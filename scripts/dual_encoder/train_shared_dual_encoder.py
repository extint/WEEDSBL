"""
SharedDualHeadNet v2  —  improved decoder + stride-4 skip + ASPP + aux head
============================================================================

Changes vs v1  (each labelled FIX-1 … FIX-4):

FIX-1  Stride-4 skip connection
       proj0 / aff0 fuse RGB[0] + NIR[0] (stride-4, sharpest features).
       F0 is upsampled and concatenated as a skip inside every decoder head,
       giving the decoder fine-grained spatial info to recover plant edges.

FIX-2  3×3 depthwise-separable spatial conv in decoder fuse block
       Replaces the pure 1×1 fuse (zero spatial context) with:
         1×1 reduce → DW 3×3 → 1×1 project → BN → ReLU
       Adds ~0.3 M params, gives the decoder local neighbourhood context.

FIX-3  Lightweight ASPP on deepest fused feature (F3, stride-32)
       Dilation rates {1,3,6,9} + global avg pool.  Much smaller than
       DeepLabV3+'s rates {6,12,18} because our feature map is already
       multi-scale-fused.  Adds ~0.5 M params.

FIX-4  Auxiliary direct 3-class head (training only, removed at inference)
       A thin 1×1 head on F1 predicts {BG, Crop, Weed} directly with
       standard CrossEntropyLoss (weight=0.4).  Gives the shared encoder
       a clean gradient signal aligned with the final evaluation metric.

FIX-5  Training tweaks
       - WEED_WEIGHT: 3.0 → 2.0  (3.0 was over-penalising background)
       - Phase-2 scheduler: T_0=40 + 2 restarts instead of T_0=80×1
       - Augmentation: added vertical flip + random 90° rotation

Runs dir layout
---------------
  runs/shared_dual_v2_YYYYMMDD_HHMMSS/
    checkpoints/
      best_model.pth          ← best val score across all epochs
      last_model.pth          ← last epoch (safe resume point)
      epoch_020.pth           ← end of warm-up
    tensorboard/              ← tensorboard --logdir runs/.../tensorboard
    config.json               ← all hyperparams
    train_log.csv             ← per-epoch metrics (loss, iou, f1 …)
    train.log                 ← full console mirror

Usage:
    python -m dual_encoder.train_shared_dual_encoder_v2
"""

import os
import sys
import csv
import json
import time
import logging
import datetime
import shutil
from pathlib import Path
from typing import Tuple
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dual_encoder.updated_architecture import (
    RGBTransformerEncoder,
    NIRLightEncoder,
    AFFModule,
    StageProjector,
)


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS  (all written to config.json for reproducibility)
# ══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    # paths
    data_root   = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET",
    runs_dir    = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs",
    # model
    rgb_variant = "small",
    nir_base_ch = 20,
    embed_dim   = 96,
    skip_ch     = 32,          # FIX-1: stride-4 skip channels
    aspp_rates  = [1, 3, 6, 9],# FIX-3: ASPP dilation rates
    aspp_out_ch = 128,         # FIX-3: ASPP output channels
    # training
    size        = (512, 512),
    batch_size  = 4,
    num_workers = 4,
    warmup_epochs = 20,
    joint_epochs  = 80,
    lr_warmup   = 1e-4,
    lr_joint    = 5e-5,
    t0_joint    = 40,          # FIX-5: cosine restart period
    lambda_cls  = 1.0,
    lambda_aux  = 0.4,         # FIX-4: aux head loss weight
    weed_weight = 2.0,         # FIX-5: reduced from 3.0
    # misc
    seed        = 42,
)


# ══════════════════════════════════════════════════════════════════════════════
# RUN DIRECTORY SETUP
# ══════════════════════════════════════════════════════════════════════════════

def make_run_dir(runs_root: str) -> Path:
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = Path(runs_root) / f"shared_dual_v2_{ts}"
    (run / "checkpoints").mkdir(parents=True)
    (run / "tensorboard").mkdir()
    return run


def setup_logging(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh  = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def setup_csv(run_dir: Path):
    path = run_dir / "train_log.csv"
    fields = [
        "epoch", "phase",
        "tr_loss", "tr_l_veg", "tr_l_cls", "tr_l_aux",
        "tr_veg_iou", "tr_cls_iou",
        "val_loss", "val_l_veg", "val_l_cls",
        "val_veg_iou", "val_cls_iou",
        "val_weed_f1", "val_crop_f1",
        "lr", "elapsed_s",
    ]
    fh = open(path, "w", newline="")
    w  = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    return fh, w


# ══════════════════════════════════════════════════════════════════════════════
# FIX-3  LIGHTWEIGHT ASPP
# ══════════════════════════════════════════════════════════════════════════════

class LightASPP(nn.Module):
    """
    Lightweight ASPP applied to the deepest fused feature (F3, stride-32).
    4 parallel dilated convs + global avg pool, all depthwise-separable to
    keep param count low.
    """
    def __init__(self, in_ch: int, out_ch: int, rates=(1, 3, 6, 9)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            self.branches.append(nn.Sequential(
                # DW conv with dilation
                nn.Conv2d(in_ch, in_ch, 3, padding=r, dilation=r,
                          groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        # global context branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        n_branches = len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n_branches, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        size = x.shape[2:]
        outs = [b(x) for b in self.branches]
        gap  = F.interpolate(self.gap(x), size=size,
                             mode="bilinear", align_corners=False)
        outs.append(gap)
        return self.project(torch.cat(outs, dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# FIX-2  SPATIAL FUSE BLOCK  (shared by both heads)
# ══════════════════════════════════════════════════════════════════════════════

def make_fuse_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    1×1 reduce → DW 3×3 spatial → 1×1 project → BN → ReLU
    Replaces the pure-1×1 fuse in v1.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),           # channel reduce
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1,            # DW spatial
                  groups=out_ch, bias=False),
        nn.Conv2d(out_ch, out_ch, 1, bias=False),           # PW project
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CBAM  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

class CBAMChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.mlp(x).view(x.size(0), x.size(1), 1, 1)


class CBAMSpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAM(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.ch = CBAMChannelGate(ch, reduction)
        self.sp = CBAMSpatialGate()
    def forward(self, x):
        return self.sp(self.ch(x))


# ══════════════════════════════════════════════════════════════════════════════
# FIX-1 + FIX-2  IMPROVED DECODER HEADS
# ══════════════════════════════════════════════════════════════════════════════

class VegHeadV2(nn.Module):
    """
    Veg head with:
      FIX-1: stride-4 skip (F0) concatenated after upsampling to stride-8 res
      FIX-2: 3×3 DW spatial conv in fuse block
    """
    def __init__(self, skip_ch: int, in_ch2: int, in_ch3: int, in_ch4: int,
                 embed_dim: int = 96):
        super().__init__()
        # FIX-1: project stride-4 skip to embed_dim
        self.proj0 = nn.Sequential(
            nn.Conv2d(skip_ch, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.proj2 = nn.Conv2d(in_ch2, embed_dim, 1, bias=False)
        self.proj3 = nn.Conv2d(in_ch3, embed_dim, 1, bias=False)
        self.proj4 = nn.Conv2d(in_ch4, embed_dim, 1, bias=False)
        # FIX-2: 4 × embed_dim in (3 scales + skip), spatial fuse
        self.fuse  = make_fuse_block(embed_dim * 4, embed_dim)
        self.head  = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, f0, f2, f3, f4, input_size):
        _, _, H2, W2 = f2.shape
        p0 = F.interpolate(self.proj0(f0), (H2, W2),
                           mode="bilinear", align_corners=False)  # FIX-1
        p2 = self.proj2(f2)
        p3 = F.interpolate(self.proj3(f3), (H2, W2),
                           mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(f4), (H2, W2),
                           mode="bilinear", align_corners=False)
        x  = self.fuse(torch.cat([p0, p2, p3, p4], dim=1))       # FIX-2
        return F.interpolate(self.head(x), input_size,
                             mode="bilinear", align_corners=False)


class ClsHeadV2(nn.Module):
    """
    Cls head with FIX-1 + FIX-2, plus CBAM (unchanged from v1).
    """
    def __init__(self, skip_ch: int, in_ch2: int, in_ch3: int, in_ch4: int,
                 embed_dim: int = 96):
        super().__init__()
        self.proj0 = nn.Sequential(
            nn.Conv2d(skip_ch, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.proj2 = nn.Conv2d(in_ch2, embed_dim, 1, bias=False)
        self.proj3 = nn.Conv2d(in_ch3, embed_dim, 1, bias=False)
        self.proj4 = nn.Conv2d(in_ch4, embed_dim, 1, bias=False)
        self.fuse  = make_fuse_block(embed_dim * 4, embed_dim)
        self.cbam  = CBAM(embed_dim)
        self.head  = nn.Conv2d(embed_dim, 2, 1)   # crop=0, weed=1

    def forward(self, f0, f2, f3, f4, input_size):
        _, _, H2, W2 = f2.shape
        p0 = F.interpolate(self.proj0(f0), (H2, W2),
                           mode="bilinear", align_corners=False)
        p2 = self.proj2(f2)
        p3 = F.interpolate(self.proj3(f3), (H2, W2),
                           mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(f4), (H2, W2),
                           mode="bilinear", align_corners=False)
        x  = self.fuse(torch.cat([p0, p2, p3, p4], dim=1))
        x  = self.cbam(x)
        return F.interpolate(self.head(x), input_size,
                             mode="bilinear", align_corners=False)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class SharedDualHeadNetV2(nn.Module):
    """
    SharedDualHeadNet v2 with all 4 fixes applied.

    Forward (training):
        returns (veg_logits, cls_logits, aux_logits)
        aux_logits is None at inference time

    Forward (inference / eval):
        call model.eval() first; aux_logits = None
    """
    def __init__(self, cfg: dict):
        super().__init__()
        rgb_variant = cfg["rgb_variant"]
        nir_base_ch = cfg["nir_base_ch"]
        embed_dim   = cfg["embed_dim"]
        skip_ch     = cfg["skip_ch"]
        aspp_rates  = cfg["aspp_rates"]
        aspp_out_ch = cfg["aspp_out_ch"]

        # ── Shared encoders ───────────────────────────────────────────────────
        self.rgb_encoder = RGBTransformerEncoder(variant=rgb_variant)
        self.nir_encoder = NIRLightEncoder(base_ch=nir_base_ch)

        if rgb_variant == "tiny":
            rgb_dims = [32,  64, 128, 256]
        elif rgb_variant == "small":
            rgb_dims = [32,  64, 160, 256]
        elif rgb_variant == "base":
            rgb_dims = [64, 128, 320, 512]
        else:
            raise ValueError(f"Unknown rgb_variant: {rgb_variant}")

        nir_dims = [nir_base_ch * 2, nir_base_ch * 4,
                    nir_base_ch * 8, nir_base_ch * 16]

        # FIX-1  stride-4 fusion ──────────────────────────────────────────────
        self.proj0 = StageProjector(rgb_dims[0], nir_dims[0], skip_ch)
        self.aff0  = AFFModule(skip_ch)

        # stride-8 / 16 / 32 fusion (same as v1) ─────────────────────────────
        fused = [64, 128, 256]
        self.proj1 = StageProjector(rgb_dims[1], nir_dims[1], fused[0])
        self.aff1  = AFFModule(fused[0])
        self.proj2 = StageProjector(rgb_dims[2], nir_dims[2], fused[1])
        self.aff2  = AFFModule(fused[1])
        self.proj3 = StageProjector(rgb_dims[3], nir_dims[3], fused[2])
        self.aff3  = AFFModule(fused[2])

        # FIX-3  ASPP on deepest fused feature ────────────────────────────────
        self.aspp  = LightASPP(fused[2], aspp_out_ch, rates=aspp_rates)

        # Task heads ──────────────────────────────────────────────────────────
        self.veg_head = VegHeadV2(skip_ch, fused[0], fused[1], aspp_out_ch,
                                   embed_dim)
        self.cls_head = ClsHeadV2(skip_ch, fused[0], fused[1], aspp_out_ch,
                                   embed_dim)

        # FIX-4  Auxiliary 3-class head (training only) ───────────────────────
        self.aux_head = nn.Conv2d(fused[0], 3, 1)   # on F1 (stride-8)

    def forward(self, x_rgb, x_nir):
        input_size = x_rgb.shape[-2:]

        R = self.rgb_encoder(x_rgb)   # [s4, s8, s16, s32]
        N = self.nir_encoder(x_nir)   # [s4, s8, s16, s32]

        # FIX-1: stride-4 skip
        r0, n0 = self.proj0(R[0], N[0])
        F0 = self.aff0(r0, n0)                      # (B, skip_ch, H/4, W/4)

        # stride-8 / 16
        r1, n1 = self.proj1(R[1], N[1]); F1 = self.aff1(r1, n1)
        r2, n2 = self.proj2(R[2], N[2]); F2 = self.aff2(r2, n2)

        # FIX-3: ASPP on stride-32
        r3, n3 = self.proj3(R[3], N[3])
        F3_raw = self.aff3(r3, n3)
        F3     = self.aspp(F3_raw)                  # (B, aspp_out_ch, H/32, W/32)

        # Heads
        veg_logits = self.veg_head(F0, F1, F2, F3, input_size)
        cls_logits = self.cls_head(F0, F1, F2, F3, input_size)

        # FIX-4: aux head — only compute during training to save memory
        if self.training:
            aux_logits = F.interpolate(
                self.aux_head(F1), input_size,
                mode="bilinear", align_corners=False)
        else:
            aux_logits = None

        return veg_logits, cls_logits, aux_logits

    @torch.no_grad()
    def predict(self, x_rgb, x_nir, veg_thresh: float = 0.5):
        """Returns 3-class mask: 0=BG, 1=Crop, 2=Weed."""
        self.eval()
        veg_logits, cls_logits, _ = self.forward(x_rgb, x_nir)
        veg_mask = torch.sigmoid(veg_logits).squeeze(1) > veg_thresh
        cls_pred = cls_logits.argmax(dim=1) + 1
        cls_pred[~veg_mask] = 0
        return cls_pred


# ══════════════════════════════════════════════════════════════════════════════
# DATASET  (FIX-5: added vertical flip + random 90° rotation augmentation)
# ══════════════════════════════════════════════════════════════════════════════

class DualTaskDataset(Dataset):
    def __init__(self, root: str, split: str = "train",
                 target_size: Tuple[int, int] = (512, 512),
                 augment: bool = True):
        self.root      = root
        self.target_h, self.target_w = target_size
        self.augment   = augment and split == "train"
        self.rgb_mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        split_file = os.path.join(root, "splits", f"{split}.txt")
        with open(split_file) as f:
            ids = [l.strip() for l in f if l.strip()]

        self.samples = [dict(
            rgb  = os.path.join(root, "rgb",    f"rgb_{i}.png"),
            nir  = os.path.join(root, "nir",    f"nir_{i}.png"),
            mask = os.path.join(root, "masks",  f"mask_{i}.png"),
        ) for i in ids]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s       = self.samples[idx]
        rgb     = cv2.cvtColor(cv2.imread(s["rgb"]), cv2.COLOR_BGR2RGB)
        nir     = cv2.imread(s["nir"], cv2.IMREAD_UNCHANGED)
        if nir.ndim == 3:
            nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)
        raw_mask = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)

        rgb      = cv2.resize(rgb,      (self.target_w, self.target_h))
        nir      = cv2.resize(nir,      (self.target_w, self.target_h))
        raw_mask = cv2.resize(raw_mask, (self.target_w, self.target_h),
                              interpolation=cv2.INTER_NEAREST)

        # FIX-5: richer augmentation
        if self.augment:
            if np.random.rand() > 0.5:            # horizontal flip
                rgb = np.fliplr(rgb).copy()
                nir = np.fliplr(nir).copy()
                raw_mask = np.fliplr(raw_mask).copy()
            if np.random.rand() > 0.5:            # vertical flip
                rgb = np.flipud(rgb).copy()
                nir = np.flipud(nir).copy()
                raw_mask = np.flipud(raw_mask).copy()
            k = np.random.randint(0, 4)           # random 90° rotation
            if k > 0:
                rgb      = np.rot90(rgb,      k).copy()
                nir      = np.rot90(nir,      k).copy()
                raw_mask = np.rot90(raw_mask, k).copy()

        veg_mask = ((raw_mask == 1) | (raw_mask == 2)).astype(np.float32)
        cls_mask = np.full_like(raw_mask, 255, dtype=np.int64)
        cls_mask[raw_mask == 1] = 0
        cls_mask[raw_mask == 2] = 1

        # FIX-4: raw 3-class GT for aux head
        gt3 = raw_mask.astype(np.int64)           # 0=BG, 1=Crop, 2=Weed

        rgb = (rgb.astype(np.float32) / 255.0 - self.rgb_mean) / self.rgb_std
        nir = nir.astype(np.float32) / 255.0

        return dict(
            rgb      = torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
            nir      = torch.from_numpy(nir[None]).float(),
            veg_mask = torch.from_numpy(veg_mask),
            cls_mask = torch.from_numpy(cls_mask),
            gt3      = torch.from_numpy(gt3),
        )


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════

class JointLossV2(nn.Module):
    """
    L = L_veg + λ_cls * L_cls + λ_aux * L_aux

    L_veg : BCE  — all pixels (model learns where vegetation is)
    L_cls : CE   — vegetation pixels only (ignore_index=255)
    L_aux : CE   — all pixels, direct 3-class supervision (FIX-4)
    """
    def __init__(self, lambda_cls=1.0, lambda_aux=0.4, weed_weight=2.0):
        super().__init__()
        self.lam_cls = lambda_cls
        self.lam_aux = lambda_aux
        self.bce     = nn.BCEWithLogitsLoss()
        self.register_buffer("cls_w", torch.tensor([1.0, weed_weight]))
        self.ce_cls  = nn.CrossEntropyLoss(ignore_index=255, weight=self.cls_w)
        self.ce_aux  = nn.CrossEntropyLoss()   # standard, no ignore

    def forward(self, veg_logits, cls_logits, aux_logits,
                veg_mask, cls_mask, gt3):
        l_veg = self.bce(veg_logits.squeeze(1), veg_mask)
        l_cls = self.ce_cls(cls_logits, cls_mask)
        l_aux = self.ce_aux(aux_logits, gt3) if aux_logits is not None else torch.tensor(0.)
        total = l_veg + self.lam_cls * l_cls + self.lam_aux * l_aux
        return total, l_veg.item(), l_cls.item(), l_aux.item() if aux_logits is not None else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_veg_iou(veg_prob, veg_mask, thresh=0.5):
    pred  = (veg_prob > thresh).long()
    tgt   = veg_mask.long()
    inter = (pred & tgt).sum().float()
    union = (pred | tgt).sum().float()
    return (inter + 1e-6) / (union + 1e-6)


def compute_cls_metrics(cls_logits, cls_mask):
    """Per-class IoU + F1 on vegetation pixels only."""
    pred  = cls_logits.argmax(dim=1)
    valid = cls_mask != 255
    pred  = pred[valid]; tgt = cls_mask[valid]
    out   = {}
    for c, name in [(0, "crop"), (1, "weed")]:
        p = (pred == c); t = (tgt == c)
        tp = (p & t).sum().float()
        fp = (p & ~t).sum().float()
        fn = (~p & t).sum().float()
        iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
        f1  = 2 * tp / (2 * tp + fp + fn + 1e-6)
        out[f"{name}_iou"] = iou.item()
        out[f"{name}_f1"]  = f1.item()
    out["mean_iou"] = (out["crop_iou"] + out["weed_iou"]) / 2
    out["mean_f1"]  = (out["crop_f1"]  + out["weed_f1"])  / 2
    return out


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def set_head_grad(model, head: str, requires_grad: bool):
    target = model.veg_head if head == "veg" else model.cls_head
    for p in target.parameters():
        p.requires_grad = requires_grad
    # aux_head follows cls_head (only relevant in joint phase)
    if head == "cls":
        for p in model.aux_head.parameters():
            p.requires_grad = requires_grad


def train_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    totals = defaultdict(float)
    for batch in tqdm(loader, desc="  train", leave=False):
        rgb      = batch["rgb"].to(device)
        nir      = batch["nir"].to(device)
        veg_mask = batch["veg_mask"].to(device)
        cls_mask = batch["cls_mask"].to(device)
        gt3      = batch["gt3"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            veg_l, cls_l, aux_l = model(rgb, nir)
            loss, lv, lc, la = criterion(veg_l, cls_l, aux_l,
                                          veg_mask, cls_mask, gt3)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        veg_prob = torch.sigmoid(veg_l.detach().squeeze(1))
        cm = compute_cls_metrics(cls_l.detach(), cls_mask)
        totals["loss"]    += loss.item()
        totals["l_veg"]   += lv
        totals["l_cls"]   += lc
        totals["l_aux"]   += la
        totals["veg_iou"] += compute_veg_iou(veg_prob, veg_mask).item()
        totals["cls_iou"] += cm["mean_iou"]

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    totals = defaultdict(float)
    for batch in loader:
        rgb      = batch["rgb"].to(device)
        nir      = batch["nir"].to(device)
        veg_mask = batch["veg_mask"].to(device)
        cls_mask = batch["cls_mask"].to(device)
        gt3      = batch["gt3"].to(device)

        veg_l, cls_l, _ = model(rgb, nir)   # aux_logits=None in eval
        loss, lv, lc, _ = criterion(veg_l, cls_l, None, veg_mask, cls_mask, gt3)

        veg_prob = torch.sigmoid(veg_l.squeeze(1))
        cm = compute_cls_metrics(cls_l, cls_mask)
        totals["loss"]      += loss.item()
        totals["l_veg"]     += lv
        totals["l_cls"]     += lc
        totals["veg_iou"]   += compute_veg_iou(veg_prob, veg_mask).item()
        totals["cls_iou"]   += cm["mean_iou"]
        totals["weed_f1"]   += cm["weed_f1"]
        totals["crop_f1"]   += cm["crop_f1"]
        totals["weed_iou"]  += cm["weed_iou"]
        totals["crop_iou"]  += cm["crop_iou"]

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_ckpt(model, optimizer, scheduler, epoch, score, val_metrics,
              path: Path, cfg: dict):
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "sched_state_dict": scheduler.state_dict(),
        "score":            score,
        "val_metrics":      val_metrics,
        "cfg":              cfg,
    }, path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = make_run_dir(CFG["runs_dir"])
    log     = setup_logging(run_dir)
    csv_fh, csv_writer = setup_csv(run_dir)
    tb      = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    with open(run_dir / "config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    log.info(f"Run dir : {run_dir}")
    log.info(f"Device  : {DEVICE}")
    log.info(f"Config  : {CFG}")

    # ── Data ─────────────────────────────────────────────────────────────────
    SIZE = tuple(CFG["size"])
    train_ds = DualTaskDataset(CFG["data_root"], "train", SIZE, augment=True)
    val_ds   = DualTaskDataset(CFG["data_root"], "val",   SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=CFG["num_workers"],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=CFG["num_workers"],
                              pin_memory=True)
    log.info(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SharedDualHeadNetV2(CFG).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"SharedDualHeadNetV2 params: {total_params/1e6:.2f}M")

    # Log model summary to tensorboard text
    tb.add_text("model/params", f"{total_params/1e6:.2f}M parameters")
    tb.add_text("model/config", json.dumps(CFG, indent=2))

    criterion = JointLossV2(
        lambda_cls  = CFG["lambda_cls"],
        lambda_aux  = CFG["lambda_aux"],
        weed_weight = CFG["weed_weight"],
    ).to(DEVICE)
    scaler    = torch.amp.GradScaler("cuda")
    best_score = 0.0
    global_epoch = 0     # absolute epoch counter across both phases

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — warm-up: train VegHead + shared encoder, cls/aux frozen
    # ─────────────────────────────────────────────────────────────────────────
    WARMUP = CFG["warmup_epochs"]
    log.info("\n" + "="*60)
    log.info("PHASE 1 — Veg head warm-up")
    log.info("="*60)

    set_head_grad(model, "cls", False)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr_warmup"], weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=WARMUP)

    for epoch in range(1, WARMUP + 1):
        global_epoch += 1
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, scaler, criterion, DEVICE)
        vl = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        score  = vl["veg_iou"]   # warmup scores on veg only

        # TensorBoard
        tb.add_scalar("loss/train",     tr["loss"],    global_epoch)
        tb.add_scalar("loss/val",       vl["loss"],    global_epoch)
        tb.add_scalar("veg_iou/train",  tr["veg_iou"], global_epoch)
        tb.add_scalar("veg_iou/val",    vl["veg_iou"], global_epoch)
        tb.add_scalar("lr",             lr_now,        global_epoch)

        # CSV
        csv_writer.writerow({
            "epoch": global_epoch, "phase": "warmup",
            "tr_loss": f"{tr['loss']:.4f}", "tr_l_veg": f"{tr['l_veg']:.4f}",
            "tr_l_cls": "0", "tr_l_aux": "0",
            "tr_veg_iou": f"{tr['veg_iou']:.4f}", "tr_cls_iou": "0",
            "val_loss": f"{vl['loss']:.4f}", "val_l_veg": f"{vl['l_veg']:.4f}",
            "val_l_cls": f"{vl['l_cls']:.4f}",
            "val_veg_iou": f"{vl['veg_iou']:.4f}", "val_cls_iou": f"{vl['cls_iou']:.4f}",
            "val_weed_f1": f"{vl.get('weed_f1',0):.4f}",
            "val_crop_f1": f"{vl.get('crop_f1',0):.4f}",
            "lr": f"{lr_now:.2e}", "elapsed_s": f"{elapsed:.1f}",
        })
        csv_fh.flush()

        log.info(
            f"[WU {epoch:02d}/{WARMUP}] "
            f"tr_loss={tr['loss']:.4f} veg_iou={tr['veg_iou']:.4f} | "
            f"val_loss={vl['loss']:.4f} val_veg_iou={vl['veg_iou']:.4f} | "
            f"lr={lr_now:.2e} {elapsed:.0f}s"
        )

    # Save end-of-warmup checkpoint
    save_ckpt(model, optimizer, scheduler, global_epoch, score,
              vl, run_dir / "checkpoints" / "epoch_warmup.pth", CFG)
    log.info(f"Warmup checkpoint saved → {run_dir/'checkpoints'/'epoch_warmup.pth'}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — joint training: all heads
    # ─────────────────────────────────────────────────────────────────────────
    JOINT = CFG["joint_epochs"]
    T0    = CFG["t0_joint"]
    log.info("\n" + "="*60)
    log.info("PHASE 2 — Joint training")
    log.info("="*60)

    set_head_grad(model, "cls", True)
    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG["lr_joint"], weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0)

    for epoch in range(1, JOINT + 1):
        global_epoch += 1
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, scaler, criterion, DEVICE)
        vl = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        # Score: weighted toward weed F1 (the hard task)
        score  = 0.4 * vl["veg_iou"] + 0.3 * vl["cls_iou"] + 0.3 * vl["weed_f1"]

        # TensorBoard
        tb.add_scalar("loss/train",      tr["loss"],      global_epoch)
        tb.add_scalar("loss/val",        vl["loss"],      global_epoch)
        tb.add_scalar("loss/l_aux_tr",   tr["l_aux"],     global_epoch)
        tb.add_scalar("veg_iou/train",   tr["veg_iou"],   global_epoch)
        tb.add_scalar("veg_iou/val",     vl["veg_iou"],   global_epoch)
        tb.add_scalar("cls_iou/train",   tr["cls_iou"],   global_epoch)
        tb.add_scalar("cls_iou/val",     vl["cls_iou"],   global_epoch)
        tb.add_scalar("weed_f1/val",     vl["weed_f1"],   global_epoch)
        tb.add_scalar("crop_f1/val",     vl["crop_f1"],   global_epoch)
        tb.add_scalar("weed_iou/val",    vl["weed_iou"],  global_epoch)
        tb.add_scalar("crop_iou/val",    vl["crop_iou"],  global_epoch)
        tb.add_scalar("lr",              lr_now,          global_epoch)
        tb.add_scalar("score/val",       score,           global_epoch)

        # CSV
        csv_writer.writerow({
            "epoch": global_epoch, "phase": "joint",
            "tr_loss": f"{tr['loss']:.4f}", "tr_l_veg": f"{tr['l_veg']:.4f}",
            "tr_l_cls": f"{tr['l_cls']:.4f}", "tr_l_aux": f"{tr['l_aux']:.4f}",
            "tr_veg_iou": f"{tr['veg_iou']:.4f}", "tr_cls_iou": f"{tr['cls_iou']:.4f}",
            "val_loss": f"{vl['loss']:.4f}", "val_l_veg": f"{vl['l_veg']:.4f}",
            "val_l_cls": f"{vl['l_cls']:.4f}",
            "val_veg_iou": f"{vl['veg_iou']:.4f}", "val_cls_iou": f"{vl['cls_iou']:.4f}",
            "val_weed_f1": f"{vl['weed_f1']:.4f}", "val_crop_f1": f"{vl['crop_f1']:.4f}",
            "lr": f"{lr_now:.2e}", "elapsed_s": f"{elapsed:.1f}",
        })
        csv_fh.flush()

        log.info(
            f"[J {epoch:02d}/{JOINT}] "
            f"loss={tr['loss']:.4f} veg={tr['veg_iou']:.4f} cls={tr['cls_iou']:.4f} | "
            f"val veg={vl['veg_iou']:.4f} cls={vl['cls_iou']:.4f} "
            f"weedF1={vl['weed_f1']:.4f} cropF1={vl['crop_f1']:.4f} | "
            f"score={score:.4f} lr={lr_now:.2e} {elapsed:.0f}s"
        )

        # Best checkpoint
        if score > best_score:
            best_score = score
            save_ckpt(model, optimizer, scheduler, global_epoch, score,
                      vl, run_dir / "checkpoints" / "best_model.pth", CFG)
            log.info(f"  ✓ Best model saved (score={score:.4f})")

        # Latest checkpoint (safe resume point)
        save_ckpt(model, optimizer, scheduler, global_epoch, score,
                  vl, run_dir / "checkpoints" / "last_model.pth", CFG)

    # ── Teardown ──────────────────────────────────────────────────────────────
    tb.close()
    csv_fh.close()
    log.info(f"\nDone. Best score: {best_score:.4f}")
    log.info(f"All outputs in: {run_dir}")


if __name__ == "__main__":
    main()