"""
benchmark_inference.py
======================
Modular inference benchmarking harness for weed-segmentation models.

Design goals
------------
* One shared DataLoader across every model — apples-to-apples comparison.
* Drop-in model registration: add a new model in ONE place (MODEL_REGISTRY).
* Industry-standard latency protocol: GPU warm-up → timed runs → statistics.
* Publishes every metric seen in segmentation papers:
    model size (M params / MB on disk), FLOPs, GPU memory, throughput (img/s),
    latency (mean ± std, P50/P95/P99), pixel-acc, mean-acc, mIoU, mDice,
    mPrecision, mRecall, mF1, per-class IoU/Dice/F1.
* Saves results to CSV + pretty console table.

Usage
-----
    python benchmark_inference.py                        # benchmark all registered models
    python benchmark_inference.py --models dual_encoder_mini   # one model only
    python benchmark_inference.py --batch-size 8 --num-batches 100
"""

from __future__ import annotations

import argparse
import csv
import os
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import 
from torch.utils.data import DataLoader, Subset


import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Optional: fvcore / thop for FLOPs — install with  pip install fvcore  or
#           pip install thop.  We degrade gracefully if neither is present.
# ---------------------------------------------------------------------------
try:
    from fvcore.nn import FlopCountAnalysis
    _FVCORE = True
except ImportError:
    _FVCORE = False

try:
    from thop import profile as thop_profile
    _THOP = True
except ImportError:
    _THOP = False


# =============================================================================
# CONFIG  ── edit the paths / hyper-params here
# =============================================================================

@dataclass
class BenchConfig:
    # ── Dataset ───────────────────────────────────────────────────────────────
    data_root:   str             = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET"
    target_size: Tuple[int,int]  = (640, 640)
    batch_size:  int             = 4
    num_workers: int             = 4
    num_batches: int             = 50      # how many batches to time  (≥30 recommended)
    warmup_batches: int          = 10      # GPU warm-up before timing starts

    # ── Classes ───────────────────────────────────────────────────────────────
    num_classes:  int            = 3
    class_names:  List[str]      = field(default_factory=lambda: ["background", "crop", "weed"])

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir:  str             = "./benchmark_results"
    csv_name:    str             = "benchmark_results.csv"

    # ── Precision ─────────────────────────────────────────────────────────────
    use_amp:     bool            = True    # run inference under torch.autocast
    use_compile: bool            = False   # torch.compile (PyTorch ≥ 2.0)

    # ── Split to evaluate on ─────────────────────────────────────────────────
    split: str                   = "test"  # "val" | "test"

    # ── Visualisation ─────────────────────────────────────────────────────────
    num_vis_images: int          = 10      # images per model to visualise
    vis_dpi:        int          = 150     # output PNG DPI

    kfold_seed: int = 42
    batch_size_kfold:int = 1
    batch_size_eff: int = 1  
    num_eff_runs: int = 100            # for efficiency benchmark


CFG = BenchConfig()


# =============================================================================
# MODEL DESCRIPTOR
# =============================================================================
# Each model needs three things wired together in one place:
#   loader_fn  : () -> nn.Module  — build + load weights, return eval-ready model
#   forward_fn : (model, batch, device, use_amp) -> (logits_BCHW, mask_BHW)
#                extracts the right keys from a batch and calls the model
#   loader_cfg : dict passed to build_dataloader() to get the right DataLoader
#                Must include "loader_type" ("dual_encoder" | "sugarbeets")
#                and any overrides (target_size, batch_size, …).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelDescriptor:
    loader_fn:  Callable                  # () -> nn.Module
    forward_fn: Callable                  # (model, batch, device, use_amp) -> (logits, mask)
    loader_cfg: Dict                      # passed to build_dataloader()
    ckpt_path:  str  = ""                 # for disk-size reporting


# =============================================================================
# FIXED VISUALIZATION SAMPLES (same images across models)
# =============================================================================

# _FIXED_VIS_INDICES = None

# def get_fixed_indices(loader, num_vis):
#     global _FIXED_VIS_INDICES

#     if _FIXED_VIS_INDICES is not None:
#         return _FIXED_VIS_INDICES

#     np.random.seed(42)  # deterministic
#     dataset_len = len(loader.dataset)

#     _FIXED_VIS_INDICES = np.random.choice(
#         dataset_len,
#         size=min(num_vis, dataset_len),
#         replace=False
#     )

#     return _FIXED_VIS_INDICES

# ─── DualEncoderMini ──────────────────────────────────────────────────────────

def _load_dual_encoder_mini() -> nn.Module:
    from dual_encoder.dual_encoder_mini import DualEncoderMini

    ckpt_path = (
        "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/"
        "dual_encoder_mini_20260412_130647/checkpoints/best_model.pth"
    )
    model = DualEncoderMini(rgb_base_ch=16, nir_base_ch=8, aspp_ch=64, num_classes=3)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print(f"  [✓] dual_encoder_mini  ←  {ckpt_path}")
    return model


def _fwd_dual_encoder(
    model: nn.Module, batch: Dict, device: str, use_amp: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch keys: rgb (B,3,H,W)  nir (B,1,H,W)  mask (B,H,W)
    Model call: model(rgb, nir) -> (logits, aux)  or  logits

    NOTE: inputs are always cast to float32 on device before the forward pass.
    torch.autocast then handles internal mixed-precision casting safely — this
    avoids the "Input HalfTensor / weight FloatTensor" mismatch that occurs when
    pinned-memory loaders deliver float32 tensors that autocast inadvertently
    promotes to float16 before the first conv sees them.
    """
    rgb  = batch["rgb"].to(device=device, dtype=torch.float32, non_blocking=True)
    nir  = batch["nir"].to(device=device, dtype=torch.float32, non_blocking=True)
    mask = batch["mask"].to(device=device, non_blocking=True).long()

    with torch.no_grad():
        if use_amp and device != "cpu":
            with torch.autocast(device_type=device):
                out = model(rgb, nir)
        else:
            out = model(rgb, nir)

    logits = out[0] if isinstance(out, (tuple, list)) else out
    # Return float32 logits — if autocast emitted float16, upcast for metrics/vis
    return logits.float(), mask


# ─── DeepLabV3-MobileNet (finetuned, 4-ch RGBNIR) ────────────────────────────

def _load_deeplabv3_mobilenet() -> nn.Module:
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights,
    )
    from collections import deque

    ckpt_path = (
        "/home/vjti-comp/WEEDSBL/scripts/sota/experiments/"
        "sugarbeets_deeplabv3_mobilenet_4ch_RGBNIR_20260413_000804/"
        "checkpoints/best_model.pth"
    )

    # ── Build architecture (no pretrained weights — we load our own) ──────
    model = deeplabv3_mobilenet_v3_large(weights=None)

    # ── Patch first conv: 3-ch → 4-ch (same as finetune.py) ──────────────
    def _get_first_conv(module):
        q = deque()
        for name, child in module.named_children():
            q.append((module, name, child))
        while q:
            parent, attr, mod = q.popleft()
            if isinstance(mod, nn.Conv2d):
                return parent, attr, mod
            for name, child in mod.named_children():
                q.append((mod, name, child))
        return None

    result = _get_first_conv(model.backbone)
    if result is None:
        raise RuntimeError("Could not locate first Conv2d in MobileNet backbone.")
    parent, attr, old_conv = result
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    setattr(parent, attr, new_conv)

    # ── Replace classifier head (21 COCO classes → 3) ─────────────────────
    in_main = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_main, 3, kernel_size=1)
    if model.aux_classifier is not None:
        in_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_aux, 3, kernel_size=1)

    # ── Load fine-tuned weights ────────────────────────────────────────────
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print(f"  [✓] deeplabv3_mobilenet  ←  {ckpt_path}")
    return model


def _fwd_deeplabv3(
    model: nn.Module, batch: Dict, device: str, use_amp: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch keys: images (B,4,H,W)  labels (B,H,W)
    Model call: model(x) -> {"out": logits, "aux": ...}

    Inputs cast to float32 explicitly; autocast handles internal mixed-precision.
    """
    x    = batch["images"].to(device=device, dtype=torch.float32, non_blocking=True)
    mask = batch["labels"].to(device=device, non_blocking=True).long()

    with torch.no_grad():
        if use_amp and device != "cpu":
            with torch.autocast(device_type=device):
                out = model(x)
        else:
            out = model(x)

    logits = out["out"] if isinstance(out, dict) else out
    return logits.float(), mask


# ─── Distilled UNet (RGB-only student, knowledge-distilled from DeepLabV3) ───

def _load_distilled_unet() -> nn.Module:
    """Original distilled UNet (from DeepLabV3 teacher)"""
    from models import UNet as unetdeeplabs

    ckpt_path = (
        "/home/vjti-comp/WEEDSBL/scripts/sota/experiments_distill/"
        "distill_deeplabv3_mobilenet_UNet_S8_20260417_223539/"
        "checkpoints/best_student.pth"
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt.get("model_config", {})
    base_ch = model_config.get("student", {}).get("base_ch", 16)
    print(f"  [distilled_unet] base_ch={base_ch} (from checkpoint metadata)")

    model = unetdeeplabs(in_channels=3, base_ch=base_ch, out_channels=3)

    state_dict = ckpt.get("student_state_dict", ckpt.get("model_state_dict", ckpt))

    # Only strip DataParallel prefix if present
    if state_dict and list(state_dict.keys())[0].startswith("module."):
        print("  [distilled_unet] removing 'module.' prefix")
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        
    if any("double_conv" in k for k in state_dict):
            print(" [distilled_unet] remapping 'double_conv' → 'net' in state_dict")
            state_dict = {
                k.replace(".double_conv.", ".net."): v for k, v in state_dict.items()
            }

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    if missing or unexpected:
        print(f"  [WARN] distilled_unet missing={len(missing)} unexpected={len(unexpected)}")

    print(f"  [✓] distilled_unet loaded (epoch={ckpt.get('epoch','?')}, best_mIoU={ckpt.get('best_val_miou',0):.4f})")
    return model

def _load_distilled_unet_from_dual() -> nn.Module:
    """New distilled UNet (from DualEncoderMini teacher)"""
    from sota.models import UNet as unetdual   # same UNet class used in the new training script

    ckpt_path = (
        "/home/vjti-comp/WEEDSBL/scripts/sota/experiments_distill/distill_MiniDualEnc_UNet_S8_20260413_104444/checkpoints/best_student.pth"
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt.get("model_config", {})
    base_ch = model_config.get("student", {}).get("base_ch", 8)   # default 8 in new script
    print(f"  [distilled_unet_from_dual] base_ch={base_ch} (from checkpoint metadata)")

    model = unetdual(in_channels=3, base_ch=base_ch, out_channels=3)

    state_dict = ckpt.get("student_state_dict", ckpt.get("model_state_dict", ckpt))

    if state_dict and list(state_dict.keys())[0].startswith("module."):
        print("  [distilled_unet_from_dual] removing 'module.' prefix")
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # NO remapping — checkpoint matches current UNet (.double_conv.)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    if missing or unexpected:
        print(f"  [WARN] distilled_unet_from_dual missing={len(missing)} unexpected={len(unexpected)}")

    print(f"  [✓] distilled_unet_from_dual loaded (epoch={ckpt.get('epoch','?')}, best_mIoU={ckpt.get('best_val_miou',0):.4f})")
    return model


def _fwd_distilled_unet_dual(
    model: nn.Module, batch: Dict, device: str, use_amp: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward for the new dual-encoder distilled UNet.
    Batch comes from dual_encoder loader → keys: "rgb", "nir", "mask"
    """
    rgb  = batch["rgb"].to(device=device, dtype=torch.float32, non_blocking=True)
    mask = batch["mask"].to(device=device, non_blocking=True).long()

    with torch.no_grad():
        if use_amp and device != "cpu":
            with torch.autocast(device_type=device):
                logits = model(rgb)
        else:
            logits = model(rgb)

    logits = logits[0] if isinstance(logits, (tuple, list)) else logits

    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:],
                               mode="bilinear", align_corners=False)

    return logits.float(), mask

def _fwd_distilled_unet(
    model: nn.Module, batch: Dict, device: str, use_amp: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch keys: 'images' (B, 3or4, H, W)  'labels' (B, H, W)
    The UNet student was trained on RGB only — strip to first 3 channels.
    Model call: model(rgb) → logits  (plain tensor, NOT a dict)
    """
    imgs = batch["images"].to(device=device, dtype=torch.float32, non_blocking=True)
    rgb  = imgs[:, :3]   # student is RGB-only; drop NIR channel if present
    mask = batch["labels"].to(device=device, non_blocking=True).long()

    with torch.no_grad():
        if use_amp and device != "cpu":
            with torch.autocast(device_type=device):
                logits = model(rgb)
        else:
            logits = model(rgb)

    # UNet returns a plain tensor (not a dict)
    logits = logits[0] if isinstance(logits, (tuple, list)) else logits

    # Spatial alignment guard: UNet should be same H×W as input, but be safe
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:],
                               mode="bilinear", align_corners=False)

    return logits.float(), mask


# ─── Registry ────────────────────────────────────────────────────────────────
#  To add a new model:
#    1. Write _load_yourmodel() and _fwd_yourmodel()
#    2. Add one ModelDescriptor entry below — nothing else changes.

MODEL_REGISTRY: Dict[str, ModelDescriptor] = {
    "dual_encoder_mini": ModelDescriptor(
        loader_fn  = _load_dual_encoder_mini,
        forward_fn = _fwd_dual_encoder,
        loader_cfg = {
            "loader_type": "dual_encoder",
            "target_size": (640, 640),
        },
        ckpt_path  = (
            "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/"
            "dual_encoder_mini_20260412_130647/checkpoints/best_model.pth"
        ),
    ),
    "deeplabv3_mobilenet": ModelDescriptor(
        loader_fn  = _load_deeplabv3_mobilenet,
        forward_fn = _fwd_deeplabv3,
        loader_cfg = {
            "loader_type": "sugarbeets",
            "target_size": (966, 1296),
            "use_rgbnir":  True,
            "nir_drop":    0.0,
        },
        ckpt_path  = (
            "/home/vjti-comp/WEEDSBL/scripts/sota/experiments/"
            "sugarbeets_deeplabv3_mobilenet_4ch_RGBNIR_20260413_000804/"
            "checkpoints/best_model.pth"
        ),
    ),
    "distilled_unet_from_deeplabsv3": ModelDescriptor(
        loader_fn  = _load_distilled_unet,
        forward_fn = _fwd_distilled_unet,
        loader_cfg = {
            "loader_type": "sugarbeets",
            "target_size": (640, 640),
            "use_rgbnir":  True,   # DataLoader loads 4ch; forward_fn strips to RGB
            "nir_drop":    0.0,
        },
        ckpt_path  = (
            "/home/vjti-comp/WEEDSBL/scripts/sota/experiments_distill/"
            "distill_deeplabv3_mobilenet_UNet_S8_20260417_223539/"
            "checkpoints/best_student.pth"
        ),
    ),
    "distilled_unet_from_dual": ModelDescriptor(
        loader_fn  = _load_distilled_unet_from_dual,
        forward_fn = _fwd_distilled_unet_dual,
        loader_cfg = {
            "loader_type": "dual_encoder",
            "target_size": (640, 640),
        },
        ckpt_path  = (
            "/home/vjti-comp/WEEDSBL/scripts/sota/experiments_distill/"
            "distill_MiniDualEnc_UNet_S8_20260413_104444/checkpoints/best_student.pth"
        ),
    ),
    # ── add future models here ──────────────────────────────────────────────
    # "deeplabv3_resnet50": ModelDescriptor(...),
    # "lraspp_mobilenet":   ModelDescriptor(...),
}


# =============================================================================
# DATA LOADER  (one loader per model — each model declares its own loader_cfg)
# =============================================================================

def build_dataloader(cfg: BenchConfig, loader_cfg: Optional[Dict] = None) -> DataLoader:
    """
    Build the correct DataLoader for a model.

    loader_cfg keys
    ---------------
    loader_type : "dual_encoder"  →  create_dual_encoder_dataloaders
                  "sugarbeets"    →  create_sugarbeets_dataloaders
    target_size : (H, W) override (default: cfg.target_size)
    use_rgbnir  : bool  (sugarbeets only, default True)
    nir_drop    : float (sugarbeets only, default 0.0)
    """
    lc          = loader_cfg or {}
    loader_type = lc.get("loader_type", "dual_encoder")
    target_size = lc.get("target_size", cfg.target_size)

    if loader_type == "dual_encoder":
        from dual_encoder.dual_encoder_data_loader_weedcrop import (
            create_dual_encoder_dataloaders,
        )
        train_loader, val_loader, test_loader = create_dual_encoder_dataloaders(
            data_root=cfg.data_root,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            target_size=target_size,
            mask_mode="multiclass",
        )

    elif loader_type == "sugarbeets":
        from sugarbeets_data_loader import create_sugarbeets_dataloaders
        train_loader, val_loader, test_loader = create_sugarbeets_dataloaders(
            data_root=cfg.data_root,
            use_rgbnir=lc.get("use_rgbnir", True),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            target_size=target_size,
            nir_drop_prob=lc.get("nir_drop", 0.0),
        )

    else:
        raise ValueError(f"Unknown loader_type: {loader_type!r}. "
                         f"Choose 'dual_encoder' or 'sugarbeets'.")

    split_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader    = split_map[cfg.split]
    print(f"[DataLoader/{loader_type}] split={cfg.split}  "
          f"res={target_size}  samples={len(loader.dataset)}  "
          f"batches={len(loader)}  batch_size={cfg.batch_size}")
    return loader


# =============================================================================
# SEGMENTATION METRICS  (mirrors SegmentationMetrics from training)
# =============================================================================

class SegmentationMetrics:
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
        mask = (t >= 0) & (t < self.num_classes)
        np.add.at(self.conf_matrix, (t[mask], p[mask]), 1)

    def compute(self) -> Dict:
        cm  = self.conf_matrix.astype(np.float64)
        TP  = np.diag(cm)
        FP  = cm.sum(0) - TP
        FN  = cm.sum(1) - TP
        TN  = cm.sum() - (TP + FP + FN)
        eps = 1e-6

        iou       = TP / (TP + FP + FN + eps)
        dice      = 2 * TP / (2 * TP + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        spec      = TN  / (TN + FP + eps)

        results: Dict = {
            "pixel_accuracy": float(TP.sum() / (cm.sum() + eps)),
            "mean_accuracy":  float((TP / (cm.sum(1) + eps)).mean()),
            "mean_iou":       float(iou.mean()),
            "mean_dice":      float(dice.mean()),
            "mean_precision": float(precision.mean()),
            "mean_recall":    float(recall.mean()),
            "mean_f1":        float(dice.mean()),
        }
        for c, name in enumerate(self.class_names):
            results[f"iou_{name}"]         = float(iou[c])
            results[f"dice_{name}"]        = float(dice[c])
            results[f"precision_{name}"]   = float(precision[c])
            results[f"recall_{name}"]      = float(recall[c])
            results[f"f1_{name}"]          = float(dice[c])
            results[f"specificity_{name}"] = float(spec[c])
        return results


# =============================================================================
# MODEL-SIZE UTILS
# =============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_disk_size_mb(ckpt_path: Optional[str]) -> float:
    """Size of the .pth file on disk in MB, or -1 if path unknown."""
    if ckpt_path and Path(ckpt_path).exists():
        return Path(ckpt_path).stat().st_size / 1e6
    return -1.0


def compute_flops(
    model: nn.Module,
    dummy_inputs: Tuple[torch.Tensor, ...],
    device: str,
) -> float:
    """
    Returns GFLOPs (giga multiply-add ops).
    Tries fvcore first, then thop, else returns -1.

    IMPORTANT: profiles on a CPU clone so the original model stays on `device`.
    """
    # Work on a CPU copy — do NOT move the original model off device
    import copy
    model_cpu  = copy.deepcopy(model).cpu().eval()
    inputs_cpu = tuple(x.cpu().float() for x in dummy_inputs)

    gflops = -1.0
    if _FVCORE:
        try:
            flops = FlopCountAnalysis(model_cpu, inputs_cpu)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            gflops = flops.total() / 1e9
        except Exception:
            pass

    if gflops < 0 and _THOP:
        try:
            macs, _ = thop_profile(model_cpu, inputs=inputs_cpu, verbose=False)
            gflops = macs / 1e9
        except Exception:
            pass

    del model_cpu   # free CPU memory immediately
    return gflops


# =============================================================================
# LATENCY BENCHMARKING  (industry-standard protocol)
# =============================================================================

def _forward_pass(
    model:      nn.Module,
    batch:      Dict,
    device:     str,
    use_amp:    bool,
    forward_fn: Optional[Callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward pass via the model's forward_fn.

    Returns (preds_BHW, mask_BHW) — predictions as argmax class indices.

    forward_fn signature:
        (model, batch, device, use_amp) -> (logits_BCHW, mask_BHW)

    Falls back to dual-encoder convention when forward_fn is None.
    """
    if forward_fn is not None:
        logits, mask = forward_fn(model, batch, device, use_amp)
    else:
        # Legacy fallback: dual-encoder convention
        logits, mask = _fwd_dual_encoder(model, batch, device, use_amp)

    return logits.argmax(dim=1), mask


def benchmark_model(
    model_name:  str,
    model:       nn.Module,
    loader:      DataLoader,
    cfg:         BenchConfig,
    device:      str,
    descriptor:  "ModelDescriptor",
) -> Dict:
    """
    Full benchmark for one model:
      1. Model-size metrics
      2. FLOPs (single image)
      3. GPU warm-up (cfg.warmup_batches batches, not timed)
      4. Timed run (cfg.num_batches batches)
         - per-batch GPU latency (CUDA events for sub-ms accuracy)
         - segmentation metrics accumulated over the full timed run
      5. GPU peak memory

    The descriptor supplies forward_fn (handles different batch schemas)
    and ckpt_path (for disk-size reporting).
    Returns a flat dict of all metrics.
    """
    lc          = descriptor.loader_cfg
    target_size = lc.get("target_size", cfg.target_size)
    in_channels = 4 if lc.get("use_rgbnir", False) else 3

    print(f"\n{'='*70}")
    print(f"  Benchmarking : {model_name}")
    print(f"  Resolution   : {target_size[0]}x{target_size[1]}  "
          f"  Channels: {in_channels}")
    print(f"{'='*70}")

    model = model.to(device).eval()
    if cfg.use_compile:
        try:
            model = torch.compile(model)
            print("  [torch.compile] enabled")
        except Exception as e:
            print(f"  [torch.compile] skipped: {e}")

    # ── 1. Model size ─────────────────────────────────────────────────────────
    total_params, trainable_params = count_parameters(model)
    disk_mb = model_disk_size_mb(descriptor.ckpt_path)

    param_mb = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    ) / 1e6
    buffer_mb = sum(
        b.nelement() * b.element_size() for b in model.buffers()
    ) / 1e6

    print(f"  Params (total / trainable) : {total_params/1e6:.3f}M / {trainable_params/1e6:.3f}M")
    print(f"  In-memory size (params+buf): {param_mb+buffer_mb:.2f} MB")
    print(f"  Checkpoint on disk         : {disk_mb:.2f} MB" if disk_mb > 0 else "  Checkpoint on disk: N/A")

    # ── 2. FLOPs — build correct dummy input for this model ───────────────────
    dummy_input = torch.zeros(1, in_channels, *target_size)
    # For dual-encoder, model takes (rgb, nir) separately; for others single tensor
    loader_type = lc.get("loader_type", "dual_encoder")
    if loader_type == "dual_encoder":
        dummy_inputs = (
            torch.zeros(1, 3, *target_size),   # rgb
            torch.zeros(1, 1, *target_size),   # nir
        )
    else:
        dummy_inputs = (dummy_input,)           # single 4-ch (or 3-ch) tensor

    gflops = compute_flops(model, dummy_inputs, device)
    if gflops >= 0:
        print(f"  GFLOPs (single image)      : {gflops:.3f}")
    else:
        print("  GFLOPs                     : N/A  (install fvcore or thop)")

    # ── 3. Warm-up ────────────────────────────────────────────────────────────
    print(f"\n  Warming up ({cfg.warmup_batches} batches)…")
    loader_iter = iter(loader)
    for _ in range(cfg.warmup_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        _forward_pass(model, batch, device, cfg.use_amp, descriptor.forward_fn)

    if device != "cpu":
        torch.cuda.synchronize()

    # ── 4. Timed run ──────────────────────────────────────────────────────────
    metrics_tracker = SegmentationMetrics(cfg.num_classes, cfg.class_names)
    latencies_ms: List[float] = []

    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"  Timing {cfg.num_batches} batches…")
    batches_done = 0

    # Restart iterator so timed batches are independent of warm-up batches
    loader_iter = iter(loader)

    while batches_done < cfg.num_batches:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        # --- CUDA event timing (most accurate on GPU) -----------------------
        if device != "cpu":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            preds, mask = _forward_pass(model, batch, device, cfg.use_amp,
                                        descriptor.forward_fn)
            end_evt.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_evt.elapsed_time(end_evt))
        else:
            t0 = time.perf_counter()
            preds, mask = _forward_pass(model, batch, device, cfg.use_amp,
                                        descriptor.forward_fn)
            latencies_ms.append((time.perf_counter() - t0) * 1e3)

        metrics_tracker.update(preds, mask)
        batches_done += 1

    # ── 5. GPU memory ─────────────────────────────────────────────────────────
    if device != "cpu":
        gpu_peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        gpu_reserved_mb = torch.cuda.max_memory_reserved(device) / 1e6
    else:
        gpu_peak_mb = -1.0
        gpu_reserved_mb = -1.0

    # ── Latency statistics ────────────────────────────────────────────────────
    lat = np.array(latencies_ms)
    mean_lat  = float(lat.mean())
    std_lat   = float(lat.std())
    p50_lat   = float(np.percentile(lat, 50))
    p95_lat   = float(np.percentile(lat, 95))
    p99_lat   = float(np.percentile(lat, 99))
    min_lat   = float(lat.min())
    max_lat   = float(lat.max())

    # Throughput: images / second
    total_images = batches_done * cfg.batch_size
    total_time_s = lat.sum() / 1e3
    throughput   = total_images / total_time_s

    print(f"\n  ── Latency (ms per batch, bs={cfg.batch_size}) ──")
    print(f"     Mean±Std : {mean_lat:.2f} ± {std_lat:.2f}")
    print(f"     P50/P95/P99 : {p50_lat:.2f} / {p95_lat:.2f} / {p99_lat:.2f}")
    print(f"     Min / Max : {min_lat:.2f} / {max_lat:.2f}")
    print(f"     Throughput : {throughput:.1f} img/s")
    if device != "cpu":
        print(f"     GPU peak mem (allocated): {gpu_peak_mb:.1f} MB")
        print(f"     GPU peak mem (reserved) : {gpu_reserved_mb:.1f} MB")

    # ── Segmentation metrics ──────────────────────────────────────────────────
    seg = metrics_tracker.compute()

    print(f"\n  ── Segmentation metrics (over {batches_done} batches) ──")
    print(f"     Pixel Acc  : {seg['pixel_accuracy']:.4f}")
    print(f"     Mean Acc   : {seg['mean_accuracy']:.4f}")
    print(f"     mIoU       : {seg['mean_iou']:.4f}")
    print(f"     mDice/mF1  : {seg['mean_dice']:.4f}")
    print(f"     mPrecision : {seg['mean_precision']:.4f}")
    print(f"     mRecall    : {seg['mean_recall']:.4f}")
    for c in cfg.class_names:
        print(f"     {c:<12} IoU={seg[f'iou_{c}']:.4f}  "
              f"Dice={seg[f'dice_{c}']:.4f}  "
              f"Prec={seg[f'precision_{c}']:.4f}  "
              f"Rec={seg[f'recall_{c}']:.4f}")

    # ── Assemble flat result dict ─────────────────────────────────────────────
    result = {
        "model_name":          model_name,
        "batch_size":          cfg.batch_size,
        "input_resolution":    f"{target_size[0]}x{target_size[1]}",
        "num_batches_timed":   batches_done,
        "total_params_M":      round(total_params / 1e6, 3),
        "trainable_params_M":  round(trainable_params / 1e6, 3),
        "model_memory_MB":     round(param_mb + buffer_mb, 2),
        "checkpoint_MB":       round(disk_mb, 2) if disk_mb >= 0 else "N/A",
        "gflops":              round(gflops, 3) if gflops >= 0 else "N/A",
        "use_amp":             cfg.use_amp,
        "device":              device,
        "lat_mean_ms":         round(mean_lat, 3),
        "lat_std_ms":          round(std_lat, 3),
        "lat_p50_ms":          round(p50_lat, 3),
        "lat_p95_ms":          round(p95_lat, 3),
        "lat_p99_ms":          round(p99_lat, 3),
        "lat_min_ms":          round(min_lat, 3),
        "lat_max_ms":          round(max_lat, 3),
        "throughput_img_s":    round(throughput, 2),
        "gpu_peak_alloc_MB":   round(gpu_peak_mb, 1) if gpu_peak_mb >= 0 else "N/A",
        "gpu_peak_reserved_MB":round(gpu_reserved_mb, 1) if gpu_reserved_mb >= 0 else "N/A",
        **{k: round(v, 4) for k, v in seg.items()},
    }
    # Return the raw confusion matrix alongside the flat result dict so callers
    # can generate confusion matrix plots without a second pass.
    return result, metrics_tracker.conf_matrix.copy()


# =============================================================================
# SUMMARY TABLE  (pretty console print)
# =============================================================================

def print_summary_table(results: List[Dict], class_names: List[str]):
    sep = "─" * 130
    print(f"\n{'='*130}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*130}")
    print(f"  {'Model':<28} {'Params(M)':>9} {'GFLOPs':>8} {'ckpt(MB)':>8} "
          f"{'Lat(ms)':>9} {'P95':>7} {'img/s':>7} "
          f"{'mIoU':>7} {'mDice':>7} {'PixAcc':>7} "
          + "  ".join(f"{c[:6]:>7}" for c in class_names))
    print(sep)
    for r in results:
        iou_cols = "  ".join(
            f"{r.get(f'iou_{c}', 0):.4f}" for c in class_names
        )
        print(
            f"  {r['model_name']:<28} "
            f"{r['total_params_M']:>9.3f} "
            f"{str(r['gflops']):>8} "
            f"{str(r['checkpoint_MB']):>8} "
            f"{r['lat_mean_ms']:>9.2f} "
            f"{r['lat_p95_ms']:>7.2f} "
            f"{r['throughput_img_s']:>7.1f} "
            f"{r['mean_iou']:>7.4f} "
            f"{r['mean_dice']:>7.4f} "
            f"{r['pixel_accuracy']:>7.4f} "
            f"  {iou_cols}"
        )
    print(sep)


# =============================================================================
# CSV EXPORT
# =============================================================================

def save_csv(results: List[Dict], output_dir: str, csv_name: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / csv_name
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[✓] Results saved → {out_path}")


# =============================================================================
# JSON METRICS EXPORT  — paper-ready, full per-class breakdown
# =============================================================================

def save_metrics_json(results: List[Dict], output_dir: str,
                      json_name: str = "metrics_results.json"):
    """
    Save benchmark results as a structured JSON file alongside the CSV.

    Structure
    ---------
    {
      "metadata": { "split": ..., "timestamp": ..., "num_models": ... },
      "models": {
        "model_name": {
          "efficiency":   { params, gflops, latency, throughput, … },
          "segmentation": { pixel_acc, mean_iou, mean_dice, … },
          "per_class":    { "background": {iou, dice, …}, … }
        },
        …
      }
    }

    Standard practice note
    ----------------------
    In segmentation papers (MICCAI, CVPR, ECCV, IEEE TPAMI) metrics are
    *always* reported on the held-out TEST set — never on val or train.
    The val split is only used for model selection / early stopping.
    Quoting val-set numbers as final performance is a methodological error
    that reviewers will flag.  This script evaluates on cfg.split (default
    \"test\") — make sure you keep that setting.
    """
    import json
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / json_name
    if not results:
        return

    # Infer class names from per-class keys (iou_<name>)
    sample = results[0]
    class_names = [k[4:] for k in sample if k.startswith("iou_")]

    structured: Dict = {
        "metadata": {
            "timestamp":    datetime.now().isoformat(),
            "split":        sample.get("split", "test"),
            "num_models":   len(results),
            "class_names":  class_names,
            "note": (
                "Metrics computed on the TEST split — standard practice for "
                "research papers (val split is for model selection only)."
            ),
        },
        "models": {},
    }

    for r in results:
        name = r["model_name"]

        per_class: Dict = {}
        for cls in class_names:
            per_class[cls] = {
                "iou":         r.get(f"iou_{cls}"),
                "dice_f1":     r.get(f"dice_{cls}"),
                "precision":   r.get(f"precision_{cls}"),
                "recall":      r.get(f"recall_{cls}"),
                "f1":          r.get(f"f1_{cls}"),
                "specificity": r.get(f"specificity_{cls}"),
            }

        structured["models"][name] = {
            "efficiency": {
                "total_params_M":      r.get("total_params_M"),
                "trainable_params_M":  r.get("trainable_params_M"),
                "model_memory_MB":     r.get("model_memory_MB"),
                "checkpoint_MB":       r.get("checkpoint_MB"),
                "gflops":              r.get("gflops"),
                "input_resolution":    r.get("input_resolution"),
                "batch_size":          r.get("batch_size"),
                "lat_mean_ms":         r.get("lat_mean_ms"),
                "lat_std_ms":          r.get("lat_std_ms"),
                "lat_p50_ms":          r.get("lat_p50_ms"),
                "lat_p95_ms":          r.get("lat_p95_ms"),
                "lat_p99_ms":          r.get("lat_p99_ms"),
                "throughput_img_s":    r.get("throughput_img_s"),
                "gpu_peak_alloc_MB":   r.get("gpu_peak_alloc_MB"),
                "device":              r.get("device"),
            },
            "segmentation": {
                "pixel_accuracy": r.get("pixel_accuracy"),
                "mean_accuracy":  r.get("mean_accuracy"),
                "mean_iou":       r.get("mean_iou"),
                "mean_dice":      r.get("mean_dice"),
                "mean_precision": r.get("mean_precision"),
                "mean_recall":    r.get("mean_recall"),
                "mean_f1":        r.get("mean_f1"),
            },
            "per_class": per_class,
        }

    with open(out_path, "w") as f:
        import json as _json
        _json.dump(structured, f, indent=2, default=str)

    print(f"[✓] Structured metrics JSON → {out_path}")


# =============================================================================
# K-FOLD STATISTICAL EVALUATION
# =============================================================================
# Split held-out TEST set into K non-overlapping folds, run full inference on
# each, report mean ± std and 95% CI (t-distribution, n-1 dof).
# Required by CEA/MICCAI/CVPR reviewers for statistical rigour.
# =============================================================================

def run_kfold_statistical_eval(
    model_name:  str,
    model:       nn.Module,
    loader:      DataLoader,
    cfg:         BenchConfig,
    device:      str,
    forward_fn:  Callable,
    n_folds:     int = 5,
    output_dir:  str = "./benchmark_results",
    seed:        int = 42,
) -> Dict:
    import json
    from scipy import stats as scipy_stats
    from torch.utils.data import Subset

    dataset   = loader.dataset
    n         = len(dataset)
    rng       = np.random.default_rng(seed)
    indices   = rng.permutation(n)
    fold_size = n // n_folds

    print(f"\n{'='*70}")
    print(f"  K-Fold Statistical Eval : {model_name}")
    print(f"  Dataset size={n}  folds={n_folds}  fold_size=~{fold_size}")
    print(f"{'='*70}")

    model = model.to(device).eval()
    fold_metrics: List[Dict] = []

    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        end   = n if fold_idx == n_folds - 1 else start + fold_size
        subset = Subset(dataset, indices[start:end].tolist())
        fold_loader = DataLoader(
            subset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )
        tracker = SegmentationMetrics(cfg.num_classes, cfg.class_names)
        for batch in fold_loader:
            logits, mask = forward_fn(model, batch, device, cfg.use_amp)
            tracker.update(logits.argmax(dim=1), mask)
        m = tracker.compute()
        fold_metrics.append(m)
        per_cls = "  ".join(f"{c}={m[f'iou_{c}']:.4f}" for c in cfg.class_names)
        print(f"  Fold {fold_idx+1}/{n_folds}  n={end-start}  "
              f"mIoU={m['mean_iou']:.4f}  [{per_cls}]")

    # mean, std, 95% CI via Student-t (dof = n_folds-1)
    aggregated: Dict = {}
    for key in fold_metrics[0].keys():
        vals  = np.array([fm[key] for fm in fold_metrics], dtype=np.float64)
        mean  = float(vals.mean())
        std   = float(vals.std(ddof=1))
        se    = std / np.sqrt(n_folds)
        t_crit = float(scipy_stats.t.ppf(0.975, df=n_folds - 1))
        aggregated[key] = {
            "mean":     round(mean, 5),
            "std":      round(std,  5),
            "ci95_lo":  round(mean - t_crit * se, 5),
            "ci95_hi":  round(mean + t_crit * se, 5),
            "per_fold": [round(float(v), 5) for v in vals],
        }

    print(f"\n  ── {n_folds}-Fold Aggregated Results ──")
    for key in ["mean_iou", "mean_dice", "mean_precision", "mean_recall",
                "pixel_accuracy"] + [f"iou_{c}" for c in cfg.class_names]:
        if key not in aggregated:
            continue
        a = aggregated[key]
        print(f"  {key:<25}  {a['mean']:.4f} ± {a['std']:.4f}  "
              f"95% CI [{a['ci95_lo']:.4f}, {a['ci95_hi']:.4f}]")

    os.makedirs(output_dir, exist_ok=True)
    out = {
        "model": model_name, "n_folds": n_folds, "dataset_size": n,
        "seed": seed, "split": cfg.split,
        "note": ("mean±std over non-overlapping test folds. "
                 "95% CI: Student-t, dof=n_folds-1."),
        "aggregated": aggregated, "per_fold": fold_metrics,
    }
    json_path = Path(output_dir) / f"kfold_{model_name}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [✓] K-fold results → {json_path}\n")
    return out


def save_kfold_summary_csv(kfold_results: List[Dict], output_dir: str,
                            csv_name: str = "kfold_summary.csv"):
    """Flat CSV: model | metric | mean | std | ci95_lo | ci95_hi — paste into LaTeX."""
    if not kfold_results:
        return
    report_keys = (["mean_iou", "mean_dice", "mean_precision", "mean_recall",
                     "pixel_accuracy", "mean_accuracy"]
                   + sorted([k for k in kfold_results[0]["aggregated"]
                              if k.startswith("iou_")]))
    rows = []
    for res in kfold_results:
        for key in report_keys:
            if key not in res["aggregated"]:
                continue
            a = res["aggregated"][key]
            rows.append({"model": res["model"], "metric": key,
                         "mean": a["mean"], "std": a["std"],
                         "ci95_lo": a["ci95_lo"], "ci95_hi": a["ci95_hi"],
                         "n_folds": res["n_folds"]})
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / csv_name
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[✓] K-fold summary CSV → {out_path}")


# =============================================================================
# ABLATION STUDY — loss weight (λ) sweep
# =============================================================================
# Ablation table expected by CEA reviewers:
#   λ₁ (task=CE+Dice)  |  λ₂ (logit KD)  |  λ₃ (feature KD)
#
# IMPORTANT: λ₃ (feature loss) is NOT in the current training script
# (feature adapters were intentionally omitted — see distill_train_deeplabs_to_unet.py).
# At inference time we therefore treat λ₃ as a *post-hoc soft label interpolation*
# weight between the student logits and the teacher logits, which is the closest
# valid proxy measurable without retraining.
#
# If you later add feature distillation to the trainer, replace the
# `_interpolate_with_teacher` call with actual feature-alignment loss values
# logged to the checkpoint, and re-run.
# =============================================================================

# Ablation configurations: (label, λ₁_task, λ₂_kd, λ₃_feat, description)
ABLATION_CONFIGS: List[Tuple] = [
    ("w_kd=0.0", 1.0, 0.0, 0.0, "No KD (pure student)"),
    ("w_kd=0.3", 1.0, 0.3, 0.0, "Light KD (30% teacher)"),
    ("w_kd=0.5", 1.0, 0.5, 0.0, "TRAINED (50% teacher)"),
    ("w_kd=0.7", 1.0, 0.7, 0.0, "Medium KD (70% teacher)"),
    ("w_kd=1.0", 1.0, 1.0, 0.0, "Heavy KD (100% teacher)"),
]

def _ablation_logit_blend(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    w_kd: float,
    w_feat: float = 0.0,  # ignored for now
) -> torch.Tensor:
    """
    Simple linear interpolation between student and teacher predictions.
    α directly represents teacher influence:
        α=0.0 → pure student (w_kd=0)
        α=0.3 → 70% student, 30% teacher (w_kd=0.3)
        α=0.5 → 50% student, 50% teacher (w_kd=0.5)
        α=1.0 → pure teacher (w_kd=1.0)
    """
    alpha = w_kd  # Direct: α = w_kd
    
    s_prob = F.softmax(student_logits.float(), dim=1)
    t_prob = F.softmax(teacher_logits.float(), dim=1)
    blended = (1.0 - alpha) * s_prob + alpha * t_prob
    return blended

def run_ablation_study(
    student_name:   str,
    student_model:  nn.Module,
    teacher_name:   str,
    teacher_model:  nn.Module,
    loader:         DataLoader,
    cfg:            BenchConfig,
    device:         str,
    student_fwd_fn: Callable,
    teacher_fwd_fn: Callable,
    output_dir:     str = "./benchmark_results",
) -> List[Dict]:
    """
    Run the λ ablation sweep and save results.
    Both student and teacher must be loaded and on CPU before calling;
    this function moves them to device internally.

    Returns list of result dicts (one per config) — also saved to CSV + JSON.
    """
    import json

    student_model = student_model.to(device).eval()
    teacher_model = teacher_model.to(device).eval()

    print(f"\n{'='*70}")
    print(f"  Ablation Study : {student_name}  (teacher={teacher_name})")
    print(f"  Configs : {len(ABLATION_CONFIGS)}  |  Split: {cfg.split}")
    print(f"{'='*70}")

    all_rows: List[Dict] = []

    for label, lam1, lam2, lam3, desc in ABLATION_CONFIGS:
        tracker = SegmentationMetrics(cfg.num_classes, cfg.class_names)

        for batch in loader:
            # Student logits
            s_logits, mask = student_fwd_fn(student_model, batch, device, cfg.use_amp)

            if lam2 == 0.0 and lam3 == 0.0:
                # Pure task loss config — use student logits as-is
                preds = s_logits.argmax(dim=1)
            else:
                # Get teacher logits for blending
                with torch.no_grad():
                    t_logits, _ = teacher_fwd_fn(teacher_model, batch, device, cfg.use_amp)
                    if t_logits.shape[-2:] != s_logits.shape[-2:]:
                        t_logits = F.interpolate(t_logits, size=s_logits.shape[-2:],
                                                 mode="bilinear", align_corners=False)
                blended = _ablation_logit_blend(s_logits, t_logits, lam2, lam3)
                preds = blended.argmax(dim=1)

            tracker.update(preds, mask)

        m = tracker.compute()

        row: Dict = {
            "config":      label,
            "lambda_task": lam1,
            "lambda_kd":   lam2,
            "lambda_feat": lam3,
            "description": desc,
            **{k: round(v, 4) for k, v in m.items()},
        }
        all_rows.append(row)

        per_cls = "  ".join(f"{c}={m[f'iou_{c}']:.4f}" for c in cfg.class_names)
        print(f"  {label:<22}  mIoU={m['mean_iou']:.4f}  "
              f"mF1={m['mean_f1']:.4f}  [{per_cls}]  ← {desc}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / f"ablation_{student_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  [✓] Ablation CSV   → {csv_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = Path(output_dir) / f"ablation_{student_name}.json"
    chosen = max(all_rows, key=lambda r: r["mean_iou"])
    out = {
        "student":        student_name,
        "teacher":        teacher_name,
        "split":          cfg.split,
        "configs_tested": len(ABLATION_CONFIGS),
        "best_config":    chosen["config"],
        "best_miou":      chosen["mean_iou"],
        "note": (
            "λ₃ at inference = post-hoc soft-label interpolation proxy "
            "(Hinton et al. 2015). For true feature-KD ablation, add "
            "FPN/intermediate hooks to the trainer and re-run."
        ),
        "results": all_rows,
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [✓] Ablation JSON  → {json_path}")

    # ── Console table ─────────────────────────────────────────────────────────
    sep = "─" * 100
    print(f"\n  {'Config':<22} {'λ₁':>5} {'λ₂':>5} {'λ₃':>5} "
          f"{'mIoU':>7} {'mF1':>7} "
          + "  ".join(f"{c[:5]:>6}" for c in cfg.class_names)
          + "  Description")
    print(f"  {sep}")
    for r in all_rows:
        cls_ious = "  ".join(f"{r[f'iou_{c}']:>6.4f}" for c in cfg.class_names)
        marker = " ← BEST" if r["config"] == chosen["config"] else ""
        print(f"  {r['config']:<22} {r['lambda_task']:>5.1f} {r['lambda_kd']:>5.1f} "
              f"{r['lambda_feat']:>5.1f} {r['mean_iou']:>7.4f} {r['mean_f1']:>7.4f}  "
              f"{cls_ious}  {r['description']}{marker}")
    print(f"  {sep}")

    return all_rows


# =============================================================================
# CROSS-MODEL COMPARISON GRID — 2-row layout (paper-ready + fixed alignment)
# =============================================================================
def save_comparison_grid(
    all_model_samples: Dict[str, List[Dict]],
    cfg: BenchConfig,
    output_dir: str,
):
    """
    ONE figure, 2 rows (2 best scenes).
    Layout per row:
        RGB | GT | Model1 | Model2 | Model3 | Model4 | Error Map (best model)
    SAME scene is now guaranteed across all models.
    """
    if not all_model_samples:
        print("  [comparison_grid] No samples — skipping.")
        return

    model_names = list(all_model_samples.keys())
    num_models = len(model_names)
    num_samples = min(len(v) for v in all_model_samples.values())

    if num_samples < 2:
        print("  [comparison_grid] Need at least 2 samples — skipping.")
        return

    class_names = cfg.class_names
    num_classes = cfg.num_classes
    palette = _CLASS_COLORS[:num_classes]

    # Compute average mIoU per scene across ALL models
    per_scene_avg_iou = []
    for scene_idx in range(num_samples):
        scene_ious = [float(np.mean([_iou_single(s["pred"], s["gt"], c)
                                     for c in range(num_classes)]))
                      for s in (all_model_samples[m][scene_idx] for m in model_names)]
        per_scene_avg_iou.append(float(np.mean(scene_ious)))

    top_scene_indices = np.argsort(per_scene_avg_iou)[-2:][::-1]

    print(f"  [comparison_grid] Selected 2 best scenes (same images for all models)")

    # Columns
    n_cols = 2 + num_models + 1
    col_titles = ["RGB Input", "Ground Truth"] + \
                 [f"{m}\nPrediction" for m in model_names] + \
                 ["Error Map\n(best model)"]

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(3.3 * n_cols, 7.0),
        gridspec_kw={"wspace": 0.015, "hspace": 0.08},
    )
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.04, right=0.98, top=0.91, bottom=0.09)

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9, fontweight="bold", pad=6)

    for row_idx, scene_idx in enumerate(top_scene_indices):
        # Shared RGB / GT
        s0 = all_model_samples[model_names[0]][scene_idx]
        rgb_disp = _tensor_to_display_rgb(s0["rgb"])
        gt_disp = _mask_to_rgb(s0["gt"], palette)

        axes[row_idx, 0].imshow(rgb_disp, interpolation="nearest")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_ylabel(f"Scene {row_idx+1}", fontsize=10, rotation=0,
                                    labelpad=45, va="center")

        axes[row_idx, 1].imshow(gt_disp, interpolation="nearest")
        axes[row_idx, 1].axis("off")

        # Models + best error map
        best_iou = -1.0
        best_pred = None
        best_gt = None

        for m_idx, model_name in enumerate(model_names):
            s = all_model_samples[model_name][scene_idx]
            pred_disp = _mask_to_rgb(s["pred"], palette)
            ax = axes[row_idx, 2 + m_idx]
            ax.imshow(pred_disp, interpolation="nearest")
            ax.axis("off")

            ious = [_iou_single(s["pred"], s["gt"], c) for c in range(num_classes)]
            miou = float(np.mean(ious))
            ax.set_xlabel(f"mIoU={miou:.3f}", fontsize=8, color="#1a5276",
                          fontfamily="monospace")

            if miou > best_iou:
                best_iou = miou
                best_pred = s["pred"]
                best_gt = s["gt"]

        # Error map
        err = np.zeros((*best_gt.shape, 3), dtype=np.uint8)
        tp = (best_pred == best_gt) & (best_gt > 0)
        fp = (best_pred != best_gt) & (best_pred > 0)
        fn = (best_pred != best_gt) & (best_gt > 0)
        err[tp] = [0, 180, 80]
        err[fp] = [230, 40, 40]
        err[fn] = [255, 160, 0]

        axes[row_idx, -1].imshow(err, interpolation="nearest")
        axes[row_idx, -1].axis("off")
        axes[row_idx, -1].set_xlabel(f"Best mIoU={best_iou:.3f}",
                                     fontsize=8, color="#145a32", fontfamily="monospace")

    # Legend
    legend_patches = [mpatches.Patch(color=tuple(c/255 for c in palette[i]),
                                     label=class_names[i]) for i in range(num_classes)]
    error_patches = [
        mpatches.Patch(color=(0, 0.71, 0.31), label="TP"),
        mpatches.Patch(color=(0.90, 0.16, 0.16), label="FP"),
        mpatches.Patch(color=(1.0, 0.63, 0.0), label="FN"),
    ]
    fig.legend(handles=legend_patches + error_patches, loc="lower center",
               ncol=num_classes + 3, fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.005), title="Segmentation classes & error types")

    fig.suptitle("Cross-Model Qualitative Comparison — Test Split (2 Best Scenes)",
                 fontsize=13, fontweight="bold", y=0.96)

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / "comparison_grid_2row.png"
    fig.savefig(out_path, dpi=cfg.vis_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Fixed cross-model grid (same images) → {out_path}")

# =============================================================================
# VISUALISATION — Research-paper quality (fixed + improved)
# =============================================================================

# Colour palette — vivid, distinguishable, print-friendly
_CLASS_COLORS = np.array([
    [45,  45,  45],   # background
    [0,   200, 80],   # crop
    [230, 30,  30],   # weed
], dtype=np.uint8)

_OVERLAY_COLORS = np.array([
    [0.18, 0.18, 0.18, 0.00],   # bg — transparent
    [0.00, 0.78, 0.31, 0.55],   # crop — green
    [0.90, 0.12, 0.12, 0.65],   # weed — red
])


def _mask_to_rgb(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, colour in enumerate(palette):
        rgb[mask == c] = colour
    return rgb


def _tensor_to_display_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().float().numpy()          # (C, H, W)
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
    img  = img * std + mean
    img  = np.clip(img, 0.0, 1.0)
    img  = (img * 255).astype(np.uint8)
    return img.transpose(1, 2, 0)


def _nir_to_display(t: torch.Tensor) -> np.ndarray:
    nir = t.cpu().float().numpy().squeeze()
    nir = np.clip(nir, 0.0, 1.0)
    nir_u8 = (nir * 255).astype(np.uint8)
    return np.stack([nir_u8, nir_u8, nir_u8], axis=-1)


def _blend_overlay(rgb_hw3: np.ndarray, mask_hw: np.ndarray,
                   alpha_table: np.ndarray) -> np.ndarray:
    base = rgb_hw3.astype(np.float32) / 255.0
    out = base.copy()
    for c, rgba in enumerate(alpha_table):
        colour = np.array(rgba[:3], dtype=np.float32)
        alpha = rgba[3]
        region = (mask_hw == c)
        out[region] = (1 - alpha) * base[region] + alpha * colour
    return np.clip(out, 0.0, 1.0)


def _iou_single(pred_hw: np.ndarray, gt_hw: np.ndarray, cls: int, eps: float = 1e-6) -> float:
    p = (pred_hw == cls)
    g = (gt_hw == cls)
    return float((p & g).sum()) / float((p | g).sum() + eps)


def _extract_vis_channels(batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Always returns (rgb_B3HW, nir_B1HW, mask_BHW) regardless of loader type."""
    if "rgb" in batch:
        return batch["rgb"], batch["nir"], batch["mask"]
    else:
        # sugarbeets loader
        imgs = batch["images"]          # (B,4,H,W) or (B,3,H,W)
        lbls = batch["labels"]
        rgb = imgs[:, :3]
        nir = imgs[:, 3:4] if imgs.shape[1] == 4 else torch.zeros_like(imgs[:, :1])
        return rgb, nir, lbls


def visualize_predictions(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    cfg: BenchConfig,
    device: str,
    output_dir: str,
    forward_fn: Optional[Callable] = None,
    reference_indices: Optional[List[int]] = None,   # ← NEW: forces same images
) -> Tuple[str, List[Dict]]:
    """
    Research-paper quality visualisation grid + individual images.
    Now supports reference_indices to guarantee identical scenes across models.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    num_vis = min(cfg.num_vis_images, len(loader.dataset))
    class_names = cfg.class_names
    num_classes = cfg.num_classes
    palette = _CLASS_COLORS[:num_classes]

    # Use reference indices (from first model) if provided
    if reference_indices is not None:
        fixed_indices = reference_indices[:num_vis]
    else:
        fixed_indices = list(range(num_vis))

    # ── Collect samples (forward pass once per image) ───────────────────────
    samples: List[Dict] = []
    for idx in fixed_indices:
        batch = loader.dataset[idx]
        # Convert single sample → batched (B=1)
        if isinstance(batch, dict):
            batch = {
                k: v.unsqueeze(0) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

        rgb_b, nir_b, mask_b = _extract_vis_channels(batch)

        fn = forward_fn if forward_fn is not None else _fwd_dual_encoder
        logits, _ = fn(model, batch, device, cfg.use_amp)
        logits = logits.cpu().float()
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        for i in range(rgb_b.shape[0]):
            if len(samples) >= num_vis:
                break
            samples.append({
                "rgb":  rgb_b[i],
                "nir":  nir_b[i],
                "gt":   mask_b[i].numpy(),
                "pred": preds[i].numpy(),
            })

    # ── Best-3 grid per model (unchanged) ───────────────────────────────────
    for s in samples:
        ious = [_iou_single(s["pred"], s["gt"], c) for c in range(num_classes)]
        s["miou"] = float(np.mean(ious))

    samples_sorted = sorted(samples, key=lambda x: x["miou"], reverse=True)
    best3_samples = samples_sorted[:3]

    if best3_samples:
        # (Best-3 grid code is identical to what you already had — kept clean)
        n_cols = 6
        col_titles = ["RGB Input", "NIR Input", "Ground Truth",
                      "Prediction", "Pred Overlay", "Error Map"]
        cell_px = 3.4
        fig_w = cell_px * n_cols + 1.5
        fig_h = cell_px * 3 + 1.8

        fig, axes = plt.subplots(3, n_cols, figsize=(fig_w, fig_h),
                                 gridspec_kw={"wspace": 0.01, "hspace": 0.01})
        fig.patch.set_facecolor("white")
        plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.06)

        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=11, color="black",
                                 fontweight="bold", pad=8)

        for row_idx, s in enumerate(best3_samples):
            rgb_disp = _tensor_to_display_rgb(s["rgb"])
            nir_disp = _nir_to_display(s["nir"])
            gt_disp = _mask_to_rgb(s["gt"], palette)
            pred_disp = _mask_to_rgb(s["pred"], palette)
            overlay = _blend_overlay(rgb_disp, s["pred"], _OVERLAY_COLORS[:num_classes])

            err = np.zeros((*s["gt"].shape, 3), dtype=np.uint8)
            tp = (s["pred"] == s["gt"]) & (s["gt"] > 0)
            fp = (s["pred"] != s["gt"]) & (s["pred"] > 0)
            fn = (s["pred"] != s["gt"]) & (s["gt"] > 0)
            err[tp] = [0, 180, 80]
            err[fp] = [230, 40, 40]
            err[fn] = [255, 160, 0]

            panels = [rgb_disp, nir_disp, gt_disp, pred_disp, overlay, err]
            for col_idx, panel in enumerate(panels):
                ax = axes[row_idx, col_idx]
                ax.axis("off")
                ax.imshow(panel, interpolation="nearest")

            per_img_ious = [_iou_single(s["pred"], s["gt"], c) for c in range(num_classes)]
            iou_str = "  ".join(
                f"{class_names[c][0].upper()}:{per_img_ious[c]:.3f}"
                for c in range(num_classes)
            )
            axes[row_idx, 3].set_xlabel(iou_str, fontsize=8.5, color="#006400",
                                        labelpad=4, fontfamily="monospace")
            axes[row_idx, 0].set_ylabel(f"Best {row_idx+1}", fontsize=9,
                                        color="black", rotation=0, labelpad=35, va="center")

        # Legend (same as before)
        legend_patches = [mpatches.Patch(color=tuple(c/255 for c in palette[i]),
                                         label=class_names[i]) for i in range(num_classes)]
        error_patches = [
            mpatches.Patch(color=(0, 0.71, 0.31), label="TP"),
            mpatches.Patch(color=(0.90, 0.16, 0.16), label="FP"),
            mpatches.Patch(color=(1.0, 0.63, 0.0), label="FN"),
        ]
        fig.legend(handles=legend_patches + error_patches, loc="lower center",
                   ncol=num_classes + 3, fontsize=9, framealpha=0.9,
                   bbox_to_anchor=(0.5, 0.005), title="Legend")

        fig.suptitle(f"{model_name} — Top 3 Best Images (by mIoU)",
                     fontsize=13, color="black", fontweight="bold", y=0.98)

        best3_path = Path(output_dir) / f"{model_name}_best3_grid.png"
        fig.savefig(best3_path, dpi=cfg.vis_dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [✓] Best-3 grid → {best3_path}")

    # ── Grid layout (6 columns)
    n_cols = 6
    col_titles = [
        "RGB Input", "NIR Input", "Ground Truth",
        "Prediction", "Pred Overlay", "Error Map"
    ]

    cell_px = 3.4
    fig_w = cell_px * n_cols + 1.5
    fig_h = cell_px * len(samples) + 1.8

    fig, axes = plt.subplots(
        len(samples), n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.01, "hspace": 0.01},
    )
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.06)

    # Column titles
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, color="black",
                             fontweight="bold", pad=8)

    for row_idx, s in enumerate(samples):
        rgb_disp = _tensor_to_display_rgb(s["rgb"])
        nir_disp = _nir_to_display(s["nir"])
        gt_disp = _mask_to_rgb(s["gt"], palette)
        pred_disp = _mask_to_rgb(s["pred"], palette)
        overlay = _blend_overlay(rgb_disp, s["pred"], _OVERLAY_COLORS[:num_classes])

        # Error map
        err = np.zeros((*s["gt"].shape, 3), dtype=np.uint8)
        tp = (s["pred"] == s["gt"]) & (s["gt"] > 0)
        fp = (s["pred"] != s["gt"]) & (s["pred"] > 0)
        fn = (s["pred"] != s["gt"]) & (s["gt"] > 0)
        err[tp] = [0, 180, 80]      # green
        err[fp] = [230, 40, 40]     # red
        err[fn] = [255, 160, 0]     # orange

        panels = [rgb_disp, nir_disp, gt_disp, pred_disp, overlay, err]

        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            ax.imshow(panel, interpolation="nearest")

        # Per-image IoU annotation under Prediction column
        per_img_ious = [_iou_single(s["pred"], s["gt"], c) for c in range(num_classes)]
        iou_str = "  ".join(
            f"{class_names[c][0].upper()}:{per_img_ious[c]:.3f}"
            for c in range(num_classes)
        )
        axes[row_idx, 3].set_xlabel(iou_str, fontsize=8.5, color="#006400",
                                    labelpad=4, fontfamily="monospace")

        # Row label
        axes[row_idx, 0].set_ylabel(f"Sample {row_idx+1}", fontsize=9,
                                    color="black", rotation=0,
                                    labelpad=35, va="center")

    # Legend
    legend_patches = [
        mpatches.Patch(color=tuple(c/255 for c in palette[i]), label=class_names[i])
        for i in range(num_classes)
    ]
    error_patches = [
        mpatches.Patch(color=(0, 0.71, 0.31), label="TP"),
        mpatches.Patch(color=(0.90, 0.16, 0.16), label="FP"),
        mpatches.Patch(color=(1.0, 0.63, 0.0), label="FN"),
    ]
    fig.legend(
        handles=legend_patches + error_patches,
        loc="lower center", ncol=num_classes + 3,
        fontsize=9, framealpha=0.9, facecolor="white",
        edgecolor="black", labelcolor="black",
        bbox_to_anchor=(0.5, 0.005),
        title="Legend",
    )

    fig.suptitle(
        f"{model_name} — Qualitative Results (test split)",
        fontsize=13, color="black", fontweight="bold", y=0.98,
    )

    grid_path = Path(output_dir) / f"{model_name}_vis_grid.png"
    fig.savefig(grid_path, dpi=cfg.vis_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Grid saved → {grid_path}")

    # ── Individual publication-ready images (same layout)
    for row_idx, s in enumerate(samples):
        rgb_disp = _tensor_to_display_rgb(s["rgb"])
        nir_disp = _nir_to_display(s["nir"])
        gt_disp = _mask_to_rgb(s["gt"], palette)
        pred_disp = _mask_to_rgb(s["pred"], palette)
        overlay = _blend_overlay(rgb_disp, s["pred"], _OVERLAY_COLORS[:num_classes])

        err = np.zeros((*s["gt"].shape, 3), dtype=np.uint8)
        tp = (s["pred"] == s["gt"]) & (s["gt"] > 0)
        fp = (s["pred"] != s["gt"]) & (s["pred"] > 0)
        fn = (s["pred"] != s["gt"]) & (s["gt"] > 0)
        err[tp] = [0, 180, 80]
        err[fp] = [230, 40, 40]
        err[fn] = [255, 160, 0]

        per_img_ious = [_iou_single(s["pred"], s["gt"], c) for c in range(num_classes)]
        iou_str = "  |  ".join(
            f"{class_names[c]} IoU={per_img_ious[c]:.3f}" for c in range(num_classes)
        )

        n_single = 6
        f2, axs = plt.subplots(1, n_single, figsize=(3.8 * n_single, 4.2))
        f2.patch.set_facecolor("white")

        panels_single = [rgb_disp, nir_disp, gt_disp, pred_disp, overlay, err]
        titles_single = ["RGB Input", "NIR Input", "Ground Truth",
                         "Prediction", "Prediction Overlay", "Error Map"]

        for j, (panel, title) in enumerate(zip(panels_single, titles_single)):
            ax = axs[j]
            ax.axis("off")
            ax.imshow(panel, interpolation="nearest")
            ax.set_title(title, fontsize=10, color="black", fontweight="bold", pad=6)

        f2.suptitle(
            f"{model_name} — Sample {row_idx+1}   |   {iou_str}",
            fontsize=11, color="black", fontweight="bold", y=1.02,
        )

        f2.legend(
            handles=legend_patches + error_patches,
            loc="lower center", ncol=num_classes + 3,
            fontsize=9, framealpha=0.9, facecolor="white",
            edgecolor="black", labelcolor="black",
            bbox_to_anchor=(0.5, -0.08),
        )

        single_path = Path(output_dir) / f"{model_name}_vis_{row_idx+1:02d}.png"
        f2.savefig(single_path, dpi=cfg.vis_dpi * 1.2, bbox_inches="tight")
        plt.close(f2)

    print(f"  [✓] {len(samples)} individual high-res images saved → {output_dir}/")
    return str(grid_path), samples

def save_confusion_matrix(
    model_name:  str,
    conf_matrix: np.ndarray,       # shape (C, C) int64, from SegmentationMetrics
    class_names: List[str],
    output_dir:  str,
    dpi:         int = 150,
):
    """
    Save a row-normalised confusion matrix as a publication-quality PNG.
    Row = ground truth, column = predicted (standard convention).
    """
    cm_norm = conf_matrix.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm_norm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(class_names)
    ax.set_xticks(range(n));  ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=11, fontweight="bold")
    ax.set_title(f"{model_name}\nNormalised Confusion Matrix", fontsize=11,
                 fontweight="bold", pad=10)

    for i in range(n):
        for j in range(n):
            colour = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, color=colour, fontweight="bold")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{model_name}_confusion_matrix.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Confusion matrix → {out_path}")


def run_efficiency_benchmark(
    model_name: str, display_name: str, model: nn.Module,
    loader: DataLoader, device: str, cfg: BenchConfig,
    descriptor: "ModelDescriptor",
) -> Dict:
    """Batch=1, FP32, 100 runs → CSV D.
    Builds dummy input using the descriptor's loader_cfg so each model gets
    exactly the right number of input channels (no channel-mismatch crash).
    """
    model = model.to(device).eval()
    lc          = descriptor.loader_cfg
    target_size = lc.get("target_size", cfg.target_size)
    loader_type = lc.get("loader_type", "sugarbeets")

    # Build dummy that matches what forward_fn will actually receive
    if loader_type == "dual_encoder":
        # model(rgb_B3HW, nir_B1HW) — tuple
        dummy = (
            torch.zeros(1, 3, *target_size, device=device),
            torch.zeros(1, 1, *target_size, device=device),
        )
    else:
        # sugarbeets: model may be 4-ch teacher or 3-ch student.
        # For UNet students the forward_fn strips to [:3] — so the dummy
        # must replicate what forward_fn sends to the model, i.e. 3 channels.
        # For DeepLabV3 (use_rgbnir=True) it's 4 channels.
        # We check by inspecting the first conv of the model.
        first_conv = next(
            (m for m in model.modules() if isinstance(m, nn.Conv2d)), None
        )
        in_ch = first_conv.in_channels if first_conv is not None else 3
        dummy = torch.zeros(1, in_ch, *target_size, device=device)

    latencies = []
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"\n[Efficiency] {display_name} | batch=1 | {cfg.num_eff_runs} runs | "
          f"{target_size[0]}×{target_size[1]}")
    with torch.no_grad():
        for i in range(cfg.num_eff_runs + 10):   # +10 warmup
            if device != "cpu":
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(*dummy) if isinstance(dummy, tuple) else model(dummy)
                end.record()
                torch.cuda.synchronize()
                if i >= 10:
                    latencies.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(*dummy) if isinstance(dummy, tuple) else model(dummy)
                if i >= 10:
                    latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    gpu_peak = torch.cuda.max_memory_allocated(device) / 1e6 if device != "cpu" else -1

    result = {
        "model":             display_name,
        "lat_mean_ms":       round(float(lat.mean()), 3),
        "lat_std_ms":        round(float(lat.std()),  3),
        "lat_p95_ms":        round(float(np.percentile(lat, 95)), 3),
        "throughput_img_s":  round(1000 / lat.mean(), 2),
        "gpu_peak_MB":       round(gpu_peak, 1),
        "model_size_MB":     round(
            sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6, 2
        ),
        "hardware":          torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "runs":              cfg.num_eff_runs,
    }

    # Save / append CSV D
    Path(cfg.output_dir).mkdir(exist_ok=True)
    csv_path = Path(cfg.output_dir) / "efficiency_benchmark.csv"
    mode = "a" if csv_path.exists() else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if mode == "w":
            writer.writeheader()
        writer.writerow(result)

    print(f"  lat={result['lat_mean_ms']:.2f}±{result['lat_std_ms']:.2f} ms  "
          f"p95={result['lat_p95_ms']:.2f} ms  "
          f"tput={result['throughput_img_s']:.1f} img/s  "
          f"peak={result['gpu_peak_MB']:.0f} MB")
    print(f"  [✓] Efficiency → efficiency_benchmark.csv")
    return result

def run_kfold_statistical_eval_paper(
    model_name: str,
    display_name: str,
    model: nn.Module,
    loader: DataLoader,
    cfg: BenchConfig,
    device: str,
    forward_fn: Callable,
    n_folds: int = 5,
    output_dir: str = "/home/vjti-comp/WEEDSBL/scripts/benchmark_results",
) -> Dict:
    """Exactly as requested: deterministic folds, per-fold CSV, aggregated CSV B."""
    dataset = loader.dataset
    n = len(dataset)
    rng = np.random.default_rng(cfg.kfold_seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    print(f"\n[KFold] {display_name} | Test set N={n} | {n_folds} folds | seed={cfg.kfold_seed}")

    # Save fold indices for reproducibility
    fold_dict = {"seed": cfg.kfold_seed, "test_set_size": n, "folds": {}}
    for f in range(n_folds):
        start = f * fold_size
        end = n if f == n_folds - 1 else start + fold_size
        fold_dict["folds"][f"f{f}"] = indices[start:end].tolist()
    json.dump(fold_dict, open(Path(output_dir) / "fold_indices.json", "w"), indent=2)
    print(f"  [✓] Fold indices saved → {output_dir}/fold_indices.json")

    model = model.to(device).eval()
    per_fold_rows = []

    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        end = n if fold_idx == n_folds - 1 else start + fold_size
        subset = Subset(dataset, indices[start:end].tolist())
        fold_loader = DataLoader(subset, batch_size=cfg.batch_size_kfold,
                                 shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        tracker = SegmentationMetrics(cfg.num_classes, cfg.class_names)
        for batch in fold_loader:
            logits, mask = forward_fn(model, batch, device, cfg.use_amp)
            tracker.update(logits.argmax(dim=1), mask)

        m = tracker.compute()
        row = {"model": display_name, "fold": fold_idx + 1, "n_samples": end - start, **m}
        per_fold_rows.append(row)

    # Save CSV A: per-fold
    Path(output_dir).mkdir(exist_ok=True)
    csv_per_fold = Path(output_dir) / "kfold_per_fold.csv"
    csv_summary  = Path(output_dir) / "kfold_summary.csv"
    # Per-fold CSV (CSV A)
    mode = "a" if csv_per_fold.exists() else "w"
    with open(csv_per_fold, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_fold_rows[0].keys())
        if mode == "w":
            writer.writeheader()
        writer.writerows(per_fold_rows)

    # Aggregate → CSV B
    agg_rows = []
    report_keys = ["pixel_accuracy", "mean_accuracy", "mean_iou", "mean_dice"] + [f"iou_{c}" for c in cfg.class_names]
    for key in report_keys:
        vals = np.array([r[key] for r in per_fold_rows])
        mean = float(vals.mean())
        std = float(vals.std(ddof=1))
        se = std / np.sqrt(n_folds)
        t_crit = 2.776  # df=4, 95%
        ci_lo = mean - t_crit * se
        ci_hi = mean + t_crit * se
        agg_rows.append({
            "model": display_name,
            "metric": key,
            "mean": round(mean, 5),
            "std": round(std, 5),
            "ci95_lo": round(ci_lo, 5),
            "ci95_hi": round(ci_hi, 5),
            "n_folds": n_folds
        })

    # Aggregated summary CSV (CSV B)
    mode = "a" if csv_summary.exists() else "w"
    with open(csv_summary, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "mean", "std", "ci95_lo", "ci95_hi", "n_folds"])
        if mode == "w":
            writer.writeheader()
        writer.writerows(agg_rows)

    print(f"  [✓] CSV A (per-fold) → kfold_per_fold.csv")
    print(f"  [✓] CSV B (summary)  → kfold_summary.csv")
    return {"model": display_name, "n_test": n, "aggregated": agg_rows}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference benchmark harness")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Names of models to benchmark (default: all registered)",
    )
    parser.add_argument("--batch-size",   type=int, default=CFG.batch_size)
    parser.add_argument("--num-batches",  type=int, default=CFG.num_batches)
    parser.add_argument("--warmup",       type=int, default=CFG.warmup_batches)
    parser.add_argument("--split",        type=str, default=CFG.split,
                        choices=["train", "val", "test"])
    parser.add_argument("--no-amp",       action="store_true")
    parser.add_argument("--compile",      action="store_true")
    parser.add_argument("--output-dir",   type=str, default=CFG.output_dir)
    parser.add_argument("--no-vis",       action="store_true",
                        help="Skip visualisation (benchmark metrics only)")
    parser.add_argument("--num-vis",      type=int, default=CFG.num_vis_images,
                        help="Number of images to visualise per model (default 10)")
    parser.add_argument("--vis-dpi",      type=int, default=CFG.vis_dpi)
    parser.add_argument("--kfold",        type=int, default=5,
                        help="K-fold statistical eval on test set (e.g. --kfold 5). "
                             "0=disabled. Requires scipy.")
    parser.add_argument("--ablation",     action="store_true",
                        help="Run λ ablation study. Requires --ablation-teacher.")
    parser.add_argument("--ablation-student", type=str, default="distilled_unet_from_dual",
                        help="Model name (in registry) to use as student for ablation.")
    parser.add_argument("--ablation-teacher", type=str, default="dual_encoder_mini",
                        help="Model name (in registry) to use as teacher for ablation.")
    parser.add_argument("--efficiency",    action="store_true",
                        help="Run batch=1 latency benchmark for each model (saves efficiency_benchmark.csv).")
    args = parser.parse_args()

    # Apply CLI overrides to shared config
    CFG.batch_size      = args.batch_size
    CFG.num_batches     = args.num_batches
    CFG.warmup_batches  = args.warmup
    CFG.split           = args.split
    CFG.use_amp         = not args.no_amp
    CFG.use_compile     = args.compile
    CFG.output_dir      = args.output_dir
    CFG.num_vis_images  = args.num_vis
    CFG.vis_dpi         = args.vis_dpi

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Setup] device={device}  AMP={CFG.use_amp}  compile={CFG.use_compile}")
    if device == "cuda":
        print(f"        GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"        PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"        batch_size={CFG.batch_size}  num_batches={CFG.num_batches}  "
          f"warmup={CFG.warmup_batches}")

    # ── Determine which models to run ─────────────────────────────────────────
    to_run = args.models if args.models else list(MODEL_REGISTRY.keys())
    unknown = [m for m in to_run if m not in MODEL_REGISTRY]
    if unknown:
        print(f"[ERROR] Unknown model(s): {unknown}")
        print(f"        Registered: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    # ── Benchmark each model — each gets its own DataLoader ─────────────────
    #   Models can have different resolutions / batch schemas, so we build a
    #   fresh loader per model from its descriptor.loader_cfg.
    all_results: List[Dict] = []
    _loader_cache: Dict[str, DataLoader] = {}   # cache by loader_type+target_size

    all_model_samples: Dict[str, List[Dict]] = {}

    for name in to_run:
        desc = MODEL_REGISTRY[name]
        lc   = desc.loader_cfg

        # Build (or reuse) the DataLoader for this model
        cache_key = f"{lc.get('loader_type')}_{lc.get('target_size', CFG.target_size)}"
        if cache_key not in _loader_cache:
            print(f"\n[DataLoader] Building loader for {name}  "
                  f"({lc.get('loader_type')} / {lc.get('target_size', CFG.target_size)})…")
            _loader_cache[cache_key] = build_dataloader(CFG, lc)
        loader = _loader_cache[cache_key]

        print(f"\n[Loading] {name}…")
        model = desc.loader_fn()

        result, conf_matrix = benchmark_model(
            model_name=name,
            model=model,
            loader=loader,
            cfg=CFG,
            device=device,
            descriptor=desc,
        )
        all_results.append(result)

        # ── Confusion matrix ───────────────────────────────────────────────────
        vis_dir = Path(CFG.output_dir) / "visualisations"
        save_confusion_matrix(
            model_name=name,
            conf_matrix=conf_matrix,
            class_names=CFG.class_names,
            output_dir=str(vis_dir),
            dpi=CFG.vis_dpi,
        )

        # ── Visualise ─────────────────────────────────────────────────────────
        if not args.no_vis:
            print(f"\n[Visualise] {name}  ({CFG.num_vis_images} images)…")
            
            # First model becomes the REFERENCE (guarantees same images)
            reference_indices = None if not all_model_samples else list(range(CFG.num_vis_images))
            
            _, model_samples = visualize_predictions(
                model_name=name,
                model=model,
                loader=loader,
                cfg=CFG,
                device=device,
                output_dir=str(vis_dir),
                forward_fn=desc.forward_fn,
                reference_indices=reference_indices,   # ← this fixes the mismatch
            )
            all_model_samples[name] = model_samples

        # Free GPU memory between models
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Print + save ──────────────────────────────────────────────────────────
    print_summary_table(all_results, CFG.class_names)
    save_csv(all_results, CFG.output_dir, CFG.csv_name)

    # ── Structured JSON metrics (test-set numbers for the paper) ──────────────
    # Standard practice: always report on the TEST split, not val.
    # Val is for model selection; test is the final held-out evaluation.
    for r in all_results:
        r["split"] = CFG.split   # annotate which split was used
    save_metrics_json(all_results, CFG.output_dir)

    # ── Cross-model qualitative comparison grid ────────────────────────────────
    if not args.no_vis and len(all_model_samples) > 1:
        print("\n[ComparisonGrid] Building cross-model comparison figure…")
        vis_dir = Path(CFG.output_dir) / "visualisations"
        save_comparison_grid(
            all_model_samples=all_model_samples,
            cfg=CFG,
            output_dir=str(vis_dir),
        )
    elif not args.no_vis and len(all_model_samples) == 1:
        print("\n[ComparisonGrid] Only one model evaluated — "
              "skipping cross-model grid (run all models together for it).")

    # ── K-fold statistical evaluation ─────────────────────────────────────────
    if args.kfold > 1:
        print(f"\n[KFold] k={args.kfold}  split={CFG.split}")
        if CFG.split != "test":
            print("  [WARN] split is not 'test' — CEA reviewers expect test-set numbers!")
        all_kfold: List[Dict] = []
        for name in to_run:
            desc = MODEL_REGISTRY[name]
            lc   = desc.loader_cfg
            cache_key = f"{lc.get('loader_type')}_{lc.get('target_size', CFG.target_size)}"
            loader = _loader_cache[cache_key]
            print(f"\n[KFold] Loading {name}…")
            model = desc.loader_fn()
            kf_result = run_kfold_statistical_eval_paper(
                name, name, model, loader, CFG, device,
                desc.forward_fn, n_folds=args.kfold, output_dir=CFG.output_dir
            )
            all_kfold.append(kf_result)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        # save_kfold_summary_csv(all_kfold, CFG.output_dir)

    # ── λ Ablation study ──────────────────────────────────────────────────────
    if args.ablation:
        s_name = args.ablation_student
        t_name = args.ablation_teacher
        for nm in [s_name, t_name]:
            if nm not in MODEL_REGISTRY:
                print(f"[ERROR] --ablation: '{nm}' not in MODEL_REGISTRY. "
                      f"Registered: {list(MODEL_REGISTRY.keys())}")
                sys.exit(1)

        s_desc = MODEL_REGISTRY[s_name]
        t_desc = MODEL_REGISTRY[t_name]
        lc_s   = s_desc.loader_cfg
        cache_key = f"{lc_s.get('loader_type')}_{lc_s.get('target_size', CFG.target_size)}"
        if cache_key not in _loader_cache:
            _loader_cache[cache_key] = build_dataloader(CFG, lc_s)
        abl_loader = _loader_cache[cache_key]

        print(f"\n[Ablation] Loading student={s_name}  teacher={t_name}…")
        s_model = s_desc.loader_fn()
        t_model = t_desc.loader_fn()

        run_ablation_study(
            student_name=s_name,   student_model=s_model,
            teacher_name=t_name,   teacher_model=t_model,
            loader=abl_loader,     cfg=CFG,
            device=device,
            student_fwd_fn=s_desc.forward_fn,
            teacher_fwd_fn=t_desc.forward_fn,
            output_dir=CFG.output_dir,
        )
        del s_model, t_model
        if device == "cuda":
            torch.cuda.empty_cache()


# ckpt_path is now stored in each ModelDescriptor — see MODEL_REGISTRY above.


if __name__ == "__main__":
    main()