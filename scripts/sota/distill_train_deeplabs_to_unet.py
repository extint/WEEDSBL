"""
Knowledge Distillation: DeepLabV3 (RGB or RGB+NIR) → UNet (RGB only)

Strategy:
  - Task loss : CE + Dice (same combo used in teacher training: MultiClassDiceLoss)
  - Logit KD  : KL divergence with temperature scaling (T=4), boundary-pixel upweighting
  - Lovász    : Optionally added on top of task loss (disabled by default, flag to enable)

Teacher always in eval mode, fully frozen.
Student trains on RGB only — even if the teacher was trained on RGB+NIR (4ch).

Key design choices
──────────────────
• Teacher outputs a dict  {"out": logits, "aux": ...}  (torchvision DeepLabV3 / LRASPP API).
  We only use the "out" key for distillation.
• Student UNet returns a plain tensor — imported from sota.models.
• Data loader is `sugarbeets_data_loader.create_sugarbeets_dataloaders`.
  Batch keys: "images" (B,3,H,W  or  B,4,H,W) and "labels" (B,H,W).
  The student always receives only the first 3 channels (RGB) regardless of how
  the teacher was trained.
• Feature adapters are intentionally omitted: DeepLabV3's internal feature maps
  are not trivially hookable without architecture surgery, and the UNet bottleneck
  dimensionality differs significantly — logit KD alone provides strong guidance.

Usage
─────
  python distill_deeplab_to_unet.py \\
      --teacher_ckpt /path/to/best_model.pth \\
      --teacher_arch deeplabv3_resnet50 \\
      --teacher_rgbnir \\                          # omit if teacher was RGB-only
      --data_root /path/to/SUGARBEETS_AUGMENTED_DATASET \\
      --output_dir ./experiments_distill \\
      --epochs 100 --batch_size 4

Supported teacher architectures (--teacher_arch):
  deeplabv3_resnet50   (~42M)
  deeplabv3_resnet101  (~61M)
  deeplabv3_mobilenet  (~11M)
  lraspp_mobilenet     (~3.2M)
"""

import argparse
import json
import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import (
    deeplabv3_resnet50,    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet101,   DeepLabV3_ResNet101_Weights,
    deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,    LRASPP_MobileNet_V3_Large_Weights,
)
from tqdm import tqdm

from sugarbeets_data_loader import create_sugarbeets_dataloaders
from sota.models import UNet

# ── optional FLOPs counter ───────────────────────────────────────────────────
try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# ── constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['background', 'crop', 'weed']
NUM_CLASSES  = 3
RGB_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD      = np.array([0.229, 0.224, 0.225],  dtype=np.float32)
MASK_COLORS  = np.array([[38, 38, 38], [51, 204, 51], [230, 51, 51]], dtype=np.uint8)

# ── teacher registry (mirrors finetune.py) ───────────────────────────────────
_MODEL_REGISTRY = {
    "deeplabv3_resnet50":  (deeplabv3_resnet50,  DeepLabV3_ResNet50_Weights,  "resnet"),
    "deeplabv3_resnet101": (deeplabv3_resnet101,  DeepLabV3_ResNet101_Weights, "resnet"),
    "deeplabv3_mobilenet": (deeplabv3_mobilenet_v3_large,
                             DeepLabV3_MobileNet_V3_Large_Weights, "mobilenet"),
    "lraspp_mobilenet":    (lraspp_mobilenet_v3_large,
                             LRASPP_MobileNet_V3_Large_Weights, "lraspp"),
}


# =============================================================================
# 1.  Teacher builder — reconstructs architecture exactly as in finetune.py
# =============================================================================

def _get_first_conv2d_path(module: nn.Module):
    """BFS to find the first Conv2d in a module tree."""
    queue = deque()
    for name, child in module.named_children():
        queue.append((module, name, child))
    while queue:
        parent, attr, mod = queue.popleft()
        if isinstance(mod, nn.Conv2d):
            return parent, attr, mod
        for name, child in mod.named_children():
            queue.append((mod, name, child))
    return None


def _patch_input_conv_resnet(model: nn.Module):
    old = model.backbone.conv1
    new = nn.Conv2d(4, old.out_channels, old.kernel_size,
                    old.stride, old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
    model.backbone.conv1 = new


def _patch_input_conv_mobilenet(model: nn.Module):
    result = _get_first_conv2d_path(model.backbone)
    if result is None:
        raise RuntimeError("Could not find first Conv2d in MobileNet backbone.")
    parent, attr, old = result
    new = nn.Conv2d(4, old.out_channels, old.kernel_size,
                    old.stride, old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(parent, attr, new)


def build_teacher(architecture: str, num_classes: int, use_rgbnir: bool) -> nn.Module:
    """
    Rebuild the teacher architecture exactly as finetune.py did, then load weights.
    We pass pretrained=False here because we will load the fine-tuned checkpoint.
    """
    if architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{architecture}'. "
                         f"Choose from: {list(_MODEL_REGISTRY.keys())}")

    loader_fn, weights_cls, family = _MODEL_REGISTRY[architecture]
    # Load with COCO weights first so we can replicate the exact param layout,
    # then overwrite with the fine-tuned checkpoint below.
    model = loader_fn(weights=None)

    # Patch input channels for RGB+NIR (must match how teacher was trained)
    if use_rgbnir:
        if family == "resnet":
            _patch_input_conv_resnet(model)
        else:
            _patch_input_conv_mobilenet(model)

    # Replace classifier heads (mirrors finetune.py exactly)
    if family == "lraspp":
        in_low  = model.classifier.low_classifier.in_channels
        in_high = model.classifier.high_classifier.in_channels
        model.classifier.low_classifier  = nn.Conv2d(in_low,  num_classes, 1)
        model.classifier.high_classifier = nn.Conv2d(in_high, num_classes, 1)
    else:
        in_main = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_main, num_classes, 1)
        if model.aux_classifier is not None:
            in_aux = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(in_aux, num_classes, 1)

    return model


def load_teacher(ckpt_path: str, architecture: str,
                 use_rgbnir: bool, num_classes: int, device) -> nn.Module:
    model = build_teacher(architecture, num_classes, use_rgbnir)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  [WARN] Teacher missing keys: {missing[:3]}'
              f'{"…" if len(missing) > 3 else ""}')
    if unexpected:
        print(f'  [WARN] Unexpected keys: {unexpected[:3]}'
              f'{"…" if len(unexpected) > 3 else ""}')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def teacher_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the teacher.
    torchvision DeepLabV3 / LRASPP always returns a dict; we extract 'out'.
    Input x may be 3ch or 4ch depending on teacher training.
    """
    out = model(x)
    if isinstance(out, dict):
        return out['out']
    return out  # safety fallback


# =============================================================================
# 2.  Loss functions
# =============================================================================

class TaskLoss(nn.Module):
    """
    CE + Soft-Dice  (same formula as finetune.py MultiClassDiceLoss,
    extended with optional per-class weighting for the CE term).
    """
    def __init__(self, class_weights: List[float], num_classes: int = 3):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer('w', w)
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_ce = F.cross_entropy(logits, targets, weight=self.w)

        probs = F.softmax(logits, dim=1)
        oh    = F.one_hot(targets.clamp(0), self.num_classes) \
                 .permute(0, 3, 1, 2).float()
        dims  = (0, 2, 3)
        inter = (probs * oh).sum(dims)
        union = probs.sum(dims) + oh.sum(dims)
        dice  = 1.0 - (2 * inter + 1e-6) / (union + 1e-6)
        l_dice = dice.mean()

        return 0.5 * l_ce + 0.5 * l_dice


class KLDivLoss(nn.Module):
    """
    Soft-target KL divergence logit distillation.
    Boundary pixels are up-weighted 3× to sharpen edge learning.
    Loss is scaled by T² so gradients stay on the same scale as task loss.
    """
    def __init__(self, T: float = 4.0):
        super().__init__()
        self.T = T

    def forward(self, s_logits: torch.Tensor, t_logits: torch.Tensor,
                boundary_mask: torch.Tensor = None) -> torch.Tensor:
        s = F.log_softmax(s_logits / self.T, dim=1)
        t = F.softmax(t_logits / self.T,     dim=1)
        kl = F.kl_div(s, t, reduction='none').sum(dim=1)   # (B,H,W)
        if boundary_mask is not None:
            kl = kl * (1.0 + 2.0 * boundary_mask.float())
        return kl.mean() * (self.T ** 2)


def _boundary_mask(targets: torch.Tensor) -> torch.Tensor:
    """Return a float boundary map (B,H,W) using dilation-erosion difference."""
    t = targets.float().unsqueeze(1)
    dilated = F.max_pool2d(t,  3, stride=1, padding=1)
    eroded  = -F.max_pool2d(-t, 3, stride=1, padding=1)
    return ((dilated - eroded).abs() > 0.1).squeeze(1)


# =============================================================================
# 3.  Metrics
# =============================================================================

class SegMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p  = preds.cpu().numpy().flatten()
        t  = targets.cpu().numpy().flatten()
        ok = (t >= 0) & (t < NUM_CLASSES)
        np.add.at(self.cm, (t[ok], p[ok]), 1)

    def compute(self) -> Dict:
        cm  = self.cm.astype(np.float64)
        TP  = np.diag(cm)
        FP  = cm.sum(0) - TP
        FN  = cm.sum(1) - TP
        TN  = cm.sum() - (TP + FP + FN)
        eps = 1e-6
        iou  = TP / (TP + FP + FN + eps)
        dice = 2 * TP / (2 * TP + FP + FN + eps)
        prec = TP / (TP + FP + eps)
        rec  = TP / (TP + FN + eps)
        spec = TN / (TN + FP + eps)
        pacc = TP.sum() / (cm.sum() + eps)
        macc = (TP / (cm.sum(1) + eps)).mean()
        r = {
            'pixel_acc':  float(pacc),
            'mean_acc':   float(macc),
            'miou':       float(iou.mean()),
            'mean_dice':  float(dice.mean()),
            'mean_f1':    float(dice.mean()),
            'mean_prec':  float(prec.mean()),
            'mean_rec':   float(rec.mean()),
        }
        for c, n in enumerate(CLASS_NAMES):
            r[f'iou_{n}']  = float(iou[c])
            r[f'dice_{n}'] = float(dice[c])
            r[f'f1_{n}']   = float(dice[c])
            r[f'prec_{n}'] = float(prec[c])
            r[f'rec_{n}']  = float(rec[c])
            r[f'spec_{n}'] = float(spec[c])
        return r

    def print_table(self, r: Dict, phase: str = 'Val'):
        print(f"\n{'─'*65}")
        print(f"  {phase} | PixAcc={r['pixel_acc']:.4f}  MeanAcc={r['mean_acc']:.4f}"
              f"  mIoU={r['miou']:.4f}  mF1={r['mean_f1']:.4f}")
        print(f"{'─'*65}")
        print(f"  {'Class':<12} {'IoU':>7} {'Dice/F1':>8} {'Prec':>7} {'Rec':>7} {'Spec':>7}")
        for n in CLASS_NAMES:
            print(f"  {n:<12} {r[f'iou_{n}']:>7.4f} {r[f'dice_{n}']:>8.4f}"
                  f" {r[f'prec_{n}']:>7.4f} {r[f'rec_{n}']:>7.4f} {r[f'spec_{n}']:>7.4f}")
        print(f"{'─'*65}\n")


# =============================================================================
# 4.  Visualisation helpers
# =============================================================================

def _denorm(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().float().numpy().transpose(1, 2, 0)
    img = img * RGB_STD + RGB_MEAN
    return (img.clip(0, 1) * 255).astype(np.uint8)


def _colorise(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(MASK_COLORS):
        out[mask == c] = col
    return out


def _add_legend(ax):
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=MASK_COLORS[i] / 255., label=CLASS_NAMES[i])
               for i in range(NUM_CLASSES)]
    ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.7)


def save_vis_grid(samples, student, teacher, device,
                  save_path: str, epoch: int, teacher_uses_rgbnir: bool):
    """
    5-column grid per sample:
      col0: RGB  col1: GT  col2: Student pred  col3: Teacher ref  col4: Error map
    Samples are dicts with keys 'images' (3 or 4 ch) and 'labels'.
    """
    student.eval()
    teacher.eval()
    n   = len(samples)
    fig = plt.figure(figsize=(20, 4 * n))
    gs  = gridspec.GridSpec(n, 5, figure=fig, hspace=0.05, wspace=0.05)
    col_titles = ['RGB', 'GT', 'Student pred', 'Teacher ref', 'Error map']

    for row, s in enumerate(samples):
        img_t  = s['images'].unsqueeze(0).to(device)   # (1, 3or4, H, W)
        rgb_t  = img_t[:, :3]                           # (1, 3, H, W) — student input
        mask   = s['labels'].cpu().numpy()              # (H, W)

        with torch.no_grad():
            s_pred = student(rgb_t).argmax(1)[0].cpu().numpy()
            # Teacher may need 3ch or 4ch depending on training
            t_in   = img_t if teacher_uses_rgbnir else rgb_t
            t_pred = teacher_logits(teacher, t_in).argmax(1)[0].cpu().numpy()

        rgb_img = _denorm(s['images'][:3])
        gt_col  = _colorise(mask)
        st_col  = _colorise(s_pred)
        te_col  = _colorise(t_pred)

        err = np.ones((*mask.shape, 3), dtype=np.uint8) * 180
        err[(s_pred > 0) & (mask == 0)]                     = [0,   220, 220]   # FP cyan
        err[(s_pred == 0) & (mask > 0)]                     = [220, 220, 0]    # FN yellow
        err[(s_pred != mask) & (s_pred > 0) & (mask > 0)]  = [200, 0,   200]   # wrong magenta

        ious = []
        for c in range(NUM_CLASSES):
            tp = ((s_pred == c) & (mask == c)).sum()
            fp = ((s_pred == c) & (mask != c)).sum()
            fn = ((s_pred != c) & (mask == c)).sum()
            ious.append(tp / (tp + fp + fn + 1e-6))
        miou    = np.mean(ious)
        iou_str = '/'.join(f'{v:.2f}' for v in ious)

        imgs = [rgb_img, gt_col, st_col, te_col, err]
        for col, (img, title) in enumerate(zip(imgs, col_titles)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=9, fontweight='bold', pad=3)
            if col == 2:
                ax.set_xlabel(f'mIoU={miou:.3f}\n[{iou_str}]', fontsize=7, labelpad=2)
            if col == 1 and row == 0:
                _add_legend(ax)

    fig.suptitle(f'Distillation — Epoch {epoch}', fontsize=11, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 5.  Latency helper
# =============================================================================

def measure_latency(model, in_ch: int, H: int, W: int,
                    device, runs: int = 50, warmup: int = 10) -> float:
    model.eval()
    dummy = torch.randn(1, in_ch, H, W, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            out = model(dummy)
            # consume dict output so CUDA kernel completes
            _ = out['out'] if isinstance(out, dict) else out

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            out = model(dummy)
            _ = out['out'] if isinstance(out, dict) else out
        if device.type == 'cuda':
            torch.cuda.synchronize()

    return (time.perf_counter() - t0) / runs * 1000.0


# =============================================================================
# 6.  Training epoch
# =============================================================================

def train_one_epoch(teacher, student, loader, optimizer, scaler,
                    task_fn, kd_fn, device, epoch,
                    w_task: float, w_kd: float,
                    teacher_uses_rgbnir: bool, writer: SummaryWriter):
    student.train()
    teacher.eval()
    metrics = SegMetrics()

    tot = tot_task = tot_kd = 0.0
    n = 0

    pbar = tqdm(loader, desc=f'Ep {epoch:03d} [Distill]')
    for batch in pbar:
        images = batch['images'].to(device)   # (B, 3or4, H, W)
        mask   = batch['labels'].to(device).long()

        rgb = images[:, :3]   # student always gets RGB only

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            with torch.no_grad():
                t_in     = images if teacher_uses_rgbnir else rgb
                t_logits = teacher_logits(teacher, t_in)

            s_logits = student(rgb)

            # Resize student logits to match mask spatial size if needed
            # (UNet should output same H×W as input, but guard anyway)
            if s_logits.shape[-2:] != mask.shape[-2:]:
                s_logits = F.interpolate(s_logits, size=mask.shape[-2:],
                                         mode='bilinear', align_corners=False)

            # Align teacher logits spatial size to student (both should be H×W,
            # but DeepLabV3 can sometimes differ by 1 px)
            if t_logits.shape[-2:] != s_logits.shape[-2:]:
                t_logits = F.interpolate(t_logits, size=s_logits.shape[-2:],
                                         mode='bilinear', align_corners=False)

            l_task = task_fn(s_logits, mask)
            bmask  = _boundary_mask(mask)
            l_kd   = kd_fn(s_logits, t_logits, bmask)
            loss   = w_task * l_task + w_kd * l_kd

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            metrics.update(s_logits.argmax(1), mask)

        tot      += loss.item()
        tot_task += l_task.item()
        tot_kd   += l_kd.item()
        n += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'task': f'{l_task.item():.4f}',
                          'kd':   f'{l_kd.item():.4f}'})

    r = metrics.compute()
    r.update({'loss': tot / n, 'loss_task': tot_task / n, 'loss_kd': tot_kd / n})
    for k, v in r.items():
        writer.add_scalar(f'Train/{k}', v, epoch)
    return r


# =============================================================================
# 7.  Validation / test evaluation
# =============================================================================

@torch.no_grad()
def evaluate(student, loader, task_fn, device, epoch,
             writer: SummaryWriter, split: str = 'Val'):
    student.eval()
    metrics = SegMetrics()
    tot = 0.0
    n   = 0
    for batch in tqdm(loader, desc=f'[{split}]'):
        rgb    = batch['images'][:, :3].to(device)
        mask   = batch['labels'].to(device).long()
        logits = student(rgb)
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(logits, size=mask.shape[-2:],
                                   mode='bilinear', align_corners=False)
        tot += task_fn(logits, mask).item()
        metrics.update(logits.argmax(1), mask)
        n += 1
    r = metrics.compute()
    r['loss'] = tot / n
    for k, v in r.items():
        writer.add_scalar(f'{split}/{k}', v, epoch)
    return r


# =============================================================================
# 8.  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Distill DeepLabV3 teacher → UNet RGB-only student')

    # ── paths ─────────────────────────────────────────────────────────────────
    parser.add_argument('--teacher_ckpt', type=str, required=True,
                        help='Path to the fine-tuned DeepLabV3 checkpoint (best_model.pth)')
    parser.add_argument('--data_root', type=str,
                        default='/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET')
    parser.add_argument('--output_dir', type=str,
                        default='/home/vjti-comp/WEEDSBL/scripts/sota/experiments_distill')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--resume',   type=str, default=None,
                        help='Path to student distillation checkpoint to resume')

    # ── teacher config ────────────────────────────────────────────────────────
    parser.add_argument('--teacher_arch', type=str, default='deeplabv3_mobilenet',
                        choices=list(_MODEL_REGISTRY.keys()),
                        help='Architecture that was used to train the teacher')
    parser.add_argument('--teacher_rgbnir', action='store_true',
                        help='Teacher was trained on RGB+NIR (4ch). '
                             'Omit this flag if teacher was trained on RGB only.')

    # ── student ───────────────────────────────────────────────────────────────
    parser.add_argument('--student_base_ch', type=int, default=8,
                        help='UNet base channels: 8→~0.1M, 16→~0.5M, 32→~1.9M')

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument('--height',      type=int, default=640)
    parser.add_argument('--width',       type=int, default=640)
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    # ── training ──────────────────────────────────────────────────────────────trainin
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--min_lr',        type=float, default=1e-6)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)

    # ── distillation weights ──────────────────────────────────────────────────
    parser.add_argument('--w_task', type=float, default=1.0,
                        help='Task loss weight (CE + Dice)')
    parser.add_argument('--w_kd',   type=float, default=0.5,
                        help='KL divergence distillation weight')
    parser.add_argument('--kd_temp', type=float, default=4.0,
                        help='Temperature for KD softmax')

    # ── class weights ─────────────────────────────────────────────────────────
    parser.add_argument('--class_weights', type=float, nargs=3,
                        default=[0.3, 1.0, 3.0],
                        help='Per-class CE weights: [bg, crop, weed]')

    # ── visualisation ─────────────────────────────────────────────────────────
    parser.add_argument('--vis_samples', type=int, default=5)
    parser.add_argument('--vis_every',   type=int, default=10)

    args = parser.parse_args()

    # ── directories ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.exp_name is None:
        args.exp_name = (f'distill_{args.teacher_arch}_UNet'
                         f'_S{args.student_base_ch}_{ts}')

    run_dir  = Path(args.output_dir) / args.exp_name
    ckpt_dir = run_dir / 'checkpoints'
    vis_dir  = run_dir / 'vis'
    tb_dir   = run_dir / 'tensorboard'
    for d in [ckpt_dir, vis_dir, tb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config = {**vars(args),
              'class_names': CLASS_NAMES,
              'teacher': args.teacher_arch,
              'student': 'UNet'}
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f'  Distillation: {args.teacher_arch} → UNet (RGB only)')
    print(f'  Teacher input: {"RGB+NIR (4ch)" if args.teacher_rgbnir else "RGB (3ch)"}')
    print(f'  Student input: RGB (3ch)  |  base_ch={args.student_base_ch}')
    print(f'  Run dir: {run_dir}')
    print(f"{'='*70}\n")

    writer = SummaryWriter(str(tb_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {device}')

    # ── data ──────────────────────────────────────────────────────────────────
    print('[INFO] Loading data...')
    # We load with use_rgbnir=True if teacher needs NIR so vis samples include
    # the NIR channel; the student always strips it to [:3] during training.
    load_rgbnir = args.teacher_rgbnir
    train_loader, val_loader, test_loader = create_sugarbeets_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.height, args.width),
        use_rgbnir=load_rgbnir,
    )
    print(f'[INFO] Train={len(train_loader.dataset)} '
          f'Val={len(val_loader.dataset)} Test={len(test_loader.dataset)}')

    all_val     = list(val_loader.dataset)
    vis_samples = random.sample(all_val, min(args.vis_samples, len(all_val)))

    # ── teacher ───────────────────────────────────────────────────────────────
    print(f'[INFO] Loading teacher ({args.teacher_arch})...')
    teacher = load_teacher(args.teacher_ckpt, args.teacher_arch,
                           args.teacher_rgbnir, NUM_CLASSES, device)
    t_in_ch = 4 if args.teacher_rgbnir else 3
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f'[INFO] Teacher params: {t_params / 1e6:.3f}M')

    # ── student ───────────────────────────────────────────────────────────────
    print(f'[INFO] Building student UNet (base_ch={args.student_base_ch})...')
    student = UNet(
        in_channels=3,
        base_ch=args.student_base_ch,
        out_channels=NUM_CLASSES,
    ).to(device)
    s_params = sum(p.numel() for p in student.parameters())
    print(f'[INFO] Student params:  {s_params / 1e6:.3f}M')
    print(f'[INFO] Compression:     {s_params / t_params:.4f}× '
          f'({(1 - s_params / t_params) * 100:.1f}% smaller)')

    # ── losses ────────────────────────────────────────────────────────────────
    task_fn = TaskLoss(args.class_weights, NUM_CLASSES).to(device)
    kd_fn   = KLDivLoss(T=args.kd_temp)

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(student.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))
    warmup  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                        total_iters=args.warmup_epochs)
    cosine  = CosineAnnealingLR(optimizer,
                                 T_max=max(1, args.epochs - args.warmup_epochs),
                                 eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                              milestones=[args.warmup_epochs])
    scaler = GradScaler('cuda')

    # ── latency ───────────────────────────────────────────────────────────────
    t_lat = measure_latency(teacher, t_in_ch, args.height, args.width, device)
    s_lat = measure_latency(student, 3,        args.height, args.width, device)
    print(f'[INFO] Teacher latency: {t_lat:.1f} ms  ({t_in_ch}ch input)')
    print(f'[INFO] Student latency: {s_lat:.1f} ms  (3ch RGB input)')

    writer.add_scalar('Model/Teacher_params_M',   t_params / 1e6, 0)
    writer.add_scalar('Model/Student_params_M',   s_params / 1e6, 0)
    writer.add_scalar('Model/Teacher_latency_ms', t_lat, 0)
    writer.add_scalar('Model/Student_latency_ms', s_lat, 0)

    model_cfg = {
        'teacher': {
            'class':          args.teacher_arch,
            'input_channels': t_in_ch,
            'num_classes':    NUM_CLASSES,
            'params_M':       t_params / 1e6,
            'latency_ms':     t_lat,
        },
        'student': {
            'class':          'UNet',
            'base_ch':        args.student_base_ch,
            'input_channels': 3,
            'num_classes':    NUM_CLASSES,
            'params_M':       s_params / 1e6,
            'latency_ms':     s_lat,
        },
        'distill': {
            'w_task':   args.w_task,
            'w_kd':     args.w_kd,
            'kd_temp':  args.kd_temp,
        },
    }
    with open(run_dir / 'model_config.json', 'w') as f:
        json.dump(model_cfg, f, indent=2)

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_miou = 0.0
    if args.resume:
        print(f'[INFO] Resuming from {args.resume}')
        r = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(r['student_state_dict'])
        optimizer.load_state_dict(r['optimizer_state_dict'])
        if 'scheduler_state_dict' in r:
            scheduler.load_state_dict(r['scheduler_state_dict'])
        start_epoch   = r['epoch'] + 1
        best_val_miou = r.get('best_val_miou', 0.0)
        print(f'[INFO] Resumed ep={r["epoch"]}  best_miou={best_val_miou:.4f}')

    # ── training loop ─────────────────────────────────────────────────────────
    print(f'\n[INFO] Starting distillation for {args.epochs} epochs...\n')

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f'Epoch {epoch}/{args.epochs}  '
              f'LR={optimizer.param_groups[0]["lr"]:.2e}')
        print('='*60)

        train_r = train_one_epoch(
            teacher, student, train_loader, optimizer, scaler,
            task_fn, kd_fn, device, epoch,
            args.w_task, args.w_kd, args.teacher_rgbnir, writer,
        )
        scheduler.step()

        val_r = evaluate(student, val_loader, task_fn, device, epoch, writer, 'Val')

        for phase, r in [('Train', train_r), ('Val', val_r)]:
            m = SegMetrics()
            m.print_table(r, phase)

        print(f"  KD losses → task={train_r['loss_task']:.4f} "
              f"kd={train_r['loss_kd']:.4f}")

        writer.add_scalar('Epoch/LR', optimizer.param_groups[0]['lr'], epoch)

        is_best   = val_r['miou'] > best_val_miou
        is_vis_ep = (epoch % args.vis_every == 0 or epoch == 1)

        if is_best:
            best_val_miou = val_r['miou']
            best_path = ckpt_dir / 'best_student.pth'
            torch.save({
                'epoch':                epoch,
                'student_state_dict':   student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_miou':        best_val_miou,
                'val_metrics':          val_r,
                'model_config':         model_cfg,
                'train_config':         config,
            }, best_path)
            print(f'  [✓] Best student saved  (mIoU={best_val_miou:.4f})')
            vis_path = vis_dir / f'best_epoch_{epoch:03d}.png'
            save_vis_grid(vis_samples, student, teacher, device,
                          str(vis_path), epoch, args.teacher_rgbnir)

        if is_vis_ep and not is_best:
            vis_path = vis_dir / f'epoch_{epoch:03d}.png'
            save_vis_grid(vis_samples, student, teacher, device,
                          str(vis_path), epoch, args.teacher_rgbnir)

        if epoch % 10 == 0:
            ep_path = ckpt_dir / f'epoch_{epoch:03d}.pth'
            torch.save({
                'epoch':                epoch,
                'student_state_dict':   student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_miou':        best_val_miou,
                'model_config':         model_cfg,
            }, ep_path)
            # Keep only the last 3 periodic checkpoints
            all_ep = sorted(ckpt_dir.glob('epoch_*.pth'))
            for old in all_ep[:-3]:
                old.unlink()

    # ── final test evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print('[INFO] Final test evaluation with best student checkpoint...')
    best_ckpt = torch.load(ckpt_dir / 'best_student.pth',
                           map_location=device, weights_only=False)
    student.load_state_dict(best_ckpt['student_state_dict'])

    test_r       = evaluate(student, test_loader, task_fn, device,
                            args.epochs + 1, writer, 'Test')
    s_final_lat  = measure_latency(student, 3, args.height, args.width, device)
    test_r['latency_ms'] = s_final_lat
    test_r['params_M']   = s_params / 1e6

    test_samples = random.sample(list(test_loader.dataset),
                                 min(args.vis_samples, len(test_loader.dataset)))
    save_vis_grid(test_samples, student, teacher, device,
                  str(vis_dir / 'final_test.png'), args.epochs,
                  args.teacher_rgbnir)

    sm = SegMetrics()
    sm.print_table(test_r, 'Test')
    print(f'[Test] Params={s_params / 1e6:.3f}M  Latency={s_final_lat:.1f}ms')

    results = {
        'test_metrics':       test_r,
        'best_val_miou':      best_val_miou,
        'teacher_params_M':   t_params / 1e6,
        'student_params_M':   s_params / 1e6,
        'teacher_latency_ms': t_lat,
        'student_latency_ms': s_final_lat,
        'compression_ratio':  s_params / t_params,
        'model_config':       model_cfg,
        'train_config':       config,
    }
    with open(run_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    for k, v in test_r.items():
        writer.add_scalar(f'Final/Test_{k}', v, 0)

    writer.close()
    print(f'\n[INFO] Done. Results → {run_dir}')
    print(f'[INFO] Best Val mIoU={best_val_miou:.4f}  '
          f'Test mIoU={test_r["miou"]:.4f}')
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()