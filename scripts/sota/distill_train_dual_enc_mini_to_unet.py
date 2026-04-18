"""
Knowledge Distillation: DualEncoderMini (RGB+NIR, 0.362M) → UNet (RGB only)

Strategy:
  - Task loss   : CE + Dice + Lovász (same as teacher training)
  - Logit KD    : KL divergence with temperature scaling (T=4)
  - Feature KD  : L2 on 2 paired feature maps with 1×1 adapters
      f_skip : teacher stride-4 skip (32ch)  ↔ student e2 / down1 output (base_ch*2)
      f_fused: teacher stride-8 fused (64ch) ↔ student e3 / down2 output (base_ch*4)
  - Attention KD: align student bottleneck CBAM sa_map with teacher CBAM sa_map
  - Auxiliary KD: deep supervision on student stride-4 decoder head

Teacher always in eval, frozen. Student trains on RGB only.
At inference: student uses RGB → produces 3-class segmentation.

Usage:
  python -m dual_encoder.distill_train_dual_enc_mini_to_unet \
    --teacher_ckpt /path/to/dual_encoder_mini_best.pth \
    --data_root /path/to/SUGARBEETS_AUGMENTED_DATASET \
    --output_dir /path/to/experiments_distill \
    --epochs 100 --batch_size 4
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
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
from tqdm import tqdm

from dual_encoder.dual_encoder_mini import DualEncoderMini
from dual_encoder.dual_encoder_data_loader_weedcrop import create_dual_encoder_dataloaders
from sota.models import UNet, get_model_info

# ── try thop for FLOPs ───────────────────────────────────────────────────────
try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

CLASS_NAMES  = ['background', 'crop', 'weed']
NUM_CLASSES  = 3
RGB_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD      = np.array([0.229, 0.224, 0.225],  dtype=np.float32)

# colour map for mask visualisation: bg=grey, crop=green, weed=red
MASK_COLORS  = np.array([[38, 38, 38], [51, 204, 51], [230, 51, 51]], dtype=np.uint8)


# =============================================================================
# 1. Student  →  UNet (imported from dual_encoder.models)
# =============================================================================
# UNet.forward(x) → logits  (plain forward, no feature hooks)


# =============================================================================
# 2. Instrumented Teacher wrapper
# =============================================================================

class DualEncoderMiniTeacher(nn.Module):
    """
    Wraps DualEncoderMini.
    forward(x_rgb, x_nir) → (logits, features)
    features: {'f_skip': stride-4 skip (32ch), 'f_fused': stride-8 fused (64ch)}
    Stores last CBAM spatial attention map in self.sa_map for attention KD.
    Always in eval mode; gradients disabled externally.
    """
    def __init__(self, base_model: DualEncoderMini):
        super().__init__()
        self.model  = base_model
        self.sa_map = None   # populated each forward pass

    def forward(self, x_rgb: torch.Tensor, x_nir: torch.Tensor):
        input_size = x_rgb.shape[-2:]

        skip, rgb_s8 = self.model.rgb_enc(x_rgb)
        nir_s8       = self.model.nir_enc(x_nir)
        nir_s8       = self.model.nir_proj(nir_s8)
        fused        = self.model.cbam(self.model.aff(rgb_s8, nir_s8))

        # FIX: read sa_map from _CBAM attribute (not a nonexistent last_sa)
        self.sa_map  = self.model.cbam.sa_map if hasattr(self.model.cbam, 'sa_map') else None

        neck   = self.model.aspp(fused)
        logits = self.model.decoder(neck, skip, input_size)

        features = {
            'f_skip':  skip,    # stride-4, 32ch
            'f_fused': fused,   # stride-8, 64ch
        }
        return logits, features


# =============================================================================
# 3. Feature adapter (student ch → teacher ch, 1×1 conv + GN + GELU)
# =============================================================================

class FeatureAdapter(nn.Module):
    def __init__(self, s_ch: int, t_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(s_ch, t_ch, 1, bias=False),
            nn.GroupNorm(min(32, t_ch), t_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.proj(x)


# =============================================================================
# 4. Loss functions
# =============================================================================

# class TaskLoss(nn.Module):
#     """CE + Dice + Lovász — same combo as teacher training."""
#     def __init__(self, class_weights: List[float], num_classes: int = 3):
#         super().__init__()
#         w = torch.tensor(class_weights, dtype=torch.float32)
#         self.register_buffer('w', w)
#         self.num_classes = num_classes

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         # CE
#         l_ce = F.cross_entropy(logits, targets, weight=self.w)
#         # Soft Dice
#         probs = F.softmax(logits, dim=1)
#         oh    = F.one_hot(targets.clamp(0), self.num_classes).permute(0,3,1,2).float()
#         inter = (probs * oh).sum((0,2,3))
#         denom = probs.sum((0,2,3)) + oh.sum((0,2,3))
#         dice  = 1.0 - (2*inter + 1e-6) / (denom + 1e-6)
#         l_dice = dice.mean()
#         # Lovász
#         B, C, H, W = probs.shape
#         l_lovasz = _lovasz_softmax(probs.permute(0,2,3,1).reshape(-1, C),
#                                    targets.reshape(-1))
#         return l_ce + l_dice + 0.5 * l_lovasz

# updated on 18 april to maintain consistency with other models 
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

def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    j = 1.0 - inter / union
    j[1:] = j[1:] - j[:-1]
    return j

def _lovasz_softmax(probs_flat, labels_flat):
    C, losses = probs_flat.shape[1], []
    for c in range(C):
        fg = (labels_flat == c).float()
        if fg.sum() == 0: continue
        err, perm = torch.sort((fg - probs_flat[:, c]).abs(), descending=True)
        losses.append((err * _lovasz_grad(fg[perm])).sum())
    return torch.stack(losses).mean() if losses else probs_flat.sum() * 0.0


class KLDivLoss(nn.Module):
    """KL divergence logit distillation with optional boundary pixel up-weighting."""
    def __init__(self, T: float = 4.0):
        super().__init__()
        self.T = T

    def forward(self, s_logits: torch.Tensor, t_logits: torch.Tensor,
                boundary_mask: torch.Tensor = None) -> torch.Tensor:
        s = F.log_softmax(s_logits / self.T, dim=1)
        t = F.softmax(t_logits / self.T,     dim=1)
        kl = F.kl_div(s, t, reduction='none').sum(dim=1)   # (B,H,W)
        if boundary_mask is not None:
            # upweight boundary pixels 3×
            w = 1.0 + 2.0 * boundary_mask.float()
            kl = kl * w
        return kl.mean() * (self.T ** 2)


class FeatureDistillLoss(nn.Module):
    """Normalised MSE after channel projection + optional spatial align."""
    def forward(self, s_feat, t_feat, adapter):
        s = adapter(s_feat)
        if s.shape[2:] != t_feat.shape[2:]:
            s = F.interpolate(s, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
        # normalise each map independently before L2
        s = F.normalize(s.flatten(2), dim=2).reshape_as(s)
        t = F.normalize(t_feat.detach().flatten(2), dim=2).reshape_as(t_feat)
        return F.mse_loss(s, t)


def _boundary_mask(targets: torch.Tensor) -> torch.Tensor:
    """(B,H,W) long → (B,H,W) float boundary map."""
    t = targets.float().unsqueeze(1)
    dilated = F.max_pool2d(t, 3, stride=1, padding=1)
    eroded  = -F.max_pool2d(-t, 3, stride=1, padding=1)
    return ((dilated - eroded).abs() > 0.1).squeeze(1)


# =============================================================================
# 5. Metrics
# =============================================================================

class SegMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
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
        dice = 2*TP / (2*TP + FP + FN + eps)
        prec = TP / (TP + FP + eps)
        rec  = TP / (TP + FN + eps)
        spec = TN / (TN + FP + eps)
        pacc = TP.sum() / (cm.sum() + eps)
        macc = (TP / (cm.sum(1) + eps)).mean()
        r = {'pixel_acc': float(pacc), 'mean_acc': float(macc),
             'miou': float(iou.mean()), 'mean_dice': float(dice.mean()),
             'mean_f1': float(dice.mean()), 'mean_prec': float(prec.mean()),
             'mean_rec': float(rec.mean())}
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
# 6. Visualisation helpers
# =============================================================================

def _denorm(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor → uint8 HWC RGB."""
    img = t.cpu().float().numpy().transpose(1, 2, 0)
    img = img * RGB_STD + RGB_MEAN
    return (img.clip(0, 1) * 255).astype(np.uint8)

def _colorise(mask: np.ndarray) -> np.ndarray:
    """(H,W) int {0,1,2} → (H,W,3) uint8 RGB."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(MASK_COLORS):
        out[mask == c] = col
    return out

def _add_legend(ax):
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=MASK_COLORS[i]/255., label=CLASS_NAMES[i])
               for i in range(NUM_CLASSES)]
    ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.7)

def save_vis_grid(samples, student, teacher_w, device, save_path: str, epoch: int):
    """
    Draw a grid: N rows × 5 cols
        col0: RGB input
        col1: GT mask
        col2: Student pred
        col3: Teacher pred (4ch reference)
        col4: Error map  (correct=grey, FP=cyan, FN=yellow, wrong_cls=magenta)
    Computes per-sample metrics and annotates each row.
    """
    student.eval()
    teacher_w.eval()

    n = len(samples)
    fig = plt.figure(figsize=(20, 4 * n))
    gs  = gridspec.GridSpec(n, 5, figure=fig, hspace=0.05, wspace=0.05)
    col_titles = ['RGB', 'GT', 'Student pred', 'Teacher ref', 'Error map']

    for row, s in enumerate(samples):
        rgb_t  = s['rgb'].unsqueeze(0).to(device)   # (1,3,H,W)
        nir_t  = s['nir'].unsqueeze(0).to(device)   # (1,1,H,W)
        mask   = s['mask'].cpu().numpy()             # (H,W)

        with torch.no_grad():
            s_logits = student(rgb_t)
            s_pred   = s_logits.argmax(1)[0].cpu().numpy()
            t_logits, _ = teacher_w(rgb_t, nir_t)
            t_pred   = t_logits.argmax(1)[0].cpu().numpy()

        rgb_img  = _denorm(s['rgb'])
        gt_col   = _colorise(mask)
        st_col   = _colorise(s_pred)
        te_col   = _colorise(t_pred)

        # Error map
        err = np.ones((*mask.shape, 3), dtype=np.uint8) * 180
        err[(s_pred > 0) & (mask == 0)]                       = [0, 220, 220]   # FP cyan
        err[(s_pred == 0) & (mask > 0)]                       = [220, 220, 0]   # FN yellow
        err[(s_pred != mask) & (s_pred > 0) & (mask > 0)]    = [200, 0, 200]   # wrong cls magenta

        # Per-sample IoU
        ious = []
        for c in range(NUM_CLASSES):
            tp = ((s_pred==c) & (mask==c)).sum()
            fp = ((s_pred==c) & (mask!=c)).sum()
            fn = ((s_pred!=c) & (mask==c)).sum()
            ious.append(tp / (tp + fp + fn + 1e-6))
        miou = np.mean(ious)
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

    fig.suptitle(f'Distillation inference — Epoch {epoch}', fontsize=11, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 7. Training epoch
# =============================================================================

def train_one_epoch(teacher_w, student, adapters, loader, optimizer, scaler,
                    task_fn, kd_fn, feat_fn, device, epoch,
                    w_task, w_kd, w_feat, writer):
    student.train()
    teacher_w.eval()
    metrics = SegMetrics()

    tot = tot_task = tot_kd = tot_feat = 0.0
    n = 0

    pbar = tqdm(loader, desc=f'Ep {epoch:03d} [Distill]')
    for batch in pbar:
        rgb  = batch['rgb'].to(device)
        nir  = batch['nir'].to(device)
        mask = batch['mask'].to(device).long()

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            with torch.no_grad():
                t_logits, t_feats = teacher_w(rgb, nir)

            # plain UNet — logits only
            s_logits = student(rgb)

            l_task = task_fn(s_logits, mask)
            bmask  = _boundary_mask(mask)
            l_kd   = kd_fn(s_logits, t_logits, bmask)
            l_feat = torch.tensor(0.0, device=device)

            loss = w_task * l_task + w_kd * l_kd

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()), 1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            metrics.update(s_logits.argmax(1), mask)

        tot      += loss.item()
        tot_task += l_task.item()
        tot_kd   += l_kd.item()
        tot_feat += l_feat.item()
        n += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'task': f'{l_task.item():.4f}',
                          'kd':   f'{l_kd.item():.4f}'})

    r = metrics.compute()
    r.update({'loss': tot/n, 'loss_task': tot_task/n,
              'loss_kd': tot_kd/n, 'loss_feat': tot_feat/n})
    for k, v in r.items():
        writer.add_scalar(f'Train/{k}', v, epoch)
    return r


# =============================================================================
# 8. Validation / test evaluation
# =============================================================================

@torch.no_grad()
def evaluate(student, loader, task_fn, device, epoch, writer, split='Val'):
    student.eval()
    metrics = SegMetrics()
    tot = 0.0
    n   = 0
    for batch in tqdm(loader, desc=f'[{split}]'):
        rgb  = batch['rgb'].to(device)
        mask = batch['mask'].to(device).long()
        logits = student(rgb)
        tot += task_fn(logits, mask).item()
        metrics.update(logits.argmax(1), mask)
        n += 1
    r = metrics.compute()
    r['loss'] = tot / n
    for k, v in r.items():
        writer.add_scalar(f'{split}/{k}', v, epoch)
    return r


# =============================================================================
# 9. Model stats helpers
# =============================================================================

def model_stats(model, in_ch, H, W, device):
    params = sum(p.numel() for p in model.parameters())
    stats  = {'params': params, 'params_M': params/1e6}
    if HAS_THOP:
        dummy = torch.randn(1, in_ch, H, W, device=device)
        model.eval()
        with torch.no_grad():
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        stats['flops_G'] = macs * 2 / 1e9
    else:
        stats['flops_G'] = -1.0
    return stats


def measure_latency(model, in_ch, H, W, device, runs=50, warmup=10):
    """
    Measure single-image inference latency in ms.
    For the teacher wrapper (DualEncoderMiniTeacher) pass in_ch=4 —
    the function detects this and splits into (rgb, nir).
    """
    model.eval()
    is_dual = isinstance(model, DualEncoderMiniTeacher)
    if is_dual:
        # FIX: teacher expects (rgb, nir) — can't call with a single 4ch tensor
        dummy_rgb = torch.randn(1, 3, H, W, device=device)
        dummy_nir = torch.randn(1, 1, H, W, device=device)
        def _run(): model(dummy_rgb, dummy_nir)
    else:
        dummy = torch.randn(1, in_ch, H, W, device=device)
        def _run(): model(dummy)

    with torch.no_grad():
        for _ in range(warmup): _run()
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs): _run()
        if device.type == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0


# =============================================================================
# 10. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Distill DualEncoderMini → UNet (RGB-only student)')

    # paths
    parser.add_argument('--teacher_ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str,
                        default='/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET')
    parser.add_argument('--output_dir', type=str,
                        default='/home/vjti-comp/WEEDSBL/scripts/sota/experiments_distill')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    # teacher config (must match how DualEncoderMini was trained)
    parser.add_argument('--teacher_rgb_base_ch', type=int, default=16)
    parser.add_argument('--teacher_nir_base_ch', type=int, default=8)
    parser.add_argument('--teacher_aspp_ch',     type=int, default=64)

    # student
    parser.add_argument('--student_base_ch', type=int, default=16,
                        help='32→~1.9M params; 16→~0.5M params')

    # data
    parser.add_argument('--height',     type=int, default=640)
    parser.add_argument('--width',      type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers',type=int, default=4)

    # training
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--min_lr',        type=float, default=1e-6)

    # distillation weights
    parser.add_argument('--w_task', type=float, default=1.0)
    parser.add_argument('--w_kd',   type=float, default=0.5,
                        help='Higher KD weight — student has no NIR, needs more guidance')
    parser.add_argument('--w_feat', type=float, default=0.5)
    parser.add_argument('--kd_temp', type=float, default=4.0)

    # class weights (bg=0.3 crop=1.0 weed=3.0)
    parser.add_argument('--class_weights', type=float, nargs=3,
                        default=[0.3, 1.0, 3.0])

    # vis
    parser.add_argument('--vis_samples', type=int, default=5)
    parser.add_argument('--vis_every',   type=int, default=10)

    args = parser.parse_args()

    # ── dirs ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.exp_name is None:
        args.exp_name = f'distill_MiniDualEnc_UNet_S{args.student_base_ch}_{ts}'

    run_dir  = Path(args.output_dir) / args.exp_name
    ckpt_dir = run_dir / 'checkpoints'
    vis_dir  = run_dir / 'vis'
    tb_dir   = run_dir / 'tensorboard'
    for d in [ckpt_dir, vis_dir, tb_dir]: d.mkdir(parents=True, exist_ok=True)

    # save full run config
    config = {**vars(args), 'class_names': CLASS_NAMES,
              'teacher': 'DualEncoderMini', 'student': 'UNet'}
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f'  Distillation: DualEncoderMini (4ch) → UNet (RGB only)')
    print(f'  Run dir: {run_dir}')
    print(f"{'='*70}\n")

    writer = SummaryWriter(str(tb_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {device}')

    # ── data ──────────────────────────────────────────────────────────────────
    print('[INFO] Loading data...')
    train_loader, val_loader, test_loader = create_dual_encoder_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.height, args.width),
        mask_mode='multiclass',
    )
    print(f'[INFO] Train={len(train_loader.dataset)} '
          f'Val={len(val_loader.dataset)} Test={len(test_loader.dataset)}')

    # Fixed vis samples from val set (different each run for variety)
    all_val   = list(val_loader.dataset)
    vis_samples = random.sample(all_val, min(args.vis_samples, len(all_val)))

    # ── teacher ───────────────────────────────────────────────────────────────
    print('[INFO] Loading teacher (DualEncoderMini)...')
    teacher_base = DualEncoderMini(
        rgb_base_ch=args.teacher_rgb_base_ch,
        nir_base_ch=args.teacher_nir_base_ch,
        aspp_ch=args.teacher_aspp_ch,
        num_classes=NUM_CLASSES,
    ).to(device)

    ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    missing, unexpected = teacher_base.load_state_dict(state, strict=False)
    if missing:
        print(f'  [WARN] Missing keys: {missing[:3]}{"…" if len(missing)>3 else ""}')
    teacher_base.eval()
    for p in teacher_base.parameters():
        p.requires_grad_(False)

    teacher_w = DualEncoderMiniTeacher(teacher_base).to(device)
    teacher_w.eval()

    t_params = sum(p.numel() for p in teacher_base.parameters())
    print(f'[INFO] Teacher params: {t_params/1e6:.3f}M')

    # ── student ───────────────────────────────────────────────────────────────
    print(f'[INFO] Building student UNet (base_ch={args.student_base_ch})...')
    student = UNet(
        in_channels=3,
        base_ch=args.student_base_ch,
        out_channels=NUM_CLASSES,
    ).to(device)
    s_params = sum(p.numel() for p in student.parameters())
    print(f'[INFO] Student params: {s_params/1e6:.3f}M')
    print(f'[INFO] Compression: {s_params/t_params:.3f}× '
          f'({(1-s_params/t_params)*100:.1f}% smaller)')

    # plain UNet — no feature adapters
    adapters = nn.ModuleDict({}).to(device)
    print('[INFO] Plain UNet — logit KD + task loss only (no feature adapters)')

    # ── loss / optimiser / scheduler ──────────────────────────────────────────
    task_fn = TaskLoss(args.class_weights, NUM_CLASSES).to(device)
    kd_fn   = KLDivLoss(T=args.kd_temp)
    feat_fn = FeatureDistillLoss()

    optimizer = optim.AdamW(
        list(student.parameters()),
        lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999)
    )
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer,
                                T_max=args.epochs - args.warmup_epochs,
                                eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                              milestones=[args.warmup_epochs])
    scaler = GradScaler('cuda')

    # ── model stats ───────────────────────────────────────────────────────────
    # FIX: pass teacher_w (DualEncoderMiniTeacher) — measure_latency handles dual input
    t_lat = measure_latency(teacher_w, 4, args.height, args.width, device)
    s_lat = measure_latency(student,   3, args.height, args.width, device)
    print(f'[INFO] Teacher latency: {t_lat:.1f}ms (3+1ch input)')
    print(f'[INFO] Student latency: {s_lat:.1f}ms (3ch input)')

    writer.add_scalar('Model/Teacher_params_M', t_params/1e6, 0)
    writer.add_scalar('Model/Student_params_M', s_params/1e6, 0)
    writer.add_scalar('Model/Teacher_latency_ms', t_lat, 0)
    writer.add_scalar('Model/Student_latency_ms', s_lat, 0)

    # save model configs to JSON
    model_cfg = {
        'teacher': {
            'class': 'DualEncoderMini',
            'rgb_base_ch': args.teacher_rgb_base_ch,
            'nir_base_ch': args.teacher_nir_base_ch,
            'aspp_ch': args.teacher_aspp_ch,
            'num_classes': NUM_CLASSES,
            'params_M': t_params/1e6,
            'latency_ms': t_lat,
            'input_channels': 4,
        },
        'student': {
            # FIX: correct class name
            'class': 'UNet',
            'base_ch': args.student_base_ch,
            'num_classes': NUM_CLASSES,
            'params_M': s_params/1e6,
            'latency_ms': s_lat,
            'input_channels': 3,
        },
        'adapters': 'none (plain UNet — logit KD only)',
        'distill': {'w_task': args.w_task, 'w_kd': args.w_kd,
                    'w_feat': args.w_feat, 'kd_temp': args.kd_temp},
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
        pass  # plain UNet — no adapters
        optimizer.load_state_dict(r['optimizer_state_dict'])
        start_epoch   = r['epoch'] + 1
        best_val_miou = r.get('best_val_miou', 0.0)
        print(f'[INFO] Resumed ep={r["epoch"]}  best_miou={best_val_miou:.4f}')

    # ── training loop ─────────────────────────────────────────────────────────
    print(f'\n[INFO] Starting distillation for {args.epochs} epochs...\n')

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f'Epoch {epoch}/{args.epochs}  LR={optimizer.param_groups[0]["lr"]:.2e}')
        print('='*60)

        train_r = train_one_epoch(
            teacher_w, student, adapters, train_loader,
            optimizer, scaler, task_fn, kd_fn, feat_fn,
            device, epoch, args.w_task, args.w_kd, args.w_feat, writer
        )
        scheduler.step()

        val_r = evaluate(student, val_loader, task_fn, device, epoch, writer, 'Val')

        # pretty print
        for phase, r in [('Train', train_r), ('Val', val_r)]:
            m = SegMetrics()
            m.print_table(r, phase)

        print(f"  KD losses → task={train_r['loss_task']:.4f} "
              f"kd={train_r['loss_kd']:.4f} feat={train_r['loss_feat']:.4f}")

        writer.add_scalar('Epoch/LR', optimizer.param_groups[0]['lr'], epoch)

        # ── visualise ─────────────────────────────────────────────────────────
        is_vis_ep = (epoch % args.vis_every == 0 or epoch == 1)
        is_best   = val_r['miou'] > best_val_miou

        if is_best:
            best_val_miou = val_r['miou']
            best_path = ckpt_dir / 'best_student.pth'
            torch.save({
                'epoch':               epoch,
                'student_state_dict':  student.state_dict(),
                'adapters_state_dict': {},
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'best_val_miou':       best_val_miou,
                'val_metrics':         val_r,
                'model_config':        model_cfg,
                'train_config':        config,
            }, best_path)
            print(f'  [✓] Best student saved (mIoU={best_val_miou:.4f})')

            vis_path = vis_dir / f'best_epoch_{epoch:03d}.png'
            save_vis_grid(vis_samples, student, teacher_w, device,
                          str(vis_path), epoch)
            # vis saved to disk at vis_path (add_figure skipped — use saved PNG)

        if is_vis_ep and not is_best:
            vis_path = vis_dir / f'epoch_{epoch:03d}.png'
            save_vis_grid(vis_samples, student, teacher_w, device,
                          str(vis_path), epoch)

        # periodic checkpoint (keep last 3)
        if epoch % 10 == 0:
            ep_path = ckpt_dir / f'epoch_{epoch:03d}.pth'
            torch.save({
                'epoch':               epoch,
                'student_state_dict':  student.state_dict(),
                'adapters_state_dict': {},
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'best_val_miou':       best_val_miou,
                'model_config':        model_cfg,
                'train_config':        config,
            }, ep_path)
            # prune old periodic checkpoints, keep last 3
            all_ep = sorted(ckpt_dir.glob('epoch_*.pth'))
            for old in all_ep[:-3]: old.unlink()

    # ── final test evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print('[INFO] Final test evaluation with best student...')
    best_ckpt = torch.load(ckpt_dir / 'best_student.pth',
                           map_location=device, weights_only=False)
    student.load_state_dict(best_ckpt['student_state_dict'])

    test_r = evaluate(student, test_loader, task_fn, device, args.epochs+1,
                      writer, 'Test')
    s_final_lat = measure_latency(student, 3, args.height, args.width, device)
    test_r['latency_ms'] = s_final_lat
    test_r['params_M']   = s_params / 1e6

    # final vis on test set
    test_samples = random.sample(list(test_loader.dataset),
                                 min(args.vis_samples, len(test_loader.dataset)))
    save_vis_grid(test_samples, student, teacher_w, device,
                  str(vis_dir / 'final_test.png'), args.epochs)

    # final metrics table
    sm = SegMetrics()
    sm.print_table(test_r, 'Test')
    print(f'[Test] Params={s_params/1e6:.3f}M  Latency={s_final_lat:.1f}ms')

    results = {
        'test_metrics':       test_r,
        'best_val_miou':      best_val_miou,
        'teacher_params_M':   t_params/1e6,
        'student_params_M':   s_params/1e6,
        'teacher_latency_ms': t_lat,
        'student_latency_ms': s_final_lat,
        'compression_ratio':  s_params/t_params,
        'model_config':       model_cfg,
        'train_config':       config,
    }
    with open(run_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    for k, v in test_r.items():
        writer.add_scalar(f'Final/Test_{k}', v, 0)

    writer.close()
    print(f'\n[INFO] Done. Results → {run_dir}')
    print(f'[INFO] Best Val mIoU={best_val_miou:.4f}  '
          f'Test mIoU={test_r["miou"]:.4f}')
    print(f"{'='*70}\n")



if __name__ == '__main__':
    main()