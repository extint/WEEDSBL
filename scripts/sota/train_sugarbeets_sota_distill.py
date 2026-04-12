#!/usr/bin/env python3
"""
Knowledge Distillation Training Script
Teacher → Student: DeepLabV3+ (4ch) → DeepLabV3+ (4ch, fewer params)

Strategy: SOTA Feature-level + Logit-level distillation
  - Logit KD   : KL-divergence on soft predictions (Hinton et al., 2015)
  - Feature KD : L2 loss on intermediate encoder/ASPP/decoder feature maps
                 with 1x1 projection adapters to align teacher→student channels
  - Task loss  : CE + Dice (same as base training)

Metrics logged (train + val, class-wise):
  Inference time, Params, FLOPs, mIoU, per-class IoU, Dice, Pixel Accuracy,
  Precision, Recall, F1 — for Background / Crop / Weed

Usage:
  python train_distill_deeplabv3plus.py \
    --teacher_ckpt /path/to/teacher_best_model.pth \
    --teacher_base_ch 64 \
    --student_base_ch 32 \
    --data_root /path/to/SUGARBEETS_AUGMENTED_DATASET \
    --use_rgbnir \
    --epochs 100 \
    --batch_size 8 \
    --mixed_precision
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
from sota.sugarbeets_data_loader import create_sugarbeets_dataloaders
from sota.models import DeepLabV3Plus, get_model_info

# ─────────────────────────────────────────────────────────────────────────────
# Optional: thop for FLOPs (pip install thop). Falls back gracefully.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


# =============================================================================
# 1.  Instrumented DeepLabV3+ — exposes intermediate feature maps
# =============================================================================

class DeepLabV3PlusWithFeatures(DeepLabV3Plus):
    """Subclass that returns intermediate features alongside the final logits.
    
    forward() returns:
        logits  : (B, C, H, W)
        features: dict with keys 'enc2', 'enc4', 'aspp', 'decoder'
    """

    def forward(self, x):                       # type: ignore[override]
        size = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # ASPP
        aspp_out = self.aspp(F.max_pool2d(e4, 2))
        aspp_out_up = F.interpolate(aspp_out, size=e2.shape[2:],
                                    mode='bilinear', align_corners=False)

        # Low-level + decoder
        low_level = self.low_level_conv(e2)
        decoder_in = torch.cat([aspp_out_up, low_level], dim=1)
        dec = self.decoder(decoder_in)

        # Final upsample + head
        out = F.interpolate(dec, size=size, mode='bilinear', align_corners=False)
        logits = self.final(out)

        features = {
            'enc2':    e2,
            'enc4':    e4,
            'aspp':    aspp_out,
            'decoder': dec,
        }
        return logits, features


# =============================================================================
# 2.  Channel-projection adapters  (student → teacher channel space)
# =============================================================================

class FeatureAdapter(nn.Module):
    """1×1 conv to project student channels → teacher channels for L2 alignment."""

    def __init__(self, student_ch: int, teacher_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(student_ch, teacher_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =============================================================================
# 3.  Loss functions
# =============================================================================

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * targets_oh).sum(dims)
        union = probs.sum(dims) + targets_oh.sum(dims)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 0.5 * ce_loss + 0.5 * (1 - dice.mean())


class MultiClassIoULoss(nn.Module):
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * targets_oh).sum(dims)
        union = probs.sum(dims) + targets_oh.sum(dims)
        iou = (inter + self.eps) / (union + self.eps)
        return 1 - iou.mean()


class KLDivSoftLoss(nn.Module):
    """Hinton-style KL divergence on softened logits (temperature scaling)."""

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(self, s_logits: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
        s_log_soft = F.log_softmax(s_logits / self.T, dim=1)
        t_soft     = F.softmax(t_logits / self.T, dim=1)
        loss = F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (self.T ** 2)
        return loss


class FeatureDistillLoss(nn.Module):
    """Normalised L2 (MSE) between projected student and teacher feature maps.
    
    Spatial sizes may differ (teacher base_ch larger → same spatial grid but
    different channel counts). We project student→teacher channels, then
    interpolate spatial dims if necessary.
    """

    def forward(self,
                s_feat: torch.Tensor,
                t_feat: torch.Tensor,
                adapter: nn.Module) -> torch.Tensor:
        s_proj = adapter(s_feat)
        if s_proj.shape[2:] != t_feat.shape[2:]:
            s_proj = F.interpolate(s_proj, size=t_feat.shape[2:],
                                   mode='bilinear', align_corners=False)
        return F.mse_loss(s_proj, t_feat.detach())


# =============================================================================
# 4.  Metrics
# =============================================================================

CLASS_NAMES = ['Background', 'Crop', 'Weed']


def _accumulate_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    tp_acc: List[float],
    fp_acc: List[float],
    fn_acc: List[float],
    inter_acc: List[float],
    union_acc: List[float],
    dice_inter_acc: List[float],
    dice_sum_acc: List[float],
    correct_acc: List[float],
    total_acc: List[float],
) -> None:
    """In-place accumulation of confusion-matrix statistics."""
    correct_acc[0] += (preds == targets).sum().item()
    total_acc[0]   += targets.numel()

    for cls in range(num_classes):
        pm = (preds == cls)
        tm = (targets == cls)
        tp = (pm & tm).sum().item()
        fp = (pm & ~tm).sum().item()
        fn = (~pm & tm).sum().item()
        union = (pm | tm).sum().item()

        tp_acc[cls]         += tp
        fp_acc[cls]         += fp
        fn_acc[cls]         += fn
        inter_acc[cls]      += tp
        union_acc[cls]      += union
        dice_inter_acc[cls] += tp
        dice_sum_acc[cls]   += pm.sum().item() + tm.sum().item()


def _compute_metrics(
    num_classes: int,
    tp_acc, fp_acc, fn_acc,
    inter_acc, union_acc,
    dice_inter_acc, dice_sum_acc,
    correct_acc, total_acc,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics['pixel_accuracy'] = correct_acc[0] / (total_acc[0] + 1e-9)

    ious, precs, recs, f1s, dices = [], [], [], [], []
    for cls in range(num_classes):
        iou  = inter_acc[cls] / (union_acc[cls] + 1e-6)
        prec = tp_acc[cls] / (tp_acc[cls] + fp_acc[cls] + 1e-6)
        rec  = tp_acc[cls] / (tp_acc[cls] + fn_acc[cls] + 1e-6)
        f1   = 2 * prec * rec / (prec + rec + 1e-6)
        dice = (2 * dice_inter_acc[cls]) / (dice_sum_acc[cls] + 1e-6)
        name = CLASS_NAMES[cls]
        metrics[f'iou_{name}']       = iou
        metrics[f'precision_{name}'] = prec
        metrics[f'recall_{name}']    = rec
        metrics[f'f1_{name}']        = f1
        metrics[f'dice_{name}']      = dice
        ious.append(iou); precs.append(prec); recs.append(rec)
        f1s.append(f1);   dices.append(dice)

    metrics['miou']           = float(np.mean(ious))
    metrics['mean_precision'] = float(np.mean(precs))
    metrics['mean_recall']    = float(np.mean(recs))
    metrics['mean_f1']        = float(np.mean(f1s))
    metrics['mean_dice']      = float(np.mean(dices))
    return metrics


# =============================================================================
# 5.  FLOPs + Params helper
# =============================================================================

def compute_model_stats(
    model: nn.Module,
    in_channels: int,
    height: int,
    width: int,
    device: torch.device,
) -> Dict[str, float]:
    info = get_model_info(model)
    stats: Dict[str, float] = {
        'params_total':   float(info['total_parameters']),
        'params_million': float(info['total_parameters_million']),
    }

    if HAS_THOP:
        dummy = torch.randn(1, in_channels, height, width).to(device)
        model.eval()
        with torch.no_grad():
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        stats['flops_giga'] = macs * 2 / 1e9   # MACs → FLOPs
    else:
        stats['flops_giga'] = -1.0  # unavailable

    return stats


def measure_inference_time(
    model: nn.Module,
    in_channels: int,
    height: int,
    width: int,
    device: torch.device,
    n_runs: int = 50,
    warmup: int = 10,
) -> float:
    """Returns mean inference latency in milliseconds (single image)."""
    model.eval()
    dummy = torch.randn(1, in_channels, height, width).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000.0  # ms


# =============================================================================
# 6.  Training epoch (distillation)
# =============================================================================

def train_one_epoch_distill(
    teacher: DeepLabV3PlusWithFeatures,
    student: DeepLabV3PlusWithFeatures,
    adapters: nn.ModuleDict,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    task_loss_fn: MultiClassDiceLoss,
    kd_logit_loss: KLDivSoftLoss,
    feat_loss_fn: FeatureDistillLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    num_classes: int = 3,
    scaler=None,
    # loss weights
    w_task: float = 1.0,
    w_kd:   float = 1.0,
    w_feat: float = 0.5,
    feat_keys: Tuple[str, ...] = ('enc2', 'enc4', 'aspp', 'decoder'),
) -> Dict[str, float]:

    student.train()
    teacher.eval()

    total_loss = total_task = total_kd = total_feat = 0.0
    tp_acc    = [0.0] * num_classes
    fp_acc    = [0.0] * num_classes
    fn_acc    = [0.0] * num_classes
    inter_acc = [0.0] * num_classes
    union_acc = [0.0] * num_classes
    dice_inter_acc = [0.0] * num_classes
    dice_sum_acc   = [0.0] * num_classes
    correct_acc = [0.0]
    total_acc   = [0.0]

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Distill-Train]")

    for batch_idx, batch in enumerate(pbar):
        x = batch['images'].to(device)
        y = batch['labels'].to(device).long()

        optimizer.zero_grad(set_to_none=True)

        def _forward():
            with torch.no_grad():
                t_logits, t_feats = teacher(x)

            s_logits, s_feats = student(x)

            # Task loss
            l_task = task_loss_fn(s_logits, y)

            # Logit KD loss (KL on soft targets)
            l_kd = kd_logit_loss(s_logits, t_logits)

            # Feature distillation loss (sum over chosen layers)
            l_feat = torch.tensor(0.0, device=device)
            for key in feat_keys:
                l_feat = l_feat + feat_loss_fn(s_feats[key], t_feats[key], adapters[key])
            l_feat = l_feat / len(feat_keys)

            loss = w_task * l_task + w_kd * l_kd + w_feat * l_feat
            return loss, l_task, l_kd, l_feat, s_logits

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, l_task, l_kd, l_feat, s_logits = _forward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, l_task, l_kd, l_feat, s_logits = _forward()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_task += l_task.item()
        total_kd   += l_kd.item()
        total_feat += l_feat.item()

        with torch.no_grad():
            preds = torch.argmax(s_logits, dim=1)
            _accumulate_batch(
                preds, y, num_classes,
                tp_acc, fp_acc, fn_acc,
                inter_acc, union_acc,
                dice_inter_acc, dice_sum_acc,
                correct_acc, total_acc,
            )

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'task': f'{l_task.item():.4f}',
                          'kd':   f'{l_kd.item():.4f}',
                          'feat': f'{l_feat.item():.4f}'})

        if batch_idx % 10 == 0:
            gs = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar('Train/BatchLoss_total', loss.item(), gs)
            writer.add_scalar('Train/BatchLoss_task',  l_task.item(), gs)
            writer.add_scalar('Train/BatchLoss_kd',    l_kd.item(), gs)
            writer.add_scalar('Train/BatchLoss_feat',  l_feat.item(), gs)

    n = len(loader)
    metrics = _compute_metrics(
        num_classes,
        tp_acc, fp_acc, fn_acc,
        inter_acc, union_acc,
        dice_inter_acc, dice_sum_acc,
        correct_acc, total_acc,
    )
    metrics['loss']      = total_loss / n
    metrics['loss_task'] = total_task / n
    metrics['loss_kd']   = total_kd   / n
    metrics['loss_feat'] = total_feat / n
    return metrics


# =============================================================================
# 7.  Validation epoch
# =============================================================================

@torch.no_grad()
def evaluate(
    model: DeepLabV3PlusWithFeatures,
    loader: DataLoader,
    loss_fn: MultiClassDiceLoss,
    device: torch.device,
    num_classes: int = 3,
    split: str = 'Val',
) -> Dict[str, float]:
    model.eval()
    iou_loss_fn = MultiClassIoULoss(num_classes=num_classes)
    total_loss = total_iou_loss = 0.0

    tp_acc    = [0.0] * num_classes
    fp_acc    = [0.0] * num_classes
    fn_acc    = [0.0] * num_classes
    inter_acc = [0.0] * num_classes
    union_acc = [0.0] * num_classes
    dice_inter_acc = [0.0] * num_classes
    dice_sum_acc   = [0.0] * num_classes
    correct_acc = [0.0]
    total_acc   = [0.0]

    pbar = tqdm(loader, desc=f'[{split}]')
    for batch in pbar:
        x = batch['images'].to(device)
        y = batch['labels'].to(device).long()

        logits, _ = model(x)
        loss = loss_fn(logits, y)
        total_loss     += loss.item()
        total_iou_loss += iou_loss_fn(logits, y).item()

        preds = torch.argmax(logits, dim=1)
        _accumulate_batch(
            preds, y, num_classes,
            tp_acc, fp_acc, fn_acc,
            inter_acc, union_acc,
            dice_inter_acc, dice_sum_acc,
            correct_acc, total_acc,
        )

    n = len(loader)
    metrics = _compute_metrics(
        num_classes,
        tp_acc, fp_acc, fn_acc,
        inter_acc, union_acc,
        dice_inter_acc, dice_sum_acc,
        correct_acc, total_acc,
    )
    metrics['loss']      = total_loss / n
    metrics['iou_loss']  = total_iou_loss / n
    return metrics


# =============================================================================
# 8.  Logging helpers
# =============================================================================

def _log_to_tensorboard(writer: SummaryWriter,
                        metrics: Dict[str, float],
                        prefix: str,
                        epoch: int) -> None:
    for k, v in metrics.items():
        writer.add_scalar(f'Epoch/{prefix}_{k}', v, epoch)


def _print_epoch(epoch: int, total: int,
                 train_m: Dict[str, float],
                 val_m: Dict[str, float]) -> None:
    def _row(m, tag):
        bg, cr, wd = CLASS_NAMES
        print(
            f"  {tag} | loss={m['loss']:.4f} | mIoU={m['miou']:.4f} "
            f"| mDice={m['mean_dice']:.4f} | Acc={m['pixel_accuracy']:.4f}"
        )
        print(
            f"       IoU   [{bg}/{cr}/{wd}]: "
            f"[{m[f'iou_{bg}']:.3f}/{m[f'iou_{cr}']:.3f}/{m[f'iou_{wd}']:.3f}]"
        )
        print(
            f"       Dice  [{bg}/{cr}/{wd}]: "
            f"[{m[f'dice_{bg}']:.3f}/{m[f'dice_{cr}']:.3f}/{m[f'dice_{wd}']:.3f}]"
        )
        print(
            f"       F1    [{bg}/{cr}/{wd}]: "
            f"[{m[f'f1_{bg}']:.3f}/{m[f'f1_{cr}']:.3f}/{m[f'f1_{wd}']:.3f}]"
        )
        print(
            f"       Prec  [{bg}/{cr}/{wd}]: "
            f"[{m[f'precision_{bg}']:.3f}/{m[f'precision_{cr}']:.3f}/{m[f'precision_{wd}']:.3f}]"
        )
        print(
            f"       Rec   [{bg}/{cr}/{wd}]: "
            f"[{m[f'recall_{bg}']:.3f}/{m[f'recall_{cr}']:.3f}/{m[f'recall_{wd}']:.3f}]"
        )

    print(f'\n[Epoch {epoch:03d}/{total}]')
    _row(train_m, 'Train')
    if 'loss_kd' in train_m:
        print(
            f"       KD losses → task={train_m['loss_task']:.4f} "
            f"kd={train_m['loss_kd']:.4f} feat={train_m['loss_feat']:.4f}"
        )
    _row(val_m, 'Val  ')


# =============================================================================
# 9.  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Distill DeepLabV3+ 4ch teacher → smaller 4ch student'
    )

    # ── paths ──────────────────────────────────────────────────────────────
    parser.add_argument('--teacher_ckpt', type=str, default="/home/vjti-comp/WEEDSBL/experiments/sugarbeets_deeplabsv3+_4ch_RGBNIR_20260327_043451/checkpoints/best_model.pth",
                        help='Path to trained teacher .pth checkpoint')
    parser.add_argument('--data_root', type=str,
                        default='/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET')
    parser.add_argument('--output_dir', type=str, default='/home/vjti-comp/WEEDSBL/scripts/sota/experiments')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume student training from checkpoint')

    # ── model ──────────────────────────────────────────────────────────────
    parser.add_argument('--teacher_base_ch', type=int, default=16,
                        help='base_ch of the teacher model')
    parser.add_argument('--student_base_ch', type=int, default=8,
                        help='base_ch of the student (smaller) model')
    parser.add_argument('--num_classes', type=int, default=3)

    # ── data ───────────────────────────────────────────────────────────────
    parser.add_argument('--use_rgbnir', action='store_true')
    parser.add_argument('--height', type=int, default=966)
    parser.add_argument('--width', type=int, default=1296)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nir_drop', type=float, default=0.0)

    # ── training ───────────────────────────────────────────────────────────
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mixed_precision', action='store_true')

    # ── distillation weights ───────────────────────────────────────────────
    parser.add_argument('--w_task', type=float, default=1.0,
                        help='Weight for CE+Dice task loss')
    parser.add_argument('--w_kd',   type=float, default=1.0,
                        help='Weight for logit KL-divergence loss')
    parser.add_argument('--w_feat', type=float, default=0.5,
                        help='Weight for feature-level MSE loss')
    parser.add_argument('--kd_temperature', type=float, default=4.0,
                        help='Softmax temperature for logit KD')

    args = parser.parse_args()

    # ── setup dirs ─────────────────────────────────────────────────────────
    if args.exp_name is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ch = '4ch_RGBNIR' if args.use_rgbnir else '3ch_RGB'
        args.exp_name = (
            f"distill_deeplabv3plus_T{args.teacher_base_ch}"
            f"_S{args.student_base_ch}_{ch}_{ts}"
        )

    run_dir  = os.path.join(args.output_dir, args.exp_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    log_dir  = os.path.join(run_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    config_dict = {**vars(args), 'class_names': CLASS_NAMES}
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n{'='*80}")
    print(f'[INFO] Distillation: DeepLabV3+ teacher(base_ch={args.teacher_base_ch})'
          f' → student(base_ch={args.student_base_ch})')
    print(f'[INFO] Run dir: {run_dir}')
    print(f"{'='*80}\n")

    writer = SummaryWriter(log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {device}')
    in_ch = 4 if args.use_rgbnir else 3

    # ── data loaders ───────────────────────────────────────────────────────
    print('[INFO] Creating dataloaders...')
    train_loader, val_loader, test_loader = create_sugarbeets_dataloaders(
        data_root=args.data_root,
        use_rgbnir=args.use_rgbnir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.height, args.width),
        nir_drop_prob=args.nir_drop,
    )
    print(f'[INFO] Train={len(train_loader.dataset)} '
          f'Val={len(val_loader.dataset)} '
          f'Test={len(test_loader.dataset)}')

    # ── teacher ────────────────────────────────────────────────────────────
    print('[INFO] Loading teacher...')
    teacher = DeepLabV3PlusWithFeatures(
        in_ch,
        args.teacher_base_ch,
        args.num_classes
    ).to(device)

    ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    teacher.load_state_dict(state, strict=False)  # strict=False: extra keys ok
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    t_info = get_model_info(teacher)
    print(f'[INFO] Teacher params: {t_info["total_parameters"]:,} '
          f'({t_info["total_parameters_million"]:.2f}M)')

    # ── student ────────────────────────────────────────────────────────────
    print('[INFO] Building student...')
    student = DeepLabV3PlusWithFeatures(
        in_ch,
        args.student_base_ch,
        args.num_classes,
    ).to(device)

    s_info = get_model_info(student)
    print(f'[INFO] Student params: {s_info["total_parameters"]:,} '
          f'({s_info["total_parameters_million"]:.2f}M)')
    ratio = s_info['total_parameters'] / t_info['total_parameters']
    print(f'[INFO] Compression ratio: {ratio:.3f}x  '
          f'({(1-ratio)*100:.1f}% smaller)')

    # ── channel sizes for adapters ─────────────────────────────────────────
    # enc2   : base_ch*2 → base_ch*2  (both)
    # enc4   : base_ch*8 → base_ch*8
    # aspp   : base_ch*4 → base_ch*4
    # decoder: base_ch*4 → base_ch*4
    t_ch = {
        'enc2':    args.teacher_base_ch * 2,
        'enc4':    args.teacher_base_ch * 8,
        'aspp':    args.teacher_base_ch * 4,
        'decoder': args.teacher_base_ch * 4,
    }
    s_ch = {
        'enc2':    args.student_base_ch * 2,
        'enc4':    args.student_base_ch * 8,
        'aspp':    args.student_base_ch * 4,
        'decoder': args.student_base_ch * 4,
    }
    feat_keys = ('enc2', 'enc4', 'aspp', 'decoder')
    adapters = nn.ModuleDict({
        k: FeatureAdapter(s_ch[k], t_ch[k])
        for k in feat_keys
    }).to(device)

    # ── losses ─────────────────────────────────────────────────────────────
    task_loss_fn  = MultiClassDiceLoss(num_classes=args.num_classes)
    kd_logit_loss = KLDivSoftLoss(temperature=args.kd_temperature)
    feat_loss_fn  = FeatureDistillLoss()

    # ── optimiser: student params + adapters ───────────────────────────────
    optimizer = optim.AdamW(
        list(student.parameters()) + list(adapters.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = (torch.cuda.amp.GradScaler()
              if (args.mixed_precision and device.type == 'cuda') else None)

    # ── compute & log static model stats ───────────────────────────────────
    t_stats = compute_model_stats(teacher, in_ch, args.height, args.width, device)
    s_stats = compute_model_stats(student, in_ch, args.height, args.width, device)
    t_lat   = measure_inference_time(teacher, in_ch, args.height, args.width, device)
    s_lat   = measure_inference_time(student, in_ch, args.height, args.width, device)

    print(f'\n[INFO] Teacher — params={t_stats["params_million"]:.2f}M '
          f'FLOPs={t_stats["flops_giga"]:.2f}G '
          f'latency={t_lat:.1f}ms')
    print(f'[INFO] Student — params={s_stats["params_million"]:.2f}M '
          f'FLOPs={s_stats["flops_giga"]:.2f}G '
          f'latency={s_lat:.1f}ms\n')

    # Log static stats once
    writer.add_scalar('Model/Teacher_params_M',   t_stats['params_million'],  0)
    writer.add_scalar('Model/Student_params_M',   s_stats['params_million'],  0)
    writer.add_scalar('Model/Teacher_FLOPs_G',    t_stats['flops_giga'],      0)
    writer.add_scalar('Model/Student_FLOPs_G',    s_stats['flops_giga'],      0)
    writer.add_scalar('Model/Teacher_latency_ms', t_lat,                      0)
    writer.add_scalar('Model/Student_latency_ms', s_lat,                      0)

    # ── optionally resume ──────────────────────────────────────────────────
    start_epoch   = 1
    best_val_miou = 0.0
    if args.resume:
        print(f'[INFO] Resuming from {args.resume}')
        ckpt_r = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(ckpt_r['student_state_dict'])
        adapters.load_state_dict(ckpt_r['adapters_state_dict'])
        optimizer.load_state_dict(ckpt_r['optimizer_state_dict'])
        start_epoch   = ckpt_r['epoch'] + 1
        best_val_miou = ckpt_r.get('best_val_miou', 0.0)
        print(f'[INFO] Resumed epoch={ckpt_r["epoch"]}  best_miou={best_val_miou:.4f}')

    # ── training loop ──────────────────────────────────────────────────────
    print(f'[INFO] Starting distillation for {args.epochs} epochs...\n')

    for epoch in range(start_epoch, args.epochs + 1):
        train_m = train_one_epoch_distill(
            teacher=teacher,
            student=student,
            adapters=adapters,
            loader=train_loader,
            optimizer=optimizer,
            task_loss_fn=task_loss_fn,
            kd_logit_loss=kd_logit_loss,
            feat_loss_fn=feat_loss_fn,
            device=device,
            epoch=epoch,
            writer=writer,
            num_classes=args.num_classes,
            scaler=scaler,
            w_task=args.w_task,
            w_kd=args.w_kd,
            w_feat=args.w_feat,
            feat_keys=feat_keys,
        )

        val_m = evaluate(
            student, val_loader, task_loss_fn,
            device, args.num_classes, split='Val',
        )

        # log inference time every 10 epochs (non-trivial to measure every epoch)
        if epoch % 10 == 0 or epoch == 1:
            s_lat_ep = measure_inference_time(
                student, in_ch, args.height, args.width, device
            )
            writer.add_scalar('Epoch/Student_latency_ms', s_lat_ep, epoch)
            val_m['inference_time_ms'] = s_lat_ep

        _log_to_tensorboard(writer, train_m, 'Train', epoch)
        _log_to_tensorboard(writer, val_m,   'Val',   epoch)
        writer.add_scalar('Epoch/LR', optimizer.param_groups[0]['lr'], epoch)
        _print_epoch(epoch, args.epochs, train_m, val_m)

        # ── save best ──────────────────────────────────────────────────────
        cur_miou = val_m['miou']
        if cur_miou > best_val_miou:
            best_val_miou = cur_miou
            best_path = os.path.join(ckpt_dir, 'best_student.pth')
            torch.save({
                'epoch':                epoch,
                'student_state_dict':   student.state_dict(),
                'adapters_state_dict':  adapters.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_miou':        best_val_miou,
                'val_metrics':          val_m,
                'student_stats':        s_stats,
                'config':               config_dict,
            }, best_path)
            print(f'  [✓] Saved best student (mIoU={best_val_miou:.4f})')

        # ── periodic checkpoint ────────────────────────────────────────────
        if epoch % 10 == 0:
            ep_path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth')
            torch.save({
                'epoch':                epoch,
                'student_state_dict':   student.state_dict(),
                'adapters_state_dict':  adapters.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_miou':        best_val_miou,
                'config':               config_dict,
            }, ep_path)

        scheduler.step()

    # ── final test evaluation ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print('[INFO] Final test evaluation with best student checkpoint...')
    best_ckpt = torch.load(os.path.join(ckpt_dir, 'best_student.pth'),
                           map_location=device, weights_only=False)
    student.load_state_dict(best_ckpt['student_state_dict'])

    test_m = evaluate(
        student, test_loader, task_loss_fn,
        device, args.num_classes, split='Test',
    )

    # Final inference time + FLOPs
    final_lat = measure_inference_time(
        student, in_ch, args.height, args.width, device
    )
    test_m['inference_time_ms'] = final_lat
    test_m['params_million']    = s_stats['params_million']
    test_m['flops_giga']        = s_stats['flops_giga']

    print(f'\n[Test Results — Student]')
    print(f'  Params={s_stats["params_million"]:.2f}M  '
          f'FLOPs={s_stats["flops_giga"]:.2f}G  '
          f'Latency={final_lat:.1f}ms')
    print(f'  Loss={test_m["loss"]:.4f} | mIoU={test_m["miou"]:.4f} | '
          f'mDice={test_m["mean_dice"]:.4f} | Acc={test_m["pixel_accuracy"]:.4f}')
    for cls in CLASS_NAMES:
        print(f'  [{cls}]  IoU={test_m[f"iou_{cls}"]:.4f}  '
              f'Dice={test_m[f"dice_{cls}"]:.4f}  '
              f'F1={test_m[f"f1_{cls}"]:.4f}  '
              f'Prec={test_m[f"precision_{cls}"]:.4f}  '
              f'Rec={test_m[f"recall_{cls}"]:.4f}')

    for k, v in test_m.items():
        writer.add_scalar(f'Final/Test_{k}', v, 0)

    results = {
        'test_metrics':     test_m,
        'best_val_miou':    best_val_miou,
        'teacher_stats':    t_stats,
        'student_stats':    s_stats,
        'teacher_latency_ms': t_lat,
        'student_latency_ms': final_lat,
        'compression_ratio':  s_stats['params_million'] / t_stats['params_million'],
        'config':           config_dict,
    }
    with open(os.path.join(run_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    writer.close()
    print(f'\n[INFO] Done. Results → {run_dir}')
    print(f'[INFO] Best Val mIoU={best_val_miou:.4f}  '
          f'Test mIoU={test_m["miou"]:.4f}')
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()


# Install thop for FLOPs (optional but recommended)
# pip install thop

# python train_distill_deeplabv3plus.py \
#   --teacher_ckpt ./experiments/your_run/checkpoints/best_model.pth \
#   --teacher_base_ch 64 \
#   --student_base_ch 32 \
#   --data_root /path/to/SUGARBEETS_AUGMENTED_DATASET \
#   --use_rgbnir \
#   --epochs 100 \
#   --batch_size 8 \
#   --mixed_precision \
#   --w_task 1.0 --w_kd 1.0 --w_feat 0.5 \
#   --kd_temperature 4.0