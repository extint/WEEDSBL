import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dual_encoder.updated_architecture import DualEncoderAFFNet
from dual_encoder.heavy import *
from dual_encoder.dual_encoder_data_loader_weedcrop import *
from dual_encoder.logging_utils_cropweed import TrainingLogger

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    data_root: str = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET"

    # 3-class: 0=background, 1=crop, 2=weed
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["background", "crop", "weed"])

    # class weights for CE — downweight easy BG, upweight hard weed
    class_weights: List[float] = field(default_factory=lambda: [0.3, 1.0, 2.0])

    batch_size: int = 4
    num_workers: int = 4
    target_size: Tuple[int, int] = (640, 640)

    num_epochs: int = 100
    warmup_epochs: int = 6

    # Optimizer
    lr: float = 3e-4          # higher initial LR — warmup handles the ramp
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Scheduler: one cosine cycle — no restarts (they caused your epoch-140 regression)
    t_max: int = 94           # epochs of cosine decay after warmup

    # Loss weights
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    boundary_weight: float = 0.5
    lovasz_weight: float = 0.5   # optional but powerful for IoU-oriented training

    # Early stopping
    patience: int = 30

    # Checkpoint / logging
    log_dir_prefix: str = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/dual_encoder"
    num_vis_samples: int = 4
    vis_every: int = 10
    heavy_log_every: int = 10


CFG = TrainConfig()

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """Multi-class focal loss."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer_weight = weight  # stored externally

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        logits : (B, C, H, W)  — raw logits
        targets: (B, H, W)     — long integer class indices
        """
        ce = F.cross_entropy(logits, targets, weight=weight, reduction='none')  # (B,H,W)
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


def multiclass_dice_loss(logits: torch.Tensor, targets: torch.Tensor,
                         num_classes: int, smooth: float = 1e-6,
                         ignore_index: int = -1) -> torch.Tensor:
    """
    Soft Dice loss averaged over classes.
    logits : (B, C, H, W)
    targets: (B, H, W)  long
    """
    probs = F.softmax(logits, dim=1)            # (B, C, H, W)
    targets_oh = F.one_hot(targets.clamp(0), num_classes)  # (B, H, W, C)
    targets_oh = targets_oh.permute(0, 3, 1, 2).float()   # (B, C, H, W)

    dice_per_class = []
    for c in range(num_classes):
        p = probs[:, c].reshape(-1)
        t = targets_oh[:, c].reshape(-1)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice_per_class.append(1.0 - (2.0 * inter + smooth) / (denom + smooth))

    return torch.stack(dice_per_class).mean()


def boundary_loss_multiclass(logits: torch.Tensor, targets: torch.Tensor,
                              num_classes: int) -> torch.Tensor:
    """
    Emphasise pixels near class boundaries.
    """
    probs = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets.clamp(0), num_classes).permute(0, 3, 1, 2).float()

    # Max-pool minus avg-pool gives boundary map per channel
    k = 3
    target_dilated = F.max_pool2d(targets_oh, k, stride=1, padding=k // 2)
    target_eroded  = F.avg_pool2d(targets_oh, k, stride=1, padding=k // 2)
    boundary       = (target_dilated - target_eroded).abs().sum(dim=1, keepdim=True)  # (B,1,H,W)
    edge_weight    = (1.0 + 2.0 * (boundary > 0.1).float())  # (B,1,H,W)

    ce = F.cross_entropy(logits, targets, reduction='none').unsqueeze(1)  # (B,1,H,W)
    return (ce * edge_weight).mean()


# ---- Lovász-Softmax (compact, no external dep) -----
def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor,
                         classes: str = 'present') -> torch.Tensor:
    """Flat Lovász-softmax. probs: (P, C), labels: (P,)"""
    C = probs.shape[1]
    losses = []
    present = labels.unique()
    for c in range(C):
        if classes == 'present' and c not in present:
            continue
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append((errors_sorted * lovasz_grad(fg_sorted)).sum())
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0


def lovasz_softmax(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)                    # (B,C,H,W)
    B, C, H, W = probs.shape
    probs_flat  = probs.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = targets.reshape(-1)
    return lovasz_softmax_flat(probs_flat, labels_flat)


class MulticlassCombinedLoss(nn.Module):
    """
    Focal + Dice + Boundary + Lovász.
    All four encourage different aspects of segmentation quality.
    """
    def __init__(self, cfg: TrainConfig, device: str):
        super().__init__()
        self.cfg = cfg
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        weight = torch.tensor(cfg.class_weights, dtype=torch.float32).to(device)
        self.register_buffer('class_weight', weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        logits : (B, C, H, W)
        targets: (B, H, W)   long integer labels  {0,1,2}
        Returns dict with individual loss components and 'total'.
        """
        C = self.cfg.num_classes

        l_focal    = self.focal(logits, targets, weight=self.class_weight)
        l_dice     = multiclass_dice_loss(logits, targets, C)
        l_boundary = boundary_loss_multiclass(logits, targets, C)
        l_lovasz   = lovasz_softmax(logits, targets)

        total = (self.cfg.ce_weight      * l_focal
               + self.cfg.dice_weight    * l_dice
               + self.cfg.boundary_weight* l_boundary
               + self.cfg.lovasz_weight  * l_lovasz)

        return {
            'total':    total,
            'focal':    l_focal.detach(),
            'dice':     l_dice.detach(),
            'boundary': l_boundary.detach(),
            'lovasz':   l_lovasz.detach(),
        }

    # nn.Module buffers require explicit register
    def register_buffer(self, name, tensor):
        super().register_buffer(name, tensor)


# =============================================================================
# METRICS  (all per-class + macro)
# =============================================================================

class SegmentationMetrics:
    """
    Accumulates predictions over a full epoch and computes:
      IoU, Dice/F1, Precision, Recall, Accuracy — per class and macro.
    Uses confusion matrix for efficiency.
    """

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes  = num_classes
        self.class_names  = class_names
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds  : (B, H, W) long — predicted class indices
        targets: (B, H, W) long — ground-truth class indices
        """
        preds_np   = preds.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        mask       = (targets_np >= 0) & (targets_np < self.num_classes)
        np.add.at(
            self.conf_matrix,
            (targets_np[mask], preds_np[mask]),
            1
        )

    def compute(self) -> Dict[str, float]:
        cm     = self.conf_matrix.astype(np.float64)
        TP     = np.diag(cm)
        FP     = cm.sum(axis=0) - TP
        FN     = cm.sum(axis=1) - TP
        TN     = cm.sum() - (TP + FP + FN)

        eps = 1e-6

        iou       = TP / (TP + FP + FN + eps)
        dice      = 2 * TP / (2 * TP + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        f1        = dice   # same thing
        specificity = TN / (TN + FP + eps)

        pixel_acc = TP.sum() / (cm.sum() + eps)
        mean_acc  = (TP / (cm.sum(axis=1) + eps)).mean()

        results = {
            'pixel_accuracy': float(pixel_acc),
            'mean_accuracy':  float(mean_acc),
            'mean_iou':       float(iou.mean()),
            'mean_dice':      float(dice.mean()),
            'mean_precision': float(precision.mean()),
            'mean_recall':    float(recall.mean()),
            'mean_f1':        float(f1.mean()),
        }

        for c, name in enumerate(self.class_names):
            results[f'iou_{name}']        = float(iou[c])
            results[f'dice_{name}']       = float(dice[c])
            results[f'precision_{name}']  = float(precision[c])
            results[f'recall_{name}']     = float(recall[c])
            results[f'f1_{name}']         = float(f1[c])
            results[f'specificity_{name}']= float(specificity[c])

        return results

    def pretty_print(self, results: Dict[str, float], phase: str = "Val"):
        print(f"\n{'─'*60}")
        print(f"  {phase} Metrics")
        print(f"{'─'*60}")
        print(f"  Pixel Acc : {results['pixel_accuracy']:.4f}   Mean Acc: {results['mean_accuracy']:.4f}")
        print(f"  Mean IoU  : {results['mean_iou']:.4f}   Mean F1 : {results['mean_f1']:.4f}")
        print(f"{'─'*60}")
        header = f"  {'Class':<14} {'IoU':>7} {'Dice/F1':>8} {'Prec':>7} {'Recall':>7}"
        print(header)
        for name in self.class_names:
            print(f"  {name:<14} "
                  f"{results[f'iou_{name}']:>7.4f} "
                  f"{results[f'dice_{name}']:>8.4f} "
                  f"{results[f'precision_{name}']:>7.4f} "
                  f"{results[f'recall_{name}']:>7.4f}")
        print(f"{'─'*60}\n")


# =============================================================================
# LABEL SMOOTHING CROSS ENTROPY  (optional, helps with over-confidence)
# =============================================================================

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.05, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C       = logits.size(1)
        log_prb = F.log_softmax(logits, dim=1)
        # smooth target: (1-ε) on correct class, ε/(C-1) elsewhere
        with torch.no_grad():
            smooth_label = torch.full_like(log_prb, self.smoothing / (C - 1))
            smooth_label.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_label * log_prb)
        if self.weight is not None:
            # weight per-class
            loss = loss * self.weight.view(1, C, 1, 1)
        return loss.sum(dim=1).mean()


# =============================================================================
# ARCHITECTURE NOTE
# (changes needed in DualEncoderAFFNet — described below via assertions)
# =============================================================================
# Your current model outputs (B,1,H,W) — binary.
# For 3-class you need:
#   - Final conv: nn.Conv2d(in_ch, num_classes=3, 1)   ← change num_classes arg
#   - Remove sigmoid in forward (loss handles raw logits)
#   - Output shape: (B, 3, H, W)
#
# ARCHITECTURAL IMPROVEMENTS (implement in updated_architecture.py):
#
# 1. ASPP (Atrous Spatial Pyramid Pooling) in the decoder bottleneck
#    — This is what gives DeepLabV3 its edge. Adds multi-scale context.
#    Rates: [1, 6, 12, 18] for 640px input.
#
# 2. Deep Supervision on intermediate decoder stages
#    — Attach auxiliary heads at 1/8 and 1/4 scale, weight 0.4 * main_loss.
#    — Dramatically speeds up gradient flow to early layers.
#
# 3. CBAM or SE on skip connections
#    — Channel + spatial attention on encoder features before fusion.
#
# 4. Replace standard BN with Group Norm (GN) if batch_size=4
#    — BN is unstable at small batch sizes; GN(32) is more stable.
#
# Quick ASPP drop-in you can add to your decoder:

class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256, rates=(1, 6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList()
        # 1x1
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU()))
        # dilated 3x3
        for r in rates[1:]:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU()))
        # global average pool branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())
        n_branches = len(self.convs) + 1
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * n_branches, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Dropout2d(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W    = x.shape[-2:]
        feats   = [c(x) for c in self.convs]
        feats  += [F.interpolate(self.gap(x), size=(H, W), mode='bilinear', align_corners=False)]
        return self.proj(torch.cat(feats, dim=1))


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    scheduler, metrics_tracker: SegmentationMetrics,
                    logger=None, max_grad_norm: float = 1.0):
    model.train()
    metrics_tracker.reset()

    running_loss      = 0.0
    loss_components   = {'focal': 0.0, 'dice': 0.0, 'boundary': 0.0, 'lovasz': 0.0}
    nan_count         = 0
    valid_batches     = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        rgb    = batch["rgb"].to(device)
        nir    = batch["nir"].to(device)
        # IMPORTANT: mask must be long integer class indices {0,1,2} — not float
        mask   = batch["mask"].to(device).long()

        if torch.isnan(rgb).any() or torch.isnan(nir).any():
            nan_count += 1
            continue

        optimizer.zero_grad()

        with autocast():
            out = model(rgb, nir)             # tuple during train, tensor during eval
            logits, aux1, aux2 = out          # (B, C, H, W) each

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_count += 1
                continue

            loss_dict  = criterion(logits, mask)
            loss_aux1  = criterion(aux1, mask)['total']
            loss_aux2  = criterion(aux2, mask)['total']
            loss       = loss_dict['total'] + 0.4 * loss_aux1 + 0.4 * loss_aux2

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Step per-batch only if using CosineAnnealingWarmRestarts
        # With SequentialLR (warmup + cosine) step per epoch in main loop
        # scheduler.step(epoch + batch_idx / len(loader))  ← remove if using SequentialLR

        with torch.no_grad():
            preds = logits.argmax(dim=1)          # (B, H, W)
            metrics_tracker.update(preds, mask)

        running_loss += loss.item()
        for k in loss_components:
            loss_components[k] += loss_dict[k].item()
        valid_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr":   f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    if nan_count > 0:
        print(f"[WARNING] Skipped {nan_count} batches due to NaN/Inf")

    denom     = max(valid_batches, 1)
    avg_loss  = running_loss / denom
    for k in loss_components:
        loss_components[k] /= denom

    results = metrics_tracker.compute()
    results['loss'] = avg_loss
    results.update({f'loss_{k}': v for k, v in loss_components.items()})
    return results


# =============================================================================
# VALIDATE
# =============================================================================

def validate(model, loader, criterion, device, epoch,
             metrics_tracker: SegmentationMetrics,
             logger=None, use_tta: bool = False):
    model.eval()
    metrics_tracker.reset()

    running_loss  = 0.0
    nan_count     = 0
    valid_batches = 0

    all_probs   = []
    all_preds   = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            rgb  = batch["rgb"].to(device)
            nir  = batch["nir"].to(device)
            mask = batch["mask"].to(device).long()

            if torch.isnan(rgb).any() or torch.isnan(nir).any():
                nan_count += 1
                continue

            if use_tta:
                # H-flip TTA for multiclass
                logits1 = model(rgb, nir)
                logits2 = model(rgb.flip(-1), nir.flip(-1)).flip(-1)
                logits  = (logits1 + logits2) / 2.0
            else:
                logits = model(rgb, nir)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_count += 1
                continue

            loss_dict = criterion(logits, mask)
            loss      = loss_dict['total']

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

            preds = logits.argmax(dim=1)
            metrics_tracker.update(preds, mask)

            running_loss += loss.item()
            valid_batches += 1

            if logger and epoch % 10 == 0:
                probs = F.softmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, logits.size(1)))
                all_preds.extend(preds.cpu().numpy().flatten().astype(int))
                all_targets.extend(mask.cpu().numpy().flatten())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if nan_count > 0:
        print(f"[WARNING] Val: Skipped {nan_count} batches")

    denom    = max(valid_batches, 1)
    avg_loss = running_loss / denom

    if logger and epoch % 10 == 0 and len(all_preds) > 0:
        all_preds   = np.array(all_preds)
        all_targets = np.array(all_targets)
        logger.log_confusion_matrix(epoch, all_preds, all_targets)
        # PR curve per class can be added here if logger supports multiclass

    results = metrics_tracker.compute()
    results['loss'] = avg_loss
    return results


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch  = ckpt.get('epoch', 0) + 1
    best_miou    = ckpt.get('metrics', {}).get('val_mean_iou', 0.0)
    print(f"[INFO] Resumed from epoch {start_epoch-1}, Best mIoU: {best_miou:.4f}")
    return start_epoch, best_miou


# =============================================================================
# MAIN
# =============================================================================

def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """
    Linear warmup for warmup_epochs, then a single cosine annealing cycle.
    No restarts — they caused the regression you saw at epoch 140.
    """
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=cfg.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.t_max,
        eta_min=cfg.min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.warmup_epochs]
    )
    return scheduler


def main():
    cfg = CFG

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR   = f"{cfg.log_dir_prefix}_{timestamp}"
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"[INFO] Logs → {LOG_DIR}")
    print(f"[INFO] TensorBoard: tensorboard --logdir={LOG_DIR}/tensorboard")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[INFO] Loading data...")
    train_loader, val_loader, test_loader = create_dual_encoder_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        target_size=cfg.target_size,
        mask_mode="multiclass",
    )
    print(f"[INFO] Train={len(train_loader.dataset)}, "
          f"Val={len(val_loader.dataset)}, "
          f"Test={len(test_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # CHANGE num_classes to 3 in your architecture constructor
    print("[INFO] Building DualEncoderAFFNet (3-class)...")
    model = DualEncoderAFFNet(
        rgb_variant='small',
        nir_base_ch=20,
        num_classes=cfg.num_classes,   # ← was 1, now 3
        embed_dim=96
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params / 1e6:.2f}M")

    model.eval()
    # Sanity check output shape
    with torch.no_grad():
        dummy_rgb = torch.zeros(1, 3, *cfg.target_size, device=DEVICE)
        dummy_nir = torch.zeros(1, 1, *cfg.target_size, device=DEVICE)
        dummy_out = model(dummy_rgb, dummy_nir)
        assert dummy_out.shape == (1, cfg.num_classes, *cfg.target_size), \
            f"Expected (1,{cfg.num_classes},...), got {dummy_out.shape}"
    print(f"[INFO] Output shape verified: {dummy_out.shape}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = MulticlassCombinedLoss(cfg, DEVICE)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate LR for encoder (pretrained) vs decoder (random init)
    # Uncomment and adapt once you have param groups in your architecture.
    # encoder_params = [p for n,p in model.named_parameters() if 'encoder' in n]
    # decoder_params = [p for n,p in model.named_parameters() if 'encoder' not in n]
    # optimizer = optim.AdamW([
    #     {'params': encoder_params, 'lr': cfg.lr * 0.1},
    #     {'params': decoder_params, 'lr': cfg.lr},
    # ], weight_decay=cfg.weight_decay)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    scaler = GradScaler(init_scale=2.**10, growth_factor=1.5, backoff_factor=0.5)

    # ── Metrics ───────────────────────────────────────────────────────────────
    train_metrics = SegmentationMetrics(cfg.num_classes, cfg.class_names)
    val_metrics   = SegmentationMetrics(cfg.num_classes, cfg.class_names)

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = TrainingLogger(
        log_dir=LOG_DIR,
        model=model,
        val_loader=val_loader,
        device=DEVICE,
        num_vis_samples=cfg.num_vis_samples
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_miou          = 0.0
    epochs_no_improvement  = 0
    start_epoch            = 1

    # To resume: uncomment below
    # start_epoch, best_val_miou = load_checkpoint("path/to/ckpt.pth", model, optimizer, scheduler)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.num_epochs}  LR={optimizer.param_groups[0]['lr']:.2e}")
        print('='*60)

        # ── Train ──
        train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            DEVICE, epoch, scheduler, train_metrics,
            logger=logger, max_grad_norm=cfg.max_grad_norm
        )
        train_metrics.pretty_print(train_results, phase="Train")

        # Step scheduler once per epoch (SequentialLR / LinearLR+CosineAnnealingLR)
        scheduler.step()

        # ── Heavy logging ──
        if epoch % cfg.heavy_log_every == 0 or epoch == 1:
            logger.log_layer_statistics(epoch)

        # ── Validate ──
        use_tta = (epoch % cfg.heavy_log_every == 0)
        val_results = validate(
            model, val_loader, criterion, DEVICE, epoch,
            val_metrics, logger=logger, use_tta=use_tta
        )
        val_metrics.pretty_print(val_results, phase="Val")

        # ── Log to TensorBoard ──
        current_lr = optimizer.param_groups[0]['lr']
        # Flatten metrics dict for logger
        logger.log_epoch(
            epoch,
            train_results['loss'], train_results['mean_iou'],
            val_results['loss'],   val_results['mean_iou'],
            current_lr
        )
        # Log per-class IoU separately if your logger supports scalars
        if hasattr(logger, 'writer'):
            for name in cfg.class_names:
                logger.writer.add_scalar(f"IoU/train_{name}", train_results[f'iou_{name}'], epoch)
                logger.writer.add_scalar(f"IoU/val_{name}",   val_results[f'iou_{name}'],   epoch)
                logger.writer.add_scalar(f"Dice/train_{name}",train_results[f'dice_{name}'],epoch)
                logger.writer.add_scalar(f"Dice/val_{name}",  val_results[f'dice_{name}'],  epoch)
                logger.writer.add_scalar(f"F1/val_{name}",    val_results[f'f1_{name}'],    epoch)
            logger.writer.add_scalar("Metrics/val_mean_iou",  val_results['mean_iou'],  epoch)
            logger.writer.add_scalar("Metrics/val_pixel_acc", val_results['pixel_accuracy'], epoch)
            for k in ['focal','dice','boundary','lovasz']:
                key = f'loss_{k}'
                if key in train_results:
                    logger.writer.add_scalar(f"Loss/train_{k}", train_results[key], epoch)

        # ── Visualize ──
        if epoch % cfg.vis_every == 0 or epoch == 1:
            print(f"[INFO] Visualising epoch {epoch}...")
            logger.visualize_predictions(epoch)
            logger.visualize_attention_gates_only(epoch)
            logger.visualize_layer_learning(epoch)
            logger.visualize_gradient_flow(epoch)

        # ── Summary line ──
        print(
            f"[Epoch {epoch}] "
            f"Train  loss={train_results['loss']:.4f}  mIoU={train_results['mean_iou']:.4f}  "
            f"crop={train_results['iou_crop']:.4f}  weed={train_results['iou_weed']:.4f}\n"
            f"         Val    loss={val_results['loss']:.4f}  mIoU={val_results['mean_iou']:.4f}  "
            f"crop={val_results['iou_crop']:.4f}  weed={val_results['iou_weed']:.4f}"
            f"{' (TTA)' if use_tta else ''}"
        )

        # ── Checkpoint ──
        is_best = val_results['mean_iou'] > best_val_miou
        if is_best:
            best_val_miou       = val_results['mean_iou']
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        save_metrics = {
            'val_mean_iou':   val_results['mean_iou'],
            'val_loss':       val_results['loss'],
            'train_mean_iou': train_results['mean_iou'],
            'val_iou_crop':   val_results['iou_crop'],
            'val_iou_weed':   val_results['iou_weed'],
        }
        logger.save_checkpoint(epoch, model, optimizer, scheduler, save_metrics, is_best=is_best)

        # ── Early stopping ──
        if epochs_no_improvement >= cfg.patience:
            print(f"[INFO] Early stopping — no improvement for {cfg.patience} epochs.")
            break

    print(f"\n[INFO] Training complete! Best Val mIoU: {best_val_miou:.4f}")
    logger.close()
    print(f"[INFO] Results → {LOG_DIR}")
    print(f"[INFO] TensorBoard: tensorboard --logdir={LOG_DIR}/tensorboard")


if __name__ == "__main__":
    main()