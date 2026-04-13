import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dual_encoder.dual_encoder_mini import DualEncoderMini
from dual_encoder.dual_encoder_data_loader_weedcrop import create_dual_encoder_dataloaders
from dual_encoder.logging_utils_cropweed import TrainingLogger

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    data_root: str = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET"

    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["background", "crop", "weed"])

    # Weed is rare and hard — upweight it
    class_weights: List[float] = field(default_factory=lambda: [0.3, 1.0, 3.0])

    batch_size: int = 4
    num_workers: int = 4
    target_size: Tuple[int, int] = (640, 640)

    num_epochs: int = 100
    warmup_epochs: int = 6
    t_max: int = 94

    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Loss weights
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    boundary_weight: float = 0.5
    lovasz_weight: float = 0.5

    # Aux head loss weight
    aux_weight: float = 0.4

    patience: int = 30

    log_dir_prefix: str = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/dual_encoder_mini"
    num_vis_samples: int = 4
    vis_every: int = 10
    heavy_log_every: int = 10


CFG = TrainConfig()

# =============================================================================
# LOSS FUNCTIONS  (identical to your existing train file)
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, weight=None):
        ce    = F.cross_entropy(logits, targets, weight=weight, reduction='none')
        pt    = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


def multiclass_dice_loss(logits, targets, num_classes, smooth=1e-6):
    probs      = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets.clamp(0), num_classes).permute(0, 3, 1, 2).float()
    losses = []
    for c in range(num_classes):
        p = probs[:, c].reshape(-1)
        t = targets_oh[:, c].reshape(-1)
        inter = (p * t).sum()
        losses.append(1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth))
    return torch.stack(losses).mean()


def boundary_loss_multiclass(logits, targets, num_classes):
    targets_oh     = F.one_hot(targets.clamp(0), num_classes).permute(0, 3, 1, 2).float()
    k              = 3
    boundary       = (F.max_pool2d(targets_oh, k, stride=1, padding=k//2)
                      - F.avg_pool2d(targets_oh, k, stride=1, padding=k//2)).abs().sum(1, keepdim=True)
    edge_weight    = 1.0 + 2.0 * (boundary > 0.1).float()
    ce             = F.cross_entropy(logits, targets, reduction='none').unsqueeze(1)
    return (ce * edge_weight).mean()


def lovasz_grad(gt_sorted):
    gts          = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union        = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard      = 1.0 - intersection / union
    jaccard[1:]  = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probs, labels):
    C, losses = probs.shape[1], []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors, perm = torch.sort((fg - probs[:, c]).abs(), descending=True)
        losses.append((errors * lovasz_grad(fg[perm])).sum())
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0


def lovasz_softmax(logits, targets):
    probs = F.softmax(logits, dim=1)
    B, C, H, W = probs.shape
    return lovasz_softmax_flat(probs.permute(0, 2, 3, 1).reshape(-1, C), targets.reshape(-1))


class MulticlassCombinedLoss(nn.Module):
    def __init__(self, cfg: TrainConfig, device: str):
        super().__init__()
        self.cfg   = cfg
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.register_buffer('class_weight',
                             torch.tensor(cfg.class_weights, dtype=torch.float32).to(device))

    def forward(self, logits, targets):
        C          = self.cfg.num_classes
        l_focal    = self.focal(logits, targets, weight=self.class_weight)
        l_dice     = multiclass_dice_loss(logits, targets, C)
        l_boundary = boundary_loss_multiclass(logits, targets, C)
        l_lovasz   = lovasz_softmax(logits, targets)
        total      = (self.cfg.ce_weight       * l_focal
                    + self.cfg.dice_weight     * l_dice
                    + self.cfg.boundary_weight * l_boundary
                    + self.cfg.lovasz_weight   * l_lovasz)
        return {
            'total':    total,
            'focal':    l_focal.detach(),
            'dice':     l_dice.detach(),
            'boundary': l_boundary.detach(),
            'lovasz':   l_lovasz.detach(),
        }


# =============================================================================
# METRICS  (identical to your existing train file)
# =============================================================================

class SegmentationMetrics:
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
        mask = (t >= 0) & (t < self.num_classes)
        np.add.at(self.conf_matrix, (t[mask], p[mask]), 1)

    def compute(self):
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
        spec      = TN / (TN + FP + eps)

        results = {
            'pixel_accuracy': float(TP.sum() / (cm.sum() + eps)),
            'mean_accuracy':  float((TP / (cm.sum(1) + eps)).mean()),
            'mean_iou':       float(iou.mean()),
            'mean_dice':      float(dice.mean()),
            'mean_precision': float(precision.mean()),
            'mean_recall':    float(recall.mean()),
            'mean_f1':        float(dice.mean()),
        }
        for c, name in enumerate(self.class_names):
            results[f'iou_{name}']         = float(iou[c])
            results[f'dice_{name}']        = float(dice[c])
            results[f'precision_{name}']   = float(precision[c])
            results[f'recall_{name}']      = float(recall[c])
            results[f'f1_{name}']          = float(dice[c])
            results[f'specificity_{name}'] = float(spec[c])
        return results

    def pretty_print(self, results, phase="Val"):
        print(f"\n{'─'*60}")
        print(f"  {phase} Metrics")
        print(f"{'─'*60}")
        print(f"  Pixel Acc : {results['pixel_accuracy']:.4f}   Mean Acc: {results['mean_accuracy']:.4f}")
        print(f"  Mean IoU  : {results['mean_iou']:.4f}   Mean F1 : {results['mean_f1']:.4f}")
        print(f"{'─'*60}")
        print(f"  {'Class':<14} {'IoU':>7} {'Dice/F1':>8} {'Prec':>7} {'Recall':>7}")
        for name in self.class_names:
            print(f"  {name:<14} "
                  f"{results[f'iou_{name}']:>7.4f} "
                  f"{results[f'dice_{name}']:>8.4f} "
                  f"{results[f'precision_{name}']:>7.4f} "
                  f"{results[f'recall_{name}']:>7.4f}")
        print(f"{'─'*60}\n")


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    scheduler, metrics_tracker, logger=None, max_grad_norm=1.0,
                    aux_weight=0.4):
    model.train()
    metrics_tracker.reset()

    running_loss    = 0.0
    loss_components = {'focal': 0.0, 'dice': 0.0, 'boundary': 0.0, 'lovasz': 0.0}
    nan_count       = 0
    valid_batches   = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        rgb  = batch["rgb"].to(device)
        nir  = batch["nir"].to(device)
        mask = batch["mask"].to(device).long()

        if torch.isnan(rgb).any() or torch.isnan(nir).any():
            nan_count += 1
            continue

        optimizer.zero_grad()

        with autocast('cuda'):
            logits, aux = model(rgb, nir)   # DualEncoderMini: one aux head

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_count += 1
                continue

            loss_dict = criterion(logits, mask)
            loss_aux  = criterion(aux, mask)['total']
            loss      = loss_dict['total'] + aux_weight * loss_aux

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            metrics_tracker.update(logits.argmax(dim=1), mask)

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

    denom = max(valid_batches, 1)
    results = metrics_tracker.compute()
    results['loss'] = running_loss / denom
    results.update({f'loss_{k}': v / denom for k, v in loss_components.items()})
    return results


# =============================================================================
# VALIDATE
# =============================================================================

def validate(model, loader, criterion, device, epoch, metrics_tracker,
             logger=None, use_tta=False):
    model.eval()
    metrics_tracker.reset()

    running_loss  = 0.0
    nan_count     = 0
    valid_batches = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch in pbar:
            rgb  = batch["rgb"].to(device)
            nir  = batch["nir"].to(device)
            mask = batch["mask"].to(device).long()

            if torch.isnan(rgb).any() or torch.isnan(nir).any():
                nan_count += 1
                continue

            if use_tta:
                logits = (model(rgb, nir) + model(rgb.flip(-1), nir.flip(-1)).flip(-1)) / 2.0
            else:
                logits = model(rgb, nir)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_count += 1
                continue

            loss = criterion(logits, mask)['total']
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

            preds = logits.argmax(dim=1)
            metrics_tracker.update(preds, mask)
            running_loss  += loss.item()
            valid_batches += 1

            if logger and epoch % 10 == 0:
                all_preds.extend(preds.cpu().numpy().flatten().astype(int))
                all_targets.extend(mask.cpu().numpy().flatten())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if nan_count > 0:
        print(f"[WARNING] Val: Skipped {nan_count} batches")

    if logger and epoch % 10 == 0 and len(all_preds) > 0:
        logger.log_confusion_matrix(epoch, np.array(all_preds), np.array(all_targets))

    results = metrics_tracker.compute()
    results['loss'] = running_loss / max(valid_batches, 1)
    return results


# =============================================================================
# SCHEDULER
# =============================================================================

def build_scheduler(optimizer, cfg):
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=cfg.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.t_max, eta_min=cfg.min_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[cfg.warmup_epochs])


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    print(f"[INFO] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start  = ckpt.get('epoch', 0) + 1
    best   = ckpt.get('metrics', {}).get('val_mean_iou', 0.0)
    print(f"[INFO] Resumed from epoch {start-1}, Best mIoU: {best:.4f}")
    return start, best


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg    = CFG
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR   = f"{cfg.log_dir_prefix}_{timestamp}"
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"[INFO] Logs → {LOG_DIR}")
    print(f"[INFO] TensorBoard: tensorboard --logdir={LOG_DIR}/tensorboard")

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
          f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[INFO] Building DualEncoderMini (3-class) ...")
    model = DualEncoderMini(
        rgb_base_ch=16,
        nir_base_ch=8,
        aspp_ch=64,
        num_classes=cfg.num_classes,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params / 1e6:.3f}M")

    # Sanity check
    model.eval()
    with torch.no_grad():
        dummy_out = model(torch.zeros(1, 3, *cfg.target_size, device=DEVICE),
                          torch.zeros(1, 1, *cfg.target_size, device=DEVICE))
    assert dummy_out.shape == (1, cfg.num_classes, *cfg.target_size), \
        f"Shape mismatch: {dummy_out.shape}"
    print(f"[INFO] Output shape verified: {dummy_out.shape}")
    model.train()

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────────────
    criterion = MulticlassCombinedLoss(cfg, DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler('cuda', init_scale=2.**10, growth_factor=1.5, backoff_factor=0.5)

    # ── Metrics / Logger ──────────────────────────────────────────────────────
    train_metrics = SegmentationMetrics(cfg.num_classes, cfg.class_names)
    val_metrics   = SegmentationMetrics(cfg.num_classes, cfg.class_names)

    logger = TrainingLogger(
        log_dir=LOG_DIR, model=model, val_loader=val_loader,
        device=DEVICE, num_vis_samples=cfg.num_vis_samples
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_miou         = 0.0
    epochs_no_improvement = 0
    start_epoch           = 1
    # start_epoch, best_val_miou = load_checkpoint("path/to/ckpt.pth", model, optimizer, scheduler)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.num_epochs}  LR={optimizer.param_groups[0]['lr']:.2e}")
        print('='*60)

        train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            DEVICE, epoch, scheduler, train_metrics,
            logger=logger, max_grad_norm=cfg.max_grad_norm, aux_weight=cfg.aux_weight
        )
        train_metrics.pretty_print(train_results, phase="Train")
        scheduler.step()

        if epoch % cfg.heavy_log_every == 0 or epoch == 1:
            logger.log_layer_statistics(epoch)

        use_tta     = (epoch % cfg.heavy_log_every == 0)
        val_results = validate(model, val_loader, criterion, DEVICE, epoch,
                               val_metrics, logger=logger, use_tta=use_tta)
        val_metrics.pretty_print(val_results, phase="Val")

        # TensorBoard
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_results['loss'], train_results['mean_iou'],
                         val_results['loss'], val_results['mean_iou'], current_lr)
        if hasattr(logger, 'writer'):
            for name in cfg.class_names:
                logger.writer.add_scalar(f"IoU/train_{name}", train_results[f'iou_{name}'], epoch)
                logger.writer.add_scalar(f"IoU/val_{name}",   val_results[f'iou_{name}'],   epoch)
                logger.writer.add_scalar(f"F1/val_{name}",    val_results[f'f1_{name}'],     epoch)
            logger.writer.add_scalar("Metrics/val_mean_iou",  val_results['mean_iou'],       epoch)
            logger.writer.add_scalar("Metrics/val_pixel_acc", val_results['pixel_accuracy'], epoch)
            for k in ['focal', 'dice', 'boundary', 'lovasz']:
                if f'loss_{k}' in train_results:
                    logger.writer.add_scalar(f"Loss/train_{k}", train_results[f'loss_{k}'], epoch)

        if epoch % cfg.vis_every == 0 or epoch == 1:
            print(f"[INFO] Visualising epoch {epoch}...")
            logger.visualize_predictions(epoch)
            logger.visualize_attention_gates_only(epoch)
            logger.visualize_layer_learning(epoch)
            logger.visualize_gradient_flow(epoch)

        print(f"[Epoch {epoch}] "
              f"Train loss={train_results['loss']:.4f}  mIoU={train_results['mean_iou']:.4f}  "
              f"crop={train_results['iou_crop']:.4f}  weed={train_results['iou_weed']:.4f}\n"
              f"         Val   loss={val_results['loss']:.4f}  mIoU={val_results['mean_iou']:.4f}  "
              f"crop={val_results['iou_crop']:.4f}  weed={val_results['iou_weed']:.4f}"
              f"{' (TTA)' if use_tta else ''}")

        is_best = val_results['mean_iou'] > best_val_miou
        if is_best:
            best_val_miou         = val_results['mean_iou']
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        logger.save_checkpoint(epoch, model, optimizer, scheduler, {
            'val_mean_iou':   val_results['mean_iou'],
            'val_loss':       val_results['loss'],
            'train_mean_iou': train_results['mean_iou'],
            'val_iou_crop':   val_results['iou_crop'],
            'val_iou_weed':   val_results['iou_weed'],
        }, is_best=is_best)

        if epochs_no_improvement >= cfg.patience:
            print(f"[INFO] Early stopping — {cfg.patience} epochs without improvement.")
            break

    print(f"\n[INFO] Training complete! Best Val mIoU: {best_val_miou:.4f}")
    logger.close()
    print(f"[INFO] Results → {LOG_DIR}")
    print(f"[INFO] TensorBoard: tensorboard --logdir={LOG_DIR}/tensorboard")


if __name__ == "__main__":
    main()