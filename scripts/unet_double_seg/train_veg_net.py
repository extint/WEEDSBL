# ===================== STAGE 1: VEGETATION TRAINING =====================
# Binary segmentation: vegetation (1) vs background (0)
#
# Model: UNet with 4-channel input (RGB + NIR concatenated)

import os
import cv2
import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from models import UNet                                         # <-- UNet from models.py
from scripts.dual_encoder.dual_encoder_data_loader_veg import (
    create_dual_encoder_dataloaders,
    DualEncoderWeedyRiceDataset,
)
from scripts.dual_encoder.logging_utils_veg import TrainingLogger


# ============ LOSS FUNCTIONS ============

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce  = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt   = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


def dice_loss(pred, target, smooth=1e-6):
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2)

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets.float())

        pred_probs = torch.sigmoid(logits)
        loss_dice  = dice_loss(pred_probs, targets)

        # Boundary-weighted BCE
        target_f    = targets.float().unsqueeze(1)
        target_edge = (
            F.max_pool2d(target_f, 3, stride=1, padding=1)
            - F.avg_pool2d(target_f, 3, stride=1, padding=1)
        )
        target_edge  = (target_edge.abs() > 0.1).float().squeeze(1)
        edge_weight  = 1 + 2 * target_edge
        loss_boundary = (
            F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
            * edge_weight
        ).mean()

        return loss_focal + loss_dice + 0.5 * loss_boundary


def compute_iou(pred, target, threshold=0.5):
    pred         = (pred > threshold).float()
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).item()


# ============ DATASET WRAPPER ============
# The existing DualEncoderWeedyRiceDataset returns {"rgb", "nir", "mask", "path"}.
# We subclass it to concatenate rgb+nir into a single (4, H, W) tensor so the
# UNet receives one input instead of two.

class UNetVegDataset(DualEncoderWeedyRiceDataset):
    """Wraps DualEncoderWeedyRiceDataset → merges rgb+nir into a 4-ch tensor."""

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        img    = torch.cat([sample["rgb"], sample["nir"]], dim=0)  # (4, H, W)
        return {"img": img, "mask": sample["mask"], "path": sample["path"]}


def create_unet_dataloaders(data_root, batch_size=4, num_workers=4,
                            target_size=(640, 640)):
    from torch.utils.data import DataLoader

    train_ds = UNetVegDataset(data_root, split="train",
                              target_size=target_size, augment=True)
    val_ds   = UNetVegDataset(data_root, split="val",
                              target_size=target_size, augment=False)
    test_ds  = UNetVegDataset(data_root, split="test",
                              target_size=target_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# ============ TRAINING ============

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    epoch, scheduler, logger=None, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    running_iou  = 0.0
    nan_count    = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        img  = batch["img"].to(device)    # (B, 4, H, W)
        mask = batch["mask"].to(device)   # (B, H, W)  binary 0/1

        if torch.isnan(img).any():
            nan_count += 1
            continue

        optimizer.zero_grad()
        with autocast():
            logits = model(img).squeeze(1)   # (B, H, W)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_count += 1
                continue

            loss = criterion(logits, mask)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step(epoch + batch_idx / len(loader))

        pred_probs    = torch.sigmoid(logits.detach())
        running_loss += loss.item()
        running_iou  += compute_iou(pred_probs, mask)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
        })

    if nan_count > 0:
        print(f"[WARNING] Skipped {nan_count} batches due to NaN/Inf")

    n = max(len(loader) - nan_count, 1)
    return running_loss / n, running_iou / n


def validate(model, loader, criterion, device, epoch,
             logger=None, use_tta=False):
    model.eval()
    running_loss = 0.0
    running_iou  = 0.0
    nan_count    = 0

    all_probs   = []
    all_preds   = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            img  = batch["img"].to(device)
            mask = batch["mask"].to(device)

            if torch.isnan(img).any():
                nan_count += 1
                continue

            if use_tta:
                p1          = torch.sigmoid(model(img).squeeze(1))
                p2          = torch.sigmoid(model(img.flip(-1)).squeeze(1)).flip(-1)
                pred_probs  = (p1 + p2) / 2.0
                logits      = torch.logit(pred_probs.clamp(1e-7, 1 - 1e-7))
            else:
                out = model(img)
                if torch.isnan(out).any() or torch.isinf(out).any():
                    nan_count += 1
                    continue
                logits     = out.squeeze(1)
                pred_probs = torch.sigmoid(logits)

            loss = criterion(logits, mask)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

            running_loss += loss.item()
            running_iou  += compute_iou(pred_probs, mask)

            if logger and epoch % 10 == 0:
                all_probs.extend(pred_probs.cpu().numpy().flatten())
                all_preds.extend((pred_probs > 0.5).cpu().numpy().flatten().astype(int))
                all_targets.extend(mask.cpu().numpy().flatten())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if nan_count > 0:
        print(f"[WARNING] Val: Skipped {nan_count} batches")

    if logger and epoch % 10 == 0 and len(all_probs) > 0:
        logger.log_confusion_matrix(epoch, np.array(all_preds), np.array(all_targets))
        logger.log_pr_curve(epoch, np.array(all_probs), np.array(all_targets))

    n = max(len(loader) - nan_count, 1)
    return running_loss / n, running_iou / n


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_iou    = ckpt.get('metrics', {}).get('val_iou', 0.0)
    print(f"[INFO] Resumed from epoch {start_epoch - 1}, Best IoU: {best_iou:.4f}")
    return start_epoch, best_iou


# ============ MAIN ============

def main():
    DATA_ROOT      = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET"
    BATCH_SIZE     = 4
    NUM_WORKERS    = 4
    TARGET_SIZE    = (640, 640)
    NUM_EPOCHS     = 150
    LR             = 5e-5
    WEIGHT_DECAY   = 1e-4
    MAX_GRAD_NORM  = 1.0
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR   = f"unet_runs/unet_veg_{timestamp}"
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"[INFO] Logs → {LOG_DIR}")

    # ── Data ────────────────────────────────────────────────────────────────
    print("[INFO] Loading data...")
    train_loader, val_loader, test_loader = create_unet_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        target_size=TARGET_SIZE,
    )
    print(f"[INFO] Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # ── Model ───────────────────────────────────────────────────────────────
    print("[INFO] Building UNet (4-ch input, binary output)...")
    model = UNet(
        in_channels=4,    # RGB (3) + NIR (1)
        base_ch=64,
        out_channels=1,   # binary vegetation mask
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params / 1e6:.2f}M")

    # ── Loss / Optimiser / Scheduler ────────────────────────────────────────
    criterion = CombinedLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler    = GradScaler(init_scale=2.**10, growth_factor=1.5, backoff_factor=0.5)

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = TrainingLogger(
        log_dir=LOG_DIR,
        model=model,
        val_loader=val_loader,
        device=DEVICE,
        num_vis_samples=4,
    )

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_iou            = 0.0
    patience                = 25
    epochs_without_improve  = 0
    start_epoch             = 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print('='*60)

        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            DEVICE, epoch, scheduler, logger=logger,
            max_grad_norm=MAX_GRAD_NORM,
        )

        if epoch % 10 == 0 or epoch == 1:
            logger.log_layer_statistics(epoch)

        use_tta = (epoch % 10 == 0)
        val_loss, val_iou = validate(
            model, val_loader, criterion, DEVICE, epoch,
            logger=logger, use_tta=use_tta,
        )

        current_lr = scheduler.get_last_lr()[0]
        logger.log_epoch(epoch, train_loss, train_iou, val_loss, val_iou, current_lr)

        if epoch % 10 == 0 or epoch == 1:
            logger.visualize_predictions(epoch)
            logger.visualize_gradient_flow(epoch)
            logger.visualize_layer_learning(epoch)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f}"
              f"{' (TTA)' if use_tta else ''}")

        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou          = val_iou
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        metrics = {
            "train_loss": train_loss, "train_iou": train_iou,
            "val_loss":   val_loss,   "val_iou":   val_iou,
            "lr":         current_lr,
        }
        logger.save_checkpoint(epoch, model, optimizer, scheduler, metrics, is_best=is_best)

        if epochs_without_improve >= patience:
            print(f"[INFO] Early stopping after {patience} epochs without improvement")
            break

    print(f"\n[INFO] Training complete! Best Val IoU: {best_val_iou:.4f}")
    logger.close()
    print(f"[INFO] All results saved to: {LOG_DIR}")
    print(f"[INFO] View TensorBoard: tensorboard --logdir={LOG_DIR}/tensorboard")


if __name__ == "__main__":
    main()