import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np

from dual_encoder.updated_architecture import DualEncoderAFFNet
from dual_encoder.heavy import *
from dual_encoder.logging_utils import TrainingLogger

# ============ IMPROVED LOSS FUNCTIONS ============

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss - pred should already be probabilities"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2)
    
    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets.float())
        
        pred_probs = torch.sigmoid(logits)
        loss_dice = dice_loss(pred_probs, targets)
        
        # Boundary loss
        target_f = targets.float().unsqueeze(1)
        target_edge = F.max_pool2d(target_f, 3, stride=1, padding=1) - F.avg_pool2d(target_f, 3, stride=1, padding=1)
        target_edge = (target_edge.abs() > 0.1).float().squeeze(1)
        edge_weight = 1 + 2 * target_edge
        loss_boundary = (F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none') * edge_weight).mean()
        
        return loss_focal + loss_dice + 0.5 * loss_boundary


def compute_iou(pred, target, threshold=0.5):
    """Compute IoU for binary segmentation"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


# ============ TRAINING ============
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, scheduler, 
                    logger=None, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    nan_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        rgb = batch["rgb"].to(device)
        nir = batch["nir"].to(device)
        mask = batch["mask"].to(device)
        
        if torch.isnan(rgb).any() or torch.isnan(nir).any():
            nan_count += 1
            continue

        optimizer.zero_grad()

        with autocast():
            logits = model(rgb, nir)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_count += 1
                continue
            
            logits_squeezed = logits.squeeze(1)
            loss = criterion(logits_squeezed, mask)
            
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step(epoch + batch_idx / len(loader))

        pred_probs = torch.sigmoid(logits_squeezed.detach())
        running_loss += loss.item()
        running_iou += compute_iou(pred_probs, mask)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        
        # Log gradients every 100 batches
        # if logger and batch_idx % 100 == 0:
        #     logger.log_layer_statistics(epoch)

    if nan_count > 0:
        print(f"[WARNING] Skipped {nan_count} batches due to NaN/Inf")

    avg_loss = running_loss / max(len(loader) - nan_count, 1)
    avg_iou = running_iou / max(len(loader) - nan_count, 1)
    return avg_loss, avg_iou


def validate(model, loader, criterion, device, epoch, logger=None, use_tta=False):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    nan_count = 0
    
    # For PR curve and confusion matrix
    all_probs = []
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            rgb = batch["rgb"].to(device)
            nir = batch["nir"].to(device)
            mask = batch["mask"].to(device)
            
            if torch.isnan(rgb).any() or torch.isnan(nir).any():
                nan_count += 1
                continue

            if use_tta:
                pred1 = torch.sigmoid(model(rgb, nir).squeeze(1))
                pred2 = torch.sigmoid(model(rgb.flip(-1), nir.flip(-1)).squeeze(1)).flip(-1)
                pred_probs = (pred1 + pred2) / 2.0
                logits_squeezed = torch.logit(pred_probs.clamp(1e-7, 1-1e-7))
            else:
                logits = model(rgb, nir)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    nan_count += 1
                    continue
                logits_squeezed = logits.squeeze(1)
                pred_probs = torch.sigmoid(logits_squeezed)
            
            loss = criterion(logits_squeezed, mask)
            
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

            running_loss += loss.item()
            running_iou += compute_iou(pred_probs, mask)
            
            # Collect for PR curve and confusion matrix
            if logger and epoch % 10 == 0:  # Only every 10 epochs to save memory
                all_probs.extend(pred_probs.cpu().numpy().flatten())
                all_preds.extend((pred_probs > 0.5).cpu().numpy().flatten().astype(int))
                all_targets.extend(mask.cpu().numpy().flatten())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if nan_count > 0:
        print(f"[WARNING] Val: Skipped {nan_count} batches")

    avg_loss = running_loss / max(len(loader) - nan_count, 1)
    avg_iou = running_iou / max(len(loader) - nan_count, 1)
    
    # Log additional metrics every 10 epochs
    if logger and epoch % 10 == 0 and len(all_probs) > 0:
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        logger.log_confusion_matrix(epoch, all_preds, all_targets)
        logger.log_pr_curve(epoch, all_probs, all_targets)
    
    return avg_loss, avg_iou
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint and resume training"""
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_iou = checkpoint.get('metrics', {}).get('val_iou', 0.0)
    
    print(f"[INFO] Resumed from epoch {start_epoch-1}, Best IoU: {best_iou:.4f}")
    return start_epoch, best_iou


def main():
    DATA_ROOT = "/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB"
    
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    TARGET_SIZE = (640, 640)
    NUM_EPOCHS = 150
    LR = 5e-5
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    WARMUP_EPOCHS = 10
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ============ LOGGING SETUP ============
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = f"dual_encoder/runs/dual_encoder_{timestamp}"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"[INFO] Logs will be saved to: {LOG_DIR}")
    print(f"[INFO] To view TensorBoard: tensorboard --logdir={LOG_DIR}/tensorboard")

    # ============ Data ============
    print("[INFO] Loading data...")
    train_loader, val_loader, test_loader = create_heavy_augmented_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        target_size=TARGET_SIZE,
        crops_per_image_train=3
    )

    print(f"[INFO] Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # ============ Model ============
    print("[INFO] Building DualEncoderAFFNet...")
    model = DualEncoderAFFNet(
        rgb_variant='small',
        nir_base_ch=20,
        num_classes=1,
        embed_dim=96
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params / 1e6:.2f}M")

    # ============ Loss & Optimizer =====
    #                 , =======
    criterion = CombinedLoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )
    
    scaler = GradScaler(init_scale=2.**10, growth_factor=1.5, backoff_factor=0.5)
    
    # ============ INITIALIZE LOGGER ============
    logger = TrainingLogger(
        log_dir=LOG_DIR,
        model=model,
        val_loader=val_loader,
        device=DEVICE,
        num_vis_samples=4  # Visualize 4 fixed samples
    )

    # ============ Training Loop ============
    best_val_iou = 0.0
    patience = 25
    epochs_without_improvement = 0
    # start_epoch, best_val_iou = load_checkpoint("path/to/checkpoint.pth", model, optimizer, scheduler)
    start_epoch = 1
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print('='*60)
        
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, 
            DEVICE, epoch, scheduler, logger=logger, max_grad_norm=MAX_GRAD_NORM
        )
        
        if epoch % 10 == 0 or epoch == 1:
            logger.log_layer_statistics(epoch)

        # Then validation
        use_tta = (epoch % 10 == 0)
        val_loss, val_iou = validate(
            model, val_loader, criterion, DEVICE, epoch, 
            logger=logger, use_tta=use_tta
        )

        # ============ LOG METRICS ============
        current_lr = scheduler.get_last_lr()[0]
        logger.log_epoch(epoch, train_loss, train_iou, val_loss, val_iou, current_lr)
        
        # Log model weights every 20 epochs
        # if epoch % 20 == 0:
        #     logger.log_model_weights(epoch)
        
        # ============ VISUALIZE EVERY 10 EPOCHS ============
        if epoch % 10 == 0 or epoch == 1:
            print(f"[INFO] Creating visualizations for epoch {epoch}...")
            
            # Predictions and attention gates
            logger.visualize_predictions(epoch)
            logger.visualize_attention_gates_only(epoch)
            
            # Layer-wise learning (NEW!)
            logger.log_layer_statistics(epoch)
            logger.visualize_layer_learning(epoch)
            logger.visualize_gradient_flow(epoch)
            # logger.visualize_feature_maps(epoch)

        
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}{' (TTA)' if use_tta else ''}")

        # ============ SAVE CHECKPOINTS ============
        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        metrics = {
            'train_loss': train_loss,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'lr': current_lr
        }
        
        logger.save_checkpoint(epoch, model, optimizer, scheduler, metrics, is_best=is_best)
        
        if epochs_without_improvement >= patience:
            print(f"[INFO] Early stopping after {patience} epochs without improvement")
            break

    print(f"\n[INFO] Training complete! Best Val IoU: {best_val_iou:.4f}")
    logger.close()
    
    print(f"\n[INFO] All results saved to: {LOG_DIR}")
    print(f"[INFO] View TensorBoard: tensorboard --logdir={LOG_DIR}/tensorboard")


if __name__ == "__main__":
    main()