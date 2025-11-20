#!/usr/bin/env python3

"""
Training script for WeedsGalore multi-class semantic segmentation
Dataset: WeedsGalore (3 classes: Background, Crop, Weed)
Features:
- RGB+NIR (4 channels) or RGB-only (3 channels)
- Advanced agricultural augmentations
- Multi-class metrics: mIoU, class-wise IoU, pixel accuracy, F1, precision, recall
- FLOPs and parameter counting
- TensorBoard logging with run-specific directories
- Comprehensive checkpointing
"""

import argparse
import os
import json
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from weedsgalore_data_loader import create_weedsgalore_dataloaders
from models import create_model, get_model_info


class MultiClassIoULossMetric(nn.Module):
    """
    IoU Loss Metric for multi-class segmentation.
    
    This computes the mean IoU loss across all classes, useful as a
    differentiable metric for monitoring (not necessarily for training loss).
    
    Args:
        num_classes: Number of classes (3 for WeedsGalore: Background, Crop, Weed)
        eps: Small epsilon for numerical stability
        ignore_background: If True, compute IoU only for foreground classes (Crop, Weed)
    """
    def __init__(self, num_classes=3, eps=1e-6, ignore_background=False):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw model outputs before softmax
            targets: (B, H, W) - ground truth class indices
        
        Returns:
            1 - mean_iou: Loss value (lower is better)
        """
        # Get class probabilities
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        
        # Convert targets to one-hot encoding
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Compute IoU per class
        class_ious = []
        start_class = 1 if self.ignore_background else 0
        
        for c in range(start_class, self.num_classes):
            # Get predictions and targets for this class
            pred_c = probs[:, c, :, :]  # (B, H, W)
            target_c = targets_one_hot[:, c, :, :]  # (B, H, W)
            
            # Flatten spatial dimensions
            pred_c = pred_c.reshape(pred_c.size(0), -1)  # (B, H*W)
            target_c = target_c.reshape(target_c.size(0), -1)  # (B, H*W)
            
            # Compute intersection and union
            intersection = (pred_c * target_c).sum(1)  # (B,)
            union = pred_c.sum(1) + target_c.sum(1) - intersection  # (B,)
            
            # Compute IoU for this class
            iou = (intersection + self.eps) / (union + self.eps)
            class_ious.append(iou.mean())
        
        # Mean IoU across classes
        mean_iou = torch.stack(class_ious).mean()
        
        # Return loss (1 - IoU, so minimizing this maximizes IoU)
        return 1 - mean_iou

# ======================== Loss Function ========================

class MultiClassDiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy loss for multi-class segmentation"""
    def __init__(self, num_classes: int = 3, dice_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) where C = num_classes
            targets: (B, H, W) with class indices
        """
        # CrossEntropy loss
        ce_loss = self.ce(logits, targets)

        # Dice loss per class
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean()

        return self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss

# ======================== Metrics ========================

def compute_multiclass_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for multi-class segmentation.

    Args:
        preds: (B, H, W) predicted class indices
        targets: (B, H, W) ground truth class indices
        num_classes: Number of classes

    Returns:
        dict: Metrics including mIoU, class-wise IoU, pixel accuracy, F1, precision, recall
    """
    metrics = {}

    # Pixel accuracy
    correct = (preds == targets).sum().item()
    total = targets.numel()
    metrics['pixel_accuracy'] = correct / total

    # Per-class metrics
    class_ious = []
    class_precisions = []
    class_recalls = []
    class_f1s = []

    class_names = ['Background', 'Crop', 'Weed']

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        # IoU
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        iou = (intersection / (union + 1e-6)).item()
        class_ious.append(iou)
        metrics[f'iou_{class_names[cls]}'] = iou

        # Precision, Recall, F1
        tp = (pred_mask & target_mask).sum().float()
        fp = (pred_mask & ~target_mask).sum().float()
        fn = (~pred_mask & target_mask).sum().float()

        precision = (tp / (tp + fp + 1e-6)).item()
        recall = (tp / (tp + fn + 1e-6)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-6))

        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)

        metrics[f'precision_{class_names[cls]}'] = precision
        metrics[f'recall_{class_names[cls]}'] = recall
        metrics[f'f1_{class_names[cls]}'] = f1

    # Mean metrics
    metrics['miou'] = np.mean(class_ious)
    metrics['mean_precision'] = np.mean(class_precisions)
    metrics['mean_recall'] = np.mean(class_recalls)
    metrics['mean_f1'] = np.mean(class_f1s)

    return metrics


# ======================== Training & Evaluation ========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn,
    device,
    epoch: int,
    writer: SummaryWriter,
    num_classes: int = 3,
    scaler=None
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]")

    for batch_idx, batch in enumerate(pbar):
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()  # (B, H, W)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)  # (B, C, H, W)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        # Collect predictions for metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)  # (B, H, W)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Log to TensorBoard every 10 batches
        if batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)

    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_multiclass_metrics(all_preds, all_targets, num_classes)
    metrics['loss'] = epoch_loss / len(loader)

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device,
    num_classes: int = 3,
    split: str = "Val"
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_iou_loss = 0.0
    all_preds = []
    all_targets = []
    iou_loss_fn = MultiClassIoULossMetric(num_classes=3)
    pbar = tqdm(loader, desc=f"[{split}]")

    for batch in pbar:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()  # (B, H, W)

        logits = model(x)  # (B, C, H, W)
        loss = loss_fn(logits, y)
        total_loss += loss.item()

        iou_loss_value = iou_loss_fn(logits, y).item()
        total_iou_loss += iou_loss_value

        preds = torch.argmax(logits, dim=1)  # (B, H, W)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_multiclass_metrics(all_preds, all_targets, num_classes)
    metrics['loss'] = total_loss / len(loader)
    metrics['iou_loss'] = total_iou_loss / len(loader)

    return metrics


# ======================== Main Training Loop ========================

def main():
    parser = argparse.ArgumentParser(
        description="Train WeedsGalore Multi-Class Segmentation Model"
    )

    # Model architecture
    parser.add_argument("--model", type=str, default="pspnet",
                        choices=["pspnet","deeplabsv3+","lightsegnet","unet++","unet3+"],
                        help="Model architecture to use")
    parser.add_argument("--base_ch", type=int, default=32,
                        help="Base number of channels for the model")

    # Data
    parser.add_argument("--data_root", type=str, default="/home/vjtiadmin/Desktop/BTechGroup/weedsgalore-dataset",
                        help="Path to weedsgalore-dataset folder")
    parser.add_argument("--use_rgbnir", action="store_true",
                        help="Use RGB+NIR (4 channels), otherwise RGB only (3 channels)")
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--width", type=int, default=600)

    # Augmentation
    parser.add_argument("--augment", default=True, action="store_true",
                        help="Enable advanced agricultural augmentations")
    parser.add_argument("--nir_drop_prob", type=float, default=0.0,
                        help="Probability of dropping NIR channel during training")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")

    # Loss
    parser.add_argument("--dice_weight", type=float, default=0.5,
                        help="Weight for Dice loss (1-weight for CE loss)")

    # Checkpointing
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--output_dir", type=str, default="./experiments_weedsgalore",
                        help="Root directory for all experiments")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    # Multi-class segmentation (3 classes)
    num_classes = 3
    class_names = ["Background", "Crop", "Weed"]

    # Create run-specific directory
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        channels = "4ch_RGBNIR" if args.use_rgbnir else "3ch_RGB"
        aug_str = "aug" if args.augment else "noaug"
        args.exp_name = f"weedsgalore_{args.model}_{channels}_{aug_str}_{timestamp}"

    run_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(run_dir, "config.json")
    config_dict = {
        **vars(args),
        "num_classes": num_classes,
        "class_names": class_names,
        "dataset": "weedsgalore"
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n{'='*70}")
    print(f"[INFO] WeedsGalore Multi-Class Segmentation Training")
    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Config saved to: {config_path}")
    print(f"{'='*70}\n")

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Dataloaders
    print("[INFO] Creating dataloaders...")
    train_loader, val_loader = create_weedsgalore_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_rgbnir=args.use_rgbnir,
        target_size=(args.height, args.width),
        nir_drop_prob=args.nir_drop_prob,
        augment=args.augment
    )

    print(f"[INFO] Train samples: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}")

    # Model
    in_ch = 4 if args.use_rgbnir else 3
    model = create_model(
        architecture=args.model,
        in_channels=in_ch,
        num_classes=num_classes,
        base_ch=args.base_ch
    )
    model = model.to(device)

    # Model info (FLOPs and parameters)
    model_info = get_model_info(model)
    print(f"\n[INFO] Model: {model_info['architecture']}")
    print(f"[INFO] Total parameters: {model_info['total_parameters']:,} "
          f"({model_info['total_parameters_million']:.2f}M)")

    # Compute FLOPs
    try:
        from thop import profile
        dummy_input = torch.randn(1, in_ch, args.height, args.width).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_giga = flops / 1e9
        print(f"[INFO] FLOPs: {flops:,} ({flops_giga:.2f} GFLOPs)")
        model_info['flops'] = float(flops)
        model_info['flops_gflops'] = flops_giga
    except ImportError:
        print("[WARNING] thop not installed. Install with: pip install thop")
        print("[WARNING] FLOPs calculation skipped")
        model_info['flops'] = None
        model_info['flops_gflops'] = None

    # Save model info
    model_info_path = os.path.join(run_dir, "model_info.json")
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"[INFO] Model info saved to: {model_info_path}")

    # Log model graph to TensorBoard
    try:
        dummy_input = torch.randn(1, in_ch, args.height, args.width).to(device)
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print(f"[WARNING] Could not log model graph: {e}")

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = MultiClassDiceCELoss(num_classes=num_classes, dice_weight=args.dice_weight)

    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None

    # Resume training if checkpoint provided
    start_epoch = 1
    best_miou = 0.0

    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_val_miou", 0.0)
        print(f"[INFO] Resumed from epoch {checkpoint['epoch']}, best mIoU: {best_miou:.4f}")

    # Training loop
    print(f"\n[INFO] Starting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, writer, num_classes, scaler
        )

        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, device, num_classes, split="Val")

        # Log to TensorBoard
        for key, val in train_metrics.items():
            writer.add_scalar(f"Epoch/Train_{key}", val, epoch)
        for key, val in val_metrics.items():
            writer.add_scalar(f"Epoch/Val_{key}", val, epoch)
        writer.add_scalar("Epoch/LR", optimizer.param_groups[0]["lr"], epoch)

        # Print epoch summary
        print(f"\n[Epoch {epoch:03d}/{args.epochs}]")
        print(f"{'─'*70}")
        print(f"TRAIN → Loss: {train_metrics['loss']:.4f} | mIoU: {train_metrics['miou']*100:.2f}% | "
              f"PixAcc: {train_metrics['pixel_accuracy']*100:.2f}%")
        print(f"        IoU → BG: {train_metrics['iou_Background']*100:.2f}% | "
              f"Crop: {train_metrics['iou_Crop']*100:.2f}% | Weed: {train_metrics['iou_Weed']*100:.2f}%")
        print(f"VAL   → Loss: {val_metrics['loss']:.4f} | mIoU: {val_metrics['miou']*100:.2f}% | "
              f"PixAcc: {val_metrics['pixel_accuracy']*100:.2f}%")
        print(f"        IoU → BG: {val_metrics['iou_Background']*100:.2f}% | "
              f"Crop: {val_metrics['iou_Crop']*100:.2f}% | Weed: {val_metrics['iou_Weed']*100:.2f}%")
        print(f"        F1  → BG: {val_metrics['f1_Background']*100:.2f}% | "
              f"Crop: {val_metrics['f1_Crop']*100:.2f}% | Weed: {val_metrics['f1_Weed']*100:.2f}%")
        print(f"{'─'*70}")

        # Save best checkpoint
        current_miou = val_metrics['miou']
        if current_miou > best_miou:
            best_miou = current_miou
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_miou": best_miou,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "config": config_dict,
                "model_info": model_info
            }, best_ckpt_path)
            print(f"Saved best checkpoint: {best_ckpt_path} (mIoU: {best_miou*100:.2f}%)")

        # Save latest checkpoint
        latest_ckpt_path = os.path.join(ckpt_dir, "latest_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_miou": best_miou,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "config": config_dict,
            "model_info": model_info
        }, latest_ckpt_path)

        scheduler.step()

    # Save final results
    final_results = {
        "best_val_miou": best_miou,
        "final_val_metrics": {k: v for k, v in val_metrics.items()},
        "model_info": model_info,
        "config": config_dict
    }

    results_path = os.path.join(run_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    writer.close()

    print(f"\n{'='*70}")
    print(f"[INFO] Training complete! Results saved to: {run_dir}")
    print(f"[INFO] Best validation mIoU: {best_miou*100:.2f}%")
    print(f"[INFO] Final results saved to: {results_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
