#!/usr/bin/env python3

"""
Universal training script for agricultural weed segmentation
Supports:
  - Weedy Rice dataset (binary segmentation: background/weed)
  - WeedsGalore dataset (3-class segmentation: background/crop/weed)
Architectures: U-Net, LightMANet, U-Net_SA
Features:
  - RGB+NIR (4 channels) or RGB-only (3 channels)
  - TensorBoard logging with run-specific directories
  - Modular preprocessing for reuse in inference
  - Comprehensive checkpointing and metrics
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

# Import dataset loaders
from rice_weed_data_loader import create_weedy_rice_rgbnir_dataloaders
from weedsgalore_data_loader import create_weedsgalore_dataloaders

from models import create_model, get_model_info


# ======================== Loss Functions ========================

class BCEDiceLoss(nn.Module):
    """Binary segmentation loss (for Weedy Rice dataset)"""
    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = bce_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets.float())
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dims)
        union = probs.sum(dims) + targets.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean()
        return self.w * bce + (1 - self.w) * dice_loss


class MultiClassDiceLoss(nn.Module):
    """Multi-class segmentation loss (for WeedsGalore dataset)"""
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) where C = num_classes
            targets: (B, H, W) with class indices
        """
        ce_loss = self.ce(logits, targets)

        # Compute Dice loss per class
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean()

        return 0.5 * ce_loss + 0.5 * dice_loss


# ======================== Training & Evaluation ========================

def train_one_epoch_binary(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn,
    device,
    epoch: int,
    writer: SummaryWriter,
    scaler=None
) -> Dict[str, float]:
    """Train for one epoch - BINARY segmentation (Weedy Rice)"""
    model.train()
    epoch_loss = 0.0
    ious, f1s = [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]")

    for batch_idx, batch in enumerate(pbar):
        x = batch["images"].to(device)
        y = batch["labels"].to(device).unsqueeze(1)  # (B, 1, H, W)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        # Compute batch metrics
        epoch_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().squeeze(1)
            y_cpu = y.squeeze(1).long()

            inter = (preds & y_cpu).sum(dim=(1, 2)).float()
            union = (preds | y_cpu).sum(dim=(1, 2)).float().clamp_min(1)
            iou = (inter / union).mean().item()
            ious.append(iou)

            tp = (preds & y_cpu).sum(dim=(1, 2)).float()
            fp = (preds & (1 - y_cpu)).sum(dim=(1, 2)).float()
            fn = ((1 - preds) & y_cpu).sum(dim=(1, 2)).float()
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            f1 = (2 * prec * rec / (prec + rec + 1e-6)).mean().item()
            f1s.append(f1)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}"})

        # Log to TensorBoard every 10 batches
        if batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            writer.add_scalar("Train/BatchIoU", iou, global_step)

    metrics = {
        "loss": epoch_loss / len(loader),
        "iou": np.mean(ious),
        "f1": np.mean(f1s)
    }
    return metrics


def train_one_epoch_multiclass(
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
    """Train for one epoch - MULTI-CLASS segmentation (WeedsGalore)"""
    model.train()
    epoch_loss = 0.0
    mious, f1s = [], []
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

        # Compute batch metrics
        epoch_loss += loss.item()
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)  # (B, H, W)

            # Mean IoU across all classes
            class_ious = []
            for cls in range(num_classes):
                pred_mask = (preds == cls)
                target_mask = (y == cls)
                inter = (pred_mask & target_mask).sum(dim=(1, 2)).float()
                union = (pred_mask | target_mask).sum(dim=(1, 2)).float().clamp_min(1)
                class_iou = (inter / union).mean().item()
                class_ious.append(class_iou)
            miou = np.mean(class_ious)
            mious.append(miou)

            # F1 score
            correct = (preds == y).sum().float()
            total = y.numel()
            acc = (correct / total).item()
            f1s.append(acc)  # Using accuracy as proxy for F1 in multiclass

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "mIoU": f"{miou:.4f}"})

        # Log to TensorBoard every 10 batches
        if batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            writer.add_scalar("Train/BatchmIoU", miou, global_step)

    metrics = {
        "loss": epoch_loss / len(loader),
        "miou": np.mean(mious),
        "acc": np.mean(f1s)
    }
    return metrics


@torch.no_grad()
def evaluate_binary(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device,
    split: str = "Val"
) -> Dict[str, float]:
    """Evaluate model - BINARY segmentation"""
    model.eval()
    total_loss = 0.0
    ious, f1s, precs, recs = [], [], [], []
    pbar = tqdm(loader, desc=f"[{split}]")

    for batch in pbar:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()
        y_1 = y.unsqueeze(1)

        logits = model(x)
        loss = loss_fn(logits, y_1)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().squeeze(1)

        inter = (preds & y).sum(dim=(1, 2)).float()
        union = (preds | y).sum(dim=(1, 2)).float().clamp_min(1)
        iou = (inter / union).mean().item()
        ious.append(iou)

        tp = (preds & y).sum(dim=(1, 2)).float()
        fp = (preds & (1 - y)).sum(dim=(1, 2)).float()
        fn = ((1 - preds) & y).sum(dim=(1, 2)).float()
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = (2 * prec * rec / (prec + rec + 1e-6)).mean().item()
        f1s.append(f1)
        precs.append(prec.mean().item())
        recs.append(rec.mean().item())

    metrics = {
        "loss": total_loss / len(loader),
        "iou": np.mean(ious),
        "f1": np.mean(f1s),
        "prec": np.mean(precs),
        "rec": np.mean(recs)
    }
    return metrics


@torch.no_grad()
def evaluate_multiclass(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device,
    num_classes: int = 3,
    split: str = "Val"
) -> Dict[str, float]:
    """Evaluate model - MULTI-CLASS segmentation"""
    model.eval()
    total_loss = 0.0
    mious, accs = [], []
    class_ious_all = [[] for _ in range(num_classes)]
    pbar = tqdm(loader, desc=f"[{split}]")

    for batch in pbar:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()  # (B, H, W)

        logits = model(x)  # (B, C, H, W)
        loss = loss_fn(logits, y)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)  # (B, H, W)

        # Per-class IoU
        class_ious = []
        for cls in range(num_classes):
            pred_mask = (preds == cls)
            target_mask = (y == cls)
            inter = (pred_mask & target_mask).sum(dim=(1, 2)).float()
            union = (pred_mask | target_mask).sum(dim=(1, 2)).float().clamp_min(1)
            class_iou = (inter / union).mean().item()
            class_ious.append(class_iou)
            class_ious_all[cls].append(class_iou)

        miou = np.mean(class_ious)
        mious.append(miou)

        # Accuracy
        correct = (preds == y).sum().float()
        total = y.numel()
        acc = (correct / total).item()
        accs.append(acc)

    metrics = {
        "loss": total_loss / len(loader),
        "miou": np.mean(mious),
        "acc": np.mean(accs),
        "iou_class0": np.mean(class_ious_all[0]),
        "iou_class1": np.mean(class_ious_all[1]),
        "iou_class2": np.mean(class_ious_all[2])
    }
    return metrics


# ======================== Main Training Loop ========================

def main():
    parser = argparse.ArgumentParser(
        description="Train Agricultural Weed Segmentation Model"
    )

    # Dataset selection
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["weedy_rice", "weedsgalore"],
                        help="Dataset to use: weedy_rice (binary) or weedsgalore (3-class)")

    # Model architecture
    parser.add_argument("--model", type=str, default="lightmanet",
                        choices=["unet", "unet_sa", "lightmanet"],
                        help="Model architecture to use")
    parser.add_argument("--base_ch", type=int, default=32,
                        help="Base number of channels for the model")

    # Data
    parser.add_argument("--data_root", type=str,
                        help="Dataset root directory", default="/home/vjtiadmin/Desktop/BTechGroup/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB")
    parser.add_argument("--use_rgbnir", action="store_true",
                        help="Use RGB+NIR (4 channels), otherwise RGB only (3 channels)")
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--width", type=int, default=600)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--nir_drop", type=float, default=0.0)

    # Checkpointing
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Root directory for all experiments")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    # Determine number of classes based on dataset
    if args.dataset == "weedy_rice":
        num_classes = 1  # Binary segmentation
        class_names = ["background", "weed"]
    else:  # weedsgalore
        num_classes = 3  # Multi-class segmentation
        class_names = ["background", "crop", "weed"]

    # Create run-specific directory
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        channels = "4ch_RGBNIR" if args.use_rgbnir else "3ch_RGB"
        args.exp_name = f"{args.dataset}_{args.model}_{channels}_{timestamp}"

    run_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(run_dir, "config.json")
    config_dict = {**vars(args), "num_classes": num_classes, "class_names": class_names}
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n{'='*70}")
    print(f"[INFO] Dataset: {args.dataset} ({num_classes} classes)")
    print(f"[INFO] Class names: {class_names}")
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
    if args.dataset == "weedy_rice":
        train_loader, val_loader, test_loader = create_weedy_rice_rgbnir_dataloaders(
            data_root=args.data_root,
            use_rgbnir=args.use_rgbnir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=(args.height, args.width),
            nir_drop_prob=args.nir_drop
        )
    else:  # weedsgalore
        train_loader, val_loader = create_weedsgalore_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_rgbnir=args.use_rgbnir,
            target_size=(args.height, args.width),
            nir_drop_prob=args.nir_drop
        )
        test_loader = None  # WeedsGalore doesn't have separate test set

    print(f"[INFO] Train samples: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}" + 
          (f", Test: {len(test_loader.dataset)}" if test_loader else ""))
    print(f"[INFO] Using nir drop prob={args.nir_drop}")

    # Model
    in_ch = 4 if args.use_rgbnir else 3
    model = create_model(
        architecture=args.model,
        in_channels=in_ch,
        num_classes=num_classes,
        base_ch=args.base_ch
    )
    model = model.to(device)

    # Model info
    model_info = get_model_info(model)
    print(f"[INFO] Model: {model_info['architecture']}")
    print(f"[INFO] Total parameters: {model_info['total_parameters']:,} "
          f"({model_info['total_parameters_million']:.2f}M)")

    # Log model graph to TensorBoard
    dummy_input = torch.randn(1, in_ch, args.height, args.width).to(device)
    writer.add_graph(model, dummy_input)

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.dataset == "weedy_rice":
        loss_fn = BCEDiceLoss(bce_weight=0.5)
    else:
        loss_fn = MultiClassDiceLoss(num_classes=num_classes)

    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None

    # Resume training if checkpoint provided
    start_epoch = 1
    best_val_metric = 0.0
    metric_name = "iou" if args.dataset == "weedy_rice" else "miou"

    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_metric = checkpoint.get(f"best_val_{metric_name}", 0.0)
        print(f"[INFO] Resumed from epoch {checkpoint['epoch']}, best {metric_name}: {best_val_metric:.4f}")

    # Training loop
    print(f"\n[INFO] Starting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        if args.dataset == "weedy_rice":
            train_metrics = train_one_epoch_binary(
                model, train_loader, optimizer, loss_fn, device, epoch, writer, scaler
            )
            val_metrics = evaluate_binary(model, val_loader, loss_fn, device, split="Val")
        else:
            train_metrics = train_one_epoch_multiclass(
                model, train_loader, optimizer, loss_fn, device, epoch, writer, num_classes, scaler
            )
            val_metrics = evaluate_multiclass(model, val_loader, loss_fn, device, num_classes, split="Val")

        # Log to TensorBoard
        for key, val in train_metrics.items():
            writer.add_scalar(f"Epoch/Train_{key}", val, epoch)
        for key, val in val_metrics.items():
            writer.add_scalar(f"Epoch/Val_{key}", val, epoch)
        writer.add_scalar("Epoch/LR", optimizer.param_groups[0]["lr"], epoch)

        # Print epoch summary
        if args.dataset == "weedy_rice":
            print(f"[Epoch {epoch:03d}/{args.epochs}] "
                  f"Train Loss: {train_metrics['loss']:.4f} | IoU: {train_metrics['iou']:.4f} | F1: {train_metrics['f1']:.4f} || "
                  f"Val Loss: {val_metrics['loss']:.4f} | IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f} | "
                  f"Precision: {val_metrics['prec']:.4f} | Recall: {val_metrics['rec']:.4f}")
        else:
            print(f"[Epoch {epoch:03d}/{args.epochs}] "
                  f"Train Loss: {train_metrics['loss']:.4f} | mIoU: {train_metrics['miou']:.4f} | Acc: {train_metrics['acc']:.4f} || "
                  f"Val Loss: {val_metrics['loss']:.4f} | mIoU: {val_metrics['miou']:.4f} | Acc: {val_metrics['acc']:.4f} | "
                  f"IoU[BG/Crop/Weed]: [{val_metrics['iou_class0']:.3f}/{val_metrics['iou_class1']:.3f}/{val_metrics['iou_class2']:.3f}]")

        # Save best checkpoint
        current_metric = val_metrics[metric_name]
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                f"best_val_{metric_name}": best_val_metric,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "config": config_dict
            }, best_ckpt_path)
            print(f"[INFO] ✓ Saved best checkpoint: {best_ckpt_path} ({metric_name}: {best_val_metric:.4f})")

        # Save latest checkpoint
        latest_ckpt_path = os.path.join(ckpt_dir, "latest_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            f"best_val_{metric_name}": best_val_metric,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "config": config_dict
        }, latest_ckpt_path)

        scheduler.step()

    # Final test evaluation (only for weedy_rice)
    if test_loader is not None:
        print(f"\n{'='*70}")
        print("[INFO] Running final test evaluation...")
        print(f"{'='*70}\n")

        if args.dataset == "weedy_rice":
            test_metrics = evaluate_binary(model, test_loader, loss_fn, device, split="Test")
            print(f"[Test Results] Loss: {test_metrics['loss']:.4f} | "
                  f"IoU: {test_metrics['iou']:.4f} | F1: {test_metrics['f1']:.4f} | "
                  f"Precision: {test_metrics['prec']:.4f} | Recall: {test_metrics['rec']:.4f}")

        for key, val in test_metrics.items():
            writer.add_scalar(f"Final/Test_{key}", val, 0)

        # Save test results
        results_path = os.path.join(run_dir, "test_results.json")
        with open(results_path, "w") as f:
            json.dump({
                **test_metrics,
                f"best_val_{metric_name}": best_val_metric,
                "model_info": model_info
            }, f, indent=2)

    writer.close()
    print(f"\n{'='*70}")
    print(f"[INFO] Training complete! Results saved to: {run_dir}")
    print(f"[INFO] Best validation {metric_name}: {best_val_metric:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
