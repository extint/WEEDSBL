#!/usr/bin/env python3

"""
Training script for Sugar Beets weed segmentation
Dataset: 3-class segmentation (background/crop/weed)
Architectures: U-Net, LightMANet, U-Net_SA
Features:
  - RGB+NIR (4 channels) or RGB-only (3 channels)
  - Comprehensive metrics (mIoU, per-class IoU, Dice, precision, recall, F1)
  - TensorBoard logging with run-specific directories
  - Mixed precision training support
  - Checkpointing and resumption
"""

import argparse
import os
import json
from datetime import datetime
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import dataset loader
from sota.sugarbeets_data_loader import create_sugarbeets_dataloaders

from sota.models import create_model, get_model_info


# ======================== Loss Functions ========================

class MultiClassDiceLoss(nn.Module):
    """Multi-class segmentation loss with Cross Entropy + Dice"""
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean()

        return 0.5 * ce_loss + 0.5 * dice_loss


class MultiClassIoULoss(nn.Module):
    """IoU loss for monitoring (not used for training)"""
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)
        iou = (intersection + self.eps) / (union + self.eps)
        iou_loss = 1 - iou.mean()

        return iou_loss


# ======================== Metrics Computation ========================

def compute_multiclass_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3
) -> Dict[str, float]:
    metrics = {}

    correct = (preds == targets).sum().item()
    total = targets.numel()
    metrics['pixel_accuracy'] = correct / total

    class_ious = []
    class_precisions = []
    class_recalls = []
    class_f1s = []
    class_dices = []

    class_names = ['Background', 'Crop', 'Weed']

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        iou = (intersection / (union + 1e-6)).item()
        class_ious.append(iou)
        metrics[f'iou_{class_names[cls]}'] = iou

        tp = (pred_mask & target_mask).sum().float()
        fp = (pred_mask & ~target_mask).sum().float()
        fn = (~pred_mask & target_mask).sum().float()

        precision = (tp / (tp + fp + 1e-6)).item()
        recall = (tp / (tp + fn + 1e-6)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-6))

        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()
        dice = (2 * tp / (pred_sum + target_sum + 1e-6)).item()

        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)
        class_dices.append(dice)

        metrics[f'precision_{class_names[cls]}'] = precision
        metrics[f'recall_{class_names[cls]}'] = recall
        metrics[f'f1_{class_names[cls]}'] = f1
        metrics[f'dice_{class_names[cls]}'] = dice

    metrics['miou'] = np.mean(class_ious)
    metrics['mean_precision'] = np.mean(class_precisions)
    metrics['mean_recall'] = np.mean(class_recalls)
    metrics['mean_f1'] = np.mean(class_f1s)
    metrics['mean_dice'] = np.mean(class_dices)

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
    model.train()
    epoch_loss = 0.0

    total_correct = 0
    total_pixels = 0

    class_tp = [0] * num_classes
    class_fp = [0] * num_classes
    class_fn = [0] * num_classes
    class_intersection = [0] * num_classes
    class_union = [0] * num_classes
    # Dice accumulators: sum of pred pixels + target pixels per class
    class_dice_intersection = [0] * num_classes
    class_dice_sum = [0] * num_classes

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]")

    for batch_idx, batch in enumerate(pbar):
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()

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

        epoch_loss += loss.item()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == y).sum().item()
            total_pixels += y.numel()

            for cls in range(num_classes):
                pred_mask = (preds == cls)
                target_mask = (y == cls)

                tp = (pred_mask & target_mask).sum().item()
                fp = (pred_mask & ~target_mask).sum().item()
                fn = (~pred_mask & target_mask).sum().item()

                class_tp[cls] += tp
                class_fp[cls] += fp
                class_fn[cls] += fn

                intersection = tp  # TP == intersection for binary masks
                union = (pred_mask | target_mask).sum().item()
                class_intersection[cls] += intersection
                class_union[cls] += union

                # Dice: 2*TP / (pred_positive + target_positive)
                class_dice_intersection[cls] += tp
                class_dice_sum[cls] += pred_mask.sum().item() + target_mask.sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)

    metrics = {}
    metrics['pixel_accuracy'] = total_correct / total_pixels

    class_names = ['Background', 'Crop', 'Weed']
    class_ious = []
    class_precisions = []
    class_recalls = []
    class_f1s = []
    class_dices = []

    for cls in range(num_classes):
        iou = class_intersection[cls] / (class_union[cls] + 1e-6)
        class_ious.append(iou)
        metrics[f'iou_{class_names[cls]}'] = iou

        precision = class_tp[cls] / (class_tp[cls] + class_fp[cls] + 1e-6)
        recall = class_tp[cls] / (class_tp[cls] + class_fn[cls] + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # Dice = 2*TP / (pred_positive + target_positive)
        dice = (2 * class_dice_intersection[cls]) / (class_dice_sum[cls] + 1e-6)

        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)
        class_dices.append(dice)

        metrics[f'precision_{class_names[cls]}'] = precision
        metrics[f'recall_{class_names[cls]}'] = recall
        metrics[f'f1_{class_names[cls]}'] = f1
        metrics[f'dice_{class_names[cls]}'] = dice

    metrics['miou'] = np.mean(class_ious)
    metrics['mean_precision'] = np.mean(class_precisions)
    metrics['mean_recall'] = np.mean(class_recalls)
    metrics['mean_f1'] = np.mean(class_f1s)
    metrics['mean_dice'] = np.mean(class_dices)
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
    model.eval()
    total_loss = 0.0
    total_iou_loss = 0.0
    iou_loss_fn = MultiClassIoULoss(num_classes=num_classes)

    total_correct = 0
    total_pixels = 0

    class_tp = [0] * num_classes
    class_fp = [0] * num_classes
    class_fn = [0] * num_classes
    class_intersection = [0] * num_classes
    class_union = [0] * num_classes
    # Dice accumulators
    class_dice_intersection = [0] * num_classes
    class_dice_sum = [0] * num_classes

    pbar = tqdm(loader, desc=f"[{split}]")

    for batch in pbar:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()

        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item()
        total_iou_loss += iou_loss_fn(logits, y).item()

        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == y).sum().item()
        total_pixels += y.numel()

        for cls in range(num_classes):
            pred_mask = (preds == cls)
            target_mask = (y == cls)

            tp = (pred_mask & target_mask).sum().item()
            fp = (pred_mask & ~target_mask).sum().item()
            fn = (~pred_mask & target_mask).sum().item()

            class_tp[cls] += tp
            class_fp[cls] += fp
            class_fn[cls] += fn

            union = (pred_mask | target_mask).sum().item()
            class_intersection[cls] += tp
            class_union[cls] += union

            # Dice: 2*TP / (pred_positive + target_positive)
            class_dice_intersection[cls] += tp
            class_dice_sum[cls] += pred_mask.sum().item() + target_mask.sum().item()

    metrics = {}
    metrics['pixel_accuracy'] = total_correct / total_pixels

    class_names = ['Background', 'Crop', 'Weed']
    class_ious = []
    class_precisions = []
    class_recalls = []
    class_f1s = []
    class_dices = []

    for cls in range(num_classes):
        iou = class_intersection[cls] / (class_union[cls] + 1e-6)
        class_ious.append(iou)
        metrics[f'iou_{class_names[cls]}'] = iou

        precision = class_tp[cls] / (class_tp[cls] + class_fp[cls] + 1e-6)
        recall = class_tp[cls] / (class_tp[cls] + class_fn[cls] + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        dice = (2 * class_dice_intersection[cls]) / (class_dice_sum[cls] + 1e-6)

        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)
        class_dices.append(dice)

        metrics[f'precision_{class_names[cls]}'] = precision
        metrics[f'recall_{class_names[cls]}'] = recall
        metrics[f'f1_{class_names[cls]}'] = f1
        metrics[f'dice_{class_names[cls]}'] = dice

    metrics['miou'] = np.mean(class_ious)
    metrics['mean_precision'] = np.mean(class_precisions)
    metrics['mean_recall'] = np.mean(class_recalls)
    metrics['mean_f1'] = np.mean(class_f1s)
    metrics['mean_dice'] = np.mean(class_dices)
    metrics['loss'] = total_loss / len(loader)
    metrics['iou_loss'] = total_iou_loss / len(loader)

    return metrics


# ======================== Main Training Loop ========================

def main():
    parser = argparse.ArgumentParser(
        description="Train Sugar Beets Weed Segmentation Model"
    )

    parser.add_argument("--model", type=str, default="lmanet",
                        choices=["deeplabsv3+", "lmanet"])
    parser.add_argument("--base_ch", type=int, default=16)
    parser.add_argument("--data_root", type=str,
                        default="/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET")
    parser.add_argument("--use_rgbnir", action="store_true")
    parser.add_argument("--height", type=int, default=966)
    parser.add_argument("--width", type=int, default=1296)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--nir_drop", type=float, default=0.0)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    num_classes = 3
    class_names = ["background", "crop", "weed"]

    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        channels = "4ch_RGBNIR" if args.use_rgbnir else "3ch_RGB"
        args.exp_name = f"sugarbeets_{args.model}_{channels}_{timestamp}"

    run_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "config.json")
    config_dict = {**vars(args), "num_classes": num_classes, "class_names": class_names}
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n{'='*80}")
    print(f"[INFO] Dataset: Sugar Beets ({num_classes} classes)")
    print(f"[INFO] Class names: {class_names}")
    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Config saved to: {config_path}")
    print(f"{'='*80}\n")

    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_sugarbeets_dataloaders(
        data_root=args.data_root,
        use_rgbnir=args.use_rgbnir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.height, args.width),
        nir_drop_prob=args.nir_drop
    )

    print(f"[INFO] Train samples: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")
    print(f"[INFO] Using NIR drop prob={args.nir_drop}")

    in_ch = 4 if args.use_rgbnir else 3
    model = create_model(
        architecture=args.model,
        in_channels=in_ch,
        num_classes=num_classes,
        base_ch=args.base_ch
    )
    model = model.to(device)

    model_info = get_model_info(model)
    print(f"[INFO] Model: {model_info['architecture']}")
    print(f"[INFO] Total parameters: {model_info['total_parameters']:,} "
          f"({model_info['total_parameters_million']:.2f}M)")

    dummy_input = torch.randn(1, in_ch, args.height, args.width).to(device)
    writer.add_graph(model, dummy_input)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = MultiClassDiceLoss(num_classes=num_classes)
    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None

    start_epoch = 1
    best_val_miou = 0.0

    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_miou = checkpoint.get("best_val_miou", 0.0)
        print(f"[INFO] Resumed from epoch {checkpoint['epoch']}, best mIoU: {best_val_miou:.4f}")

    print(f"\n[INFO] Starting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, writer, num_classes, scaler
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, num_classes, split="Val")

        for key, val in train_metrics.items():
            writer.add_scalar(f"Epoch/Train_{key}", val, epoch)
        for key, val in val_metrics.items():
            writer.add_scalar(f"Epoch/Val_{key}", val, epoch)
        writer.add_scalar("Epoch/LR", optimizer.param_groups[0]["lr"], epoch)

        print(f"\n[Epoch {epoch:03d}/{args.epochs}]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | mIoU: {train_metrics['miou']:.4f} | "
              f"Acc: {train_metrics['pixel_accuracy']:.4f} | mDice: {train_metrics['mean_dice']:.4f}")
        print(f"          IoU  [BG/Crop/Weed]: "
              f"[{train_metrics['iou_Background']:.3f}/"
              f"{train_metrics['iou_Crop']:.3f}/"
              f"{train_metrics['iou_Weed']:.3f}]")
        print(f"          Dice [BG/Crop/Weed]: "
              f"[{train_metrics['dice_Background']:.3f}/"
              f"{train_metrics['dice_Crop']:.3f}/"
              f"{train_metrics['dice_Weed']:.3f}]")
        print(f"          F1   [BG/Crop/Weed]: "
              f"[{train_metrics['f1_Background']:.3f}/"
              f"{train_metrics['f1_Crop']:.3f}/"
              f"{train_metrics['f1_Weed']:.3f}]")

        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | mIoU: {val_metrics['miou']:.4f} | "
              f"Acc: {val_metrics['pixel_accuracy']:.4f} | mDice: {val_metrics['mean_dice']:.4f}")
        print(f"          IoU  [BG/Crop/Weed]: "
              f"[{val_metrics['iou_Background']:.3f}/"
              f"{val_metrics['iou_Crop']:.3f}/"
              f"{val_metrics['iou_Weed']:.3f}]")
        print(f"          Dice [BG/Crop/Weed]: "
              f"[{val_metrics['dice_Background']:.3f}/"
              f"{val_metrics['dice_Crop']:.3f}/"
              f"{val_metrics['dice_Weed']:.3f}]")
        print(f"          F1   [BG/Crop/Weed]: "
              f"[{val_metrics['f1_Background']:.3f}/"
              f"{val_metrics['f1_Crop']:.3f}/"
              f"{val_metrics['f1_Weed']:.3f}]")
        print(f"          Prec/Rec: {val_metrics['mean_precision']:.4f}/{val_metrics['mean_recall']:.4f}")

        current_miou = val_metrics['miou']
        if current_miou > best_val_miou:
            best_val_miou = current_miou
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_miou": best_val_miou,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "config": config_dict
            }, best_ckpt_path)
            print(f"  [INFO] ✓ Saved best checkpoint (mIoU: {best_val_miou:.4f})")

        if epoch % 10 == 0:
            latest_ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_miou": best_val_miou,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "config": config_dict
            }, latest_ckpt_path)

        scheduler.step()

    print(f"\n{'='*80}")
    print("[INFO] Running final test evaluation...")
    print(f"{'='*80}\n")

    best_ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, loss_fn, device, num_classes, split="Test")

    print(f"[Test Results]")
    print(f"  Loss: {test_metrics['loss']:.4f} | IoU Loss: {test_metrics['iou_loss']:.4f}")
    print(f"  mIoU: {test_metrics['miou']:.4f} | mDice: {test_metrics['mean_dice']:.4f} | "
          f"Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")
    print(f"  IoU  [BG/Crop/Weed]: "
          f"[{test_metrics['iou_Background']:.4f}/"
          f"{test_metrics['iou_Crop']:.4f}/"
          f"{test_metrics['iou_Weed']:.4f}]")
    print(f"  Dice [BG/Crop/Weed]: "
          f"[{test_metrics['dice_Background']:.4f}/"
          f"{test_metrics['dice_Crop']:.4f}/"
          f"{test_metrics['dice_Weed']:.4f}]")
    print(f"  F1   [BG/Crop/Weed]: "
          f"[{test_metrics['f1_Background']:.4f}/"
          f"{test_metrics['f1_Crop']:.4f}/"
          f"{test_metrics['f1_Weed']:.4f}]")
    print(f"  Precision [BG/Crop/Weed]: "
          f"[{test_metrics['precision_Background']:.4f}/"
          f"{test_metrics['precision_Crop']:.4f}/"
          f"{test_metrics['precision_Weed']:.4f}]")
    print(f"  Recall [BG/Crop/Weed]: "
          f"[{test_metrics['recall_Background']:.4f}/"
          f"{test_metrics['recall_Crop']:.4f}/"
          f"{test_metrics['recall_Weed']:.4f}]")

    for key, val in test_metrics.items():
        writer.add_scalar(f"Final/Test_{key}", val, 0)

    results_path = os.path.join(run_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            **test_metrics,
            "best_val_miou": best_val_miou,
            "model_info": model_info
        }, f, indent=2)

    writer.close()
    print(f"\n{'='*80}")
    print(f"[INFO] Training complete! Results saved to: {run_dir}")
    print(f"[INFO] Best validation mIoU: {best_val_miou:.4f}")
    print(f"[INFO] Test mIoU: {test_metrics['miou']:.4f}")
    print(f"[INFO] Test mean Dice: {test_metrics['mean_dice']:.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()