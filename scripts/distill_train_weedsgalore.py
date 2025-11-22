#!/usr/bin/env python3

"""
Knowledge Distillation Training for WeedsGalore Multi-Class Segmentation
Teacher: RGB+NIR (4 channels) model
Student: RGB-only (3 channels) model

Features:
- Multi-class distillation (Background, Crop, Weed)
- Comprehensive metrics: mIoU, class-wise IoU, F1, precision, recall, pixel accuracy
- FLOPs and parameter counting
- TensorBoard logging
- Same config/checkpoint structure as train_weedsgalore.py
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import create_model, get_model_info
from weedsgalore_data_loader import create_weedsgalore_dataloaders


# ======================== Loss Functions ========================

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


class DistillationLoss(nn.Module):
    """Knowledge distillation loss for multi-class segmentation"""
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        num_classes: int = 3,
        dice_weight: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.task_loss = MultiClassDiceCELoss(num_classes=num_classes, dice_weight=dice_weight)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_logits: (B, C, H, W) from student model
            teacher_logits: (B, C, H, W) from teacher model
            targets: (B, H, W) ground truth class indices

        Returns:
            Combined distillation loss
        """
        # Task loss (student vs ground truth)
        task_loss = self.task_loss(student_logits, targets)

        # Distillation loss (student vs teacher)
        # Soften the logits with temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL divergence loss
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # Combine losses
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss

        return total_loss


# ======================== Metrics ========================

def compute_multiclass_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3
) -> dict:
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


# ======================== Teacher Model Loading ========================

def load_teacher_model(checkpoint_path: str, device) -> nn.Module:
    """Load pre-trained teacher model from checkpoint"""
    print(f"[INFO] Loading teacher model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract teacher config
    teacher_config = checkpoint.get('config', {})
    teacher_arch = teacher_config.get('model', 'lightmanet')
    teacher_base_ch = teacher_config.get('base_ch', 32)
    teacher_num_classes = teacher_config.get('num_classes', 3)

    print(f"[INFO] Teacher config: {teacher_arch}, in_ch=4 (RGB+NIR), num_classes={teacher_num_classes}")

    # Create teacher model (always 4 channels for RGB+NIR)
    teacher = create_model(
        architecture=teacher_arch,
        in_channels=4,
        num_classes=teacher_num_classes,
        base_ch=teacher_base_ch
    )
    
    # teacher.load_state_dict(checkpoint, strict=False)
    # teacher.load_state_dict(checkpoint['model_state_dict'])
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if not ('total_ops' in k or 'total_params' in k)
    }

    teacher.load_state_dict(filtered_state_dict)

    teacher = teacher.to(device)
    teacher.eval()  # Teacher is always in eval mode

    # Get teacher performance metrics
    teacher_miou = checkpoint.get('best_val_miou', 0.0)
    print(f"[INFO] Teacher validation mIoU: {teacher_miou*100:.2f}%")

    return teacher


# ======================== Training & Evaluation ========================

def train_one_epoch_distillation(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn: DistillationLoss,
    device,
    epoch: int,
    writer: SummaryWriter,
    num_classes: int = 3,
    scaler=None
) -> dict:
    """Train student for one epoch with knowledge distillation"""
    student.train()
    teacher.eval()  # Teacher always in eval mode

    epoch_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Distill Train]")

    for batch_idx, batch in enumerate(pbar):
        x = batch["images"].to(device)  # (B, 4, H, W) RGB+NIR
        y = batch["labels"].to(device).long()  # (B, H, W)

        # Split RGB and NIR channels
        rgb = x[:, :3, :, :]  # Student input (RGB only)

        optimizer.zero_grad(set_to_none=True)

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_logits = teacher(x)  # Use full RGB+NIR

        # Student forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                student_logits = student(rgb)  # RGB only
                loss = loss_fn(student_logits, teacher_logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            student_logits = student(rgb)
            loss = loss_fn(student_logits, teacher_logits, y)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        # Collect predictions for metrics
        with torch.no_grad():
            preds = torch.argmax(student_logits, dim=1)  # (B, H, W)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Log to TensorBoard
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
    device,
    num_classes: int = 3,
    is_teacher: bool = False,
    split: str = "Val"
) -> dict:
    """Evaluate model (student or teacher)"""
    model.eval()
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"[{split}]")

    for batch in pbar:
        x = batch["images"].to(device)  # (B, 4, H, W)
        y = batch["labels"].to(device).long()  # (B, H, W)

        if not is_teacher:
            # Student uses RGB only
            x = x[:, :3, :, :]

        logits = model(x)
        preds = torch.argmax(logits, dim=1)  # (B, H, W)

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_multiclass_metrics(all_preds, all_targets, num_classes)

    return metrics


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for WeedsGalore Multi-Class Segmentation"
    )

    # Teacher
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                       help="Path to pre-trained teacher model checkpoint")

    # Student architecture
    parser.add_argument("--student_model", type=str, default="lightmanet",
                       choices=["unet", "unet_sa", "lightmanet"],
                       help="Student model architecture")
    parser.add_argument("--student_base_ch", type=int, default=32,
                       help="Base channels for student model")

    # Data //                                                   vjti-comp
    parser.add_argument("--data_root", type=str, default="/home/vedantmehra/Downloads/weedsgalore-dataset",
                       help="Path to weedsgalore-dataset folder")
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--width", type=int, default=600)

    # Augmentation
    parser.add_argument("--augment", default=True, action="store_true",
                       help="Enable advanced agricultural augmentations")
    parser.add_argument("--nir_drop_prob", type=float, default=0.0,
                       help="Probability of dropping NIR during training")

    # Distillation
    parser.add_argument("--temperature", type=float, default=4.0,
                       help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Distillation loss weight (1-alpha for task loss)")
    parser.add_argument("--dice_weight", type=float, default=0.5,
                       help="Weight for Dice in task loss")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")

    # Checkpointing
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./distill_experiments_weedsgalore")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume student training from checkpoint")

    args = parser.parse_args()

    num_classes = 3
    class_names = ["Background", "Crop", "Weed"]

    # Create experiment directory
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        aug_str = "aug" if args.augment else "noaug"
        args.exp_name = f"distill_weedsgalore_{args.student_model}_RGB_{aug_str}_{timestamp}"

    run_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    config_dict = {
        **vars(args),
        "num_classes": num_classes,
        "class_names": class_names,
        "dataset": "weedsgalore",
        "mode": "knowledge_distillation",
        "student_input": "RGB (3 channels)",
        "teacher_input": "RGB+NIR (4 channels)"
    }

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n{'='*80}")
    print(f"[INFO] Knowledge Distillation Training - WeedsGalore")
    print(f"[INFO] Teacher: RGB+NIR (4 ch) → Student: RGB only (3 ch)")
    print(f"[INFO] Run directory: {run_dir}")
    print(f"{'='*80}\n")

    writer = SummaryWriter(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load teacher model
    teacher = load_teacher_model(args.teacher_checkpoint, device)

    # Create dataloaders
    print("\n[INFO] Creating dataloaders...")
    train_loader, val_loader = create_weedsgalore_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_rgbnir=True,  # Always load RGB+NIR for teacher
        target_size=(args.height, args.width),
        nir_drop_prob=args.nir_drop_prob,
        augment=args.augment
    )

    print(f"[INFO] Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Create student model (RGB only - 3 channels)
    print("\n[INFO] Creating student model...")
    student = create_model(
        architecture=args.student_model,
        in_channels=3,  # RGB only
        num_classes=num_classes,
        base_ch=args.student_base_ch
    )
    student = student.to(device)

    # Get model info
    student_info = get_model_info(student)
    print(f"[INFO] Student: {student_info['architecture']}")
    print(f"[INFO] Parameters: {student_info['total_parameters']:,} "
          f"({student_info['total_parameters_million']:.2f}M)")

    # Compute FLOPs
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, args.height, args.width).to(device)
        flops, params = profile(student, inputs=(dummy_input,), verbose=False)
        flops_giga = flops / 1e9
        print(f"[INFO] FLOPs: {flops:,} ({flops_giga:.2f} GFLOPs)")
        student_info['flops'] = float(flops)
        student_info['flops_gflops'] = flops_giga
    except ImportError:
        print("[WARNING] thop not installed - FLOPs calculation skipped")
        student_info['flops'] = None
        student_info['flops_gflops'] = None

    # Save model info
    model_info_path = os.path.join(run_dir, "student_model_info.json")
    with open(model_info_path, "w") as f:
        json.dump(student_info, f, indent=2)

    # Optimizer and loss
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    distill_loss = DistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        num_classes=num_classes,
        dice_weight=args.dice_weight
    )

    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None

    # Resume if needed
    start_epoch = 1
    best_student_miou = 0.0

    if args.resume:
        print(f"[INFO] Resuming student from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_student_miou = ckpt.get("best_student_miou", 0.0)

    # Evaluate teacher on validation set
    print("\n[INFO] Evaluating teacher on validation set...")
    teacher_val_metrics = evaluate(teacher, val_loader, device, num_classes, is_teacher=True, split="Teacher Val")
    print(f"[INFO] Teacher Val mIoU: {teacher_val_metrics['miou']*100:.2f}%")

    # Training loop
    print(f"\n[INFO] Starting distillation training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train with distillation
        train_metrics = train_one_epoch_distillation(
            student, teacher, train_loader, optimizer, distill_loss,
            device, epoch, writer, num_classes, scaler
        )

        # Validate student
        val_metrics = evaluate(student, val_loader, device, num_classes, is_teacher=False, split="Student Val")

        # Log to TensorBoard
        for key, val in train_metrics.items():
            writer.add_scalar(f"Epoch/Train_{key}", val, epoch)
        for key, val in val_metrics.items():
            writer.add_scalar(f"Epoch/Val_{key}", val, epoch)
        writer.add_scalar("Epoch/LR", optimizer.param_groups[0]["lr"], epoch)

        # Print summary
        print(f"\n[Epoch {epoch:03d}/{args.epochs}]")
        print(f"{'─'*80}")
        print(f"TRAIN → Loss: {train_metrics['loss']:.4f} | mIoU: {train_metrics['miou']*100:.2f}% | "
              f"PixAcc: {train_metrics['pixel_accuracy']*100:.2f}%")
        print(f"        IoU → BG: {train_metrics['iou_Background']*100:.2f}% | "
              f"Crop: {train_metrics['iou_Crop']*100:.2f}% | Weed: {train_metrics['iou_Weed']*100:.2f}%")
        print(f"VAL   → mIoU: {val_metrics['miou']*100:.2f}% | PixAcc: {val_metrics['pixel_accuracy']*100:.2f}%")
        print(f"        IoU → BG: {val_metrics['iou_Background']*100:.2f}% | "
              f"Crop: {val_metrics['iou_Crop']*100:.2f}% | Weed: {val_metrics['iou_Weed']*100:.2f}%")
        print(f"        F1  → BG: {val_metrics['f1_Background']*100:.2f}% | "
              f"Crop: {val_metrics['f1_Crop']*100:.2f}% | Weed: {val_metrics['f1_Weed']*100:.2f}%")
        print(f"{'─'*80}")

        # Save best checkpoint
        current_miou = val_metrics['miou']
        if current_miou > best_student_miou:
            best_student_miou = current_miou
            best_ckpt = os.path.join(ckpt_dir, "best_student.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_student_miou": best_student_miou,
                "teacher_val_miou": teacher_val_metrics['miou'],
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "config": config_dict,
                "student_info": student_info
            }, best_ckpt)
            print(f"✓ Saved best student: {best_ckpt} (mIoU: {best_student_miou*100:.2f}%)")

        # Save latest
        latest_ckpt = os.path.join(ckpt_dir, "latest_student.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_student_miou": best_student_miou,
            "teacher_val_miou": teacher_val_metrics['miou'],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "config": config_dict,
            "student_info": student_info
        }, latest_ckpt)

        scheduler.step()

    # Final results
    final_results = {
        "best_student_miou": best_student_miou,
        "teacher_val_miou": teacher_val_metrics['miou'],
        "final_val_metrics": {k: v for k, v in val_metrics.items()},
        "student_info": student_info,
        "config": config_dict
    }

    results_path = os.path.join(run_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    writer.close()

    print(f"\n{'='*80}")
    print(f"[INFO] Distillation complete!")
    print(f"[INFO] Teacher Val mIoU: {teacher_val_metrics['miou']*100:.2f}%")
    print(f"[INFO] Best Student Val mIoU: {best_student_miou*100:.2f}%")
    print(f"[INFO] Results saved to: {run_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()