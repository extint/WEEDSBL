#!/usr/bin/env python3
"""
Knowledge distillation training script
Teacher: RGB+NIR (4 channels) model
Student: RGB-only (3 channels) model
"""
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import create_model, get_model_info
from rice_weed_data_loader import create_weedy_rice_rgbnir_dataloaders


# ======================== Loss Function ========================
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, eps=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = bce_weight
        self.eps = eps
    
    def forward(self, logits, targets):
        bce = self.bce(logits, targets.float())
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dims)
        union = probs.sum(dims) + targets.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean()
        return self.w * bce + (1 - self.w) * dice_loss


def load_teacher(teacher_ckpt_path, device, teacher_arch="lightmanet", in_ch=4, base_ch=32):
    """Load teacher model and return it in eval + frozen state"""
    print(f"[INFO] Loading teacher model: {teacher_arch}")
    
    teacher = create_model(
        architecture=teacher_arch,
        in_channels=in_ch,
        num_classes=1,
        base_ch=base_ch
    )
    teacher = teacher.to(device)
    
    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}")
    state = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(state, dict):
        state_dict = state.get("model_state_dict", state.get("model", state))
    else:
        state_dict = state
    
    teacher.load_state_dict(state_dict, strict=True)
    
    # Freeze teacher weights
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    print(f"[INFO] Teacher loaded and frozen")
    return teacher


@torch.no_grad()
def evaluate_student(model, loader, device):
    """Evaluate student model on RGB-only data"""
    model.eval()
    ious, f1s = [], []
    
    for batch in tqdm(loader, desc="[Eval]", leave=False):
        x = batch["images"].to(device)  # (B, 3, H, W)
        y = batch["labels"].to(device).long()  # (B, H, W)
        
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().squeeze(1)
        
        # IoU
        inter = (preds & y).sum(dim=(1, 2)).float()
        union = (preds | y).sum(dim=(1, 2)).float().clamp_min(1)
        iou = (inter / union).mean().item()
        ious.append(iou)
        
        # F1
        tp = (preds & y).sum(dim=(1, 2)).float()
        fp = (preds & (1 - y)).sum(dim=(1, 2)).float()
        fn = ((1 - preds) & y).sum(dim=(1, 2)).float()
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = (2 * prec * rec / (prec + rec + 1e-6)).mean().item()
        f1s.append(f1)
    
    return {"mIoU": sum(ious) / len(ious), "F1": sum(f1s) / len(f1s)}


def train_student(
    teacher_ckpt,
    data_root,
    out_dir,
    device="cuda",
    epochs=50,
    batch_size=8,
    lr=2e-4,
    alpha=1.0,
    num_workers=4,
    target_size=(960, 1280),
    teacher_arch="unet", # change teacher arch here (unet, unet_sa, lightmanet)
    teacher_base_ch=32,
    student_arch="unet", # change student arch here
    student_base_ch=64,
    use_amp=False,
    exp_name=None
):
    """Train student model using knowledge distillation"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Create run-specific directory
    if exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"distill_{student_arch}_from_{teacher_arch}_{timestamp}"
    
    run_dir = os.path.join(out_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)
    
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    config = {
        "teacher_ckpt": teacher_ckpt,
        "teacher_arch": teacher_arch,
        "teacher_base_ch": teacher_base_ch,
        "student_arch": student_arch,
        "student_base_ch": student_base_ch,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "alpha": alpha,
        "target_size": target_size,
        "use_amp": use_amp
    }
    
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"[INFO] Knowledge Distillation Training")
    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Teacher: {teacher_arch} (RGB+NIR)")
    print(f"[INFO] Student: {student_arch} (RGB-only)")
    print(f"{'='*60}\n")
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # ---------------------------
    # Dataloaders
    # ---------------------------
    # Training loader: use_rgbnir=True so each sample gives 4-ch input (RGB+NIR)
    train_loader, _, _ = create_weedy_rice_rgbnir_dataloaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        use_rgbnir=True,
        target_size=target_size
    )
    
    # Validation loader for student: RGB-only
    _, val_loader_rgb, test_loader_rgb = create_weedy_rice_rgbnir_dataloaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        use_rgbnir=False,
        target_size=target_size
    )
    
    print(f"[INFO] Train samples: {len(train_loader.dataset)}")
    print(f"[INFO] Val samples: {len(val_loader_rgb.dataset)}")
    
    # ---------------------------
    # Models
    # ---------------------------
    # Teacher: load frozen teacher with 4-ch input
    teacher = load_teacher(
        teacher_ckpt,
        device=device,
        teacher_arch=teacher_arch,
        in_ch=4,
        base_ch=teacher_base_ch
    )
    
    teacher_info = get_model_info(teacher)
    print(f"[INFO] Teacher parameters: {teacher_info['total_non_trainable_parameters']:,} "
          f"({teacher_info['total_non_trainable_parameters_million']:.2f}M)")

    # Student: create RGB-only network (3 channels)
    student = create_model(
        architecture=student_arch,
        in_channels=3,
        num_classes=1,
        base_ch=student_base_ch
    )
    student = student.to(device)
    
    student_info = get_model_info(student)
    print(f"[INFO] Student parameters: {student_info['total_parameters']:,} "
          f"({student_info['total_parameters_million']:.2f}M)")
    print(f"[INFO] Compression ratio: {teacher_info['total_parameters'] / student_info['total_parameters']:.2f}x\n")
    
    # ---------------------------
    # Losses / optimizer
    # ---------------------------
    supervised_loss_fn = BCEDiceLoss(bce_weight=0.5)
    kd_loss_fn = nn.BCEWithLogitsLoss()  # Compare student logits vs teacher probs
    
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None
    
    best_miou = 0.0
    
    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss = 0.0
        epoch_sup_loss = 0.0
        epoch_kd_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            x4 = batch["images"].to(device)  # (B, 4, H, W)
            labels = batch["labels"].to(device).unsqueeze(1).float()  # (B, 1, H, W)
            
            # Split channels: teacher gets all 4, student gets first 3
            teacher_x = x4
            student_x = x4[:, :3, :, :].contiguous()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Teacher forward (frozen) to get soft targets
            with torch.no_grad():
                teacher_logits = teacher(teacher_x)
                teacher_probs = torch.sigmoid(teacher_logits)
            
            # Student forward
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    student_logits = student(student_x)
                    sup_loss = supervised_loss_fn(student_logits, labels)
                    kd_loss = kd_loss_fn(student_logits, teacher_probs)
                    loss = sup_loss + alpha * kd_loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                student_logits = student(student_x)
                sup_loss = supervised_loss_fn(student_logits, labels)
                kd_loss = kd_loss_fn(student_logits, teacher_probs)
                loss = sup_loss + alpha * kd_loss
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            epoch_sup_loss += sup_loss.item()
            epoch_kd_loss += kd_loss.item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sup": f"{sup_loss.item():.4f}",
                "kd": f"{kd_loss.item():.4f}"
            })
            
            # Log to TensorBoard
            if batch_idx % 10 == 0:
                global_step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
                writer.add_scalar("Train/SupervisedLoss", sup_loss.item(), global_step)
                writer.add_scalar("Train/KDLoss", kd_loss.item(), global_step)
        
        epoch_loss /= len(train_loader)
        epoch_sup_loss /= len(train_loader)
        epoch_kd_loss /= len(train_loader)
        
        # Validation
        val_metrics = evaluate_student(student, val_loader_rgb, device)
        val_miou = val_metrics["mIoU"]
        val_f1 = val_metrics["F1"]
        
        # Log to TensorBoard
        writer.add_scalar("Epoch/Train_Loss", epoch_loss, epoch)
        writer.add_scalar("Epoch/Train_SupervisedLoss", epoch_sup_loss, epoch)
        writer.add_scalar("Epoch/Train_KDLoss", epoch_kd_loss, epoch)
        writer.add_scalar("Epoch/Val_mIoU", val_miou, epoch)
        writer.add_scalar("Epoch/Val_F1", val_f1, epoch)
        writer.add_scalar("Epoch/LR", optimizer.param_groups[0]["lr"], epoch)
        
        print(f"[Epoch {epoch:03d}/{epochs}] "
              f"Loss: {epoch_loss:.4f} (Sup: {epoch_sup_loss:.4f}, KD: {epoch_kd_loss:.4f}) || "
              f"Val mIoU: {val_miou:.4f} | F1: {val_f1:.4f}")
        
        # Save best checkpoint
        if val_miou > best_miou:
            best_miou = val_miou
            ckpt_path = os.path.join(run_dir, "student_rgb_only_best.pth")
            torch.save({
                "model": student.state_dict(),
                "model_state_dict": student.state_dict(),  # For compatibility
                "epoch": epoch,
                "miou": best_miou,
                "f1": val_f1,
                "config": config
            }, ckpt_path)
            print(f"[INFO] ✓ Saved best student: {ckpt_path}")
        
        # Save latest
        latest_path = os.path.join(run_dir, "student_rgb_only_latest.pth")
        torch.save({
            "model": student.state_dict(),
            "model_state_dict": student.state_dict(),
            "epoch": epoch,
            "miou": val_miou,
            "config": config
        }, latest_path)
        
        scheduler.step()
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print("[INFO] Running final test evaluation...")
    test_metrics = evaluate_student(student, test_loader_rgb, device)
    print(f"[Test Results] mIoU: {test_metrics['mIoU']:.4f} | F1: {test_metrics['F1']:.4f}")
    print(f"[INFO] Best validation mIoU: {best_miou:.4f}")
    print(f"{'='*60}\n")
    
    writer.add_scalar("Final/Test_mIoU", test_metrics["mIoU"], 0)
    writer.add_scalar("Final/Test_F1", test_metrics["F1"], 0)
    
    # Save final results
    results_path = os.path.join(run_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            **test_metrics,
            "best_val_miou": best_miou,
            "student_info": student_info,
            "teacher_info": teacher_info,
            "compression_ratio": teacher_info['total_parameters'] / student_info['total_parameters']
        }, f, indent=2)
    
    writer.close()
    print(f"[INFO] Training complete! Results saved to: {run_dir}\n")
    
    return student


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    
    # Required
    parser.add_argument("--teacher_ckpt", required=True,
                        help="Path to pretrained teacher (RGB+NIR) checkpoint")
    parser.add_argument("--data_root", default="/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB",
                        help="Root of WeedyRice dataset")
    
    # Architecture
    parser.add_argument("--teacher_arch", default="lightmanet",
                        choices=["unet", "lightmanet"])
    parser.add_argument("--teacher_base_ch", type=int, default=32)
    parser.add_argument("--student_arch", default="unet",
                        choices=["unet", "lightmanet"])
    parser.add_argument("--student_base_ch", type=int, default=16)
    
    # Training
    parser.add_argument("--out_dir", default="./distill1_experiments",
                        help="Where to save student checkpoints")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="KD loss weight (teacher->student)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_amp", action="store_true")
    
    args = parser.parse_args()
    
    train_student(
        teacher_ckpt=args.teacher_ckpt,
        data_root=args.data_root,
        out_dir=args.out_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        num_workers=args.num_workers,
        teacher_arch=args.teacher_arch,
        teacher_base_ch=args.teacher_base_ch,
        student_arch=args.student_arch,
        student_base_ch=args.student_base_ch,
        use_amp=args.use_amp,
        exp_name=args.exp_name
    )
