# distill_train.py
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

# import your existing code (must be in same folder)
from new_main import UNet, BCEDiceLoss, evaluate, LightMANet  # classes & funcs from your file
from rice_weed_data_loader import create_weedy_rice_rgbnir_dataloaders

def load_teacher(teacher_ckpt_path, device, teacher_arch="lightman", in_ch=4):
    """Load teacher model and return it in eval + frozen state."""
    # You used LightMANet in your script; use same architecture for teacher
    # If you originally used a different model, change this accordingly.
    if teacher_arch == "lightman":
        teacher = LightMANet(in_channels=in_ch, num_classes=1, base_ch=32)
    else:
        teacher = UNet(in_channels=in_ch, base_ch=4, out_channels=1)

    teacher = teacher.to(device)

    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}")

    state = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
    # Accept either raw state_dict or saved dict with "model" key
    state_dict = state.get("model", state) if isinstance(state, dict) else state
    teacher.load_state_dict(state_dict["model_state_dict"], strict=True)

    # Freeze weights
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    return teacher

def train_student(
    teacher_ckpt,
    data_root,
    out_dir,
    device="cuda",
    epochs=100,
    batch_size=8,
    lr=1e-4,
    alpha=1.0,
    num_workers=4,
    target_size=(960, 1280),
    teacher_arch="lightman",
    student_arch="unet",
    use_amp=False
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

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

    # ---------------------------
    # Models
    # ---------------------------
    # Teacher: load frozen teacher with 4-ch input
    teacher = load_teacher(teacher_ckpt, device=device, teacher_arch=teacher_arch, in_ch=4)

    # Student: create a RGB-only network (3 channels)
    if student_arch == "unet":
        student = UNet(in_channels=3, base_ch=8, out_channels=1)  # tune base_ch if you want smaller model
    else:
        # Option: use LightMANet in_channels=3
        student = LightMANet(in_channels=3, num_classes=1, base_ch=32)

    student = student.to(device)

    # Optionally: initialize student's encoder from teacher's weights if architectures are compatible.
    # For simplicity we skip automatic weight transfer to avoid shape mismatches.

    # ---------------------------
    # Losses / optimizer
    # ---------------------------
    supervised_loss_fn = BCEDiceLoss(bce_weight=0.5)
    kd_loss_fn = nn.BCEWithLogitsLoss()   # will compare student's logits vs teacher's sigmoid probabilities

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    best_miou = 0.0
    global_step = 0
    print(device)
    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x4 = batch["images"].to(device)     # shape (B,4,H,W) because use_rgbnir=True
            labels = batch["labels"].to(device) # (B,H,W)
            labels = labels.unsqueeze(1).float() # (B,1,H,W)

            # Split: teacher_input = all 4 channels; student_input = first 3 channels
            teacher_x = x4
            student_x = x4[:, :3, :, :].contiguous()

            optimizer.zero_grad(set_to_none=True)

            # 1) teacher forward (frozen) to get soft targets
            with torch.no_grad():
                teacher_logits = teacher(teacher_x)            # (B,1,H,W)
                teacher_probs = torch.sigmoid(teacher_logits)  # (B,1,H,W), values in [0,1]

            # 2) student forward
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    student_logits = student(student_x)  # (B,1,H,W)
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
            num_batches += 1
            global_step += 1

        epoch_loss = epoch_loss / max(1, num_batches)
        # Validation (student on RGB-only validation loader)
        val_metrics = evaluate(student, val_loader_rgb, device)
        val_miou = val_metrics["mIoU"]
        val_f1 = val_metrics["F1"]
        print(f"[Epoch {epoch}] TrainLoss: {epoch_loss:.4f} | Val mIoU: {val_miou:.4f} | Val F1: {val_f1:.4f}")

        # Save best student checkpoint
        if val_miou > best_miou:
            best_miou = val_miou
            ckpt_path = os.path.join(out_dir, f"student_rgb_only_best.pth")
            torch.save({
                "model": student.state_dict(),
                "epoch": epoch,
                "miou": best_miou,
            }, ckpt_path)
            print("Saved best student:", ckpt_path)

    # final test on RGB-only test set
    test_metrics = evaluate(student, test_loader_rgb, device)
    print("FINAL TEST (student on RGB-only):", test_metrics)
    return student

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", required=True, help="Path to pretrained teacher (RGB+NIR) checkpoint")
    parser.add_argument("--data_root", default="/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB", help="Root of WeedyRice dataset (same layout as before)")
    parser.add_argument("--out_dir", default="./distill1_experiments", help="Where to save student checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=1.0, help="KD weight (teacher->student)")
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
        use_amp=args.use_amp
    )