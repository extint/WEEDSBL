#!/usr/bin/env python3
"""
Training script for SOTA architectures, refactored to be fully compatible with the
existing Weedy Rice segmentation pipeline (train.py + rice_weed_data_loader.py).
Maintains:
- TensorBoard logging
- BCEDiceLoss
- Mixed precision support
- Checkpointing + resume
- RGB or RGB+NIR support
- Agricultural augmentations
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

from rice_loader import create_weedy_rice_rgbnir_dataloaders
from models import create_model, get_model_info

class IoULossMetric(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        b = logits.size(0)
        probs = probs.view(b, -1)
        targets = targets.view(b, -1).float()

        inter = (probs * targets).sum(1)
        union = (probs + targets - probs * targets).sum(1)

        iou = (inter + self.eps) / (union + self.eps)
        return 1 - iou.mean()

# ============================================================
# Loss: BCE + Dice
# ============================================================
class BCEDiceLoss(nn.Module):
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


# ============================================================
# Training Loop
# ============================================================
def train_one_epoch(
    model, loader, optimizer, loss_fn, device, epoch, writer, scaler=None
) -> Dict[str, float]:
    model.train()
    epoch_loss = 0.0
    ious, f1s = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]")

    for batch_idx, batch in enumerate(pbar):
        x = batch["images"].to(device)
        y = batch["labels"].to(device).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        # AMP or FP32
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

        # Metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().squeeze(1)
            y_cpu = y.squeeze(1).long()

            inter = (preds & y_cpu).sum((1, 2)).float()
            union = (preds | y_cpu).sum((1, 2)).float().clamp(1)
            iou = (inter / union).mean().item()
            ious.append(iou)

            tp = (preds & y_cpu).sum((1, 2)).float()
            fp = (preds & (1 - y_cpu)).sum((1, 2)).float()
            fn = ((1 - preds) & y_cpu).sum((1, 2)).float()
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            f1 = (2 * prec * rec / (prec + rec + 1e-6)).mean().item()
            f1s.append(f1)

        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

        if batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            writer.add_scalar("Train/BatchIoU", iou, global_step)

    return {
        "loss": epoch_loss / len(loader),
        "iou": float(np.mean(ious)),
        "f1": float(np.mean(f1s)),
    }

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, split="Val") -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou_loss = 0.0

    iou_loss_fn = IoULossMetric()

    all_tp = all_fp = all_fn = all_tn = 0
    class_iou = {0: [], 1: []}

    pbar = tqdm(loader, desc=f"[{split}]")

    for batch in pbar:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long().unsqueeze(1)

        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item()

        # IoU metric loss
        iou_loss_value = iou_loss_fn(logits, y).item()
        total_iou_loss += iou_loss_value

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        B = preds.size(0)
        preds_f = preds.view(B, -1)
        y_f = y.view(B, -1)

        tp = ((preds_f == 1) & (y_f == 1)).sum().item()
        tn = ((preds_f == 0) & (y_f == 0)).sum().item()
        fp = ((preds_f == 1) & (y_f == 0)).sum().item()
        fn = ((preds_f == 0) & (y_f == 1)).sum().item()

        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_tn += tn

        # Per-class IoU
        for cls in [0, 1]:
            cls_pred = (preds_f == cls)
            cls_gt = (y_f == cls)
            inter = (cls_pred & cls_gt).sum().item()
            union = (cls_pred | cls_gt).sum().item()
            if union > 0:
                class_iou[cls].append(inter / union)

    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall    = all_tp / (all_tp + all_fn + 1e-6)
    f1        = (2 * precision * recall) / (precision + recall + 1e-6)
    pixel_acc = (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn + 1e-6)

    iou_nonweed = float(np.mean(class_iou[0])) if class_iou[0] else 0.0
    iou_weed    = float(np.mean(class_iou[1])) if class_iou[1] else 0.0
    mIoU        = (iou_nonweed + iou_weed) / 2.0

    return {
        "loss": total_loss / len(loader),
        "iou_loss": total_iou_loss / len(loader),     # metric, not training loss
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pixel_accuracy": pixel_acc,
        "iou_nonweed": iou_nonweed,
        "iou_weed": iou_weed,
        "mIoU": mIoU
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser("Train PSPNet for Weedy Rice segmentation")

    parser.add_argument("--model", type=str, default="pspnet",
                        choices=["pspnet","deeplabsv3+","lightsegnet","unet++","unet3+"], help="Use PSPNet")
    parser.add_argument("--base_ch", type=int, default=32)

    parser.add_argument("--data_root", default='/home/vjtiadmin/Desktop/BTechGroup/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB',type=str)
    parser.add_argument("--use_rgbnir", action="store_true")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=1280)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")

    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # Experiment directory setup
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.model}_{timestamp}_nir-{args.use_rgbnir}"

    run_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = create_weedy_rice_rgbnir_dataloaders(
        data_root=args.data_root,
        use_rgbnir=args.use_rgbnir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.height, args.width),
    )

    # Build PSPNet
    in_ch = 4 if args.use_rgbnir else 3
    model = create_model(
        architecture=args.model,
        in_channels=in_ch,
        num_classes=1,
        base_ch=args.base_ch
    ).to(device)

    # Model summary
    info = get_model_info(model)
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Parameters: {info['total_parameters']:,} "
          f"({info['total_parameters_million']:.2f}M)")

    # Dummy input for TensorBoard graph
    # dummy = torch.randn(1, in_ch, args.height, args.width).to(device)
    # writer.add_graph(model, dummy)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = BCEDiceLoss()
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # Resume
    start_epoch = 1
    best_val = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val_iou", 0.0)

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                  device, epoch, writer, scaler)
        val_m = evaluate(model, val_loader, loss_fn, device, "Val")

        writer.add_scalar("Epoch/Train_Loss", train_m["loss"], epoch)
        writer.add_scalar("Epoch/Train_IoU", train_m["iou"], epoch)
        writer.add_scalar("Epoch/Val_Loss", val_m["loss"], epoch)
        writer.add_scalar("Epoch/Val_IoU", val_m["mIoU"], epoch)

        writer.add_scalar("Val/IoU_Loss", val_m["iou_loss"], epoch)
        writer.add_scalar("Val/mIoU", val_m["mIoU"], epoch)
        writer.add_scalar("Val/IoU_Weed", val_m["iou_weed"], epoch)
        writer.add_scalar("Val/IoU_NonWeed", val_m["iou_nonweed"], epoch)

        writer.add_scalar("Val/Precision", val_m["precision"], epoch)
        writer.add_scalar("Val/Recall", val_m["recall"], epoch)
        writer.add_scalar("Val/F1", val_m["f1"], epoch)
        writer.add_scalar("Val/PixelAccuracy", val_m["pixel_accuracy"], epoch)


        print(
    f"[Epoch {epoch}] "
    f"Train: Loss={train_m['loss']:.4f}, IoU={train_m['iou']:.4f}, F1={train_m['f1']:.4f} | "
    f"Val: Loss={val_m['loss']:.4f}, IoU_Loss={val_m['iou_loss']:.4f}, "
    f"mIoU={val_m['mIoU']:.4f}, Weed_IoU={val_m['iou_weed']:.4f}, NonWeed_IoU={val_m['iou_nonweed']:.4f}, "
    f"Precision={val_m['precision']:.4f}, Recall={val_m['recall']:.4f}, "
    f"F1={val_m['f1']:.4f}, PixelAcc={val_m['pixel_accuracy']:.4f}"
)


        # Save best
        if val_m["mIoU"] > best_val:
            best_val = val_m["mIoU"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_iou": best_val
            }, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"[INFO] Saved best model (IoU={best_val:.4f})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_iou": best_val
        }, os.path.join(ckpt_dir, "latest_model.pth"))

        scheduler.step()

    # Test evaluation
    test_m = evaluate(model, test_loader, loss_fn, device, "Test")
    print("[TEST]", test_m)

    with open(os.path.join(run_dir, "test_results.json"), "w") as f:
        json.dump(test_m, f, indent=2)

    writer.close()


if __name__ == "__main__":
    main()
