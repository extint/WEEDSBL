#!/usr/bin/env python3
"""
Modular training script for Weedy Rice UAV semantic segmentation
Supports multiple architectures: U-Net, LightMANet, U-Net_SA
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

from rice_weed_data_loader import create_weedy_rice_rgbnir_dataloaders
from models import create_model, get_model_info
from weedutils.rgbnir_preprocessing import RGBNIRPreprocessor


# ======================== Loss Function ========================
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


# ======================== Training & Evaluation ========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn,
    device,
    epoch: int,
    writer: SummaryWriter,
    scaler=None
) -> Dict[str, float]:
    """Train for one epoch with TensorBoard logging"""
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device,
    split: str = "Val"
) -> Dict[str, float]:
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0.0
    ious, f1s, precs, recs = [], [] , [], []
    
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


# ======================== Main Training Loop ========================
def main():
    parser = argparse.ArgumentParser(
        description="Train Weedy Rice Segmentation Model (U-Net or LightMANet)"
    )
    
    # Model architecture
    parser.add_argument("--model", type=str, default="lightmanet",
                        choices=["unet", "unet_sa", "lightmanet"],
                        help="Model architecture to use")
    parser.add_argument("--base_ch", type=int, default=32,
                        help="Base number of channels for the model")
    
    # Data
    parser.add_argument("--data_root", type=str, default="/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB",
                        help="Dataset root with RGB/, Multispectral/, Masks/, Metadata/")
    parser.add_argument("--use_rgbnir", action="store_true",
                        help="Use RGB+NIR (4 channels), otherwise RGB only (3 channels)")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=1280)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    
    # Checkpointing
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment model (auto-generated if not provided)")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Root directory for all experiments")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Create run-specific directory
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        channels = "4ch_RGBNIR" if args.use_rgbnir else "3ch_RGB"
        args.exp_name = f"{args.model}_{timestamp}_nir-{args.use_rgbnir}"
    
    run_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n{'='*60}")
    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Config saved to: {config_path}")
    print(f"{'='*60}\n")
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Dataloaders
    print("[INFO] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_weedy_rice_rgbnir_dataloaders(
        data_root=args.data_root,
        use_rgbnir=args.use_rgbnir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.height, args.width)
    )
    print(f"[INFO] Train samples: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Model
    in_ch = 4 if args.use_rgbnir else 3
    model = create_model(
        architecture=args.model,
        in_channels=in_ch,
        num_classes=1,
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
    loss_fn = BCEDiceLoss(bce_weight=0.5)
    
    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None
    
    # Resume training if checkpoint provided
    start_epoch = 1
    best_val_iou = 0.0
    
    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_iou = checkpoint.get("best_val_iou", 0.0)
        print(f"[INFO] Resumed from epoch {checkpoint['epoch']}, best IoU: {best_val_iou:.4f}")
    
    # Training loop
    print(f"\n[INFO] Starting training for {args.epochs} epochs...\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, writer, scaler
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, device, split="Val")
        
        # Log to TensorBoard
        writer.add_scalar("Epoch/Train_Loss", train_metrics["loss"], epoch)
        writer.add_scalar("Epoch/Train_IoU", train_metrics["iou"], epoch)
        writer.add_scalar("Epoch/Train_F1", train_metrics["f1"], epoch)
        writer.add_scalar("Epoch/Val_Loss", val_metrics["loss"], epoch)
        writer.add_scalar("Epoch/Val_IoU", val_metrics["iou"], epoch)
        writer.add_scalar("Epoch/Val_F1", val_metrics["f1"], epoch)
        writer.add_scalar("Epoch/Val_Precision", val_metrics["prec"], epoch)
        writer.add_scalar("Epoch/Val_Recall", val_metrics["rec"], epoch)
        writer.add_scalar("Epoch/LR", optimizer.param_groups[0]["lr"], epoch)
        
        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"Train Loss: {train_metrics['loss']:.4f} | IoU: {train_metrics['iou']:.4f} | F1: {train_metrics['f1']:.4f} || "
              f"Val Loss: {val_metrics['loss']:.4f} | IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f} | Precision: {val_metrics['prec']:.4f} | Recall: {val_metrics['rec']:.4f}")
        
        # Save best checkpoint
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_iou": best_val_iou,
                "val_iou": val_metrics["iou"],
                "val_f1": val_metrics["f1"],
                "val_prec": val_metrics["prec"],
                "val_rec": val_metrics["rec"],
                "config": vars(args)
            }, best_ckpt_path)
            print(f"[INFO] ✓ Saved best checkpoint: {best_ckpt_path} (IoU: {best_val_iou:.4f})")
        
        # Save latest checkpoint
        latest_ckpt_path = os.path.join(ckpt_dir, "latest_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_iou": best_val_iou,
            "val_iou": val_metrics["iou"],
            "val_prec": val_metrics["prec"],
            "val_rec": val_metrics["rec"],
            "config": vars(args)
        }, latest_ckpt_path)
        
        scheduler.step()
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print("[INFO] Running final test evaluation...")
    print(f"{'='*60}\n")
    
    test_metrics = evaluate(model, test_loader, loss_fn, device, split="Test")
    print(f"[Test Results] Loss: {test_metrics['loss']:.4f} | "
          f"IoU: {test_metrics['iou']:.4f} | F1: {test_metrics['f1']:.4f} | Precision: {test_metrics['prec']:.4f} | Recall: {test_metrics['rec']:.4f}")
    
    writer.add_scalar("Final/Test_IoU", test_metrics["iou"], 0)
    writer.add_scalar("Final/Test_F1", test_metrics["f1"], 0)
    writer.add_scalar("Final/Test_Precision", test_metrics["prec"], 0)
    writer.add_scalar("Final/Test_Recall", test_metrics["rec"], 0)
    
    # Save test results
    results_path = os.path.join(run_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            **test_metrics,
            "best_val_iou": best_val_iou,
            "model_info": model_info
        }, f, indent=2)
    
    writer.close()
    print(f"\n{'='*60}")
    print(f"[INFO] Training complete! Results saved to: {run_dir}")
    print(f"[INFO] Best validation IoU: {best_val_iou:.4f}")
    print(f"[INFO] Test IoU: {test_metrics['iou']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
