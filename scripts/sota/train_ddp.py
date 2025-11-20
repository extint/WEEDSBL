#!/usr/bin/env python3
"""
Training script for SOTA architectures with Multi-GPU support via DDP
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

# Multi-GPU imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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


def train_one_epoch(
    model, loader, optimizer, loss_fn, device, epoch, writer, scaler=None, rank=0
) -> Dict[str, float]:
    model.train()
    epoch_loss = 0.0
    ious, f1s = [], []

    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]")
    else:
        pbar = loader

    for batch_idx, batch in enumerate(pbar):
        x = batch["images"].to(device)
        y = batch["labels"].to(device).unsqueeze(1)

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

        if rank == 0 and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

        if rank == 0 and batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            writer.add_scalar("Train/BatchIoU", iou, global_step)

    return {
        "loss": epoch_loss / len(loader),
        "iou": float(np.mean(ious)),
        "f1": float(np.mean(f1s)),
    }

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, split="Val", rank=0) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou_loss = 0.0

    iou_loss_fn = IoULossMetric()

    all_tp = all_fp = all_fn = all_tn = 0
    class_iou = {0: [], 1: []}

    if rank == 0:
        pbar = tqdm(loader, desc=f"[{split}]")
    else:
        pbar = loader

    for batch in pbar:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long().unsqueeze(1)

        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item()

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
        "iou_loss": total_iou_loss / len(loader),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pixel_accuracy": pixel_acc,
        "iou_nonweed": iou_nonweed,
        "iou_weed": iou_weed,
        "mIoU": mIoU
    }


def setup_ddp(rank, world_size, gpu_ids):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Map rank to actual GPU ID
    actual_gpu = gpu_ids[rank]
    
    # Set CUDA_VISIBLE_DEVICES to restrict to our selected GPUs
    # This ensures no process accidentally uses GPU 3
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # After setting CUDA_VISIBLE_DEVICES, GPU indices are remapped
    # rank 0 -> gpu_ids[0], rank 1 -> gpu_ids[1], etc.
    # But within the process, they appear as 0, 1, 2, 3
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def train_ddp(rank, world_size, gpu_ids, args):
    """Main training function for each GPU process."""
    setup_ddp(rank, world_size, gpu_ids)
    
    # Get actual GPU ID for this rank
    actual_gpu = gpu_ids[rank]
    
    # Experiment directory setup (only on rank 0)
    if rank == 0:
        if args.exp_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.exp_name = f"{args.model}_{timestamp}_nir-{args.use_rgbnir}_multigpu"

        run_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(run_dir, exist_ok=True)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        log_dir = os.path.join(run_dir, "logs")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        writer = SummaryWriter(log_dir)
    else:
        run_dir = os.path.join(args.output_dir, args.exp_name) if args.exp_name else None
        ckpt_dir = None
        writer = None

    device = torch.device(f"cuda:{actual_gpu}")

    # Create datasets directly (not dataloaders yet)
    from rice_loader import WeedyRiceRGBNIRDataset
    
    train_dataset = WeedyRiceRGBNIRDataset(
        args.data_root, 
        split="train", 
        use_rgbnir=args.use_rgbnir,
        target_size=(args.height, args.width), 
        augment=True, 
        nir_drop_prob=0.0
    )
    val_dataset = WeedyRiceRGBNIRDataset(
        args.data_root, 
        split="val", 
        use_rgbnir=args.use_rgbnir,
        target_size=(args.height, args.width), 
        augment=False, 
        nir_drop_prob=0.0
    )
    test_dataset = WeedyRiceRGBNIRDataset(
        args.data_root, 
        split="test", 
        use_rgbnir=args.use_rgbnir,
        target_size=(args.height, args.width), 
        augment=False, 
        nir_drop_prob=0.0
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders with distributed samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Build model
    in_ch = 4 if args.use_rgbnir else 3
    model = create_model(
        architecture=args.model,
        in_channels=in_ch,
        num_classes=1,
        base_ch=args.base_ch
    ).to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[actual_gpu], find_unused_parameters=False)

    if rank == 0:
        info = get_model_info(model.module)
        print(f"[INFO] Model: {args.model}")
        print(f"[INFO] Parameters: {info['total_parameters']:,} "
              f"({info['total_parameters_million']:.2f}M)")
        print(f"[INFO] Training on {world_size} GPUs: {gpu_ids}")
        print(f"[INFO] GPU Memory: ", end="")
        for i, gid in enumerate(gpu_ids):
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU{gid}={mem:.1f}GB ", end="")
        print()

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = BCEDiceLoss()
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # Resume
    start_epoch = 1
    best_val = 0.0
    if args.resume:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % actual_gpu}
        ckpt = torch.load(args.resume, map_location=map_location)
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val_iou", 0.0)

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        
        train_m = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device, epoch, writer, scaler, rank
        )
        val_m = evaluate(model, val_loader, loss_fn, device, "Val", rank)

        # Only rank 0 logs and saves
        if rank == 0:
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
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_iou": best_val
                }, os.path.join(ckpt_dir, "best_model.pth"))
                print(f"[INFO] Saved best model (IoU={best_val:.4f})")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_iou": best_val
            }, os.path.join(ckpt_dir, "latest_model.pth"))

        scheduler.step()
        
        # Synchronize all processes
        dist.barrier()

    # Test evaluation
    if rank == 0:
        test_m = evaluate(model, test_loader, loss_fn, device, "Test", rank)
        print("[TEST]", test_m)

        with open(os.path.join(run_dir, "test_results.json"), "w") as f:
            json.dump(test_m, f, indent=2)

        writer.close()

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser("Train model with Multi-GPU support")

    parser.add_argument("--model", type=str, default="pspnet",
                        choices=["pspnet","deeplabsv3+","lightsegnet","unet++","unet3+"])
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
    
    # GPU selection
    parser.add_argument("--gpu_ids", type=str, default="0,1,2",
                        help="Comma-separated GPU IDs to use (e.g., '0,1,2')")

    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    world_size = len(gpu_ids)
    
    if world_size < 1:
        print(f"[ERROR] No GPUs specified. Please provide valid GPU IDs.")
        return
    elif world_size == 1:
        print(f"[WARNING] Only 1 GPU specified. Consider using single GPU training for better performance.")
    
    print(f"[INFO] Training with {world_size} GPUs: {gpu_ids}")

    # Launch multi-process training
    mp.spawn(
        train_ddp,
        args=(world_size, gpu_ids, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()