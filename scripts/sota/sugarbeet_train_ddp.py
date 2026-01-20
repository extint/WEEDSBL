#!/usr/bin/env python3

"""
Multi-GPU Training script for Sugar Beets weed segmentation using DDP
Features:
  - Efficient DistributedDataParallel training
  - Dataset loaded ONCE at start (not per epoch)
  - Synchronized metrics across GPUs
  - Mixed precision training
  - Explicit GPU selection (0, 1, 2 only)
  - Detailed per-class metrics logging
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import dataset loader
from sugarbeets_data_loader import create_sugarbeets_dataloaders, SugarBeetDataset
from torch.utils.data import Subset

from models import create_model, get_model_info


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


# ======================== Helper Functions ========================

def setup_ddp(rank, world_size, gpu_ids):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Set visible devices to only the GPUs we want to use
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def reduce_dict(metrics: Dict[str, float], world_size: int) -> Dict[str, float]:
    """Reduce metrics across all processes"""
    reduced_metrics = {}
    for k, v in metrics.items():
        tensor = torch.tensor(v, device='cuda')
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced_metrics[k] = (tensor / world_size).item()
    return reduced_metrics


def print_metrics(metrics: Dict[str, float], split: str = "Train", epoch: int = None):
    """Print detailed metrics in a formatted way"""
    class_names = ['Background', 'Crop', 'Weed']
    
    if epoch is not None:
        print(f"\n{'='*80}")
        print(f"Epoch {epoch:03d} - {split} Metrics")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"{split} Metrics")
        print(f"{'='*80}")
    
    # Overall metrics
    print(f"\n{'Overall Metrics':^80}")
    print(f"{'-'*80}")
    print(f"  Loss                : {metrics['loss']:.6f}")
    print(f"  Pixel Accuracy      : {metrics['pixel_accuracy']:.4f} ({metrics['pixel_accuracy']*100:.2f}%)")
    print(f"  Mean IoU            : {metrics['miou']:.4f}")
    print(f"  Mean Precision      : {metrics['mean_precision']:.4f}")
    print(f"  Mean Recall         : {metrics['mean_recall']:.4f}")
    print(f"  Mean F1-Score       : {metrics['mean_f1']:.4f}")
    
    # Per-class metrics
    print(f"\n{'Per-Class Metrics':^80}")
    print(f"{'-'*80}")
    print(f"{'Class':<15} {'IoU':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*80}")
    
    for cls_name in class_names:
        iou = metrics[f'iou_{cls_name}']
        prec = metrics[f'precision_{cls_name}']
        rec = metrics[f'recall_{cls_name}']
        f1 = metrics[f'f1_{cls_name}']
        print(f"{cls_name:<15} {iou:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
    
    print(f"{'='*80}\n")


# ======================== Dataset Creation (FIXED SPLIT) ========================

def create_fixed_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    use_rgbnir: bool,
    target_size: tuple,
    nir_drop_prob: float,
    seed: int,
    rank: int,
    world_size: int
):
    """
    Create dataloaders with FIXED splits (computed once, not per epoch)
    Uses DistributedSampler for multi-GPU training
    """
    # Create full dataset for train (with augmentation)
    train_base_ds = SugarBeetDataset(
        root=data_root,
        split="train",
        use_rgbnir=use_rgbnir,
        target_size=target_size,
        augment=True,
        nir_drop_prob=nir_drop_prob
    )
    
    # Create base datasets for val/test (no augmentation)
    val_base_ds = SugarBeetDataset(
        root=data_root,
        split="val",
        use_rgbnir=use_rgbnir,
        target_size=target_size,
        augment=False
    )
    
    test_base_ds = SugarBeetDataset(
        root=data_root,
        split="test",
        use_rgbnir=use_rgbnir,
        target_size=target_size,
        augment=False
    )
    
    # Compute FIXED split indices
    n = len(train_base_ds)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    n_train = int(0.9 * n)
    n_val = int(0.05 * n)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create subsets
    train_ds = Subset(train_base_ds, train_idx)
    val_ds = Subset(val_base_ds, val_idx)
    test_ds = Subset(test_base_ds, test_idx)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed
    )
    
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    test_sampler = DistributedSampler(
        test_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Dataset split (FIXED):")
        print(f"  Train: {len(train_ds)}")
        print(f"  Val  : {len(val_ds)}")
        print(f"  Test : {len(test_ds)}")
    
    return train_loader, val_loader, test_loader, train_sampler


# ======================== Training & Evaluation ========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn,
    device,
    epoch: int,
    writer: SummaryWriter,
    num_classes: int,
    scaler,
    rank: int,
    world_size: int
) -> Dict[str, float]:
    """Train for one epoch with DDP"""
    model.train()
    epoch_loss = 0.0
    
    # Accumulators
    total_correct = 0
    total_pixels = 0
    class_tp = [0, 0, 0]
    class_fp = [0, 0, 0]
    class_fn = [0, 0, 0]
    class_intersection = [0, 0, 0]
    class_union = [0, 0, 0]

    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]")
    else:
        pbar = loader

    for batch_idx, batch in enumerate(pbar):
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            logits = model(x)
            loss = loss_fn(logits, y)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        # Compute metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct = (preds == y).sum().item()
            total_correct += correct
            total_pixels += y.numel()
            
            for cls in range(num_classes):
                pred_mask = (preds == cls)
                target_mask = (y == cls)
                
                class_tp[cls] += (pred_mask & target_mask).sum().item()
                class_fp[cls] += (pred_mask & ~target_mask).sum().item()
                class_fn[cls] += (~pred_mask & target_mask).sum().item()
                class_intersection[cls] += (pred_mask & target_mask).sum().item()
                class_union[cls] += (pred_mask | target_mask).sum().item()

        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute local metrics
    local_metrics = {
        'loss': epoch_loss / len(loader),
        'pixel_accuracy': total_correct / total_pixels,
    }
    
    class_names = ['Background', 'Crop', 'Weed']
    for cls in range(num_classes):
        local_metrics[f'iou_{class_names[cls]}'] = class_intersection[cls] / (class_union[cls] + 1e-6)
        prec = class_tp[cls] / (class_tp[cls] + class_fp[cls] + 1e-6)
        rec = class_tp[cls] / (class_tp[cls] + class_fn[cls] + 1e-6)
        local_metrics[f'precision_{class_names[cls]}'] = prec
        local_metrics[f'recall_{class_names[cls]}'] = rec
        local_metrics[f'f1_{class_names[cls]}'] = 2 * prec * rec / (prec + rec + 1e-6)
    
    # Reduce metrics across all GPUs
    metrics = reduce_dict(local_metrics, world_size)
    
    # Compute mean metrics
    metrics['miou'] = np.mean([metrics[f'iou_{cls}'] for cls in class_names])
    metrics['mean_precision'] = np.mean([metrics[f'precision_{cls}'] for cls in class_names])
    metrics['mean_recall'] = np.mean([metrics[f'recall_{cls}'] for cls in class_names])
    metrics['mean_f1'] = np.mean([metrics[f'f1_{cls}'] for cls in class_names])

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device,
    num_classes: int,
    rank: int,
    world_size: int,
    split: str = "Val"
) -> Dict[str, float]:
    """Evaluate model with DDP"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    class_tp = [0, 0, 0]
    class_fp = [0, 0, 0]
    class_fn = [0, 0, 0]
    class_intersection = [0, 0, 0]
    class_union = [0, 0, 0]
    
    if rank == 0:
        pbar = tqdm(loader, desc=f"[{split}]")
    else:
        pbar = loader

    for batch in pbar:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()

        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total_correct += correct
        total_pixels += y.numel()
        
        for cls in range(num_classes):
            pred_mask = (preds == cls)
            target_mask = (y == cls)
            
            class_tp[cls] += (pred_mask & target_mask).sum().item()
            class_fp[cls] += (pred_mask & ~target_mask).sum().item()
            class_fn[cls] += (~pred_mask & target_mask).sum().item()
            class_intersection[cls] += (pred_mask & target_mask).sum().item()
            class_union[cls] += (pred_mask | target_mask).sum().item()

    # Compute local metrics
    local_metrics = {
        'loss': total_loss / len(loader),
        'pixel_accuracy': total_correct / total_pixels,
    }
    
    class_names = ['Background', 'Crop', 'Weed']
    for cls in range(num_classes):
        local_metrics[f'iou_{class_names[cls]}'] = class_intersection[cls] / (class_union[cls] + 1e-6)
        prec = class_tp[cls] / (class_tp[cls] + class_fp[cls] + 1e-6)
        rec = class_tp[cls] / (class_tp[cls] + class_fn[cls] + 1e-6)
        local_metrics[f'precision_{class_names[cls]}'] = prec
        local_metrics[f'recall_{class_names[cls]}'] = rec
        local_metrics[f'f1_{class_names[cls]}'] = 2 * prec * rec / (prec + rec + 1e-6)
    
    # Reduce metrics across all GPUs
    metrics = reduce_dict(local_metrics, world_size)
    
    # Compute mean metrics
    metrics['miou'] = np.mean([metrics[f'iou_{cls}'] for cls in class_names])
    metrics['mean_precision'] = np.mean([metrics[f'precision_{cls}'] for cls in class_names])
    metrics['mean_recall'] = np.mean([metrics[f'recall_{cls}'] for cls in class_names])
    metrics['mean_f1'] = np.mean([metrics[f'f1_{cls}'] for cls in class_names])

    return metrics


# ======================== Main Training Function ========================

def train_worker(rank, world_size, gpu_ids, args):
    """Main training function for each GPU process"""
    setup_ddp(rank, world_size, gpu_ids)
    
    num_classes = 3
    class_names = ["background", "crop", "weed"]
    
    # Only rank 0 handles logging and checkpointing
    if rank == 0:
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
        config_dict = {**vars(args), "num_classes": num_classes, "class_names": class_names, "gpu_ids": gpu_ids}
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"\n{'='*80}")
        print(f"[INFO] Multi-GPU Training with {world_size} GPUs (IDs: {gpu_ids})")
        print(f"[INFO] Dataset: Sugar Beets ({num_classes} classes)")
        print(f"[INFO] Run directory: {run_dir}")
        print(f"[INFO] TensorBoard logs: {log_dir}")
        print(f"[INFO] To view logs, run: tensorboard --logdir={log_dir}")
        print(f"{'='*80}\n")

        writer = SummaryWriter(log_dir)
    else:
        writer = None
    
    # Create dataloaders ONCE (not per epoch!)
    if rank == 0:
        print("[INFO] Creating fixed dataloaders (computed once)...")
    
    train_loader, val_loader, test_loader, train_sampler = create_fixed_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_rgbnir=args.use_rgbnir,
        target_size=(args.height, args.width),
        nir_drop_prob=args.nir_drop,
        seed=42,
        rank=rank,
        world_size=world_size
    )
    
    # Model - use local rank since CUDA_VISIBLE_DEVICES is set
    device = torch.device(f"cuda:{rank}")
    in_ch = 4 if args.use_rgbnir else 3
    model = create_model(
        architecture=args.model,
        in_channels=in_ch,
        num_classes=num_classes,
        base_ch=args.base_ch
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if rank == 0:
        model_info = get_model_info(model.module)
        print(f"[INFO] Model: {model_info['architecture']}")
        print(f"[INFO] Parameters: {model_info['total_parameters']:,} "
              f"({model_info['total_parameters_million']:.2f}M)")
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = MultiClassDiceLoss(num_classes=num_classes)
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    best_val_miou = 0.0
    start_epoch = 1
    
    # Training loop
    if rank == 0:
        print(f"\n[INFO] Starting training for {args.epochs} epochs...\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Set epoch for distributed sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            epoch, writer, num_classes, scaler, rank, world_size
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, loss_fn, device, num_classes,
            rank, world_size, split="Val"
        )
        
        # Only rank 0 logs and saves
        if rank == 0:
            # Log all metrics to TensorBoard
            # Overall metrics
            writer.add_scalar("Train/Loss", train_metrics['loss'], epoch)
            writer.add_scalar("Train/Pixel_Accuracy", train_metrics['pixel_accuracy'], epoch)
            writer.add_scalar("Train/mIoU", train_metrics['miou'], epoch)
            writer.add_scalar("Train/Mean_Precision", train_metrics['mean_precision'], epoch)
            writer.add_scalar("Train/Mean_Recall", train_metrics['mean_recall'], epoch)
            writer.add_scalar("Train/Mean_F1", train_metrics['mean_f1'], epoch)
            
            writer.add_scalar("Val/Loss", val_metrics['loss'], epoch)
            writer.add_scalar("Val/Pixel_Accuracy", val_metrics['pixel_accuracy'], epoch)
            writer.add_scalar("Val/mIoU", val_metrics['miou'], epoch)
            writer.add_scalar("Val/Mean_Precision", val_metrics['mean_precision'], epoch)
            writer.add_scalar("Val/Mean_Recall", val_metrics['mean_recall'], epoch)
            writer.add_scalar("Val/Mean_F1", val_metrics['mean_f1'], epoch)
            
            # Per-class metrics
            for cls_name in ['Background', 'Crop', 'Weed']:
                writer.add_scalar(f"Train_IoU/{cls_name}", train_metrics[f'iou_{cls_name}'], epoch)
                writer.add_scalar(f"Train_Precision/{cls_name}", train_metrics[f'precision_{cls_name}'], epoch)
                writer.add_scalar(f"Train_Recall/{cls_name}", train_metrics[f'recall_{cls_name}'], epoch)
                writer.add_scalar(f"Train_F1/{cls_name}", train_metrics[f'f1_{cls_name}'], epoch)
                
                writer.add_scalar(f"Val_IoU/{cls_name}", val_metrics[f'iou_{cls_name}'], epoch)
                writer.add_scalar(f"Val_Precision/{cls_name}", val_metrics[f'precision_{cls_name}'], epoch)
                writer.add_scalar(f"Val_Recall/{cls_name}", val_metrics[f'recall_{cls_name}'], epoch)
                writer.add_scalar(f"Val_F1/{cls_name}", val_metrics[f'f1_{cls_name}'], epoch)
            
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
            
            # Print detailed metrics
            print_metrics(train_metrics, split="Train", epoch=epoch)
            print_metrics(val_metrics, split="Validation", epoch=epoch)
            
            # Save best checkpoint
            current_miou = val_metrics['miou']
            if current_miou > best_val_miou:
                best_val_miou = current_miou
                best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_miou": best_val_miou,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }, best_ckpt_path)
                print(f"{'='*80}")
                print(f"[INFO] ✓ NEW BEST MODEL - Saved checkpoint (mIoU: {best_val_miou:.4f})")
                print(f"{'='*80}\n")
        
        scheduler.step()
    
    # Final test evaluation
    try:
        if rank == 0:
            print(f"\n{'='*80}")
            print("[INFO] Running final test evaluation...")
            print(f"{'='*80}")
            
            best_ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pth"))
            model.module.load_state_dict(best_ckpt["model_state_dict"])
        
        # Synchronize before test
        dist.barrier()
        
        test_metrics = evaluate(
            model, test_loader, loss_fn, device, num_classes,
            rank, world_size, split="Test"
        )
        
        if rank == 0:
            # Log test metrics to TensorBoard
            writer.add_scalar("Test/Loss", test_metrics['loss'], args.epochs)
            writer.add_scalar("Test/Pixel_Accuracy", test_metrics['pixel_accuracy'], args.epochs)
            writer.add_scalar("Test/mIoU", test_metrics['miou'], args.epochs)
            writer.add_scalar("Test/Mean_Precision", test_metrics['mean_precision'], args.epochs)
            writer.add_scalar("Test/Mean_Recall", test_metrics['mean_recall'], args.epochs)
            writer.add_scalar("Test/Mean_F1", test_metrics['mean_f1'], args.epochs)
            
            for cls_name in ['Background', 'Crop', 'Weed']:
                writer.add_scalar(f"Test_IoU/{cls_name}", test_metrics[f'iou_{cls_name}'], args.epochs)
                writer.add_scalar(f"Test_Precision/{cls_name}", test_metrics[f'precision_{cls_name}'], args.epochs)
                writer.add_scalar(f"Test_Recall/{cls_name}", test_metrics[f'recall_{cls_name}'], args.epochs)
                writer.add_scalar(f"Test_F1/{cls_name}", test_metrics[f'f1_{cls_name}'], args.epochs)
            
            # Print detailed test metrics
            print_metrics(test_metrics, split="Test (Best Model)")
            
            results_path = os.path.join(run_dir, "test_results.json")
            with open(results_path, "w") as f:
                json.dump(test_metrics, f, indent=2)
            
            # Flush and close writer
            writer.flush()
            writer.close()
            
            print(f"{'='*80}")
            print(f"[INFO] Training complete!")
            print(f"[INFO] Results saved to: {run_dir}")
            print(f"[INFO] To view TensorBoard logs, run:")
            print(f"       tensorboard --logdir={log_dir}")
            print(f"{'='*80}\n")
        
        # Final synchronization before cleanup
        dist.barrier()
        
    finally:
        # Ensure cleanup happens even if there's an error
        cleanup_ddp()


# ======================== Entry Point ========================

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default="deeplabsv3+",
                        choices=["deeplabsv3+"])
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--data_root", type=str,
                        default="/home/vjtiadmin/Desktop/BTechGroup/FINAL_SUGARBEETS_DATASET")
    parser.add_argument("--use_rgbnir", action="store_true")
    parser.add_argument("--height", type=int, default=966)
    parser.add_argument("--width", type=int, default=1296)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--nir_drop", type=float, default=0.0)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--gpu_ids", type=str, default="1,2",
                        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')")
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    world_size = len(gpu_ids)
    
    print(f"[INFO] Using GPUs: {gpu_ids}")
    print(f"[INFO] World size: {world_size}")
    
    if world_size < 1:
        raise ValueError("Must specify at least one GPU")
    
    if world_size == 1:
        print("[INFO] Running on single GPU")
        train_worker(0, 1, gpu_ids, args)
    else:
        torch.multiprocessing.spawn(
            train_worker,
            args=(world_size, gpu_ids, args),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()

# """
# Multi-GPU Training script for Sugar Beets weed segmentation using DDP
# Features:
#   - Efficient DistributedDataParallel training
#   - Dataset loaded ONCE at start (not per epoch)
#   - Synchronized metrics across GPUs
#   - Mixed precision training
#   - Explicit GPU selection (0, 1, 2 only)
#   - Detailed per-class metrics logging
# """

# import argparse
# import os
# import json
# from datetime import datetime
# from typing import Dict
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm

# # Import dataset loader
# from sugarbeets_data_loader import create_sugarbeets_dataloaders, SugarBeetDataset
# from torch.utils.data import Subset

# from models import create_model, get_model_info


# # ======================== Loss Functions ========================

# class MultiClassDiceLoss(nn.Module):
#     """Multi-class segmentation loss with Cross Entropy + Dice"""
#     def __init__(self, num_classes: int = 3, eps: float = 1e-6):
#         super().__init__()
#         self.num_classes = num_classes
#         self.eps = eps
#         self.ce = nn.CrossEntropyLoss()

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         ce_loss = self.ce(logits, targets)
#         probs = torch.softmax(logits, dim=1)
#         targets_one_hot = torch.nn.functional.one_hot(
#             targets, num_classes=self.num_classes
#         ).permute(0, 3, 1, 2).float()

#         dims = (0, 2, 3)
#         intersection = (probs * targets_one_hot).sum(dims)
#         union = probs.sum(dims) + targets_one_hot.sum(dims)
#         dice = (2 * intersection + self.eps) / (union + self.eps)
#         dice_loss = 1 - dice.mean()

#         return 0.5 * ce_loss + 0.5 * dice_loss


# # ======================== Helper Functions ========================

# def setup_ddp(rank, world_size, gpu_ids):
#     """Initialize distributed training"""
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
    
#     # Set visible devices to only the GPUs we want to use
#     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)


# def cleanup_ddp():
#     """Cleanup distributed training"""
#     dist.destroy_process_group()


# def reduce_dict(metrics: Dict[str, float], world_size: int) -> Dict[str, float]:
#     """Reduce metrics across all processes"""
#     reduced_metrics = {}
#     for k, v in metrics.items():
#         tensor = torch.tensor(v, device='cuda')
#         dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
#         reduced_metrics[k] = (tensor / world_size).item()
#     return reduced_metrics


# def print_metrics(metrics: Dict[str, float], split: str = "Train", epoch: int = None):
#     """Print detailed metrics in a formatted way"""
#     class_names = ['Background', 'Crop', 'Weed']
    
#     if epoch is not None:
#         print(f"\n{'='*80}")
#         print(f"Epoch {epoch:03d} - {split} Metrics")
#         print(f"{'='*80}")
#     else:
#         print(f"\n{'='*80}")
#         print(f"{split} Metrics")
#         print(f"{'='*80}")
    
#     # Overall metrics
#     print(f"\n{'Overall Metrics':^80}")
#     print(f"{'-'*80}")
#     print(f"  Loss                : {metrics['loss']:.6f}")
#     print(f"  Pixel Accuracy      : {metrics['pixel_accuracy']:.4f} ({metrics['pixel_accuracy']*100:.2f}%)")
#     print(f"  Mean IoU            : {metrics['miou']:.4f}")
#     print(f"  Mean Precision      : {metrics['mean_precision']:.4f}")
#     print(f"  Mean Recall         : {metrics['mean_recall']:.4f}")
#     print(f"  Mean F1-Score       : {metrics['mean_f1']:.4f}")
    
#     # Per-class metrics
#     print(f"\n{'Per-Class Metrics':^80}")
#     print(f"{'-'*80}")
#     print(f"{'Class':<15} {'IoU':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
#     print(f"{'-'*80}")
    
#     for cls_name in class_names:
#         iou = metrics[f'iou_{cls_name}']
#         prec = metrics[f'precision_{cls_name}']
#         rec = metrics[f'recall_{cls_name}']
#         f1 = metrics[f'f1_{cls_name}']
#         print(f"{cls_name:<15} {iou:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
    
#     print(f"{'='*80}\n")


# # ======================== Dataset Creation (FIXED SPLIT) ========================

# def create_fixed_dataloaders(
#     data_root: str,
#     batch_size: int,
#     num_workers: int,
#     use_rgbnir: bool,
#     target_size: tuple,
#     nir_drop_prob: float,
#     seed: int,
#     rank: int,
#     world_size: int
# ):
#     """
#     Create dataloaders with FIXED splits (computed once, not per epoch)
#     Uses DistributedSampler for multi-GPU training
#     """
#     # Create full dataset for train (with augmentation)
#     train_base_ds = SugarBeetDataset(
#         root=data_root,
#         split="train",
#         use_rgbnir=use_rgbnir,
#         target_size=target_size,
#         augment=True,
#         nir_drop_prob=nir_drop_prob
#     )
    
#     # Create base datasets for val/test (no augmentation)
#     val_base_ds = SugarBeetDataset(
#         root=data_root,
#         split="val",
#         use_rgbnir=use_rgbnir,
#         target_size=target_size,
#         augment=False
#     )
    
#     test_base_ds = SugarBeetDataset(
#         root=data_root,
#         split="test",
#         use_rgbnir=use_rgbnir,
#         target_size=target_size,
#         augment=False
#     )
    
#     # Compute FIXED split indices
#     n = len(train_base_ds)
#     indices = np.arange(n)
#     rng = np.random.default_rng(seed)
#     rng.shuffle(indices)
    
#     n_train = int(0.9 * n)
#     n_val = int(0.05 * n)
    
#     train_idx = indices[:n_train]
#     val_idx = indices[n_train:n_train + n_val]
#     test_idx = indices[n_train + n_val:]
    
#     # Create subsets
#     train_ds = Subset(train_base_ds, train_idx)
#     val_ds = Subset(val_base_ds, val_idx)
#     test_ds = Subset(test_base_ds, test_idx)
    
#     # Create distributed samplers
#     train_sampler = DistributedSampler(
#         train_ds,
#         num_replicas=world_size,
#         rank=rank,
#         shuffle=True,
#         seed=seed
#     )
    
#     val_sampler = DistributedSampler(
#         val_ds,
#         num_replicas=world_size,
#         rank=rank,
#         shuffle=False
#     )
    
#     test_sampler = DistributedSampler(
#         test_ds,
#         num_replicas=world_size,
#         rank=rank,
#         shuffle=False
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         sampler=train_sampler,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True
#     )
    
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         sampler=val_sampler,
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=1,
#         sampler=test_sampler,
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     if rank == 0:
#         print(f"Dataset split (FIXED):")
#         print(f"  Train: {len(train_ds)}")
#         print(f"  Val  : {len(val_ds)}")
#         print(f"  Test : {len(test_ds)}")
    
#     return train_loader, val_loader, test_loader, train_sampler


# # ======================== Training & Evaluation ========================

# def train_one_epoch(
#     model: nn.Module,
#     loader: DataLoader,
#     optimizer,
#     loss_fn,
#     device,
#     epoch: int,
#     writer: SummaryWriter,
#     num_classes: int,
#     scaler,
#     rank: int,
#     world_size: int
# ) -> Dict[str, float]:
#     """Train for one epoch with DDP"""
#     model.train()
#     epoch_loss = 0.0
    
#     # Accumulators
#     total_correct = 0
#     total_pixels = 0
#     class_tp = [0, 0, 0]
#     class_fp = [0, 0, 0]
#     class_fn = [0, 0, 0]
#     class_intersection = [0, 0, 0]
#     class_union = [0, 0, 0]

#     if rank == 0:
#         pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]")
#     else:
#         pbar = loader

#     for batch_idx, batch in enumerate(pbar):
#         x = batch["images"].to(device)
#         y = batch["labels"].to(device).long()

#         optimizer.zero_grad(set_to_none=True)

#         with torch.amp.autocast('cuda', enabled=(scaler is not None)):
#             logits = model(x)
#             loss = loss_fn(logits, y)
        
#         if scaler is not None:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             optimizer.step()

#         epoch_loss += loss.item()

#         # Compute metrics
#         with torch.no_grad():
#             preds = torch.argmax(logits, dim=1)
#             correct = (preds == y).sum().item()
#             total_correct += correct
#             total_pixels += y.numel()
            
#             for cls in range(num_classes):
#                 pred_mask = (preds == cls)
#                 target_mask = (y == cls)
                
#                 class_tp[cls] += (pred_mask & target_mask).sum().item()
#                 class_fp[cls] += (pred_mask & ~target_mask).sum().item()
#                 class_fn[cls] += (~pred_mask & target_mask).sum().item()
#                 class_intersection[cls] += (pred_mask & target_mask).sum().item()
#                 class_union[cls] += (pred_mask | target_mask).sum().item()

#         if rank == 0 and isinstance(pbar, tqdm):
#             pbar.set_postfix({"loss": f"{loss.item():.4f}"})

#     # Compute local metrics
#     local_metrics = {
#         'loss': epoch_loss / len(loader),
#         'pixel_accuracy': total_correct / total_pixels,
#     }
    
#     class_names = ['Background', 'Crop', 'Weed']
#     for cls in range(num_classes):
#         local_metrics[f'iou_{class_names[cls]}'] = class_intersection[cls] / (class_union[cls] + 1e-6)
#         prec = class_tp[cls] / (class_tp[cls] + class_fp[cls] + 1e-6)
#         rec = class_tp[cls] / (class_tp[cls] + class_fn[cls] + 1e-6)
#         local_metrics[f'precision_{class_names[cls]}'] = prec
#         local_metrics[f'recall_{class_names[cls]}'] = rec
#         local_metrics[f'f1_{class_names[cls]}'] = 2 * prec * rec / (prec + rec + 1e-6)
    
#     # Reduce metrics across all GPUs
#     metrics = reduce_dict(local_metrics, world_size)
    
#     # Compute mean metrics
#     metrics['miou'] = np.mean([metrics[f'iou_{cls}'] for cls in class_names])
#     metrics['mean_precision'] = np.mean([metrics[f'precision_{cls}'] for cls in class_names])
#     metrics['mean_recall'] = np.mean([metrics[f'recall_{cls}'] for cls in class_names])
#     metrics['mean_f1'] = np.mean([metrics[f'f1_{cls}'] for cls in class_names])

#     return metrics


# @torch.no_grad()
# def evaluate(
#     model: nn.Module,
#     loader: DataLoader,
#     loss_fn,
#     device,
#     num_classes: int,
#     rank: int,
#     world_size: int,
#     split: str = "Val"
# ) -> Dict[str, float]:
#     """Evaluate model with DDP"""
#     model.eval()
#     total_loss = 0.0
#     total_correct = 0
#     total_pixels = 0
#     class_tp = [0, 0, 0]
#     class_fp = [0, 0, 0]
#     class_fn = [0, 0, 0]
#     class_intersection = [0, 0, 0]
#     class_union = [0, 0, 0]
    
#     if rank == 0:
#         pbar = tqdm(loader, desc=f"[{split}]")
#     else:
#         pbar = loader

#     for batch in pbar:
#         x = batch["images"].to(device)
#         y = batch["labels"].to(device).long()

#         logits = model(x)
#         loss = loss_fn(logits, y)
#         total_loss += loss.item()

#         preds = torch.argmax(logits, dim=1)
#         correct = (preds == y).sum().item()
#         total_correct += correct
#         total_pixels += y.numel()
        
#         for cls in range(num_classes):
#             pred_mask = (preds == cls)
#             target_mask = (y == cls)
            
#             class_tp[cls] += (pred_mask & target_mask).sum().item()
#             class_fp[cls] += (pred_mask & ~target_mask).sum().item()
#             class_fn[cls] += (~pred_mask & target_mask).sum().item()
#             class_intersection[cls] += (pred_mask & target_mask).sum().item()
#             class_union[cls] += (pred_mask | target_mask).sum().item()

#     # Compute local metrics
#     local_metrics = {
#         'loss': total_loss / len(loader),
#         'pixel_accuracy': total_correct / total_pixels,
#     }
    
#     class_names = ['Background', 'Crop', 'Weed']
#     for cls in range(num_classes):
#         local_metrics[f'iou_{class_names[cls]}'] = class_intersection[cls] / (class_union[cls] + 1e-6)
#         prec = class_tp[cls] / (class_tp[cls] + class_fp[cls] + 1e-6)
#         rec = class_tp[cls] / (class_tp[cls] + class_fn[cls] + 1e-6)
#         local_metrics[f'precision_{class_names[cls]}'] = prec
#         local_metrics[f'recall_{class_names[cls]}'] = rec
#         local_metrics[f'f1_{class_names[cls]}'] = 2 * prec * rec / (prec + rec + 1e-6)
    
#     # Reduce metrics across all GPUs
#     metrics = reduce_dict(local_metrics, world_size)
    
#     # Compute mean metrics
#     metrics['miou'] = np.mean([metrics[f'iou_{cls}'] for cls in class_names])
#     metrics['mean_precision'] = np.mean([metrics[f'precision_{cls}'] for cls in class_names])
#     metrics['mean_recall'] = np.mean([metrics[f'recall_{cls}'] for cls in class_names])
#     metrics['mean_f1'] = np.mean([metrics[f'f1_{cls}'] for cls in class_names])

#     return metrics


# # ======================== Main Training Function ========================

# def train_worker(rank, world_size, gpu_ids, args):
#     """Main training function for each GPU process"""
#     setup_ddp(rank, world_size, gpu_ids)
    
#     num_classes = 3
#     class_names = ["background", "crop", "weed"]
    
#     # Only rank 0 handles logging and checkpointing
#     if rank == 0:
#         if args.exp_name is None:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             channels = "4ch_RGBNIR" if args.use_rgbnir else "3ch_RGB"
#             args.exp_name = f"sugarbeets_{args.model}_{channels}_{timestamp}"

#         run_dir = os.path.join(args.output_dir, args.exp_name)
#         os.makedirs(run_dir, exist_ok=True)
#         ckpt_dir = os.path.join(run_dir, "checkpoints")
#         log_dir = os.path.join(run_dir, "logs")
#         os.makedirs(ckpt_dir, exist_ok=True)
#         os.makedirs(log_dir, exist_ok=True)

#         config_path = os.path.join(run_dir, "config.json")
#         config_dict = {**vars(args), "num_classes": num_classes, "class_names": class_names, "gpu_ids": gpu_ids}
#         with open(config_path, "w") as f:
#             json.dump(config_dict, f, indent=2)

#         print(f"\n{'='*80}")
#         print(f"[INFO] Multi-GPU Training with {world_size} GPUs (IDs: {gpu_ids})")
#         print(f"[INFO] Dataset: Sugar Beets ({num_classes} classes)")
#         print(f"[INFO] Run directory: {run_dir}")
#         print(f"[INFO] TensorBoard logs: {log_dir}")
#         print(f"[INFO] To view logs, run: tensorboard --logdir={log_dir}")
#         print(f"{'='*80}\n")

#         writer = SummaryWriter(log_dir)
#     else:
#         writer = None
    
#     # Create dataloaders ONCE (not per epoch!)
#     if rank == 0:
#         print("[INFO] Creating fixed dataloaders (computed once)...")
    
#     train_loader, val_loader, test_loader, train_sampler = create_fixed_dataloaders(
#         data_root=args.data_root,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         use_rgbnir=args.use_rgbnir,
#         target_size=(args.height, args.width),
#         nir_drop_prob=args.nir_drop,
#         seed=42,
#         rank=rank,
#         world_size=world_size
#     )
    
#     # Model - use local rank since CUDA_VISIBLE_DEVICES is set
#     device = torch.device(f"cuda:{rank}")
#     in_ch = 4 if args.use_rgbnir else 3
#     model = create_model(
#         architecture=args.model,
#         in_channels=in_ch,
#         num_classes=num_classes,
#         base_ch=args.base_ch
#     ).to(device)
    
#     # Wrap model with DDP
#     model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
#     if rank == 0:
#         model_info = get_model_info(model.module)
#         print(f"[INFO] Model: {model_info['architecture']}")
#         print(f"[INFO] Parameters: {model_info['total_parameters']:,} "
#               f"({model_info['total_parameters_million']:.2f}M)")
    
#     # Optimizer & Loss
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
#     loss_fn = MultiClassDiceLoss(num_classes=num_classes)
#     scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
#     best_val_miou = 0.0
#     start_epoch = 1
    
#     # Training loop
#     if rank == 0:
#         print(f"\n[INFO] Starting training for {args.epochs} epochs...\n")
    
#     for epoch in range(start_epoch, args.epochs + 1):
#         # Set epoch for distributed sampler (important for shuffling)
#         train_sampler.set_epoch(epoch)
        
#         # Train
#         train_metrics = train_one_epoch(
#             model, train_loader, optimizer, loss_fn, device,
#             epoch, writer, num_classes, scaler, rank, world_size
#         )
        
#         # Validate
#         val_metrics = evaluate(
#             model, val_loader, loss_fn, device, num_classes,
#             rank, world_size, split="Val"
#         )
        
#         # Only rank 0 logs and saves
#         if rank == 0:
#             # Log all metrics to TensorBoard
#             # Overall metrics
#             writer.add_scalar("Train/Loss", train_metrics['loss'], epoch)
#             writer.add_scalar("Train/Pixel_Accuracy", train_metrics['pixel_accuracy'], epoch)
#             writer.add_scalar("Train/mIoU", train_metrics['miou'], epoch)
#             writer.add_scalar("Train/Mean_Precision", train_metrics['mean_precision'], epoch)
#             writer.add_scalar("Train/Mean_Recall", train_metrics['mean_recall'], epoch)
#             writer.add_scalar("Train/Mean_F1", train_metrics['mean_f1'], epoch)
            
#             writer.add_scalar("Val/Loss", val_metrics['loss'], epoch)
#             writer.add_scalar("Val/Pixel_Accuracy", val_metrics['pixel_accuracy'], epoch)
#             writer.add_scalar("Val/mIoU", val_metrics['miou'], epoch)
#             writer.add_scalar("Val/Mean_Precision", val_metrics['mean_precision'], epoch)
#             writer.add_scalar("Val/Mean_Recall", val_metrics['mean_recall'], epoch)
#             writer.add_scalar("Val/Mean_F1", val_metrics['mean_f1'], epoch)
            
#             # Per-class metrics
#             for cls_name in ['Background', 'Crop', 'Weed']:
#                 writer.add_scalar(f"Train_IoU/{cls_name}", train_metrics[f'iou_{cls_name}'], epoch)
#                 writer.add_scalar(f"Train_Precision/{cls_name}", train_metrics[f'precision_{cls_name}'], epoch)
#                 writer.add_scalar(f"Train_Recall/{cls_name}", train_metrics[f'recall_{cls_name}'], epoch)
#                 writer.add_scalar(f"Train_F1/{cls_name}", train_metrics[f'f1_{cls_name}'], epoch)
                
#                 writer.add_scalar(f"Val_IoU/{cls_name}", val_metrics[f'iou_{cls_name}'], epoch)
#                 writer.add_scalar(f"Val_Precision/{cls_name}", val_metrics[f'precision_{cls_name}'], epoch)
#                 writer.add_scalar(f"Val_Recall/{cls_name}", val_metrics[f'recall_{cls_name}'], epoch)
#                 writer.add_scalar(f"Val_F1/{cls_name}", val_metrics[f'f1_{cls_name}'], epoch)
            
#             writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
            
#             # Print detailed metrics
#             print_metrics(train_metrics, split="Train", epoch=epoch)
#             print_metrics(val_metrics, split="Validation", epoch=epoch)
            
#             # Save best checkpoint
#             current_miou = val_metrics['miou']
#             if current_miou > best_val_miou:
#                 best_val_miou = current_miou
#                 best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
#                 torch.save({
#                     "epoch": epoch,
#                     "model_state_dict": model.module.state_dict(),
#                     "optimizer_state_dict": optimizer.state_dict(),
#                     "best_val_miou": best_val_miou,
#                     **{f"val_{k}": v for k, v in val_metrics.items()},
#                 }, best_ckpt_path)
#                 print(f"{'='*80}")
#                 print(f"[INFO] ✓ NEW BEST MODEL - Saved checkpoint (mIoU: {best_val_miou:.4f})")
#                 print(f"{'='*80}\n")
        
#         scheduler.step()
    
#     # Final test evaluation
#     if rank == 0:
#         print(f"\n{'='*80}")
#         print("[INFO] Running final test evaluation...")
#         print(f"{'='*80}")
        
#         best_ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pth"))
#         model.module.load_state_dict(best_ckpt["model_state_dict"])
    
#     # Synchronize before test
#     dist.barrier()
    
#     test_metrics = evaluate(
#         model, test_loader, loss_fn, device, num_classes,
#         rank, world_size, split="Test"
#     )
    
#     if rank == 0:
#         # Log test metrics to TensorBoard
#         writer.add_scalar("Test/Loss", test_metrics['loss'], args.epochs)
#         writer.add_scalar("Test/Pixel_Accuracy", test_metrics['pixel_accuracy'], args.epochs)
#         writer.add_scalar("Test/mIoU", test_metrics['miou'], args.epochs)
#         writer.add_scalar("Test/Mean_Precision", test_metrics['mean_precision'], args.epochs)
#         writer.add_scalar("Test/Mean_Recall", test_metrics['mean_recall'], args.epochs)
#         writer.add_scalar("Test/Mean_F1", test_metrics['mean_f1'], args.epochs)
        
#         for cls_name in ['Background', 'Crop', 'Weed']:
#             writer.add_scalar(f"Test_IoU/{cls_name}", test_metrics[f'iou_{cls_name}'], args.epochs)
#             writer.add_scalar(f"Test_Precision/{cls_name}", test_metrics[f'precision_{cls_name}'], args.epochs)
#             writer.add_scalar(f"Test_Recall/{cls_name}", test_metrics[f'recall_{cls_name}'], args.epochs)
#             writer.add_scalar(f"Test_F1/{cls_name}", test_metrics[f'f1_{cls_name}'], args.epochs)
        
#         # Print detailed test metrics
#         print_metrics(test_metrics, split="Test (Best Model)")
        
#         results_path = os.path.join(run_dir, "test_results.json")
#         with open(results_path, "w") as f:
#             json.dump(test_metrics, f, indent=2)
        
#         writer.close()
#         print(f"{'='*80}")
#         print(f"[INFO] Training complete!")
#         print(f"[INFO] Results saved to: {run_dir}")
#         print(f"[INFO] To view TensorBoard logs, run:")
#         print(f"       tensorboard --logdir={log_dir}")
#         print(f"{'='*80}\n")
    
#     cleanup_ddp()


# # ======================== Entry Point ========================

# def main():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument("--model", type=str, default="lightmanet",
#                         choices=["unet", "unet_sa", "lightmanet"])
#     parser.add_argument("--base_ch", type=int, default=32)
#     parser.add_argument("--data_root", type=str,
#                         default="/home/vjtiadmin/Desktop/BTechGroup/FINAL_SUGARBEETS_DATASET")
#     parser.add_argument("--use_rgbnir", action="store_true")
#     parser.add_argument("--height", type=int, default=966)
#     parser.add_argument("--width", type=int, default=1296)
#     parser.add_argument("--batch_size", type=int, default=32,
#                         help="Batch size per GPU")
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--num_workers", type=int, default=4)
#     parser.add_argument("--mixed_precision", action="store_true")
#     parser.add_argument("--nir_drop", type=float, default=0.0)
#     parser.add_argument("--exp_name", type=str, default=None)
#     parser.add_argument("--output_dir", type=str, default="./experiments")
#     parser.add_argument("--gpu_ids", type=str, default="0,1,2",
#                         help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')")
    
#     args = parser.parse_args()
    
#     # Parse GPU IDs
#     gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
#     world_size = len(gpu_ids)
    
#     print(f"[INFO] Using GPUs: {gpu_ids}")
#     print(f"[INFO] World size: {world_size}")
    
#     if world_size < 1:
#         raise ValueError("Must specify at least one GPU")
    
#     if world_size == 1:
#         print("[INFO] Running on single GPU")
#         train_worker(0, 1, gpu_ids, args)
#     else:
#         torch.multiprocessing.spawn(
#             train_worker,
#             args=(world_size, gpu_ids, args),
#             nprocs=world_size,
#             join=True
#         )


# if __name__ == "__main__":
#     main()