#!/usr/bin/env python3
"""
Training script for LightMANet on CWD dataset
Supports RGB-only semantic segmentation with 3 classes (0, 1, 2)
"""

import argparse
import os
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Import custom modules
from MANNet import LightMANet
from cwd_data_loader import create_cwd_dataloaders


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> Dict[int, float]:
    """Calculate IoU for each class"""
    ious = {}
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            ious[cls] = float('nan')
        else:
            ious[cls] = intersection / union
    
    return ious


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch: int,
    writer: SummaryWriter = None
):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_ious = {0: [], 1: [], 2: []}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        
        # Calculate IoU
        batch_ious = calculate_iou(preds, masks, num_classes=3)
        for cls in range(3):
            if not np.isnan(batch_ious[cls]):
                running_ious[cls].append(batch_ious[cls])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{np.nanmean([np.mean(running_ious[i]) for i in range(3) if len(running_ious[i]) > 0]):.4f}'
        })
        
        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
    
    # Epoch metrics
    epoch_loss = running_loss / len(dataloader)
    mean_ious = {cls: np.mean(running_ious[cls]) if len(running_ious[cls]) > 0 else 0.0 
                 for cls in range(3)}
    mean_iou = np.mean([mean_ious[cls] for cls in range(3)])
    
    return epoch_loss, mean_iou, mean_ious


def validate(
    model: nn.Module,
    dataloader,
    criterion,
    device,
    epoch: int,
    writer: SummaryWriter = None
):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_ious = {0: [], 1: [], 2: []}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate IoU
            batch_ious = calculate_iou(preds, masks, num_classes=3)
            for cls in range(3):
                if not np.isnan(batch_ious[cls]):
                    running_ious[cls].append(batch_ious[cls])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mIoU': f'{np.nanmean([np.mean(running_ious[i]) for i in range(3) if len(running_ious[i]) > 0]):.4f}'
            })
    
    # Epoch metrics
    epoch_loss = running_loss / len(dataloader)
    mean_ious = {cls: np.mean(running_ious[cls]) if len(running_ious[cls]) > 0 else 0.0 
                 for cls in range(3)}
    mean_iou = np.mean([mean_ious[cls] for cls in range(3)])
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', epoch_loss, epoch)
        writer.add_scalar('Val/mIoU', mean_iou, epoch)
        for cls in range(3):
            writer.add_scalar(f'Val/IoU_class_{cls}', mean_ious[cls], epoch)
    
    return epoch_loss, mean_iou, mean_ious


def main():
    parser = argparse.ArgumentParser(description='Train LightMANet on CWD dataset')
    
    # Data paths
    parser.add_argument('--data_root', type=str, 
                        default='/home/vjti-comp/Downloads/CWD/CWD-Github',
                        help='Root directory of CWD dataset')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--base_ch', type=int, default=32, help='Base channels for LightMANet')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes (0, 1, 2)')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_cwd', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    loaders = create_cwd_dataloaders(
        train_img_dir=os.path.join(args.data_root, 'train', 'images'),
        train_mask_dir=os.path.join(args.data_root, 'train', 'Morphed_Images'),
        val_img_dir=os.path.join(args.data_root, 'valid', 'images'),
        val_mask_dir=os.path.join(args.data_root, 'valid', 'Morphed_Images'),
        test_img_dir=os.path.join(args.data_root, 'test', 'images'),
        test_mask_dir=os.path.join(args.data_root, 'test', 'Morphed_Images'),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_size, args.img_size)
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders.get('test', None)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("Creating model...")
    model = LightMANet(
        in_channels=3,  # RGB only
        num_classes=args.num_classes,
        base_ch=args.base_ch
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    best_val_iou = 0.0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_miou, train_ious = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
        print(f"  Class IoUs: {train_ious}")
        
        # Validate
        val_loss, val_miou, val_ious = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        print(f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        print(f"  Class IoUs: {val_ious}")
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if val_miou > best_val_iou:
            best_val_iou = val_miou
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
                'val_ious': val_ious
            }, checkpoint_path)
            print(f"✓ Saved best model with mIoU: {val_miou:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
            }, checkpoint_path)
    
    # Test on best model
    if test_loader:
        print("\n" + "="*60)
        print("Testing best model...")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_miou, test_ious = validate(
            model, test_loader, criterion, device, 0, None
        )
        
        print(f"\nTest Loss: {test_loss:.4f} | Test mIoU: {test_miou:.4f}")
        print(f"  Class IoUs: {test_ious}")
    
    writer.close()
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
