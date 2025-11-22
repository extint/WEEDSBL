#!/usr/bin/env python3
"""
evaluate_checkpoint.py

Evaluate a trained checkpoint on the weedy rice validation set.
Computes: Pixel Accuracy and FLOPs (GFLOPs)

Usage:
    python evaluate_checkpoint.py \
        --checkpoint experiments/weedy_rice_unet/checkpoints/best_model.pth \
        --model unet \
        --use_rgbnir \
        --use_old_unet
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple
import sys

# Import your existing utilities
from models import create_model, count_parameters
from rice_weed_data_loader import create_weedy_rice_rgbnir_dataloaders

# Import the old UNet architecture
from UNet import UNet as OldUNet


def create_model_with_fallback(architecture, in_channels, num_classes, base_ch, use_old_unet=False):
    """
    Create model with option to use old UNet architecture from UNet.py
    """
    if architecture == 'unet' and use_old_unet:
        print("Using UNet architecture from UNet.py")
        model = OldUNet(
            in_channels=in_channels,
            base_ch=base_ch,
            out_channels=num_classes
        )
        return model
    else:
        # Use create_model from models.py
        return create_model(
            architecture=architecture,
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_ch
        )


def count_flops_manual(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """
    Manual FLOPs counting for Conv2d, Linear, BatchNorm2d layers
    """
    flops = 0
    
    def conv2d_flops(module, input, output):
        nonlocal flops
        batch_size = input[0].size(0)
        output_height, output_width = output.size(2), output.size(3)
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        output_elements = batch_size * output_height * output_width * module.out_channels
        flops += kernel_ops * output_elements
        
    def linear_flops(module, input, output):
        nonlocal flops
        batch_size = input[0].size(0)
        flops += batch_size * module.in_features * module.out_features
    
    def bn_flops(module, input, output):
        nonlocal flops
        flops += input[0].numel() * 2
    
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_flops))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_flops))
        elif isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(bn_flops))
    
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    return flops


def compute_pixel_accuracy(model: nn.Module, val_loader, device: torch.device) -> float:
    """
    Compute pixel accuracy on validation set
    Auto-detects binary (1 channel) vs multi-class (>1 channel) output
    """
    model.eval()
    correct_pixels = 0
    total_pixels = 0
    
    # Determine if binary or multi-class by checking first batch
    first_batch = next(iter(val_loader))
    with torch.no_grad():
        sample_output = model(first_batch['images'][:1].to(device))
        output_channels = sample_output.shape[1]
        is_binary = (output_channels == 1)
    
    print(f"Detected {'BINARY' if is_binary else 'MULTI-CLASS'} segmentation ({output_channels} output channel(s))")
    print("Computing pixel accuracy on validation set...")
    
    # Track class distribution
    gt_class_counts = {}
    pred_class_counts = {}
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            
            # Handle binary vs multi-class
            if is_binary:
                # Binary segmentation: sigmoid + threshold at 0.5
                preds = (torch.sigmoid(outputs.squeeze(1)) > 0.5).long()
            else:
                # Multi-class: argmax over channels
                preds = torch.argmax(outputs, dim=1)
            
            correct_pixels += (preds == labels).sum().item()
            total_pixels += labels.numel()
            
            # Track class distribution
            for c in labels.unique():
                c_val = c.item()
                gt_class_counts[c_val] = gt_class_counts.get(c_val, 0) + (labels == c).sum().item()
            
            for c in preds.unique():
                c_val = c.item()
                pred_class_counts[c_val] = pred_class_counts.get(c_val, 0) + (preds == c).sum().item()
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(val_loader)} batches...")
    
    pixel_accuracy = 100.0 * correct_pixels / total_pixels
    
    # Print class distribution analysis
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*70)
    print("\nGround Truth Distribution:")
    for c in sorted(gt_class_counts.keys()):
        percentage = 100.0 * gt_class_counts[c] / total_pixels
        print(f"  Class {c}: {gt_class_counts[c]:,} pixels ({percentage:.2f}%)")
    
    print("\nModel Prediction Distribution:")
    for c in sorted(pred_class_counts.keys()):
        percentage = 100.0 * pred_class_counts[c] / total_pixels
        print(f"  Class {c}: {pred_class_counts[c]:,} pixels ({percentage:.2f}%)")
    print("="*70 + "\n")
    
    return pixel_accuracy


def compute_gflops(model: nn.Module, input_shape: Tuple[int, int, int], device: torch.device) -> float:
    """Compute FLOPs in GFLOPs"""
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    flops = count_flops_manual(model, dummy_input)
    gflops = flops / 1e9
    
    return gflops


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint on weedy rice validation set')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to checkpoint .pth file')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['unet', 'unet_sa', 'lightmanet'],
                        help='Model architecture')
    parser.add_argument('--use_rgbnir', action='store_true', 
                        help='Use RGB+NIR (4 channels)')
    parser.add_argument('--use_old_unet', action='store_true',
                        help='Use old UNet architecture from UNet.py (for old checkpoints)')
    parser.add_argument('--data_root', type=str, 
                        default="/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB",
                        help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for validation')
    parser.add_argument('--img_size', type=int, nargs=2, default=[960, 1280],
                        help='Input image size (height width)')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of output classes (1 for binary, 3 for multi-class)')
    parser.add_argument('--base_ch', type=int, default=32,
                        help='Base number of channels')
    
    args = parser.parse_args()
    
    # Validation
    if args.use_old_unet and args.model != 'unet':
        print("Error: --use_old_unet can only be used with --model unet")
        sys.exit(1)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = 4 if args.use_rgbnir else 3
    target_size = tuple(args.img_size)
    
    print(f"\n{'='*70}")
    print(f"CHECKPOINT EVALUATION")
    print(f"{'='*70}")
    print(f"Checkpoint:       {Path(args.checkpoint).name}")
    print(f"Model:            {args.model.upper()}")
    print(f"Architecture:     {'Old UNet (UNet.py)' if args.use_old_unet else 'Current (models.py)'}")
    print(f"Channels:         {'RGB+NIR (4)' if args.use_rgbnir else 'RGB (3)'}")
    print(f"Input Size:       {target_size[0]}x{target_size[1]}")
    print(f"Num Classes:      {args.num_classes}")
    print(f"Device:           {device}")
    print(f"{'='*70}\n")
    
    # Create model
    print("Creating model...")
    model = create_model_with_fallback(
        architecture=args.model,
        in_channels=in_channels,
        num_classes=args.num_classes,
        base_ch=args.base_ch,
        use_old_unet=args.use_old_unet
    )
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'epoch' in checkpoint:
            print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
        if 'best_miou' in checkpoint:
            print(f"  Best mIoU: {checkpoint['best_miou']:.4f}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Checkpoint loaded successfully")
    except RuntimeError as e:
        print(f"⚠ Warning: Loading with strict=False due to key mismatch")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    
    # Print model info
    num_params = count_parameters(model)
    print(f"  Total parameters: {num_params:,} ({num_params/1e6:.2f}M)\n")
    
    # Create validation dataloader
    print("Loading validation dataset...")
    _, val_loader, _ = create_weedy_rice_rgbnir_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        use_rgbnir=args.use_rgbnir,
        target_size=target_size,
        nir_drop_prob=0.0
    )
    print(f"✓ Validation set: {len(val_loader.dataset)} images")
    print(f"  Batches: {len(val_loader)}\n")
    
    # Compute Pixel Accuracy
    print("="*70)
    print("COMPUTING PIXEL ACCURACY")
    print("="*70)
    pixel_acc = compute_pixel_accuracy(model, val_loader, device)
    print(f"\n✓ Pixel Accuracy: {pixel_acc:.2f}%\n")
    
    # Compute FLOPs
    print("="*70)
    print("COMPUTING FLOPs")
    print("="*70)
    input_shape = (in_channels, target_size[0], target_size[1])
    print(f"Input shape: {input_shape}")
    gflops = compute_gflops(model, input_shape, device)
    print(f"✓ FLOPs: {gflops:.3f} GFLOPs\n")
    
    # Final Summary
    print(f"{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Checkpoint:         {Path(args.checkpoint).name}")
    print(f"Model:              {args.model.upper()}")
    print(f"Architecture:       {'Old UNet (UNet.py)' if args.use_old_unet else 'Current (models.py)'}")
    print(f"Input Channels:     {in_channels}")
    print(f"Input Size:         {target_size[0]}x{target_size[1]}")
    print(f"Parameters:         {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Pixel Accuracy:     {pixel_acc:.2f}%")
    print(f"FLOPs:              {gflops:.3f} GFLOPs")
    print(f"{'='*70}\n")
    
    # Save results
    results_file = Path(args.checkpoint).parent / f"eval_{Path(args.checkpoint).stem}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Checkpoint Evaluation Results\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Checkpoint:         {args.checkpoint}\n")
        f.write(f"Model:              {args.model}\n")
        f.write(f"Architecture:       {'Old UNet (UNet.py)' if args.use_old_unet else 'Current (models.py)'}\n")
        f.write(f"Input Channels:     {in_channels}\n")
        f.write(f"Input Size:         {target_size[0]}x{target_size[1]}\n")
        f.write(f"Num Classes:        {args.num_classes}\n")
        f.write(f"Base Channels:      {args.base_ch}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Parameters:       {num_params:,} ({num_params/1e6:.2f}M)\n")
        f.write(f"  Pixel Accuracy:   {pixel_acc:.2f}%\n")
        f.write(f"  FLOPs:            {gflops:.3f} GFLOPs\n")
        f.write(f"\n{'='*70}\n")
    
    print(f"✓ Results saved to: {results_file}")


if __name__ == '__main__':
    main()
