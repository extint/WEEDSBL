#!/usr/bin/env python3
"""
Inference script for LightMANet on CWD dataset
Shows predictions with ground truth for reference
Displays metrics from saved checkpoint only
"""

import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from MANNet import LightMANet
from cwd_data_loader import get_val_transforms


# Color mapping for visualization (RGB format)
COLOR_MAP = {
    0: [0, 0, 0],        # Black - Background
    1: [0, 255, 0],      # Green - Class 1
    2: [0, 0, 255]       # Blue - Class 2
}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert class mask to RGB visualization"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        rgb[mask == class_id] = color
    
    return rgb


def print_checkpoint_metrics(checkpoint: dict):
    """Print metrics stored in checkpoint"""
    print("\n" + "="*60)
    print("CHECKPOINT METRICS (from training)")
    print("="*60)
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f" mIoU: {checkpoint.get('val_miou', 'N/A'):.4f}")
    
    if 'val_ious' in checkpoint:
        print("\nPer-Class Validation IoU:")
        for cls, iou in checkpoint['val_ious'].items():
            print(f"  Class {cls}: {iou:.4f}")
    
    if 'val_loss' in checkpoint:
        print(f"\nValidation Loss: {checkpoint['val_loss']:.4f}")
    
    if 'train_miou' in checkpoint:
        print(f"\nTraining mIoU: {checkpoint['train_miou']:.4f}")
    
    print("="*60 + "\n")


def inference_single_image(
    model: nn.Module,
    image_path: str,
    mask_path: str,
    device: torch.device,
    checkpoint_info: dict,
    img_size: tuple = (640, 640),
    save_path: str = None
):
    """
    Run inference on a single image and visualize results
    Only shows checkpoint metrics, not calculated metrics
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load ground truth mask (for visualization only)
    mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_gt = np.clip(mask_gt, 0, 2)
    
    # Prepare image for model
    transform = get_val_transforms(img_size)
    transformed = transform(image=image_rgb.copy(), mask=mask_gt.copy())
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0)
    
    # Convert masks to RGB for visualization
    pred_rgb = mask_to_rgb(pred.cpu().numpy())
    mask_gt_rgb = mask_to_rgb(mask_gt)
    
    # Resize image for display if needed
    if image_rgb.shape[:2] != img_size:
        image_rgb_resized = cv2.resize(image_rgb, (img_size[1], img_size[0]))
    else:
        image_rgb_resized = image_rgb
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Image
    axes[0].imshow(image_rgb_resized)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth (for reference only)
    axes[1].imshow(mask_gt_rgb)
    axes[1].set_title('Ground Truth (Reference)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred_rgb)
    axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color=np.array(COLOR_MAP[0])/255, label='Class 0 (Background)'),
        mpatches.Patch(color=np.array(COLOR_MAP[1])/255, label='Class 1 (Crop)'),
        mpatches.Patch(color=np.array(COLOR_MAP[2])/255, label='Class 2 (Weed)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12)
    
    # Add checkpoint metrics as text
    metrics_text = f"""
    MODEL CHECKPOINT METRICS (from training validation set)
    
    Epoch: {checkpoint_info.get('epoch', 'N/A')}
    Validation mIoU: {checkpoint_info.get('val_miou', 'N/A'):.4f}
    """
    
    # if 'val_ious' in checkpoint_info:
    #     metrics_text += "\n    Per-Class Validation IoU:\n"
    #     for cls, iou in checkpoint_info['val_ious'].items():
    #         metrics_text += f"      Class {cls}: {iou:.4f}\n"
    
    # if 'val_loss' in checkpoint_info:
    #     metrics_text += f"    Validation Loss: {checkpoint_info['train_loss']:.4f}\n"
    
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11, 
             family='monospace', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.20, 1, 0.95])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to: {save_path}")
    
    plt.show()
    
    # Print info to console
    print("\n" + "="*60)
    print(f"Inference on: {os.path.basename(image_path)}")
    print("="*60)
    print("Showing checkpoint metrics (from training validation set)")
    print("Ground truth is displayed for visual reference only")
    print("="*60)


def inference_batch(
    model: nn.Module,
    image_dir: str,
    mask_dir: str,
    device: torch.device,
    checkpoint_info: dict,
    img_size: tuple = (640, 640),
    num_samples: int = 5,
    save_dir: str = None
):
    """
    Run inference on multiple images
    Shows checkpoint metrics, ground truth for reference only
    """
    model.eval()
    
    # Get image files
    import glob
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[:num_samples]
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Inference")):
        # Load image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get corresponding mask
        img_basename = os.path.basename(img_path)
        mask_name = img_basename.replace('.jpg', '.png')
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, img_basename.replace('.jpg', '_morphed.png'))
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_basename}, skipping...")
            continue
        
        # Load ground truth (for visualization only)
        mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_gt = np.clip(mask_gt, 0, 2)
        
        # Prepare image for model
        transform = get_val_transforms(img_size)
        transformed = transform(image=image_rgb.copy(), mask=mask_gt.copy())
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0)
        
        # Denormalize image for visualization
        image_display = transformed['image'].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_display = std * image_display + mean
        image_display = np.clip(image_display * 255, 0, 255).astype(np.uint8)
        
        # Convert masks to RGB
        pred_rgb = mask_to_rgb(pred.cpu().numpy())
        mask_gt_rgb = mask_to_rgb(mask_gt)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(image_display)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(mask_gt_rgb)
        axes[1].set_title('Ground Truth (Reference)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(pred_rgb)
        axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=np.array(COLOR_MAP[0])/255, label='Class 0 (Background)'),
            mpatches.Patch(color=np.array(COLOR_MAP[1])/255, label='Class 1 (Weed)'),
            mpatches.Patch(color=np.array(COLOR_MAP[2])/255, label='Class 2 (Crop)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12)
        
        # Checkpoint metrics
        title_text = f"Model Metrics (from checkpoint): Epoch {checkpoint_info.get('epoch', 'N/A')} | Val mIoU: {checkpoint_info.get('val_miou', 'N/A'):.4f}"
        fig.suptitle(title_text, fontsize=12, y=0.92, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_dir:
            save_path = os.path.join(save_dir, f'inference_{idx}_{img_basename.replace(".jpg", ".png")}')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    print(f"\n✓ Processed {len(image_files)} images")
    if save_dir:
        print(f"✓ Results saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Inference script for LightMANet')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    
    # Single image inference
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single input image')
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to ground truth mask (for visualization reference only)')
    
    # Batch inference
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing test images')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory containing test masks (for visualization reference only)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to process in batch mode')
    
    # Model parameters
    parser.add_argument('--base_ch', type=int, default=32,
                        help='Base channels for LightMANet')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = LightMANet(
        in_channels=3,
        num_classes=args.num_classes,
        base_ch=args.base_ch
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print checkpoint information
    print_checkpoint_metrics(checkpoint)
    
    model.eval()
    
    # Prepare checkpoint info for display
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_miou': checkpoint.get('val_miou', 0.0),
        'val_ious': checkpoint.get('val_ious', {}),
        'val_loss': checkpoint.get('val_loss', None)
    }
    
    # Single image inference
    if args.image and args.mask:
        print("Running single image inference...")
        print("Note: Ground truth is shown for visual comparison only")
        print("Metrics displayed are from the checkpoint (validation set)\n")
        
        save_path = os.path.join(args.save_dir, 'single_inference.png')
        os.makedirs(args.save_dir, exist_ok=True)
        
        inference_single_image(
            model, args.image, args.mask, device, checkpoint_info,
            img_size=(args.img_size, args.img_size),
            save_path=save_path
        )
    
    # Batch inference
    elif args.image_dir and args.mask_dir:
        print("Running batch inference...")
        print("Note: Ground truth is shown for visual comparison only")
        print("Metrics displayed are from the checkpoint (validation set)\n")
        
        inference_batch(
            model, args.image_dir, args.mask_dir, device, checkpoint_info,
            img_size=(args.img_size, args.img_size),
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )
    
    else:
        print("Error: Please provide either:")
        print("  1. --image and --mask for single image inference")
        print("  2. --image_dir and --mask_dir for batch inference")


if __name__ == '__main__':
    main()
