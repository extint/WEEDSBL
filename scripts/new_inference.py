#!/usr/bin/env python3

"""
Universal inference script for agricultural weed segmentation models.
Supports both:
  - Weedy Rice dataset (binary segmentation)
  - WeedsGalore dataset (3-class segmentation with GT comparison)
"""

import argparse
import os
import cv2
import torch
import numpy as np
from pathlib import Path

from models import create_model
from weedutils.rgbnir_preprocessing import RGBNIRPreprocessor


def load_weedsgalore_bands(image_id: str, dataset_root: str, date_folder: str = None):
    """
    Load R, G, B, NIR bands for WeedsGalore dataset.

    Args:
        image_id: Image ID (e.g., '2023-05-25_0109')
        dataset_root: Path to weedsgalore-dataset folder
        date_folder: Optional date folder (auto-detected if None)

    Returns:
        tuple: (r, g, b, nir) as numpy arrays
    """
    # Auto-detect date folder if not provided
    if date_folder is None:
        date_prefix = image_id.split('_')[0]  # Extract '2023-05-25'
        date_folder = date_prefix

    images_dir = Path(dataset_root) / date_folder / "images"

    # Load individual bands
    r = cv2.imread(str(images_dir / f"{image_id}_R.png"), cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(str(images_dir / f"{image_id}_G.png"), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(images_dir / f"{image_id}_B.png"), cv2.IMREAD_GRAYSCALE)
    nir = cv2.imread(str(images_dir / f"{image_id}_NIR.png"), cv2.IMREAD_GRAYSCALE)

    if any(band is None for band in [r, g, b, nir]):
        raise FileNotFoundError(f"Could not load all bands for {image_id} in {images_dir}")

    return r, g, b, nir


def load_weedsgalore_mask(image_id: str, dataset_root: str, date_folder: str = None, target_size=None):
    """
    Load ground truth semantic mask for WeedsGalore dataset.

    Args:
        image_id: Image ID (e.g., '2023-05-25_0109')
        dataset_root: Path to weedsgalore-dataset folder
        date_folder: Optional date folder (auto-detected if None)
        target_size: Optional (H, W) to resize mask to

    Returns:
        np.ndarray: Ground truth mask with class indices {0, 1, 2}
    """
    # Auto-detect date folder if not provided
    if date_folder is None:
        date_prefix = image_id.split('_')[0]
        date_folder = date_prefix

    mask_path = Path(dataset_root) / date_folder / "semantics" / f"{image_id}.png"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Apply the same fix as in dataloader: class values >1 -> 2 (weed)
    mask[mask > 1] = 2

    # Resize if requested
    if target_size is not None:
        mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

    return mask


def create_comparison_visualization(pred_mask, gt_mask):
    """
    Create a color-coded comparison visualization.

    Color coding:
    - Black: True Negative (both predict background)
    - Green: True Positive (both predict same non-background class)
    - Red: False Positive (predicted class, but GT is background or different)
    - Yellow: False Negative (GT has class, but predicted background or different)

    Args:
        pred_mask: Predicted mask (H, W) with class indices
        gt_mask: Ground truth mask (H, W) with class indices

    Returns:
        np.ndarray: RGB visualization (H, W, 3)
    """
    h, w = pred_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Compute per-pixel correctness
    correct = (pred_mask == gt_mask)

    # True Positives (correct predictions)
    vis[correct] = [0, 255, 0]  # Green

    # False predictions
    incorrect = ~correct

    # False Positives (predicted non-zero, but GT is different)
    false_pos = incorrect & (pred_mask != gt_mask)
    vis[false_pos] = [0, 0, 255]  # Red

    # Make background black where both agree it's background
    both_bg = (pred_mask == 0) & (gt_mask == 0)
    vis[both_bg] = [0, 0, 0]  # Black

    return vis


def create_side_by_side_visualization(pred_mask, gt_mask):
    """
    Create side-by-side visualization of GT and prediction.

    Args:
        pred_mask: Predicted mask (H, W)
        gt_mask: Ground truth mask (H, W)

    Returns:
        np.ndarray: Side-by-side RGB visualization
    """
    # Color maps
    def colorize_mask(mask):
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colored[mask == 0] = [0, 0, 0]        # Black: Background
        colored[mask == 1] = [0, 255, 0]      # Green: Crop
        colored[mask == 2] = [255, 0, 0]      # Red: Weed
        return colored

    gt_colored = colorize_mask(gt_mask)
    pred_colored = colorize_mask(pred_mask)

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    gt_labeled = gt_colored.copy()
    pred_labeled = pred_colored.copy()
    cv2.putText(gt_labeled, "Ground Truth", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(pred_labeled, "Prediction", (10, 30), font, 1, (255, 255, 255), 2)

    # Concatenate side by side
    side_by_side = np.hstack([gt_labeled, pred_labeled])

    return side_by_side


def compute_metrics(pred_mask, gt_mask, num_classes=3):
    """
    Compute segmentation metrics.

    Args:
        pred_mask: Predicted mask (H, W)
        gt_mask: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        dict: Metrics including IoU, accuracy, per-class IoU
    """
    metrics = {}

    # Overall accuracy
    correct = (pred_mask == gt_mask).sum()
    total = pred_mask.size
    metrics['accuracy'] = correct / total

    # Per-class IoU
    class_ious = []
    class_names = ["Background", "Crop", "Weed"]

    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)

        intersection = (pred_cls & gt_cls).sum()
        union = (pred_cls | gt_cls).sum()

        iou = intersection / union if union > 0 else 0.0
        class_ious.append(iou)
        metrics[f'iou_{class_names[cls]}'] = iou

    metrics['mean_iou'] = np.mean(class_ious)

    # Confusion matrix info
    for cls in range(num_classes):
        gt_cls_pixels = (gt_mask == cls).sum()
        pred_cls_pixels = (pred_mask == cls).sum()
        metrics[f'{class_names[cls]}_gt_pixels'] = gt_cls_pixels
        metrics[f'{class_names[cls]}_pred_pixels'] = pred_cls_pixels

    return metrics


def preprocess_weedsgalore(r, g, b, nir, target_size=(600, 600), use_rgbnir=True):
    """
    Preprocess WeedsGalore bands for inference.

    Args:
        r, g, b, nir: Individual band images (grayscale)
        target_size: (H, W) to resize to
        use_rgbnir: If True, return 4 channels, else 3 (RGB only)

    Returns:
        torch.Tensor: Shape (1, C, H, W) where C=3 or 4
    """
    # Stack into multi-channel image
    rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3)

    if use_rgbnir:
        img = np.concatenate([rgb, nir[..., None]], axis=-1)  # (H, W, 4)
    else:
        img = rgb  # (H, W, 3)

    # Resize
    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # Scale to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Normalize (ImageNet for RGB, keep NIR as-is)
    rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img[..., :3] = (img[..., :3] - rgb_mean) / rgb_std

    # Convert to tensor: (H, W, C) -> (1, C, H, W)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()

    return tensor


def run_inference(
    checkpoint_path: str,
    dataset: str = "weedy_rice",
    rgb_path: str = None,
    nir_path: str = None,
    weedsgalore_id: str = None,
    weedsgalore_root: str = None,
    output_path: str = "output_mask.png",
    target_size: tuple = (960, 1280),
    device: str = "cuda",
    visualize: bool = False
):
    """
    Run inference on an image.

    Args:
        checkpoint_path: Path to trained model checkpoint
        dataset: 'weedy_rice' or 'weedsgalore'
        rgb_path: Path to RGB image (for weedy_rice)
        nir_path: Path to NIR .TIF (for weedy_rice)
        weedsgalore_id: Image ID for WeedsGalore (e.g., '2023-05-25_0109')
        weedsgalore_root: Root path to weedsgalore-dataset
        output_path: Path to save output mask
        target_size: (H, W) for preprocessing
        device: 'cuda' or 'cpu'
        visualize: If True, create a color visualization overlay

    Returns:
        str: Path to saved output mask
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Dataset mode: {dataset}")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    config = ckpt.get("config", {})
    num_classes = config.get("num_classes", 1)
    use_rgbnir = config.get("use_rgbnir", True)
    in_ch = 4 if use_rgbnir else 3
    model_arch = config.get("model", "lightmanet")
    base_ch = config.get("base_ch", 32)

    print(f"[INFO] Model config: {model_arch}, {num_classes} classes, {in_ch} input channels")

    # Load model
    model = create_model(
        architecture=model_arch,
        in_channels=in_ch,
        num_classes=num_classes,
        base_ch=base_ch
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded (epoch {ckpt.get('epoch', 'unknown')})")

    # Load and preprocess image based on dataset
    gt_mask = None  # Ground truth (only for WeedsGalore)

    if dataset == "weedy_rice":
        if rgb_path is None or nir_path is None:
            raise ValueError("For weedy_rice dataset, both --rgb and --nir paths are required")

        print(f"[INFO] Loading Weedy Rice image: {rgb_path}")
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")

        # Use existing preprocessor
        preprocessor = RGBNIRPreprocessor(
            target_size=target_size,
            normalize=True
        )
        x = preprocessor.preprocess(bgr, nir_path, return_tensor=True).to(device)

    elif dataset == "weedsgalore":
        if weedsgalore_id is None or weedsgalore_root is None:
            raise ValueError("For weedsgalore dataset, both --weedsgalore_id and --weedsgalore_root are required")

        print(f"[INFO] Loading WeedsGalore bands for: {weedsgalore_id}")
        r, g, b, nir = load_weedsgalore_bands(weedsgalore_id, weedsgalore_root)

        # Load ground truth mask
        print(f"[INFO] Loading ground truth mask for: {weedsgalore_id}")
        gt_mask = load_weedsgalore_mask(weedsgalore_id, weedsgalore_root, target_size=target_size)

        # Preprocess
        x = preprocess_weedsgalore(r, g, b, nir, target_size=target_size, use_rgbnir=use_rgbnir).to(device)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Inference
    print(f"[INFO] Running inference... (input shape: {x.shape})")
    with torch.no_grad():
        logits = model(x)  # (1, num_classes, H, W)

    # Post-process based on number of classes
    if num_classes == 1:
        # Binary segmentation (Weedy Rice)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = (probs > 0.5).astype(np.uint8) * 255
        pred_mask = (mask > 0).astype(np.uint8)  # 0 or 1

        # Optional visualization
        if visualize:
            vis_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            vis_mask[mask > 0] = [0, 255, 0]  # Green for weed
            vis_output = output_path.replace('.png', '_vis.png')
            cv2.imwrite(vis_output, vis_mask)
            print(f"[INFO] Saved visualization: {vis_output}")

    else:
        # Multi-class segmentation (WeedsGalore)
        preds = torch.argmax(logits, dim=1)[0].cpu().numpy()  # (H, W)
        pred_mask = preds.astype(np.uint8)

        # Save class map (0=background, 1=crop, 2=weed)
        mask = pred_mask.copy()

        # Create visualizations
        if visualize:
            # 1. Colored prediction
            vis_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            vis_mask[mask == 0] = [0, 0, 0]        # Black for background
            vis_mask[mask == 1] = [0, 255, 0]      # Green for crop
            vis_mask[mask == 2] = [255, 0, 0]      # Red for weed
            vis_output = output_path.replace('.png', '_vis.png')
            cv2.imwrite(vis_output, vis_mask)
            print(f"[INFO] Saved prediction visualization: {vis_output}")

        # WeedsGalore-specific: comparison with ground truth
        if gt_mask is not None:
            print(f"\n[INFO] Comparing prediction with ground truth...")

            # Compute metrics
            metrics = compute_metrics(pred_mask, gt_mask, num_classes=3)

            print(f"\n{'='*70}")
            print("SEGMENTATION METRICS")
            print(f"{'='*70}")
            print(f"Overall Accuracy:     {metrics['accuracy']*100:.2f}%")
            print(f"Mean IoU:             {metrics['mean_iou']*100:.2f}%")
            print(f"\nPer-Class IoU:")
            print(f"  - Background:       {metrics['iou_Background']*100:.2f}%")
            print(f"  - Crop:             {metrics['iou_Crop']*100:.2f}%")
            print(f"  - Weed:             {metrics['iou_Weed']*100:.2f}%")
            print(f"\nPixel Counts:")
            print(f"  - Background (GT/Pred): {metrics['Background_gt_pixels']:,} / {metrics['Background_pred_pixels']:,}")
            print(f"  - Crop (GT/Pred):       {metrics['Crop_gt_pixels']:,} / {metrics['Crop_pred_pixels']:,}")
            print(f"  - Weed (GT/Pred):       {metrics['Weed_gt_pixels']:,} / {metrics['Weed_pred_pixels']:,}")
            print(f"{'='*70}\n")

            # Save comparison visualization
            comparison_vis = create_comparison_visualization(pred_mask, gt_mask)
            comparison_output = output_path.replace('.png', '_comparison.png')
            cv2.imwrite(comparison_output, comparison_vis)
            print(f"[INFO] Saved comparison visualization: {comparison_output}")
            print(f"       (Green=Correct, Red=Incorrect)")

            # Save side-by-side visualization
            sidebyside_vis = create_side_by_side_visualization(pred_mask, gt_mask)
            sidebyside_output = output_path.replace('.png', '_sidebyside.png')
            cv2.imwrite(sidebyside_output, sidebyside_vis)
            print(f"[INFO] Saved side-by-side visualization: {sidebyside_output}")

        # Also save as grayscale with scaled values for visibility
        mask_vis = (mask * 127).astype(np.uint8)  # 0->0, 1->127, 2->254
        mask = mask_vis

    # Save output
    cv2.imwrite(output_path, mask)
    print(f"[INFO] Saved output mask: {output_path}")

    # Print class distribution for prediction
    if num_classes == 1:
        weed_pixels = np.sum(pred_mask > 0)
        total_pixels = pred_mask.size
        print(f"[INFO] Weed pixels: {weed_pixels}/{total_pixels} ({weed_pixels/total_pixels*100:.2f}%)")
    elif gt_mask is None:
        # Only print if we haven't already printed metrics above
        unique, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = pred_mask.size
        print(f"[INFO] Predicted class distribution:")
        class_names = ["Background", "Crop", "Weed"]
        for cls, count in zip(unique, counts):
            cls_name = class_names[cls] if cls < len(class_names) else f"Class{cls}"
            print(f"  - {cls_name}: {count}/{total_pixels} ({count/total_pixels*100:.2f}%)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on agricultural weed segmentation models"
    )

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["weedy_rice", "weedsgalore"],
                       help="Dataset type")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save output mask")

    # Weedy Rice specific
    parser.add_argument("--rgb", type=str, default=None,
                       help="Path to RGB image (for weedy_rice)")
    parser.add_argument("--nir", type=str, default=None,
                       help="Path to NIR .TIF (for weedy_rice)")

    # WeedsGalore specific
    parser.add_argument("--weedsgalore_root", type=str, default="/home/vjti-comp/Downloads/weedsgalore-dataset",
                       help="Root path to weedsgalore-dataset (for weedsgalore)")
    parser.add_argument("--weedsgalore_id", type=str, default='2023-05-25_0109',
                       help="Image ID, e.g., (for weedsgalore)")

    # Common arguments
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    parser.add_argument("--visualize", action="store_true",
                       help="Create color visualization overlay")

    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        rgb_path=args.rgb,
        nir_path=args.nir,
        weedsgalore_id=args.weedsgalore_id,
        weedsgalore_root=args.weedsgalore_root,
        output_path=args.output,
        target_size=(args.height, args.width),
        device=args.device,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()