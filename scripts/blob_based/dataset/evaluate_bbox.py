import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from blob_based.dataset.config import *
from blob_based.dataset.ndvi import compute_ndvi, ndvi_threshold

IMG_EXT = '.png'

def evaluate_binary_segmentation(split_file):
    """
    Evaluate NDVI-based binary segmentation vs ground truth masks
    
    GT mask values:
    - 0: background (soil)
    - 1: crop
    - 2: weed
    
    Our binary prediction:
    - 0: background
    - 1: vegetation (crop OR weed)
    """
    with open(split_file) as f:
        image_ids = [l.strip() for l in f.readlines()]

    all_gt = []
    all_pred = []
    
    metrics = {
        'tp': 0,  # True positives (veg predicted as veg)
        'tn': 0,  # True negatives (bg predicted as bg)
        'fp': 0,  # False positives (bg predicted as veg)
        'fn': 0   # False negatives (veg predicted as bg)
    }

    print(f"\nEvaluating binary segmentation on {len(image_ids)} images")
    print(f"NDVI Threshold: {NDVI_THRESH}\n")

    for img_id in tqdm(image_ids, desc="Processing"):
        rgb_path = os.path.join(RGB_DIR, 'rgb_' + img_id + IMG_EXT)
        nir_path = os.path.join(NIR_DIR, 'nir_' + img_id + IMG_EXT)
        mask_path = os.path.join(MASK_DIR, 'mask_' + img_id + IMG_EXT)

        rgb = cv2.imread(rgb_path)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        if nir is None:
            continue

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        # Generate binary prediction using NDVI
        ndvi = compute_ndvi(rgb, nir)
        pred_mask = ndvi_threshold(ndvi, NDVI_THRESH)

        # Convert GT mask to binary (0=background, 1=vegetation)
        gt_binary = (gt_mask > 0).astype(np.uint8)

        # Flatten for pixel-wise comparison
        gt_flat = gt_binary.flatten()
        pred_flat = pred_mask.flatten()

        all_gt.extend(gt_flat)
        all_pred.extend(pred_flat)

        # Update metrics
        metrics['tp'] += np.sum((pred_flat == 1) & (gt_flat == 1))
        metrics['tn'] += np.sum((pred_flat == 0) & (gt_flat == 0))
        metrics['fp'] += np.sum((pred_flat == 1) & (gt_flat == 0))
        metrics['fn'] += np.sum((pred_flat == 0) & (gt_flat == 1))

    # Calculate metrics
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-8)
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (metrics['tp'] + metrics['tn']) / sum(metrics.values())
    iou = metrics['tp'] / (metrics['tp'] + metrics['fp'] + metrics['fn'] + 1e-8)

    print("\n" + "="*50)
    print("BINARY SEGMENTATION EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TP (veg→veg): {metrics['tp']:,}")
    print(f"  TN (bg→bg):   {metrics['tn']:,}")
    print(f"  FP (bg→veg):  {metrics['fp']:,}")
    print(f"  FN (veg→bg):  {metrics['fn']:,}")
    print("="*50 + "\n")

    # Detailed classification report
    print("Detailed Classification Report:")
    print(classification_report(all_gt, all_pred, 
                                target_names=['Background', 'Vegetation'],
                                digits=4))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'metrics': metrics
    }


def visualize_segmentation_comparison(split_file, num_samples=5, output_dir="analysis/seg_comparison"):
    """
    Visualize side-by-side comparison of GT mask vs NDVI prediction
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(split_file) as f:
        image_ids = [l.strip() for l in f.readlines()]

    import random
    samples = random.sample(image_ids, min(num_samples, len(image_ids)))

    print(f"\nGenerating {len(samples)} comparison visualizations")

    for img_id in samples:
        rgb_path = os.path.join(RGB_DIR, 'rgb_' + img_id + IMG_EXT)
        nir_path = os.path.join(NIR_DIR, 'nir_' + img_id + IMG_EXT)
        mask_path = os.path.join(MASK_DIR, 'mask_' + img_id + IMG_EXT)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Generate prediction
        ndvi = compute_ndvi(rgb, nir)
        pred_mask = ndvi_threshold(ndvi, NDVI_THRESH)
        gt_binary = (gt_mask > 0).astype(np.uint8)

        # Create colored overlays
        gt_overlay = rgb.copy()
        pred_overlay = rgb.copy()

        gt_overlay[gt_binary == 1] = [0, 255, 0]  # Green for vegetation
        pred_overlay[pred_mask == 1] = [0, 255, 0]

        # Stack horizontally: RGB | GT | Prediction
        comparison = np.hstack([rgb, gt_overlay, pred_overlay])

        # Add labels
        cv2.putText(comparison, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Ground Truth", (rgb.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "NDVI Prediction", (2*rgb.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        output_path = os.path.join(output_dir, f"seg_comp_{img_id}.png")
        cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")

    print(f"\nComparison images saved to: {output_dir}/\n")


if __name__ == "__main__":
    os.makedirs("analysis", exist_ok=True)

    print("\n" + "="*60)
    print("EVALUATING BINARY SEGMENTATION QUALITY")
    print("="*60)

    # Evaluate on train set
    print("\n### TRAIN SET ###")
    train_results = evaluate_binary_segmentation(f"{SPLIT_DIR}/train.txt")

    # Evaluate on val set
    print("\n### VALIDATION SET ###")
    val_results = evaluate_binary_segmentation(f"{SPLIT_DIR}/val.txt")

    # Generate visualizations
    visualize_segmentation_comparison(f"{SPLIT_DIR}/val.txt", 
                                     num_samples=10,
                                     output_dir="analysis/seg_comparison")

    print("\n✅ Evaluation complete!")
    print("Next steps:")
    print("1. Check IoU score (>0.8 is good for vegetation detection)")
    print("2. If IoU is low, tune NDVI_THRESH in config.py")
    print("3. View visualizations in analysis/seg_comparison/")