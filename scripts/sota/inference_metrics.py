import os
import cv2
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from glob import glob
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
NUM_CLASSES = 3
TARGET_SIZE = (966, 1296)  # (H, W)
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Color map for visualization (BGR for OpenCV)
COLOR_MAP = {
    0: (0, 0, 0),       # background - black
    1: (0, 255, 0),     # crop - green
    2: (0, 0, 255),     # weed - red
}

# -------------------------
# IMAGE IO
# -------------------------
def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_nir(path: str) -> np.ndarray:
    nir = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if nir is None:
        raise FileNotFoundError(path)
    if nir.ndim == 3:
        nir = nir[:, :, 0]
    return nir


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask.astype(np.uint8)


# -------------------------
# PREPROCESSING
# -------------------------
def scale_to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32) / 255.0


def preprocess(
    rgb_path: str,
    nir_path: str,
    target_size: Tuple[int, int]
) -> torch.Tensor:
    rgb = read_rgb(rgb_path)
    nir = read_nir(nir_path)

    if nir.shape[:2] != rgb.shape[:2]:
        nir = cv2.resize(nir, (rgb.shape[1], rgb.shape[0]),
                         interpolation=cv2.INTER_LINEAR)

    img = np.concatenate([rgb, nir[..., None]], axis=-1)

    img = cv2.resize(
        img,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_LINEAR
    )

    img = scale_to_float(img)

    # Normalize RGB only
    img[..., :3] = (img[..., :3] - RGB_MEAN) / RGB_STD

    # HWC → CHW
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return img.unsqueeze(0)  # (1, C, H, W)


# -------------------------
# EVALUATION METRICS
# -------------------------
def compute_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 3) -> Dict[str, float]:
    """
    Compute pixel accuracy, IoU, precision, recall, and F1 score for each class.
    
    Args:
        pred: Predicted mask (H, W)
        gt: Ground truth mask (H, W)
        num_classes: Number of classes
    
    Returns:
        Dictionary containing metrics
    """
    metrics = {}
    
    # Pixel Accuracy
    correct = np.sum(pred == gt)
    total = pred.size
    pixel_acc = correct / total
    metrics['pixel_accuracy'] = pixel_acc
    
    # Per-class metrics
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)
        
        # True Positives, False Positives, False Negatives
        tp = np.sum(pred_cls & gt_cls)
        fp = np.sum(pred_cls & ~gt_cls)
        fn = np.sum(~pred_cls & gt_cls)
        
        # IoU
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Store metrics
        class_names = {0: 'background', 1: 'crop', 2: 'weed'}
        prefix = class_names.get(cls, f'class_{cls}')
        
        metrics[f'{prefix}_iou'] = iou
        metrics[f'{prefix}_precision'] = precision
        metrics[f'{prefix}_recall'] = recall
        metrics[f'{prefix}_f1'] = f1
    
    return metrics


def print_metrics(metrics: Dict[str, float], image_name: str = None):
    """Print metrics in a formatted way."""
    if image_name:
        print(f"\n{'='*60}")
        print(f"Image: {image_name}")
    print("="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\nOverall Pixel Accuracy: {metrics['pixel_accuracy']:.4f} ({metrics['pixel_accuracy']*100:.2f}%)")
    
    print("\n" + "-"*60)
    print("CROP METRICS:")
    print("-"*60)
    print(f"  IoU       : {metrics['crop_iou']:.4f}")
    print(f"  Precision : {metrics['crop_precision']:.4f}")
    print(f"  Recall    : {metrics['crop_recall']:.4f}")
    print(f"  F1 Score  : {metrics['crop_f1']:.4f}")
    
    print("\n" + "-"*60)
    print("WEED METRICS:")
    print("-"*60)
    print(f"  IoU       : {metrics['weed_iou']:.4f}")
    print(f"  Precision : {metrics['weed_precision']:.4f}")
    print(f"  Recall    : {metrics['weed_recall']:.4f}")
    print(f"  F1 Score  : {metrics['weed_f1']:.4f}")
    print("="*60 + "\n")


# -------------------------
# VISUALIZATION
# -------------------------
def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        out[mask == cls] = color
    return out


def save_comparison(
    rgb_path: str,
    gt_mask_path: str,
    pred_mask: np.ndarray,
    out_path: str
):
    rgb = cv2.imread(rgb_path)
    rgb = cv2.resize(rgb, (pred_mask.shape[1], pred_mask.shape[0]))

    gt = read_mask(gt_mask_path)
    gt = cv2.resize(gt, (pred_mask.shape[1], pred_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST)

    gt_color = colorize_mask(gt)
    pred_color = colorize_mask(pred_mask)

    combined = np.hstack([rgb, gt_color, pred_color])
    cv2.imwrite(out_path, combined)


# -------------------------
# DATASET PREPARATION
# -------------------------
def prepare_dataset(rgb_dir: str, nir_dir: str, mask_dir: str) -> List[Tuple[str, str, str]]:
    """
    Prepare dataset by matching RGB, NIR, and mask files.
    
    Returns:
        List of tuples (rgb_path, nir_path, mask_path)
    """
    rgb_files = sorted(glob(os.path.join(rgb_dir, "*.png")))
    
    dataset = []
    for rgb_path in rgb_files:
        base_name = os.path.basename(rgb_path)
        
        # Assuming naming convention: rgb_*, nir_*, mask_*
        nir_name = base_name.replace("rgb_", "nir_")
        mask_name = base_name.replace("rgb_", "mask_")
        
        nir_path = os.path.join(nir_dir, nir_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        if os.path.exists(nir_path) and os.path.exists(mask_path):
            dataset.append((rgb_path, nir_path, mask_path))
        else:
            print(f"Warning: Missing NIR or mask for {base_name}")
    
    return dataset


# -------------------------
# BATCH INFERENCE
# -------------------------
@torch.no_grad()
def run_batch_inference(
    model,
    checkpoint_path: str,
    rgb_dir: str,
    nir_dir: str,
    mask_dir: str,
    output_dir: str,
    csv_path: str,
    device: str = "cuda",
    save_visualizations: bool = True
):
    """
    Run inference on entire dataset and save metrics to CSV.
    
    Args:
        model: Segmentation model
        checkpoint_path: Path to model checkpoint
        rgb_dir: Directory containing RGB images
        nir_dir: Directory containing NIR images
        mask_dir: Directory containing ground truth masks
        output_dir: Directory to save outputs
        csv_path: Path to save CSV file with metrics
        device: Device to run inference on
        save_visualizations: Whether to save visualization images
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if save_visualizations:
        viz_dir = os.path.join(output_dir, "visualizations")
        pred_dir = os.path.join(output_dir, "predictions")
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=False, map_location=device)["model_state_dict"]
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(rgb_dir, nir_dir, mask_dir)
    print(f"Found {len(dataset)} samples\n")
    
    if len(dataset) == 0:
        print("No samples found! Please check your data directories.")
        return
    
    # Store results
    results = []
    
    # Process each sample
    print("Running inference...")
    for rgb_path, nir_path, mask_path in tqdm(dataset, desc="Processing images"):
        try:
            # Get image name
            base_name = os.path.splitext(os.path.basename(rgb_path))[0]
            image_name = base_name.replace("rgb_", "")
            
            # Preprocess and run inference
            x = preprocess(rgb_path, nir_path, TARGET_SIZE).to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            
            # Load and resize ground truth
            gt = read_mask(mask_path)
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
            
            # Compute metrics
            metrics = compute_metrics(pred, gt, NUM_CLASSES)
            
            # Store results for CSV
            result_row = {
                'image_name': image_name,
                'pixel_accuracy': metrics['pixel_accuracy'],
                'crop_iou': metrics['crop_iou'],
                'crop_precision': metrics['crop_precision'],
                'crop_recall': metrics['crop_recall'],
                'crop_f1': metrics['crop_f1'],
                'weed_iou': metrics['weed_iou'],
                'weed_precision': metrics['weed_precision'],
                'weed_recall': metrics['weed_recall'],
                'weed_f1': metrics['weed_f1']
            }
            results.append(result_row)
            
            # Save visualizations if requested
            if save_visualizations:
                pred_color = colorize_mask(pred)
                pred_path = os.path.join(pred_dir, f"{image_name}_pred.png")
                cv2.imwrite(pred_path, pred_color)
                
                viz_path = os.path.join(viz_dir, f"{image_name}_viz.png")
                save_comparison(rgb_path, mask_path, pred, viz_path)
        
        except Exception as e:
            print(f"\nError processing {base_name}: {str(e)}")
            continue
    
    # Save results to CSV
    print(f"\nSaving results to {csv_path}...")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, float_format='%.4f')
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal samples processed: {len(results)}")
    print(f"\nMean Metrics:")
    print(f"  Pixel Accuracy : {df['pixel_accuracy'].mean():.4f} ± {df['pixel_accuracy'].std():.4f}")
    print(f"\nCROP:")
    print(f"  IoU       : {df['crop_iou'].mean():.4f} ± {df['crop_iou'].std():.4f}")
    print(f"  Precision : {df['crop_precision'].mean():.4f} ± {df['crop_precision'].std():.4f}")
    print(f"  Recall    : {df['crop_recall'].mean():.4f} ± {df['crop_recall'].std():.4f}")
    print(f"  F1 Score  : {df['crop_f1'].mean():.4f} ± {df['crop_f1'].std():.4f}")
    print(f"\nWEED:")
    print(f"  IoU       : {df['weed_iou'].mean():.4f} ± {df['weed_iou'].std():.4f}")
    print(f"  Precision : {df['weed_precision'].mean():.4f} ± {df['weed_precision'].std():.4f}")
    print(f"  Recall    : {df['weed_recall'].mean():.4f} ± {df['weed_recall'].std():.4f}")
    print(f"  F1 Score  : {df['weed_f1'].mean():.4f} ± {df['weed_f1'].std():.4f}")
    print("="*60)
    print(f"\nResults saved to: {csv_path}")
    if save_visualizations:
        print(f"Visualizations saved to: {viz_dir}")
        print(f"Predictions saved to: {pred_dir}")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    from models import create_model
    
    # Create model
    model = create_model("deeplabsv3+", 4, 3, 32)
    
    # Run batch inference
    run_batch_inference(
        model=model,
        checkpoint_path="/home/vjti-comp/WEEDSBL/scripts/sota/experiments/sugarbeets_deeplabsv3+_4ch_RGBNIR_20260118_205155/checkpoints/best_model.pth",
        rgb_dir="/home/vjti-comp/Downloads/FINAL_SUGARBEETS_DATASET/rgb",
        nir_dir="/home/vjti-comp/Downloads/FINAL_SUGARBEETS_DATASET/nir",
        mask_dir="/home/vjti-comp/Downloads/FINAL_SUGARBEETS_DATASET/masks",
        output_dir="./outputs/",
        csv_path="./outputs/evaluation_results.csv",
        device="cuda",
        save_visualizations=True  # Set to False to speed up inference
    )