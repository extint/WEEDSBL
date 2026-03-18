import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

from blob_based.dataset.config import *
from blob_based.dataset.bbox_dataset import BboxDataset
from blob_based.models.improved_model import ResNetBboxClassifier
from blob_based.dataset.prepare_blobs import prepare_blob_samples


def evaluate_model(model, loader, device, class_names=['crop', 'weed']):
    """Bbox-level classification evaluation"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating bboxes"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(labels, preds),
        'precision_macro': precision_score(labels, preds, average='macro'),
        'recall_macro': recall_score(labels, preds, average='macro'),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'precision_weighted': precision_score(labels, preds, average='weighted'),
        'recall_weighted': recall_score(labels, preds, average='weighted'),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(labels, preds, average=None)
    recall_per_class = recall_score(labels, preds, average=None)
    f1_per_class = f1_score(labels, preds, average=None)
    
    for i, name in enumerate(class_names):
        results[f'precision_{name}'] = precision_per_class[i]
        results[f'recall_{name}'] = recall_per_class[i]
        results[f'f1_{name}'] = f1_per_class[i]
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    results['confusion_matrix'] = cm
    
    results['classification_report'] = classification_report(
        labels, preds, target_names=class_names, digits=4
    )
    
    results['preds'] = preds
    results['labels'] = labels
    results['probs'] = probs
    
    return results


def evaluate_pixel_metrics(model, dataset, device, class_names=['crop', 'weed'], batch_size=64):
    """
    FAST pixel-wise evaluation with BATCHED blob processing
    USES EXACT SAME PIPELINE AS TRAINING but processes blobs in batches
    """
    from blob_based.dataset.ndvi import compute_ndvi, ndvi_threshold
    from blob_based.dataset.blob_extraction import extract_blobs
    
    model.eval()
    
    # Group bboxes by image
    image_groups = {}
    for sample in dataset.samples:
        img_id = sample['img_id']
        if img_id not in image_groups:
            image_groups[img_id] = []
        image_groups[img_id].append(sample)
    
    print(f"\nCalculating pixel-wise metrics on {len(image_groups)} images...")
    print(f"Using BATCHED processing (batch_size={batch_size}) for speed")
    
    iou_per_class = {0: [], 1: []}
    pixel_acc_per_class = {0: [], 1: []}
    
    for img_id in tqdm(list(image_groups.keys()), desc="Processing images"):
        rgb_path = os.path.join(RGB_DIR, f'rgb_{img_id}.png')
        nir_path = os.path.join(NIR_DIR, f'nir_{img_id}.png')
        mask_path = os.path.join(MASK_DIR, f'mask_{img_id}.png')
        
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
        
        # Step 1: NDVI and blobs (SAME AS TRAINING)
        ndvi = compute_ndvi(rgb, nir)
        veg_mask = ndvi_threshold(ndvi, NDVI_THRESH)
        blobs = extract_blobs(veg_mask, min_area=MIN_BLOB_AREA)
        
        pred_mask = np.zeros_like(gt_mask)
        
        # Step 2: BATCH PROCESS ALL BLOBS (KEY SPEEDUP)
        blob_inputs = []
        blob_metadata = []
        
        for blob in blobs:
            ys, xs = np.where(blob)
            if len(ys) == 0:
                continue
            
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            
            # Prepare input (SAME AS TRAINING)
            patch = rgb[y1:y2+1, x1:x2+1].copy()
            blob_crop = blob[y1:y2+1, x1:x2+1]
            patch[blob_crop == 0] = 0
            
            patch_resized = cv2.resize(patch, (BLOB_SIZE, BLOB_SIZE))
            nir_crop = cv2.resize(nir[y1:y2+1, x1:x2+1], (BLOB_SIZE, BLOB_SIZE))
            
            img_input = np.dstack([patch_resized, np.expand_dims(nir_crop, 2)])
            img_input = torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0
            
            blob_inputs.append(img_input)
            blob_metadata.append({
                'bbox': (y1, y2, x1, x2),
                'blob_crop': blob_crop
            })
        
        if len(blob_inputs) == 0:
            continue
        
        # Step 3: BATCH INFERENCE (instead of one-by-one)
        all_preds = []
        for i in range(0, len(blob_inputs), batch_size):
            batch = torch.stack(blob_inputs[i:i+batch_size]).to(device)
            
            with torch.no_grad():
                outputs = model(batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        # Step 4: Map predictions to mask
        for pred_class, metadata in zip(all_preds, blob_metadata):
            y1, y2, x1, x2 = metadata['bbox']
            blob_crop = metadata['blob_crop']
            pred_mask[y1:y2+1, x1:x2+1][blob_crop > 0] = pred_class + 1
        
        # Step 5: Calculate metrics
        for class_id in [1, 2]:
            gt_binary = (gt_mask == class_id).astype(np.uint8)
            pred_binary = (pred_mask == class_id).astype(np.uint8)
            
            intersection = np.sum(gt_binary & pred_binary)
            union = np.sum(gt_binary | pred_binary)
            
            if union > 0:
                iou = intersection / union
                iou_per_class[class_id - 1].append(iou)
            
            if np.sum(gt_binary) > 0:
                pixel_acc = intersection / np.sum(gt_binary)
                pixel_acc_per_class[class_id - 1].append(pixel_acc)
    
    # Calculate mean metrics
    seg_metrics = {}
    for i, name in enumerate(class_names):
        seg_metrics[f'iou_{name}'] = np.mean(iou_per_class[i]) if iou_per_class[i] else 0.0
        seg_metrics[f'pixel_acc_{name}'] = np.mean(pixel_acc_per_class[i]) if pixel_acc_per_class[i] else 0.0
    
    seg_metrics['mean_iou'] = np.mean([seg_metrics[f'iou_{name}'] for name in class_names])
    seg_metrics['mean_pixel_acc'] = np.mean([seg_metrics[f'pixel_acc_{name}'] for name in class_names])
    
    return seg_metrics


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Bbox Level)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc_curve(labels, probs, class_names, output_path):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    
    for i, name in enumerate(class_names):
        y_true = (labels == i).astype(int)
        y_score = probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def visualize_predictions(dataset, model, device, num_samples=20, output_dir="evaluation/predictions"):
    """Visualize bbox predictions"""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    class_names = ['crop', 'weed']
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Generating visualizations"):
        img, true_label = dataset[idx]
        
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_label = torch.argmax(output, dim=1).item()
            confidence = probs[0, pred_label].item()
        
        # Convert to displayable image (RGB only)
        img_np = img[:3].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_display = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        color = (0, 255, 0) if true_label == pred_label else (0, 0, 255)
        
        cv2.putText(img_display, f"True: {true_name}", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_display, f"Pred: {pred_name} ({confidence:.2f})", (5, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        filename = f"{'correct' if true_label == pred_label else 'wrong'}_{idx:04d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), img_display)
    
    print(f"Saved {len(indices)} visualizations to: {output_dir}/")


def main():
    print("\n" + "="*70)
    print("BBOX CLASSIFICATION + PIXEL-WISE EVALUATION (OPTIMIZED)")
    print("="*70)
    
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    print(f"\nGPU Setup:")
    print(f"  Available GPUs: {num_gpus}")
    print(f"  Using device: {DEVICE}")
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_ds = BboxDataset(f"{SPLIT_DIR}/val.txt")
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE * 2,  # Larger batch for eval
        shuffle=False, 
        num_workers=8,  # More workers
        pin_memory=True
    )
    print(f"Validation samples: {len(val_ds)}")
    
    # Load model
    print("\nLoading model...")
    runs_dir = "runs"
    if os.path.exists(runs_dir):
        run_dirs = [d for d in os.listdir(runs_dir) if d.startswith("bbox_training_")]
        if run_dirs:
            latest_run = sorted(run_dirs)[-1]
            model_path="/home/vjtiadmin/Desktop/BTechGroup/WEEDSBL/scripts/blob_based/blob_cnn.pth"
            # model_path = os.path.join(runs_dir, latest_run, "best_model.pth")
            print(f"Using: {model_path}")
        else:
            print("Error: No trained models found")
            return
    else:
        print("Error: runs/ directory not found")
        return
    
    # Initialize model
    model = ResNetBboxClassifier(input_channels=4, num_classes=NUM_CLASSES)
    
    # Multi-GPU support
    if num_gpus > 1:
        print(f"Enabling DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel state dict
    if num_gpus > 1 and not list(state_dict.keys())[0].startswith('module.'):
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif num_gpus == 1 and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Evaluate bbox classification
    print("\n" + "="*70)
    print("STEP 1: BBOX-LEVEL EVALUATION")
    print("="*70)
    results = evaluate_model(model, val_loader, DEVICE)
    
    print(f"\nBbox Classification Metrics:")
    print(f"  Accuracy:        {results['accuracy']:.4f}")
    print(f"  F1 (macro):      {results['f1_macro']:.4f}")
    
    class_names = ['crop', 'weed']
    print(f"\nPer-Class Bbox Metrics:")
    for name in class_names:
        print(f"  {name.upper()}:")
        print(f"    Precision: {results[f'precision_{name}']:.4f}")
        print(f"    Recall:    {results[f'recall_{name}']:.4f}")
        print(f"    F1:        {results[f'f1_{name}']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Evaluate pixel-wise metrics (FAST VERSION)
    print("\n" + "="*70)
    print("STEP 2: PIXEL-WISE EVALUATION (BATCHED - FAST)")
    print("="*70)
    
    # Unwrap model if using DataParallel
    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    seg_metrics = evaluate_pixel_metrics(eval_model, val_ds, DEVICE, batch_size=64)
    
    print(f"\nPixel-wise Segmentation Metrics:")
    print(f"  Mean IoU:        {seg_metrics['mean_iou']:.4f}")
    print(f"  Mean Pixel Acc:  {seg_metrics['mean_pixel_acc']:.4f}")
    
    print(f"\nPer-Class Pixel Metrics:")
    for name in class_names:
        print(f"  {name.upper()}:")
        print(f"    IoU:        {seg_metrics[f'iou_{name}']:.4f}")
        print(f"    Pixel Acc:  {seg_metrics[f'pixel_acc_{name}']:.4f}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("="*70)
    eval_dir = "evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    plot_confusion_matrix(results['confusion_matrix'], class_names, 
                         os.path.join(eval_dir, 'confusion_matrix.png'))
    
    plot_roc_curve(results['labels'], results['probs'], class_names,
                  os.path.join(eval_dir, 'roc_curve.png'))
    
    visualize_predictions(val_ds, eval_model, DEVICE, num_samples=50,
                         output_dir=os.path.join(eval_dir, 'predictions'))
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n1. BBOX ACCURACY: {results['accuracy']:.4f}")
    print(f"   → {results['accuracy']*100:.1f}% of plant bboxes classified correctly")
    
    print(f"\n2. WEED RECALL (Bbox): {results['recall_weed']:.4f}")
    print(f"   → Detecting {results['recall_weed']*100:.1f}% of weeds (Target: >90%)")
    
    print(f"\n3. CROP PRECISION (Bbox): {results['precision_crop']:.4f}")
    print(f"   → {results['precision_crop']*100:.1f}% confidence when labeling crops (Target: >95%)")
    
    print(f"\n4. PIXEL IoU: {seg_metrics['mean_iou']:.4f}")
    print(f"   → End-to-end segmentation quality")
    print(f"   → Formula: NDVI quality × Bbox F1")
    
    print("\n" + "="*70)
    print(f"Evaluation complete! Check {eval_dir}/ for visualizations")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()