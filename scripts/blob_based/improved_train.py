import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import os
import json
from datetime import datetime

from blob_based.dataset.config import *
from blob_based.dataset.fast_bbox_dataset import FastBboxDataset
from blob_based.models.improved_model import ImprovedBboxCNN, ResNetBboxClassifier


class MetricsTracker:
    """Track and log training metrics"""
    def __init__(self, num_classes=2, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.total_loss = 0.0
        self.num_batches = 0
    
    def update(self, preds, labels, loss):
        """Update metrics with batch results"""
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        self.total_loss += loss
        self.num_batches += 1
    
    def compute(self):
        """Compute all metrics"""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        metrics = {
            'loss': self.total_loss / self.num_batches,
            'accuracy': accuracy_score(labels, preds),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        }
        
        # Per-class metrics (IMPORTANT for imbalanced datasets)
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        
        for i, name in enumerate(self.class_names):
            metrics[f'precision_{name}'] = precision_per_class[i]
            metrics[f'recall_{name}'] = recall_per_class[i]
            metrics[f'f1_{name}'] = f1_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def print_metrics(self, metrics, prefix=""):
        """Pretty print metrics"""
        print(f"\n{prefix} Metrics:")
        print(f"  Loss:     {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"\n  Per-class metrics:")
        for name in self.class_names:
            print(f"    {name:8s} - Precision: {metrics[f'precision_{name}']:.4f} | "
                  f"Recall: {metrics[f'recall_{name}']:.4f} | "
                  f"F1: {metrics[f'f1_{name}']:.4f}")
        
        print(f"\n  Confusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"              Predicted")
        print(f"              {self.class_names[0]:8s} {self.class_names[1]:8s}")
        for i, name in enumerate(self.class_names):
            print(f"  Actual {name:8s} {cm[i,0]:8d} {cm[i,1]:8d}")
        print()


def calculate_class_weights(dataset):
    """
    Calculate class weights for imbalanced datasets
    CRITICAL for crop/weed classification
    """
    labels = [sample['label'] for sample in dataset.samples]
    unique, counts = np.unique(labels, return_counts=True)
    
    total = len(labels)
    weights = total / (len(unique) * counts)
    
    print(f"\nClass distribution:")
    for cls, count, weight in zip(unique, counts, weights):
        class_name = "crop" if cls == 0 else "weed"
        print(f"  {class_name}: {count} samples ({count/total*100:.1f}%) - weight: {weight:.3f}")
    
    return torch.FloatTensor(weights)


def train_epoch(model, loader, criterion, optimizer, device, metrics_tracker):
    """Train for one epoch"""
    model.train()
    metrics_tracker.reset()
    
    loop = tqdm(loader, desc="Training", leave=False)
    for imgs, labels in loop:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Track metrics
        _, preds = torch.max(outputs, 1)
        metrics_tracker.update(preds, labels, loss.item())
        
        loop.set_postfix(loss=loss.item())
    
    return metrics_tracker.compute()


def validate(model, loader, criterion, device, metrics_tracker):
    """Validate model"""
    model.eval()
    metrics_tracker.reset()
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            metrics_tracker.update(preds, labels, loss.item())
    
    return metrics_tracker.compute()


def main():
    print("\n" + "="*70)
    print("BBOX-BASED CROP/WEED CLASSIFICATION TRAINING")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/bbox_training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("\n[1/4] Loading datasets...")
    train_ds = FastBboxDataset('analysis/train_bboxes_multi_index_threshold.json')
    val_ds = FastBboxDataset('analysis/train_bboxes_multi_index_threshold.json')
    
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_ds)
    class_weights = class_weights.to(DEVICE)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    # Choose architecture (ResNet is generally better for small datasets)
    model = ResNetBboxClassifier(input_channels=4, num_classes=NUM_CLASSES).to(DEVICE)
    # Alternative: model = ImprovedBboxCNN(input_channels=4, num_classes=NUM_CLASSES).to(DEVICE)
    
    print(f"  Model: ResNetBboxClassifier")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Metrics tracker
    class_names = ['crop', 'weed']
    train_tracker = MetricsTracker(num_classes=NUM_CLASSES, class_names=class_names)
    val_tracker = MetricsTracker(num_classes=NUM_CLASSES, class_names=class_names)
    
    # Training history
    history = {
        'train': [],
        'val': []
    }
    
    best_f1 = 0.0
    best_epoch = 0
    
    print("\n[3/4] Training...")
    print("="*70)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-"*70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                    DEVICE, train_tracker)
        train_tracker.print_metrics(train_metrics, prefix="Train")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, DEVICE, val_tracker)
        val_tracker.print_metrics(val_metrics, prefix="Validation")
        
        # Learning rate scheduling based on validation F1
        scheduler.step(val_metrics['f1_macro'])
        
        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best model (based on F1 score - most important metric for imbalanced data)
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(output_dir, 'best_model.pth'))
            
            print(f"  ✓ Best model saved! (F1: {best_f1:.4f})")
    
    print("\n[4/4] Training complete!")
    print("="*70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Models saved to: {output_dir}/")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            'train': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                      for k, v in m.items()} for m in history['train']],
            'val': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                    for k, v in m.items()} for m in history['val']]
        }
        json.dump(history_serializable, f, indent=2)
    
    print(f"Training history saved to: {output_dir}/history.json")
    print("\n" + "="*70)
    print("IMPORTANT METRICS FOR CROP/WEED CLASSIFICATION:")
    print("="*70)
    print(f"1. F1 Score (macro): {best_f1:.4f} - MAIN METRIC")
    print(f"2. Weed Recall: {history['val'][best_epoch-1]['recall_weed']:.4f} - Don't miss weeds!")
    print(f"3. Crop Precision: {history['val'][best_epoch-1]['precision_crop']:.4f} - Don't kill crops!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()