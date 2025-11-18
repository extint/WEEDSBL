
"""
Complete example usage of RGB-NIR Sugar Beet Architecture with Sugar Beet 2016 dataset
This file shows how to load and preprocess the Sugar Beet 2016 dataset and train the model
"""

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import our implementations
from rice_weed_data_loader import (
    create_weedy_rice_dataloaders
)

from rgb_nir_training_pipeline import RGBNIRDomainAdaptationTrainer



def main():
    """Main training function"""
    # Configuration
    config = {
        'data_root': '/path/to/sugar_beet_2016_dataset',
        'batch_size': 4,  # Reduce if GPU memory is limited
        'num_workers': 4,
        'num_epochs': 100,
        'lr_seg': 0.001,
        'lr_disc': 1e-4,
        'pretrained_rgb_path': None,  # Path to RGB ConvNeXt weights if available
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("RGB-NIR Sugar Beet 2016 Domain Adaptation Training")
    print("=" * 55)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print()

    # Create data loaders
    print("Loading Sugar Beet 2016 dataset...")
    source_loader, target_loader, val_loader, test_loader = create_sugar_beet_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    print(f"Source training samples: {len(source_loader.dataset)}")
    print(f"Target training samples: {len(target_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()

    # Initialize model
    print("Initializing RGB-NIR model...")
    model = RGBNIRSugarBeetSegmentationNetwork(num_classes=2)

    # Initialize weights
    if config['pretrained_rgb_path'] and os.path.exists(config['pretrained_rgb_path']):
        initialize_rgb_nir_weights(model, config['pretrained_rgb_path'])
    else:
        print("No pretrained weights provided, using random initialization")

    # Initialize trainer
    device = torch.device(config['device'])
    trainer = RGBNIRDomainAdaptationTrainer(
        model=model,
        device=device,
        lr_seg=config['lr_seg'],
        lr_disc=config['lr_disc']
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: ~{sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.1f} MB")
    print()

    # Start training
    trainer.train(
        source_loader=source_loader,
        target_loader=target_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        log_interval=10
    )

    print("Training completed!")

    # Optional: Test evaluation
    print("\nEvaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    print(f"Test mIoU: {test_metrics['miou']:.4f}")
    print(f"Test IoU per class: {test_metrics['iou_per_class']}")

if __name__ == "__main__":
    # Example: Compute dataset statistics first (run once)
    def compute_dataset_statistics(data_root: str):
        """Compute RGB and NIR channel statistics from Sugar Beet 2016 dataset"""
        print("Computing dataset statistics...")

        # This is a simplified example - you should compute from your actual dataset
        rgb_means = []
        rgb_stds = []
        nir_means = []
        nir_stds = []

        # Load a sample of images and compute statistics
        sample_dataset = SugarBeet2016Dataset(data_root, split='train', augment=False)

        for i in range(min(1000, len(sample_dataset))):  # Sample 1000 images
            sample = sample_dataset[i]
            rgbnir = sample['images'].numpy()

            # RGB channels (0, 1, 2)
            rgb = rgbnir[:3]
            rgb_means.append(rgb.mean(axis=(1, 2)))
            rgb_stds.append(rgb.std(axis=(1, 2)))

            # NIR channel (3)
            nir = rgbnir[3:4]
            nir_means.append(nir.mean())
            nir_stds.append(nir.std())

        rgb_mean = np.array(rgb_means).mean(axis=0)
        rgb_std = np.array(rgb_stds).mean(axis=0)
        nir_mean = np.array(nir_means).mean()
        nir_std = np.array(nir_stds).mean()

        print(f"RGB mean: {rgb_mean}")
        print(f"RGB std: {rgb_std}")
        print(f"NIR mean: {nir_mean}")
        print(f"NIR std: {nir_std}")

        return rgb_mean, rgb_std, nir_mean, nir_std

    # Uncomment to run training
    # main()

    print("\nExample usage file created successfully!")
    print("Modify the data paths and run main() to start training.")
