import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_dataset_distribution(root_path, dataset_name="Dataset"):
    mask_path = Path(root_path) / "masks"
    
    # Initialize lists to store percentages
    bg_percents = []
    crop_percents = []
    weed_percents = []

    mask_files = [f for f in os.listdir(mask_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    if not mask_files:
        print(f"No masks found in {mask_path}")
        return

    print(f"Analyzing {len(mask_files)} images in {dataset_name}...")

    for filename in mask_files:
        mask = cv2.imread(str(mask_path / filename), cv2.IMREAD_GRAYSCALE)
        
        total_pixels = mask.size
        # Count occurrences of each class
        bg_percents.append((np.sum(mask == 0) / total_pixels) * 100)
        crop_percents.append((np.sum(mask == 1) / total_pixels) * 100)
        weed_percents.append((np.sum(mask == 2) / total_pixels) * 100)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(bg_percents, bins=30, alpha=0.5, label='Background (0)', color='gray')
    plt.hist(crop_percents, bins=30, alpha=0.5, label='Crop (1)', color='green')
    plt.hist(weed_percents, bins=30, alpha=0.5, label='Weed (2)', color='red')
    
    plt.title(f'Pixel Percentage Distribution per Image - {dataset_name}')
    plt.xlabel('Percentage of Image Area (%)')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# Usage:
analyze_dataset_distribution('/home/vjti-comp/Downloads/FINAL_SUGARBEETS_DATASET', "Original Dataset")
analyze_dataset_distribution('/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET', "Augmented Dataset Subset")