#!/usr/bin/env python3
"""
Standalone script to analyze semantic segmentation masks.
Works independently - just point it to your dataset root folder.

Analyzes pixel distribution across classes in all semantic masks.
"""

import os
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm


def analyze_mask(mask_path):
    """
    Analyze a single mask and count pixels per class.

    Returns:
        dict: {
            'total_pixels': int,
            'class_0': int,
            'class_1': int, 
            'class_2_plus': int,
            'unique_classes': list
        }
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    total_pixels = mask.size
    unique_values = np.unique(mask)

    class_0_pixels = np.sum(mask == 0)
    class_1_pixels = np.sum(mask == 1)
    class_2_plus_pixels = np.sum(mask >= 2)

    return {
        'total_pixels': total_pixels,
        'class_0': class_0_pixels,
        'class_1': class_1_pixels,
        'class_2_plus': class_2_plus_pixels,
        'unique_classes': unique_values.tolist()
    }


def analyze_all_masks(dataset_root):
    """
    Analyze all semantic masks in the dataset.

    Args:
        dataset_root: Path to weedsgalore-dataset folder

    Returns:
        dict: Aggregated statistics
    """
    dataset_path = Path(dataset_root)

    # Find all semantic mask folders across date directories
    date_folders = ['2023-05-25', '2023-05-30', '2023-06-06', '2023-06-15']

    all_stats = {
        'total_images': 0,
        'total_pixels': 0,
        'class_0_pixels': 0,
        'class_1_pixels': 0,
        'class_2_plus_pixels': 0,
        'all_unique_classes': set()
    }

    per_image_stats = []

    print(f"\n{'='*70}")
    print(f"Analyzing masks in: {dataset_root}")
    print(f"{'='*70}\n")

    # Process each date folder
    for date_folder in date_folders:
        semantics_dir = dataset_path / date_folder / 'semantics'

        if not semantics_dir.exists():
            print(f"⚠️  Skipping {date_folder} - semantics folder not found")
            continue

        # Get all PNG files in semantics folder
        mask_files = sorted(semantics_dir.glob('*.png'))

        if not mask_files:
            print(f"⚠️  No masks found in {date_folder}/semantics/")
            continue

        print(f"📁 Processing {date_folder}/semantics/ ({len(mask_files)} masks)...")

        for mask_path in tqdm(mask_files, desc=f"  {date_folder}", leave=False):
            stats = analyze_mask(mask_path)

            if stats is None:
                print(f"  ⚠️  Failed to read: {mask_path.name}")
                continue

            # Update aggregated stats
            all_stats['total_images'] += 1
            all_stats['total_pixels'] += stats['total_pixels']
            all_stats['class_0_pixels'] += stats['class_0']
            all_stats['class_1_pixels'] += stats['class_1']
            all_stats['class_2_plus_pixels'] += stats['class_2_plus']
            all_stats['all_unique_classes'].update(stats['unique_classes'])

            # Store per-image stats
            per_image_stats.append({
                'filename': f"{date_folder}/{mask_path.name}",
                **stats
            })

    return all_stats, per_image_stats


def print_summary(all_stats, per_image_stats):
    """Print comprehensive analysis summary"""

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")

    print(f"📊 OVERALL STATISTICS")
    print(f"{'-'*70}")
    print(f"Total masks analyzed:     {all_stats['total_images']:,}")
    print(f"Total pixels:             {all_stats['total_pixels']:,}")
    print()

    # Class distribution
    total_px = all_stats['total_pixels']
    class_0_px = all_stats['class_0_pixels']
    class_1_px = all_stats['class_1_pixels']
    class_2_px = all_stats['class_2_plus_pixels']

    print(f"Class 0 (Background):     {class_0_px:,} pixels ({class_0_px/total_px*100:.2f}%)")
    print(f"Class 1 (Crop):           {class_1_px:,} pixels ({class_1_px/total_px*100:.2f}%)")
    print(f"Class 2+ (Weed):          {class_2_px:,} pixels ({class_2_px/total_px*100:.2f}%)")
    print()

    print(f"Unique class values found: {sorted(all_stats['all_unique_classes'])}")
    print()

    # Per-image statistics
    if per_image_stats:
        print(f"\n📈 PER-IMAGE STATISTICS")
        print(f"{'-'*70}")

        # Calculate averages
        avg_total = np.mean([s['total_pixels'] for s in per_image_stats])
        avg_class0 = np.mean([s['class_0'] for s in per_image_stats])
        avg_class1 = np.mean([s['class_1'] for s in per_image_stats])
        avg_class2 = np.mean([s['class_2_plus'] for s in per_image_stats])

        print(f"Average pixels per image: {avg_total:,.0f}")
        print(f"  - Class 0 (Background): {avg_class0:,.0f} ({avg_class0/avg_total*100:.2f}%)")
        print(f"  - Class 1 (Crop):       {avg_class1:,.0f} ({avg_class1/avg_total*100:.2f}%)")
        print(f"  - Class 2+ (Weed):      {avg_class2:,.0f} ({avg_class2/avg_total*100:.2f}%)")
        print()

        # Find images with most/least weed pixels
        sorted_by_weed = sorted(per_image_stats, key=lambda x: x['class_2_plus'], reverse=True)

        print(f"\n🔝 TOP 5 IMAGES WITH MOST WEED PIXELS:")
        for i, stat in enumerate(sorted_by_weed[:5], 1):
            weed_pct = stat['class_2_plus'] / stat['total_pixels'] * 100
            print(f"  {i}. {stat['filename']:40s} - {stat['class_2_plus']:,} pixels ({weed_pct:.2f}%)")

        print(f"\n🔻 TOP 5 IMAGES WITH LEAST WEED PIXELS:")
        for i, stat in enumerate(sorted_by_weed[-5:][::-1], 1):
            weed_pct = stat['class_2_plus'] / stat['total_pixels'] * 100
            print(f"  {i}. {stat['filename']:40s} - {stat['class_2_plus']:,} pixels ({weed_pct:.2f}%)")

    print(f"\n{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze pixel distribution in semantic segmentation masks'
    )
    parser.add_argument(
        '--dataset_root', 
        type=str, 
        required=True,
        help='Path to weedsgalore-dataset folder'
    )
    parser.add_argument(
        '--save_csv',
        type=str,
        default=None,
        help='Optional: Save per-image statistics to CSV file'
    )

    args = parser.parse_args()

    # Run analysis
    all_stats, per_image_stats = analyze_all_masks(args.dataset_root)

    # Print summary
    print_summary(all_stats, per_image_stats)

    # Save to CSV if requested
    if args.save_csv:
        import csv

        with open(args.save_csv, 'w', newline='') as f:
            if per_image_stats:
                fieldnames = ['filename', 'total_pixels', 'class_0', 'class_1', 
                             'class_2_plus', 'unique_classes']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_image_stats)

        print(f"✅ Per-image statistics saved to: {args.save_csv}\n")


if __name__ == '__main__':
    main()
