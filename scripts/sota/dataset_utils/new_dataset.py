import os
import shutil
from pathlib import Path

# Configuration
INPUT_LIST = "/home/vjti-comp/WEEDSBL/scripts/sota/dataset_utils/5000_mixed_images.txt"
SOURCE_DIR = "/home/vjti-comp/Downloads/FINAL_SUGARBEETS_DATASET"  # Update this
OUTPUT_DIR = "/home/vjti-comp/Downloads/SUGARBEETS_MIXED_DATASET"   # Update this

# Folder names
RGB_FOLDER = "rgb"
NIR_FOLDER = "nir"
MASK_FOLDER = "masks"

def convert_filename(mask_filename, new_prefix):
    """
    Convert mask filename to rgb or nir filename.
    e.g., 'mask_bonirob_2016-05-09-11-10-10_10_frame143.png' 
    -> 'rgb_bonirob_2016-05-09-11-10-10_10_frame143.png'
    """
    # Remove 'mask_' prefix and add new prefix
    if mask_filename.startswith('mask_'):
        base_name = mask_filename[5:]  # Remove 'mask_'
        return f"{new_prefix}_{base_name}"
    return mask_filename

def main():
    # Read the list of mask filenames
    with open(INPUT_LIST, 'r') as f:
        mask_files = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(mask_files)} images in list")
    
    # Create output directories
    output_rgb = Path(OUTPUT_DIR) / RGB_FOLDER
    output_nir = Path(OUTPUT_DIR) / NIR_FOLDER
    output_mask = Path(OUTPUT_DIR) / MASK_FOLDER
    
    output_rgb.mkdir(parents=True, exist_ok=True)
    output_nir.mkdir(parents=True, exist_ok=True)
    output_mask.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories in {OUTPUT_DIR}")
    
    # Source directories
    source_rgb = Path(SOURCE_DIR) / RGB_FOLDER
    source_nir = Path(SOURCE_DIR) / NIR_FOLDER
    source_mask = Path(SOURCE_DIR) / MASK_FOLDER
    
    copied_count = 0
    missing_count = 0
    
    for mask_filename in mask_files:
        # Convert mask filename to rgb and nir filenames
        rgb_filename = convert_filename(mask_filename, "rgb")
        nir_filename = convert_filename(mask_filename, "nir")
        
        # Construct source paths
        rgb_src = source_rgb / rgb_filename
        nir_src = source_nir / nir_filename
        mask_src = source_mask / mask_filename
        
        # Construct destination paths (same names)
        rgb_dst = output_rgb / rgb_filename
        nir_dst = output_nir / nir_filename
        mask_dst = output_mask / mask_filename
        
        # Check if all three files exist
        if not rgb_src.exists():
            print(f"WARNING: RGB not found: {rgb_src}")
            missing_count += 1
            continue
        if not nir_src.exists():
            print(f"WARNING: NIR not found: {nir_src}")
            missing_count += 1
            continue
        if not mask_src.exists():
            print(f"WARNING: Mask not found: {mask_src}")
            missing_count += 1
            continue
        
        # Copy files
        shutil.copy2(rgb_src, rgb_dst)
        shutil.copy2(nir_src, nir_dst)
        shutil.copy2(mask_src, mask_dst)
        
        copied_count += 1
        
        if copied_count % 100 == 0:
            print(f"Copied {copied_count} images...")
    
    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"{'='*60}")
    print(f"Total images in list: {len(mask_files)}")
    print(f"Successfully copied: {copied_count}")
    print(f"Missing/skipped: {missing_count}")
    print(f"\nNew dataset location: {OUTPUT_DIR}")
    print(f"  - {RGB_FOLDER}/: {copied_count} images (rgb_*.png)")
    print(f"  - {NIR_FOLDER}/: {copied_count} images (nir_*.png)")
    print(f"  - {MASK_FOLDER}/: {copied_count} images (mask_*.png)")

if __name__ == "__main__":
    main()
