import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# Configuration
SOURCE_DIR = "/home/vjtiadmin/Desktop/BTechGroup/SUGARBEETS_AUG_MIXED_DATASET"
OUTPUT_DIR = "/home/vjtiadmin/Desktop/BTechGroup/SUGARBEETS_AUGMENTED_DATASET"

# Folder names
RGB_FOLDER = "rgb"
NIR_FOLDER = "nir"
MASK_FOLDER = "masks"

# Augmentation names (for file naming)
AUG_NAMES = [
    "hflip",          # 1. Horizontal Flip
    "vflip",          # 2. Vertical Flip
    "rot15",          # 3. Rotation +15 degrees
    "rot_minus15",    # 4. Rotation -15 degrees
    "crop",           # 5. Random Crop & Resize
    "brightness",     # 6. Brightness Adjustment
    "contrast",       # 7. Contrast Adjustment
    "hue",            # 8. Hue Shift (RGB only)
    "noise",          # 9. Gaussian Noise
]


def create_augmentation_pipeline(aug_type, apply_to_nir=True):
    """
    Create augmentation pipeline for specific augmentation type.
    
    Args:
        aug_type: Type of augmentation to apply
        apply_to_nir: Whether this augmentation should be applied to NIR
                     (False for hue shift which is RGB-only)
    
    Returns:
        Albumentations transform
    """
    if aug_type == "hflip":
        return A.HorizontalFlip(p=1.0)
    
    elif aug_type == "vflip":
        return A.VerticalFlip(p=1.0)
    
    elif aug_type == "rot15":
        return A.Rotate(limit=(15, 15), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0)
    
    elif aug_type == "rot_minus15":
        return A.Rotate(limit=(-15, -15), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0)
    
    elif aug_type == "crop":
        return A.RandomResizedCrop(height=512, width=512, scale=(0.7, 0.9), p=1.0)
    
    elif aug_type == "brightness":
        return A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0)
    
    elif aug_type == "contrast":
        return A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1.0)
    
    elif aug_type == "hue":
        # Only for RGB images
        return A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0, val_shift_limit=0, p=1.0)
    
    elif aug_type == "noise":
        return A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
    
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def get_image_size(image_path):
    """Get the size of the first image to use for all augmentations."""
    img = cv2.imread(str(image_path))
    return img.shape[:2]  # (height, width)


def apply_augmentation(rgb_path, nir_path, mask_path, aug_type, output_rgb, output_nir, output_mask, image_size):
    """
    Apply augmentation to RGB, NIR, and mask triplet and save results.
    
    Args:
        rgb_path: Path to RGB image
        nir_path: Path to NIR image
        mask_path: Path to mask image
        aug_type: Type of augmentation to apply
        output_rgb: Output path for augmented RGB
        output_nir: Output path for augmented NIR
        output_mask: Output path for augmented mask
        image_size: (height, width) tuple for resizing
    """
    # Read images
    rgb = cv2.imread(str(rgb_path))
    nir = cv2.imread(str(nir_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if rgb is None or nir is None or mask is None:
        print(f"ERROR: Failed to read images for {rgb_path.stem}")
        return False
    
    # Resize all images to consistent size if needed
    height, width = image_size
    rgb = cv2.resize(rgb, (width, height))
    nir = cv2.resize(nir, (width, height))
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Special handling for hue shift (RGB only)
    if aug_type == "hue":
        # Apply hue shift to RGB
        transform_rgb = create_augmentation_pipeline(aug_type, apply_to_nir=False)
        augmented_rgb = transform_rgb(image=rgb)['image']
        
        # Apply geometric equivalent for NIR (just copy for hue)
        # For color augmentations, we keep NIR unchanged or apply brightness/contrast
        augmented_nir = nir.copy()
        augmented_mask = mask.copy()
        
    else:
        # For geometric and spatial transformations, apply to all three
        # For brightness/contrast/noise, apply to RGB and NIR separately but not to mask
        
        if aug_type in ["brightness", "contrast", "noise"]:
            # Apply to RGB and NIR independently, keep mask with geometric transform only
            transform = create_augmentation_pipeline(aug_type)
            augmented_rgb = transform(image=rgb)['image']
            augmented_nir = transform(image=nir)['image']
            augmented_mask = mask.copy()
            
        else:
            # Geometric transformations - apply same transform to all three
            transform = create_augmentation_pipeline(aug_type)
            
            # Apply transform with additional_targets for synchronized augmentation
            transform_sync = A.Compose([transform], 
                                      additional_targets={'nir': 'image', 'mask': 'mask'})
            
            augmented = transform_sync(image=rgb, nir=nir, mask=mask)
            augmented_rgb = augmented['image']
            augmented_nir = augmented['nir']
            augmented_mask = augmented['mask']
    
    # Save augmented images
    cv2.imwrite(str(output_rgb), augmented_rgb)
    cv2.imwrite(str(output_nir), augmented_nir)
    cv2.imwrite(str(output_mask), augmented_mask)
    
    return True


def generate_augmented_filename(original_filename, aug_name):
    """
    Generate new filename with augmentation suffix.
    
    Example: 
        'rgb_bonirob_2016-05-09-11-10-10_10_frame143.png' + 'hflip'
        -> 'rgb_bonirob_2016-05-09-11-10-10_10_frame143_hflip.png'
    """
    stem = Path(original_filename).stem
    ext = Path(original_filename).suffix
    return f"{stem}_{aug_name}{ext}"


def main():
    print("="*80)
    print("DATASET AUGMENTATION SCRIPT")
    print("="*80)
    
    # Setup paths
    source_rgb_dir = Path(SOURCE_DIR) / RGB_FOLDER
    source_nir_dir = Path(SOURCE_DIR) / NIR_FOLDER
    source_mask_dir = Path(SOURCE_DIR) / MASK_FOLDER
    
    output_rgb_dir = Path(OUTPUT_DIR) / RGB_FOLDER
    output_nir_dir = Path(OUTPUT_DIR) / NIR_FOLDER
    output_mask_dir = Path(OUTPUT_DIR) / MASK_FOLDER
    
    # Create output directories
    output_rgb_dir.mkdir(parents=True, exist_ok=True)
    output_nir_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSource directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Get list of all mask files (we use masks as reference)
    mask_files = sorted(list(source_mask_dir.glob("mask_*.png")))
    
    if len(mask_files) == 0:
        print(f"\nERROR: No mask files found in {source_mask_dir}")
        return
    
    print(f"\nFound {len(mask_files)} original images")
    print(f"Will generate {len(mask_files) * len(AUG_NAMES)} augmented images")
    print(f"Total dataset size: {len(mask_files) * (1 + len(AUG_NAMES))} images")
    
    # Get image size from first image
    first_rgb = source_rgb_dir / mask_files[0].name.replace("mask_", "rgb_")
    image_size = get_image_size(first_rgb)
    print(f"\nUsing image size: {image_size[0]}x{image_size[1]}")
    
    print("\n" + "="*80)
    print("AUGMENTATION TYPES:")
    for i, aug_name in enumerate(AUG_NAMES, 1):
        print(f"  {i:2d}. {aug_name}")
    print("="*80 + "\n")
    
    # First, copy original images to output directory
    print("Copying original images...")
    for mask_file in tqdm(mask_files, desc="Original images"):
        mask_filename = mask_file.name
        rgb_filename = mask_filename.replace("mask_", "rgb_")
        nir_filename = mask_filename.replace("mask_", "nir_")
        
        rgb_src = source_rgb_dir / rgb_filename
        nir_src = source_nir_dir / nir_filename
        mask_src = source_mask_dir / mask_filename
        
        # Copy to output
        import shutil
        shutil.copy2(rgb_src, output_rgb_dir / rgb_filename)
        shutil.copy2(nir_src, output_nir_dir / nir_filename)
        shutil.copy2(mask_src, output_mask_dir / mask_filename)
    
    # Apply augmentations
    print("\nApplying augmentations...")
    
    total_augmented = 0
    failed_count = 0
    
    for aug_name in AUG_NAMES:
        print(f"\nProcessing augmentation: {aug_name}")
        
        for mask_file in tqdm(mask_files, desc=f"  {aug_name}"):
            mask_filename = mask_file.name
            rgb_filename = mask_filename.replace("mask_", "rgb_")
            nir_filename = mask_filename.replace("mask_", "nir_")
            
            # Source paths
            rgb_path = source_rgb_dir / rgb_filename
            nir_path = source_nir_dir / nir_filename
            mask_path = source_mask_dir / mask_filename
            
            # Output paths with augmentation suffix
            aug_rgb_filename = generate_augmented_filename(rgb_filename, aug_name)
            aug_nir_filename = generate_augmented_filename(nir_filename, aug_name)
            aug_mask_filename = generate_augmented_filename(mask_filename, aug_name)
            
            output_rgb = output_rgb_dir / aug_rgb_filename
            output_nir = output_nir_dir / aug_nir_filename
            output_mask = output_mask_dir / aug_mask_filename
            
            # Apply augmentation
            success = apply_augmentation(
                rgb_path, nir_path, mask_path,
                aug_name,
                output_rgb, output_nir, output_mask,
                image_size
            )
            
            if success:
                total_augmented += 1
            else:
                failed_count += 1
    
    # Final summary
    print("\n" + "="*80)
    print("AUGMENTATION COMPLETE!")
    print("="*80)
    print(f"Original images copied: {len(mask_files)}")
    print(f"Augmented images created: {total_augmented}")
    print(f"Failed augmentations: {failed_count}")
    print(f"\nTotal dataset size: {len(mask_files) + total_augmented} images")
    print(f"\nOutput location: {OUTPUT_DIR}")
    print(f"  - {RGB_FOLDER}/: {len(mask_files) + total_augmented} images")
    print(f"  - {NIR_FOLDER}/: {len(mask_files) + total_augmented} images")
    print(f"  - {MASK_FOLDER}/: {len(mask_files) + total_augmented} images")
    print("="*80)


if __name__ == "__main__":
    # Check if albumentations is installed
    try:
        import albumentations as A
    except ImportError:
        print("ERROR: albumentations library not found!")
        print("Please install it using: pip install albumentations")
        exit(1)
    
    # Check if opencv is installed
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv library not found!")
        print("Please install it using: pip install opencv-python")
        exit(1)
    
    # Check if tqdm is installed
    try:
        from tqdm import tqdm
    except ImportError:
        print("ERROR: tqdm library not found!")
        print("Please install it using: pip install tqdm")
        exit(1)
    
    main()