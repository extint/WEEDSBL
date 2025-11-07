#!/usr/bin/env python3
"""
Preprocess RGB+NIR images and save as 4-channel binary for Jetson Nano inference
No need for cv2 on Nano - just load the preprocessed binary!
"""

import numpy as np
import cv2
import os

IMG_HEIGHT = 960
IMG_WIDTH = 1280
IN_CHANNELS = 4

# ImageNet normalization (matches training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_rgbnir(bgr_path, nir_path, output_npy, target_hw=(960, 1280)):
    """
    Preprocess RGB+NIR images and save as .npy file
    
    Args:
        bgr_path: Path to RGB image (BGR format from cv2)
        nir_path: Path to NIR image
        output_npy: Output path for preprocessed .npy file
        target_hw: Target (height, width)
    """
    th, tw = target_hw
    
    print(f"Loading RGB: {bgr_path}")
    bgr = cv2.imread(bgr_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB image not found: {bgr_path}")
    
    print(f"Loading NIR: {nir_path}")
    nir = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)
    if nir is None:
        raise FileNotFoundError(f"NIR image not found: {nir_path}")
    
    print(f"Original RGB shape: {bgr.shape}, NIR shape: {nir.shape}")
    
    # Convert NIR to grayscale if needed
    if nir.ndim == 3:
        nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)
    
    # Resize RGB
    bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    
    # Resize NIR
    nir = cv2.resize(nir, (tw, th), interpolation=cv2.INTER_LINEAR)
    
    # BGR -> RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Normalize RGB with ImageNet stats
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    
    # Scale NIR to [0, 1]
    if nir.dtype == np.uint16:
        nir = nir.astype(np.float32) / 65535.0
    else:
        nir = nir.astype(np.float32) / 255.0
    
    # Stack R, G, B, NIR -> (H, W, 4)
    x_hw4 = np.dstack([rgb, nir[..., None]])
    
    # Convert to NCHW format
    x_chw = np.transpose(x_hw4, (2, 0, 1))  # (4, H, W)
    x_nchw = np.expand_dims(x_chw, axis=0)  # (1, 4, H, W)
    
    # Ensure float32 and contiguous
    x_nchw = np.ascontiguousarray(x_nchw.astype(np.float32))
    
    print(f"Preprocessed shape: {x_nchw.shape}")
    print(f"Preprocessed range: [{x_nchw.min():.3f}, {x_nchw.max():.3f}]")
    
    # Save as .npy
    os.makedirs(os.path.dirname(output_npy) or ".", exist_ok=True)
    np.save(output_npy, x_nchw)
    
    print(f"✓ Saved preprocessed input to: {output_npy}")
    print(f"  Shape: {x_nchw.shape}")
    print(f"  Size: {x_nchw.nbytes / (1024*1024):.1f} MB")
    
    return x_nchw


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess RGB+NIR for Jetson Nano")
    parser.add_argument("--rgb", default= "/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB/RGB/DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m.JPG", type=str, required=False, help="Path to RGB image")
    parser.add_argument("--nir", default="/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB/Multispectral/DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m_NIR.TIF", type=str, required=False, help="Path to NIR image")
    parser.add_argument("--output",default="misc" ,type=str, required=False, help="Output .npy path")
    
    args = parser.parse_args()
    
    preprocess_rgbnir(args.rgb, args.nir, args.output)
    print("\n✅ Ready to transfer to Jetson Nano!")
