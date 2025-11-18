#!/usr/bin/env python3
"""
Inference script using modular preprocessing
"""
import argparse
import os
import cv2
import torch
import numpy as np

from MANet import LightMANet
from weedutils.rgbnir_preprocessing import RGBNIRPreprocessor


def run_inference(
    checkpoint_path: str,
    rgb_path: str,
    nir_path: str,
    output_path: str,
    target_size: tuple = (960, 1280),
    device: str = "cuda"
):
    """
    Run inference on a single RGB+NIR image pair
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        rgb_path: Path to RGB image
        nir_path: Path to NIR .TIF
        output_path: Path to save output mask
        target_size: (H, W) for preprocessing
        device: "cuda" or "cpu"
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine input channels from config
    config = ckpt.get("config", {})
    in_ch = 4 if config.get("use_rgbnir", True) else 3
    
    # Load model
    model = LightMANet(in_channels=in_ch, num_classes=1, base_ch=32)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded (epoch {ckpt.get('epoch', 'unknown')})")
    
    # Initialize preprocessor
    preprocessor = RGBNIRPreprocessor(
        target_size=target_size,
        normalize=True
    )
    
    # Load and preprocess image
    print(f"[INFO] Loading image: {rgb_path}")
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    
    x = preprocessor.preprocess(bgr, nir_path, return_tensor=True).to(device)
    
    # Inference
    print("[INFO] Running inference...")
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    # Post-process
    mask = (probs > 0.5).astype(np.uint8) * 255
    
    # Save
    cv2.imwrite(output_path, mask)
    print(f"[INFO] Saved output mask: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run inference on RGB+NIR image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--nir", type=str, required=True, help="Path to NIR .TIF")
    parser.add_argument("--output", type=str, required=True, help="Path to save output mask")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        rgb_path=args.rgb,
        nir_path=args.nir,
        output_path=args.output,
        target_size=(args.height, args.width),
        device=args.device
    )


if __name__ == "__main__":
    main()
