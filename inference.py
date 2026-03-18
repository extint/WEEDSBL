#!/usr/bin/env python3
"""
Inference script for Dual-Encoder RGB-NIR Crop/Weed Segmentation.
Loads model, processes RGB+NIR inputs, and saves prediction mask overlay.
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Import your model
from dual_encoder.updated_architecture import DualEncoderAFFNet


class DualEncoderInference:
    def __init__(self, checkpoint_path, device='cuda', rgb_base_ch=32, nir_base_ch=16):
        """
        Args:
            checkpoint_path: Path to trained model weights (.pth)
            device: 'cuda' or 'cpu'
            rgb_base_ch, nir_base_ch: Must match training config
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"[INFO] Loading model from: {checkpoint_path}")
        self.model = DualEncoderAFFNet(
            # rgb_base_ch=rgb_base_ch,
            nir_base_ch=nir_base_ch,
            num_classes=1,
            embed_dim=64,
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # If checkpoint was saved as a dict (training-style)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # fallback if someone saved only model.state_dict()
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=True)


        # state_dict = torch.load(checkpoint_path, map_location=self.device)
        # self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"[INFO] Model loaded on {self.device}")
        
        # ImageNet normalization (same as training)
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def load_and_preprocess(self, rgb_path, nir_path, target_size=None):
        """
        Load RGB and NIR images and preprocess.
        
        Args:
            rgb_path: Path to RGB image (.jpg, .png)
            nir_path: Path to NIR image (.tif, .png)
            target_size: (H, W) or None (uses original size)
        
        Returns:
            rgb_tensor: (1, 3, H, W)
            nir_tensor: (1, 1, H, W)
            original_size: (orig_H, orig_W) for resizing output back
        """
        # Load RGB
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        original_size = rgb.shape[:2]
        
        # Load NIR
        nir = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)
        if nir is None:
            raise FileNotFoundError(f"NIR image not found: {nir_path}")
        if nir.ndim == 3:
            nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)
        
        # Align NIR to RGB size if needed
        if nir.shape[:2] != rgb.shape[:2]:
            nir = cv2.resize(nir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Resize if target_size specified
        if target_size is not None:
            H, W = target_size
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            nir = cv2.resize(nir, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Scale to [0, 1]
        rgb = rgb.astype(np.float32)
        nir = nir.astype(np.float32)
        
        if rgb.max() > 1.0:
            rgb /= 255.0
        if nir.max() > 1.0:
            # Handle uint16 NIR
            if nir.max() > 255:
                nir /= 65535.0
            else:
                nir /= 255.0
        
        # Normalize RGB with ImageNet stats
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        
        # To tensor
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)  # (1, 3, H, W)
        nir_tensor = torch.from_numpy(nir).float().unsqueeze(0).unsqueeze(0)       # (1, 1, H, W)
        
        return rgb_tensor.to(self.device), nir_tensor.to(self.device), original_size
    
    @torch.no_grad()
    def predict(self, rgb_tensor, nir_tensor, threshold=0.5):
        """
        Run inference.
        
        Args:
            rgb_tensor: (1, 3, H, W)
            nir_tensor: (1, 1, H, W)
            threshold: Binary threshold for mask
        
        Returns:
            pred_mask: (H, W) numpy array, binary mask
            pred_prob: (H, W) numpy array, continuous probabilities
        """
        logits = self.model(rgb_tensor, nir_tensor)  # (1, 1, H, W)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]  # (H, W)
        mask = (probs > threshold).astype(np.uint8)
        return mask, probs
    
    def create_overlay(self, rgb_path, mask, alpha=0.5):
        """
        Create visualization with mask overlay on RGB.
        
        Args:
            rgb_path: Path to original RGB (for loading unprocessed version)
            mask: (H, W) binary mask
            alpha: Overlay transparency
        
        Returns:
            overlay: (H, W, 3) RGB image with mask overlay
        """
        # Load original RGB
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Resize mask to match original if needed
        if mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask: green=crop(0), red=weed(1)
        overlay = rgb.copy()
        
        # Weed regions: red overlay
        weed_mask = mask == 1
        overlay[weed_mask] = overlay[weed_mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
        
        # Optional: highlight crop with slight green tint
        crop_mask = mask == 0
        overlay[crop_mask] = overlay[crop_mask] * (1 - alpha*0.3) + np.array([0, 255, 0]) * (alpha*0.3)
        
        return overlay.astype(np.uint8)
    
    def save_outputs(self, rgb_path, mask, probs, output_dir, base_name=None):
        """
        Save all outputs: mask, overlay, probability map.
        
        Args:
            rgb_path: Original RGB path
            mask: Binary mask
            probs: Probability map
            output_dir: Directory to save outputs
            base_name: Output filename base (auto-generated if None)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if base_name is None:
            base_name = Path(rgb_path).stem
        
        # 1. Save binary mask
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        print(f"[INFO] Saved mask: {mask_path}")
        
        # 2. Save overlay
        overlay = self.create_overlay(rgb_path, mask, alpha=0.5)
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved overlay: {overlay_path}")
        
        # 3. Save probability heatmap
        prob_colored = (probs * 255).astype(np.uint8)
        prob_colored = cv2.applyColorMap(prob_colored, cv2.COLORMAP_JET)
        prob_path = os.path.join(output_dir, f"{base_name}_prob.png")
        cv2.imwrite(prob_path, prob_colored)
        print(f"[INFO] Saved probability map: {prob_path}")
        
        # 4. Save combined visualization
        rgb_orig = cv2.imread(rgb_path)
        rgb_orig = cv2.cvtColor(rgb_orig, cv2.COLOR_BGR2RGB)
        
        # Resize all to same size
        H, W = rgb_orig.shape[:2]
        overlay_resized = cv2.resize(overlay, (W, H))
        prob_resized = cv2.resize(prob_colored, (W, H))
        mask_resized = cv2.resize((mask * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2RGB)
        
        # Create side-by-side
        combined = np.hstack([rgb_orig, overlay_resized, mask_rgb, cv2.cvtColor(prob_resized, cv2.COLOR_BGR2RGB)])
        combined_path = os.path.join(output_dir, f"{base_name}_combined.png")
        cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved combined: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Dual-Encoder RGB-NIR Inference")
    
    parser.add_argument('--checkpoint', type=str,
                       help='Path to trained model checkpoint (.pth)'
                       , default= "checkpoints_dual_encoder/best_dual_encoder.pth")
    parser.add_argument('--rgb', type=str,
                       help='Path to RGB input image',
                       default= "/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB/RGB/DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m.JPG")
    parser.add_argument('--nir', type=str,
                       help='Path to NIR input image',
                       default="/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB/Multispectral/DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m_NIR.TIF")
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                       help='Directory to save outputs (default: ./inference_output)')
    parser.add_argument('--threshold', type=float, default=0.67,
                       help='Binary threshold for segmentation (default: 0.5)')
    parser.add_argument('--target_size', type=int, nargs=2, default=None,
                       help='Resize input to [H W], e.g., --target_size 512 512 (default: original size)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--rgb_base_ch', type=int, default=32,
                       help='RGB encoder base channels (must match training, default: 32)')
    parser.add_argument('--nir_base_ch', type=int, default=16,
                       help='NIR encoder base channels (must match training, default: 16)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.rgb):
        raise FileNotFoundError(f"RGB image not found: {args.rgb}")
    if not os.path.exists(args.nir):
        raise FileNotFoundError(f"NIR image not found: {args.nir}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Initialize inference
    inference = DualEncoderInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        rgb_base_ch=args.rgb_base_ch,
        nir_base_ch=args.nir_base_ch
    )
    
    # Load and preprocess
    print(f"[INFO] Loading images...")
    target_size = tuple(args.target_size) if args.target_size else None
    rgb_tensor, nir_tensor, original_size = inference.load_and_preprocess(
        args.rgb, args.nir, target_size=target_size
    )
    
    # Predict
    print(f"[INFO] Running inference...")
    mask, probs = inference.predict(rgb_tensor, nir_tensor, threshold=args.threshold)
    
    # Compute stats
    weed_pixels = (mask == 1).sum()
    crop_pixels = (mask == 0).sum()
    total_pixels = mask.size
    weed_pct = weed_pixels / total_pixels * 100
    
    print(f"\n[RESULTS]")
    print(f"  Weed coverage: {weed_pct:.2f}% ({weed_pixels} pixels)")
    print(f"  Crop coverage: {100-weed_pct:.2f}% ({crop_pixels} pixels)")
    
    # Save outputs
    inference.save_outputs(args.rgb, mask, probs, args.output_dir)
    
    print(f"\n[DONE] All outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
