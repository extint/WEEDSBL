#!/usr/bin/env python3
"""
Inference script for distilled student model (RGB-only)
Uses modular preprocessing from preprocessing.py
"""
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models import create_model


class RGBOnlyPreprocessor:
    """Simplified RGB-only preprocessor for student model"""
    
    def __init__(
        self,
        target_size=(960, 1280),
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225)
    ):
        self.target_h, self.target_w = target_size
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
    
    def preprocess(self, img_path, return_original=True):
        """Preprocess RGB image for student inference"""
        # Read BGR image
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Resize to target size
        if bgr.shape[:2] != (self.target_h, self.target_w):
            bgr = cv2.resize(bgr, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB and normalize to [0, 1]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # ImageNet normalization
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        
        # Convert to CHW format and add batch dimension
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)  # (1, 3, H, W)
        
        if return_original:
            return x, bgr
        return x


def run_student_inference(
    model,
    img_path,
    ckpt_path,
    out_mask_path="output_mask.png",
    out_overlay_path="output_overlay.png",
    device="cuda",
    target_size=(960, 1280)
):
    """
    Run inference with the trained student model (RGB-only)
    
    Args:
        model: Student model instance (UNet or LightMANet)
        img_path: Path to input RGB image
        ckpt_path: Path to trained student checkpoint
        out_mask_path: Where to save the binary mask
        out_overlay_path: Where to save the overlay visualization
        device: 'cuda' or 'cpu'
        target_size: (height, width) tuple
    """
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load model
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    state = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[INFO] Loaded checkpoint from: {ckpt_path}")
    
    # Initialize preprocessor
    preprocessor = RGBOnlyPreprocessor(target_size=target_size)
    
    # Preprocess image
    x, bgr = preprocessor.preprocess(img_path, return_original=True)
    x = x.to(device)
    print(f"[INFO] Input shape: {x.shape}")
    
    # Run inference
    with torch.no_grad():
        logits = model(x)  # (1, 1, H, W)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W)
    
    # Create binary mask (threshold at 0.5)
    mask_binary = (probs > 0.5).astype(np.uint8) * 255
    
    # Save binary mask
    cv2.imwrite(out_mask_path, mask_binary)
    print(f"[INFO] Saved binary mask to: {out_mask_path}")
    
    # Create overlay visualization
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Create colored overlay (weeds in red)
    overlay = rgb.copy()
    overlay[mask_binary > 0] = [255, 0, 0]  # Red for detected weeds
    
    # Blend original and overlay
    alpha = 0.4
    blended = cv2.addWeighted(rgb, 1 - alpha, overlay, alpha, 0)
    
    # Save overlay
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_overlay_path, blended_bgr)
    print(f"[INFO] Saved overlay visualization to: {out_overlay_path}")
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask_binary, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    
    axes[2].imshow(blended.astype(np.uint8))
    axes[2].set_title("Overlay (Weeds in Red)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("inference_visualization.png", dpi=150, bbox_inches='tight')
    print("[INFO] Saved visualization to: inference_visualization.png")
    plt.show()
    
    return probs, mask_binary


if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    DATA_ROOT = "/home/vjti-comp/Desktop/WeedyRice-RGBMS-DB"
    CKPT_PATH = "./student_ckpts/student_rgb_only_best.pth"
    
    # Example image path from your dataset
    IMG_PATH = os.path.join(
        DATA_ROOT,
        "RGB",
        "DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m.JPG"
    )
    
    # Model architecture - match what you used for training
    STUDENT_ARCH = "unet"  # Options: "unet" or "lightmanet"
    STUDENT_BASE_CH = 64    # Match training config
    
    # Create student model instance using model factory
    student_model = create_model(
        architecture=STUDENT_ARCH,
        in_channels=3,
        num_classes=1,
        base_ch=STUDENT_BASE_CH
    )
    
    print(f"[INFO] Created {STUDENT_ARCH.upper()} student model with {STUDENT_BASE_CH} base channels")
    
    # Run inference
    probs, mask = run_student_inference(
        model=student_model,
        img_path=IMG_PATH,
        ckpt_path=CKPT_PATH,
        out_mask_path="./output_mask.png",
        out_overlay_path="./output_overlay.png",
        device="cuda",  # Use "cpu" if no GPU available
        target_size=(960, 1280)
    )
    
    print("\n[INFO] Inference complete!")
    print(f"[INFO] Prediction probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"[INFO] Pixels classified as weed: {(mask > 0).sum()} / {mask.size}")
