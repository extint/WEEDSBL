# FROM LINE 135 CHANGE PATHS AS PER YOUR SETUP
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from new_main import UNet, LightMANet

def preprocess_rgb_only(img_path, target_hw=(960, 1280)):
    """
    Preprocess a single RGB image for the student model (3 channels).
    """
    # Read BGR image
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Resize to target size
    th, tw = target_hw
    if bgr.shape[:2] != (th, tw):
        bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB and normalize to [0, 1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    
    # Convert to CHW format and add batch dimension
    x_chw = rgb.transpose(2, 0, 1)  # (3, H, W)
    x = torch.from_numpy(x_chw).float().unsqueeze(0)  # (1, 3, H, W)
    
    return x, bgr

def run_student_inference(
    model,
    img_path,
    ckpt_path,
    out_mask_path="output_mask.png",
    out_overlay_path="output_overlay.png",
    device="cuda",
    target_size=(960, 1280),
    model_arch="unet",
    in_channels=3
):
    """
    Run inference with the trained student model (RGB-only).
    
    Args:
        model: Student model instance (UNet or LightMANet)
        img_path: Path to input RGB image
        ckpt_path: Path to trained student checkpoint
        out_mask_path: Where to save the binary mask
        out_overlay_path: Where to save the overlay visualization
        device: 'cuda' or 'cpu'
        target_size: (height, width) tuple
        model_arch: 'unet' or 'lightman'
        in_channels: Number of input channels (3 for student)
    """
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded checkpoint from: {ckpt_path}")
    
    # Preprocess image
    x, bgr = preprocess_rgb_only(img_path, target_hw=target_size)
    x = x.to(device)
    print(f"Input shape: {x.shape}")
    
    # Run inference
    with torch.no_grad():
        logits = model(x)  # (1, 1, H, W)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W)
    
    # Create binary mask (threshold at 0.5)
    mask_binary = (probs > 0.5).astype(np.uint8) * 255
    
    # Save binary mask
    cv2.imwrite(out_mask_path, mask_binary)
    print(f"Saved binary mask to: {out_mask_path}")
    
    # Create overlay visualization
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Create colored overlay (weeds in red)
    overlay = rgb.copy()
    overlay[mask_binary > 0] = [255, 0, 0]  # Red for detected weeds
    
    # Blend original and overlay
    alpha = 0.4
    blended = cv2.addWeighted(rgb, 1-alpha, overlay, alpha, 0)
    
    # Save overlay
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_overlay_path, blended_bgr)
    print(f"Saved overlay visualization to: {out_overlay_path}")
    
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
    plt.savefig("inference_visualization1.png", dpi=150, bbox_inches='tight')
    print("Saved visualization to: inference_visualization.png")
    plt.show()
    
    return probs, mask_binary


if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    DATA_ROOT = "/home/vjti-comp/Desktop/WeedyRice-RGBMS-DB"  # Update this
    CKPT_PATH = "./student_ckpts/student_rgb_only_best.pth"  # Path to your student checkpoint
    
    # Example image path from your dataset
    IMG_PATH = os.path.join(
        DATA_ROOT,
        "RGB",
        "DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m.JPG"
    )
    
    # Model architecture - match what you used for training
    # Options: "unet" or "lightman"
    STUDENT_ARCH = "unet"  # Change to "lightman" if you used LightMANet
    
    # Create student model instance
    if STUDENT_ARCH == "unet":
        student_model = UNet(in_channels=3, base_ch=4, out_channels=1)
    else:
        student_model = LightMANet(in_channels=3, num_classes=1, base_ch=32)
    
    # Run inference
    probs, mask = run_student_inference(
        model=student_model,
        img_path=IMG_PATH,
        ckpt_path=CKPT_PATH,
        out_mask_path="./output_mask.png",
        out_overlay_path="./output_overlay.png",
        device="cuda",  # Use "cpu" if no GPU available
        target_size=(960, 1280),
        model_arch=STUDENT_ARCH,
        in_channels=3
    )
    
    print("\nInference complete!")
    print(f"Prediction probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Pixels classified as weed: {(mask > 0).sum()} / {mask.size}")
