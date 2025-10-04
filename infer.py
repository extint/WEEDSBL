import os
import cv2
from models.unet import UNet
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import random

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_nir_as_gray(nir_input):
    # Accept a path or a preloaded ndarray
    if isinstance(nir_input, str):
        nir = cv2.imread(nir_input, cv2.IMREAD_UNCHANGED)
        if nir is None:
            raise FileNotFoundError(f"NIR not found: {nir_input}")
    else:
        nir = nir_input
    if nir is None:
        raise ValueError("NIR array is None")
    # If 3-channel, convert to grayscale
    if nir.ndim == 3:
        nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)
    return nir

def scale_to_unit(arr):
    # Handle uint16 NIR; otherwise assume uint8-like range
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    a = arr.astype(np.float32)
    # If values look like 0..255, scale accordingly
    return a / 255.0 if a.max() > 1.0 else a

def preprocess_rgbnir(bgr, nir_input, target_hw=(960, 1280)):
    # Resize RGB to target
    th, tw = target_hw
    if bgr.shape[:2] != (th, tw):
        bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    # Read/resize NIR to match target
    nir = read_nir_as_gray(nir_input)
    if nir.shape[:2] != (th, tw):
        nir = cv2.resize(nir, (tw, th), interpolation=cv2.INTER_LINEAR)
    # BGR->RGB, float32 [0,1], ImageNet normalize
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    # NIR to [0,1]
    nir = scale_to_unit(nir)
    # Stack R,G,B,NIR -> CHW -> add batch dim
    x_hw4 = np.dstack([rgb, nir[..., None]])  # (H, W, 4) in R,G,B,NIR
    x_chw = x_hw4.transpose(2, 0, 1)  # (4, H, W)
    x = torch.from_numpy(x_chw).float().unsqueeze(0)  # (1, 4, H, W)
    return x, rgb, nir

def load_ground_truth(gt_path, target_hw=(960, 1280)):
    """Load and resize ground truth mask"""
    if not gt_path or not os.path.exists(gt_path):
        if gt_path:
            print(f"Warning: Ground truth not found at {gt_path}")
        return None
    
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    
    th, tw = target_hw
    if gt.shape[:2] != (th, tw):
        gt = cv2.resize(gt, (tw, th), interpolation=cv2.INTER_NEAREST)
    
    return gt

def extract_training_params(ckpt_path):
    """Extract training parameters from checkpoint"""
    try:
        state = torch.load(ckpt_path, map_location='cpu')
        params = {}
        
        if isinstance(state, dict):
            # Extract metrics from checkpoint
            if 'loss' in state:
                params['Loss'] = f"{state['loss']:.4f}"
            if 'miou' in state:
                params['mIoU'] = f"{state['miou']:.4f}"
            if 'F1' in state:
                params['F1'] = f"{state['F1']:.4f}"
            
            # Extract training config if available
            if 'train_config' in state:
                train_config = state['train_config']
                if 'lr' in train_config:
                    params['LR'] = train_config['lr']
                if 'batch_size' in train_config:
                    params['Batch'] = train_config['batch_size']
                if 'epochs' in train_config:
                    params['Epochs'] = train_config['epochs']
        
        return params if params else None
    except Exception as e:
        print(f"Could not extract training params: {e}")
        return None

def visualize_results(rgb, nir, pred_mask, gt_mask, model_name, training_params, save_path="visualization.png"):
    """Create visualization with RGB, NIR, predicted mask, and ground truth"""
    
    # Denormalize RGB for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_display = rgb * std + mean
    rgb_display = np.clip(rgb_display, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Title with model info and training params
    title = f"Model: {model_name}"
    if training_params:
        param_str = " | ".join([f"{k}: {v}" for k, v in training_params.items()])
        title += f"\n{param_str}"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # RGB Image
    axes[0, 0].imshow(rgb_display)
    axes[0, 0].set_title('RGB Input', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # NIR Image
    axes[0, 1].imshow(nir, cmap='gray')
    axes[0, 1].set_title('NIR Input', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Predicted Mask
    axes[1, 0].imshow(pred_mask, cmap='jet')
    axes[1, 0].set_title('Predicted Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Ground Truth
    if gt_mask is not None:
        axes[1, 1].imshow(gt_mask, cmap='jet')
        axes[1, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Ground Truth\nNot Available', 
                        ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()

def run_inference(model, config, device="cuda", nir_drop=False):
    """Run inference using configuration, with optional NIR zero-out probability p"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load checkpoint
    ckpt_path = config['inference']['checkpoint_path']
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Extract training parameters
    training_params = extract_training_params(ckpt_path)
    img_path = None
    nir_path = None
    gt_path = None
    # --- Random sample support ---
    if config['inference'].get('use_random', False):
        data_root = config['inference']['data_root']
        rgb_dir = os.path.join(data_root, "RGB")
        nir_dir = os.path.join(data_root, "Multispectral")
        mask_dir = os.path.join(data_root, "Masks")

        rgb_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        if not rgb_files:
            raise RuntimeError(f"No RGB images found in {rgb_dir}")

        chosen = random.choice(rgb_files)
        base_name, _ = os.path.splitext(chosen)

        img_path = os.path.join(rgb_dir, chosen)
        nir_path = os.path.join(nir_dir, base_name + "_NIR.TIF")   # adjust suffix if needed
        gt_path  = os.path.join(mask_dir, base_name + ".png")      # adjust extension if needed
    else:
        img_path = config['inference']['rgb_image_path']
        nir_path = config['inference']['nir_image_path']
        gt_path  = config['inference'].get('ground_truth_path', None)
    # -----------------------------

    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB image not found: {img_path}")
    
    # Get target size from config
    target_hw = tuple(config['inference'].get('target_size', [960, 1280]))
    
    x, rgb, nir_img = preprocess_rgbnir(bgr, nir_path, target_hw=target_hw)
    
    # Zero out NIR channel with probability p
    if nir_drop:
        x[:, 3, :, :] = 0.0  # Zero out NIR channel (channel index 3)
        nir_img = np.zeros_like(nir_img)
    
    x = x.to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    # Get threshold from config
    threshold = config['inference'].get('threshold', 0.5)
    mask = (probs > threshold).astype(np.uint8) * 255
    
    # Save output mask
    out_path = config['inference']['output_path'] + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{config['model']['type']}_{'droppedNIR' if config['inference']['nir_drop'] else 'withNIR'}.png"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    cv2.imwrite(out_path, mask)
    print(f"Saved mask: {out_path}")

    if gt_path:
        gt_mask = load_ground_truth(gt_path, target_hw=target_hw)
    
    # Get model name
    model_name = f"UNet (in_ch={config['model']['in_channels']}, base_ch={config['model']['base_channels']})"
    
    # Create visualization
    viz_path = out_path.replace('.png', '_visualization.png')
    visualize_results(rgb, nir_img, mask, gt_mask, model_name, training_params, viz_path)
    
    return out_path

def main():
    config = load_config("infer_config.yaml")
    
    # Create model from config
    model = UNet(
        in_channels=config['model']['in_channels'],
        base_ch=config['model']['base_channels'],
        out_channels=config['model']['out_channels']
    )
    
    # Run inference
    device = config['inference'].get('device', 'cuda')
    out_path = run_inference(model, config, device=device, nir_drop=True)
    
    print("Inference complete!")

if __name__ == "__main__":
    main()