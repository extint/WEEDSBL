#!/usr/bin/env python3
"""
ONNX Inference for LightMANet (RGB only, 3 classes)
Fixed for proper logits output
"""

import argparse
import numpy as np
import cv2
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Color mapping for 3 classes
COLOR_MAP = {
    0: [0, 0, 0],      # Black - Background
    1: [0, 255, 0],    # Green - Class 1 (Weed)
    2: [0, 0, 255]     # Blue - Class 2 (Crop)
}


def preprocess_rgb(image_path, img_size=(640, 640)):
    """Preprocess RGB image for ONNX model"""
    # Load image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert to RGB and resize
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, img_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize with ImageNet stats
    rgb = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    
    # Convert to NCHW format (batch_size=1, channels=3, height, width)
    x = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, axis=0)     # Add batch dimension
    
    return np.ascontiguousarray(x, dtype=np.float32), rgb


def mask_to_rgb(mask):
    """Convert class mask to RGB visualization"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        rgb[mask == class_id] = color
    
    return rgb


def run_onnx_inference(onnx_path, image_path, img_size=(640, 640), save_path=None):
    """Run ONNX inference and visualize results"""
    
    print(f"\n{'='*60}")
    print(f"Running ONNX Inference")
    print(f"{'='*60}")
    
    # Preprocess image
    print(f"Loading: {image_path}")
    x, rgb_original = preprocess_rgb(image_path, img_size)
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Create ONNX session
    print(f"\nLoading ONNX model: {onnx_path}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    
    # Get input/output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    print(f"Input name: {input_name}, shape: {sess.get_inputs()[0].shape}")
    print(f"Output name: {output_name}, shape: {sess.get_outputs()[0].shape}")
    print(f"Using provider: {sess.get_providers()[0]}")
    
    # Run inference
    print("\nRunning inference...")
    outputs = sess.run([output_name], {input_name: x})
    
    # Process output (logits: [batch, num_classes, H, W])
    logits = outputs[0]
    print(f"Output shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    # Debug: Check logits per class
    print("\nLogits statistics per class:")
    for cls in range(3):
        cls_logits = logits[0, cls]
        print(f"  Class {cls}: min={cls_logits.min():.2f}, max={cls_logits.max():.2f}, mean={cls_logits.mean():.2f}")
    
    # Get class predictions (argmax over class dimension)
    if len(logits.shape) == 4:  # [batch, num_classes, H, W]
        pred_mask = np.argmax(logits[0], axis=0)  # [H, W]
    else:
        pred_mask = np.argmax(logits, axis=0)
    
    # Print prediction statistics
    print(f"\nPrediction shape: {pred_mask.shape}")
    print(f"Unique classes: {np.unique(pred_mask)}")
    print("Pixel distribution:")
    for cls in range(3):
        count = (pred_mask == cls).sum()
        percent = count / pred_mask.size * 100
        print(f"  Class {cls}: {count:,} pixels ({percent:.2f}%)")
    
    # Denormalize original image for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = std * rgb_original + mean
    img_display = np.clip(img_display * 255, 0, 255).astype(np.uint8)
    
    # Convert prediction to RGB
    pred_rgb = mask_to_rgb(pred_mask)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(pred_rgb)
    axes[1].set_title('Prediction (ONNX)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=np.array(COLOR_MAP[0])/255, label='Class 0 (Background)'),
        mpatches.Patch(color=np.array(COLOR_MAP[1])/255, label='Class 1 (Crop)'),
        mpatches.Patch(color=np.array(COLOR_MAP[2])/255, label='Class 2 (Weed)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12)
    
    # Add model info
    info_text = f"Model: LightMANet | Validation mIoU: 88.86%"
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to: {save_path}")
    
    plt.show()
    
    # Save masks separately
    if save_path:
        # Raw mask (0, 1, 2 values)
        mask_path = save_path.replace('.png', '_mask.png')
        cv2.imwrite(mask_path, pred_mask.astype(np.uint8))
        print(f"✓ Saved raw mask to: {mask_path}")
        
        # Color-coded mask
        color_mask_path = save_path.replace('.png', '_color_mask.png')
        cv2.imwrite(color_mask_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved color mask to: {color_mask_path}")
    
    print(f"{'='*60}\n")
    
    return pred_mask


def main():
    parser = argparse.ArgumentParser(description='ONNX Inference for LightMANet')
    
    parser.add_argument('--onnx_model', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--save_path', type=str, default='./onnx_inference_result.png',
                        help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Run inference
    run_onnx_inference(
        onnx_path=args.onnx_model,
        image_path=args.image,
        img_size=(args.img_size, args.img_size),
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
