import argparse
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------
def build_session(onnx_path, device):
    providers = ['CPUExecutionProvider']
    if device.lower() in ('cuda', 'gpu'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess

def preprocess_image(img_path, size):
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return img, arr  # return original PIL and normalized CHW

def decode_segmentation(pred_mask, class_colors):
    rgb_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        rgb_mask[pred_mask == class_id] = color
    return rgb_mask

def overlay_mask_on_image(image, mask, alpha=0.5):
    image = np.asarray(image).astype(np.float32)
    mask = mask.astype(np.float32)
    overlay = (alpha * mask + (1 - alpha) * image).astype(np.uint8)
    return overlay

# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize ONNX segmentation output with overlay and comparison.")
    parser.add_argument('--onnx', required=True, help='Path to ONNX model.')
    parser.add_argument('--image', required=True, help='Path to input image.')
    parser.add_argument('--output-dir', default='onnx_results', help='Directory to save results.')
    parser.add_argument('--size', nargs=2, type=int, default=[640, 640], help='Model input size (H W).')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--alpha', type=float, default=0.5, help='Transparency for overlay.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Setup ONNX Runtime
    sess = build_session(args.onnx, args.device)
    input_name = sess.get_inputs()[0].name

    # 2. Preprocess
    orig_img, arr = preprocess_image(args.image, size=args.size)
    input_tensor = np.expand_dims(arr, axis=0).astype(np.float32)

    # 3. Inference
    preds = sess.run(None, {input_name: input_tensor})[0]
    pred_mask = torch.argmax(torch.from_numpy(preds), dim=1).squeeze().numpy()

    # 4. Decode mask
    class_colors = {
        0: (0, 0, 0),        # background - black
        1: (0, 255, 0),      # crop - green
        2: (255, 0, 0)       # weed - red
    }
    rgb_mask = decode_segmentation(pred_mask, class_colors)
    overlay = overlay_mask_on_image(orig_img.resize(rgb_mask.shape[1::-1]), rgb_mask, alpha=args.alpha)

    # 5. Save results
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
    overlay_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
    comp_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")

    Image.fromarray(rgb_mask).save(mask_path)
    Image.fromarray(overlay).save(overlay_path)

    # 6. Visualization (side-by-side)
    # 6. Visualization (side-by-side)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(orig_img)
    ax[0].set_title("Original Image", fontsize=13, weight="bold")
    ax[0].axis("off")

    ax[1].imshow(rgb_mask)
    ax[1].set_title("Prediction (ONNX)", fontsize=13, weight="bold")
    ax[1].axis("off")

    # legend (placed above both images)
    patches = [
        mpatches.Patch(color=np.array(class_colors[0])/255.0, label="Class 0: Background"),
        mpatches.Patch(color=np.array(class_colors[1])/255.0, label="Class 1: Crop"),
        mpatches.Patch(color=np.array(class_colors[2])/255.0, label="Class 2: Weed"),
    ]

    # Adjust layout to reserve space for legend
    fig.subplots_adjust(top=0.85, wspace=0.05)
    legend = fig.legend(
        handles=patches,
        loc='upper center',
        ncol=3,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02)
    )

    # Save figure
    comp_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")
    fig.savefig(comp_path, dpi=250, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
