import os
import cv2
import torch
import numpy as np

from dual_encoder.updated_architecture import DualEncoderAFFNet


class Inference:
    def __init__(self, checkpoint, device="cuda", target_size=(640, 640)):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_size = target_size

        print(f"[INFO] Loading model: {checkpoint}")

        self.model = DualEncoderAFFNet(
            rgb_variant='small',
            nir_base_ch=20,
            num_classes=1,
            embed_dim=96
        ).to(self.device)

        ckpt = torch.load(checkpoint, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print("[INFO] Model loaded")

        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _scale_uint(self, img):
        if img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
        return img.astype(np.float32) / 255.0

    def preprocess(self, rgb_path, nir_path):
        # --- Load ---
        rgb = cv2.imread(rgb_path)
        nir = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)

        if rgb is None:
            raise FileNotFoundError(rgb_path)
        if nir is None:
            raise FileNotFoundError(nir_path)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if nir.ndim == 3:
            nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)

        orig_size = rgb.shape[:2]

        # --- Resize (same as training) ---
        H, W = self.target_size
        rgb = cv2.resize(rgb, (W, H))
        nir = cv2.resize(nir, (W, H))

        # --- Scale ---
        rgb = self._scale_uint(rgb)
        nir = self._scale_uint(nir)

        # --- Normalize RGB ---
        rgb = (rgb - self.rgb_mean) / self.rgb_std

        # --- To tensor ---
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
        nir = torch.from_numpy(nir[None, ...]).float().unsqueeze(0)

        print("NIR dtype:", nir.dtype)
        print("NIR min/max:", nir.min(), nir.max())
        return rgb.to(self.device), nir.to(self.device), orig_size

    @torch.no_grad()
    def run(self, rgb_path, nir_path, out_dir="output", threshold=0.5):
        os.makedirs(out_dir, exist_ok=True)

        rgb_t, nir_t, orig_size = self.preprocess(rgb_path, nir_path)

        # --- Forward ---
        logits = self.model(rgb_t, nir_t)
        ()
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        print(f"[DEBUG] prob stats → min: {probs.min():.4f}, max: {probs.max():.4f}, mean: {probs.mean():.4f}")

        mask = (probs > threshold).astype(np.uint8)

        # --- Resize back ---
        H0, W0 = orig_size
        mask_resized = cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_NEAREST)
        probs_resized = cv2.resize(probs, (W0, H0))

        # --- Save ---
        base = os.path.splitext(os.path.basename(rgb_path))[0]

        cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"), mask_resized * 255)

        heatmap = cv2.applyColorMap((probs_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_dir, f"{base}_prob.png"), heatmap)

        # --- Overlay ---
        rgb_orig = cv2.imread(rgb_path)
        overlay = rgb_orig.copy()

        overlay[mask_resized == 1] = [0, 0, 255]  # red

        cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), overlay)

        # --- Stats ---
        weed = mask_resized.sum()
        total = mask_resized.size
        print(f"\n[RESULT]")
        print(f"Weed %: {weed/total*100:.2f}%")

        return mask_resized, probs_resized


# ================== RUN ==================

if __name__ == "__main__":
    CHECKPOINT = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/dual_encoder_20260322_193044/checkpoints/best_model.pth"
    RGB = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET/rgb/rgb_bonirob_2016-05-23-10-52-28_3_frame34_vflip.png"
    NIR = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET/nir/nir_bonirob_2016-05-23-10-52-28_3_frame34_vflip.png"

    infer = Inference(CHECKPOINT)
    infer.run(RGB, NIR)

# def main():
#     parser = argparse.ArgumentParser(description="Dual-Encoder RGB-NIR Inference")
    
#     parser.add_argument('--checkpoint', type=str,
#                        help='Path to trained model checkpoint (.pth)'
#                        , default= "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/dual_encoder_20260322_193044/checkpoints/best_model.pth")
#     parser.add_argument('--rgb', type=str,
#                        help='Path to RGB input image',
#                        default= "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET/rgb/rgb_bonirob_2016-05-23-10-52-28_3_frame34_vflip.png")
#     parser.add_argument('--nir', type=str,
#                        help='Path to NIR input image',
#                        default="/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET/nir/nir_bonirob_2016-05-23-10-52-28_3_frame34_vflip.png")
#     parser.add_argument('--output_dir', type=str, default='dual_encoder/inference_output',
#                        help='Directory to save outputs (default: dual_encoder/inference_output)')
#     parser.add_argument('--threshold', type=float, default=0.5,
#                        help='Binary threshold for segmentation (default: 0.5)')
#     parser.add_argument('--target_size', type=int, nargs=2, default=(640, 640),
#                        help='Resize input to [H W], e.g., --target_size 640 640 (default: original size)')
#     parser.add_argument('--device', type=str, default='cuda',
#                        help='Device: cuda or cpu (default: cuda)')
#     parser.add_argument('--rgb_base_ch', type=int, default=32,
#                        help='RGB encoder base channels (must match training, default: 32)')
#     parser.add_argument('--nir_base_ch', type=int, default=16,
#                        help='NIR encoder base channels (must match training, default: 16)')
    
#     args = parser.parse_args()
    
#     # Validate inputs
#     if not os.path.exists(args.rgb):
#         raise FileNotFoundError(f"RGB image not found: {args.rgb}")
#     if not os.path.exists(args.nir):
#         raise FileNotFoundError(f"NIR image not found: {args.nir}")
#     if not os.path.exists(args.checkpoint):
#         raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
#     # Initialize inference
#     inference = DualEncoderInference(
#         checkpoint_path=args.checkpoint,
#         device=args.device,
#         rgb_base_ch=args.rgb_base_ch,
#         nir_base_ch=args.nir_base_ch
#     )
    
#     # Load and preprocess
#     print(f"[INFO] Loading images...")
#     target_size = tuple(args.target_size) if args.target_size else None
#     rgb_tensor, nir_tensor, original_size = inference.load_and_preprocess(
#         args.rgb, args.nir, target_size=target_size
#     )
#     print("RGB range:", rgb_tensor.min().item(), rgb_tensor.max().item())
#     print("NIR range:", nir_tensor.min().item(), nir_tensor.max().item())
#         # Predict
#     print(f"[INFO] Running inference...")
#     mask, probs = inference.predict(rgb_tensor, nir_tensor, threshold=args.threshold)
    
#     # Compute stats
#     weed_pixels = (mask == 1).sum()
#     crop_pixels = (mask == 0).sum()
#     total_pixels = mask.size
#     weed_pct = weed_pixels / total_pixels * 100
    
#     print(f"\n[RESULTS]")
#     print(f"  Weed coverage: {weed_pct:.2f}% ({weed_pixels} pixels)")
#     print(f"  Crop coverage: {100-weed_pct:.2f}% ({crop_pixels} pixels)")
    
#     # Save outputs
#     inference.save_outputs(args.rgb, mask, probs, args.output_dir)
    
#     print(f"\n[DONE] All outputs saved to: {args.output_dir}/")


# if __name__ == "__main__":
#     main()
