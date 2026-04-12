import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

# Import your components (using the exact import paths from your provided scripts)
from dual_encoder.weedcrop_classifier import CropWeedDataset, compute_metrics
from dual_encoder.updated_architecture import DualEncoderAFFNet


def load_model(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load either Stage-1 (num_classes=1) or Stage-2 (num_classes=2) model.
    Handles both checkpoint formats:
      - Direct state_dict (used by weedcrop_classifier.py)
      - Wrapped dict with 'model_state_dict' (used by train_dual_encoder.py / infer_sugarbeets.py)
    """
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model = DualEncoderAFFNet(
        rgb_variant='small',
        nir_base_ch=20,
        num_classes=num_classes,
        embed_dim=96
    ).to(device)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"[INFO] Loaded {num_classes}-class model from {checkpoint_path}")
    return model


def visualize_combined_predictions(
    model_veg,
    model_cw,
    val_loader,
    device,
    num_samples=8,
    save_dir="combined_crop_weed_inference",
    veg_threshold=0.5
):
    """Full pipeline inference:
    1. Stage 1 (veg vs bg) → binary vegetation mask
    2. Stage 2 (crop vs weed) → applied ONLY on predicted vegetation pixels
    3. Final mask: 0=crop, 1=weed, 255=background
    Saves visualizations and prints summary metrics (exactly like your classifier_infer.py)
    """
    os.makedirs(save_dir, exist_ok=True)

    model_veg.eval()
    model_cw.eval()

    crop_ious, weed_ious = [], []
    crop_accs, weed_accs = [], []
    pix_accs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Combined Inference")):
            if batch_idx * val_loader.batch_size >= num_samples:
                break

            rgb = batch["rgb"].to(device)
            nir = batch["nir"].to(device)
            gt_mask = batch["mask"].to(device)          # 0=crop, 1=weed, 255=bg (from CropWeedDataset)

            # ===================== STAGE 1: Vegetation detection =====================
            veg_logits = model_veg(rgb, nir)                    # (B, 1, H, W)
            veg_prob = torch.sigmoid(veg_logits.squeeze(1))     # (B, H, W) probability

            # ===================== STAGE 2: Crop vs Weed =====================
            cw_logits = model_cw(rgb, nir)                      # (B, 2, H, W)
            pred_cw = torch.argmax(torch.softmax(cw_logits, dim=1), dim=1)  # (B, H, W) 0 or 1

            # ===================== COMBINE: Final mask =====================
            veg_mask = (veg_prob > veg_threshold)               # (B, H, W) bool
            final_pred = torch.full_like(pred_cw, 255, dtype=torch.long)
            final_pred[veg_mask] = pred_cw[veg_mask]

            # Metrics (uses your exact compute_metrics – only evaluates on true vegetation pixels)
            ious, accs, pix_acc = compute_metrics(final_pred, gt_mask)

            crop_ious.append(ious['crop'])
            weed_ious.append(ious['weed'])
            crop_accs.append(accs['crop'])
            weed_accs.append(accs['weed'])
            pix_accs.append(pix_acc)

            # ===================== VISUALIZATION (per sample) =====================
            for i in range(rgb.shape[0]):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'Sample {batch_idx * val_loader.batch_size + i:03d} (Combined Pipeline)', fontsize=16)

                # RGB (denormalized)
                rgb_img = rgb[i].cpu().numpy().transpose(1, 2, 0)
                rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                rgb_img = np.clip(rgb_img, 0, 1)
                axes[0, 0].imshow(rgb_img)
                axes[0, 0].set_title('RGB')
                axes[0, 0].axis('off')

                # NIR
                nir_img = nir[i, 0].cpu().numpy()
                nir_img = cv2.normalize(nir_img, None, 0, 1, cv2.NORM_MINMAX)
                axes[0, 1].imshow(nir_img, cmap='gray')
                axes[0, 1].set_title('NIR')
                axes[0, 1].axis('off')

                # GT Mask
                gt_np = gt_mask[i].cpu().numpy()
                gt_vis = np.zeros((*gt_np.shape, 3))
                gt_vis[gt_np == 0] = [0, 1, 0]      # Green = crop
                gt_vis[gt_np == 1] = [1, 0, 0]      # Red = weed
                gt_vis[gt_np == 255] = [0.5, 0.5, 0.5]  # Gray = background
                axes[0, 2].imshow(gt_vis)
                axes[0, 2].set_title('GT Mask\n(Green=crop, Red=weed, Gray=bg)')
                axes[0, 2].axis('off')

                # Final Predicted Mask
                pred_np = final_pred[i].cpu().numpy()
                pred_vis = np.zeros((*pred_np.shape, 3))
                pred_vis[pred_np == 0] = [0, 1, 0]      # Green = crop
                pred_vis[pred_np == 1] = [1, 0, 0]      # Red = weed
                pred_vis[pred_np == 255] = [0.5, 0.5, 0.5]  # Gray = background
                axes[1, 0].imshow(pred_vis)
                axes[1, 0].set_title('Final Pred Mask\n(Green=crop, Red=weed, Gray=bg)')
                axes[1, 0].axis('off')

                # RGB + predicted weed overlay
                overlay = rgb_img.copy()
                overlay[pred_np == 1] = [1, 0.3, 0.3]   # semi-transparent red on weeds
                axes[1, 1].imshow(overlay)
                axes[1, 1].set_title('RGB + Pred Overlay\n(Red=predicted weed)')
                axes[1, 1].axis('off')

                # Metrics box
                axes[1, 2].axis('off')
                metrics_text = f"""Metrics (veg pixels only):
Crop IoU : {ious['crop']:.3f}
Weed IoU : {ious['weed']:.3f}
Crop Acc : {accs['crop']:.3f}
Weed Acc : {accs['weed']:.3f}
Pixel Acc: {pix_acc:.3f}
Veg thresh: {veg_threshold:.2f}"""
                axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                                fontsize=11, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                plt.tight_layout()
                save_path = f"{save_dir}/sample_{batch_idx * val_loader.batch_size + i:03d}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

    # ===================== SUMMARY =====================
    n_samples = len(crop_ious)
    print(f"\n=== COMBINED PIPELINE INFERENCE SUMMARY ({n_samples} samples) ===")
    print(f"Avg Crop IoU : {np.mean(crop_ious):.4f} ± {np.std(crop_ious):.4f}")
    print(f"Avg Weed IoU : {np.mean(weed_ious):.4f} ± {np.std(weed_ious):.4f}")
    print(f"Mean IoU     : {np.mean([np.mean(crop_ious), np.mean(weed_ious)]):.4f}")
    print(f"Avg Crop Acc : {np.mean(crop_accs):.4f}")
    print(f"Avg Weed Acc : {np.mean(weed_accs):.4f}")
    print(f"Avg Pixel Acc: {np.mean(pix_accs):.4f}")
    print(f"Results saved to: {save_dir}/")
    print(f"[INFO] Final mask legend: Green=crop, Red=weed, Gray=background")


def main():
    parser = argparse.ArgumentParser(description="Combined Two-Stage Crop vs Weed Inference Pipeline")
    parser.add_argument("--data_root", required=True, help="Path to SUGARBEETS_AUGMENTED_DATASET")
    parser.add_argument("--model_veg_path", required=True, help="Stage 1 vegetation model checkpoint (num_classes=1)")
    parser.add_argument("--model_cw_path", required=True, help="Stage 2 crop-weed model checkpoint (num_classes=2)")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of validation samples to visualize")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--img_size", nargs=2, default=[512, 512], type=int, help="Image size H W (must match training)")
    parser.add_argument("--veg_threshold", type=float, default=0.5, help="Threshold for Stage-1 vegetation mask")
    parser.add_argument("--save_dir", default="combined_crop_weed_inference", help="Output directory for results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use your existing Stage-2 dataset (CropWeedDataset) – it provides the exact GT we need for final evaluation
    val_dataset = CropWeedDataset(
        root=args.data_root,
        split="val",
        target_size=tuple(args.img_size),
        augment=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Validation set size: {len(val_dataset)}")

    # Load both models
    model_veg = load_model(args.model_veg_path, num_classes=1, device=device)
    model_cw = load_model(args.model_cw_path, num_classes=2, device=device)

    # Run full pipeline
    visualize_combined_predictions(
        model_veg=model_veg,
        model_cw=model_cw,
        val_loader=val_loader,
        device=device,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        veg_threshold=args.veg_threshold
    )


if __name__ == "__main__":
    main()
