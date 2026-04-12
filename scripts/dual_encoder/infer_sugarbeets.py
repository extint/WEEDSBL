"""
infer_val.py — Run inference on the val split using the existing dataloader.

Usage:
    python -m dual_encoder.infer_val \
        --checkpoint dual_encoder/runs/<run>/checkpoints/best_model.pth \
        --data_root  /path/to/SUGARBEETS_AUGMENTED_DATASET \
        --output_dir ./inference_output \
        --mask_mode  vegetation
"""

import argparse
import os
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from dual_encoder.updated_architecture import DualEncoderAFFNet
from scripts.dual_encoder.dual_encoder_data_loader_veg import DualEncoderWeedyRiceDataset

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = DualEncoderAFFNet(
        rgb_variant="small", nir_base_ch=20, num_classes=1, embed_dim=96
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch   = ckpt.get("epoch", "?")
    val_iou = ckpt.get("metrics", {}).get("val_iou", float("nan"))
    print(f"[INFO] Loaded checkpoint — epoch {epoch}, val IoU {val_iou:.4f}")
    return model


def denorm_rgb(rgb_t):
    """(3,H,W) normalised tensor → (H,W,3) float32 in [0,1]"""
    rgb = rgb_t.cpu().numpy().transpose(1, 2, 0)
    rgb = rgb * RGB_STD + RGB_MEAN
    return np.clip(rgb, 0, 1)


def save_grid(rgb_vis, nir_vis, gt_mask, prob_map, threshold, save_path):
    mask_pred = (prob_map > threshold).astype(np.uint8)

    overlay = rgb_vis.copy()
    overlay[mask_pred == 1] = overlay[mask_pred == 1] * 0.35 + np.array([0.9, 0.1, 0.1]) * 0.65

    gt_color = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    gt_color[gt_mask == 1] = [0.9, 0.1, 0.1]
    gt_color[gt_mask == 0] = [0.1, 0.8, 0.1]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(rgb_vis);                                axes[0].set_title("RGB");             axes[0].axis("off")
    axes[1].imshow(nir_vis, cmap="gray");                   axes[1].set_title("NIR");             axes[1].axis("off")
    axes[2].imshow(gt_color);                               axes[2].set_title("Ground Truth");    axes[2].axis("off")
    im = axes[3].imshow(prob_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[3].set_title("Prob Map"); axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    axes[4].imshow(np.clip(overlay, 0, 1));                 axes[4].set_title("Pred Overlay");   axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def compute_iou(prob, gt, threshold=0.5):
    pred = (prob > threshold).astype(np.float32)
    gt   = gt.astype(np.float32)
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return float((intersection + 1e-6) / (union + 1e-6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--output_dir",  default="./inference_output")
    parser.add_argument("--size",        type=int,   default=640)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--mask_mode",   type=str,   default="vegetation", choices=["vegetation", "weed_only"],
                        help="Which mask logic to use (vegetation for Stage 1, weed_only for Stage 2)")
    parser.add_argument("--num_samples", type=int,   default=-1,
                        help="How many val images to visualise (-1 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Mask Mode: {args.mask_mode}")

    # ── model ────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── dataloader (val split, no augmentation, explicitly passing mask_mode) ──
    val_ds = DualEncoderWeedyRiceDataset(
        root=args.data_root,
        split="val",
        target_size=(args.size, args.size),
        augment=False,
        mask_mode=args.mask_mode  # <-- NEW: Using the correct mask logic from argparse
    )
    print(f"[INFO] Val set: {len(val_ds)} samples")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ious = []
    n = args.num_samples if args.num_samples > 0 else len(val_ds)

    for i in tqdm(range(min(n, len(val_ds))), desc="Inferring"):
        batch = val_ds[i]

        rgb_t  = batch["rgb"].unsqueeze(0).to(device)   # (1,3,H,W)
        nir_t  = batch["nir"].unsqueeze(0).to(device)   # (1,1,H,W)
        gt     = batch["mask"].numpy()                   # (H,W)
        img_path = batch["path"]

        with torch.no_grad():
            logits = model(rgb_t, nir_t)                 # (1,1,H,W)
            prob   = torch.sigmoid(logits.squeeze()).cpu().numpy()  # (H,W)

        iou = compute_iou(prob, gt, args.threshold)
        ious.append(iou)

        # visualise
        rgb_vis = denorm_rgb(batch["rgb"])
        nir_vis = batch["nir"].squeeze().numpy()         # (H,W) already [0,1]

        stem     = Path(img_path).stem
        out_path = out / f"{i:04d}_{stem}_iou{iou:.3f}.png"
        save_grid(rgb_vis, nir_vis, gt, prob, args.threshold, out_path)

    mean_iou = np.mean(ious)
    print(f"\n[RESULT] Mean IoU over {len(ious)} samples: {mean_iou:.4f}")
    print(f"[RESULT] Per-sample IoUs: min={min(ious):.4f}  max={max(ious):.4f}")
    print(f"[INFO]   Results saved to: {out}")


if __name__ == "__main__":
    main()