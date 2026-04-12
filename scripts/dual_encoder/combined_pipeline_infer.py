import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from dual_encoder.updated_architecture import DualEncoderAFFNet
from scripts.dual_encoder.dual_encoder_data_loader_veg import DualEncoderWeedyRiceDataset
from torch.utils.data import DataLoader

NUM_CLASSES = 3  # 0: BG, 1: Crop, 2: Weed

def denorm_rgb(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = (img * std + mean).clip(0, 1)
    return img

def post_process_mask(mask, min_size=60):
    """Refines 'minute' noise and fills holes."""
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    processed_mask = np.zeros_like(mask)
    for cls_val in [1, 2]:
        cls_mask   = (mask == cls_val).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                processed_mask[labels == i] = cls_val
    return processed_mask

# ─────────────────────────────────────────────
#  Per-image metric helpers
# ─────────────────────────────────────────────
def compute_confusion(pred, gt, num_classes=NUM_CLASSES):
    """Returns (num_classes, num_classes) confusion matrix C[true, pred]."""
    C = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = (gt >= 0) & (gt < num_classes)
    np.add.at(C, (gt[mask], pred[mask]), 1)
    return C

def metrics_from_confusion(C):
    """
    From a (K, K) confusion matrix compute per-class and global metrics.
    Returns a dict with:
        iou       : (K,) per-class IoU
        acc       : (K,) per-class accuracy (recall)
        f1        : (K,) per-class F1
        precision : (K,) per-class precision
        recall    : (K,) per-class recall  (= acc)
        global_acc: scalar  overall pixel accuracy
        mIoU      : scalar  mean IoU
        mClassAcc : scalar  mean class accuracy
        mPrec     : scalar  mean precision
        mRec      : scalar  mean recall
    """
    eps  = 1e-6
    K    = C.shape[0]
    tp   = np.diag(C)
    fp   = C.sum(axis=0) - tp       # predicted as class c but not actually c
    fn   = C.sum(axis=1) - tp       # actually class c but not predicted as c

    iou       = tp / (tp + fp + fn + eps)
    recall    = tp / (tp + fn + eps)   # = per-class acc
    precision = tp / (tp + fp + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    global_acc = tp.sum() / (C.sum() + eps)
    mIoU       = iou.mean()
    mClassAcc  = recall.mean()
    mPrec      = precision.mean()
    mRec       = recall.mean()

    return dict(
        iou=iou, acc=recall, f1=f1,
        precision=precision, recall=recall,
        global_acc=global_acc,
        mIoU=mIoU, mClassAcc=mClassAcc,
        mPrec=mPrec, mRec=mRec,
    )

# ─────────────────────────────────────────────
#  Pretty-print / save helpers
# ─────────────────────────────────────────────
CLASS_NAMES = ["BG", "Crop", "Weed"]

def format_block(tag, m):
    """Format a metrics dict the same way as the training logger."""
    iou_str  = "/".join(f"{v:.3f}" for v in m["iou"])
    acc_str  = "/".join(f"{v:.3f}" for v in m["acc"])
    f1_str   = "/".join(f"{v:.3f}" for v in m["f1"])
    lines = [
        f"  {tag:<6} - mIoU: {m['mIoU']:.4f} | Global Acc: {m['global_acc']:.4f} | mClassAcc: {m['mClassAcc']:.4f}",
        f"          IoU [BG/Crop/Weed]: [{iou_str}]",
        f"          Acc [BG/Crop/Weed]: [{acc_str}]",
        f"          F1  [BG/Crop/Weed]: [{f1_str}]",
        f"          Mean Prec/Rec: {m['mPrec']:.4f}/{m['mRec']:.4f}",
    ]
    return "\n".join(lines)

def save_summary(summary_h, summary_s, out_dir, n_images):
    """Aggregate per-image confusion matrices → global metrics, then save."""
    def agg(summary):
        C = sum(summary["conf"])           # sum all (3,3) confusion matrices
        m = metrics_from_confusion(C)
        return m

    mh = agg(summary_h)
    ms = agg(summary_s)

    header = (
        "=" * 60 + "\n"
        "FINAL COMPARISON SUMMARY  (aggregated over all val images)\n"
        f"Images evaluated: {n_images}\n"
        "=" * 60
    )
    body = (
        "\n── HARSH APPROACH (binary vegetation gate) ──\n"
        + format_block("Harsh", mh)
        + "\n\n── SOFT FUSION + POST-PROCESS ──\n"
        + format_block("Soft", ms)
        + "\n\n" + "=" * 60
    )
    full = header + "\n" + body

    print("\n" + full)

    txt_path = os.path.join(out_dir, "comparison_summary.txt")
    with open(txt_path, "w") as f:
        f.write(full + "\n")
    print(f"\nSummary saved → {txt_path}")

    # Also save a compact CSV for quick loading
    import csv
    csv_path = os.path.join(out_dir, "comparison_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["approach", "mIoU", "global_acc", "mClassAcc",
                         "IoU_BG", "IoU_Crop", "IoU_Weed",
                         "Acc_BG", "Acc_Crop", "Acc_Weed",
                         "F1_BG",  "F1_Crop",  "F1_Weed",
                         "mPrec",  "mRec"])
        for name, m in [("Harsh", mh), ("Soft+PP", ms)]:
            writer.writerow([
                name,
                f"{m['mIoU']:.4f}", f"{m['global_acc']:.4f}", f"{m['mClassAcc']:.4f}",
                *[f"{v:.4f}" for v in m["iou"]],
                *[f"{v:.4f}" for v in m["acc"]],
                *[f"{v:.4f}" for v in m["f1"]],
                f"{m['mPrec']:.4f}", f"{m['mRec']:.4f}",
            ])
    print(f"CSV saved      → {csv_path}")


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────
def run_comparison_pipeline(veg_ckpt, cls_ckpt, data_root,
                            out_dir="./dual_encoder/soft_fusion_results"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    # 1. LOAD MODELS
    model1 = DualEncoderAFFNet(rgb_variant="small", nir_base_ch=20,
                                num_classes=1, embed_dim=96).to(device)
    model1.load_state_dict(
        torch.load(veg_ckpt, map_location=device)["model_state_dict"])

    model2 = DualEncoderAFFNet(rgb_variant="small", nir_base_ch=20,
                                num_classes=2, embed_dim=96).to(device)
    model2.load_state_dict(torch.load(cls_ckpt, map_location=device))
    model1.eval(); model2.eval()

    val_ds     = DualEncoderWeedyRiceDataset(data_root, split="val",
                                              target_size=(640, 640), augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # accumulators: store per-image confusion matrices
    summary_h = defaultdict(list)   # harsh
    summary_s = defaultdict(list)   # soft

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Full Pipeline")):
            rgb_t  = batch["rgb"].to(device)
            nir_t  = batch["nir"].to(device)
            img_path = batch["path"][0]
            mask_id  = Path(img_path).stem.replace("rgb_", "mask_")

            # Load 3-class GT
            mask_path = os.path.join(data_root, "masks", f"{mask_id}.png")
            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (640, 640), interpolation=cv2.INTER_NEAREST)

            # ── INFERENCE ───────────────────────────────────────
            prob_veg  = torch.sigmoid(model1(rgb_t, nir_t)).squeeze()
            probs_cls = F.softmax(model2(rgb_t, nir_t), dim=1).squeeze()

            # A. HARSH APPROACH (binary gate)
            mask_veg_binary = (prob_veg > 0.5).int()
            pred_harsh_raw  = (torch.argmax(probs_cls, dim=0) + 1) * mask_veg_binary
            pred_harsh = pred_harsh_raw.cpu().numpy().astype(np.uint8)

            # B. SOFT FUSION + POST-PROCESS
            final_prob_crop = prob_veg * probs_cls[0]
            final_prob_weed = prob_veg * probs_cls[1]
            stacked = torch.stack([final_prob_crop, final_prob_weed])
            max_v, cls_idx = torch.max(stacked, dim=0)

            pred_soft_raw = np.zeros((640, 640), dtype=np.uint8)
            mask_conf = (max_v > 0.5).cpu().numpy()
            pred_soft_raw[mask_conf] = (cls_idx[mask_conf] + 1).cpu().numpy()
            pred_soft = post_process_mask(pred_soft_raw, min_size=50)

            # ── METRICS (per image confusion matrix) ─────────────
            Ch = compute_confusion(pred_harsh, gt)
            Cs = compute_confusion(pred_soft,  gt)
            summary_h["conf"].append(Ch)
            summary_s["conf"].append(Cs)

            # Per-image scalars for logging / per-image txt
            mh_img = metrics_from_confusion(Ch)
            ms_img = metrics_from_confusion(Cs)

            # ── PER-IMAGE CONSOLE LOG ────────────────────────────
            print(f"\n[{i+1:03d}] {mask_id}")
            print(format_block("Harsh", mh_img))
            print(format_block("Soft ", ms_img))

            # ── VISUALIZATION GRID (2×4) ─────────────────────────
            fig, axes = plt.subplots(2, 4, figsize=(26, 13))
            cmap = plt.cm.colors.ListedColormap(['black', 'green', 'red'])

            axes[0, 0].imshow(denorm_rgb(batch["rgb"][0]))
            axes[0, 0].set_title("Input RGB")

            axes[0, 1].imshow(batch["nir"][0].squeeze().cpu().numpy(), cmap='gray')
            axes[0, 1].set_title("Input NIR")

            im1 = axes[0, 2].imshow(prob_veg.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            axes[0, 2].set_title("Stage 1: Vegetation Prob")
            plt.colorbar(im1, ax=axes[0, 2])

            im2 = axes[0, 3].imshow(probs_cls[1].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            axes[0, 3].set_title("Stage 2: Weed Prob (within Veg)")
            plt.colorbar(im2, ax=axes[0, 3])

            axes[1, 0].imshow(gt, cmap=cmap, vmin=0, vmax=2)
            axes[1, 0].set_title("Ground Truth (0,1,2)")

            axes[1, 1].imshow(pred_harsh, cmap=cmap, vmin=0, vmax=2)
            axes[1, 1].set_title(
                f"Harsh Approach\n"
                f"mIoU: {mh_img['mIoU']:.3f} | Weed IoU: {mh_img['iou'][2]:.3f}")

            axes[1, 2].imshow(pred_soft, cmap=cmap, vmin=0, vmax=2)
            axes[1, 2].set_title(
                f"Soft + Post-Process\n"
                f"mIoU: {ms_img['mIoU']:.3f} | Weed IoU: {ms_img['iou'][2]:.3f}")

            ov      = denorm_rgb(batch["rgb"][0]).copy()
            ov_mask = np.zeros_like(ov)
            ov_mask[pred_soft == 1] = [0, 1, 0]
            ov_mask[pred_soft == 2] = [1, 0, 0]
            axes[1, 3].imshow(cv2.addWeighted(ov, 0.7, ov_mask, 0.3, 0))
            axes[1, 3].set_title("Final Soft Overlay")

            for ax in axes.ravel():
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{mask_id}_comp.png"))
            plt.close()

    # ── FINAL AGGREGATED SUMMARY ─────────────────────────────────
    save_summary(summary_h, summary_s, out_dir, n_images=len(val_loader))


if __name__ == "__main__":
    run_comparison_pipeline(
        veg_ckpt="/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/"
                 "dual_encoder_20260324_115825/checkpoints/best_model.pth",
        cls_ckpt="best_crop_weed.pth",
        data_root="/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET",
    )