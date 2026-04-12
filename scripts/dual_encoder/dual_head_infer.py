# """
# Inference + Evaluation for SharedDualHeadNetV2
# """

# import os
# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from pathlib import Path
# from tqdm import tqdm
# from collections import defaultdict

# from dual_encoder.train_shared_dual_encoder import SharedDualHeadNet
# from dual_encoder.dual_encoder_data_loader import DualEncoderWeedyRiceDataset
# from torch.utils.data import DataLoader

# NUM_CLASSES = 3  # 0=BG, 1=Crop, 2=Weed
# CLASS_NAMES = ["BG", "Crop", "Weed"]


# # ── Helpers (identical to combined_pipeline_infer.py) ────────────────────────

# def denorm_rgb(tensor):
#     mean = np.array([0.485, 0.456, 0.406])
#     std  = np.array([0.229, 0.224, 0.225])
#     img  = tensor.permute(1, 2, 0).cpu().numpy()
#     return (img * std + mean).clip(0, 1)


# def post_process_mask(mask, min_size=50):
#     kernel = np.ones((3, 3), np.uint8)
#     mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     out = np.zeros_like(mask)
#     for cls_val in [1, 2]:
#         cm = (mask == cls_val).astype(np.uint8)
#         n, labels, stats, _ = cv2.connectedComponentsWithStats(cm)
#         for i in range(1, n):
#             if stats[i, cv2.CC_STAT_AREA] >= min_size:
#                 out[labels == i] = cls_val
#     return out


# def compute_confusion(pred, gt, num_classes=NUM_CLASSES):
#     C = np.zeros((num_classes, num_classes), dtype=np.int64)
#     mask = (gt >= 0) & (gt < num_classes)
#     np.add.at(C, (gt[mask], pred[mask]), 1)
#     return C


# def metrics_from_confusion(C):
#     eps = 1e-6
#     tp  = np.diag(C)
#     fp  = C.sum(0) - tp
#     fn  = C.sum(1) - tp
#     iou       = tp / (tp + fp + fn + eps)
#     recall    = tp / (tp + fn + eps)
#     precision = tp / (tp + fp + eps)
#     f1        = 2 * precision * recall / (precision + recall + eps)
#     return dict(
#         iou=iou, acc=recall, f1=f1,
#         precision=precision, recall=recall,
#         global_acc=tp.sum() / (C.sum() + eps),
#         mIoU=iou.mean(), mClassAcc=recall.mean(),
#         mPrec=precision.mean(), mRec=recall.mean(),
#     )


# def format_block(tag, m):
#     iou_s = "/".join(f"{v:.3f}" for v in m["iou"])
#     acc_s = "/".join(f"{v:.3f}" for v in m["acc"])
#     f1_s  = "/".join(f"{v:.3f}" for v in m["f1"])
#     return "\n".join([
#         f"  {tag:<8} - mIoU: {m['mIoU']:.4f} | Global Acc: {m['global_acc']:.4f} | mClassAcc: {m['mClassAcc']:.4f}",
#         f"           IoU [BG/Crop/Weed]: [{iou_s}]",
#         f"           Acc [BG/Crop/Weed]: [{acc_s}]",
#         f"           F1  [BG/Crop/Weed]: [{f1_s}]",
#         f"           Mean Prec/Rec: {m['mPrec']:.4f}/{m['mRec']:.4f}",
#     ])


# # ── Main ──────────────────────────────────────────────────────────────────────

# def run_shared_inference(
#     ckpt_path: str,
#     data_root: str,
#     out_dir: str = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/shared_dual_head_results",
#     veg_thresh: float = 0.5,
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     os.makedirs(out_dir, exist_ok=True)

#     # Load model
#     model = SharedDualHeadNet(rgb_variant="small", nir_base_ch=20, embed_dim=96).to(device)
#     ckpt  = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(ckpt["model_state_dict"])
#     model.eval()
#     print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
#     print(f"  val_veg_iou={ckpt.get('val_veg_iou', '?'):.4f}  "
#           f"val_cls_iou={ckpt.get('val_cls_iou', '?'):.4f}")

#     val_ds = DualEncoderWeedyRiceDataset(
#         data_root, split="val", target_size=(640, 640), augment=False
#     )
#     val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

#     # Two prediction variants to compare (mirrors combined_pipeline_infer.py):
#     #   "raw"  — hard veg gate, no post-processing
#     #   "pp"   — soft veg gate + morphological post-process
#     summary_raw = defaultdict(list)
#     summary_pp  = defaultdict(list)

#     with torch.no_grad():
#         for i, batch in enumerate(tqdm(val_loader, desc="SharedDualHead")):
#             rgb_t    = batch["rgb"].to(device)
#             nir_t    = batch["nir"].to(device)
#             img_path = batch["path"][0]
#             mask_id  = Path(img_path).stem.replace("rgb_", "mask_")

#             # Load 3-class GT
#             gt = cv2.imread(
#                 os.path.join(data_root, "masks", f"{mask_id}.png"),
#                 cv2.IMREAD_GRAYSCALE,
#             )
#             gt = cv2.resize(gt, (640, 640), interpolation=cv2.INTER_NEAREST)

#             # Inference
#             veg_logits, cls_logits = model(rgb_t, nir_t)
#             veg_prob  = torch.sigmoid(veg_logits).squeeze()          # (H,W)
#             cls_probs = F.softmax(cls_logits, dim=1).squeeze()        # (2,H,W)

#             # ── Raw prediction (hard gate) ────────────────────────────────────
#             veg_bin  = (veg_prob > veg_thresh).int()
#             cls_argmax = cls_logits.squeeze().argmax(dim=0) + 1       # 1 or 2
#             pred_raw = (cls_argmax * veg_bin).cpu().numpy().astype(np.uint8)

#             # ── Post-processed prediction (soft fusion) ───────────────────────
#             soft_crop = veg_prob * cls_probs[0]
#             soft_weed = veg_prob * cls_probs[1]
#             stacked   = torch.stack([soft_crop, soft_weed])
#             max_v, cls_idx = stacked.max(dim=0)
#             pred_soft_raw = np.zeros((640, 640), dtype=np.uint8)
#             conf_mask = (max_v > 0.5).cpu().numpy()
#             pred_soft_raw[conf_mask] = (cls_idx[conf_mask] + 1).cpu().numpy()
#             pred_pp = post_process_mask(pred_soft_raw, min_size=50)

#             # Metrics
#             C_raw = compute_confusion(pred_raw, gt)
#             C_pp  = compute_confusion(pred_pp,  gt)
#             summary_raw["conf"].append(C_raw)
#             summary_pp["conf"].append(C_pp)

#             m_raw = metrics_from_confusion(C_raw)
#             m_pp  = metrics_from_confusion(C_pp)

#             print(f"\n[{i+1:03d}] {mask_id}")
#             print(format_block("Raw",  m_raw))
#             print(format_block("PP",   m_pp))

#             # Visualisation
#             fig, axes = plt.subplots(2, 4, figsize=(26, 13))
#             cmap = plt.cm.colors.ListedColormap(["black", "green", "red"])

#             axes[0, 0].imshow(denorm_rgb(batch["rgb"][0])); axes[0, 0].set_title("Input RGB")
#             axes[0, 1].imshow(batch["nir"][0].squeeze().cpu().numpy(), cmap="gray"); axes[0, 1].set_title("Input NIR")
#             im1 = axes[0, 2].imshow(veg_prob.cpu().numpy(), cmap="jet", vmin=0, vmax=1)
#             axes[0, 2].set_title("Veg Prob (head A)"); plt.colorbar(im1, ax=axes[0, 2])
#             im2 = axes[0, 3].imshow(cls_probs[1].cpu().numpy(), cmap="jet", vmin=0, vmax=1)
#             axes[0, 3].set_title("Weed Prob (head B)"); plt.colorbar(im2, ax=axes[0, 3])

#             axes[1, 0].imshow(gt, cmap=cmap, vmin=0, vmax=2); axes[1, 0].set_title("Ground Truth")
#             axes[1, 1].imshow(pred_raw, cmap=cmap, vmin=0, vmax=2)
#             axes[1, 1].set_title(f"Raw\nmIoU={m_raw['mIoU']:.3f} | WeedIoU={m_raw['iou'][2]:.3f}")
#             axes[1, 2].imshow(pred_pp, cmap=cmap, vmin=0, vmax=2)
#             axes[1, 2].set_title(f"Post-Processed\nmIoU={m_pp['mIoU']:.3f} | WeedIoU={m_pp['iou'][2]:.3f}")

#             ov = denorm_rgb(batch["rgb"][0]).copy()
#             ov_mask = np.zeros_like(ov)
#             ov_mask[pred_pp == 1] = [0, 1, 0]
#             ov_mask[pred_pp == 2] = [1, 0, 0]
#             axes[1, 3].imshow(cv2.addWeighted(ov, 0.7, ov_mask, 0.3, 0))
#             axes[1, 3].set_title("Overlay (post-processed)")

#             for ax in axes.ravel():
#                 ax.axis("off")
#             plt.tight_layout()
#             plt.savefig(os.path.join(out_dir, f"{mask_id}_shared.png"))
#             plt.close()

#     # Aggregate summary
#     def agg(summary):
#         C = sum(summary["conf"])
#         return metrics_from_confusion(C)

#     m_raw_agg = agg(summary_raw)
#     m_pp_agg  = agg(summary_pp)

#     header = (
#         "=" * 60 + "\n"
#         "SHARED DUAL-HEAD FINAL SUMMARY\n"
#         f"Images evaluated: {len(val_loader)}\n"
#         "=" * 60
#     )
#     body = (
#         "\n── RAW (hard gate) ──\n"       + format_block("Raw", m_raw_agg)
#         + "\n\n── POST-PROCESSED ──\n"    + format_block("PP",  m_pp_agg)
#         + "\n\n" + "=" * 60
#     )
#     print("\n" + header + "\n" + body)

#     with open(os.path.join(out_dir, "shared_summary.txt"), "w") as f:
#         f.write(header + "\n" + body + "\n")


# if __name__ == "__main__":
#     run_shared_inference(
#         ckpt_path="/home/vjti-comp/WEEDSBL/scripts/dual_encoder/best_shared_dual_head.pth",
#         data_root="/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET",
#     )

"""
infer_shared_dual_encoder.py
============================
Full dataset inference + evaluation for SharedDualHeadNet.

Runs on train+val (or any split you specify), saves:
  results/
    per_image/          ← one PNG visualisation per image
    metrics_per_image.csv
    summary.txt
    summary.csv
    confusion_matrix.png
    metric_distributions.png

Pixel-level metrics  (aggregated via confusion matrix)
------------------------------------------------------
  Accuracy (global pixel), Per-class accuracy
  Precision, Recall, F1     per class (BG / Crop / Weed)
  Dice coefficient           per class
  Boundary F1 (trimap)       per class  ← less sensitive to interior errors

Instance-level metrics  (connected-component analysis)
-------------------------------------------------------
  GT weed count  vs  Pred weed count           per image
  Detection rate   — % GT instances hit (IoU>0.1 with any pred instance)
  False alarm rate — % pred instances with no GT hit
  Centroid distance error (matched pairs, in pixels)
  Size ratio       — pred_area / gt_area  (1.0 = perfect, >1 = over-seg)
  GT crop count  vs  Pred crop count

Why instance metrics?
  Pixel IoU penalises every boundary pixel equally.  At 8.3M params on
  512x512 inputs, boundary sharpness is limited by decoder resolution.
  Counting weed plants correctly is the operationally relevant question.
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

import os, sys, csv, argparse, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from dual_encoder.train_shared_dual_encoder import SharedDualHeadNetV2

# ── import model (same file as training) ──────────────────────────────────────
# Run as:  python -m dual_encoder.infer_shared_dual_encoder  (from scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from dual_encoder.train_shared_dual_encoder import (
    SharedDualHeadNetV2,
    DualTaskDataset,
)
from torch.utils.data import DataLoader

CFG = dict(
    # paths
    data_root   = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET",
    runs_dir    = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs",
    # model
    rgb_variant = "small",
    nir_base_ch = 20,
    embed_dim   = 96,
    skip_ch     = 32,          # FIX-1: stride-4 skip channels
    aspp_rates  = [1, 3, 6, 9],# FIX-3: ASPP dilation rates
    aspp_out_ch = 128,         # FIX-3: ASPP output channels
    # training
    size        = (512, 512),
    batch_size  = 4,
    num_workers = 4,
    warmup_epochs = 20,
    joint_epochs  = 80,
    lr_warmup   = 1e-4,
    lr_joint    = 5e-5,
    t0_joint    = 40,          # FIX-5: cosine restart period
    lambda_cls  = 1.0,
    lambda_aux  = 0.4,         # FIX-4: aux head loss weight
    weed_weight = 2.0,         # FIX-5: reduced from 3.0
    # misc
    seed        = 42,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT  = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET"
# CKPT_PATH  = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/best_shared_dual_head.pth"
CKPT_PATH  = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/shared_dual_v2_20260410_161924/checkpoints/best_model.pth"
OUT_DIR    = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/shared_dual_head_entire_sugarbeetsAug"
# SPLITS     = ["train", "val"]        # which splits to evaluate
SPLITS     = ["train","test","val"]        # which splits to evaluate
SIZE       = (512, 512)
VEG_THRESH = 0.38
# ── Small weed rescue ──────────────────────────────────────────────────────────
# After the hard veg gate, pixels that were suppressed (veg_prob <= VEG_THRESH)
# can be rescued as weed if ALL of:
#   • veg_prob   >= RESCUE_VEG_THRESH   (some vegetation signal, just sub-threshold)
#   • weed_prob  >= RESCUE_WEED_THRESH  (cls head strongly says weed)
#   • the resulting rescued blob has area >= RESCUE_MIN_PX
#   • the rescued blob has NO other weed pred within RESCUE_ISOLATION_PX
#     (isolation check — only rescue if there's nothing nearby already covering it)
RESCUE_VEG_THRESH      = 0.28   # softer veg gate for rescue candidates
RESCUE_WEED_THRESH     = 0.45   # weed_prob must be confident
RESCUE_MIN_PX          = 120     # minimum rescued blob size (px²)
RESCUE_ISOLATION_PX    = 50     # don't rescue if existing weed pred centroid is within this
BATCH_SIZE = 32         # tune to fill VRAM — 8 fits on 8 GB
NUM_WORKERS= 10
N_VIS      = 40         # 0 = skip all vis (fastest); else worst/best/random
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
 
MATCH_IOU_THRESH  = 0.08
# How close two GT weed heads must be (centroid-to-centroid) to be treated
# as ONE cluster. Tune this3 to your typical inter-plant spacing in pixels.
GT_GROUP_RADIUS_PX   = 60
# How close a predicted centroid must be to a GT GROUP's centroid to count
# as a detection hit. Kept smaller than GT_GROUP_RADIUS_PX intentionally —
# matching is strict; grouping is lenient.
PRED_MATCH_RADIUS_PX = 40
MIN_INSTANCE_PX   = 80
# Pred blobs larger than this (px²) are considered "large weed heads" —
# centroid distance is only meaningful for these. Smaller blobs just need
# to overlap the GT weed pixel mask somewhere (any-pixel match).
LARGE_WEED_PX     = 200
NUM_CLASSES       = 3
CLASS_NAMES       = ["BG", "Crop", "Weed"]
CLASS_COLORS_BGR  = np.array([[0,0,0],[0,180,0],[220,30,30]], dtype=np.uint8)
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
 
class InferDataset(Dataset):
    def __init__(self, root, split, size=SIZE):
        self.root = root
        self.H, self.W = size
        split_file = os.path.join(root, "splits", f"{split}.txt")
        with open(split_file) as f:
            self.ids = [l.strip() for l in f if l.strip()]
 
    def __len__(self):
        return len(self.ids)
 
    def __getitem__(self, idx):
        img_id  = self.ids[idx]
        rgb_bgr = cv2.imread(os.path.join(self.root, "rgb",   f"rgb_{img_id}.png"))
        nir_raw = cv2.imread(os.path.join(self.root, "nir",   f"nir_{img_id}.png"),
                             cv2.IMREAD_UNCHANGED)
        gt_raw  = cv2.imread(os.path.join(self.root, "masks", f"mask_{img_id}.png"),
                             cv2.IMREAD_GRAYSCALE)
 
        if nir_raw.ndim == 3:
            nir_raw = cv2.cvtColor(nir_raw, cv2.COLOR_BGR2GRAY)
 
        rgb_orig = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)   # uint8, native res
 
        rgb = cv2.resize(rgb_orig, (self.W, self.H)).astype(np.float32) / 255.0
        rgb = (rgb - RGB_MEAN) / RGB_STD
        nir = cv2.resize(nir_raw,  (self.W, self.H)).astype(np.float32) / 255.0
        gt  = cv2.resize(gt_raw,   (self.W, self.H),
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
 
        return dict(
            img_id   = img_id,
            rgb_t    = torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
            nir_t    = torch.from_numpy(nir[None]).float(),
            gt       = torch.from_numpy(gt),
            rgb_orig = torch.from_numpy(rgb_orig),
            nir_orig = torch.from_numpy(nir_raw),
        )
 
 
def collate_fn(batch):
    return dict(
        img_id   = [b["img_id"]   for b in batch],
        rgb_t    = torch.stack([b["rgb_t"] for b in batch]),
        nir_t    = torch.stack([b["nir_t"] for b in batch]),
        gt       = torch.stack([b["gt"]    for b in batch]),
        rgb_orig = [b["rgb_orig"].numpy() for b in batch],
        nir_orig = [b["nir_orig"].numpy() for b in batch],
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PIXEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
 
def confusion_matrix_np(pred, gt, n=NUM_CLASSES):
    C = np.zeros((n, n), dtype=np.int64)
    valid = (gt >= 0) & (gt < n)
    np.add.at(C, (gt[valid], pred[valid]), 1)
    return C
 
 
def metrics_from_C(C):
    eps  = 1e-7
    tp   = np.diag(C).astype(float)
    fp   = C.sum(0) - tp
    fn   = C.sum(1) - tp
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    return dict(
        precision=prec, recall=rec, f1=f1, dice=f1,
        global_acc=tp.sum() / (C.sum() + eps),
        macro_f1=f1.mean(), macro_prec=prec.mean(), macro_rec=rec.mean(),
    )
 
 
def boundary_f1(pred, gt, dilation=3, n=NUM_CLASSES):
    """Trimap boundary F1 — vectorised, one pass per class."""
    k   = np.ones((dilation, dilation), np.uint8)
    bf  = np.zeros(n, dtype=np.float32)
    for c in range(n):
        gc = (gt   == c).astype(np.uint8)
        pc = (pred == c).astype(np.uint8)
        gb = cv2.dilate(gc, k) - cv2.erode(gc, k)
        pb = cv2.dilate(pc, k) - cv2.erode(pc, k)
        gd = cv2.dilate(gc, k).astype(bool)
        pd = cv2.dilate(pc, k).astype(bool)
        pr = (pb.astype(bool) & gd).sum() / (pb.sum() + 1e-7)
        re = (gb.astype(bool) & pd).sum() / (gb.sum() + 1e-7)
        bf[c] = 2 * pr * re / (pr + re + 1e-7)
    return bf
 
 
# ══════════════════════════════════════════════════════════════════════════════
# INSTANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════
 
def get_instances(mask, cls_val):
    binary = (mask == cls_val).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_INSTANCE_PX:
            continue
        out.append(dict(
            area     = int(stats[i, cv2.CC_STAT_AREA]),
            centroid = (float(centroids[i, 0]), float(centroids[i, 1])),
            mask     = (labels == i),
        ))
    return out
 
 
def _iou_inst(a, b):
    inter = (a["mask"] & b["mask"]).sum()
    union = (a["mask"] | b["mask"]).sum()
    return inter / (union + 1e-7)
 
 
def _cdist(a, b):
    return float(np.hypot(a["centroid"][0] - b["centroid"][0],
                          a["centroid"][1] - b["centroid"][1]))
 
 
# def instance_stats(gt_mask, pred_mask, cls_val):
#     gts, pds = get_instances(gt_mask, cls_val), get_instances(pred_mask, cls_val)
#     used = set()
#     matched, dists, ratios = [], [], []
#     for g in gts:
#         best_score, best_j = -1, None
#         for j, p in enumerate(pds):
#             if j in used:
#                 continue
#             s = _iou_inst(g, p)
#             if s < MATCH_IOU_THRESH and _cdist(g, p) < CENTROID_MATCH_PX:
#                 s = max(s, 0.001)       # centroid fallback
#             if s > best_score:
#                 best_score, best_j = s, j
#         if best_j is not None and best_score > 0:
#             matched.append((g, pds[best_j]))
#             dists.append(_cdist(g, pds[best_j]))
#             ratios.append(pds[best_j]["area"] / (g["area"] + 1e-7))
#             used.add(best_j)
#     nh, ng, np_ = len(matched), len(gts), len(pds)
#     return dict(
#         gt_count   = ng,
#         pred_count = np_,
#         hit        = nh,
#         miss       = ng - nh,
#         false_pos  = np_ - nh,
#         det_rate   = nh / (ng  + 1e-7),
#         fa_rate    = (np_ - nh) / (np_ + 1e-7),
#         mean_centroid_dist = float(np.mean(dists))  if dists  else float("nan"),
#         mean_size_ratio    = float(np.mean(ratios)) if ratios else float("nan"),
#     )
 
def rescue_small_weeds(pred_mask, veg_prob, weed_prob):
    """
    Soft rescue pass — runs AFTER the hard veg-gate prediction.

    Pixels suppressed by the veg gate (pred_mask == 0 there) can be rescued
    as weed class if:
      1. veg_prob  >= RESCUE_VEG_THRESH   (some vegetation signal)
      2. weed_prob >= RESCUE_WEED_THRESH  (cls head confidently says weed)
      3. Resulting connected blob has area >= RESCUE_MIN_PX
      4. No existing weed pred centroid is within RESCUE_ISOLATION_PX
         (the blob is genuinely isolated — nothing already covers it)

    Only modifies pixels currently predicted as BG (0). Never overwrites crop (1)
    or weed (2) predictions.

    Returns updated pred_mask (copy).
    """
    out = pred_mask.copy()

    # Candidate pixels: currently BG, soft veg signal, strong weed signal
    candidate = (
        (pred_mask == 0) &
        (veg_prob  >= RESCUE_VEG_THRESH) &
        (weed_prob >= RESCUE_WEED_THRESH)
    ).astype(np.uint8)

    if candidate.sum() == 0:
        return out

    # Get existing weed pred centroids for isolation check
    existing_weed_centroids = [
        inst["centroid"] for inst in get_instances(pred_mask, 2)
    ]

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < RESCUE_MIN_PX:
            continue
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])

        # Isolation check — skip if an existing weed pred is already nearby
        too_close = any(
            np.hypot(cx - ecx, cy - ecy) < RESCUE_ISOLATION_PX
            for ecx, ecy in existing_weed_centroids
        )
        if too_close:
            continue

        # Rescue: mark these pixels as weed
        out[labels == i] = 2

    return out


# def instance_stats(gt_mask, pred_mask, cls_val):
#     """
#     Instance-level evaluation.  Full logic summary:

#     ── GT side ──────────────────────────────────────────────────────────────
#     GT weed heads that are within GT_GROUP_RADIUS_PX of each other are merged
#     into one GROUP.  gt_count = number of groups (not raw annotated heads).
#     This handles the common case where an annotator places several dots on one
#     plant — we treat them as one detection target.

#     ── Matching: GT group → pred ────────────────────────────────────────────
#     A pred blob HITS a GT group if ANY of:
#       (a) pred centroid is within PRED_MATCH_RADIUS_PX of the GROUP centroid
#       (b) pred blob has IoU > MATCH_IOU_THRESH with any group member mask
#       (c) [small blobs only] pred blob overlaps any pixel of any group member
#           mask (pixel_overlap > 0).  For small heads the centroid can wander;
#           what matters is that the pred blob is sitting on the right weed.

#     One group can be hit by multiple pred blobs (e.g. the model split a weed).
#     Each group is counted as hit at most once.

#     ── False alarms ─────────────────────────────────────────────────────────
#     A pred blob is a FALSE ALARM only if it has ZERO pixel overlap with the
#     entire GT weed mask (gt_mask == cls_val).
#     This avoids penalising the model for correct weed detections where the
#     annotated head dot happened not to cover the pred centroid — annotation
#     heads are not perfectly accurate blob boundaries.

#     ── Centroid distance metric ─────────────────────────────────────────────
#     Only recorded for LARGE pred blobs (area >= LARGE_WEED_PX) matched to a
#     group.  For small blobs the centroid is noisy and the metric is meaningless.
#     """
#     gts = get_instances(gt_mask, cls_val)
#     pds = get_instances(pred_mask, cls_val)

#     gt_groups   = group_nearby_gt_instances(gts, radius_px=GT_GROUP_RADIUS_PX)
#     gt_weed_px  = (gt_mask == cls_val)   # full pixel-level GT weed mask

#     matched_pred_indices = set()   # preds that hit at least one group
#     hit_groups   = 0
#     large_dists  = []   # centroid dist — large blobs only
#     ratios       = []

#     for grp in gt_groups:
#         grp_cx, grp_cy = grp["centroid"]
#         grp_area       = grp["total_area"]
#         group_hit      = False

#         for j, p in enumerate(pds):
#             pred_cx, pred_cy = p["centroid"]

#             # (a) centroid proximity to group centroid
#             dist          = float(np.hypot(pred_cx - grp_cx, pred_cy - grp_cy))
#             centroid_ok   = dist < PRED_MATCH_RADIUS_PX

#             # (b) IoU with any member mask
#             iou_ok        = any(_iou_inst(g, p) > MATCH_IOU_THRESH
#                                 for g in grp["members"])

#             # (c) any pixel overlap with any member mask (small-blob fallback)
#             pixel_ok      = any(bool((g["mask"] & p["mask"]).any())
#                                 for g in grp["members"])

#             if centroid_ok or iou_ok or pixel_ok:
#                 matched_pred_indices.add(j)
#                 if not group_hit:
#                     group_hit = True
#                     hit_groups += 1
#                     if p["area"] >= LARGE_WEED_PX:
#                         large_dists.append(dist)
#                         ratios.append(p["area"] / (grp_area + 1e-7))

#     # ── False alarms: pred blobs with ZERO GT weed pixel overlap ─────────────
#     # This is intentionally pixel-level, not head-level: if the model predicts
#     # weed somewhere that really has weed pixels, it's not a false alarm even if
#     # no annotated head is nearby.
#     false_alarms = [
#         j for j, p in enumerate(pds)
#         if j not in matched_pred_indices
#         and not bool(gt_weed_px[p["mask"]].any())
#     ]
#     num_fa = len(false_alarms)

#     ng  = len(gt_groups)
#     np_ = len(pds)
#     nh  = hit_groups

#     return dict(
#         gt_count   = ng,
#         pred_count = np_,
#         hit        = nh,
#         miss       = ng - nh,
#         false_pos  = num_fa,
#         det_rate   = nh / (ng  + 1e-7),
#         fa_rate    = num_fa / (np_ + 1e-7),
#         # centroid dist only for large matched blobs
#         mean_centroid_dist = float(np.mean(large_dists)) if large_dists else float("nan"),
#         mean_size_ratio    = float(np.mean(ratios))      if ratios      else float("nan"),
#     )

# grok
def instance_stats(gt_mask, pred_mask, cls_val):
    """
    Instance-level evaluation. Full logic summary:
    ── GT side ──────────────────────────────────────────────────────────────
    GT weed heads that are within GT_GROUP_RADIUS_PX of each other are merged
    into one GROUP. gt_count = number of groups (not raw annotated heads).

    ── Matching: GT group → pred ────────────────────────────────────────────
    A pred blob HITS a GT group if ANY of:
      (a) pred centroid is within PRED_MATCH_RADIUS_PX of the GROUP centroid
      (b) pred blob has IoU > MATCH_IOU_THRESH with any group member mask
      (c) pred blob overlaps any pixel of any group member mask

    NEW RULE (your request):
    Even if none of the above three conditions are met, the GT group is STILL
    counted as HIT (not a miss) if there is AT LEAST ONE predicted weed pixel
    anywhere inside the GT_GROUP_RADIUS_PX circle around the group centroid.

    This removes the "miss just because there's no head nearby" case when the
    model did predict weed pixels in the vicinity (i.e. it "counted that weed
    bit as part of the nearby huge weed").
    """
    gts = get_instances(gt_mask, cls_val)
    pds = get_instances(pred_mask, cls_val)
    gt_groups = group_nearby_gt_instances(gts, radius_px=GT_GROUP_RADIUS_PX)
    gt_weed_px = (gt_mask == cls_val)  # full pixel-level GT weed mask

    matched_pred_indices = set()
    hit_groups = 0
    large_dists = []
    ratios = []

    for grp in gt_groups:
        grp_cx, grp_cy = grp["centroid"]
        grp_area = grp["total_area"]
        group_hit = False

        # ── Original strict matching (centroid / IoU / pixel overlap) ─────
        for j, p in enumerate(pds):
            pred_cx, pred_cy = p["centroid"]
            dist = float(np.hypot(pred_cx - grp_cx, pred_cy - grp_cy))
            centroid_ok = dist < PRED_MATCH_RADIUS_PX
            iou_ok = any(_iou_inst(g, p) > MATCH_IOU_THRESH for g in grp["members"])
            pixel_ok = any(bool((g["mask"] & p["mask"]).any()) for g in grp["members"])

            if centroid_ok or iou_ok or pixel_ok:
                matched_pred_indices.add(j)
                if not group_hit:
                    group_hit = True
                    hit_groups += 1
                    if p["area"] >= LARGE_WEED_PX:
                        large_dists.append(dist)
                        ratios.append(p["area"] / (grp_area + 1e-7))

        # ── NEW LENIENT RULE: remove miss if any pred weed pixel is inside radius ──
        if not group_hit:
            H, W = pred_mask.shape
            yy, xx = np.ogrid[:H, :W]
            dist_sq = (xx - grp_cx)**2 + (yy - grp_cy)**2
            in_circle = dist_sq <= GT_GROUP_RADIUS_PX ** 2
            pred_weed_in_circle = (pred_mask == cls_val) & in_circle
            if pred_weed_in_circle.any():
                group_hit = True
                hit_groups += 1
                # (no centroid dist / size ratio recorded for this lenient case)

    # ── False alarms stay unchanged (pred blobs with ZERO GT weed pixels) ─────
    false_alarms = [
        j for j, p in enumerate(pds)
        if j not in matched_pred_indices
        and not bool(gt_weed_px[p["mask"]].any())
    ]
    num_fa = len(false_alarms)

    ng = len(gt_groups)
    np_ = len(pds)
    nh = hit_groups

    return dict(
        gt_count = ng,
        pred_count = np_,
        hit = nh,
        miss = ng - nh,
        false_pos = num_fa,
        det_rate = nh / (ng + 1e-7),
        fa_rate = num_fa / (np_ + 1e-7),
        mean_centroid_dist = float(np.mean(large_dists)) if large_dists else float("nan"),
        mean_size_ratio = float(np.mean(ratios)) if ratios else float("nan"),
    )
# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION  (only for selected images, runs in thread pool)
# ══════════════════════════════════════════════════════════════════════════════
 
def _mask_color(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS_BGR):
        out[mask == c] = col
    return out
 
 
def _overlay(rgb_uint8, mask, alpha=0.45):
    H, W = mask.shape
    base = cv2.resize(rgb_uint8, (W, H)) if rgb_uint8.shape[:2] != (H, W) else rgb_uint8
    bgr  = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(bgr, 1 - alpha, _mask_color(mask), alpha, 0)
 
 
# def _draw_instance_annotations(ax, gt, pred):
#     """
#     Draws radius circles and hit/miss/FA labels on an axis.
#     Mirrors instance_stats logic exactly so what you see matches what's counted.

#     GT groups:
#       • Dashed cyan circle = GT_GROUP_RADIUS_PX  (grouping zone)
#       • Solid cyan circle  = PRED_MATCH_RADIUS_PX (centroid match zone)
#       • HIT (green) / MISS (red) label above circle

#     Pred blobs:
#       • Yellow  +  = matched to a GT group (correct detection)
#       • Orange  ×  + "FA" = pred blob has zero GT weed pixels under it
#       • White   ·  = unmatched to a group head BUT overlaps GT weed pixels
#                      (real weed, just no head nearby — not penalised)
#     """
#     import matplotlib.patches as _mp

#     gt_insts   = get_instances(gt,   2)
#     pred_insts = get_instances(pred, 2)
#     gt_groups  = group_nearby_gt_instances(gt_insts, radius_px=GT_GROUP_RADIUS_PX)
#     gt_weed_px = (gt == 2)

#     # Determine matched pred indices (same logic as instance_stats)
#     matched_pred_indices = set()
#     group_hit_flags = []
#     for grp in gt_groups:
#         grp_cx, grp_cy = grp["centroid"]
#         hit = False
#         for j, p in enumerate(pred_insts):
#             pcx, pcy  = p["centroid"]
#             dist      = float(np.hypot(pcx - grp_cx, pcy - grp_cy))
#             centroid_ok = dist < PRED_MATCH_RADIUS_PX
#             iou_ok      = any(_iou_inst(g, p) > MATCH_IOU_THRESH for g in grp["members"])
#             pixel_ok    = any(bool((g["mask"] & p["mask"]).any()) for g in grp["members"])
#             if centroid_ok or iou_ok or pixel_ok:
#                 matched_pred_indices.add(j)
#                 hit = True
#         group_hit_flags.append(hit)

#     # Draw GT groups
#     for grp, hit in zip(gt_groups, group_hit_flags):
#         cx, cy = grp["centroid"]
#         ax.add_patch(_mp.Circle(
#             (cx, cy), GT_GROUP_RADIUS_PX,
#             fill=False, edgecolor="cyan", linewidth=1.2, linestyle="--", alpha=0.7))
#         ax.add_patch(_mp.Circle(
#             (cx, cy), PRED_MATCH_RADIUS_PX,
#             fill=False, edgecolor="cyan", linewidth=1.5, linestyle="-", alpha=0.9))
#         ax.plot(cx, cy, "o", ms=5, mfc="cyan", mec="cyan", alpha=0.9)
#         color, label = ("lime", "HIT") if hit else ("red", "MISS")
#         ax.text(cx, cy - GT_GROUP_RADIUS_PX - 4, label,
#                 color=color, fontsize=6, fontweight="bold", ha="center", va="bottom",
#                 bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0))

#     # Draw pred blobs
#     for j, p in enumerate(pred_insts):
#         pcx, pcy = p["centroid"]
#         if j in matched_pred_indices:
#             # Correct detection
#             ax.plot(pcx, pcy, "+", ms=9, mec="yellow", mew=2)
#         elif gt_weed_px[p["mask"]].any():
#             # Real weed under pred, but no nearby group head — not a FA
#             ax.plot(pcx, pcy, ".", ms=6, mec="white", mfc="white", alpha=0.7)
#         else:
#             # True false alarm — pred weed with zero GT weed pixels beneath it
#             ax.plot(pcx, pcy, "x", ms=9, mec="orange", mew=2)
#             ax.text(pcx, pcy + 8, "FA", color="orange", fontsize=6,
#                     fontweight="bold", ha="center", va="top",
#                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0))

# grok
# 2. Replace the entire `_draw_instance_annotations` function with this updated version
# (keeps the visualisation 100% consistent with the new metric logic)
def _draw_instance_annotations(ax, gt, pred):
    """
    Draws radius circles and hit/miss/FA labels.
    Now uses the exact same lenient rule as instance_stats:
    A GT group is marked HIT (green) even without a matching pred blob
    if there is ANY predicted weed pixel inside the GT_GROUP_RADIUS_PX circle.
    """
    gt_insts = get_instances(gt, 2)
    pred_insts = get_instances(pred, 2)
    gt_groups = group_nearby_gt_instances(gt_insts, radius_px=GT_GROUP_RADIUS_PX)
    gt_weed_px = (gt == 2)

    matched_pred_indices = set()
    group_hit_flags = []

    for grp in gt_groups:
        grp_cx, grp_cy = grp["centroid"]
        hit = False

        # ── Original strict matching ─────────────────────────────────────
        for j, p in enumerate(pred_insts):
            pcx, pcy = p["centroid"]
            dist = float(np.hypot(pcx - grp_cx, pcy - grp_cy))
            centroid_ok = dist < PRED_MATCH_RADIUS_PX
            iou_ok = any(_iou_inst(g, p) > MATCH_IOU_THRESH for g in grp["members"])
            pixel_ok = any(bool((g["mask"] & p["mask"]).any()) for g in grp["members"])

            if centroid_ok or iou_ok or pixel_ok:
                matched_pred_indices.add(j)
                hit = True

        # ── NEW LENIENT RULE (same as instance_stats) ────────────────────
        if not hit:
            H, W = gt.shape
            yy, xx = np.ogrid[:H, :W]
            dist_sq = (xx - grp_cx)**2 + (yy - grp_cy)**2
            in_circle = dist_sq <= GT_GROUP_RADIUS_PX ** 2
            pred_weed_in_circle = (pred == 2) & in_circle
            if pred_weed_in_circle.any():
                hit = True

        group_hit_flags.append(hit)

    # Draw GT groups (unchanged visual style)
    for grp, hit in zip(gt_groups, group_hit_flags):
        cx, cy = grp["centroid"]
        ax.add_patch(mpatches.Circle(
            (cx, cy), GT_GROUP_RADIUS_PX,
            fill=False, edgecolor="cyan", linewidth=1.2, linestyle="--", alpha=0.7))
        ax.add_patch(mpatches.Circle(
            (cx, cy), PRED_MATCH_RADIUS_PX,
            fill=False, edgecolor="cyan", linewidth=1.5, linestyle="-", alpha=0.9))
        ax.plot(cx, cy, "o", ms=5, mfc="cyan", mec="cyan", alpha=0.9)

        color, label = ("lime", "HIT") if hit else ("red", "MISS")
        ax.text(cx, cy - GT_GROUP_RADIUS_PX - 4, label,
                color=color, fontsize=6, fontweight="bold", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0))

    # Draw pred blobs (unchanged)
    for j, p in enumerate(pred_insts):
        pcx, pcy = p["centroid"]
        if j in matched_pred_indices:
            ax.plot(pcx, pcy, "+", ms=9, mec="yellow", mew=2)
        elif gt_weed_px[p["mask"]].any():
            ax.plot(pcx, pcy, ".", ms=6, mec="white", mfc="white", alpha=0.7)
        else:
            ax.plot(pcx, pcy, "x", ms=9, mec="orange", mew=2)
            ax.text(pcx, pcy + 8, "FA", color="orange", fontsize=6,
                    fontweight="bold", ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0))

def save_vis_fast(path, img_id, rgb_orig, nir_orig, gt, pred,
                  veg_prob, weed_prob, row, vis_tag=""):
    """8-panel figure with radius annotations; runs in a background thread."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
 
    cmap3   = mcolors.ListedColormap(
        [c[::-1].astype(float) / 255 for c in CLASS_COLORS_BGR])
    H, W    = gt.shape
    nir_dp  = cv2.resize(nir_orig, (W, H)).astype(np.float32) / 255.0
    rgb_dp  = cv2.resize(rgb_orig, (W, H)).astype(np.float32) / 255.0

    fig, axes = plt.subplots(2, 4, figsize=(26, 13))

    # Tag in title: WORST / BEST / RAND + rank, plus all key metrics
    tag_str = f"[{vis_tag}]  " if vis_tag else ""
    fig.suptitle(
        f"{tag_str}{img_id}\n"
        f"Weed F1={row['weed_f1']:.3f}  Dice={row['weed_dice']:.3f}  "
        f"BF1={row['weed_boundary_f1']:.3f}  "
        f"Prec={row['weed_precision']:.3f}  Rec={row['weed_recall']:.3f}  "
        f"DetRate={row['weed_det_rate']:.2f}  FA={row['weed_fa_rate']:.2f}  "
        f"GT#={row['weed_gt_count']}  Pred#={row['weed_pred_count']}  "
        f"Hit={row['weed_hit']}  Miss={row['weed_miss']}  FP={row['weed_false_pos']}",
        fontsize=9, y=0.99)
 
    axes[0, 0].imshow(rgb_dp);                            axes[0, 0].set_title("RGB")
    axes[0, 1].imshow(nir_dp, cmap="gray");               axes[0, 1].set_title("NIR")
    axes[0, 2].imshow(gt,   cmap=cmap3, vmin=0, vmax=2);  axes[0, 2].set_title("GT mask")
    axes[0, 3].imshow(pred, cmap=cmap3, vmin=0, vmax=2);  axes[0, 3].set_title("Pred mask")
 
    im4 = axes[1, 0].imshow(veg_prob,  cmap="RdYlGn", vmin=0, vmax=1)
    axes[1, 0].set_title("Veg prob");   plt.colorbar(im4, ax=axes[1, 0])
    im5 = axes[1, 1].imshow(weed_prob, cmap="hot",    vmin=0, vmax=1)
    axes[1, 1].set_title("Weed prob");  plt.colorbar(im5, ax=axes[1, 1])
 
    axes[1, 2].imshow(cv2.cvtColor(_overlay(rgb_orig, gt),   cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"GT overlay  [group r={GT_GROUP_RADIUS_PX}px / match r={PRED_MATCH_RADIUS_PX}px]")
    axes[1, 3].imshow(cv2.cvtColor(_overlay(rgb_orig, pred), cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title("Pred overlay")

    # Draw radius circles + hit/miss/FA on the 4 mask/overlay panels
    for ax in [axes[0, 2], axes[0, 3], axes[1, 2], axes[1, 3]]:
        _draw_instance_annotations(ax, gt, pred)

    # Legend
    patches = [
        mpatches.Patch(color="black",                   label="BG"),
        mpatches.Patch(color=(0, 180/255, 0),           label="Crop"),
        mpatches.Patch(color=(220/255, 30/255, 30/255), label="Weed"),
        mlines.Line2D([], [], color="cyan",   linestyle="--", label=f"GT group r={GT_GROUP_RADIUS_PX}px"),
        mlines.Line2D([], [], color="cyan",   linestyle="-",  label=f"Match r={PRED_MATCH_RADIUS_PX}px"),
        mlines.Line2D([], [], color="yellow", marker="+", linestyle="None", markersize=8, label="Pred hit"),
        mlines.Line2D([], [], color="white",  marker=".", linestyle="None", markersize=6, label="Real weed/no head"),
        mlines.Line2D([], [], color="orange", marker="x", linestyle="None", markersize=8, label="True FA (no GT weed px)"),
        mpatches.Patch(color="lime",  label="HIT"),
        mpatches.Patch(color="red",   label="MISS"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=10, fontsize=7,
               bbox_to_anchor=(0.5, 0.0))
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(path, dpi=90)
    plt.close(fig)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE PLOTS  (called once, after loop)
# ══════════════════════════════════════════════════════════════════════════════
 
def make_agg_plots(global_C, all_rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    # Confusion matrix
    CN = global_C.astype(float) / (global_C.sum(1, keepdims=True) + 1e-7)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(CN, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("GT")
    ax.set_title("Normalised confusion matrix")
    plt.colorbar(im, ax=ax)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{CN[i,j]:.2f}\n({global_C[i,j]:,})",
                    ha="center", va="center",
                    color="white" if CN[i, j] > 0.6 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
 
    # Violin distributions
    vis_keys = [
        ("weed_f1",           "Weed F1"),
        ("weed_dice",         "Weed Dice"),
        ("weed_precision",    "Weed Precision"),
        ("weed_recall",       "Weed Recall"),
        ("weed_boundary_f1",  "Weed Boundary F1"),
        ("crop_f1",           "Crop F1"),
        ("weed_det_rate",     "Weed Det. Rate"),
        ("weed_fa_rate",      "Weed FA Rate"),
        ("weed_centroid_dist","Centroid dist (px)"),
        ("weed_size_ratio",   "Size ratio"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle("Per-image metric distributions", fontsize=12)
    for ax, (k, label) in zip(axes.ravel(), vis_keys):
        data = [r[k] for r in all_rows if str(r.get(k, "nan")) != "nan"]
        if not data:
            continue
        ax.violinplot(data, showmedians=True)
        ax.set_title(label, fontsize=9); ax.set_xticks([])
        med = float(np.median(data))
        ax.axhline(med, color="red", lw=1, ls="--", alpha=0.7)
        ax.text(1.02, med, f"{med:.3f}",
                transform=ax.get_yaxis_transform(), fontsize=7,
                color="red", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metric_distributions.png"), dpi=150)
    plt.close()
 
    # GT vs pred count scatter
    gtc  = [r["weed_gt_count"]   for r in all_rows]
    pdc  = [r["weed_pred_count"] for r in all_rows]
    mx   = max(max(gtc, default=1), max(pdc, default=1)) + 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gtc, pdc, alpha=0.35, s=12)
    ax.plot([0, mx], [0, mx], "r--", lw=1, label="perfect count")
    ax.set_xlabel("GT weed count"); ax.set_ylabel("Pred weed count")
    ax.set_title("Weed instance count: GT vs Pred")
    ax.legend(); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "weed_count_scatter.png"), dpi=150)
    plt.close()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
 
def write_summary(global_C, gbf1, all_rows, out_dir):
    gm  = metrics_from_C(global_C)
    n   = len(all_rows)
 
    def cmean(k):
        v = [r[k] for r in all_rows if str(r.get(k, "nan")) != "nan"]
        return float(np.mean(v)) if v else float("nan")
 
    def cstd(k):
        v = [r[k] for r in all_rows if str(r.get(k, "nan")) != "nan"]
        return float(np.std(v)) if v else float("nan")
 
    L = [
        "=" * 68,
        f"SHARED DUAL-HEAD  —  FULL DATASET EVALUATION   n={n} images",
        "=" * 68, "",
        "PIXEL-LEVEL  (aggregated confusion matrix)",
        f"  Global accuracy : {gm['global_acc']:.4f}", "",
        f"  {'Class':<8}{'Precision':>11}{'Recall':>11}{'F1':>9}"
        f"{'Dice':>9}{'Boundary F1':>13}",
        "  " + "-" * 61,
    ]
    for i, nm in enumerate(CLASS_NAMES):
        L.append(
            f"  {nm:<8}{gm['precision'][i]:>11.4f}{gm['recall'][i]:>11.4f}"
            f"{gm['f1'][i]:>9.4f}{gm['dice'][i]:>9.4f}{gbf1[i]:>13.4f}")
    L += [
        "  " + "-" * 61,
        f"  {'Macro':<8}{gm['macro_prec']:>11.4f}{gm['macro_rec']:>11.4f}"
        f"{gm['macro_f1']:>9.4f}", "",
        "INSTANCE-LEVEL  (mean ± std per image)",
        f"  {'Metric':<30}{'Mean':>9}{'Std':>9}",
        "  " + "-" * 48,
    ]
    inst_keys = [
        ("weed_gt_count",      "Weed GT count"),
        ("weed_pred_count",    "Weed pred count"),
        ("weed_det_rate",      "Weed detection rate"),
        ("weed_fa_rate",       "Weed false alarm rate"),
        ("weed_centroid_dist", "Weed centroid dist (px)"),
        ("weed_size_ratio",    "Weed size ratio"),
        ("crop_gt_count",      "Crop GT count"),
        ("crop_pred_count",    "Crop pred count"),
        ("crop_det_rate",      "Crop detection rate"),
        ("crop_fa_rate",       "Crop false alarm rate"),
    ]
    for k, lbl in inst_keys:
        L.append(f"  {lbl:<30}{cmean(k):>9.3f}{cstd(k):>9.3f}")
    L += [
        "", "PER-IMAGE PIXEL  (mean ± std)",
        f"  {'Metric':<30}{'Mean':>9}{'Std':>9}",
        "  " + "-" * 48,
    ]
    px_keys = [
        ("weed_f1",          "Weed F1"),
        ("weed_dice",        "Weed Dice"),
        ("weed_precision",   "Weed Precision"),
        ("weed_recall",      "Weed Recall"),
        ("weed_boundary_f1","Weed Boundary F1"),
        ("crop_f1",          "Crop F1"),
        ("crop_dice",        "Crop Dice"),
        ("crop_precision",   "Crop Precision"),
        ("crop_recall",      "Crop Recall"),
        ("global_acc",       "Global Accuracy"),
    ]
    for k, lbl in px_keys:
        L.append(f"  {lbl:<30}{cmean(k):>9.4f}{cstd(k):>9.4f}")
    L += ["", "=" * 68]
 
    txt = "\n".join(L)
    print("\n" + txt)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
 
    # Compact one-row CSV for cross-model comparison table
    row = {"n_images": n, "global_acc": f"{gm['global_acc']:.4f}"}
    for i, nm in enumerate(CLASS_NAMES):
        nl = nm.lower()
        row.update({
            f"{nl}_prec": f"{gm['precision'][i]:.4f}",
            f"{nl}_rec":  f"{gm['recall'][i]:.4f}",
            f"{nl}_f1":   f"{gm['f1'][i]:.4f}",
            f"{nl}_dice": f"{gm['dice'][i]:.4f}",
            f"{nl}_bf1":  f"{gbf1[i]:.4f}",
        })
    for k, _ in inst_keys:
        row[k + "_mean"] = f"{cmean(k):.3f}"
        row[k + "_std"]  = f"{cstd(k):.3f}"
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader(); w.writerow(row)
    print(f"Summary → {out_dir}/summary.txt  |  summary.csv")
 
def write_stratified_report(all_rows, out_dir):
    """
    Slice metrics by (a) weed-head COUNT bucket and (b) dominant weed SIZE bucket.

    Count buckets  : 0 | 1 | 2-3 | 4-6 | 7-10 | 11+
    Size buckets   : tiny (<200 px²) | small (200-800) | medium (800-3000) | large (3000+)
      dominant size = the size bucket that contains the most GT weed instances in the image.
      Images with 0 GT weeds are excluded from the size breakdown.

    For each bucket prints:
      n_images, mean DetRate, mean FA, mean WeedF1 (pixel),
      mean DetF1 (instance), mean leniency/image.

    Outputs: stratified_report.txt + stratified_report.csv
    """
    import math

    COUNT_BINS  = [(0,0,"0"),(1,1,"1"),(2,3,"2-3"),(4,6,"4-6"),
                   (7,10,"7-10"),(11,999,"11+")]
    SIZE_LABELS = ["tiny (<200 px)", "small (200-800)",
                   "medium (800-3000)", "large (3000+)"]
    SIZE_BREAKS = [200, 800, 3000]

    def dominant_size_bucket(gt_mask):
        """Return the SIZE_LABELS entry that covers most GT weed instances."""
        if gt_mask is None:
            return None
        binary = (gt_mask == 2).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        areas = [int(stats[i, cv2.CC_STAT_AREA])
                 for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= MIN_INSTANCE_PX]
        if not areas:
            return None
        counts = [0, 0, 0, 0]
        for a in areas:
            idx = sum(a >= b for b in SIZE_BREAKS)
            counts[idx] += 1
        dom = max(range(4), key=lambda i: (counts[i], i))
        return SIZE_LABELS[dom]

    def safe_mean(rows, k):
        v = [r[k] for r in rows if str(r.get(k, "nan")) not in ("nan", "")]
        return float(np.mean(v)) if v else float("nan")

    def bucket_metrics(rows):
        if not rows:
            return None
        return dict(
            n        = len(rows),
            det_rate = safe_mean(rows, "weed_det_rate"),
            fa_rate  = safe_mean(rows, "weed_fa_rate"),
            weed_f1  = safe_mean(rows, "weed_f1"),
            det_f1   = safe_mean(rows, "weed_det_f1"),
            leniency = safe_mean(rows, "weed_leniency"),
        )

    def nan_fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and not math.isnan(x) else "  nan"

    def fmt_row(label, m, w=22):
        if m is None:
            return f"  {label:<{w}}  (no data)"
        return (f"  {label:<{w}}  n={m['n']:>4}  "
                f"DetRate={nan_fmt(m['det_rate'])}  "
                f"FA={nan_fmt(m['fa_rate'])}  "
                f"WeedF1={nan_fmt(m['weed_f1'])}  "
                f"DetF1={nan_fmt(m['det_f1'])}  "
                f"Leniency/img={nan_fmt(m['leniency'])}")

    # annotate each row with its dominant size bucket (reuses stored _gt mask)
    for r in all_rows:
        gt_mask = r.get("_gt", None)
        r["_size_bucket"] = (dominant_size_bucket(gt_mask)
                             if isinstance(gt_mask, np.ndarray) else None)

    # ── Count stratification ──────────────────────────────────────────────────
    col_hdr = (f"  {'Bucket':<22}  {'n':>4}  "
               f"{'DetRate':>9}  {'FA':>7}  {'WeedF1':>8}  "
               f"{'DetF1':>7}  {'Leniency/img':>13}")
    sep = "  " + "-" * 98

    count_lines = ["BY WEED HEAD COUNT (gt_count buckets)", sep, col_hdr, sep]
    count_csv   = []
    for lo, hi, label in COUNT_BINS:
        subset = [r for r in all_rows if lo <= r["weed_gt_count"] <= hi]
        m = bucket_metrics(subset)
        count_lines.append(fmt_row(label, m))
        if m:
            count_csv.append({"stratify": "count", "bucket": label, **{
                k: (f"{v:.3f}" if isinstance(v, float) else v)
                for k, v in m.items()}})

    # ── Size stratification ───────────────────────────────────────────────────
    size_lines = ["", "BY DOMINANT WEED HEAD SIZE (of GT instances in image)", sep, col_hdr, sep]
    size_csv   = []
    for label in SIZE_LABELS:
        subset = [r for r in all_rows if r.get("_size_bucket") == label]
        m = bucket_metrics(subset)
        size_lines.append(fmt_row(label, m))
        if m:
            size_csv.append({"stratify": "size", "bucket": label, **{
                k: (f"{v:.3f}" if isinstance(v, float) else v)
                for k, v in m.items()}})
    zero_weed = sum(1 for r in all_rows if r["weed_gt_count"] == 0)
    size_lines.append(f"  (images with 0 GT weeds excluded from size table: {zero_weed})")

    # ── Cross-table: count bucket × size bucket — DetRate heatmap ────────────
    cross_lines = ["", "CROSS-TABLE: mean DetRate  [count bucket × size bucket]"]
    short_size  = ["tiny", "small", "medium", "large"]
    cross_lines.append("  " + f"{'Count':>10}" +
                        "".join(f"{s:>12}" for s in short_size))
    cross_lines.append("  " + "-" * (10 + 12 * 4))
    for lo, hi, clabel in COUNT_BINS:
        cells = []
        for slabel in SIZE_LABELS:
            sub = [r for r in all_rows
                   if lo <= r["weed_gt_count"] <= hi
                   and r.get("_size_bucket") == slabel]
            cells.append(f"{np.mean([r['weed_det_rate'] for r in sub]):>12.3f}"
                         if sub else f"{'—':>12}")
        cross_lines.append("  " + f"{clabel:>10}" + "".join(cells))

    # ── Assemble ──────────────────────────────────────────────────────────────
    hdr = [
        "=" * 100,
        "STRATIFIED REPORT  —  metrics by weed head count and dominant weed size",
        "=" * 100, "",
        "  DetRate     = detection recall  (TP+leniency) / (TP+leniency+FN)",
        "  FA          = false alarm rate  FP / (effective pred count)",
        "  WeedF1      = pixel-level weed F1",
        "  DetF1       = instance-level detection F1  2·P·R/(P+R)",
        "  Leniency    = mean small-weed leniency matches per image",
        "  Size breaks : tiny<200px²  small=200–800  medium=800–3000  large≥3000",
        "",
    ]
    txt = "\n".join(hdr + count_lines + size_lines + cross_lines + ["", "=" * 100])
    print("\n" + txt)
    with open(os.path.join(out_dir, "stratified_report.txt"), "w") as f:
        f.write(txt + "\n")

    fields = ["stratify", "bucket", "n", "det_rate", "fa_rate",
              "weed_f1", "det_f1", "leniency"]
    with open(os.path.join(out_dir, "stratified_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(count_csv + size_csv)
    print(f"Stratified → {out_dir}/stratified_report.txt  |  stratified_report.csv")


def group_nearby_gt_instances(instances, radius_px):
    """
    Groups GT weed instances whose centroids are within `radius_px` of each
    other. Returns a list of group dicts, each with:
        members   : list of raw instances in the group
        centroid  : area-weighted centroid of the whole group  ← used for matching
        total_area: sum of member areas

    Using the GROUP centroid (not individual member centroids) for pred matching
    keeps the effective catch-radius predictable — it does not compound with the
    grouping radius.
    """
    n = len(instances)
    if n == 0:
        return []

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            cx_i, cy_i = instances[i]["centroid"]
            cx_j, cy_j = instances[j]["centroid"]
            if np.hypot(cx_i - cx_j, cy_i - cy_j) <= radius_px:
                union(i, j)

    # Collect raw members per group
    raw_groups = defaultdict(list)
    for i in range(n):
        raw_groups[find(i)].append(instances[i])

    # Build enriched group dicts with a single representative centroid
    groups = []
    for members in raw_groups.values():
        total_area = sum(m["area"] for m in members)
        cx = sum(m["centroid"][0] * m["area"] for m in members) / (total_area + 1e-7)
        cy = sum(m["centroid"][1] * m["area"] for m in members) / (total_area + 1e-7)
        groups.append(dict(members=members, centroid=(cx, cy), total_area=total_area))

    return groups


# merge_satellite_instances is kept as a thin wrapper used only in visualisation
# (centroid dots on the vis panels). Returns a flat list compatible with the
# existing vis code (dicts with area/centroid/mask keys).
def merge_satellite_instances(instances):
    """
    Thin wrapper for visualisation: collapses nearby instances to one centroid
    per group (area-weighted). Does NOT modify any pixel masks.
    """
    groups = group_nearby_gt_instances(instances, radius_px=GT_GROUP_RADIUS_PX)
    merged = []
    for grp in groups:
        combined_mask = grp["members"][0]["mask"].copy()
        for m in grp["members"][1:]:
            combined_mask = combined_mask | m["mask"]
        merged.append(dict(
            area=grp["total_area"],
            centroid=grp["centroid"],
            mask=combined_mask,
        ))
    return merged

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
 
def run(splits, n_vis, ckpt_path, data_root, out_dir, veg_thresh, batch_size, workers):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "vis")
    if n_vis > 0:
        os.makedirs(vis_dir, exist_ok=True)
 
    # Model
    model = SharedDualHeadNetV2(CFG).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint  epoch={ckpt.get('epoch','?')}  "
          f"score={ckpt.get('score',0):.4f}  "
          f"val_veg_iou={ckpt.get('val_metrics',{}).get('veg_iou',0):.4f}  "
          f"val_weed_f1={ckpt.get('val_metrics',{}).get('weed_f1',0):.4f}")
 
    # CSV (incremental writes — no huge in-memory list)
    CSV_FIELDS = [
        "img_id", "split", "global_acc",
        "bg_precision","bg_recall","bg_f1","bg_dice","bg_boundary_f1",
        "crop_precision","crop_recall","crop_f1","crop_dice","crop_boundary_f1",
        "weed_precision","weed_recall","weed_f1","weed_dice","weed_boundary_f1",
        "weed_gt_count","weed_pred_count","weed_hit","weed_miss","weed_false_pos",
        "weed_det_rate","weed_fa_rate","weed_centroid_dist","weed_size_ratio",
        "crop_gt_count","crop_pred_count","crop_det_rate","crop_fa_rate",
    ]
    csv_path = os.path.join(out_dir, "metrics_per_image.csv")
    csv_fh   = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    writer.writeheader()
 
    global_C = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    gbf1_acc = np.zeros(NUM_CLASSES, dtype=np.float64)
    all_rows = []          # lightweight dicts for summary + vis selection
    vis_pool = ThreadPoolExecutor(max_workers=3)
    n_total  = 0
    t0       = time.time()
 
    for split in splits:
        ds = InferDataset(data_root, split)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=workers, collate_fn=collate_fn,
            pin_memory=(DEVICE == "cuda"), prefetch_factor=2,
        )
        print(f"\n── {split.upper()}  ({len(ds)} images) ──")
 
        for batch in tqdm(loader, desc=split, unit="batch"):
            rgb_t = batch["rgb_t"].to(DEVICE, non_blocking=True)
            nir_t = batch["nir_t"].to(DEVICE, non_blocking=True)
 
            with torch.no_grad():
                veg_logits, cls_logits, _ = model(rgb_t, nir_t)
 
            # Decode on GPU, one CPU transfer per batch
            veg_prob_gpu = torch.sigmoid(veg_logits).squeeze(1)
            cls_argmax   = cls_logits.argmax(dim=1)
            veg_bin      = (veg_prob_gpu > veg_thresh)
            pred_gpu     = torch.where(veg_bin, cls_argmax + 1,
                                       torch.zeros_like(cls_argmax))
 
            veg_prob_np  = veg_prob_gpu.cpu().numpy()
            weed_prob_np = F.softmax(cls_logits, dim=1)[:, 1].cpu().numpy()
            pred_np      = pred_gpu.cpu().numpy().astype(np.uint8)
            gt_np        = batch["gt"].numpy().astype(np.int64)
 
            for i in range(len(batch["img_id"])):
                img_id = batch["img_id"][i]
                pred   = pred_np[i]
                gt     = gt_np[i]
                vp     = veg_prob_np[i]
                wp     = weed_prob_np[i]

                # Rescue isolated small weed regions suppressed by the veg gate.
                # vp/wp are the raw probability maps; pred is updated in-place copy.
                pred = rescue_small_weeds(pred, vp, wp)

                C  = confusion_matrix_np(pred, gt)
                global_C += C
                pm = metrics_from_C(C)
                bf = boundary_f1(pred, gt)
                gbf1_acc += bf
 
                wi = instance_stats(gt, pred, 2)
                ci = instance_stats(gt, pred, 1)
                n_total += 1
 
                row = dict(
                    img_id=img_id, split=split,
                    global_acc=float(pm["global_acc"]),
                    bg_precision=float(pm["precision"][0]),
                    bg_recall=float(pm["recall"][0]),
                    bg_f1=float(pm["f1"][0]),
                    bg_dice=float(pm["dice"][0]),
                    bg_boundary_f1=float(bf[0]),
                    crop_precision=float(pm["precision"][1]),
                    crop_recall=float(pm["recall"][1]),
                    crop_f1=float(pm["f1"][1]),
                    crop_dice=float(pm["dice"][1]),
                    crop_boundary_f1=float(bf[1]),
                    weed_precision=float(pm["precision"][2]),
                    weed_recall=float(pm["recall"][2]),
                    weed_f1=float(pm["f1"][2]),
                    weed_dice=float(pm["dice"][2]),
                    weed_boundary_f1=float(bf[2]),
                    weed_gt_count=wi["gt_count"],
                    weed_pred_count=wi["pred_count"],
                    weed_hit=wi["hit"],
                    weed_miss=wi["miss"],
                    weed_false_pos=wi["false_pos"],
                    weed_det_rate=float(wi["det_rate"]),
                    weed_fa_rate=float(wi["fa_rate"]),
                    weed_centroid_dist=wi["mean_centroid_dist"],
                    weed_size_ratio=wi["mean_size_ratio"],
                    crop_gt_count=ci["gt_count"],
                    crop_pred_count=ci["pred_count"],
                    crop_det_rate=float(ci["det_rate"]),
                    crop_fa_rate=float(ci["fa_rate"]),
                )
                writer.writerow(row)
                all_rows.append({
                    **row,
                    "_pred": pred, "_gt": gt,
                    "_veg_prob": vp, "_weed_prob": wp,
                    "_rgb_orig": batch["rgb_orig"][i],
                    "_nir_orig": batch["nir_orig"][i],
                })
 
    csv_fh.close()
    elapsed = time.time() - t0
    print(f"\nInference done: {n_total} images in {elapsed:.1f}s "
          f"({n_total/elapsed:.1f} img/s)")
 
    # Select images to visualise
    if n_vis > 0 and all_rows:
        by_wf1  = sorted(all_rows, key=lambda r: r["weed_f1"])
        n_worst = min(n_vis // 4, len(by_wf1))
        n_best  = min(n_vis // 4, len(by_wf1))
        worst   = by_wf1[:n_worst]
        best    = by_wf1[-n_best:]
        chosen  = {id(r) for r in worst + best}
        rest    = [r for r in all_rows if id(r) not in chosen]
        np.random.shuffle(rest)
        rand    = rest[:max(0, n_vis - n_worst - n_best)]

        # Build (row, tag, filename) triples with rank-prefixed names
        vis_triples = []
        for rank, r in enumerate(worst, 1):
            tag  = f"WORST {rank:02d}/{n_worst}"
            fname = f"worst_{rank:02d}_{r['img_id']}.png"
            vis_triples.append((r, tag, fname))
        for rank, r in enumerate(reversed(best), 1):
            tag  = f"BEST {rank:02d}/{n_best}"
            fname = f"best_{rank:02d}_{r['img_id']}.png"
            vis_triples.append((r, tag, fname))
        for rank, r in enumerate(rand, 1):
            tag  = f"RAND {rank:02d}"
            fname = f"rand_{rank:02d}_{r['img_id']}.png"
            vis_triples.append((r, tag, fname))

        print(f"Queuing {len(vis_triples)} vis "
              f"({n_worst} worst / {n_best} best / {len(rand)} random)…")

        futures = [
            vis_pool.submit(
                save_vis_fast,
                os.path.join(vis_dir, fname),
                r["img_id"], r["_rgb_orig"], r["_nir_orig"],
                r["_gt"], r["_pred"], r["_veg_prob"], r["_weed_prob"], r, tag,
            )
            for r, tag, fname in vis_triples
        ]
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc="Writing vis", unit="img"):
            pass
 
    vis_pool.shutdown(wait=False)
 
    gbf1 = gbf1_acc / max(n_total, 1)
    write_summary(global_C, gbf1, all_rows, out_dir)
    write_stratified_report(all_rows, out_dir)
    make_agg_plots(global_C, all_rows, out_dir)
    print(f"\nAll outputs → {out_dir}")
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits",     nargs="+", default=SPLITS)
    p.add_argument("--ckpt",       default=CKPT_PATH)
    p.add_argument("--data_root",  default=DATA_ROOT)
    p.add_argument("--out_dir",    default=OUT_DIR)
    p.add_argument("--veg_thresh", type=float, default=VEG_THRESH)
    p.add_argument("--n_vis",      type=int,   default=N_VIS,
                   help="images to visualise (0 = none, fastest)")
    p.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    p.add_argument("--workers",    type=int,   default=NUM_WORKERS)
    a = p.parse_args()
    run(splits=a.splits, n_vis=a.n_vis, ckpt_path=a.ckpt,
        data_root=a.data_root, out_dir=a.out_dir, veg_thresh=a.veg_thresh,
        batch_size=a.batch_size, workers=a.workers)