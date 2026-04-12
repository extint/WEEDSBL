"""
debug_dual_head_infer_v2.py
============================
Drop-in replacement for debug_dual_head_infer.py with:

  1. PROPER GT↔Pred matching:
       - IoU-first match  (threshold: IOU_MATCH_THRESH, default 0.1)
       - Fallback centroid-distance match (CENTROID_TOL_PX, default 40px)
       → Fixes: small GT blobs that have a nearby pred blob but zero mask overlap
                were previously counted as FN even though pred covered them.

  2. TOLERANT small-weed detection:
       - Any GT group whose merged mask has ANY pred pixel of class-2 within
         SMALL_WEED_COVER_PX pixels is considered "covered" even if IoU < threshold.
       - Controlled by: SMALL_WEED_MAX_AREA (groups ≤ this px count get the leniency)

  3. WHITE-BACKGROUND visualization panel (new 4th row):
       - TP = green fill + green centroid cross
       - FN = red fill + red centroid cross
       - FP = orange fill + orange centroid cross
       - Unmatched tiny GT = grey outline (shown as context, not penalised)
       - Clean white bg so small blobs are visible

  4. Per-image metrics table:
       Precision, Recall, F1 at both the raw-CC and post-merge level.

Run exactly like before:
    python debug_dual_head_infer_v2.py
    python debug_dual_head_infer_v2.py --ids 000123 --merge_px 80

New flags:
    --iou_thresh        float, default 0.10
    --centroid_tol      int,   default 40   (pixels)
    --small_area        int,   default 300  (px; groups below this get leniency)
    --small_cover_px    int,   default 25   (distance tolerance for small weed leniency)
"""

import os, sys, argparse, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from dual_encoder.train_shared_dual_encoder import SharedDualHeadNet

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATA_ROOT  = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET"
CKPT_PATH  = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/best_shared_dual_head.pth"
OUT_DIR    = "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/debug_shared_dual_head_entire_sugarbeetsAug"
SPLIT      = "val"
SIZE       = (512, 512)
VEG_THRESH = 0.5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SATELLITE_AREA_RATIO = 0.15
SATELLITE_MERGE_PX   = 120
MIN_INSTANCE_PX      = 30

# ── New matching thresholds ───────────────────────────────────────────────────
IOU_MATCH_THRESH   = 0.10   # any overlap ≥ 10% counts as a match
CENTROID_TOL_PX    = 40     # fallback: centroid within 40px → match
SMALL_WEED_MAX_AREA = 300   # GT groups ≤ 300px get lenient coverage check
SMALL_WEED_COVER_PX = 25    # pred pixels within 25px of small GT centroid → covered

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_COLORS_BGR = np.array([[0,0,0],[0,180,0],[0,0,220]], dtype=np.uint8)

GROUP_PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
    "#dcbeff","#9a6324","#fffac8","#800000","#aaffc3",
    "#808000","#ffd8b1","#000075","#a9a9a9",
]

# ══════════════════════════════════════════════════════════════════════════════
# DATA / MODEL HELPERS  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

def load_image(data_root, img_id, size=SIZE):
    H, W = size
    rgb_bgr = cv2.imread(os.path.join(data_root, "rgb",   f"rgb_{img_id}.png"))
    nir_raw = cv2.imread(os.path.join(data_root, "nir",   f"nir_{img_id}.png"),
                         cv2.IMREAD_UNCHANGED)
    gt_raw  = cv2.imread(os.path.join(data_root, "masks", f"mask_{img_id}.png"),
                         cv2.IMREAD_GRAYSCALE)
    if nir_raw.ndim == 3:
        nir_raw = cv2.cvtColor(nir_raw, cv2.COLOR_BGR2GRAY)
    rgb_orig = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb_orig, (W, H)).astype(np.float32) / 255.0
    rgb  = (rgb - RGB_MEAN) / RGB_STD
    nir  = cv2.resize(nir_raw,  (W, H)).astype(np.float32) / 255.0
    gt   = cv2.resize(gt_raw,   (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    rgb_t = torch.from_numpy(rgb.transpose(2,0,1)).float().unsqueeze(0)
    nir_t = torch.from_numpy(nir[None]).float().unsqueeze(0)
    return rgb_t, nir_t, gt, rgb_orig, nir_raw


def run_model(model, rgb_t, nir_t, veg_thresh=VEG_THRESH):
    with torch.no_grad():
        veg_logits, cls_logits = model(rgb_t.to(DEVICE), nir_t.to(DEVICE))
    veg_prob   = torch.sigmoid(veg_logits).squeeze(1)
    cls_argmax = cls_logits.argmax(dim=1)
    veg_bin    = (veg_prob > veg_thresh)
    pred = torch.where(veg_bin, cls_argmax + 1,
                       torch.zeros_like(cls_argmax))
    vp = veg_prob.squeeze().cpu().numpy()
    wp = F.softmax(cls_logits, dim=1)[:, 1].squeeze().cpu().numpy()
    return pred.squeeze().cpu().numpy().astype(np.uint8), vp, wp


def get_raw_instances(mask, cls_val, min_px=MIN_INSTANCE_PX):
    binary = (mask == cls_val).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        out.append(dict(
            area     = area,
            centroid = (float(centroids[i, 0]), float(centroids[i, 1])),
            mask     = (labels == i),
            tiny     = area < min_px,
        ))
    return out


def _nearest_edge_dist(mask_a, mask_b):
    dist_to_b = ndi.distance_transform_edt(~mask_b)
    edge_a = (cv2.dilate(mask_a.astype(np.uint8), np.ones((3,3),np.uint8))
              - mask_a.astype(np.uint8))
    edge_pixels = dist_to_b[edge_a == 1]
    return float(edge_pixels.min()) if edge_pixels.size > 0 else float("inf")


def merge_satellites(instances, area_ratio=SATELLITE_AREA_RATIO,
                     merge_px=SATELLITE_MERGE_PX, max_abs=4):
    if not instances:
        return [], []
    active = [dict(inst, members=[inst]) for inst in instances if not inst["tiny"]]
    noise  = [inst for inst in instances if inst["tiny"]]
    changed = True
    while changed:
        changed = False
        active.sort(key=lambda x: x["area"], reverse=True)
        flags = [False] * len(active)
        new_active = []
        for i in range(len(active)):
            if flags[i]:
                continue
            blob = active[i]
            for j in range(i + 1, len(active)):
                if flags[j]:
                    continue
                frag = active[j]
                if frag["area"] >= area_ratio * blob["area"]:
                    continue
                dist = _nearest_edge_dist(frag["mask"], blob["mask"])
                if dist > merge_px:
                    continue
                if len(blob["members"]) - 1 >= max_abs:
                    continue
                combined_area = blob["area"] + frag["area"]
                cx = (blob["centroid"][0]*blob["area"] +
                      frag["centroid"][0]*frag["area"]) / combined_area
                cy = (blob["centroid"][1]*blob["area"] +
                      frag["centroid"][1]*frag["area"]) / combined_area
                blob = dict(
                    area     = combined_area,
                    centroid = (cx, cy),
                    mask     = blob["mask"] | frag["mask"],
                    members  = blob["members"] + frag["members"],
                )
                flags[j] = True
                changed   = True
            new_active.append(blob)
        active = new_active
    return active, noise


# ══════════════════════════════════════════════════════════════════════════════
# ★ NEW: GT↔Pred matching with IoU + centroid fallback + small-weed leniency
# ══════════════════════════════════════════════════════════════════════════════

def compute_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else 0.0


def centroid_dist(g1, g2):
    cx1, cy1 = g1["centroid"]
    cx2, cy2 = g2["centroid"]
    return np.hypot(cx2 - cx1, cy2 - cy1)


def small_weed_covered(gt_group, pred_mask_binary,
                       small_area=SMALL_WEED_MAX_AREA,
                       cover_px=SMALL_WEED_COVER_PX):
    """
    For tiny GT groups that had no IoU/centroid match:
    Check if ANY pred pixel of the correct class falls within cover_px of
    the GT group's centroid. This handles the case where the pred blob is
    very close but not overlapping (e.g. slightly shifted small weed).
    """
    if gt_group["area"] > small_area:
        return False
    cx, cy = gt_group["centroid"]
    cx, cy = int(round(cx)), int(round(cy))
    H, W   = pred_mask_binary.shape
    r      = cover_px
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    patch   = pred_mask_binary[y0:y1, x0:x1]
    return bool(patch.any())


def match_groups(gt_groups, pred_groups, pred_mask,
                 iou_thresh=IOU_MATCH_THRESH,
                 centroid_tol=CENTROID_TOL_PX,
                 small_area=SMALL_WEED_MAX_AREA,
                 cover_px=SMALL_WEED_COVER_PX):
    """
    Returns:
        tp_pairs  : list of (gt_idx, pred_idx)
        fn_idxs   : list of unmatched gt_idx
        fp_idxs   : list of unmatched pred_idx
        leniency_matches : list of gt_idx matched via small-weed leniency
    """
    pred_binary = (pred_mask == 2)
    n_gt   = len(gt_groups)
    n_pred = len(pred_groups)

    # Build IoU matrix
    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, gt_g in enumerate(gt_groups):
        for j, pr_g in enumerate(pred_groups):
            iou_mat[i, j] = compute_iou(gt_g["mask"], pr_g["mask"])

    matched_gt   = set()
    matched_pred = set()
    tp_pairs     = []

    # Greedy match by descending IoU
    flat_order = np.argsort(-iou_mat.ravel())
    for flat_idx in flat_order:
        i, j = divmod(int(flat_idx), n_pred)
        if iou_mat[i, j] < iou_thresh:
            break
        if i in matched_gt or j in matched_pred:
            continue
        tp_pairs.append((i, j))
        matched_gt.add(i)
        matched_pred.add(j)

    # Fallback: centroid-distance match for still-unmatched pairs
    for i, gt_g in enumerate(gt_groups):
        if i in matched_gt:
            continue
        best_j, best_d = None, float("inf")
        for j, pr_g in enumerate(pred_groups):
            if j in matched_pred:
                continue
            d = centroid_dist(gt_g, pr_g)
            if d < centroid_tol and d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            tp_pairs.append((i, best_j))
            matched_gt.add(i)
            matched_pred.add(best_j)

    # Small-weed leniency for still-unmatched GT
    leniency_matches = []
    fn_idxs = []
    for i, gt_g in enumerate(gt_groups):
        if i in matched_gt:
            continue
        if small_weed_covered(gt_g, pred_binary, small_area, cover_px):
            leniency_matches.append(i)
        else:
            fn_idxs.append(i)

    fp_idxs = [j for j in range(n_pred) if j not in matched_pred]

    return tp_pairs, fn_idxs, fp_idxs, leniency_matches


def compute_metrics(tp, fn, fp, leniency=0):
    """
    leniency matches are counted as TP for recall but not for FP reduction.
    Returns dict with precision, recall, f1.
    """
    eff_tp = tp + leniency
    prec   = eff_tp / (eff_tp + fp)  if (eff_tp + fp)  > 0 else 0.0
    rec    = eff_tp / (eff_tp + fn)  if (eff_tp + fn)  > 0 else 0.0
    f1     = 2*prec*rec / (prec+rec) if (prec + rec)   > 0 else 0.0
    return dict(tp=tp, fn=fn, fp=fp, leniency=leniency,
                precision=prec, recall=rec, f1=f1)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))


def _mask_color_rgb(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colors_rgb = CLASS_COLORS_BGR[:, ::-1]
    for c, col in enumerate(colors_rgb):
        out[mask == c] = col
    return out


def _overlay(rgb_uint8, mask, alpha=0.45):
    H, W = mask.shape
    base = cv2.resize(rgb_uint8, (W, H)) if rgb_uint8.shape[:2] != (H, W) else rgb_uint8
    return (base * (1-alpha) + _mask_color_rgb(mask) * alpha).clip(0,255).astype(np.uint8)


def draw_instance_panel(ax, mask, cls_val, groups, noise_instances, title, show_arrows=True):
    H, W = mask.shape
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9)
    bg = np.zeros((H, W, 4), dtype=np.float32)
    ax.imshow(bg, extent=[0, W, H, 0])
    legend_handles = []
    for gi, group in enumerate(groups):
        col  = GROUP_PALETTE[gi % len(GROUP_PALETTE)]
        col3 = hex_to_rgb01(col)
        host = group["members"][0]
        sats = group["members"][1:]
        host_rgba = np.zeros((H, W, 4), dtype=np.float32)
        host_rgba[host["mask"]] = [*col3, 0.75]
        ax.imshow(host_rgba, extent=[0, W, H, 0])
        for sat in sats:
            sat_rgba = np.zeros((H, W, 4), dtype=np.float32)
            sat_rgba[sat["mask"]] = [*col3, 0.55]
            ax.imshow(sat_rgba, extent=[0, W, H, 0])
            border = cv2.dilate(sat["mask"].astype(np.uint8),
                                np.ones((3,3),np.uint8)) - sat["mask"].astype(np.uint8)
            border_rgba = np.zeros((H, W, 4), dtype=np.float32)
            border_rgba[border == 1] = [1, 0.1, 0.1, 1.0]
            ax.imshow(border_rgba, extent=[0, W, H, 0])
            if show_arrows:
                ax.annotate("", xy=group["centroid"], xytext=sat["centroid"],
                            arrowprops=dict(arrowstyle="->", color="white",
                                           lw=1.2, linestyle="dashed",
                                           connectionstyle="arc3,rad=0.15"))
        cx, cy = group["centroid"]
        ax.plot(cx, cy, "+", ms=11, mec="yellow", mew=2.0, zorder=10)
        patch = mpatches.Patch(facecolor=col3,
                               label=f"Group {gi+1} ({len(group['members'])} blobs)")
        legend_handles.append(patch)
    for inst in noise_instances:
        noise_rgba = np.zeros((H, W, 4), dtype=np.float32)
        noise_rgba[inst["mask"]] = [0.5, 0.5, 0.5, 0.5]
        ax.imshow(noise_rgba, extent=[0, W, H, 0])
    return legend_handles


# ★ NEW: white-background TP/FN/FP panel
def draw_match_panel(ax, H, W,
                     gt_groups, pred_groups, gt_noise,
                     tp_pairs, fn_idxs, fp_idxs, leniency_idxs,
                     metrics, title):
    """
    White background panel showing detection outcome per instance:
      TP  = green fill
      FN  = red fill  (missed GT)
      FP  = orange fill (spurious pred)
      Leniency TP = light-green fill + dotted border (small weed covered nearby)
      Tiny GT noise = grey outline (not scored)
    """
    ax.set_facecolor("white")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.axis("off")

    # white canvas
    canvas = np.ones((H, W, 4), dtype=np.float32)
    canvas[:, :, 3] = 1.0
    ax.imshow(canvas, extent=[0, W, H, 0])

    def fill(mask, rgba):
        layer = np.zeros((H, W, 4), dtype=np.float32)
        layer[mask] = rgba
        ax.imshow(layer, extent=[0, W, H, 0])

    def border(mask, color_rgba, dilation=3):
        kern = np.ones((dilation, dilation), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kern)
        edge = dilated - mask.astype(np.uint8)
        layer = np.zeros((H, W, 4), dtype=np.float32)
        layer[edge == 1] = color_rgba
        ax.imshow(layer, extent=[0, W, H, 0])

    tp_pred_idxs = {j for _, j in tp_pairs}

    # TP: green
    for i, j in tp_pairs:
        m = gt_groups[i]["mask"]
        fill(m, [0.08, 0.72, 0.24, 0.55])
        border(m, [0.04, 0.55, 0.15, 1.0])
        cx, cy = gt_groups[i]["centroid"]
        ax.plot(cx, cy, "+", ms=12, mec="#0a9e30", mew=2.2, zorder=12)

    # FN: red
    for i in fn_idxs:
        m = gt_groups[i]["mask"]
        fill(m, [0.85, 0.1, 0.1, 0.55])
        border(m, [0.7, 0.0, 0.0, 1.0])
        cx, cy = gt_groups[i]["centroid"]
        ax.plot(cx, cy, "x", ms=12, mec="#cc0000", mew=2.5, zorder=12)

    # Leniency TP: light green, dashed border
    for i in leniency_idxs:
        m = gt_groups[i]["mask"]
        fill(m, [0.5, 0.95, 0.55, 0.45])
        border(m, [0.2, 0.8, 0.3, 0.8], dilation=2)
        cx, cy = gt_groups[i]["centroid"]
        ax.plot(cx, cy, "+", ms=10, mec="#33cc55", mew=1.8, mfc="none",
                linestyle="dashed", zorder=12)

    # FP: orange
    for j in fp_idxs:
        m = pred_groups[j]["mask"]
        fill(m, [1.0, 0.55, 0.1, 0.55])
        border(m, [0.9, 0.4, 0.0, 1.0])
        cx, cy = pred_groups[j]["centroid"]
        ax.plot(cx, cy, "D", ms=7, mec="#e06000", mfc="none", mew=2.0, zorder=12)

    # Tiny GT noise: grey outline (context only)
    for inst in gt_noise:
        border(inst["mask"], [0.6, 0.6, 0.6, 0.7], dilation=2)

    # Metrics text box
    m = metrics
    txt = (f"TP={m['tp']}  FN={m['fn']}  FP={m['fp']}  Leniency={m['leniency']}\n"
           f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  F1={m['f1']:.3f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="left", fontfamily="monospace",
            color="#111111",
            bbox=dict(facecolor="white", edgecolor="#aaaaaa", boxstyle="round,pad=0.3",
                      alpha=0.85))

    ax.set_title(title, fontsize=9, color="#111111")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SAVE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def save_debug_vis(out_path, img_id, rgb_orig, nir_orig, gt, pred,
                   veg_prob, weed_prob,
                   area_ratio=SATELLITE_AREA_RATIO, merge_px=SATELLITE_MERGE_PX,
                   iou_thresh=IOU_MATCH_THRESH, centroid_tol=CENTROID_TOL_PX,
                   small_area=SMALL_WEED_MAX_AREA, cover_px=SMALL_WEED_COVER_PX):

    H, W = gt.shape
    nir_dp = cv2.resize(nir_orig, (W, H)).astype(np.float32) / 255.0
    rgb_dp = cv2.resize(rgb_orig, (W, H)).astype(np.float32) / 255.0

    gt_raw_inst   = get_raw_instances(gt,   2)
    pred_raw_inst = get_raw_instances(pred, 2)

    gt_groups,   gt_noise   = merge_satellites(gt_raw_inst,   area_ratio, merge_px)
    pred_groups, pred_noise = merge_satellites(pred_raw_inst, area_ratio, merge_px)

    n_gt_raw    = sum(1 for x in gt_raw_inst   if not x["tiny"])
    n_pred_raw  = sum(1 for x in pred_raw_inst if not x["tiny"])
    n_gt_merged   = len(gt_groups)
    n_pred_merged = len(pred_groups)

    # ── Matching ──────────────────────────────────────────────────────────────
    tp_pairs, fn_idxs, fp_idxs, leniency_idxs = match_groups(
        gt_groups, pred_groups, pred,
        iou_thresh=iou_thresh, centroid_tol=centroid_tol,
        small_area=small_area, cover_px=cover_px,
    )

    # Raw-CC level metrics (IoU/centroid, no leniency)
    tp_raw_pairs, fn_raw, fp_raw, len_raw = match_groups(
        [dict(g, members=[g]) for g in gt_raw_inst if not g["tiny"]],
        [dict(g, members=[g]) for g in pred_raw_inst if not g["tiny"]],
        pred,
        iou_thresh=iou_thresh, centroid_tol=centroid_tol,
        small_area=small_area, cover_px=cover_px,
    )

    merged_metrics = compute_metrics(len(tp_pairs), len(fn_idxs), len(fp_idxs),
                                     leniency=len(leniency_idxs))
    raw_metrics    = compute_metrics(len(tp_raw_pairs), len(fn_raw), len(fp_raw),
                                     leniency=len(len_raw))

    print(f"  [merged] TP={merged_metrics['tp']} FN={merged_metrics['fn']} "
          f"FP={merged_metrics['fp']} len={merged_metrics['leniency']} "
          f"P={merged_metrics['precision']:.3f} R={merged_metrics['recall']:.3f} "
          f"F1={merged_metrics['f1']:.3f}")
    print(f"  [raw CC] TP={raw_metrics['tp']} FN={raw_metrics['fn']} "
          f"FP={raw_metrics['fp']} len={raw_metrics['leniency']} "
          f"P={raw_metrics['precision']:.3f} R={raw_metrics['recall']:.3f} "
          f"F1={raw_metrics['f1']:.3f}")

    # ── Layout: 4 rows × 4 cols ───────────────────────────────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(28, 28))
    fig.patch.set_facecolor("#1a1a1a")

    fig.suptitle(
        f"{img_id}  |  "
        f"GT raw={n_gt_raw}→merged={n_gt_merged}   "
        f"Pred raw={n_pred_raw}→merged={n_pred_merged}   "
        f"area_ratio={area_ratio}  merge_px={merge_px}  "
        f"iou_thresh={iou_thresh}  centroid_tol={centroid_tol}px",
        fontsize=10, y=0.995, color="white",
    )

    # ── Row 0: raw inputs ─────────────────────────────────────────────────────
    axes[0,0].imshow(rgb_dp);                             axes[0,0].set_title("RGB", color="w")
    axes[0,1].imshow(nir_dp, cmap="gray");                axes[0,1].set_title("NIR", color="w")
    cmap3 = mcolors.ListedColormap(["black", "green", "blue"])
    axes[0,2].imshow(gt,   cmap=cmap3, vmin=0, vmax=2);  axes[0,2].set_title("GT mask", color="w")
    axes[0,3].imshow(pred, cmap=cmap3, vmin=0, vmax=2);  axes[0,3].set_title("Pred mask", color="w")
    for inst in gt_raw_inst:
        col = "grey" if inst["tiny"] else "cyan"
        axes[0,2].plot(*inst["centroid"], "o", ms=4 if inst["tiny"] else 7,
                       mfc="none", mec=col, mew=1.5)
    for inst in pred_raw_inst:
        col = "grey" if inst["tiny"] else "yellow"
        axes[0,3].plot(*inst["centroid"], "+", ms=4 if inst["tiny"] else 7,
                       mec=col, mew=1.5)

    # ── Row 1: GT merge groups ────────────────────────────────────────────────
    axes[1,0].imshow(_overlay(rgb_orig, gt));             axes[1,0].set_title("GT overlay", color="w")
    ax_gt_raw = axes[1,1]
    ax_gt_raw.set_title(f"GT raw CC  ({n_gt_raw} blobs, grey=noise)", fontsize=9, color="w")
    ax_gt_raw.imshow(np.zeros((H,W,3),dtype=np.uint8), extent=[0,W,H,0])
    ax_gt_raw.set_xlim(0,W); ax_gt_raw.set_ylim(H,0); ax_gt_raw.axis("off")
    for ri, inst in enumerate([x for x in gt_raw_inst if not x["tiny"]]):
        col3 = hex_to_rgb01(GROUP_PALETTE[ri % len(GROUP_PALETTE)])
        tmp  = np.zeros((H,W,4),dtype=np.float32)
        tmp[inst["mask"]] = [*col3, 0.8]
        ax_gt_raw.imshow(tmp, extent=[0,W,H,0])
        ax_gt_raw.plot(*inst["centroid"], "o", ms=7, mfc="none", mec="white", mew=1.5)
    for inst in gt_noise:
        tmp = np.zeros((H,W,4),dtype=np.float32)
        tmp[inst["mask"]] = [0.5,0.5,0.5,0.5]
        ax_gt_raw.imshow(tmp, extent=[0,W,H,0])

    gt_legend = draw_instance_panel(axes[1,2], gt, 2, gt_groups, gt_noise,
                                    f"GT after merge  ({n_gt_merged} groups)")
    axes[1,3].axis("off"); axes[1,3].set_title("GT merge legend", color="w")
    if gt_legend:
        axes[1,3].legend(handles=gt_legend, loc="upper left", fontsize=8,
                         framealpha=0.3, ncol=1, labelcolor="white")

    # ── Row 2: Pred merge groups ──────────────────────────────────────────────
    axes[2,0].imshow(_overlay(rgb_orig, pred));           axes[2,0].set_title("Pred overlay", color="w")
    ax_pred_raw = axes[2,1]
    ax_pred_raw.set_title(f"Pred raw CC  ({n_pred_raw} blobs, grey=noise)", fontsize=9, color="w")
    ax_pred_raw.imshow(np.zeros((H,W,3),dtype=np.uint8), extent=[0,W,H,0])
    ax_pred_raw.set_xlim(0,W); ax_pred_raw.set_ylim(H,0); ax_pred_raw.axis("off")
    for ri, inst in enumerate([x for x in pred_raw_inst if not x["tiny"]]):
        col3 = hex_to_rgb01(GROUP_PALETTE[ri % len(GROUP_PALETTE)])
        tmp  = np.zeros((H,W,4),dtype=np.float32)
        tmp[inst["mask"]] = [*col3, 0.8]
        ax_pred_raw.imshow(tmp, extent=[0,W,H,0])
        ax_pred_raw.plot(*inst["centroid"], "+", ms=8, mec="white", mew=1.5)
    for inst in pred_noise:
        tmp = np.zeros((H,W,4),dtype=np.float32)
        tmp[inst["mask"]] = [0.5,0.5,0.5,0.5]
        ax_pred_raw.imshow(tmp, extent=[0,W,H,0])

    pred_legend = draw_instance_panel(axes[2,2], pred, 2, pred_groups, pred_noise,
                                      f"Pred after merge  ({n_pred_merged} groups)")
    axes[2,3].axis("off"); axes[2,3].set_title("Pred merge legend", color="w")
    if pred_legend:
        axes[2,3].legend(handles=pred_legend, loc="upper left", fontsize=8,
                         framealpha=0.3, ncol=1, labelcolor="white")

    # ── Row 3: ★ WHITE-BG Match Panel ─────────────────────────────────────────
    # Col 0: RGB reference on white
    rgb_white_ax = axes[3,0]
    rgb_white_ax.set_facecolor("white")
    rgb_white_ax.imshow(rgb_dp)
    rgb_white_ax.set_title("RGB reference", color="#111111", fontsize=9)
    rgb_white_ax.axis("off")

    # Col 1: Merged-level TP/FN/FP on white bg
    draw_match_panel(
        axes[3,1], H, W,
        gt_groups, pred_groups, gt_noise,
        tp_pairs, fn_idxs, fp_idxs, leniency_idxs,
        merged_metrics,
        title=f"★ Merged match  (iou≥{iou_thresh} or dist<{centroid_tol}px)",
    )

    # Col 2: Raw-CC level TP/FN/FP on white bg
    gt_raw_active = [dict(g, members=[g]) for g in gt_raw_inst if not g["tiny"]]
    pred_raw_active = [dict(g, members=[g]) for g in pred_raw_inst if not g["tiny"]]
    draw_match_panel(
        axes[3,2], H, W,
        gt_raw_active, pred_raw_active,
        [g for g in gt_raw_inst if g["tiny"]],
        tp_raw_pairs, fn_raw, fp_raw, len_raw,
        raw_metrics,
        title="★ Raw-CC match",
    )

    # Col 3: Summary text
    axes[3,3].set_facecolor("white")
    axes[3,3].axis("off")
    axes[3,3].set_title("Summary", color="#111111")
    summary = [
        ["", "Merged", "Raw CC"],
        ["TP",        str(merged_metrics["tp"]),        str(raw_metrics["tp"])],
        ["FN",        str(merged_metrics["fn"]),        str(raw_metrics["fn"])],
        ["FP",        str(merged_metrics["fp"]),        str(raw_metrics["fp"])],
        ["Leniency",  str(merged_metrics["leniency"]),  str(raw_metrics["leniency"])],
        ["Precision", f"{merged_metrics['precision']:.3f}", f"{raw_metrics['precision']:.3f}"],
        ["Recall",    f"{merged_metrics['recall']:.3f}",    f"{raw_metrics['recall']:.3f}"],
        ["F1",        f"{merged_metrics['f1']:.3f}",        f"{raw_metrics['f1']:.3f}"],
    ]
    legend_elems = [
        mpatches.Patch(facecolor="#15b83e", alpha=0.7, label="TP (matched GT)"),
        mpatches.Patch(facecolor="#cc2222", alpha=0.7, label="FN (missed GT)"),
        mpatches.Patch(facecolor="#e06000", alpha=0.7, label="FP (spurious Pred)"),
        mpatches.Patch(facecolor="#77ee88", alpha=0.7, label="Leniency TP (small weed nearby)"),
        mpatches.Patch(facecolor="#aaaaaa", alpha=0.5, label="Tiny GT noise (not scored)"),
    ]
    axes[3,3].legend(handles=legend_elems, loc="lower left", fontsize=8,
                     framealpha=0.8, edgecolor="#aaaaaa")
    y = 0.96
    for row in summary:
        is_hdr = row[0] == ""
        color  = "#333333" if is_hdr else "#111111"
        weight = "bold" if is_hdr else "normal"
        txt    = f"{row[0]:<12}{row[1]:<12}{row[2]}"
        axes[3,3].text(0.04, y, txt, transform=axes[3,3].transAxes,
                       fontsize=9, va="top", color=color, fontweight=weight,
                       fontfamily="monospace")
        y -= 0.085

    # ── Common style ──────────────────────────────────────────────────────────
    for ax in axes[:3].ravel():
        ax.set_facecolor("black")
    for ax in axes[3].ravel():
        ax.set_facecolor("white")

    # Bottom legend
    legend_elems2 = [
        mpatches.Patch(facecolor="none", edgecolor="cyan",   label="GT raw centroid (circle)"),
        mpatches.Patch(facecolor="none", edgecolor="yellow", label="Pred raw centroid (+)"),
        mpatches.Patch(facecolor="none", edgecolor="red",    label="Absorbed satellite border"),
        mpatches.Patch(facecolor="grey", edgecolor="grey",   label="Noise (below min size)"),
    ]
    fig.legend(handles=legend_elems2, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.4, labelcolor="white")

    plt.tight_layout(rect=[0, 0.025, 1, 0.995])
    plt.savefig(out_path, dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return merged_metrics, raw_metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(description="Debug satellite-merge visualiser v2")
    pa.add_argument("--ids",          nargs="*",  default=None)
    pa.add_argument("--n_random",     type=int,   default=6)
    pa.add_argument("--split",        default=SPLIT)
    pa.add_argument("--data_root",    default=DATA_ROOT)
    pa.add_argument("--ckpt",         default=CKPT_PATH)
    pa.add_argument("--out_dir",      default=OUT_DIR)
    pa.add_argument("--veg_thresh",   type=float, default=VEG_THRESH)
    pa.add_argument("--area_ratio",   type=float, default=SATELLITE_AREA_RATIO)
    pa.add_argument("--merge_px",     type=float, default=SATELLITE_MERGE_PX)
    pa.add_argument("--min_px",       type=int,   default=MIN_INSTANCE_PX)
    # new flags
    pa.add_argument("--iou_thresh",   type=float, default=IOU_MATCH_THRESH,
                    help="Minimum IoU to count GT↔Pred as matched")
    pa.add_argument("--centroid_tol", type=int,   default=CENTROID_TOL_PX,
                    help="Fallback: centroid within N px → match")
    pa.add_argument("--small_area",   type=int,   default=SMALL_WEED_MAX_AREA,
                    help="GT groups ≤ this area get small-weed leniency")
    pa.add_argument("--small_cover",  type=int,   default=SMALL_WEED_COVER_PX,
                    help="Pred must be within N px of small GT centroid")
    a = pa.parse_args()

    out_dir = os.path.join(a.out_dir, "debug_merge_v2")
    os.makedirs(out_dir, exist_ok=True)

    split_file = os.path.join(a.data_root, "splits", f"{a.split}.txt")
    with open(split_file) as f:
        all_ids = [l.strip() for l in f if l.strip()]

    chosen_ids = a.ids if a.ids else random.sample(all_ids, min(a.n_random, len(all_ids)))

    print(f"Visualising {len(chosen_ids)} images: {chosen_ids}")
    print(f"  area_ratio={a.area_ratio}  merge_px={a.merge_px}  min_px={a.min_px}")
    print(f"  iou_thresh={a.iou_thresh}  centroid_tol={a.centroid_tol}px")
    print(f"  small_area≤{a.small_area}px  small_cover={a.small_cover}px")

    model = SharedDualHeadNet(rgb_variant="small", nir_base_ch=20, embed_dim=96).to(DEVICE)
    ckpt  = torch.load(a.ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded  epoch={ckpt.get('epoch','?')}")

    all_merged, all_raw = [], []
    for img_id in chosen_ids:
        print(f"\n[{img_id}]")
        rgb_t, nir_t, gt, rgb_orig, nir_orig = load_image(a.data_root, img_id)
        pred, vp, wp = run_model(model, rgb_t, nir_t, a.veg_thresh)
        out_path = os.path.join(out_dir, f"{img_id}_match_debug.png")
        mm, rm = save_debug_vis(
            out_path, img_id, rgb_orig, nir_orig, gt, pred, vp, wp,
            area_ratio=a.area_ratio, merge_px=a.merge_px,
            iou_thresh=a.iou_thresh, centroid_tol=a.centroid_tol,
            small_area=a.small_area, cover_px=a.small_cover,
        )
        all_merged.append(mm)
        all_raw.append(rm)

    # Dataset-level aggregate
    if len(all_merged) > 1:
        def agg(lst):
            tp = sum(x["tp"] + x["leniency"] for x in lst)
            fn = sum(x["fn"] for x in lst)
            fp = sum(x["fp"] for x in lst)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
            return prec, rec, f1
        mp, mr, mf = agg(all_merged)
        rp, rr, rf = agg(all_raw)
        print(f"\n{'='*60}")
        print(f"DATASET AGGREGATE ({len(chosen_ids)} images)")
        print(f"  Merged  → P={mp:.4f}  R={mr:.4f}  F1={mf:.4f}")
        print(f"  Raw CC  → P={rp:.4f}  R={rr:.4f}  F1={rf:.4f}")
        print(f"{'='*60}")

    print(f"\nDone. All debug images in: {out_dir}")


if __name__ == "__main__":
    main()