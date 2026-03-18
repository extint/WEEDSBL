# infer_and_visualize.py
# ------------------------------------------------------------
# Blob-wise Sugar Beet vs Weed inference + visualization
# FIXED:
# - Consistent class mapping
# - Correct GT vs Prediction comparison
# - Side-by-side visualizations:
#   (1) Bounding boxes: GT vs Prediction
#   (2) Segmentation colors: GT vs Prediction
# ------------------------------------------------------------

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from scripts.blob_based.dataset.config import *
from models.blob_cnn import BlobCNN
from dataset.ndvi import compute_ndvi, ndvi_threshold
from dataset.blob_extraction import extract_blobs

# ============================================================
# CLASS CONVENTIONS (IMPORTANT)
# ------------------------------------------------------------
# Ground Truth mask:
#   0 = background
#   1 = crop
#   2 = weed
#
# Prediction (CNN output):
#   0 = crop
#   1 = weed
#   (background handled separately as -1)
# ============================================================


# -------------------------
# NDVI + Blob classification
# -------------------------
def classify_blobs(rgb, nir, model, ndvi_thresh):
    ndvi = compute_ndvi(rgb, nir)
    veg_mask = ndvi_threshold(ndvi, ndvi_thresh)
    blobs = extract_blobs(veg_mask, MIN_BLOB_AREA)

    results = []
    for blob in blobs:
        ys, xs = np.where(blob)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        patch = rgb[y1:y2 + 1, x1:x2 + 1]
        blob_crop = blob[y1:y2 + 1, x1:x2 + 1]

        patch[blob_crop == 0] = 0
        patch = cv2.resize(patch, (BLOB_SIZE, BLOB_SIZE))

        x = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(x).argmax(1).item()  # 0=crop, 1=weed

        results.append((blob, pred))

    return results


# -------------------------
# GT blob extraction (for visualization)
# -------------------------
def extract_gt_blobs(gt_mask):
    veg = (gt_mask > 0).astype(np.uint8)
    return extract_blobs(veg, MIN_BLOB_AREA)


# -------------------------
# Convert blobs to pixel mask
# -------------------------
def blobs_to_pixel_mask(results, shape):
    pred_mask = np.full(shape[:2], -1, dtype=np.int8)  # -1 = background
    for blob, pred in results:
        pred_mask[blob] = pred
    return pred_mask


# -------------------------
# Colorization helpers
# -------------------------
def colorize_gt_mask(gt_mask):
    col = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    col[gt_mask == 1] = (0, 255, 0)    # crop
    col[gt_mask == 2] = (255, 0, 0)    # weed
    return col


def colorize_pred_mask(pred_mask):
    col = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    col[pred_mask == 0] = (0, 255, 0)  # crop
    col[pred_mask == 1] = (255, 0, 0)  # weed
    return col


# -------------------------
# Draw bounding boxes
# -------------------------
def draw_blob_boxes(image, blobs, color):
    vis = image.copy()
    for blob in blobs:
        ys, xs = np.where(blob)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    return vis


# -------------------------
# Pixel metrics (ignore background)
# -------------------------
def compute_pixel_metrics(pred_mask, gt_mask):
    valid = pred_mask != -1
    if valid.sum() == 0:
        return 0.0, 0.0

    gt_valid = gt_mask[valid]
    pred_valid = pred_mask[valid]

    # map GT: 1->0 (crop), 2->1 (weed)
    gt_valid = np.where(gt_valid == 1, 0, 1)

    pixel_acc = (gt_valid == pred_valid).mean()

    ious = []
    for cls in [0, 1]:
        inter = np.logical_and(pred_valid == cls, gt_valid == cls).sum()
        union = np.logical_or(pred_valid == cls, gt_valid == cls).sum()
        ious.append(inter / max(1, union))

    return pixel_acc, np.mean(ious)


# ============================================================
# SINGLE IMAGE DEMO (SIDE-BY-SIDE VISUALIZATION)
# ============================================================
def run_single_image_demo(model):
    rgb_path = "/home/vjtiadmin/Desktop/BTechGroup/SUGARBEETS_MIXED_DATASET/rgb/rgb_1461671153_17012367.png"
    nir_path = "/home/vjtiadmin/Desktop/BTechGroup/SUGARBEETS_MIXED_DATASET/nir/nir_1461671153_17012367.png"
    gt_path  = "/home/vjtiadmin/Desktop/BTechGroup/SUGARBEETS_MIXED_DATASET/masks/mask_1461671153_17012367.png"

    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    results = classify_blobs(rgb, nir, model, NDVI_THRESH)
    pred_mask = blobs_to_pixel_mask(results, rgb.shape)
    gt_blobs = extract_gt_blobs(gt_mask)

    # -------- Bounding boxes --------
    gt_boxes   = draw_blob_boxes(rgb, gt_blobs, (0, 255, 0))
    pred_boxes = draw_blob_boxes(rgb, [b for b, _ in results], (255, 0, 0))
    bbox_vis = np.concatenate([gt_boxes, pred_boxes], axis=1)

    cv2.imwrite(
        "sample_bbox_gt_vs_pred.png",
        cv2.cvtColor(bbox_vis, cv2.COLOR_RGB2BGR)
    )

    # -------- Segmentation --------
    gt_col   = colorize_gt_mask(gt_mask)
    pred_col = colorize_pred_mask(pred_mask)
    seg_vis = np.concatenate([gt_col, pred_col], axis=1)

    cv2.imwrite(
        "sample_seg_gt_vs_pred.png",
        cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR)
    )

    print("Saved:")
    print(" - sample_bbox_gt_vs_pred.png")
    print(" - sample_seg_gt_vs_pred.png")


# ============================================================
# TEST SET EVALUATION
# ============================================================
def evaluate_test_set(model, image_ids):
    blob_correct, blob_total = 0, 0
    pix_accs, miou_list = [], []

    for img_id in tqdm(image_ids, desc="Evaluating test set"):
        rgb = cv2.imread(os.path.join(RGB_DIR, "rgb_" + img_id + ".png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        nir = cv2.imread(os.path.join(NIR_DIR, "nir_" + img_id + ".png"), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(os.path.join(MASK_DIR, "mask_" + img_id + ".png"), cv2.IMREAD_GRAYSCALE)

        results = classify_blobs(rgb, nir, model, NDVI_THRESH)
        pred_mask = blobs_to_pixel_mask(results, rgb.shape)

        # blob accuracy
        for blob, pred in results:
            crop = (gt_mask[blob] == 1).sum()
            weed = (gt_mask[blob] == 2).sum()
            if crop + weed == 0:
                continue
            gt_label = 0 if crop > weed else 1
            blob_correct += int(pred == gt_label)
            blob_total += 1

        pa, miou = compute_pixel_metrics(pred_mask, gt_mask)
        pix_accs.append(pa)
        miou_list.append(miou)

    print("\n=== TEST RESULTS ===")
    print("Blob-wise Accuracy:", blob_correct / max(1, blob_total))
    print("Pixel Accuracy:", np.mean(pix_accs))
    print("Mean IoU:", np.mean(miou_list))

# ============================================================
# FULL EVALUATION: pixel + blob + NDVI miss analysis
# ============================================================

import numpy as np
import cv2
from tqdm import tqdm
import os

from dataset.blob_extraction import extract_blobs
from dataset.ndvi import compute_ndvi, ndvi_threshold
from scripts.blob_based.dataset.config import *


def evaluate_test_set_full(model, image_ids):
    # ------------------------------
    # Pixel-level accumulators
    # ------------------------------
    pixel_correct = 0
    pixel_total = 0

    class_pixel_correct = {0: 0, 1: 0}
    class_pixel_total   = {0: 0, 1: 0}

    iou_intersection = {0: 0, 1: 0}
    iou_union        = {0: 0, 1: 0}

    # ------------------------------
    # Blob-level accumulators
    # ------------------------------
    total_blob_correct = 0
    total_blob_count = 0

    # ------------------------------
    # NDVI miss analysis
    # ------------------------------
    total_gt_blobs = 0
    total_ndvi_missed_blobs = 0

    print("\nEvaluating test set...\n")

    for img_id in tqdm(image_ids):
        # -------- Load data --------
        rgb = cv2.imread(os.path.join(RGB_DIR, "rgb_" + img_id + ".png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        nir = cv2.imread(os.path.join(NIR_DIR, "nir_" + img_id + ".png"),
                         cv2.IMREAD_GRAYSCALE)

        gt_mask = cv2.imread(os.path.join(MASK_DIR, "mask_" + img_id + ".png"),
                              cv2.IMREAD_GRAYSCALE)

        # -------- NDVI vegetation mask --------
        ndvi = compute_ndvi(rgb, nir)
        veg_mask = ndvi_threshold(ndvi, NDVI_THRESH)

        # -------- GT vegetation blobs --------
        gt_veg_mask = (gt_mask > 0).astype(np.uint8)
        gt_blobs = extract_blobs(gt_veg_mask, MIN_BLOB_AREA)
        total_gt_blobs += len(gt_blobs)

        # -------- Predicted blobs --------
        pred_blobs = extract_blobs(veg_mask, MIN_BLOB_AREA)

        # -------- NDVI miss count --------
        for gt_blob in gt_blobs:
            overlap = False
            for pb in pred_blobs:
                if np.logical_and(gt_blob, pb).sum() > 0:
                    overlap = True
                    break
            if not overlap:
                total_ndvi_missed_blobs += 1

        # -------- Classify predicted blobs --------
        results = classify_blobs(rgb, nir, model, NDVI_THRESH)

        # -------- Blob-wise correctness (per image) --------
        img_blob_correct = 0
        img_blob_total = 0

        for blob, pred in results:
            crop_pixels = (gt_mask[blob] == 1).sum()
            weed_pixels = (gt_mask[blob] == 2).sum()

            if crop_pixels + weed_pixels == 0:
                continue

            gt_label = 0 if crop_pixels > weed_pixels else 1

            img_blob_correct += int(pred == gt_label)
            img_blob_total += 1

        total_blob_correct += img_blob_correct
        total_blob_count += img_blob_total

        print(f"{img_id}: {img_blob_correct}/{img_blob_total} blobs correct")

        # -------- Pixel-level evaluation --------
        pred_pixel_mask = np.full(gt_mask.shape, -1, dtype=np.int8)
        for blob, pred in results:
            pred_pixel_mask[blob] = pred

        valid = pred_pixel_mask != -1
        gt_valid = gt_mask[valid]
        pred_valid = pred_pixel_mask[valid]

        # Map GT: 1->0 (crop), 2->1 (weed)
        gt_valid = np.where(gt_valid == 1, 0, 1)

        pixel_correct += (gt_valid == pred_valid).sum()
        pixel_total += len(gt_valid)

        for cls in [0, 1]:
            cls_mask = gt_valid == cls
            class_pixel_correct[cls] += (pred_valid[cls_mask] == cls).sum()
            class_pixel_total[cls] += cls_mask.sum()

            inter = np.logical_and(pred_valid == cls, gt_valid == cls).sum()
            union = np.logical_or(pred_valid == cls, gt_valid == cls).sum()

            iou_intersection[cls] += inter
            iou_union[cls] += union

    # =====================================================
    # FINAL METRICS
    # =====================================================
    pixel_accuracy = pixel_correct / max(1, pixel_total)

    class_pixel_accuracy = {
        cls: class_pixel_correct[cls] / max(1, class_pixel_total[cls])
        for cls in [0, 1]
    }

    class_iou = {
        cls: iou_intersection[cls] / max(1, iou_union[cls])
        for cls in [0, 1]
    }

    mean_iou = np.mean(list(class_iou.values()))

    blob_accuracy = total_blob_correct / max(1, total_blob_count)
    ndvi_blob_recall = (total_gt_blobs - total_ndvi_missed_blobs) / max(1, total_gt_blobs)

    # =====================================================
    # PRINT RESULTS (PAPER-READY)
    # =====================================================
    print("\n================ FINAL RESULTS ================\n")

    print("PIXEL METRICS")
    print(f"Pixel Accuracy (overall): {pixel_accuracy:.4f}")
    print(f"Pixel Accuracy (crop):    {class_pixel_accuracy[0]:.4f}")
    print(f"Pixel Accuracy (weed):    {class_pixel_accuracy[1]:.4f}\n")

    print("IoU METRICS")
    print(f"IoU (crop): {class_iou[0]:.4f}")
    print(f"IoU (weed): {class_iou[1]:.4f}")
    print(f"Mean IoU:   {mean_iou:.4f}\n")

    print("BLOB METRICS")
    print(f"Blob-wise Accuracy: {blob_accuracy:.4f}")
    print(f"Total GT vegetation blobs: {total_gt_blobs}")
    print(f"NDVI missed blobs:         {total_ndvi_missed_blobs}")
    print(f"NDVI blob recall:          {ndvi_blob_recall:.4f}")

    print("\n================================================\n")


# ============================================================
# MAIN
# ============================================================
def main():
    model = BlobCNN().to(DEVICE)
    model.load_state_dict(torch.load("blob_cnn.pth", map_location=DEVICE))
    model.eval()

    # Single image qualitative check
    run_single_image_demo(model)

    # Quantitative test evaluation
    with open(os.path.join(SPLIT_DIR, "test.txt")) as f:
        image_ids = [l.strip() for l in f.readlines()]

    evaluate_test_set_full(model, image_ids)
    # evaluate_test_set(model, image_ids)



if __name__ == "__main__":
    main()
