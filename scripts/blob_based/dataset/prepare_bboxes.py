import cv2
import os
import numpy as np
from blob_based.dataset.ndvi import compute_ndvi, ndvi_threshold
from blob_based.dataset.blob_extraction import extract_blobs

def multi_index_threshold(rgb, nir):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
    # Multiple indices
    exg = 2*g - r - b
    ndvi = (nir - r) / (nir + r + 1e-6)
    gndvi = (nir - g) / (nir + g + 1e-6)
    
    # Normalize to [0, 1]
    exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)
    ndvi_norm = (ndvi + 1) / 2  # NDVI is [-1, 1]
    gndvi_norm = (gndvi + 1) / 2
    
    # Weighted combination
    combined = 0.4 * exg_norm + 0.3 * ndvi_norm + 0.3 * gndvi_norm
    
    veg_mask = (combined > 0.5).astype(np.uint8)
    return veg_mask

def save_debug_visuals(rgb, veg_mask, blobs, img_id, out_dir="/home/vjti-comp/WEEDSBL/scripts/analysis"):
    os.makedirs(out_dir, exist_ok=True)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ---------- 1. Veg Mask Overlay ----------
    overlay = rgb_bgr.copy()
    overlay[veg_mask > 0] = (0, 255, 0)  # green

    blended = cv2.addWeighted(rgb_bgr, 0.7, overlay, 0.3, 0)
    cv2.imwrite(os.path.join(out_dir, f"{img_id}_veg_overlay.png"), blended)

    # ---------- 2. Blob Visualization ----------
    blob_vis = np.zeros_like(rgb_bgr)

    for blob_id in range(1, blobs.max() + 1):
        color = np.random.randint(0, 255, size=3)
        blob_vis[blobs == blob_id] = color

    cv2.imwrite(os.path.join(out_dir, f"{img_id}_blobs.png"), blob_vis)

    # ---------- 3. BBoxes ----------
    bbox_img = rgb_bgr.copy()

    for blob_id in range(1, blobs.max() + 1):
        blob = (blobs == blob_id)

        if blob.sum() < 50:
            continue

        y, x = np.where(blob)
        x1, x2 = x.min(), x.max()
        y1, y2 = y.min(), y.max()

        cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(out_dir, f"{img_id}_bboxes.png"), bbox_img)

def prepare_bbox_samples(img_id, rgb, nir, gt_mask, ndvi_thresh, min_area=100):
    """
    Extract bboxes from vegetation blobs with labels from gt_mask
    Returns: list of (bbox, label, img_id) tuples
    """
    ndvi = compute_ndvi(rgb, nir)
    # Combine with voting or weighted average
    veg_mask = ndvi_threshold(ndvi, ndvi_thresh)
    
    # Normalize NDVI to 0–255
    ndvi_norm = ((ndvi + 1) / 2 * 255).astype(np.uint8)

    # Adaptive threshold (Otsu)
    _, veg_mask = cv2.threshold(
        ndvi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # =========================
    # CLEAN MASK
    # =========================

    # Remove noise
    kernel_small = np.ones((3,3), np.uint8)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel_small)

    # Fill holes (VERY IMPORTANT)
    from scipy.ndimage import binary_fill_holes
    veg_mask = binary_fill_holes(veg_mask > 0).astype(np.uint8) * 255

    # Mild closing (connect leaves)
    kernel = np.ones((5,5), np.uint8)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, kernel)

    blobs, label_map = extract_blobs(veg_mask, min_area)
    save_debug_visuals(rgb, veg_mask, label_map, img_id)
    bboxes = []
    for blob in blobs:
        ys, xs = np.where(blob)
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        
        img_area = rgb.shape[0] * rgb.shape[1]
        if (x2 - x1) * (y2 - y1) > 0.3 * img_area:
            continue

        # Remove elongated weird shapes
        width = x2 - x1
        height = y2 - y1

        aspect_ratio = width / (height + 1e-6)

        if aspect_ratio > 5 or aspect_ratio < 0.2:
            continue

        # Get label from GT mask within blob region
        roi_mask = gt_mask[y1:y2+1, x1:x2+1]
        blob_crop = blob[y1:y2+1, x1:x2+1]
        
        crop_pixels = (roi_mask[blob_crop] == 1).sum()
        weed_pixels = (roi_mask[blob_crop] == 2).sum()

        if crop_pixels + weed_pixels == 0:
            continue

        # label = 0 if crop_pixels > weed_pixels else 1
        total = crop_pixels + weed_pixels
        if total == 0:
            continue

        crop_ratio = crop_pixels / total
        weed_ratio = weed_pixels / total

        if max(crop_ratio, weed_ratio) < 0.6:
            continue  # discard ambiguous blobs

        label = 0 if crop_ratio > weed_ratio else 1

        bboxes.append({
            'bbox': [x1, y1, x2, y2],
            'label': label
        })

    return bboxes