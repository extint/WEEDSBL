import cv2
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

def prepare_bbox_samples(rgb, nir, gt_mask, ndvi_thresh, min_area=100):
    """
    Extract bboxes from vegetation blobs with labels from gt_mask
    Returns: list of (bbox, label, img_id) tuples
    """
    ndvi = compute_ndvi(rgb, nir)
    # Combine with voting or weighted average
    # veg_mask = ndvi_threshold(ndvi, ndvi_thresh)
    veg_mask = multi_index_threshold(rgb, nir)
    blobs = extract_blobs(veg_mask, min_area)

    bboxes = []
    for blob in blobs:
        ys, xs = np.where(blob)
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()

        # Get label from GT mask within blob region
        roi_mask = gt_mask[y1:y2+1, x1:x2+1]
        blob_crop = blob[y1:y2+1, x1:x2+1]
        
        crop_pixels = (roi_mask[blob_crop] == 1).sum()
        weed_pixels = (roi_mask[blob_crop] == 2).sum()

        if crop_pixels + weed_pixels == 0:
            continue

        label = 0 if crop_pixels > weed_pixels else 1
        bboxes.append({
            'bbox': [x1, y1, x2, y2],
            'label': label
        })

    return bboxes