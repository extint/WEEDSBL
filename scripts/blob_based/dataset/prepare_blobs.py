import cv2
import numpy as np
from blob_based.dataset.ndvi import compute_ndvi, ndvi_threshold
from blob_based.dataset.blob_extraction import extract_blobs

def prepare_blob_samples(rgb, nir, gt_mask, blob_size, ndvi_thresh):
    """
    gt_mask is used ONLY to assign labels (training)
    """
    samples = []

    ndvi = compute_ndvi(rgb, nir)
    veg_mask = ndvi_threshold(ndvi, ndvi_thresh)

    blobs = extract_blobs(veg_mask)

    for blob in blobs:
        ys, xs = np.where(blob)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        patch = rgb[y1:y2+1, x1:x2+1]
        blob_crop = blob[y1:y2+1, x1:x2+1]

        patch[blob_crop == 0] = 0
        patch = cv2.resize(patch, (blob_size, blob_size))

        # label using GT mask (training only)
        crop_pixels = (gt_mask[y1:y2+1, x1:x2+1][blob_crop] == 1).sum()
        weed_pixels = (gt_mask[y1:y2+1, x1:x2+1][blob_crop] == 2).sum()

        if crop_pixels + weed_pixels == 0:
            continue

        label = 0 if crop_pixels > weed_pixels else 1
        samples.append((patch, label))

    return samples
