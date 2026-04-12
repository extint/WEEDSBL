import cv2
import numpy as np

def extract_blobs(veg_mask, min_area=100):
    """
    Takes CLEAN veg_mask → returns blobs + label map
    """

    veg_mask = (veg_mask > 0).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(veg_mask)

    valid_blobs = []
    valid_labels = np.zeros_like(labels)

    new_id = 1

    for i in range(1, num_labels):
        blob = (labels == i)
        area = blob.sum()

        if area < min_area:
            continue

        # remove giant blobs
        img_area = labels.shape[0] * labels.shape[1]
        if area > 0.3 * img_area:
            continue

        valid_blobs.append(blob)
        valid_labels[blob] = new_id
        new_id += 1

    return valid_blobs, valid_labels