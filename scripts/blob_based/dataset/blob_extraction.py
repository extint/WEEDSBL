import cv2

def extract_blobs(veg_mask, min_area=100):
    num_labels, labels = cv2.connectedComponents(veg_mask)

    blobs = []
    for i in range(1, num_labels):
        blob = (labels == i)
        if blob.sum() < min_area:
            continue
        blobs.append(blob)

    return blobs
