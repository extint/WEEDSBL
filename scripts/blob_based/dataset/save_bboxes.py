import json
import os
from tqdm import tqdm
import cv2
import numpy as np

from blob_based.dataset.config import *
from blob_based.dataset.prepare_bboxes import prepare_bbox_samples
from blob_based.dataset.ndvi import compute_ndvi, ndvi_threshold

IMG_EXT = '.png'

def save_bboxes_to_json(split_file, output_json):
    """
    Extract all bboxes and save to JSON for analysis
    """
    with open(split_file) as f:
        image_ids = [l.strip() for l in f.readlines()]

    all_data = []

    print(f"\nProcessing {len(image_ids)} images from {split_file}")
    
    for img_id in tqdm(image_ids, desc="Extracting bboxes"):
        rgb_path = os.path.join(RGB_DIR, 'rgb_' + img_id + IMG_EXT)
        nir_path = os.path.join(NIR_DIR, 'nir_' + img_id + IMG_EXT)
        mask_path = os.path.join(MASK_DIR, 'mask_' + img_id + IMG_EXT)

        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"Warning: RGB not found: {rgb_path}")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        if nir is None:
            print(f"Warning: NIR not found: {nir_path}")
            continue

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"Warning: Mask not found: {mask_path}")
            continue

        bboxes = prepare_bbox_samples(
            rgb=rgb,
            nir=nir,
            gt_mask=gt_mask,
            ndvi_thresh=NDVI_THRESH,
            min_area=MIN_BLOB_AREA
        )

        img_data = {
            'img_id': img_id,
            'image_shape': list(rgb.shape[:2]),  # [H, W]
            'num_bboxes': len(bboxes),
            'bboxes': []
        }

        for bbox_dict in bboxes:
            x1, y1, x2, y2 = bbox_dict['bbox']
            img_data['bboxes'].append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'label': int(bbox_dict['label']),
                'label_name': 'crop' if bbox_dict['label'] == 0 else 'weed',
                'width': int(x2 - x1),
                'height': int(y2 - y1),
                'area': int((x2 - x1) * (y2 - y1))
            })

        all_data.append(img_data)

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(all_data, f, indent=2)

    # Print statistics
    total_bboxes = sum(d['num_bboxes'] for d in all_data)
    total_crops = sum(sum(1 for b in d['bboxes'] if b['label'] == 0) for d in all_data)
    total_weeds = sum(sum(1 for b in d['bboxes'] if b['label'] == 1) for d in all_data)

    print(f"\n=== BBOX EXTRACTION SUMMARY ===")
    print(f"Total images: {len(all_data)}")
    print(f"Total bboxes: {total_bboxes}")
    print(f"  - Crop: {total_crops}")
    print(f"  - Weed: {total_weeds}")
    print(f"Average bboxes per image: {total_bboxes/len(all_data):.2f}")
    print(f"Saved to: {output_json}\n")


if __name__ == "__main__":
    os.makedirs("analysis", exist_ok=True)
    print("using multi index thresholding")
    # Process train and val splits
    save_bboxes_to_json(f"{SPLIT_DIR}/train.txt", "analysis/train_bboxes_multi_index_threshold.json")
    save_bboxes_to_json(f"{SPLIT_DIR}/val.txt", "analysis/val_bboxes_multi_index_threshold.json")