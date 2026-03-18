import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from tqdm import tqdm

from blob_based.dataset.config import *
from blob_based.dataset.prepare_bboxes import *

IMG_EXT = '.png'

class BboxDataset(Dataset):
    def __init__(self, split_file):
        with open(split_file) as f:
            self.image_ids = [l.strip() for l in f.readlines()]

        self.samples = []  # List of (img_id, bbox_dict)

        print(f"\nLoading dataset from {split_file}")
        print(f"Total images: {len(self.image_ids)}")

        for img_id in tqdm(self.image_ids, desc="Extracting bboxes"):
            rgb_path = os.path.join(RGB_DIR, 'rgb_' + img_id + IMG_EXT)
            nir_path = os.path.join(NIR_DIR, 'nir_' + img_id + IMG_EXT)
            mask_path = os.path.join(MASK_DIR, 'mask_' + img_id + IMG_EXT)

            rgb = cv2.imread(rgb_path)
            if rgb is None:
                raise FileNotFoundError(f"RGB image not found: {rgb_path}")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
            if nir is None:
                raise FileNotFoundError(f"NIR image not found: {nir_path}")

            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            bboxes = prepare_bbox_samples(
                rgb=rgb,
                nir=nir,
                gt_mask=gt_mask,
                ndvi_thresh=NDVI_THRESH,
                min_area=100
            )

            # Store img_id with each bbox for later retrieval
            for bbox_dict in bboxes:
                self.samples.append({
                    'img_id': img_id,
                    'bbox': bbox_dict['bbox'],
                    'label': bbox_dict['label']
                })

        print(f"Total bboxes extracted: {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample['img_id']
        bbox = sample['bbox']
        label = sample['label']

        # Load full image
        rgb_path = os.path.join(RGB_DIR, 'rgb_' + img_id + IMG_EXT)
        nir_path = os.path.join(NIR_DIR, 'nir_' + img_id + IMG_EXT)
        
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

        # Crop bbox region
        x1, y1, x2, y2 = bbox
        rgb_crop = rgb[y1:y2+1, x1:x2+1]
        nir_crop = nir[y1:y2+1, x1:x2+1]

        # Resize to fixed size
        rgb_crop = cv2.resize(rgb_crop, (BLOB_SIZE, BLOB_SIZE))
        nir_crop = cv2.resize(nir_crop, (BLOB_SIZE, BLOB_SIZE))

        # Stack RGB + NIR as 4-channel input
        img = np.dstack([rgb_crop, nir_crop])
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label