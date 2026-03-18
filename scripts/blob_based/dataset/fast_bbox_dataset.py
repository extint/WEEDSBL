import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import json

from blob_based.dataset.config import *

IMG_EXT = '.png'


class FastBboxDataset(Dataset):
    """
    Fast bbox dataset that loads from pre-computed JSON
    No on-the-fly NDVI computation or blob extraction
    """
    def __init__(self, bbox_json_path):
        """
        Args:
            bbox_json_path: 'analysis/train_bboxes.json' or 'analysis/val_bboxes.json'
        """
        print(f"Loading from: {bbox_json_path}")
        
        with open(bbox_json_path, 'r') as f:
            data = json.load(f)
        
        # Flatten to individual samples
        self.samples = []
        for img_data in data:
            img_id = img_data['img_id']
            for bbox in img_data['bboxes']:
                self.samples.append({
                    'img_id': img_id,
                    'bbox': bbox['bbox'],
                    'label': bbox['label']
                })
        
        print(f"Loaded {len(self.samples)} bboxes\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample['img_id']
        x1, y1, x2, y2 = sample['bbox']
        label = sample['label']

        # Load images
        rgb = cv2.imread(os.path.join(RGB_DIR, f'rgb_{img_id}{IMG_EXT}'))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        nir = cv2.imread(os.path.join(NIR_DIR, f'nir_{img_id}{IMG_EXT}'), cv2.IMREAD_GRAYSCALE)

        # Crop bbox
        rgb_crop = rgb[y1:y2+1, x1:x2+1]
        nir_crop = nir[y1:y2+1, x1:x2+1]

        # Resize
        rgb_crop = cv2.resize(rgb_crop, (BLOB_SIZE, BLOB_SIZE))
        nir_crop = cv2.resize(nir_crop, (BLOB_SIZE, BLOB_SIZE))

        # Stack RGB + NIR (4 channels)
        nir_crop = np.expand_dims(nir_crop, axis=2)
        img = np.dstack([rgb_crop, nir_crop])
        
        # To tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label