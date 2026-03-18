import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from tqdm import tqdm

from scripts.blob_based.dataset.config import *
from dataset.prepare_blobs import prepare_blob_samples

IMG_EXT='.png'

class BlobDataset(Dataset):
    def __init__(self, split_file):
        with open(split_file) as f:
            self.image_ids = [l.strip() for l in f.readlines()]

        self.samples = []

        print(f"\nLoading dataset from {split_file}")
        print(f"Total images: {len(self.image_ids)}")

        for img_id in tqdm(self.image_ids, desc="Preparing blobs"):
            rgb_path = os.path.join(RGB_DIR, 'rgb_'+img_id + IMG_EXT)
            nir_path = os.path.join(NIR_DIR, 'nir_'+img_id + IMG_EXT)
            mask_path = os.path.join(MASK_DIR, 'mask_'+img_id + IMG_EXT)

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

            blob_samples = prepare_blob_samples(
                rgb=rgb,
                nir=nir,
                gt_mask=gt_mask,
                blob_size=BLOB_SIZE,
                ndvi_thresh=NDVI_THRESH
            )

            self.samples.extend(blob_samples)

        print(f"Total blobs extracted: {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(label, dtype=torch.long)
        return img, label
