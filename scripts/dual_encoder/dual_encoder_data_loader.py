import os
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# Reuse your helper functions if they're in the same file
# Otherwise import them: from rice_weed_data_loader import _read_split_list, ...

class DualEncoderWeedyRiceDataset(Dataset):
    """
    Dataset that returns RGB and NIR as SEPARATE tensors for dual-encoder architecture.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_size: Tuple[int, int] = (512, 512),  # smaller for Jetson
        augment: bool = True,
    ):
        self.root = root
        self.split = split
        self.target_h, self.target_w = target_size
        self.augment = augment and split == "train"

        self.rgb_dir = os.path.join(root, "RGB")
        self.ms_dir = os.path.join(root, "Multispectral")
        self.mask_dir = os.path.join(root, "Masks")
        self.meta_dir = os.path.join(root, "Metadata")

        # Reuse your helper functions
        from rice_weed_data_loader import _load_filename_mapping, _read_split_list, _load_nir_band
        
        self.orig2std, self.std2orig = _load_filename_mapping(self.meta_dir)
        split_list = _read_split_list(self.meta_dir, split)

        if split_list is not None:
            rgb_files = split_list
        else:
            rgb_files = sorted([f for f in os.listdir(self.rgb_dir)
                               if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        self.use_standardized_names = (
            (rgb_files[0] in self.std2orig or "DateTime" in rgb_files[0])
            if rgb_files else True
        )

        self.samples = self._index_samples(rgb_files, _load_nir_band)

        # ImageNet normalization for RGB
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Optional: augmentation (import your AgriculturalAugmentation if needed)
        self.aug = None
        if self.augment:
            try:
                from weedutils.augmentations import AgriculturalAugmentation
                self.aug = AgriculturalAugmentation(
                    hue_shift=15, sat_shift=30, val_shift=20,
                    brightness_limit=0.25, contrast_limit=0.25,
                    noise_std=0.05, flip_prob=0.5,
                    hsv_prob=0.7, brightness_prob=0.6, noise_prob=0.3
                )
            except ImportError:
                print("[WARN] AgriculturalAugmentation not found, skipping augmentation.")

    def _index_samples(self, rgb_files, _load_nir_band):
        samples = []
        for rgb_name in rgb_files:
            rgb_path = os.path.join(self.rgb_dir, rgb_name)
            if not os.path.exists(rgb_path):
                continue

            base = os.path.splitext(rgb_name)[0]
            mask_candidates = [
                os.path.join(self.mask_dir, f"{base}.png"),
                os.path.join(self.mask_dir, f"{base}.PNG"),
            ]
            mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)

            # Fallback mask search
            if mask_path is None:
                if self.use_standardized_names and (rgb_name in self.std2orig):
                    orig_base = os.path.splitext(self.std2orig[rgb_name])[0]
                    for ext in (".png", ".PNG"):
                        cand = os.path.join(self.mask_dir, f"{orig_base}{ext}")
                        if os.path.exists(cand):
                            mask_path = cand
                            break
                elif (not self.use_standardized_names) and (rgb_name in self.orig2std):
                    std_base = os.path.splitext(self.orig2std[rgb_name])[0]
                    cand = os.path.join(self.mask_dir, f"{std_base}.png")
                    if os.path.exists(cand):
                        mask_path = cand

            nir_path = _load_nir_band(
                self.ms_dir, rgb_name, self.use_standardized_names,
                self.orig2std, self.std2orig
            )

            if mask_path is not None and nir_path is not None:
                samples.append({"rgb": rgb_path, "mask": mask_path, "nir": nir_path})

        return samples

    def __len__(self):
        return len(self.samples)

    def _read_rgb(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"RGB not found: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_nir(self, path: str) -> np.ndarray:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"NIR not found: {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr

    def _read_mask(self, path: str) -> np.ndarray:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Mask not found: {path}")
        return (m == 255).astype(np.uint8)

    def _scale_uint(self, img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
        return img.astype(np.float32) / 255.0

    def _resize(self, rgb, nir, mask):
        rgb = cv2.resize(rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        nir = cv2.resize(nir, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
        return rgb, nir, mask

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        rgb = self._read_rgb(item["rgb"])    # (H, W, 3)
        nir = self._read_nir(item["nir"])    # (H, W)
        mask = self._read_mask(item["mask"]) # (H, W)

        # Resize
        rgb, nir, mask = self._resize(rgb, nir, mask)

        # Scale to [0, 1]
        rgb = self._scale_uint(rgb)
        nir = self._scale_uint(nir)

        # Augmentation: apply to RGB+mask (can extend to NIR if needed)
        if self.aug is not None:
            # Stack temporarily for augmentation
            rgbnir_temp = np.concatenate([rgb, nir[..., None]], axis=-1)
            rgbnir_temp, mask = self.aug(rgbnir_temp, mask)
            rgb = rgbnir_temp[..., :3]
            nir = rgbnir_temp[..., 3]

        # Normalize RGB with ImageNet stats
        rgb = (rgb - self.rgb_mean) / self.rgb_std

        # Convert to torch
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()  # (3, H, W)
        nir_t = torch.from_numpy(nir[None, ...]).float()  # (1, H, W)
        mask_t = torch.from_numpy(mask.astype(np.int64))  # (H, W)

        return {
            "rgb": rgb_t,
            "nir": nir_t,
            "mask": mask_t,
            "path": item["rgb"]
        }


def create_dual_encoder_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (512, 512),
):
    train_ds = DualEncoderWeedyRiceDataset(data_root, split="train", target_size=target_size, augment=True)
    val_ds = DualEncoderWeedyRiceDataset(data_root, split="val", target_size=target_size, augment=False)
    test_ds = DualEncoderWeedyRiceDataset(data_root, split="test", target_size=target_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
