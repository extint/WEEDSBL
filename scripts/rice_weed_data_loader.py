import os
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import random

def _read_split_list(meta_dir: str, split: str) -> Optional[List[str]]:
    split_file = os.path.join(meta_dir, f"{split}.txt")
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            items = [ln.strip() for ln in f if ln.strip()]
        return items
    return None

def _load_filename_mapping(meta_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    map_path = os.path.join(meta_dir, "filename_mapping.csv")
    orig2std, std2orig = {}, {}
    if os.path.exists(map_path):
        with open(map_path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                o = row["original_filename"]
                s = row["standardized_filename"]
                orig2std[o] = s
                std2orig[s] = o
    return orig2std, std2orig

def _is_uint16(img: np.ndarray) -> bool:
    return img is not None and img.dtype == np.uint16

def _load_nir_band(ms_dir: str, rgb_name: str, use_standardized: bool,
                   orig2std: Dict[str, str], std2orig: Dict[str, str]) -> Optional[str]:
    base_std = os.path.splitext(rgb_name)[0]
    if use_standardized:
        std_candidate = os.path.join(ms_dir, f"{base_std}_NIR.TIF")
        if os.path.exists(std_candidate):
            return std_candidate
        if base_std + ".JPG" in std2orig:
            orig_rgb = std2orig[base_std + ".JPG"]
            base_org = os.path.splitext(orig_rgb)[0].replace("_D", "")
            org_candidate = os.path.join(ms_dir, f"{base_org}_MS_NIR.TIF")
            if os.path.exists(org_candidate):
                return org_candidate
    else:
        base_org = os.path.splitext(rgb_name)[0].replace("_D", "")
        org_candidate = os.path.join(ms_dir, f"{base_org}_MS_NIR.TIF")
        if os.path.exists(org_candidate):
            return org_candidate
        if rgb_name in orig2std:
            std_base = os.path.splitext(orig2std[rgb_name])[0]
            std_candidate = os.path.join(ms_dir, f"{std_base}_NIR.TIF")
            if os.path.exists(std_candidate):
                return std_candidate
    return None


class WeedyRiceRGBNIRDataset(Dataset):
    """4-channel RGB+NIR dataset loader without Albumentations."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_size: Tuple[int, int] = (960, 1280),
        use_rgbnir: bool = True,
        augment: bool = True
    ):
        self.root = root
        self.split = split
        self.target_h, self.target_w = target_size
        self.use_rgbnir = use_rgbnir
        self.augment = augment and split == "train"
        
        self.rgb_dir = os.path.join(root, "RGB")
        self.ms_dir = os.path.join(root, "Multispectral")
        self.mask_dir = os.path.join(root, "Masks")
        self.meta_dir = os.path.join(root, "Metadata")
        
        self.orig2std, self.std2orig = _load_filename_mapping(self.meta_dir)
        split_list = _read_split_list(self.meta_dir, split)
        if split_list is not None:
            rgb_files = split_list
        else:
            rgb_files = sorted([f for f in os.listdir(self.rgb_dir)
                               if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        
        self.use_standardized_names = None
        if rgb_files:
            sample = rgb_files[0]
            self.use_standardized_names = sample in self.std2orig or ("DateTime" in sample)
        else:
            self.use_standardized_names = True
        
        self.samples = self._index_samples(rgb_files)
        
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _index_samples(self, rgb_files: List[str]) -> List[Dict]:
        samples = []
        for rgb_name in rgb_files:
            rgb_path = os.path.join(self.rgb_dir, rgb_name)
            if not os.path.exists(rgb_path):
                continue
            
            base = os.path.splitext(rgb_name)[0]
            mask_candidates = [
                os.path.join(self.mask_dir, f"{base}.png"),
                os.path.join(self.mask_dir, f"{base}.PNG")
            ]
            mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)
            
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
            
            nir_path = None
            if self.use_rgbnir:
                nir_path = _load_nir_band(
                    self.ms_dir, rgb_name, self.use_standardized_names, 
                    self.orig2std, self.std2orig
                )
            
            if mask_path is not None and (not self.use_rgbnir or nir_path is not None):
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
        if _is_uint16(img):
            return img.astype(np.float32) / 65535.0
        return img.astype(np.float32) / 255.0

    def _resize(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_resized = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
        return img_resized, mask_resized

    def _random_flip(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img, mask = np.flip(img, axis=1), np.flip(mask, axis=1)
        if random.random() < 0.25:
            img, mask = np.flip(img, axis=0), np.flip(mask, axis=0)
        return img, mask

    def _random_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            alpha = 1.0 + 0.2 * (random.random() - 0.5)
            beta = 0.1 * (random.random() - 0.5)
            img = np.clip(img * alpha + beta, 0, 1)
        return img

    def _random_noise(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        rgb = self._read_rgb(item["rgb"])
        mask = self._read_mask(item["mask"])
        nir = None

        if self.use_rgbnir and item["nir"] is not None:
            nir = self._read_nir(item["nir"])
            if nir.shape[:2] != rgb.shape[:2]:
                nir = cv2.resize(nir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            rgb = np.concatenate([rgb, nir[..., None]], axis=-1)

        rgb, mask = self._resize(rgb, mask)
        rgb = self._scale_uint(rgb)

        if self.augment:
            rgb, mask = self._random_flip(rgb, mask)
            rgb = self._random_brightness_contrast(rgb)
            rgb = self._random_noise(rgb)

        # Normalize RGB (first 3 channels)
        rgb[..., :3] = (rgb[..., :3] - self.rgb_mean) / self.rgb_std

        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        y = torch.from_numpy(mask.astype(np.int64))
        return {"images": x, "labels": y, "paths": item["rgb"]}


def create_weedy_rice_rgbnir_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    use_rgbnir: bool = True,
    target_size: Tuple[int, int] = (960, 1280)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = WeedyRiceRGBNIRDataset(data_root, split="train", use_rgbnir=use_rgbnir,
                                     target_size=target_size, augment=True)
    val_ds = WeedyRiceRGBNIRDataset(data_root, split="val", use_rgbnir=use_rgbnir,
                                   target_size=target_size, augment=False)
    test_ds = WeedyRiceRGBNIRDataset(data_root, split="test", use_rgbnir=use_rgbnir,
                                    target_size=target_size, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
