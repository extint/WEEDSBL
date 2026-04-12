import os
from typing import Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class DualEncoderWeedyRiceDataset(Dataset):
    """
    Mask convention (raw PNG pixel values):
        0   = background
        128 = crop        (vegetation)
        255 = weed        (vegetation)

    THE BUG IN THE OLD VERSION
    --------------------------
    Old _read_mask did:
        m = m / 255.0          →  bg=0.0, crop=0.502, weed=1.0
        return (m > 0.5)       →  bg=0,   crop=0,     weed=1

    Crop pixels (0.502) barely pass 0.5 and in practice got rounded to 0
    by floating-point, so the model was trained with crop = background.
    The stated goal — vegetation (crop+weed) vs background — was NEVER
    being trained correctly.

    THE FIX
    -------
    Work directly on the raw uint8 values before any division:
        vegetation = (raw > 0)   →  bg=0, crop=1, weed=1
    This is unambiguous regardless of exact pixel values.

    A 'weed_only' mode is also provided for the next stage (weed vs crop
    binary classification within vegetation pixels).
    """

    # Raw mask pixel values in the PNG (uint8)
    BG_VAL   = 0
    CROP_VAL = 1
    WEED_VAL = 2

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_size: Tuple[int, int] = (512, 512),
        augment: bool = True,
        mask_mode: str = "vegetation",   # "vegetation" | "weed_only"
    ):
        """
        mask_mode:
            "vegetation"  →  binary: (crop OR weed)=1, background=0
            "weed_only"   →  binary: weed=1, (background OR crop)=0
            "multiclass"  →  3-class: bg=0, crop=1, weed=2  ← use this for 3-class training
        """
        assert mask_mode in ("vegetation", "weed_only", "multiclass"), \
            f"mask_mode must be 'vegetation'|'weed_only'|'multiclass', got {mask_mode}"

        self.root       = root
        self.split      = split
        self.mask_mode  = mask_mode
        self.target_h, self.target_w = target_size
        self.augment    = augment and split == "train"

        self.rgb_dir   = os.path.join(root, "rgb")
        self.nir_dir   = os.path.join(root, "nir")
        self.mask_dir  = os.path.join(root, "masks")
        self.split_dir = os.path.join(root, "splits")

        split_file = os.path.join(self.split_dir, f"{split}.txt")
        with open(split_file, "r") as f:
            image_ids = [line.strip() for line in f if line.strip()]
        self.samples = self._index_samples(image_ids)

        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
                print("[WARN] AgriculturalAugmentation not found.")

    def _index_samples(self, image_ids):
        samples = []
        for img_id in image_ids:
            rgb_path  = os.path.join(self.rgb_dir,  f"rgb_{img_id}.png")
            nir_path  = os.path.join(self.nir_dir,  f"nir_{img_id}.png")
            mask_path = os.path.join(self.mask_dir, f"mask_{img_id}.png")

            if os.path.exists(rgb_path) and os.path.exists(nir_path) and os.path.exists(mask_path):
                samples.append({"rgb": rgb_path, "nir": nir_path, "mask": mask_path})
            else:
                print(f"[WARN] Missing pair for {img_id}")
        return samples

    def __len__(self):
        return len(self.samples)

    def _read_rgb(self, path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # uint8 (H,W,3)

    def _read_nir(self, path):
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr                                      # uint8 or uint16 (H,W)

    def _read_mask(self, path):
        """
        Read raw uint8 mask and convert to binary according to mask_mode.

        Works on raw pixel values (0 / 128 / 255) BEFORE any division,
        so there is zero ambiguity from floating-point thresholding.
        """
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    # uint8, values in {0,128,255}
        if m is None:
            raise FileNotFoundError(path)

        if self.mask_mode == "vegetation":
            binary = (m > self.BG_VAL).astype(np.uint8)
        elif self.mask_mode == "multiclass":
            # Values are already {0=bg, 1=crop, 2=weed} — pass through directly
            return m.astype(np.uint8)
        else:
            binary = (m == self.WEED_VAL).astype(np.uint8)

        return binary

    def _scale_uint(self, img):
        if img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
        return img.astype(np.float32) / 255.0

    def _resize(self, rgb, nir, mask):
        rgb  = cv2.resize(rgb,  (self.target_w, self.target_h))
        nir  = cv2.resize(nir,  (self.target_w, self.target_h))
        mask = cv2.resize(mask, (self.target_w, self.target_h),
                          interpolation=cv2.INTER_NEAREST)   # no interpolation on labels
        return rgb, nir, mask

    def __getitem__(self, idx):
        item = self.samples[idx]

        rgb  = self._read_rgb(item["rgb"])
        nir  = self._read_nir(item["nir"])
        mask = self._read_mask(item["mask"])

        rgb, nir, mask = self._resize(rgb, nir, mask)

        rgb = self._scale_uint(rgb)
        nir = self._scale_uint(nir)

        if self.aug is not None:
            rgbnir = np.concatenate([rgb, nir[..., None]], axis=-1)
            rgbnir, mask = self.aug(rgbnir, mask)
            rgb  = rgbnir[..., :3]
            nir  = rgbnir[..., 3]

        rgb = (rgb - self.rgb_mean) / self.rgb_std

        rgb_t  = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        nir_t  = torch.from_numpy(nir[None, ...]).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))

        return {
            "rgb":  rgb_t,
            "nir":  nir_t,
            "mask": mask_t,
            "path": item["rgb"],
        }


def create_dual_encoder_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (512, 512),
    mask_mode: str = "multiclass",   # "vegetation" | "weed_only" | "multiclass"
):
    train_ds = DualEncoderWeedyRiceDataset(data_root, split="train",
                                           target_size=target_size,
                                           augment=True,  mask_mode=mask_mode)
    val_ds   = DualEncoderWeedyRiceDataset(data_root, split="val",
                                           target_size=target_size,
                                           augment=False, mask_mode=mask_mode)
    test_ds  = DualEncoderWeedyRiceDataset(data_root, split="test",
                                           target_size=target_size,
                                           augment=False, mask_mode=mask_mode)

    sample_mask = train_ds[0]["mask"].numpy()
    unique_vals = np.unique(sample_mask)
    counts = {v: int((sample_mask == v).sum()) for v in unique_vals}
    total  = sample_mask.size
    print(f"[DataLoader] mask_mode='{mask_mode}'  unique={unique_vals}  "
          + "  ".join(f"cls{v}={counts[v]/total*100:.1f}%" for v in unique_vals))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader