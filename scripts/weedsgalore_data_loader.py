import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from weedutils.augmentations import AgriculturalAugmentation


def _read_split_list(splits_dir: str, split: str) -> Optional[List[str]]:
    """Read image IDs from train.txt or val.txt in splits folder"""
    split_file = os.path.join(splits_dir, f"{split}.txt")
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            items = [ln.strip() for ln in f if ln.strip()]
        return items
    return None


class WeedsGaloreRGBNIRDataset(Dataset):
    """
    WeedsGalore 4-channel RGB+NIR dataset for crop-weed segmentation.

    Dataset structure:
    root/
        2023-05-25/
            images/
                2023-05-25_0109_B.png
                2023-05-25_0109_G.png
                2023-05-25_0109_NIR.png
                2023-05-25_0109_R.png
                2023-05-25_0109_RE.png
                ...
            semantics/
                2023-05-25_0109.png
                ...
        2023-05-30/
        2023-06-06/
        2023-06-15/
        splits/
            train.txt
            val.txt

    The semantics masks contain 3 classes:
        - 0: Background
        - 1: Crop (maize)
        - 2: Weed
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_size: Tuple[int, int] = (600, 600),
        use_rgbnir: bool = True,
        augment: bool = False,
        nir_drop_prob: float = 0.0,
        # Agricultural augmentation parameters
        hue_shift: int = 15,
        sat_shift: int = 30,
        val_shift: int = 20,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        noise_std: float = 0.05,
        flip_prob: float = 0.5,
        hsv_prob: float = 0.7,
        brightness_prob: float = 0.5,
        noise_prob: float = 0.3,
    ):
        """
        Args:
            root: Path to weedsgalore-dataset folder
            split: 'train' or 'val'
            target_size: (height, width) to resize images
            use_rgbnir: If True, load 4 channels (RGB+NIR), else 3 channels (RGB only)
            augment: Apply advanced agricultural augmentations
            nir_drop_prob: Probability of zeroing out NIR channel during training (0.0 to 1.0)
            hue_shift: HSV hue shift range (±degrees)
            sat_shift: HSV saturation shift range (±)
            val_shift: HSV value shift range (±)
            brightness_limit: Brightness adjustment limit (±)
            contrast_limit: Contrast adjustment limit (±)
            noise_std: Gaussian noise standard deviation
            flip_prob: Probability of horizontal flip
            hsv_prob: Probability of HSV jittering
            brightness_prob: Probability of brightness/contrast adjustment
            noise_prob: Probability of adding noise
        """
        self.root = root
        self.split = split
        self.target_h, self.target_w = target_size
        self.use_rgbnir = use_rgbnir
        self.augment = augment and split == "train"
        self.nir_drop_prob = nir_drop_prob

        # Date folders
        self.date_folders = ["2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15"]
        self.splits_dir = os.path.join(root, "splits")

        # Read split file
        split_list = _read_split_list(self.splits_dir, split)
        if split_list is None:
            raise FileNotFoundError(f"Split file not found: {self.splits_dir}/{split}.txt")

        # Index all valid samples
        self.samples = self._index_samples(split_list)

        # ImageNet normalization for RGB
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Initialize agricultural augmentation pipeline
        if self.augment:
            self.augmentor = AgriculturalAugmentation(
                hue_shift=hue_shift,
                sat_shift=sat_shift,
                val_shift=val_shift,
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                noise_std=noise_std,
                flip_prob=flip_prob,
                hsv_prob=hsv_prob,
                brightness_prob=brightness_prob,
                noise_prob=noise_prob
            )
            print(f"[INFO] Loaded {len(self.samples)} samples for {split} split with ADVANCED augmentations")
        else:
            self.augmentor = None
            print(f"[INFO] Loaded {len(self.samples)} samples for {split} split (no augmentation)")

    def _index_samples(self, split_list: List[str]) -> List[Dict]:
        """
        Build sample list by checking existence of R, G, B, NIR bands and semantics mask

        Args:
            split_list: List of image IDs from train.txt/val.txt (e.g., '2023-05-25_0109')

        Returns:
            List of dicts with paths to bands and mask
        """
        samples = []

        for img_id in split_list:
            # Extract date prefix (e.g., '2023-05-25' from '2023-05-25_0109')
            date_prefix = img_id.split('_')[0]  # YYYY-MM-DD format

            # Find which date folder this belongs to
            date_folder = None
            for df in self.date_folders:
                if date_prefix == df:
                    date_folder = df
                    break

            if date_folder is None:
                print(f"Warning: Could not find date folder for {img_id}")
                continue

            # Paths to band images and semantic mask
            images_dir = os.path.join(self.root, date_folder, "images")
            semantics_dir = os.path.join(self.root, date_folder, "semantics")

            r_path = os.path.join(images_dir, f"{img_id}_R.png")
            g_path = os.path.join(images_dir, f"{img_id}_G.png")
            b_path = os.path.join(images_dir, f"{img_id}_B.png")
            nir_path = os.path.join(images_dir, f"{img_id}_NIR.png")
            mask_path = os.path.join(semantics_dir, f"{img_id}.png")

            # Check if all required files exist
            required_paths = [r_path, g_path, b_path, mask_path]
            if self.use_rgbnir:
                required_paths.append(nir_path)

            if all(os.path.exists(p) for p in required_paths):
                samples.append({
                    "r": r_path,
                    "g": g_path,
                    "b": b_path,
                    "nir": nir_path if self.use_rgbnir else None,
                    "mask": mask_path,
                    "id": img_id
                })
            else:
                missing = [p for p in required_paths if not os.path.exists(p)]
                print(f"Warning: Missing files for {img_id}: {missing}")

        return samples

    def __len__(self):
        return len(self.samples)

    def _read_band(self, path: str) -> np.ndarray:
        """Read single band image (grayscale)"""
        band = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if band is None:
            raise FileNotFoundError(f"Band not found: {path}")
        return band

    def _read_mask(self, path: str) -> np.ndarray:
        """
        Read semantic mask with 3 classes:
        0: Background
        1: Crop
        2: Weed
        """
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path}")
        return mask

    def _build_rgbnir(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, 
                      nir: Optional[np.ndarray] = None) -> np.ndarray:
        """Stack R, G, B (and optionally NIR) into multi-channel image"""
        rgb = np.stack([r, g, b], axis=-1)  # Shape: (H, W, 3)

        if self.use_rgbnir and nir is not None:
            # Add NIR as 4th channel
            rgbnir = np.concatenate([rgb, nir[..., None]], axis=-1)  # Shape: (H, W, 4)
            return rgbnir
        return rgb

    def _resize(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image and mask to target size"""
        img_resized = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
        return img_resized, mask_resized

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image:
        - RGB channels: ImageNet normalization
        - NIR channel: Scale to [0, 1] only (no normalization)
        """
        img_norm = img.copy()

        # Normalize RGB channels (first 3 channels)
        img_norm[..., :3] = (img_norm[..., :3] - self.rgb_mean) / self.rgb_std

        # NIR channel (if present) is already in [0, 1], keep as is
        return img_norm

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
                - 'images': Tensor of shape (C, H, W) where C=3 (RGB) or C=4 (RGB+NIR)
                - 'labels': Tensor of shape (H, W) with class indices {0, 1, 2}
                - 'paths': Image ID string
        """
        item = self.samples[idx]

        # Read individual bands
        r = self._read_band(item["r"])
        g = self._read_band(item["g"])
        b = self._read_band(item["b"])
        nir = self._read_band(item["nir"]) if self.use_rgbnir else None
        mask = self._read_mask(item["mask"])

        # Build multi-channel image
        img = self._build_rgbnir(r, g, b, nir)

        # Resize
        img, mask = self._resize(img, mask)

        # Scale to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Apply ADVANCED agricultural augmentation
        # (includes: flips, HSV jitter, brightness/contrast, noise)
        if self.augment and self.augmentor is not None:
            img, mask = self.augmentor(img, mask)

        # NIR drop augmentation (applied AFTER augmentation, BEFORE normalization)
        if self.use_rgbnir and self.nir_drop_prob > 0 and self.split == "train":
            if np.random.rand() < self.nir_drop_prob:
                # Zero out the NIR channel (channel index 3)
                img[..., 3] = 0.0

        # Normalize
        img = self._normalize(img)

        # Convert to torch tensors
        # Image: (H, W, C) -> (C, H, W)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()

        # Mask: (H, W) with class indices
        y = torch.from_numpy(mask.astype(np.int64))
        y[y > 1] = 2  # Fix: class ids 0,1,2,...5 where >1 are weeds

        return {
            "images": x,
            "labels": y,
            "paths": item["id"]
        }


def create_weedsgalore_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    use_rgbnir: bool = True,
    target_size: Tuple[int, int] = (600, 600),
    nir_drop_prob: float = 0.0,
    # Augmentation parameters
    augment: bool = True,
    hue_shift: int = 15,
    sat_shift: int = 30,
    val_shift: int = 20,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    noise_std: float = 0.05,
    flip_prob: float = 0.5,
    hsv_prob: float = 0.7,
    brightness_prob: float = 0.5,
    noise_prob: float = 0.3,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for WeedsGalore dataset.

    Args:
        data_root: Path to weedsgalore-dataset folder
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        use_rgbnir: If True, load 4 channels (RGB+NIR), else 3 channels (RGB only)
        target_size: (height, width) to resize images
        nir_drop_prob: Probability of zeroing out NIR channel during training (0.0 to 1.0)
        augment: Enable advanced agricultural augmentations for training
        hue_shift: HSV hue shift range (±degrees)
        sat_shift: HSV saturation shift range (±)
        val_shift: HSV value shift range (±)
        brightness_limit: Brightness adjustment limit (±)
        contrast_limit: Contrast adjustment limit (±)
        noise_std: Gaussian noise standard deviation
        flip_prob: Probability of horizontal flip
        hsv_prob: Probability of HSV jittering
        brightness_prob: Probability of brightness/contrast adjustment
        noise_prob: Probability of adding noise

    Returns:
        (train_loader, val_loader)
    """
    train_ds = WeedsGaloreRGBNIRDataset(
        data_root, 
        split="train", 
        use_rgbnir=use_rgbnir,
        target_size=target_size, 
        augment=augment,
        nir_drop_prob=nir_drop_prob,
        hue_shift=hue_shift,
        sat_shift=sat_shift,
        val_shift=val_shift,
        brightness_limit=brightness_limit,
        contrast_limit=contrast_limit,
        noise_std=noise_std,
        flip_prob=flip_prob,
        hsv_prob=hsv_prob,
        brightness_prob=brightness_prob,
        noise_prob=noise_prob
    )

    val_ds = WeedsGaloreRGBNIRDataset(
        data_root, 
        split="val", 
        use_rgbnir=use_rgbnir,
        target_size=target_size, 
        augment=False,  # No augmentation for validation
        nir_drop_prob=0.0  # Never drop NIR during validation
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader


# Example usage:
if __name__ == "__main__":
    # Set your dataset root path //vjti-comp
    dataset_root = "/home/vedantmehra/Downloads/weedsgalore-dataset"

    # Create dataloaders with RGB+NIR (4 channels), NIR drop, and ADVANCED augmentation
    train_loader, val_loader = create_weedsgalore_dataloaders(
        data_root=dataset_root,
        batch_size=8,
        num_workers=4,
        use_rgbnir=True,
        target_size=(600, 600),
        nir_drop_prob=0.3,  # 30% chance to drop NIR during training
        augment=True,  # Enable advanced augmentation
        # Augmentation parameters (defaults shown, can be customized)
        hue_shift=15,
        sat_shift=30,
        val_shift=20,
        brightness_limit=0.2,
        contrast_limit=0.2,
        noise_std=0.05,
        flip_prob=0.5,
        hsv_prob=0.7,
        brightness_prob=0.5,
        noise_prob=0.3
    )

    # Test loading a batch
    print("\nTesting data loading with augmentation...")
    for batch in train_loader:
        images = batch["images"]  # Shape: (B, 4, 600, 600) for RGB+NIR
        labels = batch["labels"]  # Shape: (B, 600, 600)
        paths = batch["paths"]

        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique classes in batch: {torch.unique(labels)}")
        print(f"Sample IDs: {paths[:3]}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        break
