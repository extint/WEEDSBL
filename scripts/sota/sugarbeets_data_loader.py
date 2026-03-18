import os
from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from weedutils.augmentations import AgriculturalAugmentation

from torch.utils.data import Subset
import numpy as np

def _is_uint16(img: np.ndarray) -> bool:
    """Check if image is uint16 format"""
    return img is not None and img.dtype == np.uint16


class SugarBeetDataset(Dataset):
    """
    Dataset loader for RGB + NIR + Mask triplets with 3-class segmentation
    
    Classes:
        - 0: background
        - 1: crop
        - 2: weed
    
    Expected directory structure:
        data_root/
            rgb/
                rgb_146_1021.png
                rgb_147_1022.png
                ...
            nir/
                nir_146_1021.png
                nir_147_1022.png
                ...
            masks/
                mask_146_1021.png
                mask_147_1022.png
                ...
    
    Args:
        root: Root directory containing rgb/, nir/, masks/ folders
        split: Dataset split name (e.g., 'train', 'val', 'test')
        target_size: Target (height, width) for resizing
        use_rgbnir: If True, concatenate NIR as 4th channel; if False, use RGB only
        augment: Whether to apply data augmentation
        nir_drop_prob: Probability of zeroing out NIR channel during training
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_size: Tuple[int, int] = (966, 1296),
        use_rgbnir: bool = True,
        augment: bool = True,
        nir_drop_prob: float = 0.0
    ):
        self.root = root
        self.split = split
        self.target_h, self.target_w = target_size
        self.use_rgbnir = use_rgbnir
        self.nir_drop_prob = nir_drop_prob
        self.augment = augment and split == "train"
        
        # Define directories
        self.rgb_dir = os.path.join(root, "rgb")
        self.nir_dir = os.path.join(root, "nir")
        self.mask_dir = os.path.join(root, "masks")
        
        # Validate directories exist
        for dir_path, name in [(self.rgb_dir, "rgb"), 
                                (self.mask_dir, "masks")]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
        
        if self.use_rgbnir and not os.path.exists(self.nir_dir):
            raise FileNotFoundError(f"NIR directory not found: {self.nir_dir}")
        
        # Index all valid triplets
        self.samples = self._index_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {root}")
        
        print(f"[{split}] Found {len(self.samples)} valid samples")
        
        # ImageNet normalization for RGB channels
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Agricultural augmentation pipeline
        if self.augment:
            self.aug = AgriculturalAugmentation(
                hue_shift=15,
                sat_shift=30,
                val_shift=20,
                brightness_limit=0.25,
                contrast_limit=0.25,
                noise_std=0.05,
                flip_prob=0.5,
                hsv_prob=0.7,
                brightness_prob=0.6,
                noise_prob=0.3
            )
    
    def _get_image_id(self, filename: str) -> str:
        """
        Extract image ID from filename (without extension and prefix)
        Example: 'rgb_146_1021.png' -> '146_1021'
        """
        base = os.path.splitext(filename)[0]  # Remove extension
        # Remove prefix (rgb_, nir_, mask_)
        if base.startswith('rgb_'):
            return base[4:]
        elif base.startswith('nir_'):
            return base[4:]
        elif base.startswith('mask_'):
            return base[5:]
        return base
    
    def _find_matching_file(self, img_id: str, directory: str, 
                           prefix: str) -> str:
        """
        Find file with matching ID in directory with given prefix
        Example: img_id='146_1021', prefix='mask_' -> 'mask_146_1021.png'
        """
        candidate = os.path.join(directory, f"{prefix}{img_id}.png")
        if os.path.exists(candidate):
            return candidate
        return None
    
    def _index_samples(self) -> List[Dict[str, str]]:
        """
        Index all valid triplets (RGB + Mask + optional NIR)
        Uses RGB files as the reference and finds matching mask/NIR files
        """
        samples = []
        
        # Get all RGB files (they start with 'rgb_')
        rgb_files = sorted([f for f in os.listdir(self.rgb_dir)
                           if f.startswith('rgb_') and f.endswith('.png')])
        
        for rgb_file in rgb_files:
            img_id = self._get_image_id(rgb_file)  # Extract ID without 'rgb_' prefix
            rgb_path = os.path.join(self.rgb_dir, rgb_file)
            
            # Find matching mask with 'mask_' prefix
            mask_path = self._find_matching_file(img_id, self.mask_dir, 'mask_')
            if mask_path is None:
                print(f"Warning: No mask found for {rgb_file}, skipping...")
                continue
            
            # Find matching NIR with 'nir_' prefix (only if using RGBNIR)
            nir_path = None
            if self.use_rgbnir:
                nir_path = self._find_matching_file(img_id, self.nir_dir, 'nir_')
                if nir_path is None:
                    print(f"Warning: No NIR found for {rgb_file}, skipping...")
                    continue
            
            # Valid triplet found
            samples.append({
                "rgb": rgb_path,
                "mask": mask_path,
                "nir": nir_path,
                "img_id": img_id
            })
        
        return samples
    
    def _read_rgb(self, path: str) -> np.ndarray:
        """Read RGB image and convert from BGR to RGB"""
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"RGB image not found: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    def _read_nir(self, path: str) -> np.ndarray:
        """Read NIR image (supports single-channel or multi-channel TIF)"""
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"NIR image not found: {path}")
        
        # If multi-channel, extract first channel
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        
        return arr
    
    def _read_mask(self, path: str) -> np.ndarray:
        """
        Read mask with 3 classes:
        - 0: background
        - 1: crop
        - 2: weed
        
        Assumes mask pixel values are already 0, 1, 2
        """
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path}")
        
        # Verify mask contains valid class indices
        unique_values = np.unique(mask)
        if not np.all(np.isin(unique_values, [0, 1, 2])):
            print(f"Warning: Mask {path} contains unexpected values: {unique_values}")
        
        return mask.astype(np.uint8)
    
    def _scale_to_float(self, img: np.ndarray) -> np.ndarray:
        """Scale image to [0, 1] range based on dtype"""
        if _is_uint16(img):
            return img.astype(np.float32) / 65535.0
        return img.astype(np.float32) / 255.0
    
    def _resize(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image and mask to target size"""
        img_resized = cv2.resize(
            img, (self.target_w, self.target_h), 
            interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            mask, (self.target_w, self.target_h), 
            interpolation=cv2.INTER_NEAREST
        )
        return img_resized, mask_resized
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - images: (C, H, W) tensor where C=3 (RGB) or C=4 (RGBNIR)
                - labels: (H, W) tensor with class indices (0=background, 1=crop, 2=weed)
                - paths: path to RGB image
                - img_id: image identifier
        """
        item = self.samples[idx]
        
        # Load RGB
        rgb = self._read_rgb(item["rgb"])
        
        # Load mask
        mask = self._read_mask(item["mask"])
        
        # Load and concatenate NIR if needed
        if self.use_rgbnir and item["nir"] is not None:
            nir = self._read_nir(item["nir"])
            
            # Resize NIR to match RGB dimensions if needed
            if nir.shape[:2] != rgb.shape[:2]:
                nir = cv2.resize(
                    nir, (rgb.shape[1], rgb.shape[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Concatenate NIR as 4th channel
            rgb = np.concatenate([rgb, nir[..., None]], axis=-1)
        
        # Resize to target size
        rgb, mask = self._resize(rgb, mask)
        
        # Scale to [0, 1]
        rgb = self._scale_to_float(rgb)
        
        # Apply augmentations BEFORE normalization (for HSV jittering to work)
        if self.augment:
            rgb, mask = self.aug(rgb, mask)
        
        # NIR dropout for regularization during training
        if self.use_rgbnir and self.nir_drop_prob > 0 and self.split == "train":
            if np.random.rand() < self.nir_drop_prob:
                # Zero out the NIR channel (channel index 3)
                rgb[:, :, 3] = 0.0
        
        # Normalize RGB channels (first 3) using ImageNet statistics
        rgb[..., :3] = (rgb[..., :3] - self.rgb_mean) / self.rgb_std
        
        # Convert to PyTorch tensors
        # Image: (H, W, C) -> (C, H, W)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        
        # Mask: (H, W) with class indices
        y = torch.from_numpy(mask.astype(np.int64))
        
        return {
            "images": x,
            "labels": y,
            "paths": item["rgb"],
            "img_id": item["img_id"]
        }

def create_sugarbeets_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    use_rgbnir: bool = True,
    target_size: Tuple[int, int] = (966, 1296),
    nir_drop_prob: float = 0.0,
    seed: int = 42,
    stratified: bool = False,
    stratified_dir: str = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Create ONE base dataset for ALL splits
    base_ds = SugarBeetDataset(
        root=data_root,
        split="train",  # only controls augmentation
        use_rgbnir=use_rgbnir,
        target_size=target_size,
        augment=False   # No augmentation for consistent indexing
    )
    
    n = len(base_ds)
    # print(n)
    print(f"Full dataset: {n} samples")
    
    if stratified and stratified_dir:
        print(f"[INFO] Using pre-computed stratified splits from {stratified_dir}")
        
        # Read split files
        train_file = os.path.join(stratified_dir, "train.txt")
        val_file = os.path.join(stratified_dir, "val.txt")
        test_file = os.path.join(stratified_dir, "test.txt")
        
        train_ids = set(read_txt_file(train_file))
        val_ids = set(read_txt_file(val_file))
        test_ids = set(read_txt_file(test_file))
        
        print(f"Loaded splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
        
        # FIXED: Match img_id from dataset.samples, not CSV filenames
        train_idx = []
        val_idx = []
        test_idx = []
        
        # print(train_ids)
        for i, sample in enumerate(base_ds.samples):
            img_id = sample['img_id']  # This is '146_1021' format
            # print(img_id)
            if img_id in train_ids:
                train_idx.append(i)
                # print(train_idx)
            elif img_id in val_ids:
                val_idx.append(i)
            elif img_id in test_ids:
                test_idx.append(i)
        
        print(f"Matched indices: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        if len(train_idx) == 0:
            raise ValueError("No training samples found! Check filename matching.")
    
    else:
        # Random split (unchanged)
        indices = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        
        n_train = int(train_split * n)
        n_val = int(val_split * n)
        n_test = n - n_train - n_val
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
    
    # Create subsets from SAME base dataset
    train_ds = Subset(base_ds, train_idx)
    
    # Create train-specific dataset with augmentation
    train_aug_ds = SugarBeetDataset(
        root=data_root,
        split="train",
        use_rgbnir=use_rgbnir,
        target_size=target_size,
        augment=True,
        nir_drop_prob=nir_drop_prob
    )
    train_ds = Subset(train_aug_ds, train_idx)
    
    val_ds = Subset(base_ds, val_idx)
    test_ds = Subset(base_ds, test_idx)
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    print(f"Final dataset sizes:")
    print(f" Train: {len(train_ds)}")
    print(f" Val:   {len(val_ds)}")
    print(f" Test:  {len(test_ds)}")
    
    return train_loader, val_loader, test_loader


def read_txt_file(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stratified split file not found: {file_path}")
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# def create_sugarbeets_dataloaders(
#     data_root: str,
#     batch_size: int = 8,
#     num_workers: int = 4,
#     use_rgbnir: bool = True,
#     target_size: Tuple[int, int] = (966, 1296),
#     nir_drop_prob: float = 0.0,
#     seed: int = 42
# ) -> Tuple[DataLoader, DataLoader, DataLoader]:

#     # ---------------------------
#     # Create FULL dataset (no split here)
#     # ---------------------------
#     full_ds = SugarBeetDataset(
#         root=data_root,
#         split="train",          # only controls augmentation
#         use_rgbnir=use_rgbnir,
#         target_size=target_size,
#         augment=True,
#         nir_drop_prob=nir_drop_prob
#     )

#     n = len(full_ds)
#     indices = np.arange(n)

#     # Reproducible shuffle
#     rng = np.random.default_rng(seed)
#     rng.shuffle(indices)

#     # ---------------------------
#     # Split sizes
#     # ---------------------------
#     n_train = int(0.8 * n)
#     n_val   = int(0.1 * n)
#     n_test  = n - n_train - n_val

#     train_idx = indices[:n_train]
#     val_idx   = indices[n_train:n_train + n_val]
#     test_idx  = indices[n_train + n_val:]

#     # ---------------------------
#     # Create subsets
#     # ---------------------------
#     train_ds = Subset(full_ds, train_idx)

#     # Validation & test must NOT augment
#     val_base = SugarBeetDataset(
#         root=data_root,
#         split="val",
#         use_rgbnir=use_rgbnir,
#         target_size=target_size,
#         augment=False
#     )
#     test_base = SugarBeetDataset(
#         root=data_root,
#         split="test",
#         use_rgbnir=use_rgbnir,
#         target_size=target_size,
#         augment=False
#     )

#     val_ds  = Subset(val_base, val_idx)
#     test_ds = Subset(test_base, test_idx)

#     # ---------------------------
#     # DataLoaders
#     # ---------------------------
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True
#     )

#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     test_loader = DataLoader(
#         test_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     print(f"Dataset split:")
#     print(f"  Train: {len(train_ds)}")
#     print(f"  Val  : {len(val_ds)}")
#     print(f"  Test : {len(test_ds)}")

#     return train_loader, val_loader, test_loader


