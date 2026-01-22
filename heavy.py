"""
Heavy Augmentation Pipeline for Dual-Encoder Transformer
- Multi-scale random crops (4-8 crops per image)
- Aggressive spatial transforms (rotation, scale, shear)
- CutMix and Mosaic augmentation
- Agricultural-specific color/lighting
- Effective dataset expansion: 5-10x original size
"""

import os
import csv
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional


# ============================================================
# HEAVY AUGMENTATION CLASS
# ============================================================

class HeavyAgriculturalAugmentation:
    """
    Aggressive augmentation for transformer models.
    Transformers need MORE diverse data than CNNs.
    """
    def __init__(
        self,
        # Spatial augmentations
        rotation_limit=45,
        scale_limit=(0.7, 1.3),
        shear_limit=15,
        crop_scale=(0.5, 1.0),
        aspect_ratio=(0.8, 1.2),
        
        # Color augmentations
        hue_shift=25,
        sat_shift=40,
        val_shift=30,
        brightness_limit=0.3,
        contrast_limit=0.3,
        gamma_limit=(0.7, 1.3),
        
        # Advanced augmentations
        cutmix_prob=0.3,
        grid_distort_prob=0.2,
        elastic_prob=0.15,
        blur_prob=0.2,
        noise_prob=0.4,
        shadow_prob=0.3,
        
        # Always-on
        flip_prob=0.5,
    ):
        self.rotation_limit = rotation_limit
        self.scale_limit = scale_limit
        self.shear_limit = shear_limit
        self.crop_scale = crop_scale
        self.aspect_ratio = aspect_ratio
        
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.gamma_limit = gamma_limit
        
        self.cutmix_prob = cutmix_prob
        self.grid_distort_prob = grid_distort_prob
        self.elastic_prob = elastic_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.shadow_prob = shadow_prob
        self.flip_prob = flip_prob
    
    def random_flip(self, rgb, nir, mask):
        """Horizontal and vertical flips"""
        if random.random() < self.flip_prob:
            rgb = np.flip(rgb, axis=1).copy()
            nir = np.flip(nir, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        
        if random.random() < 0.3:
            rgb = np.flip(rgb, axis=0).copy()
            nir = np.flip(nir, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        
        return rgb, nir, mask
    
    def random_rotation(self, rgb, nir, mask):
        """Random rotation with angle sampling"""
        angle = random.uniform(-self.rotation_limit, self.rotation_limit)
        H, W = rgb.shape[:2]
        center = (W // 2, H // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rgb = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        nir = cv2.warpAffine(nir, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        
        return rgb, nir, mask
    
    def random_affine(self, rgb, nir, mask):
        """Random scale, shear, translation"""
        H, W = rgb.shape[:2]
        
        # Scale
        scale = random.uniform(*self.scale_limit)
        
        # Shear
        shear_x = random.uniform(-self.shear_limit, self.shear_limit)
        shear_y = random.uniform(-self.shear_limit, self.shear_limit)
        
        # Affine matrix
        M = np.array([
            [scale, shear_x / 180 * np.pi, 0],
            [shear_y / 180 * np.pi, scale, 0]
        ], dtype=np.float32)
        
        rgb = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        nir = cv2.warpAffine(nir, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        
        return rgb, nir, mask
    
    def random_crop(self, rgb, nir, mask, output_size):
        """Random crop to output_size"""
        H, W = rgb.shape[:2]
        target_h, target_w = output_size
        
        if H < target_h or W < target_w:
            # Pad if too small
            pad_h = max(0, target_h - H)
            pad_w = max(0, target_w - W)
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            nir = np.pad(nir, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
            H, W = rgb.shape[:2]
        
        # Random top-left corner
        top = random.randint(0, H - target_h)
        left = random.randint(0, W - target_w)
        
        rgb = rgb[top:top+target_h, left:left+target_w]
        nir = nir[top:top+target_h, left:left+target_w]
        mask = mask[top:top+target_h, left:left+target_w]
        
        return rgb, nir, mask
    
    def hsv_jitter(self, rgb):
        """Aggressive HSV jittering"""
        rgb_uint8 = (np.clip(rgb * 255, 0, 255)).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        h_shift = random.uniform(-self.hue_shift, self.hue_shift)
        s_shift = random.uniform(-self.sat_shift, self.sat_shift)
        v_shift = random.uniform(-self.val_shift, self.val_shift)
        
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + h_shift, 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_shift, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v_shift, 0, 255)
        
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return rgb
    
    def brightness_contrast(self, rgb):
        """Random brightness and contrast"""
        alpha = random.uniform(1 - self.contrast_limit, 1 + self.contrast_limit)
        beta = random.uniform(-self.brightness_limit, self.brightness_limit)
        rgb = np.clip(rgb * alpha + beta, 0, 1)
        return rgb
    
    def gamma_transform(self, rgb):
        """Random gamma correction"""
        gamma = random.uniform(*self.gamma_limit)
        rgb = np.clip(rgb ** gamma, 0, 1)
        return rgb
    
    def add_shadow(self, rgb):
        """Add random shadow overlay (simulates field shadows)"""
        if random.random() > self.shadow_prob:
            return rgb
        
        H, W = rgb.shape[:2]
        
        # Random shadow region
        x1, y1 = random.randint(0, W//2), random.randint(0, H//2)
        x2, y2 = random.randint(W//2, W), random.randint(H//2, H)
        
        shadow_mask = np.zeros((H, W), dtype=np.float32)
        shadow_mask[y1:y2, x1:x2] = 1.0
        shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 30)
        
        # Darken
        shadow_strength = random.uniform(0.3, 0.7)
        rgb = rgb * (1 - shadow_mask[..., None] * shadow_strength)
        return rgb
    
    def add_blur(self, rgb, nir):
        """Random blur (motion or gaussian)"""
        if random.random() > self.blur_prob:
            return rgb, nir
        
        kernel_size = random.choice([3, 5, 7])
        
        if random.random() < 0.5:
            # Gaussian blur
            rgb = cv2.GaussianBlur(rgb, (kernel_size, kernel_size), 0)
            nir = cv2.GaussianBlur(nir, (kernel_size, kernel_size), 0)
        else:
            # Motion blur
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            rgb = cv2.filter2D(rgb, -1, kernel)
            nir = cv2.filter2D(nir, -1, kernel)
        
        return rgb, nir
    
    def add_noise(self, rgb, nir):
        """Gaussian noise"""
        if random.random() > self.noise_prob:
            return rgb, nir
        
        noise_std = random.uniform(0.01, 0.05)
        rgb = np.clip(rgb + np.random.normal(0, noise_std, rgb.shape), 0, 1)
        nir = np.clip(nir + np.random.normal(0, noise_std, nir.shape), 0, 1)
        return rgb, nir
    
    def cutmix(self, rgb, nir, mask, rgb2, nir2, mask2):
        """CutMix: cut and paste region from another image"""
        if random.random() > self.cutmix_prob or rgb2 is None:
            return rgb, nir, mask
        
        H, W = rgb.shape[:2]
        
        # Random box
        cut_ratio = random.uniform(0.2, 0.5)
        cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
        
        cx = random.randint(cut_w // 2, W - cut_w // 2)
        cy = random.randint(cut_h // 2, H - cut_h // 2)
        
        x1, x2 = cx - cut_w // 2, cx + cut_w // 2
        y1, y2 = cy - cut_h // 2, cy + cut_h // 2
        
        # Paste from second image
        rgb[y1:y2, x1:x2] = rgb2[y1:y2, x1:x2]
        nir[y1:y2, x1:x2] = nir2[y1:y2, x1:x2]
        mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
        
        return rgb, nir, mask
    
    def grid_distortion(self, rgb, nir, mask):
        """Grid-based distortion"""
        if random.random() > self.grid_distort_prob:
            return rgb, nir, mask
        
        H, W = rgb.shape[:2]
        
        # Create distortion mesh
        num_steps = 5
        distort_limit = 0.1
        
        x_step = W // num_steps
        y_step = H // num_steps
        
        xx = np.zeros((num_steps + 1, num_steps + 1), dtype=np.float32)
        yy = np.zeros((num_steps + 1, num_steps + 1), dtype=np.float32)
        
        for i in range(num_steps + 1):
            for j in range(num_steps + 1):
                xx[i, j] = j * x_step + random.uniform(-distort_limit * x_step, distort_limit * x_step)
                yy[i, j] = i * y_step + random.uniform(-distort_limit * y_step, distort_limit * y_step)
        
        # Apply distortion
        map_x = cv2.resize(xx, (W, H)).astype(np.float32)
        map_y = cv2.resize(yy, (W, H)).astype(np.float32)
        
        rgb = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        nir = cv2.remap(nir, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        
        return rgb, nir, mask
    
    def __call__(self, rgb, nir, mask, rgb2=None, nir2=None, mask2=None, target_size=(512, 512)):
        """Apply full augmentation pipeline"""
        
        # Spatial transforms
        rgb, nir, mask = self.random_flip(rgb, nir, mask)
        
        if random.random() < 0.5:
            rgb, nir, mask = self.random_rotation(rgb, nir, mask)
        
        if random.random() < 0.4:
            rgb, nir, mask = self.random_affine(rgb, nir, mask)
        
        if random.random() < 0.3:
            rgb, nir, mask = self.grid_distortion(rgb, nir, mask)
        
        # Crop to target size
        rgb, nir, mask = self.random_crop(rgb, nir, mask, target_size)
        
        # CutMix (needs second image)
        if rgb2 is not None:
            rgb, nir, mask = self.cutmix(rgb, nir, mask, rgb2, nir2, mask2)
        
        # Color transforms (RGB only)
        if random.random() < 0.8:
            rgb = self.hsv_jitter(rgb)
        
        if random.random() < 0.7:
            rgb = self.brightness_contrast(rgb)
        
        if random.random() < 0.4:
            rgb = self.gamma_transform(rgb)
        
        if random.random() < self.shadow_prob:
            rgb = self.add_shadow(rgb)
        
        # Noise and blur (both modalities)
        rgb, nir = self.add_blur(rgb, nir)
        rgb, nir = self.add_noise(rgb, nir)
        
        return rgb, nir, mask


# ============================================================
# DATASET WITH MULTI-CROP EXPANSION
# ============================================================

class HeavyAugmentedDualEncoderDataset(Dataset):
    """
    Dataset that generates MULTIPLE crops per image during training.
    Effective dataset expansion: 5-10x original size.
    
    Each image generates:
    - 4-8 random crops
    - Various augmentations per crop
    - CutMix between random pairs
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_size: Tuple[int, int] = (512, 512),
        crops_per_image: int = 6,  # DATASET EXPANSION FACTOR
        augment: bool = True,
    ):
        self.root = root
        self.split = split
        self.target_size = target_size
        self.crops_per_image = crops_per_image if split == "train" else 1
        self.augment = augment and split == "train"
        
        self.rgb_dir = os.path.join(root, "RGB")
        self.ms_dir = os.path.join(root, "Multispectral")
        self.mask_dir = os.path.join(root, "Masks")
        self.meta_dir = os.path.join(root, "Metadata")
        
        # Load filename mappings
        self.orig2std, self.std2orig = self._load_filename_mapping()
        
        # Load split
        split_list = self._read_split_list(split)
        if split_list:
            rgb_files = split_list
        else:
            rgb_files = sorted([f for f in os.listdir(self.rgb_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        self.use_standardized = (rgb_files[0] in self.std2orig or "DateTime" in rgb_files[0]) if rgb_files else True
        
        # Index samples
        self.samples = self._index_samples(rgb_files)
        
        print(f"[{split.upper()}] Loaded {len(self.samples)} images")
        print(f"[{split.upper()}] Effective dataset size: {len(self)} samples (crops_per_image={self.crops_per_image})")
        
        # ImageNet normalization
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Augmentation
        if self.augment:
            self.aug = HeavyAgriculturalAugmentation()
        else:
            self.aug = None
    
    def _load_filename_mapping(self):
        map_path = os.path.join(self.meta_dir, "filename_mapping.csv")
        orig2std, std2orig = {}, {}
        if os.path.exists(map_path):
            with open(map_path, newline="") as f:
                for row in csv.DictReader(f):
                    o, s = row["original_filename"], row["standardized_filename"]
                    orig2std[o] = s
                    std2orig[s] = o
        return orig2std, std2orig
    
    def _read_split_list(self, split):
        split_file = os.path.join(self.meta_dir, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file) as f:
                return [ln.strip() for ln in f if ln.strip()]
        return None
    
    def _load_nir_band(self, rgb_name):
        base_std = os.path.splitext(rgb_name)[0]
        
        if self.use_standardized:
            std_candidate = os.path.join(self.ms_dir, f"{base_std}_NIR.TIF")
            if os.path.exists(std_candidate):
                return std_candidate
            
            if base_std + ".JPG" in self.std2orig:
                orig_rgb = self.std2orig[base_std + ".JPG"]
                base_org = os.path.splitext(orig_rgb)[0].replace("_D", "")
                org_candidate = os.path.join(self.ms_dir, f"{base_org}_MS_NIR.TIF")
                if os.path.exists(org_candidate):
                    return org_candidate
        else:
            base_org = os.path.splitext(rgb_name)[0].replace("_D", "")
            org_candidate = os.path.join(self.ms_dir, f"{base_org}_MS_NIR.TIF")
            if os.path.exists(org_candidate):
                return org_candidate
            
            if rgb_name in self.orig2std:
                std_base = os.path.splitext(self.orig2std[rgb_name])[0]
                std_candidate = os.path.join(self.ms_dir, f"{std_base}_NIR.TIF")
                if os.path.exists(std_candidate):
                    return std_candidate
        
        return None
    
    def _index_samples(self, rgb_files):
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
            if not mask_path:
                if self.use_standardized and rgb_name in self.std2orig:
                    orig_base = os.path.splitext(self.std2orig[rgb_name])[0]
                    for ext in (".png", ".PNG"):
                        cand = os.path.join(self.mask_dir, f"{orig_base}{ext}")
                        if os.path.exists(cand):
                            mask_path = cand
                            break
            
            nir_path = self._load_nir_band(rgb_name)
            
            if mask_path and nir_path:
                samples.append({"rgb": rgb_path, "nir": nir_path, "mask": mask_path})
        
        return samples
    
    def __len__(self):
        # Effective length = num_images * crops_per_image
        return len(self.samples) * self.crops_per_image
    
    def _load_image(self, path, mode='rgb'):
        if mode == 'rgb':
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"RGB not found: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'nir':
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"NIR not found: {path}")
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:  # mask
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Mask not found: {path}")
            return (img == 255).astype(np.uint8)
    
    def _scale_uint(self, img):
        if img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
        return img.astype(np.float32) / 255.0
    
    def __getitem__(self, idx):
        # Map idx to base image and crop number
        base_idx = idx // self.crops_per_image
        crop_idx = idx % self.crops_per_image
        
        sample = self.samples[base_idx]
        
        # Load primary image
        rgb = self._load_image(sample["rgb"], 'rgb')
        nir = self._load_image(sample["nir"], 'nir')
        mask = self._load_image(sample["mask"], 'mask')
        
        # Align NIR to RGB
        if nir.shape[:2] != rgb.shape[:2]:
            nir = cv2.resize(nir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Scale to [0, 1]
        rgb = self._scale_uint(rgb)
        nir = self._scale_uint(nir)
        
        # For CutMix: load a second random image
        rgb2, nir2, mask2 = None, None, None
        if self.augment and random.random() < 0.3:
            rand_idx = random.randint(0, len(self.samples) - 1)
            if rand_idx != base_idx:
                rand_sample = self.samples[rand_idx]
                rgb2 = self._scale_uint(self._load_image(rand_sample["rgb"], 'rgb'))
                nir2 = self._scale_uint(self._load_image(rand_sample["nir"], 'nir'))
                mask2 = self._load_image(rand_sample["mask"], 'mask')
                
                # Resize to match
                if rgb2.shape[:2] != rgb.shape[:2]:
                    rgb2 = cv2.resize(rgb2, (rgb.shape[1], rgb.shape[0]))
                    nir2 = cv2.resize(nir2, (nir.shape[1], nir.shape[0]))
                    mask2 = cv2.resize(mask2, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentations
        if self.aug:
            rgb, nir, mask = self.aug(rgb, nir, mask, rgb2, nir2, mask2, target_size=self.target_size)
        else:
            # Validation: center crop
            H, W = rgb.shape[:2]
            target_h, target_w = self.target_size
            top = (H - target_h) // 2
            left = (W - target_w) // 2
            rgb = rgb[top:top+target_h, left:left+target_w]
            nir = nir[top:top+target_h, left:left+target_w]
            mask = mask[top:top+target_h, left:left+target_w]
        
        # Normalize RGB
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        
        # To tensor
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        nir_t = torch.from_numpy(nir[None, ...]).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))
        
        return {"rgb": rgb_t, "nir": nir_t, "mask": mask_t, "path": sample["rgb"]}


# ============================================================
# DATA LOADER FACTORY
# ============================================================

def create_heavy_augmented_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (384, 384),
    crops_per_image_train: int = 6,  # Effective training set = 6x original
    crops_per_image_val: int = 1,
):
    """
    Create dataloaders with heavy augmentation.
    
    Args:
        crops_per_image_train: How many crops per training image (dataset expansion)
            - 4: Conservative (4x dataset)
            - 6: Balanced (6x dataset)
            - 8: Aggressive (8x dataset)
    """
    train_ds = HeavyAugmentedDualEncoderDataset(
        root=data_root,
        split="train",
        target_size=target_size,
        crops_per_image=crops_per_image_train,
        augment=True,
    )
    
    val_ds = HeavyAugmentedDualEncoderDataset(
        root=data_root,
        split="val",
        target_size=target_size,
        crops_per_image=crops_per_image_val,
        augment=False,
    )
    
    test_ds = HeavyAugmentedDualEncoderDataset(
        root=data_root,
        split="test",
        target_size=target_size,
        crops_per_image=1,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True  # For stable batch stats
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Test the pipeline
    print("Testing Heavy Augmentation Pipeline...")
    
    DATA_ROOT = "/path/to/WeedyRice"  # CHANGE THIS
    
    train_loader, val_loader, test_loader = create_heavy_augmented_dataloaders(
        data_root=DATA_ROOT,
        batch_size=4,
        num_workers=4,
        target_size=(384, 384),
        crops_per_image_train=6,  # 6x dataset expansion
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples (effective)")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  RGB: {batch['rgb'].shape}")
    print(f"  NIR: {batch['nir'].shape}")
    print(f"  Mask: {batch['mask'].shape}")
    
    print("\n✓ Heavy augmentation pipeline ready!")
