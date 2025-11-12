import os
import glob
from typing import Tuple, List
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CWDDataset(Dataset):
    """
    CWD Dataset for semantic segmentation with 3 classes (background=0, class1=1, class2=2)
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        img_size: Tuple[int, int] = (512, 512)
    ):
        """
        Args:
            image_dir: Path to directory containing .jpg images
            mask_dir: Path to directory containing .png masks
            transform: Albumentations transforms
            img_size: Target image size (H, W)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No .jpg images found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
    # Load RGB image
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get corresponding mask
        img_basename = os.path.basename(img_path)
        # Replace .jpg with .png for mask (adjust if your naming is different)
        mask_name = img_basename.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            # Try with original name
            mask_path = os.path.join(self.mask_dir, img_basename.replace('.jpg', '_morphed.png'))
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {img_basename}")
        
        # Load mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask has values 0, 1, 2
        mask = np.clip(mask, 0, 2).astype(np.int64)  # Change to int64 here
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()  # Explicitly cast to long
        else:
            # Default resize and normalize
            image = cv2.resize(image, self.img_size)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
            mask = torch.from_numpy(mask).long()  # Cast to long
        
        return image, mask



def get_train_transforms(img_size: Tuple[int, int] = (512, 512)):
    """Training augmentations"""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(img_size: Tuple[int, int] = (512, 512)):
    """Validation transforms (no augmentation)"""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def create_cwd_dataloaders(
    train_img_dir: str,
    train_mask_dir: str,
    val_img_dir: str,
    val_mask_dir: str,
    test_img_dir: str = None,
    test_mask_dir: str = None,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (512, 512)
) -> dict:
    """
    Create DataLoaders for train, val, and optionally test sets
    
    Returns:
        dict with keys 'train', 'val', and optionally 'test'
    """
    # Create datasets
    train_dataset = CWDDataset(
        train_img_dir,
        train_mask_dir,
        transform=get_train_transforms(img_size),
        img_size=img_size
    )
    
    val_dataset = CWDDataset(
        val_img_dir,
        val_mask_dir,
        transform=get_val_transforms(img_size),
        img_size=img_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Optionally add test loader
    if test_img_dir and test_mask_dir:
        test_dataset = CWDDataset(
            test_img_dir,
            test_mask_dir,
            transform=get_val_transforms(img_size),
            img_size=img_size
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        loaders['test'] = test_loader
    
    return loaders
