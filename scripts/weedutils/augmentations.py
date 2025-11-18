"""
Agricultural-specific data augmentations without Albumentations
"""
import numpy as np
import cv2
import random
from typing import Tuple


class AgriculturalAugmentation:
    """Augmentation pipeline for agricultural weed detection"""
    
    def __init__(
        self,
        hue_shift: int = 15,
        sat_shift: int = 30,
        val_shift: int = 20,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        noise_std: float = 0.05,
        flip_prob: float = 0.5,
        hsv_prob: float = 0.7,
        brightness_prob: float = 0.5,
        noise_prob: float = 0.3
    ):
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.brightness_prob = brightness_prob
        self.noise_prob = noise_prob
    
    def random_flip(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random horizontal and vertical flips"""
        if random.random() < self.flip_prob:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        
        if random.random() < 0.25:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        
        return img, mask
    
    def hsv_jitter(self, img: np.ndarray) -> np.ndarray:
        """
        Agricultural-specific HSV color jittering
        Only applied to RGB channels (first 3), not NIR
        """
        if random.random() < self.hsv_prob:
            # Extract RGB channels (assume img is float32 [0, 1])
            rgb = img[..., :3].copy()
            
            # Convert to uint8 for HSV conversion
            rgb_uint8 = (np.clip(rgb * 255, 0, 255)).astype(np.uint8)
            hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Random shifts
            h_shift = random.uniform(-self.hue_shift, self.hue_shift)
            s_shift = random.uniform(-self.sat_shift, self.sat_shift)
            v_shift = random.uniform(-self.val_shift, self.val_shift)
            
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + h_shift, 0, 179)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_shift, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v_shift, 0, 255)
            
            rgb_jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            rgb_jittered = rgb_jittered.astype(np.float32) / 255.0
            
            # Replace RGB channels, keep NIR unchanged
            img[..., :3] = rgb_jittered
        
        return img
    
    def brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Random brightness and contrast adjustment (RGB only)"""
        if random.random() < self.brightness_prob:
            alpha = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
            beta = random.uniform(-self.brightness_limit, self.brightness_limit)
            
            # Apply only to RGB channels
            img[..., :3] = np.clip(img[..., :3] * alpha + beta, 0, 1)
        
        return img
    
    def add_noise(self, img: np.ndarray) -> np.ndarray:
        """Add Gaussian noise (all channels)"""
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
        
        return img
    
    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply full augmentation pipeline"""
        img, mask = self.random_flip(img, mask)
        img = self.hsv_jitter(img)
        img = self.brightness_contrast(img)
        img = self.add_noise(img)
        return img, mask
