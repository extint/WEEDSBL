"""
Reusable preprocessing utilities for RGB+NIR weedy rice segmentation
Can be used for both training and inference
"""
import numpy as np
import cv2
import torch
from typing import Tuple, Union, Optional


class RGBNIRPreprocessor:
    """Handles preprocessing for RGB+NIR imagery"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (960, 1280),
        rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        rgb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        normalize: bool = True
    ):
        """
        Args:
            target_size: (H, W) target resolution
            rgb_mean: ImageNet mean for RGB channels
            rgb_std: ImageNet std for RGB channels
            normalize: Whether to apply normalization
        """
        self.target_h, self.target_w = target_size
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.normalize = normalize
    
    def read_nir(self, nir_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Read NIR band from path or array
        
        Args:
            nir_input: Path to NIR .TIF or numpy array
            
        Returns:
            Grayscale NIR array
        """
        if isinstance(nir_input, str):
            nir = cv2.imread(nir_input, cv2.IMREAD_UNCHANGED)
            if nir is None:
                raise FileNotFoundError(f"NIR not found: {nir_input}")
        else:
            nir = nir_input
        
        if nir is None:
            raise ValueError("NIR array is None")
        
        # Convert to grayscale if 3-channel
        if nir.ndim == 3:
            nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)
        
        return nir
    
    def scale_to_float(self, arr: np.ndarray) -> np.ndarray:
        """Scale uint8/uint16 to [0, 1] float32"""
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) / 65535.0
        return arr.astype(np.float32) / 255.0
    
    def preprocess(
        self,
        bgr: np.ndarray,
        nir_input: Union[str, np.ndarray],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Complete preprocessing pipeline for RGB+NIR
        
        Args:
            bgr: BGR image from cv2.imread (H, W, 3)
            nir_input: NIR path or array
            return_tensor: If True, return torch.Tensor (1, 4, H, W), else numpy (H, W, 4)
            
        Returns:
            Preprocessed 4-channel tensor or array
        """
        # Resize RGB
        if bgr.shape[:2] != (self.target_h, self.target_w):
            bgr = cv2.resize(bgr, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        
        # Read and resize NIR
        nir = self.read_nir(nir_input)
        if nir.shape[:2] != (self.target_h, self.target_w):
            nir = cv2.resize(nir, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB and scale to [0, 1]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = self.scale_to_float(rgb)
        
        # Normalize RGB channels
        if self.normalize:
            rgb = (rgb - self.rgb_mean) / self.rgb_std
        
        # Scale NIR to [0, 1]
        nir = self.scale_to_float(nir)
        
        # Stack RGB + NIR -> (H, W, 4)
        rgbnir = np.dstack([rgb, nir[..., None]])
        
        if return_tensor:
            # Convert to CHW and add batch dimension
            x = torch.from_numpy(rgbnir.transpose(2, 0, 1)).float().unsqueeze(0)
            return x
        
        return rgbnir
    
    def denormalize_rgb(self, rgb_normalized: np.ndarray) -> np.ndarray:
        """Denormalize RGB for visualization (useful for debugging)"""
        rgb = rgb_normalized * self.rgb_std + self.rgb_mean
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return rgb
