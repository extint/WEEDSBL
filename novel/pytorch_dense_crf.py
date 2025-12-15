# pytorch_dense_crf.py
"""
Pure PyTorch implementation of Dense CRF for semantic segmentation.
No external dependencies (pydensecrf-free!)

Based on:
- Krähenbühl & Koltun (2011): "Efficient Inference in Fully Connected CRFs"
"""

import torch
import torch.nn.functional as F
import numpy as np


class DenseCRFPyTorch:
    """
    Pure PyTorch Dense CRF implementation for post-processing segmentation.
    
    This implementation uses mean-field approximation for inference.
    """
    
    def __init__(
        self,
        num_classes=3,
        max_iterations=10,
        pos_w=3.0,
        pos_xy_std=3.0,
        bi_w=4.0,
        bi_xy_std=67.0,
        bi_rgb_std=3.0,
        device='cpu'
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            max_iterations: Number of mean-field iterations
            pos_w: Weight for spatial (position) kernel
            pos_xy_std: Spatial standard deviation for position kernel
            bi_w: Weight for bilateral (appearance) kernel
            bi_xy_std: Spatial standard deviation for bilateral kernel
            bi_rgb_std: Color standard deviation for bilateral kernel
            device: 'cpu' or 'cuda'
        """
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.device = device
    
    def _bilateral_filter(self, Q, image, spatial_std, color_std):
        """
        Apply bilateral filtering on Q using image features.
        
        Args:
            Q: (C, H, W) - current Q distribution
            image: (H, W, 3) - RGB image in [0, 255]
            spatial_std: Spatial standard deviation
            color_std: Color standard deviation
        
        Returns:
            Filtered Q: (C, H, W)
        """
        C, H, W = Q.shape
        
        # Normalize image to [0, 1]
        image_norm = torch.from_numpy(image).float().to(self.device) / 255.0
        
        # Create coordinate grid
        y_coords = torch.arange(H, device=self.device).float().unsqueeze(1).repeat(1, W)
        x_coords = torch.arange(W, device=self.device).float().unsqueeze(0).repeat(H, 1)
        
        # Normalize spatial coordinates
        y_coords = y_coords / spatial_std
        x_coords = x_coords / spatial_std
        
        # Normalize color features
        image_norm = image_norm / color_std
        
        # Flatten everything
        coords = torch.stack([y_coords, x_coords], dim=0).view(2, -1)  # (2, H*W)
        colors = image_norm.permute(2, 0, 1).reshape(3, -1)  # (3, H*W)
        Q_flat = Q.view(C, -1)  # (C, H*W)
        
        # Combine spatial and color features
        features = torch.cat([coords, colors], dim=0)  # (5, H*W)
        
        # Compute pairwise distances (this is memory intensive for large images)
        # For efficiency, we'll use a sliding window approach
        filtered_Q = torch.zeros_like(Q_flat)
        
        # Use a sliding window to reduce memory
        window_size = min(100, H * W)  # Process in chunks
        
        for i in range(0, H * W, window_size):
            end_i = min(i + window_size, H * W)
            
            # Compute distances for this chunk
            feat_i = features[:, i:end_i].unsqueeze(2)  # (5, chunk_size, 1)
            feat_all = features.unsqueeze(1)  # (5, 1, H*W)
            
            # Euclidean distance
            dist = torch.sum((feat_i - feat_all) ** 2, dim=0)  # (chunk_size, H*W)
            
            # Gaussian kernel
            weights = torch.exp(-0.5 * dist)  # (chunk_size, H*W)
            
            # Normalize weights
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Apply weighted sum
            filtered_Q[:, i:end_i] = torch.mm(Q_flat, weights.t())
        
        return filtered_Q.view(C, H, W)
    
    def _spatial_filter(self, Q, spatial_std):
        """
        Apply spatial (Gaussian) filtering on Q.
        
        Args:
            Q: (C, H, W) - current Q distribution
            spatial_std: Spatial standard deviation
        
        Returns:
            Filtered Q: (C, H, W)
        """
        # Simple Gaussian blur using conv2d
        kernel_size = int(6 * spatial_std + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        padding = kernel_size // 2
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size, device=self.device).float() - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / spatial_std) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        # Apply convolution
        kernel = gauss_2d.unsqueeze(0).unsqueeze(0).repeat(self.num_classes, 1, 1, 1)
        Q_unsqueezed = Q.unsqueeze(0)  # (1, C, H, W)
        
        filtered = F.conv2d(Q_unsqueezed, kernel, padding=padding, groups=self.num_classes)
        
        return filtered.squeeze(0)
    
    def __call__(self, image, probabilities):
        """
        Apply Dense CRF inference.
        
        Args:
            image: (H, W, 3) numpy array in [0, 255], uint8
            probabilities: (C, H, W) numpy array with class probabilities
        
        Returns:
            Refined segmentation: (H, W) numpy array with class indices
        """
        C, H, W = probabilities.shape
        
        # Convert to torch tensors
        Q = torch.from_numpy(probabilities).float().to(self.device)
        
        # Initialize with unary potentials
        Q = torch.log(Q + 1e-8)  # Log probabilities
        
        # Mean-field inference
        for iteration in range(self.max_iterations):
            # Store old Q
            Q_old = Q.clone()
            
            # Message passing
            # 1. Spatial smoothness (Gaussian kernel)
            spatial_msg = self._spatial_filter(torch.exp(Q), self.pos_xy_std)
            
            # 2. Appearance kernel (bilateral filter)
            # For efficiency, we'll use a simpler approximation
            bilateral_msg = self._bilateral_filter(
                torch.exp(Q), image, self.bi_xy_std, self.bi_rgb_std
            )
            
            # Combine messages
            Q = Q_old - self.pos_w * torch.log(spatial_msg + 1e-8) \
                      - self.bi_w * torch.log(bilateral_msg + 1e-8)
            
            # Normalize
            Q = Q - torch.logsumexp(Q, dim=0, keepdim=True)
        
        # Get final segmentation
        refined_map = torch.argmax(Q, dim=0).cpu().numpy()
        
        return refined_map


class SimpleCRF:
    """
    Simplified CRF that's faster and doesn't require bilateral filtering.
    Uses only spatial smoothness (Gaussian kernel).
    """
    
    def __init__(
        self,
        num_classes=3,
        max_iterations=5,
        spatial_weight=3.0,
        spatial_std=3.0,
        device='cpu'
    ):
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        self.spatial_weight = spatial_weight
        self.spatial_std = spatial_std
        self.device = device
    
    def _gaussian_filter(self, Q, sigma):
        """Apply Gaussian filtering"""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        padding = kernel_size // 2
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size, device=self.device).float() - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        # Apply convolution
        kernel = gauss_2d.unsqueeze(0).unsqueeze(0).repeat(self.num_classes, 1, 1, 1)
        Q_unsqueezed = Q.unsqueeze(0)
        
        filtered = F.conv2d(Q_unsqueezed, kernel, padding=padding, groups=self.num_classes)
        
        return filtered.squeeze(0)
    
    def __call__(self, probabilities):
        """
        Apply simplified CRF (spatial smoothness only).
        
        Args:
            probabilities: (C, H, W) numpy array with class probabilities
        
        Returns:
            Refined segmentation: (H, W) numpy array with class indices
        """
        # Convert to torch
        Q = torch.from_numpy(probabilities).float().to(self.device)
        
        # Mean-field inference
        for _ in range(self.max_iterations):
            Q_exp = torch.exp(Q)
            
            # Apply spatial smoothness
            Q_smooth = self._gaussian_filter(Q_exp, self.spatial_std)
            
            # Update Q
            Q = torch.log(Q_exp / (Q_smooth + 1e-8) + 1e-8)
            
            # Normalize
            Q = Q - torch.logsumexp(Q, dim=0, keepdim=True)
        
        # Get final segmentation
        refined_map = torch.argmax(Q, dim=0).cpu().numpy()
        
        return refined_map


# Convenience functions
def apply_dense_crf(image, probabilities, use_bilateral=True, device='cpu'):
    """
    Apply Dense CRF to refine segmentation.
    
    Args:
        image: (H, W, 3) numpy array in [0, 255], uint8
        probabilities: (C, H, W) numpy array with class probabilities
        use_bilateral: If True, use full bilateral CRF; else use simple spatial CRF
        device: 'cpu' or 'cuda'
    
    Returns:
        Refined segmentation: (H, W) numpy array with class indices
    """
    if use_bilateral and image is not None:
        crf = DenseCRFPyTorch(device=device)
        return crf(image, probabilities)
    else:
        crf = SimpleCRF(device=device)
        return crf(probabilities)


# Example usage
# if __name__ == "__main__":
#     # Create dummy data
#     H, W = 256, 256
#     num_classes = 3
    
#     # Random probabilities
#     probs = np.random.rand(num_classes, H, W).astype(np.float32)
#     probs = probs / probs.sum(axis=0, keepdims=True)
    
#     # Random image
#     image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
#     # Apply CRF
#     print("Testing SimpleCRF (fast)...")
#     crf_simple = SimpleCRF(device='cpu')
#     refined_simple = crf_simple(probs)
#     print(f"Output shape: {refined_simple.shape}")
#     print(f"Unique classes: {np.unique(refined_simple)}")
    
#     print("\nTesting DenseCRFPyTorch (with bilateral)...")
#     crf_full = DenseCRFPyTorch(device='cpu')
#     refined_full = crf_full(image, probs)
#     print(f"Output shape: {refined_full.shape}")
#     print(f"Unique classes: {np.unique(refined_full)}")