# unet_crf_model.py
"""
U-Net architecture for NDVI-based crop-weed segmentation with CRF post-processing.
Based on the paper: "A Multispectral U-Net Framework for Crop-Weed Semantic Segmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import our pure PyTorch CRF (no pydensecrf needed!)
try:
    from pytorch_dense_crf import SimpleCRF, DenseCRFPyTorch
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("[WARNING] pytorch_dense_crf.py not found. CRF will be disabled.")


# ============================================================================
# U-Net Architecture (Original Ronneberger et al. 2015)
# ============================================================================

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        
        # Use transposed convolution for upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch (input might not be perfectly divisible)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    Args:
        in_channels: Number of input channels (3 for NDVI + Filtered-NIR + Green)
        num_classes: Number of output classes (3 for Background, Crop, Weed)
        base_filters: Number of filters in first layer (doubles each layer)
        dropout: Dropout probability
    """
    def __init__(self, in_channels=3, num_classes=3, base_filters=16, dropout=0.05):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_filters, dropout)
        self.down1 = Down(base_filters, base_filters * 2, dropout)
        self.down2 = Down(base_filters * 2, base_filters * 4, dropout)
        self.down3 = Down(base_filters * 4, base_filters * 8, dropout)
        self.down4 = Down(base_filters * 8, base_filters * 16, dropout)
        
        # Decoder
        self.up1 = Up(base_filters * 16, base_filters * 8, dropout)
        self.up2 = Up(base_filters * 8, base_filters * 4, dropout)
        self.up3 = Up(base_filters * 4, base_filters * 2, dropout)
        self.up4 = Up(base_filters * 2, base_filters, dropout)
        
        # Output
        self.outc = nn.Conv2d(base_filters, num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output logits
        logits = self.outc(x)
        
        return logits


# ============================================================================
# CRF Post-Processing (Pure PyTorch - No External Dependencies!)
# ============================================================================

class DenseCRF:
    """
    Wrapper for Pure PyTorch Dense CRF implementation.
    Uses spatial smoothness and optional bilateral filtering.
    """
    def __init__(
        self,
        num_classes=3,
        max_iterations=10,
        pos_w=3,
        pos_xy_std=3,
        bi_w=4,
        bi_xy_std=67,
        bi_rgb_std=3,
        use_bilateral=False,
        device='cpu'
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            max_iterations: Number of CRF iterations
            pos_w: Weight for spatial kernel
            pos_xy_std: Spatial standard deviation for spatial kernel
            bi_w: Weight for bilateral kernel
            bi_xy_std: Spatial standard deviation for bilateral kernel
            bi_rgb_std: Color standard deviation for bilateral kernel
            use_bilateral: Use bilateral filtering (slower) or just spatial (faster)
            device: 'cpu' or 'cuda'
        """
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        self.use_bilateral = use_bilateral
        self.device = device
        
        if not CRF_AVAILABLE:
            print("[WARNING] CRF not available. Install pytorch_dense_crf.py")
            self.crf = None
            return
        
        if use_bilateral:
            self.crf = DenseCRFPyTorch(
                num_classes=num_classes,
                max_iterations=max_iterations,
                pos_w=pos_w,
                pos_xy_std=pos_xy_std,
                bi_w=bi_w,
                bi_xy_std=bi_xy_std,
                bi_rgb_std=bi_rgb_std,
                device=device
            )
        else:
            # Use simplified CRF (faster, spatial only)
            self.crf = SimpleCRF(
                num_classes=num_classes,
                max_iterations=max_iterations,
                spatial_weight=pos_w,
                spatial_std=pos_xy_std,
                device=device
            )
    
    def __call__(self, image, probabilities):
        """
        Apply CRF to refine segmentation.
        
        Args:
            image: Original image (H, W, 3) in [0, 255], uint8 (optional if use_bilateral=False)
            probabilities: Class probabilities (num_classes, H, W), softmax output
        
        Returns:
            Refined segmentation map (H, W) with class indices
        """
        if self.crf is None:
            # Fallback to argmax if CRF not available
            return np.argmax(probabilities, axis=0)
        
        if self.use_bilateral:
            return self.crf(image, probabilities)
        else:
            return self.crf(probabilities)


# ============================================================================
# Combined Model with CRF
# ============================================================================

class UNetWithCRF(nn.Module):
    """
    U-Net with optional CRF post-processing for inference.
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=3,
        base_filters=16,
        dropout=0.05,
        use_crf=True,
        use_bilateral_crf=False,  # New: use full bilateral or just spatial
        crf_params=None
    ):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_filters: Base number of filters in U-Net
            dropout: Dropout rate
            use_crf: Whether to apply CRF during inference
            use_bilateral_crf: Use bilateral CRF (slower) or spatial only (faster)
            crf_params: Dictionary of CRF parameters
        """
        super().__init__()
        
        self.unet = UNet(in_channels, num_classes, base_filters, dropout)
        self.use_crf = use_crf
        self.use_bilateral_crf = use_bilateral_crf
        
        if use_crf:
            if crf_params is None:
                crf_params = {}
            crf_params['use_bilateral'] = use_bilateral_crf
            crf_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.crf = DenseCRF(num_classes=num_classes, **crf_params)
        else:
            self.crf = None
    
    def forward(self, x):
        """Forward pass returns logits (for training)"""
        return self.unet(x)
    
    def predict_with_crf(self, x, original_images):
        """
        Inference with CRF post-processing.
        
        Args:
            x: Normalized input tensor (B, 3, H, W)
            original_images: List of original RGB images (H, W, 3) in [0, 255]
        
        Returns:
            List of refined segmentation maps (H, W)
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.unet(x)
            probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # Convert to numpy
        probs_np = probs.cpu().numpy()
        
        refined_masks = []
        for i in range(probs_np.shape[0]):
            if self.use_crf and self.crf is not None:
                # Apply CRF refinement
                refined = self.crf(original_images[i], probs_np[i])
            else:
                # Just take argmax
                refined = np.argmax(probs_np[i], axis=0)
            
            refined_masks.append(refined)
        
        return refined_masks


# ============================================================================
# Model Creation Functions
# ============================================================================

def create_unet_model(in_channels=3, num_classes=3, base_filters=16, dropout=0.05):
    """Create U-Net model for NDVI-based segmentation"""
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        dropout=dropout
    )


def create_unet_with_crf(
    in_channels=3,
    num_classes=3,
    base_filters=16,
    dropout=0.05,
    use_crf=True,
    use_bilateral_crf=False,  # New parameter
    crf_max_iterations=10,
    crf_pos_w=3,
    crf_pos_xy_std=3,
    crf_bi_w=4,
    crf_bi_xy_std=67,
    crf_bi_rgb_std=3
):
    """
    Create U-Net with CRF post-processing.
    
    Args:
        use_bilateral_crf: If True, use full bilateral CRF (slower, better quality)
                          If False, use spatial-only CRF (faster, good quality)
    """
    crf_params = {
        'max_iterations': crf_max_iterations,
        'pos_w': crf_pos_w,
        'pos_xy_std': crf_pos_xy_std,
        'bi_w': crf_bi_w,
        'bi_xy_std': crf_bi_xy_std,
        'bi_rgb_std': crf_bi_rgb_std,
        'use_bilateral': use_bilateral_crf
    }
    
    return UNetWithCRF(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        dropout=dropout,
        use_crf=use_crf,
        use_bilateral_crf=use_bilateral_crf,
        crf_params=crf_params
    )


def get_model_info(model):
    """Get model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'architecture': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_parameters_M': total_params / 1_000_000,
        'trainable_parameters_M': trainable_params / 1_000_000
    }


# Example usage
# if __name__ == "__main__":
#     # Create model
#     model = create_unet_with_crf(
#         in_channels=3,  # NDVI + Filtered-NIR + Green
#         num_classes=3,  # Background, Crop, Weed
#         base_filters=16,
#         dropout=0.05,
#         use_crf=True
#     )
    
#     info = get_model_info(model)
#     print(f"Model: {info['architecture']}")
#     print(f"Total parameters: {info['total_parameters']:,} ({info['total_parameters_M']:.2f}M)")
#     print(f"Trainable parameters: {info['trainable_parameters']:,} ({info['trainable_parameters_M']:.2f}M)")
    
#     # Test forward pass
#     x = torch.randn(2, 3, 512, 512)
#     logits = model(x)
#     print(f"\nInput shape: {x.shape}")
#     print(f"Output shape: {logits.shape}")