"""
Model factory for semantic segmentation architectures
Supports: U-Net, U-Net_SA (Spatial Attention), LightMANet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from MANet import LightMANet

# -------------------------
# Simple, flexible UNet
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 4, base_ch: int = 64, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)
        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ======================== U-Net with Spatial Attention ========================
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module using channel-wise pooling
    Applies attention mechanism to enhance spatial features
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                               padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv1(out))
        
        # Apply attention mask
        return x * attn


class DoubleConvSA(nn.Module):
    """(Conv2D -> ReLU) x 2 block for UNet_SA"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSA(nn.Module):
    """Downsampling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvSA(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpSA(nn.Module):
    """Upsampling with transposed conv then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConvSA(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle input sizes that are not divisible by 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet_SA(nn.Module):
    """
    U-Net architecture with Spatial Attention modules
    Spatial attention is applied after each encoder block and bridge
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        num_classes: Number of output classes (default: 1)
        base_ch: Base number of channels (default: 64)
    """
    def __init__(self, in_channels=3, num_classes=1, base_ch=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder
        self.inc = DoubleConvSA(in_channels, base_ch)
        self.down1 = DownSA(base_ch, base_ch * 2)
        self.down2 = DownSA(base_ch * 2, base_ch * 4)
        self.down3 = DownSA(base_ch * 4, base_ch * 8)
        
        # Bridge
        self.down4 = DownSA(base_ch * 8, base_ch * 16)
        
        # Decoder
        self.up1 = UpSA(base_ch * 16, base_ch * 8)
        self.up2 = UpSA(base_ch * 8, base_ch * 4)
        self.up3 = UpSA(base_ch * 4, base_ch * 2)
        self.up4 = UpSA(base_ch * 2, base_ch)
        
        # Final output layer
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)
        
        # Spatial Attention modules
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa_bridge = SpatialAttention()

    def forward(self, x):
        # Encoder with spatial attention
        x1 = self.inc(x)
        x1 = self.sa1(x1)
        
        x2 = self.down1(x1)
        x2 = self.sa2(x2)
        
        x3 = self.down2(x2)
        x3 = self.sa3(x3)
        
        x4 = self.down3(x3)
        x4 = self.sa4(x4)
        
        # Bridge with spatial attention
        x5 = self.down4(x4)
        x5 = self.sa_bridge(x5)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final output
        logits = self.outc(x)
        return logits

## DEEPLABV3+ and PSPNet just Module added as it is, not integrated ##
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()

        # Different dilation rates
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv_out(x)
        return x


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ - Excellent for crop-weed segmentation"""
    def __init__(self, in_channels=3, out_channels=3):
        super(DeepLabV3Plus, self).__init__()

        # Encoder
        self.enc1 = self._make_layer(in_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)

        # ASPP Module
        self.aspp = ASPP(512, 256)

        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(128, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.final = nn.Conv2d(256, out_channels, 1)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # ASPP
        aspp_out = self.aspp(F.max_pool2d(enc4, 2))
        aspp_out = F.interpolate(aspp_out, size=enc2.shape[2:], mode='bilinear', align_corners=False)

        # Low-level features
        low_level = self.low_level_conv(enc2)

        # Concatenate
        decoder_in = torch.cat([aspp_out, low_level], dim=1)

        # Decoder
        out = self.decoder(decoder_in)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
        out = self.final(out)

        return out


# ==============================================================================
# 2. PSPNet (Pyramid Scene Parsing Network)
# ==============================================================================
class PyramidPooling(nn.Module):
    """Pyramid Pooling Module"""
    def __init__(self, in_channels, out_channels=512):
        super(PyramidPooling, self).__init__()

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        feat1 = F.interpolate(self.conv1(self.pool1(x)), size=size, mode='bilinear', align_corners=False)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), size=size, mode='bilinear', align_corners=False)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), size=size, mode='bilinear', align_corners=False)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), size=size, mode='bilinear', align_corners=False)

        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class PSPNet(nn.Module):
    """PSPNet - Great for multi-scale feature extraction"""
    def __init__(self, in_channels=3, out_channels=3):
        super(PSPNet, self).__init__()

        # Encoder
        self.enc1 = self._make_layer(in_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)

        # Pyramid Pooling
        self.ppm = PyramidPooling(512, 512)

        # Final classifier
        self.final = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, out_channels, 1)
        )

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Pyramid pooling
        ppm_out = self.ppm(enc4)

        # Final
        out = self.final(ppm_out)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)

        return out
#########################################Above to be integrated############################################


# ======================== LightMANet (Import from MANet.py) ========================
# try:
#     from scripts.MANet import LightMANet
#     LIGHTMANET_AVAILABLE = True
# except ImportError:
#     print("[WARNING] MANNet.py not found. LightMANet will not be available.")
#     LIGHTMANET_AVAILABLE = False
    
    # Dummy LightMANet for type checking
    class LightMANet(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("MANet.py not found. Please add MANet.py to your project.")


# ======================== Model Factory ========================
def create_model(
    architecture: str,
    in_channels: int = 3,
    num_classes: int = 1,
    base_ch: int = 64,
    **kwargs
) -> nn.Module:
    """
    Factory function to create segmentation models
    
    Args:
        architecture: Model name ('unet', 'lightmanet')
        in_channels: Number of input channels (3 for RGB, 4 for RGB+NIR)
        num_classes: Number of output classes
        base_ch: Base number of channels for the model
        **kwargs: Additional model-specific arguments
    
    Returns:
        Initialized PyTorch model
    
    Raises:
        ValueError: If architecture is not supported
    """
    architecture = architecture.lower()
    
    if architecture == "unet":
        print(f"[INFO] Creating U-Net with {in_channels} input channels, {base_ch} base channels")
        return UNet(
            in_channels=in_channels,
            out_channels=num_classes,
            base_ch=base_ch,
            # bilinear=kwargs.get('bilinear', True)
        )
    
    elif architecture == "unet_sa":
        print(f"[INFO] Creating U-Net with Spatial Attention with {in_channels} input channels, {base_ch} base channels, {num_classes} classes")
        return UNet_SA(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_ch
        )

    elif architecture == "lightmanet":
        # if not LIGHTMANET_AVAILABLE:
        #     raise ImportError("LightMANet requires MANNet.py. Please add it to your project.")
        
        print(f"[INFO] Creating LightMANet with {in_channels} input channels, {base_ch} base channels, {num_classes} classes")
        return LightMANet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_ch
        )
    
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported architectures: ['unet', 'lightmanet']"
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_not_trainable(model: nn.Module) -> int:
    """Count non trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def get_model_info(model: nn.Module) -> dict:
    """Get detailed model information"""
    total_params = count_parameters(model)
    total_params_non_trainable = count_parameters_not_trainable(model)
    return {
        "total_parameters": total_params,
        "total_parameters_million": total_params / 1e6,
        "architecture": model.__class__.__name__,
        "total_non_trainable_parameters": total_params_non_trainable,
        "total_non_trainable_parameters_million": total_params_non_trainable / 1e6
    }

# Below is for sanity check of number of parameters without running train script. 

# def main():
#     # Model configuration
#     architecture = "lightmanet"
#     in_channels = 4       # change to 4 if RGB+NIR
#     num_classes = 1
#     base_ch = 32

#     # Create model
#     model = create_model(
#         architecture=architecture,
#         in_channels=in_channels,
#         num_classes=num_classes,
#         base_ch=base_ch
#     )

#     # Get model info
#     info = get_model_info(model)

#     # Print results
#     print("\n================ Model Summary ================")
#     print(f"Architecture: {info['architecture']}")
#     print(f"Input Channels: {in_channels}")
#     print(f"Output Classes: {num_classes}")
#     print(f"Base Channels: {base_ch}")
#     print("----------------------------------------------")
#     print(f"Total Parameters: {info['total_parameters']:,}")
#     print(f"Total Parameters (M): {info['total_parameters_million']:.3f} M")
#     print(f"Non-Trainable Parameters: {info['total_non_trainable_parameters']:,}")
#     print(f"Non-Trainable Parameters (M): {info['total_non_trainable_parameters_million']:.3f} M")
#     print("==============================================\n")


# if __name__ == "__main__":
#     main()
