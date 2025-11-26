"""
Model factory for semantic segmentation architectures
Supports: U-Net, U-Net_SA (Spatial Attention), LightMANet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class MemoryEfficientKernelAttention(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=4):
        super().__init__()
        self.reduction = reduction
        self.in_ch = in_ch
        self.out_ch = out_ch
        # Reduced channel attention to save memory
        self.key_conv = nn.Conv2d(in_ch, out_ch // reduction, 1)
        self.query_conv = nn.Conv2d(in_ch, out_ch // reduction, 1)  
        self.value_conv = nn.Conv2d(in_ch, out_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Reduce spatial size for attention computation (saves H*W memory)
        H_small, W_small = max(H // self.reduction, 1), max(W // self.reduction, 1)
        
        # Downsample input for key/query computation
        x_small = F.adaptive_avg_pool2d(x, (H_small, W_small))
        
        # Compute key/query on smaller spatial resolution
        key = self.key_conv(x_small).view(B, self.out_ch // self.reduction, -1)  # (B, C_red, H_s*W_s)
        query = self.query_conv(x_small).view(B, self.out_ch // self.reduction, -1).transpose(1, 2)  # (B, H_s*W_s, C_red)
        
        # Compute value on original resolution
        value = self.value_conv(x).view(B, self.out_ch, -1)  # (B, C_out, H*W)
        
        # Attention matrix: (B, H_s*W_s, H_s*W_s) - much smaller than original
        attention = torch.bmm(query, key) / (H_small * W_small) ** 0.5
        attention = F.softmax(attention, dim=-1)  # (B, H_s*W_s, H_s*W_s)
        
        # Apply attention to downsampled features, then upsample result
        value_small = F.adaptive_avg_pool2d(x, (H_small, W_small))
        value_small = self.value_conv(value_small).view(B, self.out_ch, -1)  # (B, C_out, H_s*W_s)
        
        out_small = torch.bmm(value_small, attention.transpose(1, 2))  # (B, C_out, H_s*W_s)
        out_small = out_small.view(B, self.out_ch, H_small, W_small)
        
        # Upsample back to original spatial size
        out = F.interpolate(out_small, size=(H, W), mode='bilinear', align_corners=False)
        
        return self.gamma * out + x

class SimpleChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, max(in_ch // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_ch // reduction, 1), in_ch, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(self.fc(self.avg_pool(x)))
        return x * att

class LightAttentionBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # Only use channel attention to save memory - skip spatial attention
        self.ca = SimpleChannelAttention(in_ch)
        
    def forward(self, x):
        return self.ca(x)

class LightMANet(nn.Module):
    """Memory-efficient MANet for 4-channel RGB+NIR input"""
    def __init__(self, in_channels=4, num_classes=2, base_ch=16):  # Reduced base_ch
        super().__init__()
        # Lightweight encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Reduced encoder blocks
        self.layer1 = self._make_layer(base_ch, base_ch, 1)      # Only 1 block per layer
        self.layer2 = self._make_layer(base_ch, base_ch*2, 1, stride=2)
        self.layer3 = self._make_layer(base_ch*2, base_ch*4, 1, stride=2)
        self.layer4 = self._make_layer(base_ch*4, base_ch*8, 1, stride=2)
        
        # Simple decoder without skip connections to save memory
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_ch, base_ch, 4, stride=4, padding=0),  # Back to input size
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Conv2d(base_ch, num_classes, 1)
        
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            LightAttentionBlock(out_ch)  # Only channel attention
        ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Store input size for final resize
        input_size = x.shape[-2:]  # (H, W)
        
        # Encoder path
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Final convolution
        logits = self.final(x)
        
        # *** FIX: Resize output to match input size exactly ***
        logits = F.interpolate(
            logits, 
            size=input_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        return logits

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
