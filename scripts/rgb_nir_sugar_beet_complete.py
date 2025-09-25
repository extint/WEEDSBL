import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional, Dict
import math

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted for RGB-NIR processing"""
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class RGBNIRConvNeXtEncoder(nn.Module):
    """ConvNeXt Encoder adapted for 4-channel RGB-NIR input"""
    def __init__(self, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], 
                 drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()

        # Stem - Modified for 4-channel RGB-NIR input
        self.stem = nn.Sequential(
            nn.Conv2d(4, dims[0], kernel_size=7, stride=2, padding=3),  # 4 channels for RGB+NIR
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.stages = nn.ModuleList()

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Downsampling layers between stages
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning intermediate features for skip connections"""
        features = []

        # Stem: 1296×966×4 -> 648×483×192
        x = self.stem(x)

        # Stage 1: 648×483×192
        x = self.stages[0](x)
        features.append(x)
        x = self.downsample_layers[0](x)  # -> 324×242×384

        # Stage 2: 324×242×384
        x = self.stages[1](x)
        features.append(x)
        x = self.downsample_layers[1](x)  # -> 162×121×768

        # Stage 3: 162×121×768
        x = self.stages[2](x)
        features.append(x)
        x = self.downsample_layers[2](x)  # -> 81×61×1536

        # Stage 4: 81×61×1536
        x = self.stages[3](x)
        features.append(x)

        return features

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling for multiscale context"""
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        dilations = [1, 6, 12, 18]

        self.convs = nn.ModuleList()
        for dilation in dilations:
            if dilation == 1:
                conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            else:
                conv = nn.Conv2d(in_channels, out_channels, 3, 
                               padding=dilation, dilation=dilation, bias=False)
            self.convs.append(nn.Sequential(
                conv,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        # Apply dilated convolutions
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))

        # Global pooling branch
        global_out = self.global_pool(x)
        global_out = F.interpolate(global_out, size=(h, w), 
                                 mode='bilinear', align_corners=False)
        conv_outs.append(global_out)

        # Concatenate all branches
        out = torch.cat(conv_outs, dim=1)
        return self.project(out)

class DenseAttentionModule(nn.Module):
    """Dense Attention Module for feature refinement with gating"""
    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = max(in_channels // 4, 64)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Dense feature extraction
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Attention weighting
        attention_map = self.attention(out)
        out = out * attention_map

        # Residual connection
        out = self.relu(out + residual)
        return out

class RGBNIRDecoder(nn.Module):
    """Decoder with skip connections and attention mechanisms"""
    def __init__(self, encoder_dims: List[int] = [192, 384, 768, 1536], 
                 num_classes: int = 2):
        super().__init__()

        # ASPP module on deepest features
        self.aspp = ASPPModule(encoder_dims[-1], 256)

        # Gating functions for skip connections (suppress low-level features)
        self.gate3 = nn.Conv2d(encoder_dims[2], 256, 1)  # 768 -> 256
        self.gate2 = nn.Conv2d(encoder_dims[1], 128, 1)  # 384 -> 128
        self.gate1 = nn.Conv2d(encoder_dims[0], 64, 1)   # 192 -> 64

        # Dense attention modules
        self.dam3 = DenseAttentionModule(256 + 256)  # ASPP + gated skip
        self.dam2 = DenseAttentionModule(256 + 128)  # Upsampled + gated skip
        self.dam1 = DenseAttentionModule(128 + 64)   # Upsampled + gated skip

        # Upsampling and refinement layers
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, encoder_features: List[torch.Tensor], 
                target_size: Tuple[int, int] = (966, 1296)) -> torch.Tensor:
        f1, f2, f3, f4 = encoder_features

        # Stage 1: ASPP on deepest features (81×61×1536 -> 81×61×256)
        x = self.aspp(f4)

        # Stage 2: Skip connection with stage 3 features (162×121×768)
        f3_gated = self.gate3(f3)  # 162×121×256
        x = F.interpolate(x, size=f3_gated.shape[-2:], 
                         mode='bilinear', align_corners=False)
        x = torch.cat([x, f3_gated], dim=1)  # 162×121×512
        x = self.dam3(x)  # Dense attention refinement

        # Stage 3: Skip connection with stage 2 features (324×242×384)
        x = self.up3(x)  # 324×242×256
        f2_gated = self.gate2(f2)  # 324×242×128
        x = torch.cat([x, f2_gated], dim=1)  # 324×242×384
        x = self.dam2(x)

        # Stage 4: Skip connection with stage 1 features (648×483×192)
        x = self.up2(x)  # 648×483×128
        f1_gated = self.gate1(f1)  # 648×483×64
        x = torch.cat([x, f1_gated], dim=1)  # 648×483×192
        x = self.dam1(x)

        # Final upsampling and classification
        x = self.up1(x)  # 1296×966×64

        # Ensure exact target size
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, 
                             mode='bilinear', align_corners=False)

        return self.classifier(x)  # 1296×966×2

class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for domain classification"""
    def __init__(self, input_channels: int, ndf: int = 64):
        super().__init__()

        def discriminator_block(in_ch: int, out_ch: int, normalize: bool = True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(input_channels, ndf, normalize=False),  # 64
            discriminator_block(ndf, ndf * 2),                         # 128
            discriminator_block(ndf * 2, ndf * 4),                     # 256
            discriminator_block(ndf * 4, ndf * 8),                     # 512
            nn.Conv2d(ndf * 8, 1, 4, padding=1)                       # 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class RGBNIRSugarBeetSegmentationNetwork(nn.Module):
    """Complete RGB-NIR segmentation network for Sugar Beet 2016 with domain adaptation"""
    def __init__(self, num_classes: int = 2, encoder_dims=[192, 384, 768, 1536]):
        super().__init__()

        # Main segmentation network
        self.encoder = RGBNIRConvNeXtEncoder(dims=encoder_dims)
        self.decoder = RGBNIRDecoder(encoder_dims=encoder_dims, num_classes=num_classes)

        # Discriminators for domain adaptation
        self.discriminator_encoder = PatchGANDiscriminator(encoder_dims[2])  # 768 channels
        self.discriminator_decoder = PatchGANDiscriminator(num_classes)      # 2 channels

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: RGB-NIR input (B, 4, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing segmentation output and features for discriminators
        """
        # Forward through encoder
        encoder_features = self.encoder(x)

        # Forward through decoder
        segmentation = self.decoder(encoder_features, target_size=x.shape[-2:])

        results = {
            'segmentation': segmentation,
            'encoder_features': encoder_features[2],  # Stage 3 features for discriminator
            'decoder_features': segmentation          # Final output for discriminator
        }

        if return_features:
            results['all_encoder_features'] = encoder_features

        return results

def initialize_rgb_nir_weights(model: RGBNIRSugarBeetSegmentationNetwork, 
                              pretrained_rgb_path: Optional[str] = None):
    """Initialize RGB-NIR model weights from RGB pretrained model"""
    if pretrained_rgb_path:
        try:
            # Load RGB pretrained weights
            rgb_state = torch.load(pretrained_rgb_path, map_location='cpu')

            # Handle different checkpoint formats
            if 'model' in rgb_state:
                rgb_state = rgb_state['model']
            elif 'state_dict' in rgb_state:
                rgb_state = rgb_state['state_dict']

            # Initialize first conv layer for RGB-NIR
            stem_key = 'stem.0.weight'  # ConvNeXt stem convolution
            if stem_key in rgb_state:
                rgb_conv_weight = rgb_state[stem_key]  # Shape: [192, 3, 7, 7]

                # Method 1: Average RGB weights for NIR channel initialization
                nir_weight = rgb_conv_weight.mean(dim=1, keepdim=True)  # [192, 1, 7, 7]

                # Concatenate RGB and NIR weights
                new_conv_weight = torch.cat([rgb_conv_weight, nir_weight], dim=1)
                rgb_state[stem_key] = new_conv_weight

                print(f"Initialized NIR channel weights by averaging RGB channels")
                print(f"New conv weight shape: {new_conv_weight.shape}")

            # Load modified weights (strict=False to ignore missing keys)
            missing_keys, unexpected_keys = model.load_state_dict(rgb_state, strict=False)

            print(f"Loaded pretrained RGB weights from {pretrained_rgb_path}")
            if missing_keys:
                print(f"Missing keys (newly initialized): {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys (ignored): {len(unexpected_keys)}")

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with random initialization")
            _initialize_weights(model)
    else:
        print("No pretrained weights provided, using random initialization")
        _initialize_weights(model)

def _initialize_weights(model):
    """Random weight initialization"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class RGBNIRPreprocessor:
    """Data preprocessing pipeline for RGB-NIR Sugar Beet images"""
    def __init__(self, target_size: Tuple[int, int] = (966, 1296)):
        # Sugar Beet 2016 dataset statistics (compute these from your actual dataset)
        self.rgb_mean = [0.485, 0.456, 0.406]  # Standard ImageNet means
        self.rgb_std = [0.229, 0.224, 0.225]   # Standard ImageNet stds

        # NIR channel statistics (these should be computed from Sugar Beet 2016 NIR data)
        self.nir_mean = [0.5]    # Placeholder - compute from actual NIR data
        self.nir_std = [0.25]    # Placeholder - compute from actual NIR data

        self.target_size = target_size

        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

    def normalize_rgb_nir(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize 4-channel RGB-NIR tensor"""
        if tensor.shape[0] != 4:
            raise ValueError(f"Expected 4-channel input, got {tensor.shape[0]} channels")

        # Separate RGB and NIR channels
        rgb_tensor = tensor[:3]  # First 3 channels
        nir_tensor = tensor[3:4] # Last channel

        # Normalize separately
        rgb_normalized = transforms.Normalize(self.rgb_mean, self.rgb_std)(rgb_tensor)
        nir_normalized = transforms.Normalize(self.nir_mean, self.nir_std)(nir_tensor)

        # Concatenate back
        return torch.cat([rgb_normalized, nir_normalized], dim=0)

    def __call__(self, rgb_nir_image: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing pipeline"""
        # Resize if needed
        if rgb_nir_image.shape[-2:] != self.target_size:
            rgb_nir_image = F.interpolate(
                rgb_nir_image.unsqueeze(0), 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)

        # Normalize
        return self.normalize_rgb_nir(rgb_nir_image)

# Example usage and testing
if __name__ == "__main__":
    print("RGB-NIR Sugar Beet Architecture Implementation")
    print("=" * 50)

    # Initialize model
    model = RGBNIRSugarBeetSegmentationNetwork(num_classes=2)

    # Initialize weights (uncomment if you have pretrained RGB weights)
    # initialize_rgb_nir_weights(model, "path/to/convnext_pretrained.pth")

    # Initialize data preprocessor
    preprocessor = RGBNIRPreprocessor(target_size=(966, 1296))

    # Test forward pass
    batch_size = 2
    rgb_nir_input = torch.randn(batch_size, 4, 966, 1296)  # RGB-NIR input

    print(f"Input shape: {rgb_nir_input.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(rgb_nir_input)

        segmentation = outputs['segmentation']
        encoder_features = outputs['encoder_features'] 
        decoder_features = outputs['decoder_features']

        print(f"Segmentation output: {segmentation.shape}")
        print(f"Encoder features (Stage 3): {encoder_features.shape}")
        print(f"Decoder features: {decoder_features.shape}")

        # Test discriminators
        enc_disc_out = model.discriminator_encoder(encoder_features)
        dec_disc_out = model.discriminator_decoder(decoder_features)

        print(f"Encoder discriminator output: {enc_disc_out.shape}")
        print(f"Decoder discriminator output: {dec_disc_out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    print("\n✓ Architecture implementation completed successfully!")
