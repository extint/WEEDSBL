import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DoubleConv, Down  # reuse your existing UNet blocks
# If these imports fail, adjust the path/names accordingly.

class UNetEncoder(nn.Module):
    """
    UNet-style encoder that returns multi-scale feature maps.
    Reuses DoubleConv + Down blocks from existing UNet.
    """
    def __init__(self, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        self.inc   = DoubleConv(in_channels, base_ch)        # stride 1
        self.down1 = Down(base_ch, base_ch * 2)              # stride 2
        self.down2 = Down(base_ch * 2, base_ch * 4)          # stride 4
        self.down3 = Down(base_ch * 4, base_ch * 8)          # stride 8
        self.down4 = Down(base_ch * 8, base_ch * 16)         # stride 16

    def forward(self, x):
        x1 = self.inc(x)     # B, C,  H,   W
        x2 = self.down1(x1)  # B, 2C, H/2, W/2
        x3 = self.down2(x2)  # B, 4C, H/4, W/4
        x4 = self.down3(x3)  # B, 8C, H/8, W/8
        x5 = self.down4(x4)  # B,16C, H/16,W/16

        # Return list for easy fusion/decoding
        return [x1, x2, x3, x4, x5]


class RGBEncoder(nn.Module):
    """
    RGB encoder branch. For now, reuse UNet-style encoder.
    Can later be swapped for LightMANet encoder if desired.
    """
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.encoder = UNetEncoder(in_channels=3, base_ch=base_ch)

    def forward(self, x_rgb):
        return self.encoder(x_rgb)  # [R1..R5]


class NIREncoder(nn.Module):
    """
    NIR encoder branch – same structure, but single-channel input
    and smaller width to save parameters.
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.encoder = UNetEncoder(in_channels=1, base_ch=base_ch)

    def forward(self, x_nir):
        return self.encoder(x_nir)  # [N1..N5]
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class AFFModule(nn.Module):
    """
    Attention-based Fusion Feature (AFF) module.
    Takes aligned RGB and NIR features at a given stage and outputs fused features.

    Inputs:
        rgb_feat:  (B, C, H, W)
        nir_feat:  (B, C, H, W)  # already projected to same C, H, W
    Output:
        fused:     (B, C, H, W)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Local attention: 1x1 conv -> BN -> ReLU -> 1x1 conv -> BN
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Global attention: GAP -> 1x1 conv -> LN -> ReLU -> 1x1 conv -> LN
        self.global_conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.global_ln1   = nn.LayerNorm(channels)
        self.global_conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.global_ln2   = nn.LayerNorm(channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_feat, nir_feat):
        # rgb_feat, nir_feat: (B, C, H, W)
        # 1) Shared representation
        x = rgb_feat + nir_feat  # (B, C, H, W)

        # 2) Local branch
        L = self.local(x)  # (B, C, H, W)

        # 3) Global branch: GAP over H,W then conv + LN in (B,C) space
        # GAP -> (B, C, 1, 1)
        g = F.adaptive_avg_pool2d(x, output_size=1)
        g = self.global_conv1(g)  # (B, C, 1, 1)

        # LayerNorm over C: reshape to (B, C)
        B, C, _, _ = g.shape
        g = g.view(B, C)
        g = self.global_ln1(g)
        g = F.relu(g, inplace=True)

        g = g.view(B, C, 1, 1)
        g = self.global_conv2(g)
        g = g.view(B, C)
        g = self.global_ln2(g)
        g = g.view(B, C, 1, 1)  # broadcast later to H,W

        # 4) Gate: Ws = sigmoid(Ls + Gs)
        # broadcast g to H,W automatically by addition
        W = self.sigmoid(L + g)  # (B, C, H, W)

        # 5) Fused: F = 2 * W * RGB + 2 * (1 - W) * NIR
        fused = 2.0 * W * rgb_feat + 2.0 * (1.0 - W) * nir_feat

        return fused

class StageProjector(nn.Module):
    """
    Projects RGB and NIR features at a given stage to a common channel width.
    """
    def __init__(self, in_ch_rgb: int, in_ch_nir: int, out_ch: int):
        super().__init__()
        self.proj_rgb = nn.Conv2d(in_ch_rgb, out_ch, kernel_size=1, bias=False)
        self.proj_nir = nn.Conv2d(in_ch_nir, out_ch, kernel_size=1, bias=False)

    def forward(self, feat_rgb, feat_nir):
        rgb = self.proj_rgb(feat_rgb)
        nir = self.proj_nir(feat_nir)
        return rgb, nir

class SimpleDecoder(nn.Module):
    """
    Lightweight decoder:
      - per-scale 1x1 projection to embed_dim
      - upsample all to highest resolution (stride-4)
      - concatenate and compress
      - final classifier + upsample to input size
    """
    def __init__(self, in_ch2: int, in_ch3: int, in_ch4: int,
                 embed_dim: int, num_classes: int):
        super().__init__()
        # per-stage projections
        self.proj2 = nn.Conv2d(in_ch2, embed_dim, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(in_ch3, embed_dim, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(in_ch4, embed_dim, kernel_size=1, bias=False)

        # compression after concat
        self.fuse = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1, bias=False)

        # classifier
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, f2, f3, f4, input_size):
        # f2,f3,f4: fused features at stages 2,3,4
        B, _, H2, W2 = f2.shape

        # project
        p2 = self.proj2(f2)       # (B, E, H2, W2)
        p3 = self.proj3(f3)       # (B, E, H3, W3)
        p4 = self.proj4(f4)       # (B, E, H4, W4)

        # upsample to H2,W2
        p3 = F.interpolate(p3, size=(H2, W2), mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=(H2, W2), mode="bilinear", align_corners=False)

        x = torch.cat([p2, p3, p4], dim=1)  # (B, 3E, H2, W2)
        x = self.fuse(x)                    # (B, E, H2, W2)
        logits = self.classifier(x)         # (B, K, H2, W2)

        # upsample to original input size
        logits = F.interpolate(logits, size=input_size, mode="bilinear",
                               align_corners=False)
        return logits

class DualEncoderAFFNet(nn.Module):
    """
    Dual-encoder RGB+NIR network with AFF fusion and lightweight decoder.

    Encoders: UNet-style (can later be swapped for LightMANet encoders).
    Fusion:   AFF at stages 2,3,4.
    Decoder:  Simple multi-scale MLP-like head.
    """
    def __init__(self,
                 rgb_base_ch: int = 64,
                 nir_base_ch: int = 32,
                 num_classes: int = 2,
                 embed_dim: int = 64):
        super().__init__()

        # Encoders
        self.rgb_encoder = RGBEncoder(base_ch=rgb_base_ch)
        self.nir_encoder = NIREncoder(base_ch=nir_base_ch)

        # Channel sizes from encoders
        # RGB: [C, 2C, 4C, 8C, 16C]
        # NIR: [c, 2c, 4c, 8c, 16c]
        C_rgb = rgb_base_ch
        C_nir = nir_base_ch

        # Stage 2: RGB 4C, NIR 4c -> project to 64
        self.proj2 = StageProjector(
            in_ch_rgb=C_rgb * 4,
            in_ch_nir=C_nir * 4,
            out_ch=64
        )
        self.aff2 = AFFModule(channels=64)

        # Stage 3: RGB 8C, NIR 8c -> project to 128
        self.proj3 = StageProjector(
            in_ch_rgb=C_rgb * 8,
            in_ch_nir=C_nir * 8,
            out_ch=128
        )
        self.aff3 = AFFModule(channels=128)

        # Stage 4: RGB 16C, NIR 16c -> project to 256
        self.proj4 = StageProjector(
            in_ch_rgb=C_rgb * 16,
            in_ch_nir=C_nir * 16,
            out_ch=256
        )
        self.aff4 = AFFModule(channels=256)

        # Decoder: takes fused ch [64,128,256]
        self.decoder = SimpleDecoder(
            in_ch2=64,
            in_ch3=128,
            in_ch4=256,
            embed_dim=embed_dim,
            num_classes=num_classes,
        )

    def forward(self, x_rgb, x_nir):
        """
        x_rgb: (B, 3, H, W)
        x_nir: (B, 1, H, W)
        """
        input_size = x_rgb.shape[-2:]

        # 1. Encode
        R_feats = self.rgb_encoder(x_rgb)  # [R1..R5]
        N_feats = self.nir_encoder(x_nir)  # [N1..N5]

        # We use stages 3,4,5 from encoder lists as 2,3,4 for fusion
        # R_feats: [x1, x2, x3, x4, x5] with channels [C,2C,4C,8C,16C]
        R2, R3, R4 = R_feats[2], R_feats[3], R_feats[4]
        N2, N3, N4 = N_feats[2], N_feats[3], N_feats[4]

        # 2. Project + AFF fusion at each stage
        r2, n2 = self.proj2(R2, N2)
        F2 = self.aff2(r2, n2)

        r3, n3 = self.proj3(R3, N3)
        F3 = self.aff3(r3, n3)

        r4, n4 = self.proj4(R4, N4)
        F4 = self.aff4(r4, n4)

        # 3. Decode to full-res logits
        logits = self.decoder(F2, F3, F4, input_size=input_size)
        return logits

