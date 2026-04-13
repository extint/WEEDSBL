"""
DualEncoderMini — Proposed architecture for paper.

Design philosophy:
  - RGB path : lightweight CNN, 3 stages [16, 32, 64]  — texture + colour
  - NIR path : ultra-light CNN, 3 stages [8,  16, 32]  — reflectance / physiology
  - Fusion   : AFF at stride-8 (single scale, most informative)
  - Neck     : ASPP-style multi-scale context on fused features
  - Decoder  : high-res skip from stride-4 RGB + final upsampling

Target: ~0.6–0.9M params  (fair comparison with DeepLabV3+ at 0.63M)

Paper claim: equal params, modality-aware encoding > naive RGB+NIR stacking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ──────────────────────────────────────────────────────────────────

def _gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with fallback for small channel counts."""
    g = min(groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


def _conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, bias=False),
        _gn(out_ch),
        nn.ReLU(inplace=True),
    )


# ── RGB encoder (lightweight CNN, 3 stages) ───────────────────────────────────

class RGBMiniEncoder(nn.Module):
    """
    3-stage CNN for RGB.
    Returns features at stride-4 (skip) and stride-8 (for fusion).
    Channels: 3 → 16 → 32 → 64
    """
    def __init__(self, base_ch: int = 16):
        super().__init__()
        c = base_ch
        # Stem: stride-2
        self.stem = nn.Sequential(
            _conv_bn_relu(3, c, k=3, s=2, p=1),
            _conv_bn_relu(c, c, k=3, s=1, p=1),
        )
        # Stage 1: stride-4
        self.stage1 = nn.Sequential(
            _conv_bn_relu(c, c * 2, k=3, s=2, p=1),
            _conv_bn_relu(c * 2, c * 2),
        )
        # Stage 2: stride-8
        self.stage2 = nn.Sequential(
            _conv_bn_relu(c * 2, c * 4, k=3, s=2, p=1),
            _conv_bn_relu(c * 4, c * 4),
        )
        self.out_skip = c * 2   # stride-4 channels for skip connection
        self.out_fuse = c * 4   # stride-8 channels for fusion

    def forward(self, x):
        x = self.stem(x)        # stride-2
        s4 = self.stage1(x)     # stride-4  (skip)
        s8 = self.stage2(s4)    # stride-8  (fusion)
        return s4, s8


# ── NIR encoder (ultra-light CNN, 3 stages) ───────────────────────────────────

class NIRMiniEncoder(nn.Module):
    """
    3-stage CNN for single-channel NIR.
    Intentionally narrower than RGB — NIR features are simpler.
    Channels: 1 → 8 → 16 → 32
    Returns stride-8 features for fusion.
    """
    def __init__(self, base_ch: int = 8):
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            _conv_bn_relu(1, c, k=3, s=2, p=1),
            _conv_bn_relu(c, c),
        )
        self.stage1 = _conv_bn_relu(c, c * 2, k=3, s=2, p=1)
        self.stage2 = nn.Sequential(
            _conv_bn_relu(c * 2, c * 4, k=3, s=2, p=1),
            _conv_bn_relu(c * 4, c * 4),
        )
        self.out_fuse = c * 4   # stride-8 channels for fusion

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x                # stride-8


# ── AFF fusion (unchanged from main arch) ─────────────────────────────────────

class AFFModule(nn.Module):
    """Attentive Feature Fusion — local + global gate."""
    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            _gn(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            _gn(channels),
        )
        self.gap_conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.gap_ln1   = nn.LayerNorm(channels)
        self.gap_conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.gap_ln2   = nn.LayerNorm(channels)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, rgb_feat, nir_feat):
        x = rgb_feat + nir_feat
        L = self.local(x)
        g = F.adaptive_avg_pool2d(x, 1)
        B, C, _, _ = g.shape
        g = self.gap_conv1(g).view(B, C)
        g = F.relu(self.gap_ln1(g), inplace=True).view(B, C, 1, 1)
        g = self.gap_conv2(g).view(B, C)
        g = self.gap_ln2(g).view(B, C, 1, 1)
        W = self.sigmoid(L + g)
        self.last_gate = W  # stored for visualization
        return 2.0 * W * rgb_feat + 2.0 * (1.0 - W) * nir_feat


# ── CBAM (channel + spatial attention on skip) ────────────────────────────────

class CBAM(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        mid = max(ch // r, 4)
        self.ca = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(mid, ch, 1, bias=False))
        self.sa = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.ca(x.mean(-1, keepdim=True).mean(-2, keepdim=True)) +
                            self.ca(x.amax(dim=(-1, -2), keepdim=True)))
        x  = x * ca
        sa = torch.sigmoid(self.sa(torch.cat([x.mean(1, keepdim=True),
                                               x.amax(1, keepdim=True)], 1)))
        self.sa_map = sa   # stored for attention transfer distillation
        return x * sa


# ── Lightweight ASPP neck ─────────────────────────────────────────────────────

class LightASPP(nn.Module):
    """
    4-branch ASPP on fused stride-8 features.
    Much lighter than standard ASPP: out_ch=64 instead of 256.
    Rates chosen for 640px input at stride-8 (effective receptive field).
    """
    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.b0 = _conv_bn_relu(in_ch, out_ch, k=1, p=0)           # 1×1
        self.b1 = _conv_bn_relu(in_ch, out_ch, k=3, p=6,  d=6)     # rate 6
        self.b2 = _conv_bn_relu(in_ch, out_ch, k=3, p=12, d=12)    # rate 12
        self.b3 = _conv_bn_relu(in_ch, out_ch, k=3, p=18, d=18)    # rate 18
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            _conv_bn_relu(out_ch * 5, out_ch, k=1, p=0),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        gap  = F.interpolate(self.gap(x), (H, W), mode='bilinear', align_corners=False)
        out  = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), gap], dim=1)
        return self.proj(out)


# ── Decoder ───────────────────────────────────────────────────────────────────

class MiniDecoder(nn.Module):
    """
    Two-stage decoder:
      1. Upsample ASPP output (stride-8 → stride-4) and fuse with RGB skip
      2. Upsample to full resolution and classify
    """
    def __init__(self, aspp_ch: int, skip_ch: int, num_classes: int):
        super().__init__()
        # Project skip connection to smaller dim
        self.skip_proj = _conv_bn_relu(skip_ch, aspp_ch // 2, k=1, p=0)

        # Fuse ASPP + skip
        self.fuse = nn.Sequential(
            _conv_bn_relu(aspp_ch + aspp_ch // 2, aspp_ch, k=3, p=1),
            _conv_bn_relu(aspp_ch, aspp_ch, k=3, p=1),
        )
        self.classifier = nn.Conv2d(aspp_ch, num_classes, 1)

    def forward(self, aspp_feat, skip_feat, input_size):
        # Upsample to stride-4 resolution
        up = F.interpolate(aspp_feat, size=skip_feat.shape[-2:],
                           mode='bilinear', align_corners=False)
        skip = self.skip_proj(skip_feat)
        x = self.fuse(torch.cat([up, skip], dim=1))
        logits = self.classifier(x)
        return F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)


# ── Aux head for deep supervision ─────────────────────────────────────────────

class AuxHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            _conv_bn_relu(in_ch, in_ch // 2, k=3, p=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_ch // 2, num_classes, 1),
        )

    def forward(self, x, target_size):
        return F.interpolate(self.head(x), size=target_size,
                             mode='bilinear', align_corners=False)


# ── Full model ────────────────────────────────────────────────────────────────

class DualEncoderMini(nn.Module):
    """
    Proposed DualEncoderMini for paper.

    Args:
        rgb_base_ch  : RGB encoder base channels (default 16)
        nir_base_ch  : NIR encoder base channels (default 8)
        aspp_ch      : ASPP output channels (default 64)
        num_classes  : output classes (3 for bg/crop/weed)

    ~0.7M params with defaults — comparable to DeepLabV3+ at 0.63M.
    """
    def __init__(self, rgb_base_ch=16, nir_base_ch=8, aspp_ch=64, num_classes=3):
        super().__init__()

        self.rgb_enc = RGBMiniEncoder(rgb_base_ch)
        self.nir_enc = NIRMiniEncoder(nir_base_ch)

        rgb_fuse_ch = self.rgb_enc.out_fuse   # 64
        nir_fuse_ch = self.nir_enc.out_fuse   # 32

        # Project NIR to match RGB channels before AFF
        self.nir_proj = _conv_bn_relu(nir_fuse_ch, rgb_fuse_ch, k=1, p=0)

        # AFF fusion at stride-8
        self.aff  = AFFModule(rgb_fuse_ch)
        self.cbam = CBAM(rgb_fuse_ch)

        # ASPP neck
        self.aspp = LightASPP(rgb_fuse_ch, aspp_ch)

        # Decoder with stride-4 skip from RGB
        skip_ch = self.rgb_enc.out_skip       # 32
        self.decoder = MiniDecoder(aspp_ch, skip_ch, num_classes)

        # Aux head on fused features (deep supervision)
        self.aux_head = AuxHead(rgb_fuse_ch, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_rgb, x_nir):
        input_size = x_rgb.shape[-2:]

        # Encode
        skip, rgb_s8 = self.rgb_enc(x_rgb)   # stride-4 skip, stride-8
        nir_s8       = self.nir_enc(x_nir)   # stride-8

        # Project NIR → same channels as RGB, then fuse
        nir_s8  = self.nir_proj(nir_s8)
        fused   = self.cbam(self.aff(rgb_s8, nir_s8))

        # ASPP context
        neck = self.aspp(fused)

        # Decode with high-res skip
        logits = self.decoder(neck, skip, input_size)

        if self.training:
            aux = self.aux_head(fused, input_size)
            return logits, aux

        return logits


# ── Factory / test ────────────────────────────────────────────────────────────

def create_dual_encoder_mini(num_classes=3):
    return DualEncoderMini(rgb_base_ch=16, nir_base_ch=8, aspp_ch=64,
                           num_classes=num_classes)


if __name__ == "__main__":
    model = create_dual_encoder_mini(num_classes=3)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DualEncoderMini  params: {total/1e6:.3f}M")

    rgb = torch.randn(2, 3, 640, 640)
    nir = torch.randn(2, 1, 640, 640)

    model.train()
    logits, aux = model(rgb, nir)
    print(f"Train output : {logits.shape}  aux: {aux.shape}")

    model.eval()
    with torch.no_grad():
        out = model(rgb, nir)
    print(f"Eval  output : {out.shape}")