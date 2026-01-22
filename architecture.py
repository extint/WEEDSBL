"""
Dual-Encoder Architecture with Attention-Based RGB Encoder
- RGB: Efficient Transformer (MiT-style) for long-range context
- NIR: Lightweight ResNet for local physiological features
- Fusion: AFF modules at multiple scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ============================================================
# 1. EFFICIENT TRANSFORMER RGB ENCODER (MiT-style)
# ============================================================

class OverlapPatchEmbed(nn.Module):
    """
    Overlapped patch embedding used in Mix Transformer (MiT).
    Projects image patches to embedding with stride < patch_size for overlap.
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """
    Efficient self-attention with spatial reduction (like PVT/SegFormer).
    Reduces KV spatial dimension by sr_ratio to save compute.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Spatial reduction for K, V
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Query
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Key, Value with spatial reduction
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block: Efficient self-attention + MLP.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, sr_ratio=sr_ratio)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class MiTEncoder(nn.Module):
    """
    Mix Transformer Encoder (simplified MiT-B0/B1 style).
    4-stage hierarchical transformer with overlapped patch embedding.
    
    Stage configs: (embed_dim, depth, num_heads, sr_ratio)
    B0-style (lightweight): [(32, 2), (64, 2), (160, 2), (256, 2)]
    B1-style (balanced):    [(64, 2), (128, 2), (320, 2), (512, 2)]
    """
    def __init__(self, in_chans=3, embed_dims=[32, 64, 160, 256], 
                 depths=[2, 2, 2, 2], num_heads=[1, 2, 5, 8], 
                 sr_ratios=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4]):
        super().__init__()
        
        self.stages = nn.ModuleList()
        
        # Stage 1
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias=True, sr_ratio=sr_ratios[0])
            for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        
        # Stage 2
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias=True, sr_ratio=sr_ratios[1])
            for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        
        # Stage 3
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias=True, sr_ratio=sr_ratios[2])
            for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        
        # Stage 4
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias=True, sr_ratio=sr_ratios[3])
            for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        B = x.shape[0]
        outs = []
        
        # Stage 1: stride 4
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 2: stride 8
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 3: stride 16
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 4: stride 32
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        return outs  # [stage1_feat, stage2_feat, stage3_feat, stage4_feat]


class RGBTransformerEncoder(nn.Module):
    """
    RGB encoder using efficient transformer (MiT).
    Returns multi-scale features with long-range attention.
    
    Variants:
    - 'tiny': For Jetson Nano (embed_dims=[32, 64, 128, 256])
    - 'small': Balanced (embed_dims=[32, 64, 160, 256]) 
    - 'base': Higher capacity (embed_dims=[64, 128, 320, 512])
    """
    def __init__(self, variant='small'):
        super().__init__()
        
        if variant == 'tiny':
            self.encoder = MiTEncoder(
                embed_dims=[32, 64, 128, 256],
                depths=[2, 2, 2, 2],
                num_heads=[1, 2, 4, 8],
                sr_ratios=[8, 4, 2, 1]
            )
        elif variant == 'small':
            self.encoder = MiTEncoder(
                embed_dims=[32, 64, 160, 256],
                depths=[2, 2, 2, 2],
                num_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1]
            )
        elif variant == 'base':
            self.encoder = MiTEncoder(
                embed_dims=[64, 128, 320, 512],
                depths=[3, 4, 6, 3],
                num_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1]
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")
    
    def forward(self, x_rgb):
        return self.encoder(x_rgb)


# ============================================================
# 2. LIGHTWEIGHT NIR ENCODER (Efficient ResNet-style)
# ============================================================

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv for efficiency"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, 
                                   padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightResBlock(nn.Module):
    """Lightweight residual block for NIR encoder"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class NIRLightEncoder(nn.Module):
    """
    Lightweight encoder for single-channel NIR.
    Efficient ResNet-18 style with reduced capacity.
    """
    def __init__(self, base_ch=16):
        super().__init__()
        
        # Initial conv: 1 -> base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 4 stages
        self.layer1 = self._make_layer(base_ch, base_ch * 2, blocks=2, stride=1)      # stride 4
        self.layer2 = self._make_layer(base_ch * 2, base_ch * 4, blocks=2, stride=2)  # stride 8
        self.layer3 = self._make_layer(base_ch * 4, base_ch * 8, blocks=2, stride=2)  # stride 16
        self.layer4 = self._make_layer(base_ch * 8, base_ch * 16, blocks=2, stride=2) # stride 32
    
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        layers.append(LightResBlock(in_ch, out_ch, stride))
        for _ in range(1, blocks):
            layers.append(LightResBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x_nir):
        x = self.stem(x_nir)
        
        x1 = self.layer1(x)   # stride 4
        x2 = self.layer2(x1)  # stride 8
        x3 = self.layer3(x2)  # stride 16
        x4 = self.layer4(x3)  # stride 32
        
        return [x1, x2, x3, x4]


# ============================================================
# 3. AFF FUSION MODULE (Unchanged - already optimal)
# ============================================================

class AFFModule(nn.Module):
    """
    Attention-based Fusion Feature (AFF) module.
    Adaptively fuses RGB and NIR features with local + global attention.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Local attention
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # Global attention
        self.global_conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.global_ln1 = nn.LayerNorm(channels)
        self.global_conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.global_ln2 = nn.LayerNorm(channels)
        
        self.sigmoid = nn.Sigmoid()
        self.last_gate = None  # For visualization
    
    def forward(self, rgb_feat, nir_feat):
        x = rgb_feat + nir_feat
        
        # Local branch
        L = self.local(x)
        
        # Global branch
        g = F.adaptive_avg_pool2d(x, output_size=1)
        g = self.global_conv1(g)
        B, C, _, _ = g.shape
        g = g.view(B, C)
        g = self.global_ln1(g)
        g = F.relu(g, inplace=True)
        g = g.view(B, C, 1, 1)
        g = self.global_conv2(g)
        g = g.view(B, C)
        g = self.global_ln2(g)
        g = g.view(B, C, 1, 1)
        
        # Gate
        W = self.sigmoid(L + g)
        self.last_gate = W  # Store for visualization
        
        # Fused output
        fused = 2.0 * W * rgb_feat + 2.0 * (1.0 - W) * nir_feat
        return fused


class StageProjector(nn.Module):
    """Projects RGB and NIR to common channel dimension"""
    def __init__(self, in_ch_rgb: int, in_ch_nir: int, out_ch: int):
        super().__init__()
        self.proj_rgb = nn.Conv2d(in_ch_rgb, out_ch, kernel_size=1, bias=False)
        self.proj_nir = nn.Conv2d(in_ch_nir, out_ch, kernel_size=1, bias=False)
    
    def forward(self, feat_rgb, feat_nir):
        return self.proj_rgb(feat_rgb), self.proj_nir(feat_nir)


# ============================================================
# 4. LIGHTWEIGHT DECODER (Unchanged)
# ============================================================

class SimpleDecoder(nn.Module):
    """SegFormer-style MLP decoder"""
    def __init__(self, in_ch2: int, in_ch3: int, in_ch4: int, 
                 embed_dim: int, num_classes: int):
        super().__init__()
        self.proj2 = nn.Conv2d(in_ch2, embed_dim, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(in_ch3, embed_dim, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(in_ch4, embed_dim, kernel_size=1, bias=False)
        
        self.fuse = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1, bias=False)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
    
    def forward(self, f2, f3, f4, input_size):
        B, _, H2, W2 = f2.shape
        
        p2 = self.proj2(f2)
        p3 = self.proj3(f3)
        p4 = self.proj4(f4)
        
        p3 = F.interpolate(p3, size=(H2, W2), mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=(H2, W2), mode="bilinear", align_corners=False)
        
        x = torch.cat([p2, p3, p4], dim=1)
        x = self.fuse(x)
        logits = self.classifier(x)
        
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


# ============================================================
# 5. FINAL DUAL-ENCODER NETWORK
# ============================================================

class DualEncoderAFFNet(nn.Module):
    """
    Improved Dual-Encoder with:
    - RGB: Efficient Transformer (MiT) for long-range attention
    - NIR: Lightweight ResNet for local features
    - AFF fusion at multiple scales
    - SegFormer-style decoder
    
    Args:
        rgb_variant: 'tiny' | 'small' | 'base' (transformer size)
        nir_base_ch: NIR encoder base channels (default: 16)
        num_classes: Output classes (default: 1 for binary)
        embed_dim: Decoder embedding dimension
    """
    def __init__(self, 
                 rgb_variant='tiny',
                 nir_base_ch=16,
                 num_classes=1,
                 embed_dim=64):
        super().__init__()
        
        # Encoders
        self.rgb_encoder = RGBTransformerEncoder(variant=rgb_variant)
        self.nir_encoder = NIRLightEncoder(base_ch=nir_base_ch)
        
        # Get channel dimensions based on variant
        if rgb_variant == 'tiny':
            rgb_dims = [32, 64, 128, 256]
        elif rgb_variant == 'small':
            rgb_dims = [32, 64, 160, 256]
        elif rgb_variant == 'base':
            rgb_dims = [64, 128, 320, 512]
        
        nir_dims = [nir_base_ch * 2, nir_base_ch * 4, nir_base_ch * 8, nir_base_ch * 16]
        
        # Fusion at stages 1, 2, 3 (stride 8, 16, 32)
        fused_dims = [64, 128, 256]
        
        self.proj1 = StageProjector(rgb_dims[1], nir_dims[1], fused_dims[0])
        self.aff1 = AFFModule(fused_dims[0])
        
        self.proj2 = StageProjector(rgb_dims[2], nir_dims[2], fused_dims[1])
        self.aff2 = AFFModule(fused_dims[1])
        
        self.proj3 = StageProjector(rgb_dims[3], nir_dims[3], fused_dims[2])
        self.aff3 = AFFModule(fused_dims[2])
        
        # Decoder
        self.decoder = SimpleDecoder(
            in_ch2=fused_dims[0],
            in_ch3=fused_dims[1],
            in_ch4=fused_dims[2],
            embed_dim=embed_dim,
            num_classes=num_classes
        )
    
    def forward(self, x_rgb, x_nir):
        """
        x_rgb: (B, 3, H, W)
        x_nir: (B, 1, H, W)
        """
        input_size = x_rgb.shape[-2:]
        
        # Encode
        R_feats = self.rgb_encoder(x_rgb)  # [stride4, 8, 16, 32]
        N_feats = self.nir_encoder(x_nir)   # [stride4, 8, 16, 32]
        
        # Fuse stages 2, 3, 4 (stride 8, 16, 32)
        r1, n1 = self.proj1(R_feats[1], N_feats[1])
        F1 = self.aff1(r1, n1)
        
        r2, n2 = self.proj2(R_feats[2], N_feats[2])
        F2 = self.aff2(r2, n2)
        
        r3, n3 = self.proj3(R_feats[3], N_feats[3])
        F3 = self.aff3(r3, n3)
        
        # Decode
        logits = self.decoder(F1, F2, F3, input_size=input_size)
        return logits


# ============================================================
# HELPER: Model factory
# ============================================================

def create_dual_encoder_model(variant='small', nir_base_ch=16, num_classes=1):
    """
    Factory function to create model with different configs.
    
    Variants:
    - 'tiny': For Jetson Nano (~3-5M params)
    - 'small': Balanced accuracy/speed (~5-8M params)
    - 'base': Higher accuracy (~12-15M params)
    """
    model = DualEncoderAFFNet(
        rgb_variant=variant,
        nir_base_ch=nir_base_ch,
        num_classes=num_classes,
        embed_dim=64
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Created DualEncoderAFFNet-{variant}: {total_params/1e6:.2f}M parameters")
    
    return model


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Test different variants
    for variant in ['tiny', 'small']:
        print(f"\n{'='*50}")
        print(f"Testing variant: {variant}")
        print('='*50)
        
        model = create_dual_encoder_model(variant=variant, nir_base_ch=16, num_classes=1)
        
        x_rgb = torch.randn(2, 3, 512, 512)
        x_nir = torch.randn(2, 1, 512, 512)
        
        with torch.no_grad():
            out = model(x_rgb, x_nir)
        
        print(f"Input RGB: {x_rgb.shape}")
        print(f"Input NIR: {x_nir.shape}")
        print(f"Output: {out.shape}")
        
        # FLOPs estimate (rough)
        from thop import profile
        flops, params = profile(model, inputs=(x_rgb, x_nir), verbose=False)
        print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")
