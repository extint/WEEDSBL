# models.py
"""
Lightweight PSPNet implementation + simple factory functions expected by train.py:
- create_model(architecture, in_channels, num_classes, base_ch)
- get_model_info(model)

Designed for binary segmentation (num_classes=1) but supports general num_classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. DeepLabV3+ (Highly effective for agricultural segmentation)
# ==============================================================================
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

# ============================================================
#                  LIGHTWEIGHT SEGNET
# ============================================================

class LightSegNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()

        C1 = base_ch          # 32
        C2 = base_ch * 2      # 64
        C3 = base_ch * 4      # 128
        C4 = base_ch * 8      # 256
        C5 = base_ch * 8      # 256 (kept same for final depth)

        self.enc1 = self._block(in_channels, C1)
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.enc2 = self._block(C1, C2)
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.enc3 = self._block(C2, C3)
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.enc4 = self._block(C3, C4)
        self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.enc5 = self._block(C4, C5)
        self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.out_channels = (C1, C2, C3, C4, C5)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        sizes = []

        x = self.enc1(x); sizes.append(x.size()); x, idx1 = self.pool1(x)
        x = self.enc2(x); sizes.append(x.size()); x, idx2 = self.pool2(x)
        x = self.enc3(x); sizes.append(x.size()); x, idx3 = self.pool3(x)
        x = self.enc4(x); sizes.append(x.size()); x, idx4 = self.pool4(x)
        x = self.enc5(x); sizes.append(x.size()); x, idx5 = self.pool5(x)

        return x, (idx1, idx2, idx3, idx4, idx5), sizes


class LightSegNetDecoder(nn.Module):
    def __init__(self, out_channels, num_classes=1):
        super().__init__()

        C1, C2, C3, C4, C5 = out_channels

        self.unpool = nn.MaxUnpool2d(2, stride=2)

        # reverse channel progression
        self.dec5 = self._block(C5, C4)
        self.dec4 = self._block(C4, C3)
        self.dec3 = self._block(C3, C2)
        self.dec2 = self._block(C2, C1)
        self.dec1 = self._block(C1, C1)

        self.final = nn.Conv2d(C1, num_classes, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, indices, sizes):
        idx1, idx2, idx3, idx4, idx5 = indices
        s1, s2, s3, s4, s5 = sizes

        x = self.unpool(x, idx5, output_size=s5)
        x = self.dec5(x)

        x = self.unpool(x, idx4, output_size=s4)
        x = self.dec4(x)

        x = self.unpool(x, idx3, output_size=s3)
        x = self.dec3(x)

        x = self.unpool(x, idx2, output_size=s2)
        x = self.dec2(x)

        x = self.unpool(x, idx1, output_size=s1)
        x = self.dec1(x)

        return self.final(x)


class LightSegNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_ch=32):
        super().__init__()
        self.encoder = LightSegNetEncoder(in_channels, base_ch)
        self.decoder = LightSegNetDecoder(self.encoder.out_channels, num_classes)

    def forward(self, x):
        x, indices, sizes = self.encoder(x)
        return self.decoder(x, indices, sizes)

######### UNET ++ ##################

# ---------------------------
# Basic building blocks
# ---------------------------
class DoubleConvD(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None, dropout=0.0):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        layers = [
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout and dropout > 0.0:
            layers.insert(3, nn.Dropout2d(dropout))  # after first ReLU
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


# ---------------------------
# UNet++
# ---------------------------
class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested U-Net) implementation.
    - in_channels: input channels (3 for RGB, 4 for RGB+NIR)
    - num_classes: output channels (1 for binary segmentation)
    - base_ch: number of filters in level 0 (doubles each level)
    - dropout: optional dropout probability inside DoubleConv
    """
    def __init__(self, in_channels=3, num_classes=1, base_ch=32, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_ch = base_ch
        self.dropout = dropout

        # Encoder (x_{i,0})
        self.conv00 = DoubleConvD(in_channels, base_ch, dropout=dropout)
        self.conv10 = DoubleConvD(base_ch, base_ch * 2, dropout=dropout)
        self.conv20 = DoubleConvD(base_ch * 2, base_ch * 4, dropout=dropout)
        self.conv30 = DoubleConvD(base_ch * 4, base_ch * 8, dropout=dropout)
        self.conv40 = DoubleConvD(base_ch * 8, base_ch * 16, dropout=dropout)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Nested convs: x_{i,j} for j>=1
        # For each i, j we create a DoubleConv with input channels = channels(i) + channels(i+1)
        # and output channels = channels(i)
        def ch(i): return base_ch * (2 ** i)

        # level 1
        self.conv01 = DoubleConvD(ch(0) + ch(1), ch(0), dropout=dropout)
        self.conv11 = DoubleConvD(ch(1) + ch(2), ch(1), dropout=dropout)
        self.conv21 = DoubleConvD(ch(2) + ch(3), ch(2), dropout=dropout)
        self.conv31 = DoubleConvD(ch(3) + ch(4), ch(3), dropout=dropout)

        # level 2
        self.conv02 = DoubleConvD(ch(0) + ch(1), ch(0), dropout=dropout)  # input same shape concatenation pattern maintained
        self.conv12 = DoubleConvD(ch(1) + ch(2), ch(1), dropout=dropout)
        self.conv22 = DoubleConvD(ch(2) + ch(3), ch(2), dropout=dropout)

        # level 3
        self.conv03 = DoubleConvD(ch(0) + ch(1), ch(0), dropout=dropout)
        self.conv13 = DoubleConvD(ch(1) + ch(2), ch(1), dropout=dropout)

        # level 4 (top-most nested)
        self.conv04 = DoubleConvD(ch(0) + ch(1), ch(0), dropout=dropout)

        # final classifier (we take the deepest nested output conv04 as final)
        self.final = nn.Conv2d(ch(0), num_classes, kernel_size=1)

        # initialize weights
        self._init_weights()

    def forward(self, x):
        # Encoder path
        x00 = self.conv00(x)                    # (B, C0, H, W)
        x10 = self.conv10(self.pool(x00))       # (B, C1, H/2, W/2)
        x20 = self.conv20(self.pool(x10))       # (B, C2, H/4, W/4)
        x30 = self.conv30(self.pool(x20))       # (B, C3, H/8, W/8)
        x40 = self.conv40(self.pool(x30))       # (B, C4, H/16, W/16)

        # Level 1
        x01 = self.conv01(torch.cat([x00, self._upsample(x10, x00)], dim=1))
        x11 = self.conv11(torch.cat([x10, self._upsample(x20, x10)], dim=1))
        x21 = self.conv21(torch.cat([x20, self._upsample(x30, x20)], dim=1))
        x31 = self.conv31(torch.cat([x30, self._upsample(x40, x30)], dim=1))

        # Level 2
        x02 = self.conv02(torch.cat([x01, self._upsample(x11, x01)], dim=1))
        x12 = self.conv12(torch.cat([x11, self._upsample(x21, x11)], dim=1))
        x22 = self.conv22(torch.cat([x21, self._upsample(x31, x21)], dim=1))

        # Level 3
        x03 = self.conv03(torch.cat([x02, self._upsample(x12, x02)], dim=1))
        x13 = self.conv13(torch.cat([x12, self._upsample(x22, x12)], dim=1))

        # Level 4 (final nested fusion)
        x04 = self.conv04(torch.cat([x03, self._upsample(x13, x03)], dim=1))

        logits = self.final(x04)
        return logits

    def _upsample(self, src, target):
        """Upsample src to the spatial size of target using bilinear interpolation"""
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='bilinear', align_corners=False)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
########################################################

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_ch=32, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        C = base_ch

        # Encoder path
        self.conv00 = DoubleConv(in_channels, C)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.conv10 = DoubleConv(C, C*2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv20 = DoubleConv(C*2, C*4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv30 = DoubleConv(C*4, C*8)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv40 = DoubleConv(C*8, C*16)

        # Decoder / full-scale aggregation
        # Each decoder level gets 5 inputs (from all encoder levels)
        # We use 1x1 convs to reduce channels before concatenation
        
        # Level 3 decoder (H/8, W/8) - gets inputs from e0,e1,e2,e3,e4
        self.e0_to_d3 = nn.Conv2d(C, C, kernel_size=1)
        self.e1_to_d3 = nn.Conv2d(C*2, C, kernel_size=1)
        self.e2_to_d3 = nn.Conv2d(C*4, C, kernel_size=1)
        self.e3_to_d3 = nn.Conv2d(C*8, C, kernel_size=1)
        self.e4_to_d3 = nn.Conv2d(C*16, C, kernel_size=1)
        self.conv3d = DoubleConv(C*5, C*8)
        
        # Level 2 decoder (H/4, W/4) - gets inputs from e0,e1,e2,e3,e4,d3
        self.e0_to_d2 = nn.Conv2d(C, C, kernel_size=1)
        self.e1_to_d2 = nn.Conv2d(C*2, C, kernel_size=1)
        self.e2_to_d2 = nn.Conv2d(C*4, C, kernel_size=1)
        self.e3_to_d2 = nn.Conv2d(C*8, C, kernel_size=1)
        self.e4_to_d2 = nn.Conv2d(C*16, C, kernel_size=1)
        self.d3_to_d2 = nn.Conv2d(C*8, C, kernel_size=1)
        self.conv2d = DoubleConv(C*6, C*4)
        
        # Level 1 decoder (H/2, W/2) - gets inputs from e0,e1,e2,e3,e4,d3,d2
        self.e0_to_d1 = nn.Conv2d(C, C, kernel_size=1)
        self.e1_to_d1 = nn.Conv2d(C*2, C, kernel_size=1)
        self.e2_to_d1 = nn.Conv2d(C*4, C, kernel_size=1)
        self.e3_to_d1 = nn.Conv2d(C*8, C, kernel_size=1)
        self.e4_to_d1 = nn.Conv2d(C*16, C, kernel_size=1)
        self.d3_to_d1 = nn.Conv2d(C*8, C, kernel_size=1)
        self.d2_to_d1 = nn.Conv2d(C*4, C, kernel_size=1)
        self.conv1d = DoubleConv(C*7, C*2)
        
        # Level 0 decoder (H, W) - gets inputs from e0,e1,e2,e3,e4,d3,d2,d1
        self.e0_to_d0 = nn.Conv2d(C, C, kernel_size=1)
        self.e1_to_d0 = nn.Conv2d(C*2, C, kernel_size=1)
        self.e2_to_d0 = nn.Conv2d(C*4, C, kernel_size=1)
        self.e3_to_d0 = nn.Conv2d(C*8, C, kernel_size=1)
        self.e4_to_d0 = nn.Conv2d(C*16, C, kernel_size=1)
        self.d3_to_d0 = nn.Conv2d(C*8, C, kernel_size=1)
        self.d2_to_d0 = nn.Conv2d(C*4, C, kernel_size=1)
        self.d1_to_d0 = nn.Conv2d(C*2, C, kernel_size=1)
        self.conv0d = DoubleConv(C*8, C)

        # Final classifier
        self.final = nn.Conv2d(C, num_classes, kernel_size=1)

        # Optional deep supervision side outputs
        if self.deep_supervision:
            self.ds3 = nn.Conv2d(C*8, num_classes, kernel_size=1)
            self.ds2 = nn.Conv2d(C*4, num_classes, kernel_size=1)
            self.ds1 = nn.Conv2d(C*2, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _upsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    
    def _downsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        e0 = self.conv00(x)              # H, W, C
        e1 = self.conv10(self.pool0(e0)) # H/2, W/2, C*2
        e2 = self.conv20(self.pool1(e1)) # H/4, W/4, C*4
        e3 = self.conv30(self.pool2(e2)) # H/8, W/8, C*8
        e4 = self.conv40(self.pool3(e3)) # H/16, W/16, C*16

        # Decoder level 3 (H/8, W/8)
        d3_inputs = [
            self.e0_to_d3(self._upsample(e0, e3.shape[2:])),  # upsample from H to H/8
            self.e1_to_d3(self._upsample(e1, e3.shape[2:])),  # upsample from H/2 to H/8
            self.e2_to_d3(self._upsample(e2, e3.shape[2:])),  # upsample from H/4 to H/8
            self.e3_to_d3(e3),                                 # same size
            self.e4_to_d3(self._upsample(e4, e3.shape[2:]))   # downsample from H/16 to H/8
        ]
        d3 = self.conv3d(torch.cat(d3_inputs, dim=1))

        # Decoder level 2 (H/4, W/4)
        d2_inputs = [
            self.e0_to_d2(F.max_pool2d(e0, 4)),  # downsample from H to H/4
            self.e1_to_d2(F.max_pool2d(e1, 2)),  # downsample from H/2 to H/4
            self.e2_to_d2(e2),
            self.e3_to_d2(self._upsample(e3, e2.shape[2:])),  # upsample from H/8 to H/4
            self.e4_to_d2(self._upsample(e4, e2.shape[2:])),  # upsample from H/16 to H/4
            self.d3_to_d2(self._upsample(d3, e2.shape[2:]))   # upsample from H/8 to H/4
        ]
        d2 = self.conv2d(torch.cat(d2_inputs, dim=1))

        # Decoder level 1 (H/2, W/2)
        d1_inputs = [
            self.e0_to_d1(F.max_pool2d(e0, 2)),  # downsample from H to H/2
            self.e1_to_d1(e1),
            self.e2_to_d1(self._upsample(e2, e1.shape[2:])),  # upsample from H/4 to H/2
            self.e3_to_d1(self._upsample(e3, e1.shape[2:])),  # upsample from H/8 to H/2
            self.e4_to_d1(self._upsample(e4, e1.shape[2:])),  # upsample from H/16 to H/2
            self.d3_to_d1(self._upsample(d3, e1.shape[2:])),  # upsample from H/8 to H/2
            self.d2_to_d1(self._upsample(d2, e1.shape[2:]))   # upsample from H/4 to H/2
        ]
        d1 = self.conv1d(torch.cat(d1_inputs, dim=1))

        # Decoder level 0 (H, W)
        d0_inputs = [
            self.e0_to_d0(e0),
            self.e1_to_d0(self._upsample(e1, e0.shape[2:])),  # upsample from H/2 to H
            self.e2_to_d0(self._upsample(e2, e0.shape[2:])),  # upsample from H/4 to H
            self.e3_to_d0(self._upsample(e3, e0.shape[2:])),  # upsample from H/8 to H
            self.e4_to_d0(self._upsample(e4, e0.shape[2:])),  # upsample from H/16 to H
            self.d3_to_d0(self._upsample(d3, e0.shape[2:])),  # upsample from H/8 to H
            self.d2_to_d0(self._upsample(d2, e0.shape[2:])),  # upsample from H/4 to H
            self.d1_to_d0(self._upsample(d1, e0.shape[2:]))   # upsample from H/2 to H
        ]
        d0 = self.conv0d(torch.cat(d0_inputs, dim=1))

        # Final output
        logits = self.final(d0)

        if self.deep_supervision:
            ds3 = self.ds3(d3)
            ds2 = self.ds2(d2)
            ds1 = self.ds1(d1)
            return logits, ds3, ds2, ds1

        return logits

# ------------------------------
# Factory functions expected by train.py
# ------------------------------
def create_model(architecture: str, in_channels: int = 4, num_classes: int = 1, base_ch: int = 32):
    arch = architecture.lower()
    if arch == "deeplabsv3+":
        return DeepLabV3Plus(in_channels, num_classes)
    elif arch == "pspnet":
        return PSPNet(in_channels, num_classes)
    elif arch == "lightsegnet":
        return LightSegNet(in_channels, num_classes, base_ch)
    elif arch == "unet++":
        return UNetPlusPlus(in_channels, num_classes, base_ch)
    elif arch == "unet3+":
        return UNet3Plus(in_channels, num_classes, base_ch)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Supported: 'pspnet'.")


def get_model_info(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "architecture": model.__class__.__name__,
        "total_parameters": total_params,
        "total_parameters_million": total_params / 1_000_000.0
    }

model_list = ["unet3+"]

for model in model_list:
    model_entity = create_model(
        architecture=model,
        in_channels=4,
        num_classes=1,
        base_ch = 16,
    )
    info = get_model_info(model_entity)
    print(f"[INFO] Model: {info["architecture"]}")
    print(f"[INFO] Parameters: {info["total_parameters"]:,} "
        f"({info['total_parameters_million']:.2f}M)")


