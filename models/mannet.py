import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Encoder path - no skip connections to save memory
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Simple decoder
        x = self.decoder(x)
        logits = self.final(x)
        return logits