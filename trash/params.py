import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Channel-wise max pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along channel dimension
        out = torch.cat([avg_out, max_out], dim=1)

        # Convolve, then apply sigmoid -> attention mask
        attn = self.conv1(out)
        attn = self.sigmoid(attn)

        # Multiply the attention mask with the input
        return x * attn

class UNet_SA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_SA, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bridge
        self.bridge = self.conv_block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        # Spatial Attention modules
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa_bridge = SpatialAttention()

    def conv_block(self, in_channels, out_channels):
        """
        A basic (Conv -> ReLU -> Conv -> ReLU) block,
        no Batch Normalization layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # ------------------ Encoder ------------------
        enc1 = self.enc1(x)            # (B,64,H,W)
        enc1 = self.sa1(enc1)          # Spatial attention

        enc2 = self.enc2(F.max_pool2d(enc1, 2))  # (B,128,H/2,W/2)
        enc2 = self.sa2(enc2)

        enc3 = self.enc3(F.max_pool2d(enc2, 2))  # (B,256,H/4,W/4)
        enc3 = self.sa3(enc3)

        enc4 = self.enc4(F.max_pool2d(enc3, 2))  # (B,512,H/8,W/8)
        enc4 = self.sa4(enc4)

        # ------------------ Bridge -------------------
        bridge = self.bridge(F.max_pool2d(enc4, 2))  # (B,1024,H/16,W/16)
        bridge = self.sa_bridge(bridge)

        # ------------------ Decoder ------------------
        # Up4
        dec4 = self.up4(bridge)                 # (B,512,H/8,W/8)
        dec4 = torch.cat([enc4, dec4], dim=1)   # (B,1024,H/8,W/8)
        dec4 = self.dec4(dec4)                  # (B,512,H/8,W/8)

        # Up3
        dec3 = self.up3(dec4)                   # (B,256,H/4,W/4)
        dec3 = torch.cat([enc3, dec3], dim=1)   # (B,512,H/4,W/4)
        dec3 = self.dec3(dec3)                  # (B,256,H/4,W/4)

        # Up2
        dec2 = self.up2(dec3)                   # (B,128,H/2,W/2)
        dec2 = torch.cat([enc2, dec2], dim=1)   # (B,256,H/2,W/2)
        dec2 = self.dec2(dec2)                  # (B,128,H/2,W/2)

        # Up1
        dec1 = self.up1(dec2)                   # (B,64,H,W)
        dec1 = torch.cat([enc1, dec1], dim=1)   # (B,128,H,W)
        dec1 = self.dec1(dec1)                  # (B,64,H,W)

        # Final conv
        out = self.out(dec1)                    # (B,out_channels,H,W)
        return out
# export_to_onnx.py
import torch

ckpt_path = "/home/vjti-comp/Downloads/unet_best_model.pth"
onnx_path = "/home/vjti-comp/Downloads/unet_best_model.onnx"
device = "cuda"  # or "cuda" if available

# model = UNet_SA()
# state = torch.load(ckpt_path, map_location=device, weights_only=False)
model = torch.load(ckpt_path, map_location=device, weights_only=False)
# model.load_state_dict(state)
model.eval().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params}")
