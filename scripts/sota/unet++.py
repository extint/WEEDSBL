import torch
import torch.nn as nn

# -------------------------
# UNet++ (Nested UNet)
# -------------------------

class DoubleConv(nn.Module):
    """Double convolution block used in UNet++"""
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


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, base_ch: int = 64, 
                 out_channels: int = 1, deep_supervision: bool = True):
        """
        UNet++ architecture
        
        Args:
            in_channels: Number of input channels (e.g., 3 for RGB, 4 for RGBD)
            base_ch: Base number of filters (default: 64)
            out_channels: Number of output channels (e.g., 1 for binary segmentation)
            deep_supervision: If True, returns outputs from multiple decoder levels (default: True)
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.training_mode = True  # Flag to control output behavior
        
        # Encoder (downsampling)
        self.pool = nn.MaxPool2d(2)
        self.conv0_0 = DoubleConv(in_channels, base_ch)
        self.conv1_0 = DoubleConv(base_ch, base_ch * 2)
        self.conv2_0 = DoubleConv(base_ch * 2, base_ch * 4)
        self.conv3_0 = DoubleConv(base_ch * 4, base_ch * 8)
        self.conv4_0 = DoubleConv(base_ch * 8, base_ch * 16)
        
        # Decoder (upsampling + nested skip connections)
        self.up1_0 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.conv0_1 = DoubleConv(base_ch + base_ch * 2, base_ch)
        
        self.up2_0 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.conv1_1 = DoubleConv(base_ch * 2 + base_ch * 4, base_ch * 2)
        
        self.up3_0 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.conv2_1 = DoubleConv(base_ch * 4 + base_ch * 8, base_ch * 4)
        
        self.up4_0 = nn.ConvTranspose2d(base_ch * 16, base_ch * 16, 2, stride=2)
        self.conv3_1 = DoubleConv(base_ch * 8 + base_ch * 16, base_ch * 8)
        
        # Level 2 nested connections
        self.up1_1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.conv0_2 = DoubleConv(base_ch * 2 + base_ch * 2, base_ch)
        
        self.up2_1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.conv1_2 = DoubleConv(base_ch * 2 + base_ch * 2 + base_ch * 4, base_ch * 2)
        
        self.up3_1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.conv2_2 = DoubleConv(base_ch * 4 + base_ch * 4 + base_ch * 8, base_ch * 4)
        
        # Level 3 nested connections
        self.up1_2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.conv0_3 = DoubleConv(base_ch * 3 + base_ch * 2, base_ch)
        
        self.up2_2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.conv1_3 = DoubleConv(base_ch * 2 + base_ch * 2 + base_ch * 2 + base_ch * 4, base_ch * 2)
        
        # Level 4 nested connections
        self.up1_3 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.conv0_4 = DoubleConv(base_ch * 4 + base_ch * 2, base_ch)
        
        # Final output layers
        if deep_supervision:
            self.final1 = nn.Conv2d(base_ch, out_channels, 1)
            self.final2 = nn.Conv2d(base_ch, out_channels, 1)
            self.final3 = nn.Conv2d(base_ch, out_channels, 1)
            self.final4 = nn.Conv2d(base_ch, out_channels, 1)
        else:
            self.final = nn.Conv2d(base_ch, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        # Nested decoder - Level 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        
        # Nested decoder - Level 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        
        # Nested decoder - Level 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        
        # Nested decoder - Level 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], dim=1))
        
        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)


# Testing the model
if __name__ == "__main__":
    # Test without deep supervision
    model = UNetPlusPlus(in_channels=3, base_ch=64, out_channels=1, deep_supervision=False)
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with deep supervision
    # model_ds = UNetPlusPlus(in_channels=3, base_ch=64, out_channels=1, deep_supervision=True)
    # outputs = model_ds(x)
    # print(f"\nWith deep supervision:")
    # for i, out in enumerate(outputs):
    #     print(f"Output {i+1} shape: {out.shape}")
    
    # # Count parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"\nTotal parameters: {total_params:,}")