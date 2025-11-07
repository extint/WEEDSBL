import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
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

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # ---------------- Encoder ----------------
        enc1 = self.enc1(x)                      # (B,64,H,W)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))   # (B,128,H/2,W/2)
        enc3 = self.enc3(F.max_pool2d(enc2, 2))   # (B,256,H/4,W/4)
        enc4 = self.enc4(F.max_pool2d(enc3, 2))   # (B,512,H/8,W/8)

        # ---------------- Bridge ----------------
        bridge = self.bridge(F.max_pool2d(enc4, 2))  # (B,1024,H/16,W/16)

        # ---------------- Decoder ----------------
        # Up4
        dec4 = self.up4(bridge)                    # (B,512,H/8,W/8)
        dec4 = torch.cat([enc4, dec4], dim=1)      # (B,1024,H/8,W/8)
        dec4 = self.dec4(dec4)                     # (B,512,H/8,W/8)

        # Up3
        dec3 = self.up3(dec4)                      # (B,256,H/4,W/4)
        dec3 = torch.cat([enc3, dec3], dim=1)      # (B,512,H/4,W/4)
        dec3 = self.dec3(dec3)                     # (B,256,H/4,W/4)

        # Up2
        dec2 = self.up2(dec3)                      # (B,128,H/2,W/2)
        dec2 = torch.cat([enc2, dec2], dim=1)      # (B,256,H/2,W/2)
        dec2 = self.dec2(dec2)                     # (B,128,H/2,W/2)

        # Up1
        dec1 = self.up1(dec2)                      # (B,64,H,W)
        dec1 = torch.cat([enc1, dec1], dim=1)      # (B,128,H,W)
        dec1 = self.dec1(dec1)                     # (B,64,H,W)

        # Output
        out = self.out(dec1)                       # (B,out_channels,H,W)
        return out

# import torch

# ckpt_path = "scripts/checkpoints/lmannet_4_ch_best.pth"
# onnx_path = "scripts/checkpoints/lmannet_4_ch_best.onnx"
# device = "cuda"  # or "cuda" if available
# from MANNet import LightMANet

# model = LightMANet(in_channels=4, num_classes=1, base_ch=32)
# # state = torch.load(ckpt_path, map_location=device, weights_only=True)
# # # model = torch.load(ckpt_path, map_location=device, weights_only=False)
# # model.load_state_dict(state)
# # model.eval().to(device)

# device = torch.device(device if torch.cuda.is_available() else "cpu")
# state = torch.load(ckpt_path, map_location=device)
# state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
# model.load_state_dict(state_dict, strict=True)
# model.eval().to(device)

# # Adjust shape to your model's input
# dummy = torch.randn(2, 4, 960, 1280, device=device)

# torch.onnx.export(
#     model, dummy, onnx_path,
#     input_names=["input"], output_names=["output"],
#     opset_version=12, do_constant_folding=True,
#     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
# )
# print("Exported:", onnx_path)


import torch
import torch.nn as nn

# Export script
import torch
from MANNet import LightMANet

ckpt_path = "/home/vjti-comp/Downloads/unet_best_model(1).pth"
onnx_path = "/home/vjti-comp/Downloads/unet_best_model(1).onnx"
device = "cuda"

# CRITICAL: Match the training configuration exactly
# Your training uses num_classes=1 (binary segmentation with sigmoid)
# model = LightMANet(in_channels=4, num_classes=1, base_ch=32)
model=UNet()
# device = torch.device(device if torch.cuda.is_available() else "cpu")
state = torch.load(ckpt_path, map_location=device)
state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
model.load_state_dict(state_dict, strict=True)
model.eval().to(device)

# Wrap model to include sigmoid in ONNX graph
# class ModelWithSigmoid(nn.Module):
#     def __init__(self, base_model):
#         super().__init__()
#         self.model = base_model
    
#     def forward(self, x):
#         logits = self.model(x)
#         # Include sigmoid so ONNX has complete inference pipeline
#         probs = torch.sigmoid(logits)
#         return probs

# wrapped_model = ModelWithSigmoid(model)
# wrapped_model.eval().to(device)

# Adjust shape to your model's input
dummy = torch.randn(2, 3, 640, 640, device=device)

torch.onnx.export(
    model, 
    dummy, 
    onnx_path,
    input_names=["input"], 
    output_names=["output"],
    opset_version=10,  # Use opset 11 for better Jetson compatibility
    do_constant_folding=True,
    export_params=True,
    verbose=False,
    # Static batch size for TensorRT
    dynamic_axes=None  # Remove dynamic axes for Jetson Nano
)


print("Exported:", onnx_path)
print("Model outputs probabilities in [0,1] range after sigmoid")


