#!/usr/bin/env python3
"""
Main training script for Weedy Rice UAV semantic segmentation.

Supports:
- RGB only (3 channels), or
- RGB + Multispectral (G, R, RE, NIR) = 7 channels fusion

Dataset layout (root):
  RGB/              -> RGB images (.JPG/.PNG)
  Multispectral/    -> 4 bands per RGB: *_G.TIF, *_R.TIF, *_RE.TIF, *_NIR.TIF
  Masks/            -> binary masks (PNG), 255 = weedy rice, 0 = background
  Metadata/         -> filename_mapping.csv, image_metadata.csv, train/val/test .txt (optional)

Masks are converted to {0,1} during loading; see rice_weed_data_loader.py.
"""

import argparse
import os
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2

import numpy as np

# Import the new dataset/dataloaders
# Ensure rice_weed_data_loader.py (with WeedyRiceUAVDataset and create_weedy_rice_rgbnir_dataloaders)
# is in the same directory or PYTHONPATH.
from rice_weed_data_loader import create_weedy_rice_rgbnir_dataloaders

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

# -------------------------
# Losses and metrics
# -------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = bce_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets.float())
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dims)
        union = probs.sum(dims) + targets.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean()
        return self.w * bce + (1 - self.w) * dice_loss

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ious, f1s = [], []
    for batch in loader:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()          # (B,H,W), values {0,1}
        logits = model(x)                               # (B,1,H,W)
        probs = torch.sigmoid(logits)                  # (B,1,H,W)
        preds = (probs > 0.5).long().squeeze(1)        # (B,H,W)

        inter = (preds & y).sum(dim=(1, 2))            # per-sample
        union = (preds | y).sum(dim=(1, 2)).clamp_min(1)
        iou = (inter.float() / union.float()).mean().item()
        ious.append(iou)

        tp = (preds & y).sum(dim=(1, 2)).float()
        fp = (preds & (1 - y)).sum(dim=(1, 2)).float()
        fn = ((1 - preds) & y).sum(dim=(1, 2)).float()
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f1   = (2 * prec * rec / (prec + rec + 1e-6)).mean().item()
        f1s.append(f1)

    return {"mIoU": float(np.mean(ious) if ious else 0.0),
            "F1":   float(np.mean(f1s) if f1s else 0.0)}

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None) -> float:
    model.train()
    epoch_loss = 0.0
    for batch in loader:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).unsqueeze(1)  # (B,1,H,W)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / max(1, len(loader))

import cv2
import numpy as np
import torch

def _read_nir_as_gray(nir_input):
    # Accept a path or a preloaded ndarray
    if isinstance(nir_input, str):
        nir = cv2.imread(nir_input, cv2.IMREAD_UNCHANGED)
        if nir is None:
            raise FileNotFoundError(f"NIR not found: {nir_input}")
    else:
        nir = nir_input
        if nir is None:
            raise ValueError("NIR array is None")

    # If 3-channel, convert to grayscale
    if nir.ndim == 3:
        nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)
    return nir

def _scale_to_unit(arr):
    # Handle uint16 NIR; otherwise assume uint8-like range
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    a = arr.astype(np.float32)
    # If values look like 0..255, scale accordingly
    return a / 255.0 if a.max() > 1.0 else a

def preprocess_rgbnir(bgr, nir_input, target_hw=(960, 1280)):
    # Resize RGB to target
    th, tw = target_hw
    if bgr.shape[:2] != (th, tw):
        bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LINEAR)

    # Read/resize NIR to match target
    nir = _read_nir_as_gray(nir_input)
    if nir.shape[:2] != (th, tw):
        nir = cv2.resize(nir, (tw, th), interpolation=cv2.INTER_LINEAR)

    # BGR->RGB, float32 [0,1], ImageNet normalize
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std

    # NIR to [0,1]
    nir = _scale_to_unit(nir)

    # Stack R,G,B,NIR -> CHW -> add batch dim
    x_hw4 = np.dstack([rgb, nir[..., None]])  # (H, W, 4) in R,G,B,NIR
    x_chw = x_hw4.transpose(2, 0, 1)          # (4, H, W)
    x = torch.from_numpy(x_chw).float().unsqueeze(0)  # (1, 4, H, W)
    return x

def run_inference(model, img_path, nir, ckpt_path, out_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    x = preprocess_rgbnir(bgr, nir,target_hw=(960, 1280)).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = (probs > 0.5).astype(np.uint8) * 255

    cv2.imwrite(out_path, mask)
    print("Saved:", out_path)
    return out_path

from MANNet import LightMANet

import torchvision.models as models
class MobileUNet4Ch(nn.Module):
    def __init__(self, in_channels=4, num_classes=1):
        super().__init__()
        # Adapt MobileNetV2 for 4-channel input
        mobilenet = models.mobilenet_v2(pretrained=True)
        # Replace first conv: 3->4 channels
        old_conv = mobilenet.features[0][0]
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, 
                           old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=False)
        # Initialize new conv weights
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        mobilenet.features[0][0] = new_conv
        
        self.backbone = mobilenet.features
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4, num_classes, 4, stride=2, padding=1),
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x

# model = MobileUNet4Ch(in_channels=4, num_classes=1)  # ~2M params, efficient


def main():
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--data_root", type=str, required=True,
    #                     # help="Dataset root containing RGB/, Multispectral/, Masks/, Metadata/ subfolders")
    # parser.add_argument("--use_multimodal", action="store_true",
    #                     help="Enable to use RGB+MS (7 channels). If omitted, use RGB only (3 channels).")
    # parser.add_argument("--height", type=int, default=960)
    # parser.add_argument("--width", type=int, default=1280)
    # parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    # parser.add_argument("--mixed_precision", action="store_true")
    # args = parser.parse_args()

    # # os.makedirs(args.ckpt_dir, exist_ok=True)
    # # # model = MobileUNet4Ch(in_channels=4, num_classes=1)  # ~2M params, efficient

    # # Dataloaders
    # train_loader, val_loader, test_loader = create_weedy_rice_rgbnir_dataloaders(
    #     data_root="/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB",
    #     use_rgbnir=True,  # 4 channels
    #     batch_size=8,
    #     num_workers=args.num_workers,
    #     # use_multimodal=args.use_multimodal,
    #     target_size=(args.height, args.width)
    # )

    # # # Model
    # # in_ch = 4 #if args.use_multimodal else 3
    # # model = UNet(in_channels=in_ch, base_ch=4, out_channels=1)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    # # # Optimizer, loss
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # loss_fn = BCEDiceLoss(bce_weight=0.5)
    # scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None

    # best_miou = 0.0
    # for epoch in range(1, args.epochs + 1):
    #     train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
    #     metrics = evaluate(model, val_loader, device)
    #     print(f"[Epoch {epoch:03d}] Loss: {train_loss:.4f} | Val mIoU: {metrics['mIoU']:.4f} | Val F1: {metrics['F1']:.4f}")

    #     # Save best
    #     if metrics["mIoU"] > best_miou:
    #         best_miou = metrics["mIoU"]
    #         ckpt_path = os.path.join(args.ckpt_dir, f"unet_4_ch_best.pth")
    #         torch.save({"model": model.state_dict(),
    #                     "epoch": epoch,
    #                     "miou": best_miou}, ckpt_path)
    #         print(f"Saved checkpoint: {ckpt_path}")

    # # Final test
    # test_metrics = evaluate(model, test_loader, device)
    # print(f"[Test] mIoU: {test_metrics['mIoU']:.4f} | F1: {test_metrics['F1']:.4f}")

    model = LightMANet(in_channels=4, num_classes=1, base_ch=32)  # ~3M params
    out = run_inference(
        model = model,
        # img_path="/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB/RGB/DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m.JPG",
        img_path='/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB/RGB/DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m.JPG',
        nir='/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB/Multispectral/DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m_NIR.TIF',
        ckpt_path="scripts/checkpoints/unet_4_ch_best.pth",
        out_path="./output_mask.png",
        device="cuda",
    )
    print("Saved:", out)

if __name__ == "__main__":
    main()


#  