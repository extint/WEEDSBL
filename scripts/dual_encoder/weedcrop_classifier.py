# ===================== STAGE 2: CROP vs WEED TRAINING =====================
# Uses GT vegetation mask → trains only on vegetation pixels
# Labels:
#   0 = crop
#   1 = weed
#   255 = ignore (background)

import os
import cv2
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# ===================== DATASET =====================

class CropWeedDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_size: Tuple[int, int] = (512, 512),
        augment: bool = True
    ):
        self.root = root
        self.split = split
        self.target_h, self.target_w = target_size
        self.augment = augment and split == "train"

        self.rgb_dir   = os.path.join(root, "rgb")
        self.nir_dir   = os.path.join(root, "nir")
        self.mask_dir  = os.path.join(root, "masks")
        self.split_dir = os.path.join(root, "splits")

        split_file = os.path.join(self.split_dir, f"{split}.txt")
        with open(split_file, "r") as f:
            image_ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        for img_id in image_ids:
            self.samples.append({
                "rgb":  os.path.join(self.rgb_dir,  f"rgb_{img_id}.png"),
                "nir":  os.path.join(self.nir_dir,  f"nir_{img_id}.png"),
                "mask": os.path.join(self.mask_dir, f"mask_{img_id}.png"),
            })

        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _read_rgb(self, path):
        bgr = cv2.imread(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_nir(self, path):
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr

    def _read_mask(self, path):
        """
        Converts:
            0   → background
            128 → crop
            255 → weed

        Output:
            0 = crop
            1 = weed
            255 = ignore background
        """
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        cw_mask = np.zeros_like(m, dtype=np.uint8)

        cw_mask[m == 1] = 0   # crop
        cw_mask[m == 2] = 1   # weed
        cw_mask[m == 0]   = 255 # ignore

        return cw_mask

    def _scale(self, img):
        return img.astype(np.float32) / 255.0

    def _resize(self, rgb, nir, mask):
        rgb  = cv2.resize(rgb,  (self.target_w, self.target_h))
        nir  = cv2.resize(nir,  (self.target_w, self.target_h))
        mask = cv2.resize(mask, (self.target_w, self.target_h),
                          interpolation=cv2.INTER_NEAREST)
        return rgb, nir, mask

    def __getitem__(self, idx):
        item = self.samples[idx]

        rgb  = self._read_rgb(item["rgb"])
        nir  = self._read_nir(item["nir"])
        mask = self._read_mask(item["mask"])

        rgb, nir, mask = self._resize(rgb, nir, mask)

        rgb = self._scale(rgb)
        nir = self._scale(nir)

        rgb = (rgb - self.rgb_mean) / self.rgb_std

        rgb_t  = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        nir_t  = torch.from_numpy(nir[None, ...]).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))

        return {
            "rgb": rgb_t,
            "nir": nir_t,
            "mask": mask_t
        }


def create_dataloaders(root, batch_size=4, num_workers=4, size=(512,512)):
    train_ds = CropWeedDataset(root, "train", size, True)
    val_ds   = CropWeedDataset(root, "val",   size, False)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )


# ===================== MODEL =====================
from dual_encoder.updated_architecture import DualEncoderAFFNet


# ===================== METRICS =====================

def compute_metrics(pred, target):
    valid = target != 255

    pred = pred[valid]
    target = target[valid]

    ious = {}
    accs = {}

    for cls, name in zip([0,1], ["crop", "weed"]):
        pred_c = (pred == cls)
        tgt_c  = (target == cls)

        intersection = (pred_c & tgt_c).sum().float()
        union = pred_c.sum() + tgt_c.sum() - intersection

        if union == 0:
            iou = torch.tensor(0.0)
        else:
            iou = (intersection + 1e-6)/(union + 1e-6)

        # class-wise accuracy
        total = tgt_c.sum()
        if total == 0:
            acc = torch.tensor(0.0)
        else:
            acc = intersection / total

        ious[name] = iou.item()
        accs[name] = acc.item()

    pixel_acc = (pred == target).float().mean().item()

    return ious, accs, pixel_acc


# ===================== TRAIN =====================

def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0
    total_iou = 0

    for batch in tqdm(loader):
        rgb = batch["rgb"].to(device)
        nir = batch["nir"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(rgb, nir)   # (B,2,H,W)
            loss = criterion(logits, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        total_loss += loss.item()
        ious, accs, pix_acc = compute_metrics(pred, mask)
        total_iou += (ious["crop"] + ious["weed"]) / 2
        
    print(f"Crop IoU: {ious['crop']:.4f}, Weed IoU: {ious['weed']:.4f}")
    print(f"Crop Acc: {accs['crop']:.4f}, Weed Acc: {accs['weed']:.4f}, Pixel Acc: {pix_acc:.4f}")
    return total_loss / len(loader), total_iou / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            nir = batch["nir"].to(device)
            mask = batch["mask"].to(device)

            logits = model(rgb, nir)
            loss = criterion(logits, mask)

            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            total_loss += loss.item()
            ious, accs, pix_acc = compute_metrics(pred, mask)
            total_iou += (ious["crop"] + ious["weed"]) / 2

        print(f"Crop IoU: {ious['crop']:.4f}, Weed IoU: {ious['weed']:.4f}")
        print(f"Crop Acc: {accs['crop']:.4f}, Weed Acc: {accs['weed']:.4f}, Pixel Acc: {pix_acc:.4f}")
    return total_loss / len(loader), total_iou / len(loader)


# ===================== MAIN =====================



def main():
    DATA_ROOT = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = create_dataloaders(DATA_ROOT)

    model = DualEncoderAFFNet(
        rgb_variant='small',
        nir_base_ch=20,
        num_classes=2,   # crop vs weed
        embed_dim=96
    ).to(DEVICE)

    class_weights = torch.tensor([1.0, 3.0]).to(DEVICE)  # weed heavier
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20)

    scaler = GradScaler()

    best_iou = 0

    for epoch in range(1, 101):
        print(f"\nEpoch {epoch}")

        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, DEVICE
        )

        val_loss, val_iou = validate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f}")
        print(f"Val   Loss: {val_loss:.4f} IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_crop_weed.pth")
            print("Saved best model")


if __name__ == "__main__":
    main()
