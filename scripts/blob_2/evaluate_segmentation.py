import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ================= CONFIG =================

DATASET_ROOT = "/home/vjti-comp/Downloads/SUGARBEETS_REDUCED_DATASET"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEED_LABEL = 2
NDVI_THRESHOLD = 0.3
MODEL_PATH = "vegetation_model.pth"  # trained UNet
# os.makedirs(MODEL_PATH, exist_ok=True)

# ==========================================


# ---------- NDVI ----------
def compute_ndvi(rgb, nir):
    red = rgb[:, :, 0].astype(np.float32)
    nir = nir.astype(np.float32)

    return (nir - red) / (nir + red + 1e-6)


# ---------- Metrics ----------
class SegmentationMetrics:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.conf_matrix = np.zeros((num_classes, num_classes))

    def update(self, gt, pred):
        mask = (gt >= 0) & (gt < self.num_classes)
        hist = np.bincount(
            self.num_classes * gt[mask].astype(int) + pred[mask].astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.conf_matrix += hist

    def compute(self):
        cm = self.conf_matrix

        # IoU
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = intersection / (union + 1e-6)
        mean_iou = np.nanmean(iou)

        # Dice
        dice = (2 * intersection) / (cm.sum(axis=1) + cm.sum(axis=0) + 1e-6)

        # Precision
        precision = intersection / (cm.sum(axis=0) + 1e-6)

        # Recall
        recall = intersection / (cm.sum(axis=1) + 1e-6)

        # Pixel accuracy
        pixel_acc = intersection.sum() / cm.sum()

        # Class-wise pixel accuracy
        class_pixel_acc = intersection / (cm.sum(axis=1) + 1e-6)

        return {
            "IoU_per_class": iou,
            "Mean_IoU": mean_iou,
            "Dice_per_class": dice,
            "Precision_per_class": precision,
            "Recall_per_class": recall,
            "Pixel_Accuracy": pixel_acc,
            "Classwise_Pixel_Accuracy": class_pixel_acc,
        }


# ---------- Load Model ----------
def load_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ---------- Evaluation ----------
def evaluate_split(split_name, model=None, use_ndvi=False):

    split_file = os.path.join(DATASET_ROOT, "splits", f"{split_name}.txt")
    rgb_dir = os.path.join(DATASET_ROOT, "rgb")
    nir_dir = os.path.join(DATASET_ROOT, "nir")
    mask_dir = os.path.join(DATASET_ROOT, "masks")

    with open(split_file) as f:
        ids = [line.strip() for line in f.readlines()]

    metrics = SegmentationMetrics(num_classes=2)

    for img_id in tqdm(ids, desc=f"Evaluating {split_name}"):

        rgb = np.array(Image.open(os.path.join(rgb_dir, f"rgb_{img_id}.png")))
        nir = np.array(Image.open(os.path.join(nir_dir, f"nir_{img_id}.png")))
        mask = np.array(Image.open(os.path.join(mask_dir, f"mask_{img_id}.png")))

        # Binary GT
        gt = (mask != 0).astype(np.uint8)

        # Prediction
        if use_ndvi:
            ndvi = compute_ndvi(rgb, nir)
            pred = (ndvi > NDVI_THRESHOLD).astype(np.uint8)

        else:
            input_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                output = torch.sigmoid(output)
                pred = (output > 0.5).float()

            pred = pred.squeeze().cpu().numpy().astype(np.uint8)

        metrics.update(gt, pred)

    return metrics.compute()


# ================= MAIN =================

if __name__ == "__main__":

    print("----- NDVI BASELINE -----")
    for split in ["train", "val"]:
        results = evaluate_split(split, use_ndvi=True)
        print(f"\nNDVI Results ({split})")
        for k, v in results.items():
            print(k, ":", v)

    print("\n\n----- SEGMENTATION MODEL -----")
    model = load_model()

    for split in ["train", "val"]:
        results = evaluate_split(split, model=model, use_ndvi=False)
        print(f"\nModel Results ({split})")
        for k, v in results.items():
            print(k, ":", v)
