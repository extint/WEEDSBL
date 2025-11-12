import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from datetime import datetime

# ======================
# CONFIG
# ======================
TEST_IMAGES_DIR = "test/images"
TEST_MASKS_DIR  = "test/Morphed_Images"
ONNX_PATH       = "model_onnx/PSPNet.onnx"
OUTPUT_EXCEL    = "metrics/PSPNet_Metrics.xlsx"

IMG_HEIGHT, IMG_WIDTH = 640, 640
NUM_CLASSES = 3
CLASS_LABELS = ["Background", "Crop", "Weed"]
MODEL_NAME = "PSPNet_ONNX"
DATASET_NAME = "CWD-TestSet"

# ======================
# PREPROCESSING
# ======================
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr[np.newaxis, :, :, :]     # BCHW

def preprocess_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
    mask_np = np.array(mask, dtype=np.int64)

    # normalize grayscale masks like [0,128,255] → [0,1,2]
    unique_vals = np.unique(mask_np)
    if np.max(unique_vals) > NUM_CLASSES - 1:
        val_map = {v: i for i, v in enumerate(sorted(unique_vals))}
        mask_np = np.vectorize(val_map.get)(mask_np)
    return mask_np

# ======================
# METRIC FUNCTIONS
# ======================
def calculate_iou(pred_mask, gt_mask, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        inter = np.logical_and(pred_mask == cls, gt_mask == cls).sum()
        union = np.logical_or(pred_mask == cls, gt_mask == cls).sum()
        iou = inter / union if union > 0 else 0
        iou_per_class.append(iou)
    return iou_per_class

def calculate_dice(pred_mask, gt_mask, num_classes):
    dice_per_class = []
    for cls in range(num_classes):
        inter = np.logical_and(pred_mask == cls, gt_mask == cls).sum()
        denom = (np.sum(pred_mask == cls) + np.sum(gt_mask == cls))
        dice = 2.0 * inter / denom if denom > 0 else 0
        dice_per_class.append(dice)
    return dice_per_class

def calculate_precision_recall(pred_mask, gt_mask, num_classes):
    precisions, recalls = [], []
    for cls in range(num_classes):
        TP = np.logical_and(pred_mask == cls, gt_mask == cls).sum()
        FP = np.logical_and(pred_mask == cls, gt_mask != cls).sum()
        FN = np.logical_and(pred_mask != cls, gt_mask == cls).sum()
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls

def pixel_accuracy(pred_mask, gt_mask):
    return (pred_mask == gt_mask).sum() / gt_mask.size

# ======================
# MAIN EVALUATION
# ======================
def evaluate_onnx_model():
    print(f"\n Evaluating {MODEL_NAME} on {DATASET_NAME}")

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(".jpg")])
    iou_sum = np.zeros(NUM_CLASSES)
    dice_sum = np.zeros(NUM_CLASSES)
    prec_sum = np.zeros(NUM_CLASSES)
    rec_sum = np.zeros(NUM_CLASSES)
    acc_sum = 0.0
    total = 0

    for img_name in tqdm(images, desc="Testing"):
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        mask_path = os.path.join(TEST_MASKS_DIR, img_name.replace(".jpg", "_morphed.png"))
        if not os.path.exists(mask_path):
            continue

        inp = preprocess_image(img_path)
        gt_mask = preprocess_mask(mask_path)
        pred_logits = sess.run(None, {input_name: inp})[0]
        pred_mask = np.argmax(pred_logits, axis=1).squeeze(0)

        iou_sum += np.array(calculate_iou(pred_mask, gt_mask, NUM_CLASSES))
        dice_sum += np.array(calculate_dice(pred_mask, gt_mask, NUM_CLASSES))
        prec, rec = calculate_precision_recall(pred_mask, gt_mask, NUM_CLASSES)
        prec_sum += np.array(prec)
        rec_sum += np.array(rec)
        acc_sum += pixel_accuracy(pred_mask, gt_mask)
        total += 1

    if total == 0:
        raise RuntimeError("No test samples found!")

    # Average across dataset
    iou_avg = iou_sum / total
    dice_avg = dice_sum / total
    prec_avg = prec_sum / total
    rec_avg = rec_sum / total
    acc_avg = acc_sum / total

    mean_iou = np.mean(iou_avg)
    mean_dice = np.mean(dice_avg)
    mean_prec = np.mean(prec_avg)
    mean_rec = np.mean(rec_avg)
    mean_acc = acc_avg
    mean_jaccard = mean_iou  # Jaccard index = IoU

    print("\n Per-class metrics:")
    for i, cls in enumerate(CLASS_LABELS):
        print(f"{cls:>12s}: IoU={iou_avg[i]*100:.2f}%, Dice={dice_avg[i]*100:.2f}%, "
              f"Prec={prec_avg[i]*100:.2f}%, Rec={rec_avg[i]*100:.2f}%")

    print("\n Summary:")
    print(f"Mean IoU: {mean_iou*100:.2f}%")
    print(f"Mean Dice: {mean_dice*100:.2f}%")
    print(f"Mean Precision: {mean_prec*100:.2f}%")
    print(f"Mean Recall: {mean_rec*100:.2f}%")
    print(f"Mean Accuracy: {mean_acc*100:.2f}%")

    # ======================
    # Log to Excel
    # ======================
    new_row = {
        "Model": MODEL_NAME,
        "Dataset": DATASET_NAME,
        "Mean IoU": round(mean_iou * 100, 2),
        "Mean Dice": round(mean_dice * 100, 2),
        "Mean Jaccard": round(mean_jaccard * 100, 2),
        "Mean Precision": round(mean_prec * 100, 2),
        "Mean Recall": round(mean_rec * 100, 2),
        "Mean Accuracy": round(mean_acc * 100, 2),
        "Background IoU": round(iou_avg[0] * 100, 2),
        "Crop IoU": round(iou_avg[1] * 100, 2),
        "Weed IoU": round(iou_avg[2] * 100, 2),
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if os.path.exists(OUTPUT_EXCEL):
        df = pd.read_excel(OUTPUT_EXCEL)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n Logged metrics to {OUTPUT_EXCEL}")

# ======================
# RUN
# ======================
if __name__ == "__main__":
    evaluate_onnx_model()
