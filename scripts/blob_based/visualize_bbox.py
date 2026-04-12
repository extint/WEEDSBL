import json
import os
import random
import cv2
from blob_based.dataset.config import *
# ========= PATHS =========
JSON_PATH = "/home/vjti-comp/WEEDSBL/scripts/analysis/train_bboxes_multi_index_threshold.json"

NUM_SAMPLES = 10  # number of images to visualize

# =========================
with open(JSON_PATH, "r") as f:
    data = json.load(f)   # <-- this is a LIST

sample_items = random.sample(data, min(NUM_SAMPLES, len(data)))

for i, item in enumerate(sample_items):
    img_id = item["img_id"]
    bboxes = item["bboxes"]

    img_path = os.path.join(RGB_DIR, f"rgb_{img_id}.png")
    image = cv2.imread(img_path)

    if image is None:
        print(f"Image not found: {img_path}")
        continue

    for bbox_data in bboxes:
        x1, y1, x2, y2 = bbox_data["bbox"]
        label = bbox_data["label"]

        color = (0, 255, 0) if label == 0 else (0, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = "crop" if label == 0 else "weed"
        cv2.putText(image, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(f"/home/vjti-comp/WEEDSBL/scripts/analysis/bbox_{i}.png", image)
    # key = cv2.waitKey(0)

    # if key == 27:
    #     break

# cv2.destroyAllWindows()