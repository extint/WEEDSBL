import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
import random

DATASET_ROOT = "/home/vjti-comp/Downloads/SUGARBEETS_MIXED_DATASET"
UPDATED_DATASET_ROOT = "/home/vjti-comp/Downloads/SUGARBEETS_REDUCED_DATASET"

MASK_DIR = os.path.join(DATASET_ROOT, "masks")

WEED_LABEL = 2
TOP_K = 400
ZERO_WEED_K = 50

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def compute_weed_percentage(mask_path):
    mask = np.array(Image.open(mask_path))
    total_pixels = mask.size
    weed_pixels = np.sum(mask == WEED_LABEL)
    return weed_pixels / total_pixels


def get_all_image_ids():
    splits_dir = os.path.join(DATASET_ROOT, "splits")
    image_ids = []

    for split_file in ["train.txt", "val.txt", "test.txt"]:
        with open(os.path.join(splits_dir, split_file), "r") as f:
            image_ids.extend([line.strip() for line in f.readlines()])

    # Remove duplicates if any
    image_ids = list(set(image_ids))
    return image_ids


def create_global_subset():
    image_ids = get_all_image_ids()
    weed_stats = []

    print("Computing weed percentages globally...")
    for img_id in tqdm(image_ids):
        mask_path = os.path.join(MASK_DIR, f"mask_{img_id}.png")
        weed_percent = compute_weed_percentage(mask_path)
        weed_stats.append((img_id, weed_percent))

    weed_stats.sort(key=lambda x: x[1], reverse=True)

    top_k = weed_stats[:TOP_K]
    zero_weed = [x for x in weed_stats if x[1] == 0.0][:ZERO_WEED_K]

    final_subset = top_k + zero_weed
    final_subset = list({x[0]: x for x in final_subset}.values())

    print("Total selected images:", len(final_subset))

    return [x[0] for x in final_subset]


def split_dataset(image_ids):
    random.seed(42)
    random.shuffle(image_ids)

    total = len(image_ids)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_ids = image_ids[:train_end]
    val_ids = image_ids[train_end:val_end]
    test_ids = image_ids[val_end:]

    return train_ids, val_ids, test_ids


def save_split(split_name, image_ids):
    split_dir = os.path.join(UPDATED_DATASET_ROOT, "splits")
    os.makedirs(split_dir, exist_ok=True)

    with open(os.path.join(split_dir, f"{split_name}.txt"), "w") as f:
        for img_id in image_ids:
            f.write(img_id + "\n")


def copy_files(image_ids):
    rgb_src = os.path.join(DATASET_ROOT, "rgb")
    nir_src = os.path.join(DATASET_ROOT, "nir")
    mask_src = os.path.join(DATASET_ROOT, "masks")

    rgb_dst = os.path.join(UPDATED_DATASET_ROOT, "rgb")
    nir_dst = os.path.join(UPDATED_DATASET_ROOT, "nir")
    mask_dst = os.path.join(UPDATED_DATASET_ROOT, "masks")

    os.makedirs(rgb_dst, exist_ok=True)
    os.makedirs(nir_dst, exist_ok=True)
    os.makedirs(mask_dst, exist_ok=True)

    print("Copying files...")
    for img_id in tqdm(image_ids):
        shutil.copy2(os.path.join(rgb_src, f"rgb_{img_id}.png"),
                     os.path.join(rgb_dst, f"rgb_{img_id}.png"))

        shutil.copy2(os.path.join(nir_src, f"nir_{img_id}.png"),
                     os.path.join(nir_dst, f"nir_{img_id}.png"))

        shutil.copy2(os.path.join(mask_src, f"mask_{img_id}.png"),
                     os.path.join(mask_dst, f"mask_{img_id}.png"))


if __name__ == "__main__":

    # Step 1: Create global subset
    subset_ids = create_global_subset()

    # Step 2: Split
    train_ids, val_ids, test_ids = split_dataset(subset_ids)

    print("Train:", len(train_ids))
    print("Val:", len(val_ids))
    print("Test:", len(test_ids))

    # Step 3: Save splits
    save_split("train", train_ids)
    save_split("val", val_ids)
    save_split("test", test_ids)

    # Step 4: Copy files
    copy_files(subset_ids)

    print("Dataset reduction complete.")
