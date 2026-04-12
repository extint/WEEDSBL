import os
import random

# Paths
dataset_root = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET"  # change this
rgb_dir = os.path.join(dataset_root, "rgb")
mask_dir = os.path.join(dataset_root, "masks")
nir_dir = os.path.join(dataset_root, "nir")
splits_dir = os.path.join(dataset_root, "splits")

# Create splits folder
os.makedirs(splits_dir, exist_ok=True)
random.seed(42) # Reproduceability
# Helper function to extract ID
def extract_id(filename):
    name = os.path.splitext(filename)[0]  # remove extension
    for prefix in ["rgb_", "mask_", "nir_"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

# Get IDs from rgb folder (assumes all folders are aligned)
ids = set()

for file in os.listdir(rgb_dir):
    if file.endswith(".png"):
        ids.add(extract_id(file))

# Optional: ensure corresponding files exist in all folders
valid_ids = []
for id_ in ids:
    rgb_file = os.path.join(rgb_dir, f"rgb_{id_}.png")
    mask_file = os.path.join(mask_dir, f"mask_{id_}.png")
    nir_file = os.path.join(nir_dir, f"nir_{id_}.png")

    if os.path.exists(rgb_file) and os.path.exists(mask_file) and os.path.exists(nir_file):
        valid_ids.append(id_)

# Shuffle for randomness
random.shuffle(valid_ids)

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

n = len(valid_ids)
train_end = int(n * train_ratio)
val_end = train_end + int(n * val_ratio)

train_ids = valid_ids[:train_end]
val_ids = valid_ids[train_end:val_end]
test_ids = valid_ids[val_end:]

# Function to write to file
def write_split(file_path, id_list):
    with open(file_path, "w") as f:
        for id_ in id_list:
            f.write(id_ + "\n")

# Write splits
write_split(os.path.join(splits_dir, "train.txt"), train_ids)
write_split(os.path.join(splits_dir, "val.txt"), val_ids)
write_split(os.path.join(splits_dir, "test.txt"), test_ids)

print("Done!")
print(f"Total samples: {n}")
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")