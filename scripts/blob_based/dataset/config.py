import os

DATA_ROOT = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET"

RGB_DIR = os.path.join(DATA_ROOT, "rgb")
NIR_DIR = os.path.join(DATA_ROOT, "nir")
MASK_DIR = os.path.join(DATA_ROOT, "masks")
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")

BLOB_SIZE = 64          # CNN input size
# MIN_BLOB_AREA = 100     # remove tiny noise blobs
NUM_CLASSES = 2         # crop / weed

NDVI_THRESH = 0.4    # tune this
MIN_BLOB_AREA = 100

DEVICE = "cuda"
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
