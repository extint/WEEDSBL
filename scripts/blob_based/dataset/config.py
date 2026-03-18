import os

DATA_ROOT = "/home/vjtiadmin/Desktop/BTechGroup/SUGARBEETS_MIXED_DATASET"

RGB_DIR = os.path.join(DATA_ROOT, "rgb")
NIR_DIR = os.path.join(DATA_ROOT, "nir")
MASK_DIR = os.path.join(DATA_ROOT, "masks")
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")

BLOB_SIZE = 64          # CNN input size
# MIN_BLOB_AREA = 100     # remove tiny noise blobs
NUM_CLASSES = 2         # crop / weed

NDVI_THRESH = 0.25     # tune this
MIN_BLOB_AREA = 100

DEVICE = "cuda"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
