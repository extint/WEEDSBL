import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset.ndvi import compute_ndvi, ndvi_threshold
from scripts.blob_based.dataset.config import *
IMG_EXT='.png'
img_id = "1462264715_676331429"   # pick ONE image id

rgb = cv2.imread(f"{RGB_DIR}/rgb_{img_id}{IMG_EXT}")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
nir = cv2.imread(f"{NIR_DIR}/nir_{img_id}{IMG_EXT}", cv2.IMREAD_GRAYSCALE)

ndvi = compute_ndvi(rgb, nir)

mask = ndvi_threshold(ndvi, NDVI_THRESH)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("RGB"); plt.imshow(rgb); plt.axis("off")
plt.subplot(1,3,2); plt.title("NDVI"); plt.imshow(ndvi, cmap="jet"); plt.colorbar()
plt.subplot(1,3,3); plt.title(f"NDVI > {NDVI_THRESH}"); plt.imshow(mask, cmap="gray"); plt.axis("off")
plt.show()
