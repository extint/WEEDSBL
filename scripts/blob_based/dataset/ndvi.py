import numpy as np

def compute_ndvi(rgb, nir):
    """
    rgb: HxWx3 (uint8)
    nir: HxW   (uint8 or uint16)
    """
    red = rgb[:,:,0].astype(np.float32)
    nir = nir.astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi


def ndvi_threshold(ndvi, thresh=0.2):
    """
    returns binary vegetation mask
    """
    return (ndvi > thresh).astype(np.uint8)
