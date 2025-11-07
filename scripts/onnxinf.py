import numpy as np
from PIL import Image
import onnxruntime as ort
import cv2

# User settings
ONNX_PATH = "scripts/checkpoints/lmannet_4_ch_best.onnx"
OUT_MASK_PATH = "/home/vjti-comp/Downloads/onnxinfmask.png"
OUT_MASK_COLOR_PATH = "/home/vjti-comp/Downloads/onnxinfmask_color.png"

# Model/input specs
IMG_HEIGHT = 960
IMG_WIDTH = 1280
IN_CHANNELS = 4
THRESHOLD = 0.5

# Providers (CPU only since you don't have CUDA/TensorRT on this machine)
USE_TRT = False
providers = ["CPUExecutionProvider"]

def _read_nir_as_gray(nir_input):
    if isinstance(nir_input, str):
        nir = cv2.imread(nir_input, cv2.IMREAD_UNCHANGED)
        if nir is None:
            raise FileNotFoundError(f"NIR not found: {nir_input}")
    else:
        nir = nir_input
        if nir is None:
            raise ValueError("NIR array is None")
    
    if nir.ndim == 3:
        nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)
    return nir

def _scale_to_unit(arr):
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    a = arr.astype(np.float32)
    return a / 255.0 if a.max() > 1.0 else a

def preprocess_rgbnir(bgr, nir_input, target_hw=(960, 1280)):
    """Pure NumPy preprocessing matching training exactly"""
    th, tw = target_hw
    
    # Resize BGR
    if bgr.shape[:2] != (th, tw):
        bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    
    # Read/resize NIR
    nir = _read_nir_as_gray(nir_input)
    if nir.shape[:2] != (th, tw):
        nir = cv2.resize(nir, (tw, th), interpolation=cv2.INTER_LINEAR)
    
    # BGR->RGB, normalize with ImageNet stats
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    
    # NIR to [0,1]
    nir = _scale_to_unit(nir).astype(np.float32)
    
    # Stack R,G,B,NIR -> (H, W, 4)
    x_hw4 = np.dstack([rgb, nir[..., None]])
    
    # Convert to NCHW
    x_chw = np.transpose(x_hw4, (2, 0, 1))
    x_nchw = np.expand_dims(x_chw, axis=0)
    
    return np.ascontiguousarray(x_nchw.astype(np.float32))

def run_inference(img_path, nir_path, device="cpu"):
    # Load image
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    
    # Preprocess
    x = preprocess_rgbnir(bgr, nir_path, target_hw=(IMG_HEIGHT, IMG_WIDTH))
    
    print(f"Input shape: {x.shape}, dtype: {x.dtype}, range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Create ONNX session
    sess = ort.InferenceSession(ONNX_PATH, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    print(f"Model input: {input_name}, output: {output_name}")
    print(f"Expected input shape: {sess.get_inputs()[0].shape}")
    print(f"Expected output shape: {sess.get_outputs()[0].shape}")
    
    # Run inference
    outputs = sess.run(None, {input_name: x})
    
    # Check output shape
    print(f"Raw output shape: {outputs[0].shape}")
    
    # Output should be probabilities [0,1] from sigmoid (1, 1, H, W)
    if len(outputs[0].shape) == 4:
        probs = outputs[0][0, 0]  # (H, W)
    else:
        probs = outputs[0][0]  # If already (H, W)
    
    print(f"Output probs shape: {probs.shape}")
    print(f"Probs range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Probs mean: {probs.mean():.3f}, median: {np.median(probs):.3f}")
    print(f"Probs distribution:")
    print(f"  < 0.1: {(probs < 0.1).sum()}")
    print(f"  0.1-0.3: {((probs >= 0.1) & (probs < 0.3)).sum()}")
    print(f"  0.3-0.5: {((probs >= 0.3) & (probs < 0.5)).sum()}")
    print(f"  0.5-0.7: {((probs >= 0.5) & (probs < 0.7)).sum()}")
    print(f"  0.7-0.9: {((probs >= 0.7) & (probs < 0.9)).sum()}")
    print(f"  > 0.9: {(probs > 0.9).sum()}")
    
    # Threshold to get binary mask
    pred = (probs > THRESHOLD).astype(np.uint8)
    
    print(f"Prediction unique values: {np.unique(pred)}, positives: {pred.sum()} / {pred.size}")
    
    # Save grayscale mask
    Image.fromarray(pred * 255).save(OUT_MASK_PATH)
    
    # Save probability heatmap for debugging
    prob_vis = (probs * 255).astype(np.uint8)
    Image.fromarray(prob_vis).save("/home/vjti-comp/Downloads/prob_heatmap.png")
    print("Saved probability heatmap: /home/vjti-comp/Downloads/prob_heatmap.png")
    
    # Create color visualization
    color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    color_mask[pred == 1] = [0, 255, 0]  # Green for weedy rice
    
    Image.fromarray(color_mask).save(OUT_MASK_COLOR_PATH)
    
    print("Saved:", OUT_MASK_PATH, "and", OUT_MASK_COLOR_PATH)

if __name__ == "__main__":
    run_inference(
        img_path='/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB/RGB/DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m.JPG',
        nir_path='/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB/Multispectral/DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m_NIR.TIF',
        device="cpu"
    )
