# benchmark_unet_pth_vs_onnx.py
import os
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import onnxruntime as ort

# -------------------- Model defs (from your code) --------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv1(out)
        attn = self.sigmoid(attn)
        return x * attn

class UNet_SA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_SA, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bridge = self.conv_block(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa_bridge = SpatialAttention()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1 = self.sa1(enc1)

        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc2 = self.sa2(enc2)

        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc3 = self.sa3(enc3)

        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc4 = self.sa4(enc4)

        bridge = self.bridge(F.max_pool2d(enc4, 2))
        bridge = self.sa_bridge(bridge)

        dec4 = self.up4(bridge)
        dec4 = torch.cat([enc4, dec4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.dec1(dec1)

        out = self.out(dec1)
        return out

# -------------------- Utilities --------------------
def list_images(folder: Path, limit: int = 5) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
    return imgs[:limit]

def load_and_preprocess(img_path: Path, size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    # Simple preprocessing: resize, to float32 [0,1], CHW
    img = Image.open(img_path).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return chw

def save_output_tensor(tensor: np.ndarray, out_path: Path):
    # For demo: clamp to [0,1] and save first 3 channels as RGB
    x = tensor
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] >= 3:
        x = x[:3]
    else:
        # tile channels if fewer than 3
        x = np.tile(x[:1], (3, 1, 1))
    x = np.clip(x, 0.0, 1.0)
    img = np.transpose(x, (1, 2, 0)) * 255.0
    Image.fromarray(img.astype(np.uint8)).save(out_path)

# -------------------- Benchmark routines --------------------
def benchmark_pytorch(pth_path: Path, image_paths: List[Path], device: str, save_dir: Path):
    # Try to load full model first
    try:
        model = torch.load(str(pth_path), map_location=device)
        # If this is a state_dict, the attribute keys won’t be module methods; detect naively
        if not hasattr(model, 'forward'):
            raise ValueError("Loaded object is not a torch.nn.Module")
    except Exception:
        # Fallback to state_dict with known architecture
        model = torch.load("/home/vjti-comp/Downloads/unet_best_model.pth", map_location=device, weights_only=False)
        # model.load_state_dict(state)
        # model.eval().to(device)
        # model = UNet_SA(in_channels=3, out_channels=3)
        # state = torch.load(str(pth_path), map_location=device)
        # if 'state_dict' in state:
        #     state = state['state_dict']
        #     # Strip possible module. prefixes
        #     new_state = {k.replace("module.", ""): v for k, v in state.items()}
        #     state = new_state
        # model.load_state_dict(state, strict=False)

    model.eval().to(device)

    # Warm-up
    dummy = torch.randn(1, 3, 640, 640, device=device)
    with torch.inference_mode():
        _ = model(dummy)

    times = []
    outputs = []
    with torch.inference_mode():
        for i, img_path in enumerate(image_paths):
            inp = load_and_preprocess(img_path)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)

            # Time per image
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(inp_t)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            times.append(dt)
            outputs.append(out.detach().cpu().numpy())

            save_output_tensor(outputs[-1], save_dir / f"pth_out_{i}.png")

    return times, outputs

def benchmark_onnx(onnx_path: Path, image_paths: List[Path], device_prefers_cuda: bool, save_dir: Path):
    # Choose providers
    avail = ort.get_available_providers()
    providers = []
    if device_prefers_cuda and 'CUDAExecutionProvider' in avail:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    # Resolve names
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Warm-up
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
    _ = sess.run([out_name], {in_name: dummy})

    times = []
    outputs = []
    for i, img_path in enumerate(image_paths):
        inp = load_and_preprocess(img_path)
        inp_b = np.expand_dims(inp, axis=0).astype(np.float32)

        t0 = time.perf_counter()
        out = sess.run([out_name], {in_name: inp_b})[0]
        dt = time.perf_counter() - t0

        times.append(dt)
        outputs.append(out)

        save_output_tensor(outputs[-1], save_dir / f"onnx_out_{i}.png")

    return times, outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--onnx', type=str, required=True, help='Path to .onnx model file')
    parser.add_argument('--images', type=str, required=True, help='Folder with at least 5 images')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device for PyTorch model')
    parser.add_argument('--count', type=int, default=5, help='How many images to benchmark')
    parser.add_argument('--outdir', type=str, default='bench_outputs', help='Where to save outputs')
    args = parser.parse_args()

    img_dir = Path(args.images)
    save_dir = Path(args.outdir)
    save_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(img_dir, args.count)
    if len(image_paths) < args.count:
        raise RuntimeError(f"Found only {len(image_paths)} images in {img_dir}")

    # PyTorch
    print("=== PyTorch .pth inference ===")
    pth_times, _ = benchmark_pytorch(Path(args.ckpt), image_paths, args.device, save_dir)
    for i, t in enumerate(pth_times):
        print(f"[PyTorch] Image {i}: {t*1000:.2f} ms")
    print(f"[PyTorch] Avg over {len(pth_times)}: {np.mean(pth_times)*1000:.2f} ms")

    # ONNX Runtime
    print("\n=== ONNX Runtime .onnx inference ===")
    prefer_cuda = (args.device == 'cuda')
    onnx_times, _ = benchmark_onnx(Path(args.onnx), image_paths, prefer_cuda, save_dir)
    for i, t in enumerate(onnx_times):
        print(f"[ONNX] Image {i}: {t*1000:.2f} ms")
    print(f"[ONNX] Avg over {len(onnx_times)}: {np.mean(onnx_times)*1000:.2f} ms")

    # Summary
    print("\n=== Summary ===")
    print(f"PyTorch avg: {np.mean(pth_times)*1000:.2f} ms")
    print(f"ONNX avg:   {np.mean(onnx_times)*1000:.2f} ms")

if __name__ == "__main__":
    main()
