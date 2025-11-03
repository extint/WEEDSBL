# single_image_infer.py
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

# try to import common model names from your file
try:
    from new_main import LightMANet, UNet
except Exception:
    LightMANet = None
    UNet = None

def make_model(arch_name="auto", device="cpu"):
    arch = arch_name.lower()
    if arch in ("lightman", "lightmanet") and LightMANet is not None:
        m = LightMANet(in_channels=3, num_classes=1, base_ch=32)
    elif arch in ("unet",) and UNet is not None:
        m = UNet(in_channels=3, base_ch=4, out_channels=1)
    elif arch == "auto":
        # try lightman first then unet
        if LightMANet is not None:
            try:
                m = LightMANet(in_channels=3, num_classes=1, base_ch=32)
            except Exception:
                m = None
        else:
            m = None
        if m is None and UNet is not None:
            m = UNet(in_channels=3, base_ch=4, out_channels=1)
    else:
        raise RuntimeError("Unknown arch or model class not found in new_main.py")
    if m is None:
        raise RuntimeError("Failed to instantiate a model. Edit the script to match your model class.")
    return m.to(device)

IMG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # converts to [0,1] float tensor (C,H,W)
])

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model

def save_mask_and_overlay(prob_map, pil_img, out_dir, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    # binary mask (0/255)
    mask = (prob_map >= threshold).astype("uint8") * 255
    mask_im = Image.fromarray(mask)
    mask_path = Path(out_dir) / "mask.png"
    mask_im.save(mask_path)

    # overlay: red mask on original
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    color_mask = Image.fromarray(np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=2).astype("uint8"))
    overlay = Image.blend(pil_img, color_mask, alpha=0.45)
    overlay_path = Path(out_dir) / "overlay.png"
    overlay.save(overlay_path)
    return str(mask_path), str(overlay_path)

def infer_image(student_ckpt, image_path, out_dir="out", device="cuda", arch="auto", threshold=0.5):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = make_model(arch, device=device)
    model = load_checkpoint(model, student_ckpt, device)

    pil = Image.open(image_path).convert("RGB")
    t = IMG_TRANSFORM(pil).unsqueeze(0).to(device)  # (1,3,H,W)

    with torch.no_grad():
        logits = model(t)              # expect (1,1,H,W)
        probs = torch.sigmoid(logits)  # (1,1,H,W)
        prob_map = probs[0,0].cpu().numpy()

    mask_path, overlay_path = save_mask_and_overlay(prob_map, pil, out_dir, threshold=threshold)
    print("Saved mask ->", mask_path)
    print("Saved overlay ->", overlay_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to student checkpoint (.pth)")
    p.add_argument("--img", required=True, help="path to RGB image file (jpg/png/tif)")
    p.add_argument("--out", default="./single_infer_out", help="output folder")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--arch", default="auto", help="model arch: 'auto'|'lightman'|'unet' (optional)")
    p.add_argument("--threshold", type=float, default=0.5, help="binarization threshold")
    args = p.parse_args()

    infer_image(args.ckpt, args.img, out_dir=args.out, device=args.device, arch=args.arch, threshold=args.threshold)