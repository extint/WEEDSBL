"""
benchmark_inference.py
======================
Modular inference benchmarking harness for weed-segmentation models.

Design goals
------------
* One shared DataLoader across every model — apples-to-apples comparison.
* Drop-in model registration: add a new model in ONE place (MODEL_REGISTRY).
* Industry-standard latency protocol: GPU warm-up → timed runs → statistics.
* Publishes every metric seen in segmentation papers:
    model size (M params / MB on disk), FLOPs, GPU memory, throughput (img/s),
    latency (mean ± std, P50/P95/P99), pixel-acc, mean-acc, mIoU, mDice,
    mPrecision, mRecall, mF1, per-class IoU/Dice/F1.
* Saves results to CSV + pretty console table.

Usage
-----
    python benchmark_inference.py                        # benchmark all registered models
    python benchmark_inference.py --models dual_encoder_mini   # one model only
    python benchmark_inference.py --batch-size 8 --num-batches 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Optional: fvcore / thop for FLOPs — install with  pip install fvcore  or
#           pip install thop.  We degrade gracefully if neither is present.
# ---------------------------------------------------------------------------
try:
    from fvcore.nn import FlopCountAnalysis
    _FVCORE = True
except ImportError:
    _FVCORE = False

try:
    from thop import profile as thop_profile
    _THOP = True
except ImportError:
    _THOP = False


# =============================================================================
# CONFIG  ── edit the paths / hyper-params here
# =============================================================================

@dataclass
class BenchConfig:
    # ── Dataset ───────────────────────────────────────────────────────────────
    data_root:   str             = "/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET"
    target_size: Tuple[int,int]  = (640, 640)
    batch_size:  int             = 4
    num_workers: int             = 4
    num_batches: int             = 50      # how many batches to time  (≥30 recommended)
    warmup_batches: int          = 10      # GPU warm-up before timing starts

    # ── Classes ───────────────────────────────────────────────────────────────
    num_classes:  int            = 3
    class_names:  List[str]      = field(default_factory=lambda: ["background", "crop", "weed"])

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir:  str             = "./benchmark_results"
    csv_name:    str             = "benchmark_results.csv"

    # ── Precision ─────────────────────────────────────────────────────────────
    use_amp:     bool            = True    # run inference under torch.autocast
    use_compile: bool            = False   # torch.compile (PyTorch ≥ 2.0)

    # ── Split to evaluate on ─────────────────────────────────────────────────
    split: str                   = "test"  # "val" | "test"

    # ── Visualisation ─────────────────────────────────────────────────────────
    num_vis_images: int          = 10      # images per model to visualise
    vis_dpi:        int          = 150     # output PNG DPI


CFG = BenchConfig()


# =============================================================================
# MODEL REGISTRY
# =============================================================================
# Each entry is a callable  () -> nn.Module  that returns the model already
# loaded with weights and ready for .eval().
# Add new models here — the rest of the script is untouched.

def _load_dual_encoder_mini() -> nn.Module:
    """Load DualEncoderMini from the trained checkpoint."""
    # ── import ─────────────────────────────────────────────────────────────
    from dual_encoder.dual_encoder_mini import DualEncoderMini          # noqa: F401

    ckpt_path = (
        "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/"
        "dual_encoder_mini_20260412_130647/checkpoints/best_model.pth"
    )

    model = DualEncoderMini(
        rgb_base_ch=16,
        nir_base_ch=8,
        aspp_ch=64,
        num_classes=CFG.num_classes,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)   # handle bare state-dict too
    model.load_state_dict(state)
    print(f"  [✓] dual_encoder_mini  ←  {ckpt_path}")
    return model


# ── Register models here ──────────────────────────────────────────────────────
#   key          : short name used on CLI and in CSV
#   value        : zero-arg callable returning nn.Module with loaded weights
MODEL_REGISTRY: Dict[str, Callable[[], nn.Module]] = {
    "dual_encoder_mini": _load_dual_encoder_mini,
    # "dual_encoder_full": _load_dual_encoder_full,  ← add future models here
    # "unet_baseline":     _load_unet_baseline,
}


# =============================================================================
# DATA LOADER  (single instance shared by every model)
# =============================================================================

def build_dataloader(cfg: BenchConfig) -> DataLoader:
    """Return the appropriate split DataLoader."""
    from dual_encoder.dual_encoder_data_loader_weedcrop import (
        create_dual_encoder_dataloaders,
    )

    train_loader, val_loader, test_loader = create_dual_encoder_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        target_size=cfg.target_size,
        mask_mode="multiclass",
    )
    split_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = split_map[cfg.split]
    print(f"[DataLoader] split={cfg.split}  samples={len(loader.dataset)}  "
          f"batches={len(loader)}  batch_size={cfg.batch_size}")
    return loader


# =============================================================================
# SEGMENTATION METRICS  (mirrors SegmentationMetrics from training)
# =============================================================================

class SegmentationMetrics:
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
        mask = (t >= 0) & (t < self.num_classes)
        np.add.at(self.conf_matrix, (t[mask], p[mask]), 1)

    def compute(self) -> Dict:
        cm  = self.conf_matrix.astype(np.float64)
        TP  = np.diag(cm)
        FP  = cm.sum(0) - TP
        FN  = cm.sum(1) - TP
        TN  = cm.sum() - (TP + FP + FN)
        eps = 1e-6

        iou       = TP / (TP + FP + FN + eps)
        dice      = 2 * TP / (2 * TP + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        spec      = TN  / (TN + FP + eps)

        results: Dict = {
            "pixel_accuracy": float(TP.sum() / (cm.sum() + eps)),
            "mean_accuracy":  float((TP / (cm.sum(1) + eps)).mean()),
            "mean_iou":       float(iou.mean()),
            "mean_dice":      float(dice.mean()),
            "mean_precision": float(precision.mean()),
            "mean_recall":    float(recall.mean()),
            "mean_f1":        float(dice.mean()),
        }
        for c, name in enumerate(self.class_names):
            results[f"iou_{name}"]         = float(iou[c])
            results[f"dice_{name}"]        = float(dice[c])
            results[f"precision_{name}"]   = float(precision[c])
            results[f"recall_{name}"]      = float(recall[c])
            results[f"f1_{name}"]          = float(dice[c])
            results[f"specificity_{name}"] = float(spec[c])
        return results


# =============================================================================
# MODEL-SIZE UTILS
# =============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_disk_size_mb(ckpt_path: Optional[str]) -> float:
    """Size of the .pth file on disk in MB, or -1 if path unknown."""
    if ckpt_path and Path(ckpt_path).exists():
        return Path(ckpt_path).stat().st_size / 1e6
    return -1.0


def compute_flops(
    model: nn.Module,
    dummy_inputs: Tuple[torch.Tensor, ...],
    device: str,
) -> float:
    """
    Returns GFLOPs (giga multiply-add ops).
    Tries fvcore first, then thop, else returns -1.
    """
    model_cpu = model.cpu().eval()
    inputs_cpu = tuple(x.cpu() for x in dummy_inputs)

    if _FVCORE:
        try:
            flops = FlopCountAnalysis(model_cpu, inputs_cpu)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            return flops.total() / 1e9
        except Exception:
            pass

    if _THOP:
        try:
            macs, _ = thop_profile(model_cpu, inputs=inputs_cpu, verbose=False)
            return macs / 1e9  # thop returns MACs; 1 MAC ≈ 2 FLOPs but convention varies
        except Exception:
            pass

    return -1.0


# =============================================================================
# LATENCY BENCHMARKING  (industry-standard protocol)
# =============================================================================

def _forward_pass(model: nn.Module, batch: Dict, device: str, use_amp: bool) -> torch.Tensor:
    """
    Run a single forward pass.  Returns the argmax prediction tensor.
    Handles models that return (logits, aux) or plain logits.
    """
    rgb  = batch["rgb"].to(device, non_blocking=True)
    nir  = batch["nir"].to(device, non_blocking=True)

    with torch.no_grad():
        if use_amp and device != "cpu":
            with torch.autocast(device):
                out = model(rgb, nir)
        else:
            out = model(rgb, nir)

    # Accept (logits, aux_logits) or plain logits
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return logits.argmax(dim=1)


def benchmark_model(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    cfg: BenchConfig,
    device: str,
    ckpt_path: Optional[str] = None,
) -> Dict:
    """
    Full benchmark for one model:
      1. Model-size metrics
      2. FLOPs (single image)
      3. GPU warm-up (cfg.warmup_batches batches, not timed)
      4. Timed run (cfg.num_batches batches)
         - per-batch GPU latency (CUDA events for sub-ms accuracy)
         - segmentation metrics accumulated over the full timed run
      5. GPU peak memory

    Returns a flat dict of all metrics.
    """
    print(f"\n{'='*70}")
    print(f"  Benchmarking: {model_name}")
    print(f"{'='*70}")

    model = model.to(device).eval()
    if cfg.use_compile:
        try:
            model = torch.compile(model)
            print("  [torch.compile] enabled")
        except Exception as e:
            print(f"  [torch.compile] skipped: {e}")

    # ── 1. Model size ─────────────────────────────────────────────────────────
    total_params, trainable_params = count_parameters(model)
    disk_mb = model_disk_size_mb(ckpt_path)

    # State-dict memory footprint (FP32 baseline)
    param_mb = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    ) / 1e6
    buffer_mb = sum(
        b.nelement() * b.element_size() for b in model.buffers()
    ) / 1e6

    print(f"  Params (total / trainable) : {total_params/1e6:.3f}M / {trainable_params/1e6:.3f}M")
    print(f"  In-memory size (params+buf): {param_mb+buffer_mb:.2f} MB")
    print(f"  Checkpoint on disk         : {disk_mb:.2f} MB" if disk_mb > 0 else "  Checkpoint on disk: N/A")

    # ── 2. FLOPs ──────────────────────────────────────────────────────────────
    dummy_rgb = torch.zeros(1, 3, *cfg.target_size)
    dummy_nir = torch.zeros(1, 1, *cfg.target_size)
    gflops = compute_flops(model, (dummy_rgb, dummy_nir), device)
    if gflops >= 0:
        print(f"  GFLOPs (single image)      : {gflops:.3f}")
    else:
        print("  GFLOPs                     : N/A  (install fvcore or thop)")

    # ── 3. Warm-up ────────────────────────────────────────────────────────────
    print(f"\n  Warming up ({cfg.warmup_batches} batches)…")
    loader_iter = iter(loader)
    for _ in range(cfg.warmup_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        _forward_pass(model, batch, device, cfg.use_amp)

    if device != "cpu":
        torch.cuda.synchronize()

    # ── 4. Timed run ──────────────────────────────────────────────────────────
    metrics_tracker = SegmentationMetrics(cfg.num_classes, cfg.class_names)
    latencies_ms: List[float] = []

    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"  Timing {cfg.num_batches} batches…")
    batches_done = 0

    # Restart iterator so timed batches are independent of warm-up batches
    loader_iter = iter(loader)

    while batches_done < cfg.num_batches:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        mask = batch["mask"].to(device, non_blocking=True).long()

        # --- CUDA event timing (most accurate on GPU) -----------------------
        if device != "cpu":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            preds = _forward_pass(model, batch, device, cfg.use_amp)
            end_evt.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_evt.elapsed_time(end_evt))
        else:
            t0 = time.perf_counter()
            preds = _forward_pass(model, batch, device, cfg.use_amp)
            latencies_ms.append((time.perf_counter() - t0) * 1e3)

        metrics_tracker.update(preds, mask)
        batches_done += 1

    # ── 5. GPU memory ─────────────────────────────────────────────────────────
    if device != "cpu":
        gpu_peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        gpu_reserved_mb = torch.cuda.max_memory_reserved(device) / 1e6
    else:
        gpu_peak_mb = -1.0
        gpu_reserved_mb = -1.0

    # ── Latency statistics ────────────────────────────────────────────────────
    lat = np.array(latencies_ms)
    mean_lat  = float(lat.mean())
    std_lat   = float(lat.std())
    p50_lat   = float(np.percentile(lat, 50))
    p95_lat   = float(np.percentile(lat, 95))
    p99_lat   = float(np.percentile(lat, 99))
    min_lat   = float(lat.min())
    max_lat   = float(lat.max())

    # Throughput: images / second
    total_images = batches_done * cfg.batch_size
    total_time_s = lat.sum() / 1e3
    throughput   = total_images / total_time_s

    print(f"\n  ── Latency (ms per batch, bs={cfg.batch_size}) ──")
    print(f"     Mean±Std : {mean_lat:.2f} ± {std_lat:.2f}")
    print(f"     P50/P95/P99 : {p50_lat:.2f} / {p95_lat:.2f} / {p99_lat:.2f}")
    print(f"     Min / Max : {min_lat:.2f} / {max_lat:.2f}")
    print(f"     Throughput : {throughput:.1f} img/s")
    if device != "cpu":
        print(f"     GPU peak mem (allocated): {gpu_peak_mb:.1f} MB")
        print(f"     GPU peak mem (reserved) : {gpu_reserved_mb:.1f} MB")

    # ── Segmentation metrics ──────────────────────────────────────────────────
    seg = metrics_tracker.compute()

    print(f"\n  ── Segmentation metrics (over {batches_done} batches) ──")
    print(f"     Pixel Acc  : {seg['pixel_accuracy']:.4f}")
    print(f"     Mean Acc   : {seg['mean_accuracy']:.4f}")
    print(f"     mIoU       : {seg['mean_iou']:.4f}")
    print(f"     mDice/mF1  : {seg['mean_dice']:.4f}")
    print(f"     mPrecision : {seg['mean_precision']:.4f}")
    print(f"     mRecall    : {seg['mean_recall']:.4f}")
    for c in cfg.class_names:
        print(f"     {c:<12} IoU={seg[f'iou_{c}']:.4f}  "
              f"Dice={seg[f'dice_{c}']:.4f}  "
              f"Prec={seg[f'precision_{c}']:.4f}  "
              f"Rec={seg[f'recall_{c}']:.4f}")

    # ── Assemble flat result dict ─────────────────────────────────────────────
    result = {
        "model_name":          model_name,
        "batch_size":          cfg.batch_size,
        "input_resolution":    f"{cfg.target_size[0]}x{cfg.target_size[1]}",
        "num_batches_timed":   batches_done,
        "total_params_M":      round(total_params / 1e6, 3),
        "trainable_params_M":  round(trainable_params / 1e6, 3),
        "model_memory_MB":     round(param_mb + buffer_mb, 2),
        "checkpoint_MB":       round(disk_mb, 2) if disk_mb >= 0 else "N/A",
        "gflops":              round(gflops, 3) if gflops >= 0 else "N/A",
        "use_amp":             cfg.use_amp,
        "device":              device,
        "lat_mean_ms":         round(mean_lat, 3),
        "lat_std_ms":          round(std_lat, 3),
        "lat_p50_ms":          round(p50_lat, 3),
        "lat_p95_ms":          round(p95_lat, 3),
        "lat_p99_ms":          round(p99_lat, 3),
        "lat_min_ms":          round(min_lat, 3),
        "lat_max_ms":          round(max_lat, 3),
        "throughput_img_s":    round(throughput, 2),
        "gpu_peak_alloc_MB":   round(gpu_peak_mb, 1) if gpu_peak_mb >= 0 else "N/A",
        "gpu_peak_reserved_MB":round(gpu_reserved_mb, 1) if gpu_reserved_mb >= 0 else "N/A",
        **{k: round(v, 4) for k, v in seg.items()},
    }
    return result


# =============================================================================
# SUMMARY TABLE  (pretty console print)
# =============================================================================

def print_summary_table(results: List[Dict], class_names: List[str]):
    sep = "─" * 130
    print(f"\n{'='*130}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*130}")
    print(f"  {'Model':<28} {'Params(M)':>9} {'GFLOPs':>8} {'ckpt(MB)':>8} "
          f"{'Lat(ms)':>9} {'P95':>7} {'img/s':>7} "
          f"{'mIoU':>7} {'mDice':>7} {'PixAcc':>7} "
          + "  ".join(f"{c[:6]:>7}" for c in class_names))
    print(sep)
    for r in results:
        iou_cols = "  ".join(
            f"{r.get(f'iou_{c}', 0):.4f}" for c in class_names
        )
        print(
            f"  {r['model_name']:<28} "
            f"{r['total_params_M']:>9.3f} "
            f"{str(r['gflops']):>8} "
            f"{str(r['checkpoint_MB']):>8} "
            f"{r['lat_mean_ms']:>9.2f} "
            f"{r['lat_p95_ms']:>7.2f} "
            f"{r['throughput_img_s']:>7.1f} "
            f"{r['mean_iou']:>7.4f} "
            f"{r['mean_dice']:>7.4f} "
            f"{r['pixel_accuracy']:>7.4f} "
            f"  {iou_cols}"
        )
    print(sep)


# =============================================================================
# CSV EXPORT
# =============================================================================

def save_csv(results: List[Dict], output_dir: str, csv_name: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / csv_name
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[✓] Results saved → {out_path}")


# =============================================================================
# VISUALISATION
# =============================================================================

# Colour palette — one vivid, distinguishable colour per class
#   0 = background  →  dark grey
#   1 = crop        →  bright green
#   2 = weed        →  vivid red
_CLASS_COLORS = np.array([
    [45,  45,  45 ],   # background  — near-black
    [0,   200, 80 ],   # crop        — green
    [230, 30,  30 ],   # weed        — red
], dtype=np.uint8)

# Soft transparent overlay colours (RGBA, 0–1 range)
_OVERLAY_COLORS = np.array([
    [0.18, 0.18, 0.18, 0.0 ],   # background  — fully transparent
    [0.0,  0.78, 0.31, 0.55],   # crop        — green semi-transparent
    [0.90, 0.12, 0.12, 0.65],   # weed        — red   semi-transparent
])


def _mask_to_rgb(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """(H, W) int mask → (H, W, 3) uint8 colour image."""
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for c, colour in enumerate(palette):
        rgb[mask == c] = colour
    return rgb


def _tensor_to_display_rgb(t: torch.Tensor) -> np.ndarray:
    """
    Convert a CHW float tensor (ImageNet-normalised or raw [0,1]) to an
    HWC uint8 array suitable for imshow.
    We try to denormalise with ImageNet stats; if values are already ≤1 we
    just clip and scale.
    """
    img = t.cpu().float().numpy()          # (C, H, W)
    # ImageNet denorm
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
    img  = img * std + mean
    img  = np.clip(img, 0.0, 1.0)
    img  = (img * 255).astype(np.uint8)
    return img.transpose(1, 2, 0)          # (H, W, 3)


def _nir_to_display(t: torch.Tensor) -> np.ndarray:
    """Single-channel NIR tensor → (H, W, 3) uint8 (grey)."""
    nir = t.cpu().float().numpy().squeeze()          # (H, W)
    nir = np.clip(nir, 0.0, 1.0)
    nir_u8 = (nir * 255).astype(np.uint8)
    return np.stack([nir_u8, nir_u8, nir_u8], axis=-1)


def _blend_overlay(rgb_hw3: np.ndarray, mask_hw: np.ndarray,
                   alpha_table: np.ndarray) -> np.ndarray:
    """
    Blend per-class semi-transparent overlays onto an RGB image.
    alpha_table: (num_classes, 4) RGBA where A controls opacity.
    Returns float32 HWC image in [0, 1].
    """
    base  = rgb_hw3.astype(np.float32) / 255.0
    out   = base.copy()
    for c, rgba in enumerate(alpha_table):
        colour = np.array(rgba[:3], dtype=np.float32)
        alpha  = rgba[3]
        region = (mask_hw == c)
        out[region] = (1 - alpha) * base[region] + alpha * colour
    return np.clip(out, 0.0, 1.0)


def _per_class_binary_masks(mask_hw: np.ndarray,
                             num_classes: int) -> List[np.ndarray]:
    """Return list of binary (H, W) bool arrays, one per class."""
    return [(mask_hw == c) for c in range(num_classes)]


def _iou_single(pred_hw: np.ndarray, gt_hw: np.ndarray,
                cls: int, eps: float = 1e-6) -> float:
    p = (pred_hw == cls)
    g = (gt_hw  == cls)
    return float((p & g).sum()) / float((p | g).sum() + eps)


def visualize_predictions(
    model_name:  str,
    model:       nn.Module,
    loader:      DataLoader,
    cfg:         BenchConfig,
    device:      str,
    output_dir:  str,
) -> str:
    """
    Collect `cfg.num_vis_images` samples and save a publication-quality grid.

    Each row = one image, columns:
      0 – RGB input
      1 – NIR input (false-colour)
      2 – Ground-truth mask  (colour-coded)
      3 – Predicted mask     (colour-coded)
      4 – Overlay: prediction blended onto RGB
      5 – Error map          (TP green / FP red / FN orange / TN dark)
      6–8 – Per-class confidence maps (softmax prob) for bg / crop / weed

    A legend and per-image IoU annotation are added automatically.
    Saves:
      <output_dir>/<model_name>_vis_grid.png   (full grid)
      <output_dir>/<model_name>_vis_N.png      (individual rows, publication-ready)
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    num_vis    = cfg.num_vis_images
    class_names = cfg.class_names
    num_classes = cfg.num_classes
    palette     = _CLASS_COLORS[:num_classes]

    # ── Collect samples ───────────────────────────────────────────────────────
    samples: List[Dict] = []
    loader_iter = iter(loader)

    while len(samples) < num_vis:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        rgb_b  = batch["rgb"]    # (B, 3, H, W)
        nir_b  = batch["nir"]    # (B, 1, H, W)
        mask_b = batch["mask"]   # (B, H, W)

        with torch.no_grad():
            rgb_d  = rgb_b.to(device)
            nir_d  = nir_b.to(device)
            if cfg.use_amp and device != "cpu":
                with torch.autocast(device):
                    out = model(rgb_d, nir_d)
            else:
                out = model(rgb_d, nir_d)

        logits = out[0] if isinstance(out, (tuple, list)) else out   # (B,C,H,W)
        probs  = F.softmax(logits, dim=1).cpu()                       # (B,C,H,W)
        preds  = probs.argmax(dim=1)                                  # (B,H,W)

        for i in range(rgb_b.shape[0]):
            if len(samples) >= num_vis:
                break
            samples.append({
                "rgb":   rgb_b[i],           # (3,H,W) cpu tensor
                "nir":   nir_b[i],           # (1,H,W) cpu tensor
                "gt":    mask_b[i].numpy(),  # (H,W) int numpy
                "pred":  preds[i].numpy(),   # (H,W) int numpy
                "probs": probs[i].numpy(),   # (C,H,W) float numpy
            })

    n_cols = 6 + num_classes    # RGB | NIR | GT | Pred | Overlay | Error | prob×C
    col_titles = (
        ["RGB Input", "NIR Input", "Ground Truth", "Prediction", "Pred Overlay", "Error Map"]
        + [f"P({c})" for c in class_names]
    )

    # ── Colour-maps for prob channels ─────────────────────────────────────────
    prob_cmaps = ["Greys_r", "Greens", "Reds"][:num_classes]

    # ── Build the big grid figure ─────────────────────────────────────────────
    cell_px   = 2.2          # inches per cell
    fig_w     = cell_px * n_cols + 1.2
    fig_h     = cell_px * len(samples) + 1.4
    fig, axes = plt.subplots(
        len(samples), n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.04, "hspace": 0.18},
    )
    if len(samples) == 1:
        axes = axes[np.newaxis, :]   # always 2-D

    fig.patch.set_facecolor("#1a1a2e")

    # Column header strip
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=7, color="white",
                             fontweight="bold", pad=3)

    for row_idx, s in enumerate(samples):
        rgb_disp  = _tensor_to_display_rgb(s["rgb"])    # (H,W,3) uint8
        nir_disp  = _nir_to_display(s["nir"])           # (H,W,3) uint8
        gt_disp   = _mask_to_rgb(s["gt"],   palette)
        pred_disp = _mask_to_rgb(s["pred"], palette)
        overlay   = _blend_overlay(rgb_disp, s["pred"], _OVERLAY_COLORS[:num_classes])

        # Error map: TP=dark green, FP=red, FN=orange, TN=near-black
        err = np.zeros((*s["gt"].shape, 3), dtype=np.uint8)
        tp = (s["pred"] == s["gt"]) & (s["gt"] > 0)
        fp = (s["pred"] != s["gt"]) & (s["pred"] > 0)
        fn = (s["pred"] != s["gt"]) & (s["gt"]  > 0)
        err[tp] = [0,   180, 80 ]   # green  — correct foreground
        err[fp] = [230, 40,  40 ]   # red    — false positive
        err[fn] = [255, 160, 0  ]   # orange — false negative
        # TN (bg correct) stays black

        panels = [rgb_disp, nir_disp, gt_disp, pred_disp, overlay, err]
        for prob_c in range(num_classes):
            panels.append(s["probs"][prob_c])   # (H,W) float

        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < 6:
                ax.imshow(panel, interpolation="nearest")
            else:
                # prob map with colour-map
                c = col_idx - 6
                ax.imshow(panel, cmap=prob_cmaps[c], vmin=0, vmax=1,
                          interpolation="bilinear")

        # Per-image IoU annotation on the GT cell (col 2)
        per_img_ious = [_iou_single(s["pred"], s["gt"], c) for c in range(num_classes)]
        iou_str = "  ".join(
            f"{class_names[c][0].upper()}:{per_img_ious[c]:.2f}"
            for c in range(num_classes)
        )
        axes[row_idx, 2].set_xlabel(iou_str, fontsize=5.5, color="#aaffaa",
                                    labelpad=2, fontfamily="monospace")

        # Row label
        axes[row_idx, 0].set_ylabel(f"img {row_idx+1}", fontsize=6,
                                    color="#cccccc", rotation=0,
                                    labelpad=28, va="center")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=tuple(c/255 for c in palette[i]), label=class_names[i])
        for i in range(num_classes)
    ]
    error_patches = [
        mpatches.Patch(color=(0, 0.71, 0.31), label="TP (fg correct)"),
        mpatches.Patch(color=(0.90, 0.16, 0.16), label="FP (false pos)"),
        mpatches.Patch(color=(1.0,  0.63, 0.0),  label="FN (false neg)"),
    ]
    fig.legend(
        handles=legend_patches + error_patches,
        loc="lower center", ncol=num_classes + 3,
        fontsize=7, framealpha=0.25, facecolor="#2a2a4a",
        labelcolor="white", edgecolor="#555577",
        bbox_to_anchor=(0.5, 0.005),
    )

    # Super-title
    fig.suptitle(
        f"{model_name}  —  Inference Visualisation  ({len(samples)} images, {cfg.split} split)",
        fontsize=11, color="white", fontweight="bold", y=0.995,
    )

    grid_path = Path(output_dir) / f"{model_name}_vis_grid.png"
    fig.savefig(grid_path, dpi=cfg.vis_dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Grid saved → {grid_path}")

    # ── Individual per-image figures (publication-ready single rows) ──────────
    for row_idx, s in enumerate(samples):
        rgb_disp  = _tensor_to_display_rgb(s["rgb"])
        nir_disp  = _nir_to_display(s["nir"])
        gt_disp   = _mask_to_rgb(s["gt"],   palette)
        pred_disp = _mask_to_rgb(s["pred"], palette)
        overlay   = _blend_overlay(rgb_disp, s["pred"], _OVERLAY_COLORS[:num_classes])

        err = np.zeros((*s["gt"].shape, 3), dtype=np.uint8)
        tp = (s["pred"] == s["gt"]) & (s["gt"] > 0)
        fp = (s["pred"] != s["gt"]) & (s["pred"] > 0)
        fn = (s["pred"] != s["gt"]) & (s["gt"]  > 0)
        err[tp] = [0, 180, 80]; err[fp] = [230, 40, 40]; err[fn] = [255, 160, 0]

        per_img_ious = [_iou_single(s["pred"], s["gt"], c) for c in range(num_classes)]

        n_single = 5 + num_classes
        f2, axs = plt.subplots(1, n_single, figsize=(3.0 * n_single, 3.4))
        f2.patch.set_facecolor("#1a1a2e")

        single_panels = [rgb_disp, nir_disp, gt_disp, pred_disp, overlay, err]
        for c in range(num_classes):
            single_panels.append(s["probs"][c])

        single_titles = (
            ["RGB", "NIR", "Ground Truth", "Prediction", "Overlay"]
            + [f"P({class_names[c]})" for c in range(num_classes)]
        )

        for j, (panel, title) in enumerate(zip(single_panels, single_titles)):
            ax = axs[j]; ax.axis("off")
            if j < 6:
                ax.imshow(panel, interpolation="nearest")
            else:
                c = j - 6
                ax.imshow(panel, cmap=prob_cmaps[c], vmin=0, vmax=1,
                          interpolation="bilinear")
            ax.set_title(title, fontsize=8, color="white", fontweight="bold", pad=3)

        iou_str = "  |  ".join(
            f"{class_names[c]}: IoU={per_img_ious[c]:.3f}"
            for c in range(num_classes)
        )
        f2.suptitle(
            f"{model_name}  —  Image {row_idx+1}  |  {iou_str}",
            fontsize=8.5, color="#aaffaa", fontweight="bold", y=1.01,
        )

        # Colour legend
        f2.legend(
            handles=legend_patches + error_patches,
            loc="lower center", ncol=num_classes + 3,
            fontsize=7, framealpha=0.2, facecolor="#2a2a4a",
            labelcolor="white", edgecolor="#444466",
            bbox_to_anchor=(0.5, -0.06),
        )

        single_path = Path(output_dir) / f"{model_name}_vis_{row_idx+1:02d}.png"
        f2.savefig(single_path, dpi=cfg.vis_dpi, bbox_inches="tight",
                   facecolor=f2.get_facecolor())
        plt.close(f2)

    print(f"  [✓] {len(samples)} individual images saved → {output_dir}/")
    return str(grid_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference benchmark harness")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Names of models to benchmark (default: all registered)",
    )
    parser.add_argument("--batch-size",   type=int, default=CFG.batch_size)
    parser.add_argument("--num-batches",  type=int, default=CFG.num_batches)
    parser.add_argument("--warmup",       type=int, default=CFG.warmup_batches)
    parser.add_argument("--split",        type=str, default=CFG.split,
                        choices=["train", "val", "test"])
    parser.add_argument("--no-amp",       action="store_true")
    parser.add_argument("--compile",      action="store_true")
    parser.add_argument("--output-dir",   type=str, default=CFG.output_dir)
    parser.add_argument("--no-vis",       action="store_true",
                        help="Skip visualisation (benchmark metrics only)")
    parser.add_argument("--num-vis",      type=int, default=CFG.num_vis_images,
                        help="Number of images to visualise per model (default 10)")
    parser.add_argument("--vis-dpi",      type=int, default=CFG.vis_dpi)
    args = parser.parse_args()

    # Apply CLI overrides to shared config
    CFG.batch_size      = args.batch_size
    CFG.num_batches     = args.num_batches
    CFG.warmup_batches  = args.warmup
    CFG.split           = args.split
    CFG.use_amp         = not args.no_amp
    CFG.use_compile     = args.compile
    CFG.output_dir      = args.output_dir
    CFG.num_vis_images  = args.num_vis
    CFG.vis_dpi         = args.vis_dpi

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Setup] device={device}  AMP={CFG.use_amp}  compile={CFG.use_compile}")
    if device == "cuda":
        print(f"        GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"        PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"        batch_size={CFG.batch_size}  num_batches={CFG.num_batches}  "
          f"warmup={CFG.warmup_batches}")

    # ── Determine which models to run ─────────────────────────────────────────
    to_run = args.models if args.models else list(MODEL_REGISTRY.keys())
    unknown = [m for m in to_run if m not in MODEL_REGISTRY]
    if unknown:
        print(f"[ERROR] Unknown model(s): {unknown}")
        print(f"        Registered: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    # ── Build the SINGLE shared DataLoader ───────────────────────────────────
    print(f"\n[DataLoader] Building shared {CFG.split} loader…")
    loader = build_dataloader(CFG)

    # ── Benchmark each model ──────────────────────────────────────────────────
    all_results: List[Dict] = []
    for name in to_run:
        print(f"\n[Loading] {name}…")
        model = MODEL_REGISTRY[name]()

        # Try to infer ckpt_path from the loader function's docstring or closure
        # (We'll just pass None; disk size is looked up inside _load_* functions)
        ckpt_path = _get_ckpt_path(name)

        result = benchmark_model(
            model_name=name,
            model=model,
            loader=loader,
            cfg=CFG,
            device=device,
            ckpt_path=ckpt_path,
        )
        all_results.append(result)

        # ── Visualise ─────────────────────────────────────────────────────────
        if not args.no_vis:
            print(f"\n[Visualise] {name}  ({CFG.num_vis_images} images)…")
            vis_dir = Path(CFG.output_dir) / "visualisations"
            visualize_predictions(
                model_name=name,
                model=model,       # still on device, already eval()
                loader=loader,
                cfg=CFG,
                device=device,
                output_dir=str(vis_dir),
            )

        # Free GPU memory between models
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Print + save ──────────────────────────────────────────────────────────
    print_summary_table(all_results, CFG.class_names)
    save_csv(all_results, CFG.output_dir, CFG.csv_name)


# ---------------------------------------------------------------------------
# Helper: centralised checkpoint path registry
# Add each model's ckpt path here so disk-size is always reported correctly.
# ---------------------------------------------------------------------------
_CKPT_PATHS: Dict[str, str] = {
    "dual_encoder_mini": (
        "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/"
        "dual_encoder_mini_20260412_130647/checkpoints/best_model.pth"
    ),
    # "dual_encoder_full": "/path/to/full/best_model.pth",
}

def _get_ckpt_path(model_name: str) -> Optional[str]:
    return _CKPT_PATHS.get(model_name, None)


if __name__ == "__main__":
    main()