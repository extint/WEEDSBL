#!/usr/bin/env python3

"""
Training script for Sugar Beets weed segmentation
using pretrained torchvision DeepLab variants

Supported architectures:
  - deeplabv3_resnet50       (~42M params, good balance)
  - deeplabv3_resnet101      (~61M params, highest accuracy)
  - deeplabv3_mobilenet      (~11M params, DeepLabV3 + MobileNetV3-Large)
  - lraspp_mobilenet         (~3.2M params, lightest — simplified ASPP head)

Dataset: 3-class segmentation (background / crop / weed)
Features:
  - RGB+NIR (4 channels) or RGB-only (3 channels)
  - NIR channel weight initialised from RGB mean of pretrained weights
  - Auxiliary loss head (weighted 0.4x) — skipped for lraspp (no aux head)
  - Freeze backbone option with optional unfreeze after N epochs
  - Comprehensive metrics (mIoU, per-class IoU, precision, recall, F1)
  - TensorBoard logging with run-specific directories
  - Mixed precision training support
  - Checkpointing and resumption
"""

import argparse
import os
import json
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import (
    deeplabv3_resnet50,    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet101,   DeepLabV3_ResNet101_Weights,
    deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,    LRASPP_MobileNet_V3_Large_Weights,
)
from tqdm import tqdm

# Import dataset loader (same as original script)
from sugarbeets_data_loader import create_sugarbeets_dataloaders


# ======================== Model Registry ========================

# Maps CLI name → (loader_fn, weights_cls, architecture_family)
#   family "resnet"   → backbone at model.backbone.conv1
#   family "mobilenet"→ backbone at model.backbone.features[0][0]
#   family "lraspp"   → same backbone path, but NO aux_classifier head
_MODEL_REGISTRY = {
    "deeplabv3_resnet50": (
        deeplabv3_resnet50,
        DeepLabV3_ResNet50_Weights,
        "resnet",
    ),
    "deeplabv3_resnet101": (
        deeplabv3_resnet101,
        DeepLabV3_ResNet101_Weights,
        "resnet",
    ),
    "deeplabv3_mobilenet": (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights,
        "mobilenet",
    ),
    "lraspp_mobilenet": (
        lraspp_mobilenet_v3_large,
        LRASPP_MobileNet_V3_Large_Weights,
        "lraspp",
    ),
}

VALID_MODELS = list(_MODEL_REGISTRY.keys())


# ======================== Model Builder ========================

def _patch_input_conv_resnet(model: nn.Module):
    """Replace ResNet stem conv1 with a 4-channel version."""
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
    model.backbone.conv1 = new_conv


def _get_first_conv2d_path(module: nn.Module):
    """
    Walk a module tree and return (parent, attr_name, Conv2d) for the
    first Conv2d encountered in named_children order (BFS).
    Returns None if not found.
    """
    from collections import deque
    queue = deque()
    for name, child in module.named_children():
        queue.append((module, name, child))

    while queue:
        parent, attr, mod = queue.popleft()
        if isinstance(mod, nn.Conv2d):
            return parent, attr, mod
        for name, child in mod.named_children():
            queue.append((mod, name, child))
    return None


def _patch_input_conv_mobilenet(model: nn.Module):
    """
    Replace the MobileNetV3 backbone's first Conv2d with a 4-channel version.

    torchvision wraps the backbone in IntermediateLayerGetter whose internal
    layout can vary across torchvision versions, so we robustly locate the
    first Conv2d via BFS rather than hard-coding a path.
    """
    result = _get_first_conv2d_path(model.backbone)
    if result is None:
        raise RuntimeError("Could not find any Conv2d in the MobileNet backbone.")

    parent, attr, old_conv = result
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    setattr(parent, attr, new_conv)
    print(f"[INFO] Patched MobileNet first conv at backbone.{attr}: "
          f"Conv2d(3→4, {tuple(old_conv.kernel_size)})")


def build_model(
    architecture: str = "deeplabv3_resnet50",
    num_classes: int = 3,
    use_rgbnir: bool = False,
    freeze_backbone: bool = False,
    pretrained: bool = True,
) -> nn.Module:
    """
    Load a pretrained DeepLab variant and adapt it for sugar beets segmentation.

    Args:
        architecture:     One of VALID_MODELS
        num_classes:      Output classes (3 for BG/Crop/Weed)
        use_rgbnir:       Patch first conv to accept 4 channels (RGB+NIR)
        freeze_backbone:  Freeze backbone weights initially
        pretrained:       Load COCO pretrained weights

    Returns:
        nn.Module ready for training
    """
    if architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{architecture}'. "
                         f"Choose from: {VALID_MODELS}")

    loader_fn, weights_cls, family = _MODEL_REGISTRY[architecture]

    weights = weights_cls.DEFAULT if pretrained else None
    model   = loader_fn(weights=weights)

    # ── Patch input conv for RGB+NIR ──────────────────────────────────────────
    if use_rgbnir:
        if family == "resnet":
            _patch_input_conv_resnet(model)
        else:  # mobilenet / lraspp share the same backbone layout
            _patch_input_conv_mobilenet(model)
        print("[INFO] Input conv replaced: 3ch → 4ch (NIR initialised from RGB mean)")

    # ── Replace classifier head(s) ────────────────────────────────────────────
    if family == "lraspp":
        # LRASPP has a different head structure with no aux classifier
        # low_classifier and high_classifier are both Conv2d
        in_low  = model.classifier.low_classifier.in_channels    # 40
        in_high = model.classifier.high_classifier.in_channels   # 128
        model.classifier.low_classifier  = nn.Conv2d(in_low,  num_classes, kernel_size=1)
        model.classifier.high_classifier = nn.Conv2d(in_high, num_classes, kernel_size=1)
        print(f"[INFO] LRASPP heads replaced: 21 → {num_classes} classes (no aux head)")
    else:
        # DeepLabV3 main head: Sequential[..., Conv2d at index 4]
        in_main = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_main, num_classes, kernel_size=1)
        if model.aux_classifier is not None:
            in_aux = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(in_aux, num_classes, kernel_size=1)
        print(f"[INFO] Classifier head replaced: 21 → {num_classes} classes")

    # ── Optionally freeze backbone ─────────────────────────────────────────────
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("[INFO] Backbone frozen — only heads will be trained")

    return model


def has_aux_head(model: nn.Module) -> bool:
    """Return True if the model has a trainable auxiliary classifier head."""
    return (
        hasattr(model, "aux_classifier")
        and model.aux_classifier is not None
        # LRASPP inherits the attribute but sets it to None
    )


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all backbone parameters (called mid-training if desired)."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("[INFO] Backbone unfrozen — full model now trainable")


def get_model_info(model: nn.Module, architecture: str) -> Dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "architecture": architecture,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "total_parameters_million": total / 1e6,
        "trainable_parameters_million": trainable / 1e6,
    }


# ======================== Loss Functions ========================

class MultiClassDiceLoss(nn.Module):
    """Multi-class segmentation loss: 0.5 * CrossEntropy + 0.5 * Dice"""
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean()

        return 0.5 * ce_loss + 0.5 * dice_loss


class MultiClassIoULoss(nn.Module):
    """IoU loss (used for monitoring only, not training)"""
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)
        iou = (intersection + self.eps) / (union + self.eps)
        return 1 - iou.mean()


# ======================== Training & Evaluation ========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn: nn.Module,
    device,
    epoch: int,
    writer: SummaryWriter,
    num_classes: int = 3,
    scaler=None,
    aux_loss_weight: float = 0.4,
) -> Dict[str, float]:
    """
    Train for one epoch.

    DeepLabV3 variants return a dict with 'out' (main) and 'aux' (auxiliary).
    LRASPP returns only 'out' — aux loss is skipped automatically.
    Total loss = (1 - aux_loss_weight) * main_loss + aux_loss_weight * aux_loss
    """
    model.train()
    epoch_loss = 0.0
    use_aux    = has_aux_head(model)

    total_correct  = 0
    total_pixels   = 0
    class_tp         = [0] * num_classes
    class_fp         = [0] * num_classes
    class_fn         = [0] * num_classes
    class_intersection = [0] * num_classes
    class_union      = [0] * num_classes

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]")

    for batch_idx, batch in enumerate(pbar):
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs   = model(x)
                main_loss = loss_fn(outputs["out"], y)
                if use_aux:
                    aux_loss = loss_fn(outputs["aux"], y)
                    loss     = (1 - aux_loss_weight) * main_loss + aux_loss_weight * aux_loss
                else:
                    aux_loss = torch.tensor(0.0)
                    loss     = main_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs   = model(x)
            main_loss = loss_fn(outputs["out"], y)
            if use_aux:
                aux_loss = loss_fn(outputs["aux"], y)
                loss     = (1 - aux_loss_weight) * main_loss + aux_loss_weight * aux_loss
            else:
                aux_loss = torch.tensor(0.0)
                loss     = main_loss
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        with torch.no_grad():
            preds = torch.argmax(outputs["out"], dim=1)

            total_correct += (preds == y).sum().item()
            total_pixels  += y.numel()

            for cls in range(num_classes):
                pred_mask   = (preds == cls)
                target_mask = (y == cls)

                class_tp[cls]           += (pred_mask & target_mask).sum().item()
                class_fp[cls]           += (pred_mask & ~target_mask).sum().item()
                class_fn[cls]           += (~pred_mask & target_mask).sum().item()
                class_intersection[cls] += (pred_mask & target_mask).sum().item()
                class_union[cls]        += (pred_mask | target_mask).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Train/BatchLoss",     loss.item(),      global_step)
            writer.add_scalar("Train/BatchMainLoss", main_loss.item(), global_step)
            writer.add_scalar("Train/BatchAuxLoss",  aux_loss.item(),  global_step)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    metrics = _aggregate_metrics(
        total_correct, total_pixels,
        class_tp, class_fp, class_fn,
        class_intersection, class_union,
        num_classes
    )
    metrics["loss"] = epoch_loss / len(loader)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device,
    num_classes: int = 3,
    split: str = "Val",
) -> Dict[str, float]:
    """Evaluate model — uses only the main output ('out') head."""
    model.eval()
    total_loss     = 0.0
    total_iou_loss = 0.0
    iou_loss_fn    = MultiClassIoULoss(num_classes=num_classes)

    total_correct    = 0
    total_pixels     = 0
    class_tp         = [0] * num_classes
    class_fp         = [0] * num_classes
    class_fn         = [0] * num_classes
    class_intersection = [0] * num_classes
    class_union      = [0] * num_classes

    pbar = tqdm(loader, desc=f"[{split}]")

    for batch in pbar:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()

        outputs = model(x)
        logits  = outputs["out"]

        total_loss     += loss_fn(logits, y).item()
        total_iou_loss += iou_loss_fn(logits, y).item()

        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == y).sum().item()
        total_pixels  += y.numel()

        for cls in range(num_classes):
            pred_mask   = (preds == cls)
            target_mask = (y == cls)

            class_tp[cls]           += (pred_mask & target_mask).sum().item()
            class_fp[cls]           += (pred_mask & ~target_mask).sum().item()
            class_fn[cls]           += (~pred_mask & target_mask).sum().item()
            class_intersection[cls] += (pred_mask & target_mask).sum().item()
            class_union[cls]        += (pred_mask | target_mask).sum().item()

    metrics = _aggregate_metrics(
        total_correct, total_pixels,
        class_tp, class_fp, class_fn,
        class_intersection, class_union,
        num_classes
    )
    metrics["loss"]     = total_loss     / len(loader)
    metrics["iou_loss"] = total_iou_loss / len(loader)
    return metrics


def _aggregate_metrics(
    total_correct, total_pixels,
    class_tp, class_fp, class_fn,
    class_intersection, class_union,
    num_classes: int,
) -> Dict[str, float]:
    """Shared helper: compute per-class and mean metrics from accumulators."""
    class_names = ["Background", "Crop", "Weed"]
    metrics = {}
    metrics["pixel_accuracy"] = total_correct / total_pixels

    class_ious       = []
    class_precisions = []
    class_recalls    = []
    class_f1s        = []

    for cls in range(num_classes):
        iou       = class_intersection[cls] / (class_union[cls] + 1e-6)
        precision = class_tp[cls] / (class_tp[cls] + class_fp[cls] + 1e-6)
        recall    = class_tp[cls] / (class_tp[cls] + class_fn[cls] + 1e-6)
        f1        = 2 * precision * recall / (precision + recall + 1e-6)

        class_ious.append(iou)
        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)

        name = class_names[cls]
        metrics[f"iou_{name}"]       = iou
        metrics[f"precision_{name}"] = precision
        metrics[f"recall_{name}"]    = recall
        metrics[f"f1_{name}"]        = f1

    metrics["miou"]           = float(np.mean(class_ious))
    metrics["mean_precision"] = float(np.mean(class_precisions))
    metrics["mean_recall"]    = float(np.mean(class_recalls))
    metrics["mean_f1"]        = float(np.mean(class_f1s))
    return metrics


# ======================== Main Training Loop ========================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained DeepLab variants on Sugar Beets"
    )

    # Model
    parser.add_argument("--model", type=str, default="deeplabv3_resnet50",
                        choices=VALID_MODELS,
                        help=(
                            "Pretrained architecture to use. Options:\n"
                            "  deeplabv3_resnet50   — ~42M params (default)\n"
                            "  deeplabv3_resnet101  — ~61M params, highest accuracy\n"
                            "  deeplabv3_mobilenet  — ~11M params, lightweight\n"
                            "  lraspp_mobilenet     — ~3.2M params, lightest (no aux head)"
                        ))

    # Data
    parser.add_argument("--data_root", type=str,
                        default="/home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET")
    parser.add_argument("--use_rgbnir", action="store_true",
                        help="Use RGB+NIR (4ch); default is RGB-only (3ch)")
    parser.add_argument("--height", type=int, default=966)
    parser.add_argument("--width",  type=int, default=1296)
    parser.add_argument("--nir_drop", type=float, default=0.0,
                        help="Probability of dropping NIR channel during training")

    # Model options
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train from scratch (no COCO weights)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone weights initially")
    parser.add_argument("--unfreeze_epoch", type=int, default=None,
                        help="Epoch at which to unfreeze backbone (requires --freeze_backbone)")
# it is done, training now ok
    # Training
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--lr",             type=float, default=1e-4,
                        help="Learning rate (recommend 1e-4 for fine-tuning)")
    parser.add_argument("--aux_loss_weight",type=float, default=0.4,
                        help="Weight for auxiliary head loss (0 to disable)")
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--mixed_precision",action="store_true")

    # Checkpointing
    parser.add_argument("--exp_name",   type=str, default=None,
                        help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    num_classes  = 3
    class_names  = ["background", "crop", "weed"]

    # ── Run directory ──────────────────────────────────────────────────────────
    if args.exp_name is None:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        channels   = "4ch_RGBNIR" if args.use_rgbnir else "3ch_RGB"
        freeze_tag = "_frozenBB" if args.freeze_backbone else ""
        args.exp_name = f"sugarbeets_{args.model}_{channels}{freeze_tag}_{timestamp}"

    run_dir  = os.path.join(args.output_dir, args.exp_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir  = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # Save config
    config_dict = {**vars(args), "num_classes": num_classes, "class_names": class_names}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n{'='*80}")
    print(f"[INFO] Model     : {args.model} (pretrained={not args.no_pretrained})")
    print(f"[INFO] Dataset   : Sugar Beets ({num_classes} classes: {class_names})")
    print(f"[INFO] Run dir   : {run_dir}")
    print(f"{'='*80}\n")

    writer = SummaryWriter(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device    : {device}")

    # ── Dataloaders ────────────────────────────────────────────────────────────
    print("[INFO] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_sugarbeets_dataloaders(
        data_root=args.data_root,
        use_rgbnir=args.use_rgbnir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.height, args.width),
        nir_drop_prob=args.nir_drop,
    )
    print(f"[INFO] Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(
        architecture=args.model,
        num_classes=num_classes,
        use_rgbnir=args.use_rgbnir,
        freeze_backbone=args.freeze_backbone,
        pretrained=not args.no_pretrained,
    )
    model = model.to(device)

    model_info = get_model_info(model, args.model)
    print(f"[INFO] Total params    : {model_info['total_parameters']:,} "
          f"({model_info['total_parameters_million']:.2f}M)")
    print(f"[INFO] Trainable params: {model_info['trainable_parameters']:,} "
          f"({model_info['trainable_parameters_million']:.2f}M)")

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    # Use a lower LR for the pretrained backbone than the new heads
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    if has_aux_head(model):
        head_params += list(model.aux_classifier.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},   # 10x lower for backbone
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = MultiClassDiceLoss(num_classes=num_classes)
    scaler    = (
        torch.cuda.amp.GradScaler()
        if (args.mixed_precision and device.type == "cuda") else None
    )

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch    = 1
    best_val_miou  = 0.0

    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_miou = ckpt.get("best_val_miou", 0.0)
        print(f"[INFO] Resumed epoch {ckpt['epoch']}, best mIoU: {best_val_miou:.4f}")

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n[INFO] Starting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):

        # Unfreeze backbone mid-training if requested
        if (
            args.freeze_backbone
            and args.unfreeze_epoch is not None
            and epoch == args.unfreeze_epoch
        ):
            unfreeze_backbone(model)
            # Reset optimizer so backbone params get their own LR entry
            optimizer = optim.AdamW([
                {"params": list(model.backbone.parameters()), "lr": args.lr * 0.1},
                {"params": head_params,                        "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch
            )

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            epoch, writer, num_classes, scaler, args.aux_loss_weight
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, loss_fn, device, num_classes, split="Val"
        )

        # Log to TensorBoard
        for key, val in train_metrics.items():
            writer.add_scalar(f"Epoch/Train_{key}", val, epoch)
        for key, val in val_metrics.items():
            writer.add_scalar(f"Epoch/Val_{key}", val, epoch)
        writer.add_scalar("Epoch/LR_backbone", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Epoch/LR_heads",    optimizer.param_groups[1]["lr"], epoch)

        # Console summary
        print(f"\n[Epoch {epoch:03d}/{args.epochs}]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | mIoU: {train_metrics['miou']:.4f} | "
              f"Acc: {train_metrics['pixel_accuracy']:.4f}")
        print(f"          IoU [BG/Crop/Weed]: "
              f"[{train_metrics['iou_Background']:.3f}/"
              f"{train_metrics['iou_Crop']:.3f}/"
              f"{train_metrics['iou_Weed']:.3f}]")
        print(f"          F1  [BG/Crop/Weed]: "
              f"[{train_metrics['f1_Background']:.3f}/"
              f"{train_metrics['f1_Crop']:.3f}/"
              f"{train_metrics['f1_Weed']:.3f}]")

        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | mIoU: {val_metrics['miou']:.4f} | "
              f"Acc: {val_metrics['pixel_accuracy']:.4f}")
        print(f"          IoU [BG/Crop/Weed]: "
              f"[{val_metrics['iou_Background']:.3f}/"
              f"{val_metrics['iou_Crop']:.3f}/"
              f"{val_metrics['iou_Weed']:.3f}]")
        print(f"          F1  [BG/Crop/Weed]: "
              f"[{val_metrics['f1_Background']:.3f}/"
              f"{val_metrics['f1_Crop']:.3f}/"
              f"{val_metrics['f1_Weed']:.3f}]")
        print(f"          Prec/Rec: {val_metrics['mean_precision']:.4f}/{val_metrics['mean_recall']:.4f}")

        # Save best checkpoint
        if val_metrics["miou"] > best_val_miou:
            best_val_miou = val_metrics["miou"]
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_miou":        best_val_miou,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "config": config_dict,
            }, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"  [INFO] ✓ Saved best checkpoint (mIoU: {best_val_miou:.4f})")

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_miou":        best_val_miou,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "config": config_dict,
            }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

        scheduler.step()

    # ── Final test evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("[INFO] Running final test evaluation with best checkpoint...")
    print(f"{'='*80}\n")

    best_ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, loss_fn, device, num_classes, split="Test")

    print(f"[Test Results]")
    print(f"  Loss: {test_metrics['loss']:.4f} | IoU Loss: {test_metrics['iou_loss']:.4f}")
    print(f"  mIoU: {test_metrics['miou']:.4f} | Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")
    print(f"  IoU  [BG/Crop/Weed]: "
          f"[{test_metrics['iou_Background']:.4f}/"
          f"{test_metrics['iou_Crop']:.4f}/"
          f"{test_metrics['iou_Weed']:.4f}]")
    print(f"  F1   [BG/Crop/Weed]: "
          f"[{test_metrics['f1_Background']:.4f}/"
          f"{test_metrics['f1_Crop']:.4f}/"
          f"{test_metrics['f1_Weed']:.4f}]")
    print(f"  Prec [BG/Crop/Weed]: "
          f"[{test_metrics['precision_Background']:.4f}/"
          f"{test_metrics['precision_Crop']:.4f}/"
          f"{test_metrics['precision_Weed']:.4f}]")
    print(f"  Rec  [BG/Crop/Weed]: "
          f"[{test_metrics['recall_Background']:.4f}/"
          f"{test_metrics['recall_Crop']:.4f}/"
          f"{test_metrics['recall_Weed']:.4f}]")

    # Log & save test results
    for key, val in test_metrics.items():
        writer.add_scalar(f"Final/Test_{key}", val, 0)

    with open(os.path.join(run_dir, "test_results.json"), "w") as f:
        json.dump({
            **test_metrics,
            "best_val_miou": best_val_miou,
            "model_info": model_info,
        }, f, indent=2)

    writer.close()
    print(f"\n{'='*80}")
    print(f"[INFO] Training complete! Results saved to: {run_dir}")
    print(f"[INFO] Best Val mIoU : {best_val_miou:.4f}")
    print(f"[INFO] Test mIoU     : {test_metrics['miou']:.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()