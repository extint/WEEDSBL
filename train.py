import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import numpy as np
import torch
from torch import optim
from datasets.weedyrice import create_weedy_rice_rgbnir_dataloaders
from models.unet import UNet
from models.losses import BCEDiceLoss
import yaml
import sys
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ious, f1s = [], []
    for batch in loader:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()          # (B,H,W), values {0,1}
        logits = model(x)                               # (B,1,H,W)
        probs = torch.sigmoid(logits)                  # (B,1,H,W)
        preds = (probs > 0.5).long().squeeze(1)        # (B,H,W)

        inter = (preds & y).sum(dim=(1, 2))            # per-sample
        union = (preds | y).sum(dim=(1, 2)).clamp_min(1)
        iou = (inter.float() / union.float()).mean().item()
        ious.append(iou)

        tp = (preds & y).sum(dim=(1, 2)).float()
        fp = (preds & (1 - y)).sum(dim=(1, 2)).float()
        fn = ((1 - preds) & y).sum(dim=(1, 2)).float()
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f1   = (2 * prec * rec / (prec + rec + 1e-6)).mean().item()
        f1s.append(f1)

    return {"mIoU": float(np.mean(ious) if ious else 0.0),
            "F1":   float(np.mean(f1s) if f1s else 0.0)}

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None) -> float:
    model.train()
    epoch_loss = 0.0
    for batch in loader:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).unsqueeze(1)  # (B,1,H,W)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / max(1, len(loader))

def main():
    config_path = "train_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = create_weedy_rice_rgbnir_dataloaders(
        data_root=config["data_root"],
        use_rgbnir=True if config["in_channels"] == 4 else False,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        target_size=(config["height"], config["width"])
    )
    model = UNet(in_channels=config["in_channels"], base_ch=4, out_channels=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    loss_fn = BCEDiceLoss(bce_weight=0.5)
    scaler = torch.cuda.amp.GradScaler() if (config["mixed_precision"] and device.type == "cuda") else None

    writer = SummaryWriter(log_dir=os.path.join(config["ckpt_dir"], "train_logs", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    best_miou = 0.0
    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        metrics = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] Loss: {train_loss:.4f} | Val mIoU: {metrics['mIoU']:.4f} | Val F1: {metrics['F1']:.4f}")

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("mIoU/val", metrics["mIoU"], epoch)
        writer.add_scalar("F1/val", metrics["F1"], epoch)

        # Save best
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            ckpt_path = os.path.join(config["ckpt_dir"], f"{type(model).__name__}_{config['in_channels']}ch_best.pth")
            torch.save({"model": model.state_dict(),
                        "loss": train_loss,
                        "miou": best_miou,
                        "F1": metrics["F1"],
                        "train_config": config}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final test
    test_metrics = evaluate(model, test_loader, device)
    print(f"[Test] mIoU: {test_metrics['mIoU']:.4f} | F1: {test_metrics['F1']:.4f}")

    # TensorBoard logging for test
    writer.add_scalar("mIoU/test", test_metrics["mIoU"])
    writer.add_scalar("F1/test", test_metrics["F1"])
    writer.close()

if __name__ == "__main__":
    main()