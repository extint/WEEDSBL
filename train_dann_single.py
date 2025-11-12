# train_dann_single.py
import os, sys, yaml, numpy as np, torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from models.dannet import DANNFromMANet
from models.losses import BCEDiceLoss
from datasets.weedyrice import create_weedy_rice_rgbnir_dataloaders  # same as your train.py

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ious, f1s = [], []
    for batch in loader:
        x = batch["images"].to(device)
        y = batch["labels"].to(device).long()
        logits = model(x)  # eval: returns logits only
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().squeeze(1)
        inter = (preds & y).sum(dim=(1, 2))
        union = (preds | y).sum(dim=(1, 2)).clamp_min(1)
        ious.append((inter.float() / union.float()).mean().item())
        tp = (preds & y).sum(dim=(1, 2)).float()
        fp = (preds & (1 - y)).sum(dim=(1, 2)).float()
        fn = ((1 - preds) & y).sum(dim=(1, 2)).float()
        prec = tp / (tp + fp + 1e-6); rec = tp / (tp + fn + 1e-6)
        f1s.append((2 * prec * rec / (prec + rec + 1e-6)).mean().item())
    return {"mIoU": float(np.mean(ious) if ious else 0.0), "F1": float(np.mean(f1s) if f1s else 0.0)}

def main():
    cfg_path = "train_config.yaml" if len(sys.argv) == 1 else sys.argv[1]
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    # Single dataset: train=source (labeled), test=target (unlabeled for domain)
    src_train, src_val, src_test = create_weedy_rice_rgbnir_dataloaders(
        data_root=cfg["data_root"], use_rgbnir=(cfg["in_channels"] == 4),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        target_size=(cfg["height"], cfg["width"]),
        nir_drop_prob=cfg.get("nir_dropout_prob", 0.0),
        test_on_rgb_only=cfg.get("test_on_rgb_only", False),
        use_ndvi=cfg.get("use_ndvi", True)
    )

    model = DANNFromMANet(
        in_channels=cfg["in_channels"],
        base_ch=cfg.get("base_channels", 16),
        out_channels=1,
        grl_lambda=cfg.get("grl_lambda", 1.0)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    loss_seg = BCEDiceLoss(bce_weight=0.5)
    loss_dom = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if (cfg.get("mixed_precision", False) and device.type == "cuda") else None
    writer = SummaryWriter(log_dir=os.path.join(cfg["ckpt_dir"], "train_logs_dann_single", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    best_miou = 0.0
    src_iter, tgt_iter = iter(src_train), iter(src_test)  # test as target

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss_epoch = 0.0

        for step in range(len(src_train)):
            try: s = next(src_iter)
            except StopIteration: src_iter = iter(src_train); s = next(src_iter)
            try: t = next(tgt_iter)
            except StopIteration: tgt_iter = iter(src_test); t = next(tgt_iter)

            xs, ys = s["images"].to(device), s["labels"].to(device).unsqueeze(1)
            xt = t["images"].to(device)

            optimizer.zero_grad(set_to_none=True)
            dom_w = cfg.get("domain_loss_weight", 0.1)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    seg_logits_s, dom_logits_s = model(xs)
                    seg_loss = loss_seg(seg_logits_s, ys)
                    dom_loss_s = loss_dom(dom_logits_s, torch.zeros(dom_logits_s.size(0), dtype=torch.long, device=device))
                    _, dom_logits_t = model(xt)
                    dom_loss_t = loss_dom(dom_logits_t, torch.ones(dom_logits_t.size(0), dtype=torch.long, device=device))
                    total = seg_loss + dom_w * (dom_loss_s + dom_loss_t)
                scaler.scale(total).backward()
                scaler.step(optimizer); scaler.update()
            else:
                seg_logits_s, dom_logits_s = model(xs)
                seg_loss = loss_seg(seg_logits_s, ys)
                dom_loss_s = loss_dom(dom_logits_s, torch.zeros(dom_logits_s.size(0), dtype=torch.long, device=device))
                _, dom_logits_t = model(xt)
                dom_loss_t = loss_dom(dom_logits_t, torch.ones(dom_logits_t.size(0), dtype=torch.long, device=device))
                total = seg_loss + dom_w * (dom_loss_s + dom_loss_t)
                total.backward(); optimizer.step()

            total_loss_epoch += float(total.item())

        metrics = evaluate(model, src_val, device)
        print(f"[Epoch {epoch:03d}] Loss: {total_loss_epoch / max(1, len(src_train)):.4f} | Val mIoU: {metrics['mIoU']:.4f} | F1: {metrics['F1']:.4f}")
        writer.add_scalar("Loss/train_total", total_loss_epoch / max(1, len(src_train)), epoch)
        writer.add_scalar("mIoU/val_src", metrics["mIoU"], epoch); writer.add_scalar("F1/val_src", metrics["F1"], epoch)

        if metrics["mIoU"] >= best_miou:
            best_miou = metrics["mIoU"]
            ckpt_path = os.path.join(cfg["ckpt_dir"], f"DANNFromMANet_{cfg['in_channels']}ch_best.pth")
            torch.save({"model": model.state_dict(), "miou": best_miou, "epoch": epoch, "train_config": cfg}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    test_metrics = evaluate(model, src_test, device)
    print(f"[Test] mIoU: {test_metrics['mIoU']:.4f} | F1: {test_metrics['F1']:.4f}")
    writer.add_scalar("mIoU/test_src", test_metrics["mIoU"]); writer.add_scalar("F1/test_src", test_metrics["F1"]); writer.close()

if __name__ == "__main__":
    main()
