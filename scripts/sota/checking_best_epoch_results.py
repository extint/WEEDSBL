import torch
import numpy as np

try:
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except:
    pass

checkpoint = torch.load(
    "/home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/dual_encoder_20260122_235537/checkpoints/best_model.pth",
    weights_only=False
)

print(f"\n{'='*80}")
print(f"BEST MODEL - DualEncoderAFFNet")
print(f"{'='*80}\n")

print(f"Epoch: {checkpoint['epoch']}")

if 'metrics' in checkpoint:
    m = checkpoint['metrics']
    print(f"\nResults:")
    print(f"  Train Loss : {m['train_loss']:.4f}")
    print(f"  Train IoU  : {m['train_iou']:.4f}")
    print(f"  Val Loss   : {m['val_loss']:.4f}")
    print(f"  Val IoU    : {m['val_iou']:.4f}")
    print(f"  LR         : {m['lr']:.2e}")

if 'model_state_dict' in checkpoint:
    total = sum(p.numel() for p in checkpoint['model_state_dict'].values())
    print(f"\nModel: {total/1e6:.2f}M parameters")

print(f"\n{'='*80}\n")
