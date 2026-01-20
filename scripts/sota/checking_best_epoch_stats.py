import torch 
checkpoint = torch.load("/home/vjtiadmin/Desktop/BTechGroup/WEEDSBL/scripts/sota/experiments/sugarbeets_deeplabsv3+_4ch_RGBNIR_20260118_205155/checkpoints/best_model.pth")

print(f"\n{'='*80}")
print(f"BEST MODEL CHECKPOINT DETAILS")
print(f"{'='*80}")
print(f"\nEpoch: {checkpoint['epoch']}")
print(f"Best Validation mIoU: {checkpoint['best_val_miou']:.4f}")

print(f"\n{'Overall Metrics':^80}")
print(f"{'-'*80}")
print(f"  Loss                : {checkpoint['val_loss']:.6f}")
print(f"  Pixel Accuracy      : {checkpoint['val_pixel_accuracy']:.4f} ({checkpoint['val_pixel_accuracy']*100:.2f}%)")
print(f"  Mean IoU            : {checkpoint['val_miou']:.4f}")
print(f"  Mean Precision      : {checkpoint['val_mean_precision']:.4f}")
print(f"  Mean Recall         : {checkpoint['val_mean_recall']:.4f}")
print(f"  Mean F1-Score       : {checkpoint['val_mean_f1']:.4f}")

print(f"\n{'Per-Class Metrics':^80}")
print(f"{'-'*80}")
print(f"{'Class':<15} {'IoU':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print(f"{'-'*80}")

class_names = ['Background', 'Crop', 'Weed']
for cls_name in class_names:
    iou = checkpoint[f'val_iou_{cls_name}']
    prec = checkpoint[f'val_precision_{cls_name}']
    rec = checkpoint[f'val_recall_{cls_name}']
    f1 = checkpoint[f'val_f1_{cls_name}']
    print(f"{cls_name:<15} {iou:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

print(f"{'='*80}\n")