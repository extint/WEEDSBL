import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

# Import your components
from scripts.dual_encoder.weedcrop_classifier import CropWeedDataset, compute_metrics  # Replace with actual import path
from dual_encoder.updated_architecture import DualEncoderAFFNet

def visualize_predictions(model, val_loader, device, num_samples=8, save_dir="inference_results"):
    """Run inference on validation set and save visualizations"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    crop_ious, weed_ious = [], []
    crop_accs, weed_accs = [], []
    pix_accs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Inference")):
            if batch_idx * val_loader.batch_size >= num_samples:
                break
                
            rgb = batch["rgb"].to(device)
            nir = batch["nir"].to(device)
            mask = batch["mask"].to(device)
            
            logits = model(rgb, nir)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            
            # Compute metrics
            ious, accs, pix_acc = compute_metrics(pred, mask)
            crop_ious.append(ious['crop'])
            weed_ious.append(ious['weed'])
            crop_accs.append(accs['crop'])
            weed_accs.append(accs['weed'])
            pix_accs.append(pix_acc)
            
            # Visualize
            for i in range(rgb.shape[0]):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'Sample {batch_idx*val_loader.batch_size + i}', fontsize=16)
                
                # RGB
                rgb_img = rgb[i].cpu().numpy().transpose(1,2,0)
                rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                rgb_img = np.clip(rgb_img, 0, 1)
                axes[0,0].imshow(rgb_img)
                axes[0,0].set_title('RGB')
                axes[0,0].axis('off')
                
                # NIR
                nir_img = nir[i,0].cpu().numpy()
                nir_img = cv2.normalize(nir_img, None, 0, 1, cv2.NORM_MINMAX)
                axes[0,1].imshow(nir_img, cmap='gray')
                axes[0,1].set_title('NIR')
                axes[0,1].axis('off')
                
                # GT Mask
                gt_mask = mask[i].cpu().numpy()
                gt_vis = np.zeros((*gt_mask.shape, 3))
                gt_vis[gt_mask == 0] = [0, 1, 0]    # Green = crop
                gt_vis[gt_mask == 1] = [1, 0, 0]    # Red = weed
                gt_vis[gt_mask == 255] = [0.5, 0.5, 0.5]  # Gray = ignore
                axes[0,2].imshow(gt_vis)
                axes[0,2].set_title('GT Mask\n(Green=crop, Red=weed)')
                axes[0,2].axis('off')
                
                # Predicted Mask
                pred_vis = np.zeros((*pred[i].cpu().numpy().shape, 3))
                pred_vis[pred[i].cpu().numpy() == 0] = [0, 1, 0]    # Green = crop
                pred_vis[pred[i].cpu().numpy() == 1] = [1, 0, 0]    # Red = weed
                axes[1,0].imshow(pred_vis)
                axes[1,0].set_title('Predicted Mask')
                axes[1,0].axis('off')
                
                # Overlay
                overlay = rgb_img.copy()
                overlay[pred[i].cpu().numpy() == 1] = [1, 0.3, 0.3]  # Red overlay on weeds (semi-transparent)
                axes[1,1].imshow(overlay)
                axes[1,1].set_title('RGB + Pred Overlay\n(Red=predicted weed)')
                axes[1,1].axis('off')
                
                # Metrics
                axes[1,2].axis('off')
                metrics_text = f"""Metrics (veg only):
Crop IoU: {ious['crop']:.3f}
Weed IoU: {ious['weed']:.3f}
Crop Acc: {accs['crop']:.3f}
Weed Acc: {accs['weed']:.3f}
Pixel Acc: {pix_acc:.3f}"""
                axes[1,2].text(0.05, 0.95, metrics_text, transform=axes[1,2].transAxes,
                             fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/sample_{batch_idx*val_loader.batch_size + i:03d}.png",
                          dpi=150, bbox_inches='tight')
                plt.close()
    
    # FIXED: Print summary metrics correctly
    print(f"\n=== Inference Summary ({len(crop_ious)} samples) ===")
    print(f"Avg Crop IoU:  {np.mean(crop_ious):.4f} ± {np.std(crop_ious):.4f}")
    print(f"Avg Weed IoU:  {np.mean(weed_ious):.4f} ± {np.std(weed_ious):.4f}")
    print(f"Mean IoU:      {np.mean([np.mean(crop_ious), np.mean(weed_ious)]):.4f}")
    print(f"Avg Crop Acc:  {np.mean(crop_accs):.4f}")
    print(f"Avg Weed Acc:  {np.mean(weed_accs):.4f}")
    print(f"Avg Pixel Acc: {np.mean(pix_accs):.4f}")
    print(f"Results saved to: {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Crop vs Weed Inference")
    parser.add_argument("--data_root", required=True, help="Path to dataset")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to visualize")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--img_size", nargs=2, default=[512, 512], type=int, help="Image size H W")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create validation dataloader (same as training)
    val_dataset = CropWeedDataset(
        root=args.data_root,
        split="val",
        target_size=tuple(args.img_size),
        augment=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Validation set size: {len(val_dataset)}")
    
    # Load model (same architecture as training)
    model = DualEncoderAFFNet(
        rgb_variant='small',
        nir_base_ch=20,
        num_classes=2,  # crop vs weed
        embed_dim=96
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded model from: {args.model_path}")
    
    # Run inference and visualization
    visualize_predictions(
        model, val_loader, device,
        num_samples=args.num_samples,
        save_dir="crop_weed_inference"
    )

if __name__ == "__main__":
    main()
