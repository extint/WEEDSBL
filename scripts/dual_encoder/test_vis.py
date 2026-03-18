import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


class RiceFieldAFFVisualizer:
    """
    Visualize AFF gates overlaid on actual rice field images.
    Shows WHERE and WHY model prefers RGB vs NIR in real scenes.
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.gates = {}
        self.hooks = []
        
    def register_hooks(self):
        def make_hook(stage_name):
            def hook(module, input, output):
                self.gates[stage_name] = module.last_gate.detach().cpu()
            return hook
        
        self.hooks.append(self.model.aff2.register_forward_hook(make_hook('stage2')))
        self.hooks.append(self.model.aff3.register_forward_hook(make_hook('stage3')))
        self.hooks.append(self.model.aff4.register_forward_hook(make_hook('stage4')))
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def visualize_field_attention(self, rgb, nir, mask, pred=None, save_path=None):
        """
        Create rice-field specific visualization:
        Row 1: RGB | NIR | Ground Truth Mask
        Row 2: Prediction | Gate Overlay on RGB | Gate with semantic zones
        Row 3: RGB regions analysis | NIR regions analysis | Interpretation
        """
        # Denormalize and prepare images
        rgb_img = self._denormalize_rgb(rgb[0])  # (H, W, 3)
        nir_img = nir[0, 0].cpu().numpy()
        mask_img = mask[0].cpu().numpy()
        
        # Get prediction if not provided
        if pred is None:
            with torch.no_grad():
                logits = self.model(rgb, nir)
                pred = torch.sigmoid(logits).cpu().numpy()[0, 0]
        else:
            pred = pred[0, 0] if pred.ndim == 4 else pred[0]
        
        # Get finest resolution gate (stage 2)
        gate = self.gates['stage2'][0].mean(dim=0).numpy()  # Average over channels
        H, W = rgb_img.shape[:2]
        gate_resized = cv2.resize(gate, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ===== ROW 1: Inputs and GT =====
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rgb_img)
        ax1.set_title("RGB Input\n(Structural info: rows, texture)", fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(nir_img, cmap='gray')
        ax2.set_title("NIR Input\n(Physiological: vegetation vigor)", fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        # Show GT with color: green=crop, red=weed
        gt_colored = np.zeros((H, W, 3))
        gt_colored[mask_img == 0] = [0.2, 0.8, 0.2]  # crop = green
        gt_colored[mask_img == 1] = [0.9, 0.2, 0.2]  # weed = red
        ax3.imshow(gt_colored)
        ax3.set_title("Ground Truth\n(Green=Crop, Red=Weed)", fontsize=11, fontweight='bold')
        ax3.axis('off')
        
        # ===== ROW 2: Prediction and Attention Gates =====
        ax4 = fig.add_subplot(gs[1, 0])
        pred_binary = (pred > 0.5).astype(np.uint8)
        pred_colored = np.zeros((H, W, 3))
        pred_colored[pred_binary == 0] = [0.2, 0.8, 0.2]
        pred_colored[pred_binary == 1] = [0.9, 0.2, 0.2]
        ax4.imshow(pred_colored)
        ax4.set_title("Model Prediction\n(Green=Crop, Red=Weed)", fontsize=11, fontweight='bold')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(rgb_img)
        im5 = ax5.imshow(gate_resized, cmap='RdYlGn', alpha=0.6, vmin=0, vmax=1)
        ax5.set_title("Attention Gate on RGB\n(Green=Uses RGB, Red=Uses NIR)", fontsize=11, fontweight='bold')
        ax5.axis('off')
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.set_label('RGB preference →', rotation=270, labelpad=15)
        
        ax6 = fig.add_subplot(gs[1, 2])
        # Semantic segmentation of gate
        semantic_viz = self._create_semantic_gate_viz(gate_resized, nir_img, rgb_img)
        ax6.imshow(semantic_viz)
        ax6.set_title("Modality Usage by Region\n(Where RGB/NIR dominate)", fontsize=11, fontweight='bold')
        ax6.axis('off')
        
        # ===== ROW 3: Regional Analysis =====
        ax7 = fig.add_subplot(gs[2, 0])
        # Highlight where RGB is preferred (gate > 0.6)
        rgb_mask = gate_resized > 0.6
        rgb_highlight = rgb_img.copy()
        rgb_highlight[~rgb_mask] = rgb_highlight[~rgb_mask] * 0.3
        ax7.imshow(rgb_highlight)
        ax7.contour(rgb_mask, colors='lime', linewidths=2, levels=[0.5])
        ax7.set_title("RGB-Dominant Regions\n(Structure, boundaries, rows)", fontsize=11, fontweight='bold')
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[2, 1])
        # Highlight where NIR is preferred (gate < 0.4)
        nir_mask = gate_resized < 0.4
        nir_highlight = np.stack([nir_img]*3, axis=-1)
        nir_highlight[~nir_mask] = nir_highlight[~nir_mask] * 0.3
        ax8.imshow(nir_highlight, cmap='gray')
        ax8.contour(nir_mask, colors='red', linewidths=2, levels=[0.5])
        ax8.set_title("NIR-Dominant Regions\n(Vegetation health, stress)", fontsize=11, fontweight='bold')
        ax8.axis('off')
        
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # Interpretation text with statistics
        rgb_pct = (gate_resized > 0.6).sum() / gate_resized.size * 100
        nir_pct = (gate_resized < 0.4).sum() / gate_resized.size * 100
        mixed_pct = 100 - rgb_pct - nir_pct
        
        interpretation = f"""
Fusion Strategy Analysis:

RGB Preference: {rgb_pct:.1f}%
• Field boundaries
• Crop row structure  
• Texture patterns

NIR Preference: {nir_pct:.1f}%
• Vegetation regions
• Health/stress detection
• Weed vigor

Mixed Fusion: {mixed_pct:.1f}%
• Transition zones
• Complex patterns

Gate mean: {gate_resized.mean():.3f}
Gate std: {gate_resized.std():.3f}
        """
        
        ax9.text(0.05, 0.95, interpretation.strip(), 
                transform=ax9.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle("Dual-Encoder AFF: Modality-Aware Fusion in Rice Field", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
    
    def _create_semantic_gate_viz(self, gate, nir_img, rgb_img):
        """Create semantic visualization showing different fusion zones"""
        H, W = gate.shape
        viz = np.zeros((H, W, 3))
        
        # Strong RGB regions (green): structure-dominated
        strong_rgb = gate > 0.65
        viz[strong_rgb] = [0.2, 0.8, 0.2]
        
        # Strong NIR regions (red): vegetation-dominated
        strong_nir = gate < 0.35
        viz[strong_nir] = [0.9, 0.2, 0.2]
        
        # Mixed regions (yellow): balanced fusion
        mixed = (gate >= 0.35) & (gate <= 0.65)
        viz[mixed] = [0.9, 0.9, 0.2]
        
        return viz
    
    def _denormalize_rgb(self, rgb_tensor):
        """Undo ImageNet normalization"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = rgb_tensor * std + mean
        rgb = torch.clamp(rgb, 0, 1)
        return rgb.permute(1, 2, 0).cpu().numpy()


    def create_multi_sample_story(self, test_loader, num_samples=3, save_dir="figures/"):
        """
        Create storytelling visualizations across multiple samples
        showing different fusion behaviors
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
        samples_processed = 0
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if samples_processed >= num_samples:
                    break
                
                rgb = batch["rgb"].to(self.device)
                nir = batch["nir"].to(self.device)
                mask = batch["mask"]
                
                # Forward pass
                logits = self.model(rgb, nir)
                pred = torch.sigmoid(logits).cpu()
                
                # Visualize
                save_path = os.path.join(save_dir, f"field_attention_sample_{i+1}.png")
                self.visualize_field_attention(rgb, nir, mask, pred, save_path=save_path)
                
                samples_processed += 1
                print(f"[INFO] Saved visualization {samples_processed}/{num_samples}")



# Import your dual encoder model
from dual_encoder.architecture import DualEncoderAFFNet

# Import data loader
from dual_encoder.dual_encoder_data_loader import create_dual_encoder_dataloaders

model = DualEncoderAFFNet(rgb_base_ch=32, nir_base_ch=16, num_classes=1, embed_dim=64)
model.load_state_dict(torch.load("checkpoints_dual_encoder/best_dual_encoder.pth"))
model = model.to('cuda')
model.eval()

# Load test data
_, _, test_loader = create_dual_encoder_dataloaders(
    data_root="/home/vjti-comp/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima(1)/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB",
    batch_size=1,
    target_size=(512, 512)
)

# Create visualizer
visualizer = RiceFieldAFFVisualizer(model, device='cuda')
visualizer.register_hooks()

# Generate visualizations for multiple samples
visualizer.create_multi_sample_story(test_loader, num_samples=5, save_dir="figures/aff_analysis/")

visualizer.remove_hooks()
