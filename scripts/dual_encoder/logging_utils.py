"""
Enhanced logging with layer-wise learning visualization and advanced monitoring.
"""

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict


class TrainingLogger:
    """
    Comprehensive training logger with layer-wise visualization.
    """
    def __init__(self, log_dir, model, val_loader, device, num_vis_samples=4):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # Create subdirectories
        self.img_dir = self.log_dir / 'inference_images'
        self.img_dir.mkdir(exist_ok=True)
        
        self.gate_dir = self.log_dir / 'attention_gates'
        self.gate_dir.mkdir(exist_ok=True)
        
        self.layer_dir = self.log_dir / 'layer_learning'
        self.layer_dir.mkdir(exist_ok=True)
        
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.model = model
        self.device = device
        
        # Track layer-wise statistics
        self.layer_stats_history = defaultdict(list)
        
        # Select fixed validation samples
        print(f"[Logger] Selecting {num_vis_samples} fixed validation samples...")
        self.vis_samples = self._select_vis_samples(val_loader, num_vis_samples)
        
        # ImageNet normalization stats
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        print(f"[Logger] Initialized. Logs saved to: {self.log_dir}")
    
    def _select_vis_samples(self, loader, num_samples):
        """Select fixed samples for visualization"""
        samples = []
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
            samples.append({
                'rgb': batch['rgb'][0:1].to(self.device),
                'nir': batch['nir'][0:1].to(self.device),
                'mask': batch['mask'][0].cpu().numpy(),
                'path': batch['path'][0] if 'path' in batch else f"sample_{i}"
            })
        return samples
    
    def log_epoch(self, epoch, train_loss, train_iou, val_loss, val_iou, lr):
        """Log scalar metrics to TensorBoard"""
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('IoU/train', train_iou, epoch)
        self.writer.add_scalar('IoU/val', val_iou, epoch)
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Also log to text file
        log_file = self.log_dir / 'training_log.txt'
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train IoU={train_iou:.4f}, "
                   f"Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}, LR={lr:.2e}\n")
    
    def log_layer_statistics(self, epoch):
        """
        Log detailed layer-wise statistics:
        - Weight norms
        - Gradient norms
        - Activation statistics
        - Dead neurons
        """
        stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Weight statistics
                weight_norm = param.data.norm().item()
                weight_mean = param.data.mean().item()
                weight_std = param.data.std().item()
                
                stats[f'{name}/weight_norm'] = weight_norm
                stats[f'{name}/weight_mean'] = weight_mean
                stats[f'{name}/weight_std'] = weight_std
                
                # Log to TensorBoard
                self.writer.add_scalar(f'Weights/{name}/norm', weight_norm, epoch)
                self.writer.add_scalar(f'Weights/{name}/mean', weight_mean, epoch)
                self.writer.add_scalar(f'Weights/{name}/std', weight_std, epoch)
                
                # Gradient statistics
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    
                    stats[f'{name}/grad_norm'] = grad_norm
                    stats[f'{name}/grad_mean'] = grad_mean
                    stats[f'{name}/grad_std'] = grad_std
                    
                    self.writer.add_scalar(f'Gradients/{name}/norm', grad_norm, epoch)
                    self.writer.add_scalar(f'Gradients/{name}/mean', grad_mean, epoch)
                    self.writer.add_scalar(f'Gradients/{name}/std', grad_std, epoch)
                    
                    # Gradient-to-weight ratio (learning rate effectiveness indicator)
                    if weight_norm > 0:
                        grad_weight_ratio = grad_norm / weight_norm
                        self.writer.add_scalar(f'Gradients/{name}/grad_weight_ratio', grad_weight_ratio, epoch)
        
        # Store history for later visualization
        for key, val in stats.items():
            self.layer_stats_history[key].append((epoch, val))
        
        return stats
    
    def visualize_layer_learning(self, epoch):
        """
        Create comprehensive layer-wise learning visualization:
        - Weight evolution over time
        - Gradient flow
        - Learning rate effectiveness per layer
        """
        # Group layers by module
        rgb_encoder_layers = [k for k in self.layer_stats_history.keys() if 'rgb_encoder' in k and 'weight_norm' in k]
        nir_encoder_layers = [k for k in self.layer_stats_history.keys() if 'nir_encoder' in k and 'weight_norm' in k]
        aff_layers = [k for k in self.layer_stats_history.keys() if 'aff' in k and 'weight_norm' in k]
        decoder_layers = [k for k in self.layer_stats_history.keys() if 'decoder' in k and 'weight_norm' in k]
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Layer-wise Learning Analysis - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # Helper function to plot layer norms
        def plot_layer_norms(ax, layer_keys, title, color):
            for key in layer_keys[:10]:  # Limit to first 10 layers for readability
                history = self.layer_stats_history[key]
                if history:
                    epochs, values = zip(*history)
                    layer_name = key.split('/')[0].split('.')[-1][:20]  # Shorten name
                    ax.plot(epochs, values, label=layer_name, alpha=0.7)
            
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Weight Norm', fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # Plot each module
        plot_layer_norms(axes[0, 0], rgb_encoder_layers, 'RGB Encoder Weight Evolution', 'blue')
        plot_layer_norms(axes[0, 1], nir_encoder_layers, 'NIR Encoder Weight Evolution', 'green')
        plot_layer_norms(axes[1, 0], aff_layers, 'AFF Fusion Weight Evolution', 'red')
        plot_layer_norms(axes[1, 1], decoder_layers, 'Decoder Weight Evolution', 'purple')
        
        plt.tight_layout()
        save_path = self.layer_dir / f'layer_learning_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Also log to TensorBoard
        self.writer.add_figure('Layer_Learning/weight_evolution', fig, epoch)
        plt.close()
        
        print(f"[Logger] Saved layer-wise learning visualization for epoch {epoch}")
    
    def visualize_gradient_flow(self, epoch):
        """
        Visualize gradient flow through the network.
        Helps detect vanishing/exploding gradients.
        """
        ave_grads = []
        max_grads= []
        layers = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and 'bias' not in name:
                layers.append(name.split('.')[-2][:15])  # Shorten layer name
                ave_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())
        
        if len(layers) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label='max-gradient')
        ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b", label='mean-gradient')
        ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        ax.set_xticks(range(0, len(ave_grads), 1))
        ax.set_xticklabels(layers, rotation=90, fontsize=8)
        ax.set_xlim(left=0, right=len(ave_grads))
        ax.set_ylim(bottom=-0.001, top=max(max_grads)*1.2)  # Zoom in on the lower gradient regions
        ax.set_xlabel("Layers", fontweight='bold')
        ax.set_ylabel("Gradient Magnitude", fontweight='bold')
        ax.set_title(f"Gradient Flow - Epoch {epoch}", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.layer_dir / f'gradient_flow_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        self.writer.add_figure('Layer_Learning/gradient_flow', fig, epoch)
        plt.close()
    
    def visualize_feature_maps(self, epoch, num_channels=8):
        """
        Visualize intermediate feature maps from different stages.
        Shows what the network is learning to detect.
        """
        self.model.eval()
        
        # Hook to capture feature maps
        feature_maps = {}
        
        def make_hook(name):
            def hook(module, input, output):
                feature_maps[name] = output.detach().cpu()
            return hook
        
        hooks = []
        # Register hooks on key layers
        if hasattr(self.model, 'rgb_encoder'):
            hooks.append(self.model.rgb_encoder.encoder.patch_embed1.register_forward_hook(make_hook('rgb_stage1')))
            hooks.append(self.model.rgb_encoder.encoder.patch_embed2.register_forward_hook(make_hook('rgb_stage2')))
        
        if hasattr(self.model, 'nir_encoder'):
            hooks.append(self.model.nir_encoder.layer1.register_forward_hook(make_hook('nir_stage1')))
            hooks.append(self.model.nir_encoder.layer2.register_forward_hook(make_hook('nir_stage2')))
        
        with torch.no_grad():
            sample = self.vis_samples[0]
            rgb = sample['rgb']
            nir = sample['nir']
            
            _ = self.model(rgb, nir)
            
            # Create visualization
            num_stages = len(feature_maps)
            fig, axes = plt.subplots(num_stages, num_channels, figsize=(num_channels*2, num_stages*2))
            
            if num_stages == 1:
                axes = axes[np.newaxis, :]
            
            for stage_idx, (stage_name, fmap) in enumerate(feature_maps.items()):
                # fmap shape varies, handle different cases
                if fmap.dim() == 3:  # (B, N, C) from transformer
                    fmap = fmap[0]  # (N, C)
                    # Reshape to approximate spatial dims
                    N, C = fmap.shape
                    H = W = int(np.sqrt(N))
                    if H * W == N:
                        fmap = fmap.transpose(0, 1).reshape(C, H, W)
                    else:
                        continue
                elif fmap.dim() == 4:  # (B, C, H, W)
                    fmap = fmap[0]  # (C, H, W)
                else:
                    continue
                
                # Select channels to display
                channels_to_show = min(num_channels, fmap.shape[0])
                
                for ch_idx in range(channels_to_show):
                    ax = axes[stage_idx, ch_idx]
                    channel_data = fmap[ch_idx].numpy()
                    im = ax.imshow(channel_data, cmap='viridis')
                    ax.axis('off')
                    if ch_idx == 0:
                        ax.set_ylabel(stage_name, fontsize=10, fontweight='bold')
                
                # Hide unused subplots
                for ch_idx in range(channels_to_show, num_channels):
                    axes[stage_idx, ch_idx].axis('off')
            
            plt.suptitle(f'Feature Maps - Epoch {epoch}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = self.layer_dir / f'feature_maps_epoch_{epoch:03d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            self.writer.add_figure('Layer_Learning/feature_maps', fig, epoch)
            plt.close()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    def visualize_predictions(self, epoch, save_individual=True):
        """Create comprehensive visualization of predictions."""
        self.model.eval()
        
        # Register hooks to capture attention gates
        gates = {}
        hooks = []
        
        def make_hook(stage_name):
            def hook(module, input, output):
                gates[stage_name] = module.last_gate.detach().cpu()
            return hook
        
        if hasattr(self.model, 'aff1'):
            hooks.append(self.model.aff1.register_forward_hook(make_hook('stage1')))
        if hasattr(self.model, 'aff2'):
            hooks.append(self.model.aff2.register_forward_hook(make_hook('stage2')))
        if hasattr(self.model, 'aff3'):
            hooks.append(self.model.aff3.register_forward_hook(make_hook('stage3')))
        
        with torch.no_grad():
            for idx, sample in enumerate(self.vis_samples):
                rgb = sample['rgb']
                nir = sample['nir']
                mask = sample['mask']
                
                # Forward pass
                logits = self.model(rgb, nir)
                pred = torch.sigmoid(logits).cpu().numpy()[0, 0]
                
                # Denormalize RGB for visualization
                rgb_vis = self._denormalize_rgb(rgb)
                nir_vis = nir[0, 0].cpu().numpy()
                
                # Get finest attention gate
                if 'stage1' in gates:
                    gate = gates['stage1'][0].mean(dim=0).numpy()
                    H, W = rgb_vis.shape[:2]
                    gate_resized = cv2.resize(gate, (W, H), interpolation=cv2.INTER_LINEAR)
                else:
                    gate_resized = np.zeros_like(mask)
                
                # Create visualization
                fig = self._create_prediction_figure(
                    rgb_vis, nir_vis, mask, pred, gate_resized, epoch, idx
                )
                
                # Save to disk
                save_path = self.img_dir / f"epoch_{epoch:03d}_sample_{idx}.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                
                # Log to TensorBoard
                self.writer.add_figure(f'Predictions/sample_{idx}', fig, epoch)
                
                plt.close(fig)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print(f"[Logger] Saved prediction visualizations for epoch {epoch}")
    
    def _create_prediction_figure(self, rgb, nir, gt, pred, gate, epoch, sample_idx):
        """Create detailed prediction visualization"""
        pred_binary = (pred > 0.5).astype(np.uint8)
        
        # Compute metrics
        intersection = (pred_binary & gt).sum()
        union = (pred_binary | gt).sum()
        iou = intersection / (union + 1e-6)
        
        # Compute errors
        fp = ((pred_binary == 1) & (gt == 0)).sum()
        fn = ((pred_binary == 0) & (gt == 1)).sum()
        tp = ((pred_binary == 1) & (gt == 1)).sum()
        tn = ((pred_binary == 0) & (gt == 0)).sum()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Inputs
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rgb)
        ax1.set_title("RGB Input", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(nir, cmap='gray')
        ax2.set_title("NIR Input", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        gt_colored = self._mask_to_color(gt)
        ax3.imshow(gt_colored)
        ax3.set_title("Ground Truth\n(Green=Crop, Red=Weed)", fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Row 2: Predictions
        ax4 = fig.add_subplot(gs[1, 0])
        pred_colored = self._mask_to_color(pred_binary)
        ax4.imshow(pred_colored)
        ax4.set_title(f"Prediction\nIoU: {iou:.3f}", fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(pred, cmap='RdYlGn', vmin=0, vmax=1)
        ax5.set_title("Confidence Map", fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        ax6 = fig.add_subplot(gs[1, 2])
        error_map = self._create_error_map(pred_binary, gt)
        ax6.imshow(error_map)
        ax6.set_title(f"Error Map\nFP:{fp}, FN:{fn}", fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # Row 3: Attention and Metrics
        ax7 = fig.add_subplot(gs[2, 0])
        im7 = ax7.imshow(gate, cmap='RdYlGn', vmin=0, vmax=1)
        ax7.set_title(f"Attention Gate\nMean: {gate.mean():.3f}", fontsize=12, fontweight='bold')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.imshow(rgb)
        ax8.imshow(gate, cmap='RdYlGn', alpha=0.5, vmin=0, vmax=1)
        ax8.set_title("Gate Overlay", fontsize=12, fontweight='bold')
        ax8.axis('off')
        
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        metrics_text = f"""
Epoch: {epoch}  Sample: {sample_idx}

Metrics:
  IoU:       {iou:.4f}
  Precision: {precision:.4f}
  Recall:    {recall:.4f}
  F1:        {f1:.4f}

Confusion:
  TP: {tp:,}  FP: {fp:,}
  FN: {fn:,}  TN: {tn:,}

Gate: μ={gate.mean():.3f}, σ={gate.std():.3f}
        """
        ax9.text(0.1, 0.95, metrics_text.strip(), 
                transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.suptitle(f"Analysis - Epoch {epoch}, Sample {sample_idx}", 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def visualize_attention_gates_only(self, epoch):
        """Visualize attention gates across stages"""
        self.model.eval()
        
        gates = {}
        hooks = []
        
        def make_hook(stage_name):
            def hook(module, input, output):
                gates[stage_name] = module.last_gate.detach().cpu()
            return hook
        
        if hasattr(self.model, 'aff1'):
            hooks.append(self.model.aff1.register_forward_hook(make_hook('stage1')))
        if hasattr(self.model, 'aff2'):
            hooks.append(self.model.aff2.register_forward_hook(make_hook('stage2')))
        if hasattr(self.model, 'aff3'):
            hooks.append(self.model.aff3.register_forward_hook(make_hook('stage3')))
        
        with torch.no_grad():
            for idx, sample in enumerate(self.vis_samples):
                rgb = sample['rgb']
                nir = sample['nir']
                
                _ = self.model(rgb, nir)
                
                rgb_vis = self._denormalize_rgb(rgb)
                nir_vis = nir[0, 0].cpu().numpy()
                
                num_stages = len(gates)
                fig, axes = plt.subplots(2, num_stages + 1, figsize=(4*(num_stages+1), 8))
                
                axes[0, 0].imshow(rgb_vis)
                axes[0, 0].set_title("RGB", fontweight='bold')
                axes[0, 0].axis('off')
                
                axes[1, 0].imshow(nir_vis, cmap='gray')
                axes[1, 0].set_title("NIR", fontweight='bold')
                axes[1, 0].axis('off')
                
                for col, (stage_name, gate_tensor) in enumerate(gates.items(), start=1):
                    gate = gate_tensor[0].mean(dim=0).numpy()
                    H, W = rgb_vis.shape[:2]
                    gate_resized = cv2.resize(gate, (W, H))
                    
                    im = axes[0, col].imshow(gate_resized, cmap='RdYlGn', vmin=0, vmax=1)
                    axes[0, col].set_title(f"{stage_name}\nμ={gate.mean():.3f}", fontweight='bold')
                    axes[0, col].axis('off')
                    plt.colorbar(im, ax=axes[0, col], fraction=0.046)
                    
                    axes[1, col].imshow(rgb_vis)
                    axes[1, col].imshow(gate_resized, cmap='RdYlGn', alpha=0.5, vmin=0, vmax=1)
                    axes[1, col].set_title(f"Overlay", fontweight='bold')
                    axes[1, col].axis('off')
                
                plt.suptitle(f"Attention Gates - Epoch {epoch}, Sample {idx}", fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                save_path = self.gate_dir / f"epoch_{epoch:03d}_sample_{idx}_gates.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        for hook in hooks:
            hook.remove()
        
        print(f"[Logger] Saved attention gates for epoch {epoch}")
    
    def log_confusion_matrix(self, epoch, all_preds, all_targets):
        """Log confusion matrix"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Crop', 'Weed'], 
                   yticklabels=['Crop', 'Weed'], ax=ax)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
        
        self.writer.add_figure('Metrics/confusion_matrix', fig, epoch)
        plt.close()
    
    def log_pr_curve(self, epoch, all_probs, all_targets):
        """Log PR curve"""
        from sklearn.metrics import precision_recall_curve, auc
        
        precision, recall, _ = precision_recall_curve(all_targets, all_probs)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'AUC={pr_auc:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'PR Curve - Epoch {epoch}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.writer.add_figure('Metrics/pr_curve', fig, epoch)
        self.writer.add_scalar('Metrics/PR_AUC', pr_auc, epoch)
        plt.close()
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, metrics, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"[Logger] ✓ Saved best model at epoch {epoch}")
        
        # Keep only last 3 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if len(checkpoints) > 3:
            for old in checkpoints[:-3]:
                old.unlink()
    
    def _denormalize_rgb(self, rgb_tensor):
        """Undo ImageNet normalization - FIXED"""
        # Handle both (3, H, W) and (B, 3, H, W)
        if rgb_tensor.dim() == 4:
            rgb_tensor = rgb_tensor[0]
        
        mean = self.rgb_mean.squeeze().to(rgb_tensor.device)
        std = self.rgb_std.squeeze().to(rgb_tensor.device)
        
        rgb = rgb_tensor * std[:, None, None] + mean[:, None, None]
        rgb = torch.clamp(rgb, 0, 1)
        return rgb.permute(1, 2, 0).cpu().numpy()
    
    def _mask_to_color(self, mask):
        """Convert mask to color"""
        colored = np.zeros((*mask.shape, 3))
        colored[mask == 0] = [0.2, 0.8, 0.2]
        colored[mask == 1] = [0.9, 0.2, 0.2]
        return colored
    
    def _create_error_map(self, pred, gt):
        """Create error map"""
        error_map = np.ones((*pred.shape, 3))
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)
        correct = (pred == gt)
        
        error_map[fp] = [0, 1, 1]  # cyan
        error_map[fn] = [1, 1, 0]  # yellow
        error_map[correct] = [0.9, 0.9, 0.9]
        
        return error_map
    
    def close(self):
        """Close writer"""
        self.writer.close()
        print(f"[Logger] Closed. Logs: {self.log_dir}")
