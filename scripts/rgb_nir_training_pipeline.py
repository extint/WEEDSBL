
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
import numpy as np
from typing import Dict, Tuple, Optional
import random
from tqdm import tqdm

class DomainAdaptationLoss(nn.Module):
    """Combined loss functions for RGB-NIR domain adaptation"""
    def __init__(self, lambda_seg: float = 2.0, lambda_ent: float = 0.5, 
                 lambda_adv: float = 0.4, lambda_aux: float = 0.4, eta: float = 2.0):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_ent = lambda_ent  
        self.lambda_adv = lambda_adv
        self.lambda_aux = lambda_aux
        self.eta = eta  # Carbonnier penalty exponent

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def segmentation_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy segmentation loss for labeled source data"""
        return self.ce_loss(predictions, targets)

    def entropy_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Robust entropy minimization with Carbonnier penalty for target data"""
        # Convert logits to probabilities
        probs = F.softmax(predictions, dim=1)

        # Compute per-pixel entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [B, H, W]

        # Normalize entropy to [0, 1] range  
        num_classes = predictions.shape[1]
        entropy_normalized = entropy / np.log(num_classes)

        # Apply Carbonnier penalty (penalizes high entropy more when eta > 0.5)
        entropy_penalty = (entropy_normalized ** 2 + 0.00012) ** self.eta

        return entropy_penalty.mean()

    def adversarial_loss(self, disc_pred_target: torch.Tensor) -> torch.Tensor:
        """Adversarial loss to fool discriminator (target predictions should look like source)"""
        # Target: discriminator should classify target features as source (label=1)
        source_labels = torch.ones_like(disc_pred_target)
        return self.bce_loss(disc_pred_target, source_labels)

    def discriminator_loss(self, disc_pred_source: torch.Tensor, 
                          disc_pred_target: torch.Tensor) -> torch.Tensor:
        """Discriminator loss to distinguish source vs target"""
        source_labels = torch.ones_like(disc_pred_source)
        target_labels = torch.zeros_like(disc_pred_target)

        loss_source = self.bce_loss(disc_pred_source, source_labels)
        loss_target = self.bce_loss(disc_pred_target, target_labels)

        return loss_source + loss_target

    def forward(self, outputs_source: Dict, outputs_target: Dict, 
                source_labels: torch.Tensor, disc_outputs: Dict) -> Dict[str, torch.Tensor]:
        """Compute total combined loss"""
        losses = {}

        # Segmentation loss (source domain only)
        if source_labels is not None:
            losses['seg'] = self.segmentation_loss(outputs_source['segmentation'], source_labels)
        else:
            losses['seg'] = torch.tensor(0.0, device=outputs_source['segmentation'].device)

        # Entropy loss (target domain)
        losses['ent'] = self.entropy_loss(outputs_target['segmentation'])

        # Adversarial losses (generator tries to fool discriminators)
        losses['adv_enc'] = self.adversarial_loss(disc_outputs['enc_target'])
        losses['adv_dec'] = self.adversarial_loss(disc_outputs['dec_target'])
        losses['adv'] = losses['adv_enc'] + losses['adv_dec']

        # Discriminator losses
        losses['disc_enc'] = self.discriminator_loss(
            disc_outputs['enc_source'], disc_outputs['enc_target']
        )
        losses['disc_dec'] = self.discriminator_loss(
            disc_outputs['dec_source'], disc_outputs['dec_target']
        )
        losses['disc'] = losses['disc_enc'] + self.lambda_aux * losses['disc_dec']

        # Total generator loss (segmentation network)
        losses['gen_total'] = (
            self.lambda_seg * losses['seg'] +
            self.lambda_ent * losses['ent'] +
            self.lambda_adv * losses['adv']
        )

        return losses

class AugmentationScheduler:
    """Progressive augmentation scheduling as described in the paper"""
    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.3, 
                 lambda_aug: int = 20, total_epochs: int = 100):
        self.alpha = alpha  # Geometric augmentations
        self.beta = beta    # Noise augmentations
        self.gamma = gamma  # Collage augmentations
        self.lambda_aug = lambda_aug
        self.total_epochs = total_epochs

    def get_augmentation_probs(self, current_epoch: int) -> Dict[str, float]:
        """Get augmentation probabilities based on current epoch"""
        progress = current_epoch / self.total_epochs

        if current_epoch < self.lambda_aug:
            # Early epochs: only identity (no augmentation)
            return {'identity': 1.0, 'geometric': 0.0, 'noise': 0.0, 'collage': 0.0}
        else:
            # Progressive increase in augmentation probability
            geometric_prob = min(self.alpha, self.alpha * (current_epoch - self.lambda_aug) / (self.total_epochs - self.lambda_aug))
            noise_prob = min(self.beta, self.beta * (current_epoch - self.lambda_aug) / (self.total_epochs - self.lambda_aug))
            collage_prob = min(self.gamma, self.gamma * (current_epoch - self.lambda_aug) / (self.total_epochs - self.lambda_aug))

            identity_prob = max(0.0, 1.0 - geometric_prob - noise_prob - collage_prob)

            return {
                'identity': identity_prob,
                'geometric': geometric_prob,
                'noise': noise_prob,
                'collage': collage_prob
            }

class RGBNIRDomainAdaptationTrainer:
    """Complete training pipeline for RGB-NIR domain adaptation"""
    def __init__(self, model: nn.Module, device: torch.device, 
                 lr_seg: float = 0.001, lr_disc: float = 1e-4):
        self.model = model.to(device)
        self.device = device

        # Separate optimizers for segmentation network and discriminators
        seg_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
        disc_params = list(model.discriminator_encoder.parameters()) + list(model.discriminator_decoder.parameters())

        self.optimizer_seg = optim.SGD(seg_params, lr=lr_seg, momentum=0.9, weight_decay=5e-4)
        self.optimizer_disc = optim.Adam(disc_params, lr=lr_disc, betas=(0.9, 0.99))

        # Learning rate schedulers
        self.scheduler_seg = None  # Will be set when training starts
        self.scheduler_disc = None

        self.criterion = DomainAdaptationLoss()
        self.aug_scheduler = AugmentationScheduler()

    def setup_schedulers(self, total_epochs: int):
        """Setup learning rate schedulers"""
        self.scheduler_seg = CosineAnnealingLR(self.optimizer_seg, T_max=total_epochs)
        # Polynomial decay for discriminator
        self.scheduler_disc = optim.lr_scheduler.LambdaLR(
            self.optimizer_disc, 
            lr_lambda=lambda epoch: (1 - epoch / total_epochs) ** 0.9
        )

    def train_step(self, source_batch: Dict, target_batch: Dict, current_epoch: int) -> Dict[str, float]:
        """Single training step with domain adaptation"""
        self.model.train()

        # Move data to device
        source_imgs = source_batch['images'].to(self.device)
        source_labels = source_batch['labels'].to(self.device)
        target_imgs = target_batch['images'].to(self.device)

        batch_size = source_imgs.shape[0]

        # Forward pass through segmentation network
        outputs_source = self.model(source_imgs)
        outputs_target = self.model(target_imgs)

        # Forward pass through discriminators
        with torch.no_grad():
            # Get discriminator predictions for loss computation
            enc_source_pred = self.model.discriminator_encoder(outputs_source['encoder_features'])
            enc_target_pred = self.model.discriminator_encoder(outputs_target['encoder_features'])
            dec_source_pred = self.model.discriminator_decoder(outputs_source['decoder_features'])
            dec_target_pred = self.model.discriminator_decoder(outputs_target['decoder_features'])

        disc_outputs = {
            'enc_source': enc_source_pred,
            'enc_target': enc_target_pred, 
            'dec_source': dec_source_pred,
            'dec_target': dec_target_pred
        }

        # Compute losses
        losses = self.criterion(outputs_source, outputs_target, source_labels, disc_outputs)

        # Update segmentation network
        self.optimizer_seg.zero_grad()
        losses['gen_total'].backward()
        self.optimizer_seg.step()

        # Update discriminators
        self.optimizer_disc.zero_grad()

        # Fresh forward pass for discriminators (no gradients from segmentation network)
        with torch.no_grad():
            outputs_source = self.model(source_imgs)
            outputs_target = self.model(target_imgs)

        # Forward through discriminators with gradients
        enc_source_pred = self.model.discriminator_encoder(outputs_source['encoder_features'].detach())
        enc_target_pred = self.model.discriminator_encoder(outputs_target['encoder_features'].detach())
        dec_source_pred = self.model.discriminator_decoder(outputs_source['decoder_features'].detach())
        dec_target_pred = self.model.discriminator_decoder(outputs_target['decoder_features'].detach())

        disc_outputs_train = {
            'enc_source': enc_source_pred,
            'enc_target': enc_target_pred,
            'dec_source': dec_source_pred,
            'dec_target': dec_target_pred
        }

        # Compute discriminator loss
        disc_loss = self.criterion.discriminator_loss(
            disc_outputs_train['enc_source'], disc_outputs_train['enc_target']
        ) + self.criterion.lambda_aux * self.criterion.discriminator_loss(
            disc_outputs_train['dec_source'], disc_outputs_train['dec_target']
        )

        disc_loss.backward()
        self.optimizer_disc.step()

        # Convert losses to float for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['disc_total'] = disc_loss.item()

        return loss_dict

    def validate(self, val_loader, num_classes: int = 2) -> Dict[str, float]:
        """Validation step computing mIoU"""
        self.model.eval()

        total_intersection = torch.zeros(num_classes).to(self.device)
        total_union = torch.zeros(num_classes).to(self.device)

        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(images)
                predictions = torch.argmax(outputs['segmentation'], dim=1)

                # Compute IoU per class
                for cls in range(num_classes):
                    pred_mask = (predictions == cls)
                    true_mask = (labels == cls)

                    intersection = (pred_mask & true_mask).sum().float()
                    union = (pred_mask | true_mask).sum().float()

                    total_intersection[cls] += intersection
                    total_union[cls] += union

        # Compute mIoU
        ious = total_intersection / (total_union + 1e-8)
        miou = ious.mean().item()

        return {
            'miou': miou,
            'iou_per_class': ious.cpu().numpy().tolist()
        }

    def train(self, source_loader, target_loader, val_loader, 
              num_epochs: int = 100, log_interval: int = 10):
        """Complete training loop"""
        self.setup_schedulers(num_epochs)

        best_miou = 0.0

        print(f"Starting RGB-NIR Sugar Beet Domain Adaptation Training")
        print(f"Total epochs: {num_epochs}")
        print("=" * 60)

        for epoch in range(num_epochs):
            # Get augmentation probabilities for current epoch
            aug_probs = self.aug_scheduler.get_augmentation_probs(epoch)

            # Training
            epoch_losses = []
            progress_bar = tqdm(zip(source_loader, target_loader), 
                              desc=f'Epoch {epoch+1}/{num_epochs}')

            for batch_idx, (source_batch, target_batch) in enumerate(progress_bar):
                loss_dict = self.train_step(source_batch, target_batch, epoch)
                epoch_losses.append(loss_dict)

                # Update progress bar
                if batch_idx % log_interval == 0:
                    avg_gen_loss = np.mean([l['gen_total'] for l in epoch_losses[-log_interval:]])
                    avg_disc_loss = np.mean([l['disc_total'] for l in epoch_losses[-log_interval:]])
                    progress_bar.set_postfix({
                        'Gen Loss': f'{avg_gen_loss:.4f}',
                        'Disc Loss': f'{avg_disc_loss:.4f}',
                        'Aug': f"G:{aug_probs['geometric']:.2f}"
                    })

            # Update learning rates
            self.scheduler_seg.step()
            self.scheduler_disc.step()

            # Validation
            if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
                val_metrics = self.validate(val_loader)
                current_miou = val_metrics['miou']

                print(f"\nEpoch {epoch+1} Validation:")
                print(f"  mIoU: {current_miou:.4f}")
                print(f"  IoU per class: {val_metrics['iou_per_class']}")

                # Save best model
                if current_miou > best_miou:
                    best_miou = current_miou
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_seg_state_dict': self.optimizer_seg.state_dict(),
                        'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
                        'best_miou': best_miou,
                    }, 'best_rgb_nir_sugar_beet_model.pth')
                    print(f"  ✓ New best model saved (mIoU: {best_miou:.4f})")

            # Epoch summary
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([l[key] for l in epoch_losses])

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Segmentation Loss: {avg_losses['seg']:.4f}")
            print(f"  Entropy Loss: {avg_losses['ent']:.4f}")
            print(f"  Adversarial Loss: {avg_losses['adv']:.4f}")
            print(f"  Discriminator Loss: {avg_losses['disc_total']:.4f}")
            print(f"  Generator Total: {avg_losses['gen_total']:.4f}")
            print(f"  Learning Rate (Seg): {self.scheduler_seg.get_last_lr()[0]:.6f}")
            print(f"  Learning Rate (Disc): {self.scheduler_disc.get_last_lr()[0]:.6f}")

        print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")

# Example usage
if __name__ == "__main__":
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    from rgb_nir_sugar_beet_complete import RGBNIRSugarBeetSegmentationNetwork
    model = RGBNIRSugarBeetSegmentationNetwork(num_classes=2)

    # Initialize trainer
    trainer = RGBNIRDomainAdaptationTrainer(model, device)

    print("RGB-NIR Sugar Beet Domain Adaptation Trainer initialized")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Ready for training with your Sugar Beet 2016 dataset!")

    # Example training call (uncomment when you have data loaders):
    # trainer.train(source_loader, target_loader, val_loader, num_epochs=100)

