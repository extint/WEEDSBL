# 1. Initialize model
from rgb_nir_sugar_beet_complete import RGBNIRSugarBeetSegmentationNetwork
from rgb_nir_training_pipeline import RGBNIRDomainAdaptationTrainer

model = RGBNIRSugarBeetSegmentationNetwork(num_classes=2)

# 2. Setup trainer
device = torch.device('cuda')
trainer = RGBNIRDomainAdaptationTrainer(model, device)

# 3. Load Sugar Beet 2016 data
source_loader, target_loader, val_loader = create_sugar_beet_dataloaders(
    data_root='/path/to/sugar_beet_2016',
    batch_size=4    
)

# 4. Train with domain adaptation
trainer.train(source_loader, target_loader, val_loader, num_epochs=100)
