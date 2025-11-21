import os
import torch
import numpy as np
from models import createmodel  # Make sure this returns the correct model given config
from rice_weed_data_loader-1 import create_weedy_rice_rgbnir_dataloaders  # Adjust as needed
from ptflops import get_model_complexity_info  # Use flopth if you prefer

def pixel_accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)  # For multi-class; adjust for binary
    correct = torch.eq(pred, target).sum().item()
    total = torch.numel(target)
    return correct / total

experiments_root = './experiments'  # Update as needed
output_file = 'metrics_results.txt'
val_batch_size = 1  # or what your GPU supports

results = []

for exp in os.listdir(experiments_root):
    ckpt_path = os.path.join(experiments_root, exp, 'checkpoints', 'best_model.pth')
    config_path = os.path.join(experiments_root, exp, 'config.json')  # Or wherever config is stored

    if not os.path.exists(ckpt_path):
        continue

    # Load config: Assume JSON dict with 'model_name', 'num_channels', etc.
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = createmodel(config['model_name'], config['num_channels'])
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()

    # Dataloader
    _, val_loader = create_weedy_rice_rgbnir_dataloaders(config, batch_size=val_batch_size)

    # Pixel accuracy computation
    total_correct, total_pixels = 0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct = torch.eq(preds, masks).sum().item()
            total = torch.numel(masks)
            total_correct += correct
            total_pixels += total
    pixel_acc = total_correct / total_pixels if total_pixels else 0

    # FLOPs computation (use ptflops or flopth as per install)
    def input_constructor(input_res):
        return {'x': torch.randn((1, config['num_channels']) + input_res)}

    flops, params = get_model_complexity_info(
        model, (config['input_size'], config['input_size']),
        input_constructor=input_constructor, as_strings=False)
    gflops = flops / 1e9

    # Save metrics for this model
    results.append(f"{exp}: Pixel Accuracy={pixel_acc:.4f}, FLOPs(G)={gflops:.4f}, Params={params/1e6:.4f}M")

# Write results to file
with open(output_file, 'w') as f:
    for line in results:
        f.write(line + '\n')
