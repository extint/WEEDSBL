import torch
import torch.nn as nn
import onnx
from onnx import helper, numpy_helper
import numpy as np

from MANNet import LightMANet

ckpt_path = "scripts/checkpoints/lmannet_4_ch_best.pth"
onnx_path = "scripts/checkpoints/lmannet_4_ch_best.onnx"
device = "cuda"

# Load model
model = LightMANet(in_channels=4, num_classes=1, base_ch=32)

device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
state = torch.load(ckpt_path, map_location=device_obj)
state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
model.load_state_dict(state_dict, strict=True)

# Wrap with sigmoid
class ModelWithSigmoid(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
    
    def forward(self, x):
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        return probs

wrapped_model = ModelWithSigmoid(model)
wrapped_model.eval().to(device_obj)

dummy = torch.randn(1, 4, 960, 1280, device=device_obj)

# Export
print("Exporting to ONNX...")
torch.onnx.export(
    wrapped_model,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True,
    export_params=True,
    verbose=False,
    training=torch.onnx.TrainingMode.EVAL,
    dynamic_axes=None
)

print(f"✓ Exported: {onnx_path}")

# NOW: Convert opset 20 ReduceMean to opset 13 compatible version
print("\n" + "="*60)
print("Converting ReduceMean opset 20 → opset 13 format...")
print("="*60)

onnx_model = onnx.load(onnx_path)

print(f"Current opset: {onnx_model.opset_import[0].version}")

# Convert ReduceMean nodes from opset 20 format to opset 13 format
# Opset 20: axes is an INPUT (second input)
# Opset 13: axes is an ATTRIBUTE
for node in onnx_model.graph.node:
    if node.op_type == "ReduceMean":
        print(f"\nProcessing {node.name} (ReduceMean)")
        print(f"  Current inputs: {list(node.input)}")
        print(f"  Current attributes: {[(attr.name, attr) for attr in node.attribute]}")
        
        # In opset 20, the second input is the axes tensor
        # We need to extract it and convert to an attribute
        if len(node.input) >= 2:
            axes_input = node.input[1]
            print(f"  Axes input tensor: {axes_input}")
            
            # Find the initializer for this axes tensor
            axes_initializer = None
            for init in onnx_model.graph.initializer:
                if init.name == axes_input:
                    axes_initializer = init
                    break
            
            if axes_initializer:
                # Extract axes value
                axes_value = numpy_helper.to_array(axes_initializer).tolist()
                if isinstance(axes_value, (int, np.integer)):
                    axes_value = [axes_value]
                print(f"  Axes values: {axes_value}")
                
                # Create axes attribute
                axes_attr = helper.make_attribute("axes", axes_value)
                node.attribute.extend([axes_attr])
                
                # Remove the second input (axes was passed as input)
                del node.input[1]
                
                print(f"  ✓ Converted to attribute format")
                print(f"  New inputs: {list(node.input)}")
        
        # Remove opset 20+ specific attributes
        attrs_to_remove = []
        for i, attr in enumerate(node.attribute):
            if attr.name == "noop_with_empty_axes":
                print(f"  Removing opset 20+ attribute: {attr.name}")
                attrs_to_remove.append(i)
        
        for i in sorted(attrs_to_remove, reverse=True):
            del node.attribute[i]

# Set opset to 13
onnx_model.opset_import[0].version = 13
print(f"\n✓ Set opset to: 13")

onnx_model.ir_version = 8
print(f"✓ Set IR version to: 8")

# Save
onnx.save(onnx_model, onnx_path)
print(f"✓ Saved opset 13 model: {onnx_path}")

# Validate
print("\nValidating...")
try:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model validated successfully")
    print(f"\n📊 Model Info:")
    print(f"  Opset Version: {onnx_model.opset_import[0].version}")
    print(f"  IR Version: {onnx_model.ir_version}")
    print(f"  Total nodes: {len(onnx_model.graph.node)}")
    reducemean_count = sum(1 for n in onnx_model.graph.node if n.op_type == "ReduceMean")
    print(f"  ReduceMean nodes: {reducemean_count}")
    print(f"\n✅ Ready for Jetson Nano deployment!")
except Exception as e:
    print(f"⚠️  Validation failed: {e}")
    print(f"   Attempting to continue anyway...")
