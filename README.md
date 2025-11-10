# Crop Weed Detection
## File Structure ##
```
├─ checkpoints/
│  └─ unet4channel/                # Trained UNet checkpoints (RGB + NIR)
│
├─ datasets/
│  └─ __pycache__/                  # Auto-generated Python cache (ignore)
│
├─ misc/
│  ├─ inference/                    # ONNX runtime helpers for Jetson Nano
│  │  ├─ CPU Runtime Provider       # Inference scripts for CPU execution
│  │  └─ TensorRT Runtime Provider  # Inference scripts for TensorRT execution
│  ├─ nano_inf/                     # Jetson Nano: image preprocessing utilities
│  └─ nano_min/                     # Jetson Nano: (Used currently) main inference module (.npy Input Output format)
│
├─ models/                          # Model architecture definitions (unused currently)
│
├─ scripts/
│  ├─ bench_outputs/                # Benchmark results on HSL dataset (.pth / .onnx)
│  ├─ checkpoints/                  # Model checkpoints (LightMANet, UNet)
│  ├─ inference_results/            # Generated inference results (can ignore)
│  ├─ student_ckpts/                # Checkpoints from student model training
│  │
│  ├─ MANNET.py                     # MANNet architecture definition
│  ├─ distill_train.py              # Knowledge distillation training (teacher → student)
│  ├─ infer_student.py              # Inference for distilled student model
│  ├─ inference_visualization.png   # Example output visualization
│  ├─ lman_pthtoonnx.py             # Convert LightMANet (.pth) → (.onnx)
│  ├─ main.py                       # Deprecated; to ignore
│  ├─ new_main.py                   # Primary LightMANet training script
│  ├─ onnxinf.py                    # ONNX inference (CPU / TensorRT)
│  ├─ pthonnxcomparison.py          # Compare runtime: PyTorch vs ONNX
│  ├─ pthtoonnx.py                  # Generic .pth → .onnx converter
│  ├─ rice_weed_data_loader.py      # Data loader for RGB + NIR Rice–Weed dataset
│  └─ (other deprecated scripts)    # Legacy; safe to ignore
│
└─ (other root files)               # Miscellaneous or non-essential files
```
