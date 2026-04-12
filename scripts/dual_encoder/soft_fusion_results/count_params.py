import torch
import sys

def count_params(pth_path):
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)

    # Handle common checkpoint wrapper formats
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt:
                print(f"  (loaded state dict from key: '{key}')")
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        print("ERROR: Could not find a state dict in this file.")
        sys.exit(1)

    total   = sum(v.numel() for v in ckpt.values())
    trainable = sum(v.numel() for v in ckpt.values() if v.is_floating_point())

    print(f"\nFile : {pth_path}")
    print(f"  Total params      : {total:,}")
    print(f"  Floating-point    : {trainable:,}  ({trainable/total*100:.1f}%)")
    print(f"  Approx size (fp32): {trainable * 4 / 1e6:.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_params.py <path_to_model.pth>")
        sys.exit(1)
    count_params(sys.argv[1])