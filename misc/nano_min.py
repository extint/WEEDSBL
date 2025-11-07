#!/usr/bin/env python3
"""
Minimal ONNX inference for Jetson Nano
Loads preprocessed .npy input and runs model
NO cv2, PIL, or other heavy dependencies needed!
"""

import numpy as np
import onnxruntime as ort
import time
import argparse
import os

ONNX_PATH = "lmannet_4_ch_best.onnx"
IMG_HEIGHT = 960
IMG_WIDTH = 1280
THRESHOLD = 0.5

# For Jetson Nano with old ONNX Runtime - use CPU only
PROVIDERS = ['CPUExecutionProvider']


def load_preprocessed_input(npy_path):
    """Load preprocessed 4-channel input from .npy file"""
    print(f"Loading preprocessed input: {npy_path}")
    x = np.load(npy_path)
    
    print(f"  Shape: {x.shape}")
    print(f"  Dtype: {x.dtype}")
    print(f"  Range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Ensure contiguous and float32
    x = np.ascontiguousarray(x.astype(np.float32))
    
    return x


def run_inference(session, x):
    """Run ONNX inference on preprocessed input"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"\nRunning inference...")
    start_time = time.time()
    
    # Run inference
    outputs = session.run([output_name], {input_name: x})
    
    inference_time = time.time() - start_time
    
    # Extract probabilities (1, 1, H, W) -> (H, W)
    probs = outputs[0][0, 0]
    
    print(f"Inference time: {inference_time*1000:.2f}ms")
    print(f"Output shape: {probs.shape}")
    print(f"Probs range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Probs mean: {probs.mean():.3f}, median: {np.median(probs):.3f}")
    
    # Distribution
    print("\nProbability distribution:")
    print(f"  < 0.1: {(probs < 0.1).sum():>7} ({100*(probs < 0.1).sum()/probs.size:.1f}%)")
    print(f"  0.1-0.3: {((probs >= 0.1) & (probs < 0.3)).sum():>7} ({100*((probs >= 0.1) & (probs < 0.3)).sum()/probs.size:.1f}%)")
    print(f"  0.3-0.5: {((probs >= 0.3) & (probs < 0.5)).sum():>7} ({100*((probs >= 0.3) & (probs < 0.5)).sum()/probs.size:.1f}%)")
    print(f"  0.5-0.7: {((probs >= 0.5) & (probs < 0.7)).sum():>7} ({100*((probs >= 0.5) & (probs < 0.7)).sum()/probs.size:.1f}%)")
    print(f"  0.7-0.9: {((probs >= 0.7) & (probs < 0.9)).sum():>7} ({100*((probs >= 0.7) & (probs < 0.9)).sum()/probs.size:.1f}%)")
    print(f"  > 0.9: {(probs > 0.9).sum():>7} ({100*(probs > 0.9).sum()/probs.size:.1f}%)")
    
    # Create binary mask
    binary_mask = (probs > THRESHOLD).astype(np.uint8)
    print(f"\nDetected pixels (> {THRESHOLD}): {binary_mask.sum()} / {binary_mask.size} ({100*binary_mask.sum()/binary_mask.size:.2f}%)")
    
    return binary_mask, probs, inference_time


def save_outputs(binary_mask, probs, output_dir="./nano_outputs"):
    """Save outputs as binary files (for quick transfer back)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binary mask
    mask_path = os.path.join(output_dir, "mask.npy")
    np.save(mask_path, binary_mask)
    print(f"\nSaved mask: {mask_path}")
    
    # Save probabilities
    probs_path = os.path.join(output_dir, "probs.npy")
    np.save(probs_path, probs)
    print(f"Saved probs: {probs_path}")
    
    return mask_path, probs_path


def benchmark(session, num_runs=50):
    """Benchmark inference speed"""
    print(f"\nBenchmarking ({num_runs} iterations)...")
    
    # Create dummy input
    dummy_input = np.random.randn(1, 4, IMG_HEIGHT, IMG_WIDTH).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Warmup
    for _ in range(5):
        _ = session.run([output_name], {input_name: dummy_input})
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = session.run([output_name], {input_name: dummy_input})
        times.append(time.time() - start)
    
    times = np.array(times) * 1000
    
    print(f"Inference time statistics (ms):")
    print(f"  Mean: {times.mean():.2f}")
    print(f"  Median: {np.median(times):.2f}")
    print(f"  Min: {times.min():.2f}")
    print(f"  Max: {times.max():.2f}")
    print(f"  FPS: {1000/times.mean():.2f}")


def main():
    parser = argparse.ArgumentParser(description="Nano ONNX inference (minimal, no cv2)")
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed .npy input")
    parser.add_argument("--onnx", type=str, default=ONNX_PATH, help="Path to ONNX model")
    parser.add_argument("--output", type=str, default="./nano_outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Segmentation threshold")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Jetson Nano ONNX Inference (Minimal)")
    print("="*60)
    
    # Load preprocessed input
    x = load_preprocessed_input(args.input)
    
    # Create session
    print(f"\nLoading ONNX model: {args.onnx}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(args.onnx, sess_options=sess_options, providers=PROVIDERS)
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Session providers: {session.get_providers()}")
    
    # Run inference
    binary_mask, probs, inf_time = run_inference(session, x)
    
    # Save outputs
    save_outputs(binary_mask, probs, args.output)
    
    print(f"\n✅ Done! Total time: {inf_time*1000:.2f}ms")
    
    # Optional benchmark
    if args.benchmark:
        benchmark(session)


if __name__ == "__main__":
    main()
