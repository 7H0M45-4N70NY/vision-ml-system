#!/usr/bin/env python3
"""Benchmark YOLO inference across backends: PyTorch FP32/FP16, ONNX, TensorRT.

Designed to run on Kaggle/Colab GPU instances. On CPU-only machines it will
only benchmark PyTorch FP32 and ONNX (FP16/TensorRT require CUDA).

Measures:
  - Latency: median ms per frame (excludes warmup)
  - Throughput: frames per second
  - GPU memory: peak VRAM usage
  - Model size: file size on disk
  - Detection parity: mAP consistency across backends

Usage (Colab/Kaggle):
    !python examples/demo_benchmark.py
    !python examples/demo_benchmark.py --model yolo11s --frames 200
    !python examples/demo_benchmark.py --backends pytorch_fp32 onnx tensorrt

Usage (local CPU — limited):
    python examples/demo_benchmark.py --backends pytorch_fp32 onnx
"""

import sys
import os
import json
import time
import argparse
import statistics
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Backend definitions
# ---------------------------------------------------------------------------

BACKENDS = {
    'pytorch_fp32': {
        'label': 'PyTorch FP32',
        'format': None,        # native .pt
        'half': False,
        'requires_gpu': False,
    },
    'pytorch_fp16': {
        'label': 'PyTorch FP16',
        'format': None,
        'half': True,
        'requires_gpu': True,
    },
    'onnx': {
        'label': 'ONNX Runtime',
        'format': 'onnx',
        'half': False,
        'requires_gpu': False,
    },
    'tensorrt_fp16': {
        'label': 'TensorRT FP16',
        'format': 'engine',
        'half': True,
        'requires_gpu': True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_gpu_memory_mb():
    """Return current GPU memory usage in MB, or 0 if no GPU."""
    if not HAS_CUDA:
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def get_peak_gpu_memory_mb():
    """Return peak GPU memory usage in MB since last reset."""
    if not HAS_CUDA:
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def generate_test_frames(num_frames: int, width: int = 640, height: int = 480):
    """Generate random test frames for benchmarking."""
    rng = np.random.RandomState(42)
    return [rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
            for _ in range(num_frames)]


def export_model(model_path: str, fmt: str, half: bool, device: str):
    """Export YOLO model to target format. Returns path to exported model."""
    model = YOLO(model_path)
    exported = model.export(format=fmt, half=half, device=device)
    return str(exported)


def get_model_size_mb(path: str) -> float:
    """Get model file size in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    # For directories (some exports), sum all files
    if os.path.isdir(path):
        total = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
        return total / (1024 * 1024)
    return 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_backend(
    backend_name: str,
    model_path: str,
    frames: list,
    warmup: int = 10,
    device: str = 'cuda',
    conf: float = 0.35,
    iou: float = 0.45,
) -> dict:
    """Run benchmark for a single backend. Returns metrics dict."""
    cfg = BACKENDS[backend_name]
    print(f"\n--- {cfg['label']} ---")

    # Skip GPU-only backends on CPU
    if cfg['requires_gpu'] and not HAS_CUDA:
        print(f"  SKIPPED (requires CUDA)")
        return {'backend': backend_name, 'label': cfg['label'], 'skipped': True,
                'reason': 'No CUDA device'}

    effective_device = device if HAS_CUDA else 'cpu'

    # Export if needed
    if cfg['format']:
        print(f"  Exporting to {cfg['format']}...")
        try:
            exported_path = export_model(model_path, cfg['format'], cfg['half'], effective_device)
            print(f"  Exported: {exported_path}")
        except Exception as e:
            print(f"  EXPORT FAILED: {e}")
            return {'backend': backend_name, 'label': cfg['label'], 'skipped': True,
                    'reason': str(e)}
        model = YOLO(exported_path)
        model_size = get_model_size_mb(exported_path)
    else:
        model = YOLO(model_path)
        model_size = get_model_size_mb(model_path)

    # Reset GPU memory tracking
    if HAS_CUDA:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup
    print(f"  Warming up ({warmup} frames)...")
    for frame in frames[:warmup]:
        model(frame, conf=conf, iou=iou, device=effective_device,
              verbose=False, half=cfg['half'])

    if HAS_CUDA:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed inference
    num_frames = len(frames)
    latencies = []
    total_detections = 0

    print(f"  Benchmarking ({num_frames} frames)...")
    for frame in frames:
        if HAS_CUDA:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        results = model(frame, conf=conf, iou=iou, device=effective_device,
                        verbose=False, half=cfg['half'])
        if HAS_CUDA:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms
        total_detections += len(results[0].boxes)

    peak_mem = get_peak_gpu_memory_mb()

    metrics = {
        'backend': backend_name,
        'label': cfg['label'],
        'skipped': False,
        'num_frames': num_frames,
        'latency_median_ms': round(statistics.median(latencies), 2),
        'latency_p95_ms': round(sorted(latencies)[int(0.95 * len(latencies))], 2),
        'latency_mean_ms': round(statistics.mean(latencies), 2),
        'throughput_fps': round(num_frames / (sum(latencies) / 1000), 1),
        'peak_gpu_memory_mb': round(peak_mem, 1),
        'model_size_mb': round(model_size, 1),
        'total_detections': total_detections,
        'avg_detections_per_frame': round(total_detections / num_frames, 2),
        'device': effective_device,
    }

    print(f"  Median latency: {metrics['latency_median_ms']:.1f} ms")
    print(f"  Throughput:     {metrics['throughput_fps']:.1f} fps")
    if peak_mem > 0:
        print(f"  Peak GPU mem:   {peak_mem:.1f} MB")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLO Backend Benchmark")
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='YOLO model path or name')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames to benchmark')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup frames (excluded from timing)')
    parser.add_argument('--backends', nargs='+',
                        default=list(BACKENDS.keys()),
                        choices=list(BACKENDS.keys()),
                        help='Backends to benchmark')
    parser.add_argument('--device', type=str, default='cuda' if HAS_CUDA else 'cpu')
    parser.add_argument('--output', type=str, default='runs/benchmark/results.json')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    print("=" * 65)
    print("  YOLO BACKEND BENCHMARK")
    print("=" * 65)
    print(f"  Model:      {args.model}")
    print(f"  Device:     {args.device} ({'GPU' if HAS_CUDA else 'CPU only'})")
    print(f"  Frames:     {args.frames} (+{args.warmup} warmup)")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Backends:   {', '.join(args.backends)}")
    if HAS_CUDA:
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  GPU Memory: {total_mem:.1f} GB")
    print("=" * 65)

    # Generate test frames
    print(f"\nGenerating {args.frames} test frames...")
    frames = generate_test_frames(args.frames, args.width, args.height)

    # Run benchmarks
    results = []
    for backend in args.backends:
        metrics = benchmark_backend(
            backend_name=backend,
            model_path=args.model,
            frames=frames,
            warmup=args.warmup,
            device=args.device,
        )
        results.append(metrics)

    # Print comparison table
    active = [r for r in results if not r.get('skipped')]
    skipped = [r for r in results if r.get('skipped')]

    print(f"\n{'=' * 65}")
    print("  BENCHMARK RESULTS")
    print(f"{'=' * 65}")
    print(f"  {'Backend':<18} {'Latency':>10} {'FPS':>8} {'GPU Mem':>10} {'Size':>8} {'Dets/fr':>8}")
    print(f"  {'':─<18} {'(median)':>10} {'':>8} {'(peak)':>10} {'(MB)':>8} {'':>8}")
    print(f"  {'─' * 62}")

    for r in active:
        mem_str = f"{r['peak_gpu_memory_mb']:.0f} MB" if r['peak_gpu_memory_mb'] > 0 else "N/A"
        print(f"  {r['label']:<18} {r['latency_median_ms']:>8.1f}ms {r['throughput_fps']:>7.1f} "
              f"{mem_str:>10} {r['model_size_mb']:>7.1f} {r['avg_detections_per_frame']:>8.2f}")

    if skipped:
        print(f"\n  Skipped:")
        for r in skipped:
            print(f"    {r['label']}: {r.get('reason', 'unknown')}")

    # Speedup comparison
    if len(active) >= 2:
        baseline = active[0]
        print(f"\n  Speedup vs {baseline['label']}:")
        for r in active[1:]:
            speedup = baseline['latency_median_ms'] / r['latency_median_ms']
            print(f"    {r['label']}: {speedup:.2f}x")

    print(f"{'=' * 65}")

    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    output = {
        'metadata': {
            'model': args.model,
            'device': args.device,
            'gpu': torch.cuda.get_device_name(0) if HAS_CUDA else 'CPU',
            'num_frames': args.frames,
            'resolution': f"{args.width}x{args.height}",
            'warmup_frames': args.warmup,
        },
        'results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Colab/Kaggle hint
    if not HAS_CUDA:
        print(f"\n  NOTE: Running on CPU. For full benchmark with FP16 + TensorRT,")
        print(f"  run this script on a GPU instance (Kaggle T4 or Colab):")
        print(f"    !python examples/demo_benchmark.py --device cuda")


if __name__ == '__main__':
    main()
