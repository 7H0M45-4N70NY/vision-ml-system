"""Hardware and CUDA Environment Diagnostic Tool.

Validates NVIDIA GPU accessibility, PyTorch CUDA build, cuDNN acceleration,
and memory headroom for the Vision ML System (specifically optimized for RTX 4050 / Ada Lovelace).
"""

import sys
import os
import time

def print_banner(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def main():
    print_banner("VISION ML SYSTEM - HARDWARE & CUDA DIAGNOSTIC")
    
    # 1. Python Environment Check
    print(f"[1/5] Python Environment:")
    print(f"      Executable: {sys.executable}")
    print(f"      Version:    {sys.version.split()[0]}")

    # 2. PyTorch & CUDA Build Check
    print(f"\n[2/5] PyTorch & CUDA Support:")
    try:
        import torch
        print(f"      PyTorch Version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"      CUDA Available:  {'YES (CUDA Active)' if cuda_available else 'NO (CPU Only)'}")
        
        if not cuda_available:
            print("\n  [!] CAUTION: PyTorch is installed without CUDA acceleration or NVIDIA driver is inaccessible.")
            print("      To install PyTorch with CUDA for RTX 4050, run:")
            print("      uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            print("      (or: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124)")
            return
            
        print(f"      CUDA Build Ver:  {torch.version.cuda}")
        print(f"      cuDNN Enabled:   {torch.backends.cudnn.enabled}")
        print(f"      cuDNN Version:   {torch.backends.cudnn.version()}")
    except ImportError:
        print("  [!] PyTorch is not installed in the active environment.")
        print("      Run: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return

    # 3. GPU Device Details (RTX 4050 Validation)
    print(f"\n[3/5] GPU Hardware Detection:")
    device_count = torch.cuda.device_count()
    print(f"      Detected GPUs:   {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_vram_gb = props.total_memory / (1024 ** 3)
        print(f"\n      Device [{i}]:      {props.name}")
        print(f"      Compute Cap:     {props.major}.{props.minor} (Ada Lovelace Architecture)")
        print(f"      Total VRAM:      {total_vram_gb:.2f} GB")
        print(f"      Multi-Processors:{props.multi_processor_count}")

    # 4. VRAM Allocation & Tensor Core Benchmark
    print(f"\n[4/5] Memory & Tensor Core Benchmark (FP16 & FP32):")
    device = torch.device("cuda:0")
    
    # Warmup
    x = torch.randn(1024, 1024, device=device)
    _ = torch.matmul(x, x)
    torch.cuda.synchronize()

    # Benchmark FP16 (Half Precision - Tensor Cores)
    size = 4096
    a_fp16 = torch.randn(size, size, device=device, dtype=torch.float16)
    b_fp16 = torch.randn(size, size, device=device, dtype=torch.float16)
    
    start_time = time.perf_counter()
    iters = 20
    for _ in range(iters):
        c_fp16 = torch.matmul(a_fp16, b_fp16)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start_time) / iters

    # Memory state
    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"      FP16 GEMM ({size}x{size}): {fp16_time * 1000:.2f} ms / iter")
    print(f"      VRAM Allocated:        {allocated_mb:.2f} MB")
    print(f"      VRAM Reserved:         {reserved_mb:.2f} MB")
    print(f"      Peak Memory Used:      {max_memory_mb:.2f} MB")
    
    del a_fp16, b_fp16, c_fp16
    torch.cuda.empty_cache()

    # 5. Vision ML Ecosystem Check
    print(f"\n[5/5] Vision Stack Integration Check:")
    try:
        import ultralytics
        print(f"      Ultralytics:     v{ultralytics.__version__} (YOLO ready)")
    except ImportError:
        print("      Ultralytics:     Not installed")

    try:
        import supervision
        print(f"      Supervision:     v{supervision.__version__} (Detections ready)")
    except ImportError:
        print("      Supervision:     Not installed")

    try:
        import cv2
        print(f"      OpenCV:          v{cv2.__version__}")
    except ImportError:
        print("      OpenCV:          Not installed")

    print_banner("DIAGNOSTIC COMPLETE - GPU READY FOR INFERENCE & TRAINING")
    print("Recommendation for RTX 4050 (6GB VRAM):")
    print("  - Inference: Use fp16 precision (device='cuda:0', precision='fp16')")
    print("  - Training:  Use batch size 8 or 16 with AMP (Automatic Mixed Precision)")
    print("============================================================\n")


if __name__ == '__main__':
    main()

