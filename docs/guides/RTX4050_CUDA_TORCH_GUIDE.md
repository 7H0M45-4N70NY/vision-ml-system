# Production Hardware Guide: CUDA, PyTorch & RTX 4050 (Ada Lovelace)

> **Audience:** ML Systems Engineers, CV Practitioners, Technical Interviewers, and Recruiters.  
> **Target Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (AD107 / Ada Lovelace, 6GB GDDR6 VRAM).  
> **Scope:** Demystifying NVIDIA drivers vs. CUDA Toolkit, configuring PyTorch with GPU acceleration, optimizing memory budgets for edge CV, and verifying system performance.

---

## 1. Hardware Architecture Overview

| Parameter | Specification | Engineering Implication |
|---|---|---|
| **GPU Architecture** | Ada Lovelace (AD107) | 4th-Gen Tensor Cores, FP8/FP16 native support |
| **Compute Capability** | **8.9** (`sm_89`) | Modern architecture supported by PyTorch 2.1+ |
| **VRAM Capacity** | **6.0 GB GDDR6** | Edge-class memory envelope (similar to Jetson Orin Nano) |
| **Memory Bus Width** | 96-bit (~192 GB/s) | Bandwidth-conscious batching required |
| **CUDA Cores** | 2,560 | Balanced for high FPS real-time detection |
| **Tensor Cores** | 80 (4th Generation) | 2x-3x matrix acceleration with AMP (FP16/BF16) |

---

## 2. Demystifying CUDA on Windows: What You Actually Need

A common misunderstanding among practitioners is that installing the multi-gigabyte **NVIDIA CUDA Toolkit (`nvcc`)** is mandatory for PyTorch. In modern production ML systems, there are three distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: NVIDIA CUDA Toolkit (nvcc, headers, profilers)    │
│ ONLY needed when compiling custom C++/CUDA kernel extensions │
└───────────────────────────────▲─────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────┐
│ Layer 2: PyTorch CUDA Runtime Wheels (bundled in torch wheel)│
│ INCLUDES: cudart64.dll, cuDNN, cuBLAS, cuFFT, NCCL           │
│ -> This is all you need for YOLO, PyTorch, & Supervision!    │
└───────────────────────────────▲─────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────┐
│ Layer 1: NVIDIA Display Driver (e.g., Driver 550+ / 610+)   │
│ Provides nvcuda.dll (CUDA Driver API) & communicates with GPU│
└─────────────────────────────────────────────────────────────┘
```

### The Takeaway
1. **NVIDIA Display Driver (Layer 1):** If `nvidia-smi` runs in your terminal, your host driver is working and exposes the CUDA Driver API (supporting up to CUDA 12.x/13.x).
2. **PyTorch Wheel (Layer 2):** PyTorch wheels from the official PyTorch wheelhouse (`https://download.pytorch.org/whl/cu124` or `cu121`) already include all necessary CUDA and cuDNN runtime binaries internally.
3. **CUDA Toolkit Installer (Layer 3):** You **do not** need to install the 3.5GB CUDA Toolkit executable unless you plan to write custom CUDA `.cu` files or compile custom C++ extensions with `nvcc`.

---

## 3. Quickstart: Setting Up PyTorch with CUDA via `uv`

The fastest, cleanest, and most reproducible method to set up the environment on Windows:

### Step 1: Create Virtual Environment
```powershell
# Using uv (recommended for speed and lockfile integrity)
uv venv .venv --python 3.13
.venv\Scripts\activate
```

### Step 2: Install PyTorch with CUDA 12.4
Do **not** run a bare `pip install torch` from PyPI on Windows, as PyPI defaults to CPU-only wheels for Windows platforms. Specify the PyTorch CUDA index:

```powershell
# Install PyTorch with CUDA 12.4 support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 3: Install Vision ML Dependencies
```powershell
uv pip install -r requirements.txt
```

### Step 4: Verify Acceleration
Run the diagnostic script included in this repository:
```powershell
python scripts/check_cuda.py
```

Expected output:
```text
============================================================
 VISION ML SYSTEM - HARDWARE & CUDA DIAGNOSTIC
============================================================
[2/5] PyTorch & CUDA Support:
      PyTorch Version: 2.x.x+cu124
      CUDA Available:  YES (CUDA Active)
      CUDA Build Ver:  12.4
      cuDNN Enabled:   True
      cuDNN Version:   90100

[3/5] GPU Hardware Detection:
      Device [0]:      NVIDIA GeForce RTX 4050 Laptop GPU
      Compute Cap:     8.9 (Ada Lovelace Architecture)
      Total VRAM:      6.00 GB
============================================================
```

---

## 4. 6GB VRAM Optimization Playbook (Edge-Class Systems)

Working with a 6GB VRAM constraint trains you to think like an **Edge Systems Engineer**. Deployments on retail edge servers, drones, and NVIDIA Jetson devices operate under identical memory caps.

### Rule 1: Use Automatic Mixed Precision (AMP / FP16)
- **Why:** FP16 cuts activation memory in half and unlocks 4th-Gen Tensor Cores.
- **Inference:** YOLOv11 supports native FP16 execution:
  ```python
  # In config/inference/base.yaml:
  model:
    precision: fp16
  inference:
    device: cuda:0
  ```
- **Training:** Ultralytics YOLO enables AMP by default (`amp=True`), reducing memory footprint during backpropagation by ~40%.

### Rule 2: Calibrated Batch Sizing
Recommended batch sizes for RTX 4050 (6GB VRAM) at standard `640x640` image resolution:

| Task / Model | Image Size | Batch Size | Estimated VRAM Peak | Headroom |
|---|---|---|---|---|
| **YOLOv11n (Inference)** | 640x640 | Stream (1) | ~1.1 GB | High (>4.5 GB free) |
| **YOLOv11s (Inference)** | 640x640 | Stream (1) | ~1.4 GB | High (>4.0 GB free) |
| **YOLOv11n (Training)** | 640x640 | 16 | ~2.6 GB | Safe |
| **YOLOv11s (Training)** | 640x640 | 8 | ~3.8 GB | Safe |
| **RF-DETR (Inference)** | 640x640 | Stream (1) | ~2.8 GB | Moderate |
| **DualDetector (Hot)** | 640x640 | Stream (1) | ~1.2 GB | Safe (RF-DETR in batch) |

### Rule 3: Enable cuDNN Benchmark
When input frame dimensions are constant (e.g., standard RTSP or 1080p video streams), tell cuDNN to benchmark and select the fastest convolution algorithms:
```python
import torch
torch.backends.cudnn.benchmark = True
```

### Rule 4: Clean Frame Loops & Avoid VRAM Leaks
In computer vision pipelines, accidental references to PyTorch tensors in long-running loops cause CUDA Out-Of-Memory (OOM) crashes:
```python
# BAD: Retains computation graph across iterations
losses.append(loss) 

# GOOD: Extract scalar float
losses.append(loss.item())

# GOOD: Convert supervision / bounding box outputs to NumPy / CPU immediately
boxes = detections.xyxy  # NumPy array
```

---

## 5. System Design & Interview Talking Points

When discussing this hardware setup in technical interviews or with recruiters, frame the RTX 4050 not as a limitation, but as a deliberate **edge design constraint**:

### Talking Point 1: Edge Simulation
> *"I deliberately designed and tested this pipeline within a 6GB VRAM budget. In retail and industrial CV, you rarely have an A100 in the ceiling; you have edge devices like Jetson Orins or localized micro-PCs with 4GB to 8GB of memory. Optimizing for this envelope ensures the system can be deployed anywhere without requiring multi-thousand dollar cloud instances."*

### Talking Point 2: Dual-Detector Architecture
> *"Rather than keeping two massive models permanently loaded in VRAM, our `DualDetector` uses a decoupled strategy: YOLOv11 runs on the fast hot path consuming just ~1.1GB VRAM, while low-confidence frames are buffered to disk (`data/low_confidence_frames/`) for batch secondary analysis or active learning, preventing VRAM contention."*

### Talking Point 3: Privacy & Zero Cloud Egress
> *"All inference, tracking, dwell-time analytics, and SQLite persistence run 100% on-device on the GPU. No customer video frames ever leave the physical premises, satisfying GDPR and strict privacy compliance out of the box."*

