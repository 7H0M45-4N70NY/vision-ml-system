# Computer Vision & ML Systems Interview Playbook

A deep technical reference for CV/ML engineering roles. Every concept links back to
this codebase so you can demonstrate real implementation experience, not textbook answers.

---

## Table of Contents

1. [The Modern CV Landscape](#1-the-modern-cv-landscape)
2. [Deep Learning Foundations](#2-deep-learning-foundations)
3. [Object Detection Deep Dive](#3-object-detection-deep-dive)
4. [CV System Design](#4-cv-system-design)
5. [MLOps & Production ML](#5-mlops--production-ml)
6. [Active Learning & Data Flywheels](#6-active-learning--data-flywheels)
7. [Edge Deployment & Optimization](#7-edge-deployment--optimization)
8. [Coding Interview Patterns](#8-coding-interview-patterns)
9. [Behavioral Interview Strategy](#9-behavioral-interview-strategy)
10. [System Design Mock Questions](#10-system-design-mock-questions)

---

## 1. The Modern CV Landscape

### What Interviewers Actually Want (2025-2026)

The industry has shifted from "what's your mAP?" to "how does your system handle
failure at 3am?" Companies hire **ML Systems Engineers** — people who build
reliable pipelines, not just accurate models.

**Three capabilities they test for:**

| Capability | What They Ask | How You Prove It |
|---|---|---|
| **Systems Thinking** | "How would you deploy this to 100K users?" | Your `InferencePipeline` orchestrates Detector → Tracker → Analytics → DriftDetector in a clean pipeline (`pipeline.py:18-50`) |
| **Self-Improving Systems** | "What happens when accuracy degrades?" | Your `DriftDetector` monitors rolling confidence, your `DualDetector` auto-collects failure cases |
| **Production Awareness** | "How do you monitor model health?" | Config-driven thresholds, SQLite analytics persistence, MLflow experiment tracking |

### The Model Landscape You Must Know

**Detection models (real-time):**

| Model | COCO mAP | Latency | Key Innovation |
|---|---|---|---|
| YOLOv8 (x) | ~53.9 | Baseline | Anchor-free, decoupled head |
| YOLOv9 (e) | ~55.6 | Slightly slower | Programmable Gradient Info (PGI) |
| YOLOv10 (x) | ~54.4 | 19.3ms | NMS-free end-to-end |
| **YOLOv11 (x)** | ~54.7 | **13.5ms** | C3k2 blocks, 22% fewer params than v10 |
| RT-DETR | ~53.1 | 108 FPS on T4 | Proved transformers match YOLO speed |
| **RF-DETR** | ~54.7 | 4.52ms on T4 | DINOv2 backbone, strong domain generalization |

> **Your project uses YOLOv11 (primary) + RF-DETR (secondary)** — this is a real
> architectural decision you can defend. YOLO for speed on the hot path, RF-DETR
> for accuracy on edge cases. See `DualDetector` (`dual_detector.py`).

**Foundation models (know the concepts):**

| Model | What It Does | Interview Angle |
|---|---|---|
| **DINOv2** | Self-supervised visual features (no labels) | "How do SSL methods reduce labeling costs?" |
| **CLIP / SigLIP** | Aligns image + text embeddings via contrastive learning | "How would you build a visual search system?" |
| **SAM2** | Promptable segmentation (point/box/text → mask), works on video | "How would you integrate SAM2 into auto-labeling?" |
| **Florence-2** | Multi-task: captioning, grounding, detection via text prompts | "Benefits of unified multi-task vision models?" |
| **Grounding DINO** | Open-vocabulary detection (text → bounding boxes) | "Detect objects you've never trained on" |

---

## 2. Deep Learning Foundations

### 2.1 Vision Transformers (ViTs) — Must-Know Architecture

**How ViT works (step by step):**

```
Input Image (224x224x3)
    │
    ▼
Patch Embedding: Split into 16x16 patches → 196 patches
    │  Each patch flattened to 768-dim vector (linear projection)
    │
    ▼
Prepend [CLS] token → 197 tokens
    │
    ▼
Add Positional Embeddings (learnable 1D, one per token)
    │
    ▼
L Transformer Encoder Blocks, each:
    ├── Layer Norm
    ├── Multi-Head Self-Attention (MHSA)
    ├── Residual Connection
    ├── Layer Norm
    ├── MLP (Linear → GELU → Linear)
    └── Residual Connection
    │
    ▼
[CLS] token output → Classification Head
```

**Self-Attention (the core mechanism):**

```
Q = X·W_Q    K = X·W_K    V = X·W_V

Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Multi-Head: h parallel heads, each with d_k = D/h dimensions
            outputs concatenated and projected
```

- Complexity: **O(n² · d)** where n = number of patches
- Every patch attends to every other patch → **global receptive field from layer 1**

**ViT vs CNN — the interview comparison:**

| Aspect | CNN | ViT |
|---|---|---|
| Inductive bias | Translation equivariance, locality | None (learns from data) |
| Data efficiency | Better on small datasets | Needs large-scale pretraining |
| Receptive field | Local, grows with depth | Global from layer 1 |
| Scaling | Diminishing returns | Continues improving with more data/compute |

**When to use which:**
- CNN for edge/mobile, small datasets, latency-critical real-time
- ViT for large-scale pretraining, transfer learning, when accuracy > latency

> **Interview answer:** "I'd choose a ViT-based model when I have access to large
> pretrained weights and accuracy is paramount. For real-time edge inference
> where latency matters, I'd stick with a CNN-based model like YOLO. In my
> project, I use both — YOLOv11 (CNN) for real-time primary detection and
> RF-DETR (DINOv2/ViT backbone) for high-accuracy secondary detection."

### 2.2 Efficient Attention Variants

| Variant | How It Works | Complexity | Trade-off |
|---|---|---|---|
| **FlashAttention** | IO-aware exact attention, minimizes HBM reads | O(n²) but much faster in practice | No accuracy loss, just faster |
| **Linear Attention** | Replace softmax(QK^T)V with φ(Q)(φ(K)^T V) | O(n) | Potentially lower expressivity |
| **Window/Swin** | Attention within shifted local windows | O(n) | Loses some global context |

> Know FlashAttention well — it's not an approximation. It computes exact
> attention but restructures memory access patterns to minimize GPU HBM
> read/writes. FlashAttention-3 supports FP8 on Hopper GPUs.

### 2.3 Contrastive Learning (CLIP/SigLIP)

**CLIP training objective:**
- Given N (image, text) pairs, compute embeddings for all images and all texts
- Create an N×N similarity matrix
- Maximize similarity on the diagonal (matching pairs)
- Minimize similarity off-diagonal (non-matching pairs)
- Loss: symmetric cross-entropy over the similarity matrix

**SigLIP improvement:**
- Replaces softmax cross-entropy with **sigmoid loss** on each pair independently
- No need for global normalization across the batch
- Better performance, more efficient training at scale

**Why this matters:** CLIP embeddings power visual search, zero-shot classification,
and content moderation systems. If asked "design a visual search engine," CLIP
is your retrieval backbone.

### 2.4 Self-Supervised Learning (DINOv2)

**Why it matters:** Labeled data is expensive. DINOv2 produces universal visual
features without any labels.

**How DINO works:**
- Student-teacher framework (both are ViTs)
- Teacher is an exponential moving average of student weights
- Student sees augmented crops, teacher sees full image
- Loss: cross-entropy between student and teacher output distributions
- No labels needed — the model learns representations by self-distillation

> **RF-DETR uses DINOv2 as its backbone** — this is why it generalizes well
> to new domains without fine-tuning. Your `rfdetr_detector.py` loads this.

---

## 3. Object Detection Deep Dive

### 3.1 Anchor-Free vs Anchor-Based

**Anchor-based (older YOLO versions):**
- Predefined bounding box templates at each spatial location
- Model predicts offsets from these templates
- Requires careful anchor design and matching strategy

**Anchor-free (YOLOv8+, your project):**
- Model directly predicts center point + width/height
- Simpler, fewer hyperparameters, better generalization
- Uses **decoupled head**: separate branches for classification and regression

### 3.2 DETR Architecture (Know for RF-DETR Questions)

```
Image → CNN/ViT Backbone → Encoder → Decoder → Set Prediction
                                        ↑
                              Object Queries (learnable)
```

**Key innovation:** Treats detection as a **set prediction** problem.
- No NMS needed — Hungarian matching during training ensures unique assignments
- Object queries are learnable embeddings that each "specialize" in finding objects
- RT-DETR/RF-DETR proved this can be as fast as YOLO

### 3.3 NMS (Non-Maximum Suppression)

You **must** be able to implement this from scratch:

```
Algorithm:
1. Sort detections by confidence (descending)
2. Pick the highest-confidence box → add to output
3. Remove all remaining boxes with IoU > threshold against picked box
4. Repeat from step 2 until no boxes remain
```

Variants to know:
- **Soft-NMS:** Instead of removing overlapping boxes, decay their scores
- **NMS-free (YOLOv10):** End-to-end training eliminates NMS entirely
- **Weighted NMS:** Merge overlapping boxes weighted by confidence

### 3.4 Evaluation Metrics

| Metric | What It Measures | When to Use |
|---|---|---|
| **mAP@50** | Mean Average Precision at IoU ≥ 0.5 | General detection quality |
| **mAP@50:95** | Average over IoU thresholds 0.5 to 0.95 | COCO standard, more strict |
| **Precision** | TP / (TP + FP) | When false positives are costly |
| **Recall** | TP / (TP + FN) | When missed detections are costly |
| **F1** | Harmonic mean of Precision & Recall | Balanced performance |

> **Your project context:** For retail visitor analytics, **recall matters more**
> than precision — missing a person is worse than a false detection that gets
> filtered by tracking. This is a real trade-off discussion point.

---

## 4. CV System Design

### 4.1 The Framework (45-minute interview)

```
1. Clarify Requirements       (~5 min)  — functional, non-functional, scale
2. High-Level Architecture    (~10 min) — block diagram, data flow
3. Deep Dive                  (~25 min) — data pipeline, models, serving, storage
4. Evaluation & Trade-offs    (~5 min)  — metrics, A/B testing, failure modes
```

### 4.2 The "Vision Flywheel" — Your Key Differentiator

This is the architecture from your project. Use it as your go-to system design pattern:

```
┌─────────────────────────────────────────────────────┐
│                   INFERENCE (Hot Path)               │
│  Video Stream → YOLOv11 → ByteTrack → Analytics DB  │
│                    │                                 │
│          confidence < threshold?                     │
│              YES │          NO │                     │
│                  ▼             ▼                     │
│         Save Frame       Discard (handled)           │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│              IMPROVEMENT (Cold Path)                 │
│  Low-conf frames → RF-DETR (pseudo-labels)          │
│       → Roboflow/Local export                       │
│       → Human review (if both models uncertain)     │
│       → Retrain YOLOv11                             │
│       → Deploy improved model                       │
│       → (cycle repeats)                             │
└─────────────────────────────────────────────────────┘
```

**How to explain this in an interview:**

> "I design systems with a dual-path architecture. The hot path uses a fast
> model (YOLOv11) for real-time inference — it handles 95%+ of traffic at
> minimal latency. When confidence drops below a threshold, those frames
> are saved for the cold path, where a higher-accuracy model (RF-DETR)
> provides pseudo-labels. This creates a self-improving data flywheel that
> automatically collects the hardest examples for the next training cycle.
> The key insight is that the system focuses on its own failures — active
> learning, not random sampling."

**Code references to cite:**
- Hot/cold path routing: `DualDetector.detect_with_source()` (`dual_detector.py:58-122`)
- Frame saving: `DualDetector._save_frame()` (`dual_detector.py:124-139`)
- Label collection: `AutoLabeler.load_dual_detector_frames()` (`auto_labeler.py:37-50`)
- Drift monitoring: `DriftDetector.check()` (`drift_detector.py:80-104`)

### 4.3 Scaling This System

**Single camera → 10 cameras → 1000 cameras:**

| Scale | Architecture | Key Changes |
|---|---|---|
| 1 camera | Monolith (your current project) | Single process, SQLite, local storage |
| 10 cameras | Multi-process + shared DB | Process pool, PostgreSQL, shared model in GPU memory |
| 100 cameras | Microservices + message queue | Kafka/Redis for frame ingestion, Triton Inference Server, K8s |
| 1000+ cameras | Distributed + edge-cloud hybrid | Edge inference (Jetson), cloud for retraining, global model registry |

**Backpressure handling (common follow-up question):**
- Frame dropping with priority queue (newest frames first)
- Adaptive batch sizing based on queue depth
- Auto-scaling inference workers (K8s HPA on GPU utilization)

### 4.4 Key Infrastructure Components

| Component | Purpose | Tools |
|---|---|---|
| **Model Serving** | Low-latency GPU inference | Triton Inference Server, TorchServe, TensorFlow Serving |
| **Message Queue** | Decouple producers/consumers | Kafka (high throughput), Redis Streams (low latency) |
| **Vector DB** | Embedding similarity search | Milvus, Pinecone, Weaviate, pgvector |
| **Orchestration** | Container management | Kubernetes + GPU operator |
| **Monitoring** | System + model health | Prometheus/Grafana, DataDog |

---

## 5. MLOps & Production ML

### 5.1 Experiment Tracking

**What to track (your project does this via MLflow + DagsHub):**
- Hyperparameters: learning rate, batch size, epochs, optimizer
- Metrics: mAP, precision, recall, loss curves (per epoch)
- Artifacts: model weights, config YAML, dataset version
- Environment: Python version, package versions, GPU type

**Reproducibility checklist:**
1. Code version → git SHA
2. Data version → DVC hash
3. Config → YAML file (committed to git)
4. Environment → Docker image or `requirements.txt`
5. Random seeds → pinned in config

> **Your project:** `Trainer` class logs all params/metrics to MLflow
> (`trainer.py`), config is YAML-driven (`config/training/base.yaml`),
> MLflow tracking URI points to DagsHub.

### 5.2 Deployment Strategies

| Strategy | How It Works | Risk Level | When to Use |
|---|---|---|---|
| **Shadow** | New model processes real traffic, responses discarded | Zero | First deployment, error analysis |
| **Canary** | Route 1-5% of traffic to new model, serve real responses | Low | Gradual rollout with real feedback |
| **Blue/Green** | Two identical envs, instant traffic switch | Medium | When you need instant rollback |
| **A/B Test** | Split users into control/treatment | Medium | Measuring business metrics |

**Key distinction interviewers test:**
- Shadow = new model output is **never shown to users** (monitoring only)
- Canary = new model **actually serves** a small percentage of users

### 5.3 Drift Detection

**Two types (know both):**

| Type | What Changes | Detection Method | Example |
|---|---|---|---|
| **Model drift** | Prediction quality degrades | Rolling avg confidence, error rate monitoring | Model accuracy drops after store renovation |
| **Data drift** | Input distribution shifts | KS-test, PSI on feature distributions | Lighting changes seasonally, new camera angle |
| **Concept drift** | Input→output relationship changes | Label distribution shift, prediction distribution | New product types appear that model never saw |

> **Your implementation:** `DriftDetector` (`drift_detector.py`) monitors model
> drift via rolling confidence window. When avg confidence drops below threshold
> (default 0.3), drift is flagged. The docstring at `drift_detector.py:17-28`
> explicitly describes how to extend this with a `DataDriftDetector` using
> KS-test on image features like brightness/contrast.

**Interview answer for "How do you detect drift?":**

> "I implement two layers of drift detection. For model drift, I maintain a
> sliding window of prediction confidence scores and flag when the rolling
> average drops below a threshold — this catches gradual degradation. For data
> drift, I'd compute lightweight image statistics (brightness histogram,
> resolution distribution) and run KS-tests against a training-time baseline.
> When either triggers, the system logs a drift event and can automatically
> initiate a retraining pipeline with the most recent active learning data."

### 5.4 CI/CD for ML

**Pipeline stages:**
```
Code Change → Unit Tests → Data Validation → Training → Evaluation Gate → Shadow Deploy → Canary → Full Deploy
                                                │
                                        mAP < threshold?
                                          YES → Block deployment
                                          NO  → Continue
```

**Evaluation gate (critical concept):**
- Define minimum metrics: mAP ≥ X, latency ≤ Y ms, throughput ≥ Z FPS
- Compare against current production model (regression testing)
- Block deployment if any metric regresses beyond tolerance

### 5.5 Model Versioning & Registry

**What a model registry provides:**
- Version control for model artifacts (weights, configs, metadata)
- Stage management: Staging → Production → Archived
- Lineage: which training run, dataset, and config produced this model
- Rollback: instant revert to any previous version

> **Your project:** MLflow Model Registry configured in `config/training/base.yaml:46`
> with `model_name: visitor-analytics-yolo11n`. The `Trainer` class handles
> registration and stage promotion.

---

## 6. Active Learning & Data Flywheels

### 6.1 Why Active Learning Matters

Labeling is the bottleneck. Random sampling wastes budget on easy examples.
Active learning focuses human effort on the examples that improve the model most.

### 6.2 Sampling Strategies

| Strategy | How It Works | Best For |
|---|---|---|
| **Least confidence** | Select where max(P(y\|x)) is lowest | Classification |
| **Margin sampling** | Select where top-2 prediction gap is smallest | Multi-class |
| **Entropy sampling** | Select where prediction entropy is highest | General |
| **Committee (QBC)** | Ensemble of models, select where they disagree | When you have compute budget |
| **Diversity-based** | Cluster unlabeled data, sample from each cluster | Avoiding redundant labeling |

> **Your project uses confidence-based sampling** — the `DualDetector` saves frames
> where primary model confidence < threshold (`dual_detector.py:74-83`). This is
> a practical, production-friendly form of least-confidence sampling.

### 6.3 The Data Flywheel Architecture

```
Production Inference → Confidence Filter
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
        High Confidence            Low Confidence
        (auto-labeled,             (saved to disk)
         no action needed)              │
                                        ▼
                                 Secondary Model
                                 (RF-DETR pseudo-labels)
                                        │
                                        ▼
                              ┌─────────┴──────────┐
                              ▼                    ▼
                        Both models           One model
                        uncertain             confident
                              │                    │
                              ▼                    ▼
                        Human Review         Auto-labeled
                        (Roboflow UI)        (pseudo-label)
                              │                    │
                              └────────┬───────────┘
                                       ▼
                              Training Data Pool
                                       │
                                       ▼
                              Retrain Primary Model
                                       │
                                       ▼
                              Deploy → Cycle Repeats
```

**How your codebase implements this:**
1. `DualDetector._save_frame()` — saves image + JSON labels (`dual_detector.py:124-139`)
2. `AutoLabeler.load_dual_detector_frames()` — loads saved pseudo-labels (`auto_labeler.py:37-50`)
3. `AutoLabeler.flush()` — exports to Roboflow or local JSON (`auto_labeler.py:52-57`)
4. `AutoLabeler._upload_roboflow()` — uploads to Roboflow for human review (`auto_labeler.py:69-116`)
5. `Trainer` — retrains with new data, tracks in MLflow

**Interview talking point:**

> "Our data flywheel reduced the labeling burden by focusing exclusively on
> model failures. Instead of randomly sampling frames, we save only the frames
> where the primary detector's confidence drops below a threshold. A secondary
> model (RF-DETR) provides pseudo-labels for these hard cases. This means human
> annotators only need to verify or correct predictions — micro-annotation — rather
> than labeling from scratch, which is 3-5x faster."

---

## 7. Edge Deployment & Optimization

### 7.1 Optimization Stack

Apply in this order: **Prune → Distill → Quantize**

**Quantization (highest impact, lowest effort):**

| Precision | Memory | Speed Gain | Accuracy Impact |
|---|---|---|---|
| FP32 → FP16 | 2x reduction | 1.5-2x faster | Negligible (<0.1% mAP) |
| FP32 → INT8 | 4x reduction | 1.5-3x faster | Small (0-3% mAP drop) |
| FP32 → INT4 | 8x reduction | 2-4x faster | Moderate (needs calibration) |

**Two approaches:**
- **Post-Training Quantization (PTQ):** Quick. Calibrate with ~100-1000 representative images. No retraining.
- **Quantization-Aware Training (QAT):** Train with simulated quantization nodes. Better accuracy but more effort.

> **Interview tip:** Always mention calibration dataset. PTQ quality depends heavily
> on having representative calibration data from the target domain.

**Pruning (architectural simplification):**
- **Structured:** Remove entire filters/channels. Hardware-friendly. Typical: 50-80% pruning with <1% accuracy loss.
- **Unstructured:** Zero individual weights. Needs sparse hardware support.

**Knowledge Distillation (model compression):**
- Large teacher model trains small student model
- Student learns from teacher's soft labels (logits) + intermediate features
- Can achieve 5-50x model size reduction

### 7.2 Deployment Frameworks

| Framework | Platform | Best For |
|---|---|---|
| **TensorRT** | NVIDIA GPUs (Jetson, T4, A100) | Maximum NVIDIA performance |
| **ONNX Runtime** | Cross-platform | Portability |
| **TFLite** | Android, microcontrollers | Mobile inference |
| **CoreML** | Apple (iOS, macOS) | Apple ecosystem |
| **OpenVINO** | Intel CPUs/VPUs | Intel hardware |

**Real numbers to cite:**
- TensorRT on Jetson AGX Orin: SqueezeNet 0.66ms, ResNet152 2.28ms
- TensorRT typically **2-5x faster** than vanilla PyTorch on the same GPU
- YOLO11n INT8: ~2ms inference on T4 GPU

### 7.3 Latency vs Accuracy Trade-off

**The answer framework:**

> "The trade-off depends on the use case. For real-time applications like
> autonomous driving, I'd accept 1-2% mAP drop for 3x latency improvement
> via INT8 quantization — the safety margin comes from sensor fusion and
> temporal consistency, not single-frame accuracy. For offline batch
> processing like our auto-labeling pipeline, accuracy is paramount so
> I'd use FP32 with a larger model."

---

## 8. Coding Interview Patterns

### 8.1 IoU (Intersection over Union)

```python
def iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area (0 if no overlap)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
```

**Variants to know:**
- **GIoU:** Adds penalty for non-overlapping area → better gradient for non-overlapping boxes
- **DIoU:** Adds center distance penalty → faster convergence
- **CIoU:** Adds aspect ratio penalty → most complete loss function

### 8.2 Non-Maximum Suppression

```python
def nms(boxes, scores, iou_threshold=0.5):
    """Standard NMS. boxes: Nx4 [x1,y1,x2,y2], scores: N."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order
                 if iou(boxes[i], boxes[j]) < iou_threshold]

    return keep
```

**Soft-NMS variant (know conceptually):**
Instead of removing overlapping boxes, decay their scores:
`score_j = score_j * exp(-iou² / sigma)` — this preserves nearby detections.

### 8.3 Image Convolution

```python
def convolve2d(image, kernel):
    """2D convolution (no padding, stride=1)."""
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)

    return output
```

**Output size formula:** `(W - K + 2P) / S + 1`
- W = input size, K = kernel size, P = padding, S = stride

**Common kernels to know:**
- Sobel (edge detection): `[[-1,0,1],[-2,0,2],[-1,0,1]]`
- Gaussian blur: weighted average kernel
- Sharpening: center-heavy kernel with negative surroundings

### 8.4 Connected Components (BFS)

```python
from collections import deque

def connected_components(grid):
    """Count connected components in a binary grid (4-connectivity)."""
    rows, cols = len(grid), len(grid[0])
    visited = set()
    count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in visited:
                # BFS flood fill
                queue = deque([(r, c)])
                visited.add((r, c))
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and grid[nr][nc] == 1
                                and (nr, nc) not in visited):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                count += 1

    return count
```

### 8.5 Camera Geometry (3D → 2D Projection)

```
p_2d = K · [R | t] · P_3d

K = [fx  0  cx]    Intrinsic matrix (focal length, principal point)
    [0  fy  cy]
    [0   0   1]

[R|t] = 3x4 extrinsic matrix (rotation + translation)

P_3d = [X, Y, Z, 1]^T   (homogeneous 3D point)
p_2d = [u, v, w]^T       (homogeneous 2D, divide by w for pixel coords)
```

**Homography:** 3x3 matrix mapping between two planes.
- Requires 4+ point correspondences to solve
- Use RANSAC for robust estimation with outliers

### 8.6 Additional Patterns

| Pattern | Data Structure | Application |
|---|---|---|
| Sliding window | Array/Matrix | Kernel operations, feature extraction |
| Priority queue | Heap | Top-K detections, NMS |
| Union-Find | Disjoint set | Connected components at scale |
| KD-Tree | Spatial tree | Nearest neighbor in feature space |
| 2D BFS/DFS | Grid/Matrix | Segmentation, flood fill |

---

## 9. Behavioral Interview Strategy

### 9.1 The STAR-L Framework

- **S**ituation: Context and challenge
- **T**ask: Your specific responsibility
- **A**ction: What YOU did (be specific, use "I" not "we")
- **R**esult: Quantified outcome
- **L**earning: What you'd do differently (shows growth)

### 9.2 Your Project Story (Prepared Answer)

> **S:** "I needed to build a production-grade retail analytics system for
> person detection and tracking that could improve itself over time without
> constant manual intervention."
>
> **T:** "Design and implement the full pipeline: detection, tracking,
> analytics persistence, drift monitoring, and an active learning loop."
>
> **A:** "I implemented a dual-detector architecture with YOLOv11 as the
> primary model for real-time inference and RF-DETR as a secondary model
> for edge-case collection. The system automatically identifies frames where
> the primary model struggles, saves them with pseudo-labels, and feeds them
> into the retraining pipeline through Roboflow. I built a config-driven
> drift detector that monitors rolling confidence scores and flags degradation.
> Everything is tracked through MLflow on DagsHub for experiment reproducibility."
>
> **R:** "The system processes video at 30+ FPS for single-camera setups,
> automatically collects the hardest examples for retraining, and provides
> real-time visitor analytics through a Streamlit dashboard."
>
> **L:** "I learned that the MLOps infrastructure — config management, drift
> detection, active learning loops — delivers more production value than
> optimizing the model architecture itself. A mediocre model with great
> infrastructure beats a perfect model with no observability."

### 9.3 Stories Every Candidate Should Prepare

| Story Type | What It Proves | Your Project Angle |
|---|---|---|
| **Performance improvement** | Methodology | DualDetector ensemble → better recall on hard cases |
| **Failed experiment** | Resilience | Tried high-confidence auto-labeling → realized it doesn't improve the model, pivoted to active learning (low-confidence only) |
| **Shipped with limitations** | Pragmatism | SQLite for analytics (not Postgres) — right choice for MVP scope |
| **Infrastructure investment** | Communication | Built centralized logging, config system, MLflow tracking before model tuning |
| **Production debugging** | Problem-solving | Drift detection caught confidence degradation → triggered retraining |
| **Disagreement** | Collaboration | Chose active learning over random sampling despite being more complex |

### 9.4 How to Talk About Failed Experiments

**Structure:** Hypothesis → Experiment → Result → Insight → Pivot

> "We originally collected both high-confidence and low-confidence frames for
> auto-labeling. The hypothesis was that high-confidence frames provide clean
> training data. But we realized that the model already handles those cases
> well — adding more 'easy' examples doesn't improve performance. The insight
> was that active learning should focus on failures, not successes. We stripped
> out high-confidence collection entirely and focused on the frames where the
> model struggles most."

### 9.5 How to Talk About Technical Debt

> "I chose SQLite for analytics persistence because it's zero-config and
> perfect for single-camera MVP scope. I explicitly documented in the
> architecture that this should migrate to PostgreSQL when scaling beyond
> ~10 concurrent camera streams. The migration path is clean because all
> DB access goes through the `AnalyticsDB` class — swap the backend,
> keep the interface."

---

## 10. System Design Mock Questions

### Q1: "Design a Real-Time Video Analytics Pipeline"

**This is literally your project.** Walk through:

1. **Requirements:** Single camera, person detection + tracking, dwell time analytics, <100ms latency
2. **Architecture:** Frame capture → Detection (YOLO) → Tracking (ByteTrack) → Analytics (SQLite) → Dashboard (Streamlit)
3. **Deep dive:** Factory pattern for detectors, singleton model caching, config-driven thresholds
4. **Scaling:** Add message queue, move to Triton, K8s auto-scaling
5. **Monitoring:** Confidence drift detection, frame processing latency, throughput metrics

### Q2: "Design a Content Moderation System"

**Requirements:** Classify images as safe/unsafe across categories (NSFW, violence, hate symbols). Scale to billions of images/day.

**Architecture:**
```
Upload → Lightweight Hash Check (known bad content) → ML Classification Pipeline
    │                                                         │
    │                                              ┌──────────┴──────────┐
    │                                              ▼                     ▼
    │                                        High Confidence       Low Confidence
    │                                        (auto-action)         (human review queue)
    │                                              │                     │
    │                                              ▼                     ▼
    │                                        Block/Allow            Human Decision
    │                                                                    │
    │                                                                    ▼
    │                                                              Training Data
    └──── Log all decisions for audit trail
```

**Key design choices:**
- Multi-stage pipeline: hash → lightweight model → heavyweight model (cost optimization)
- Confidence thresholds determine auto-action vs. human review (same pattern as your DualDetector)
- Active learning: low-confidence cases improve the model over time
- Regional content policies: different thresholds per geography

### Q3: "Design a Visual Search Engine"

**Requirements:** User uploads a photo, system returns visually similar products. Scale to 100M product catalog.

**Architecture:**
```
Query Image → Feature Extraction (CLIP/DINOv2) → Embedding Vector
    │
    ▼
Vector Database (Milvus/Pinecone) → Approximate Nearest Neighbor Search
    │
    ▼
Top-K Candidates → Re-ranking (cross-attention model) → Results
```

**Key design choices:**
- **Embedding model:** CLIP for text+image search, DINOv2 for pure visual similarity
- **ANN algorithm:** HNSW or IVF-PQ for sub-millisecond search at 100M+ scale
- **Index updates:** Batch pipeline for new products, real-time for removals
- **Re-ranking:** Lightweight model on top-100 candidates for final ordering

### Q4: "Design a Face Recognition System"

**Key concepts:**
- **Enrollment:** Extract face embedding → store in database with identity
- **Verification (1:1):** Compare two embeddings, threshold on cosine similarity
- **Identification (1:N):** Query embedding against all enrolled faces, return closest
- **Liveness detection:** Prevent spoofing with depth/IR sensors or challenge-response
- **Privacy:** Embedding is one-way (can't reconstruct face from vector), GDPR compliance

### Q5: "How Would You Scale Your Current System to 1000 Cameras?"

**The graduated scaling answer:**

> "My current architecture is a monolith optimized for single-camera MVP — that's
> intentional. To scale to 1000 cameras, I'd make three architectural changes:
>
> First, **decouple frame capture from inference** using Kafka. Each camera pushes
> frames to a topic, inference workers consume from the topic. This handles
> backpressure — if workers are saturated, we drop oldest frames, not newest.
>
> Second, **move model serving to Triton Inference Server** with dynamic batching.
> Instead of one inference per frame, Triton batches frames across cameras and
> runs them as a single GPU operation — this 4-8x throughput.
>
> Third, **edge-cloud hybrid** for bandwidth. Run YOLO on edge devices (Jetson)
> at each camera. Only send low-confidence frames to the cloud for RF-DETR
> processing. This reduces bandwidth by 95%+ since most frames are routine.
>
> The factory pattern in my codebase (`DetectorFactory`, `TrackerFactory`) makes
> this migration clean — swap implementations without changing the pipeline logic."

---

## Quick Reference Card

### Concepts You Must Define on the Spot

| Term | One-Line Definition |
|---|---|
| **mAP** | Mean average precision across IoU thresholds — the standard detection metric |
| **IoU** | Intersection over Union — measures overlap between predicted and ground truth boxes |
| **NMS** | Non-Maximum Suppression — removes duplicate overlapping detections |
| **FPN** | Feature Pyramid Network — multi-scale feature maps for detecting objects at different sizes |
| **Anchor-free** | Directly predict center + size instead of offsets from predefined templates |
| **Attention** | Mechanism where each token computes weighted sum of all other tokens based on similarity |
| **FlashAttention** | IO-aware exact attention that minimizes GPU memory reads/writes — no accuracy trade-off |
| **Quantization** | Reduce numerical precision (FP32→INT8) for faster inference with minimal accuracy loss |
| **Knowledge Distillation** | Train small student model to mimic large teacher model's predictions |
| **Active Learning** | Strategically select the most informative samples for labeling |
| **Data Drift** | Input distribution shifts from what the model was trained on |
| **Model Drift** | Model's prediction quality degrades over time |
| **Contrastive Learning** | Learn representations by pulling similar pairs together, pushing dissimilar apart |
| **Self-Supervised Learning** | Learn representations from unlabeled data (no human annotation) |
| **ONNX** | Open Neural Network Exchange — portable model format across frameworks |
| **Triton** | NVIDIA inference server with dynamic batching and multi-model serving |

### Your Project's Design Decisions (Ready-Made Interview Answers)

| Decision | Why | Alternative Considered |
|---|---|---|
| YOLOv11 primary | Fastest inference (13.5ms), 22% fewer params than v10 | YOLOv8 (more mature but slower) |
| RF-DETR secondary | DINOv2 backbone generalizes well to new domains | Grounding DINO (too slow for our use case) |
| ByteTrack tracker | Simple, fast, no re-ID model needed for single-camera | DeepSORT (heavier, needs re-ID model) |
| SQLite analytics | Zero-config, perfect for single-camera MVP | PostgreSQL (overkill for MVP) |
| YAML config | Human-readable, git-trackable, environment-overridable | JSON (less readable), env vars only (no structure) |
| MLflow + DagsHub | Free tier, Git-integrated, model registry included | W&B (paid), custom solution (maintenance burden) |
| Factory pattern | Swap detectors/trackers without changing pipeline code | Direct instantiation (tight coupling) |
| Active learning only | Model improves from failures, not redundant easy examples | Random sampling (wastes labeling budget) |

---

*Use this playbook as a learning resource. Every concept connects back to code
you've written. When an interviewer asks "have you done this?", you can point
to specific files and explain real implementation decisions.*
