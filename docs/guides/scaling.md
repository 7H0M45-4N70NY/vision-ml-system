# 📈 Vision ML System — Scaling Analysis

---

# 1. Baseline Measurement

CPU Full Pipeline Inference:
~257 ms per image (~3.9 FPS)

Includes:
- Preprocessing
- Forward pass
- Postprocessing

---

# 2. Batch Scaling Analysis

Batch sizes tested:
1, 4, 8

Goal:
Determine if per-image latency decreases.

If yes:
→ CPU vectorization effective.

If no:
→ Compute-bound architecture.
→ GPU required.

---

# 3. Resolution Scaling

Attention complexity ≈ O((H × W)^2)

Higher resolution dramatically increases cost.

Testing resolutions:
- 640x640
- 800x800
- 1024x1024

---

# 4. CPU vs GPU Tradeoffs

CPU:
- Limited parallelism
- Memory-bound risk

GPU:
- Massive parallel compute
- Transformer acceleration
- Higher throughput

---

# 5. Multiprocessing Considerations

- GIL limitations
- Memory duplication risk
- GPU context sharing issues
- Worker pool sizing strategy

---

# 6. Deployment Insight

Real-time streaming (30 FPS) requires:
- GPU acceleration
- Resolution control
- Possibly quantization

CPU suitable for:
- Offline batch processing
- Low-throughput pipelines

---

# 7. Model Batch Scaling (RF-DETR vs YOLO)

## Transformer-Based (RF-DETR Nano)
- Batch 1 → ~40 ms per image (~25 FPS)
- Batch 4 → ~38 ms per image (~26 FPS)
- Batch 16 → ~43 ms per image (~23 FPS)
- **Insight**: Model is memory-bandwidth bound due to global self-attention (O(N²) complexity). Increasing batch size beyond 4 increases memory usage linearly but does not improve throughput.

## CNN-Based (YOLO Nano)
- Batch 1 → ~34 ms per image (~29 FPS)
- Batch 4 → ~4.6 ms per image (~218 FPS)
- Batch 16 → ~4.5 ms per image (~220 FPS)
- **Insight**: Convolution operations scale efficiently with batch size. Better GPU tensor core utilization and memory locality lead to massive batch scaling (7.5x throughput improvement).

## Scaling Strategy
- **Real-Time Systems**: Latency-constrained. Small batch sizes. Both models viable depending on accuracy needs.
- **Offline Analytics**: Throughput-constrained. CNNs (YOLO) are dramatically more cost-efficient for batch systems due to high hardware utilization.