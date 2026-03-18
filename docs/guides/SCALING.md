# Scaling Guide (Current)

This guide captures practical scaling guidance for the current Vision ML stack.

## 1) Baseline First

Before optimizing, measure baseline latency and throughput for:

- detector inference
- tracking
- end-to-end pipeline (decode -> detect -> track -> analytics)

Use a fixed dataset and fixed hardware profile when comparing runs.

## 2) Throughput vs Latency Tradeoff

- **Real-time mode**: optimize tail latency (p95/p99), small batches
- **Batch analytics mode**: optimize throughput, larger batches when hardware allows

## 3) Model-Level Considerations

- YOLO-style CNN detectors generally scale well with batching
- Transformer-based detectors may become memory-bandwidth bound earlier
- choose model per workload, not by single benchmark number

## 4) Resolution Strategy

Higher resolution increases compute and memory significantly.

Recommended process:

1. pick target FPS / SLA
2. benchmark at candidate resolutions (e.g., 640, 800, 1024)
3. select the smallest resolution meeting accuracy goals

## 5) CPU vs GPU

- CPU is often sufficient for low-throughput or offline jobs
- GPU is typically required for stable real-time multi-stream workloads
- profile both to avoid assumptions

## 6) Optimization Order

Apply optimizations in this order:

1. profiling and bottleneck identification
2. input pipeline improvements (decode, preprocessing, batching)
3. model/runtime optimization (ONNX/TensorRT/quantization if applicable)
4. horizontal scaling (process isolation, worker pools, stream sharding)

## 7) Metrics to Track

Track these consistently:

- p50/p95/p99 latency
- FPS / throughput
- GPU utilization and memory
- CPU utilization
- dropped frame ratio
- detection confidence distribution drift

## 8) Production Recommendations

- define SLOs (latency + quality)
- keep one benchmark harness for repeatability
- log benchmark metadata (model version, config hash, hardware, commit SHA)
- compare only like-for-like runs

## Related Docs

- [Repository Architecture](../architecture/REPO_ARCHITECTURE.md)
- [Training Pipeline](./TRAINING_PIPELINE.md)
- [DagsHub + MLflow + DVC](./DAGSHUB_MLFLOW_DVC_INTEGRATION.md)
- [Documentation Index](../INDEX.md)
