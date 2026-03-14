# 📋 Portfolio Readiness To-Do

**Status**: ~70% ready for interviews. Strong architecture foundation, needs production hardening & working examples.

---

## 🟢 STRENGTHS (Already Portfolio-Ready)

- ✅ Clear MVP scope & strategic docs (not overengineered)
- ✅ Modular detection/tracking/training/inference architecture
- ✅ Config-driven system (no magic numbers, YAML-based)
- ✅ MLflow integration for experiment tracking
- ✅ Multiple detector backends (YOLO, RF-DETR abstraction)
- ✅ Precision control (FP32, FP16, INT8 quantization)
- ✅ Online/offline inference modes
- ✅ ByteTrack multi-object tracking integration
- ✅ Analytics module (visitor counting, dwell time)
- ✅ Comprehensive documentation (STRATEGY, REPO_ARCHITECTURE, etc.)

---

## 🔴 CRITICAL GAPS (Block Interviews)

### 1. **Zero Working Examples**
- [ ] Add `examples/` folder with runnable scripts:
  - `example_inference.py` — Load video, run pipeline, show output
  - `example_train.py` — Complete training run with MLflow tracking
  - `example_drift_detection.py` — Simulate drift, trigger retrain
- **Why**: Interviewers will ask "Can you show me this working?" You need proof.

### 2. **No Benchmarking / Performance Data**
- [ ] Run & document inference benchmarks:
  - Latency (PyTorch vs ONNX vs INT8)
  - Throughput (FPS at different batch sizes)
  - GPU/CPU memory usage
  - Comparative analysis with 2-3 sentences explaining findings
- [ ] Add `benchmarks/` folder with results (markdown + plots)
- **Why**: "How would you scale this?" needs concrete numbers, not theory.

### 3. **Test Coverage < 5%**
- [ ] Expand `tests/` to cover:
  - Detector initialization & detection (all backends)
  - Tracker state management
  - Config loading & merging
  - Pipeline end-to-end (mock video)
  - Drift detector logic
- [ ] Run `pytest` in CI/CD (add GitHub Actions workflow)
- **Why**: Portfolio projects show production discipline.

### 4. **Drift Detection Not Visible**
- [ ] Implement `src/vision_ml/training/drift_detector.py` fully:
  - Confidence score distribution shift
  - Trigger-based retraining logic
  - Clear threshold tuning
- [ ] Add example: `examples/example_drift_detection.py`
- **Why**: Core MLOps concept. Must explain it confidently.

### 5. **DVC Not Wired In**
- [ ] Initialize DVC & add sample dataset versioning:
  - `dvc.yaml` with training/evaluation pipeline
  - Track at least 1-2 model versions
  - Document: "How to reproduce any past experiment"
- **Why**: Shows reproducibility discipline (FAANG asks for this).

---

## 🟡 IMPORTANT (Elevates Portfolio)

### 6. **No Executable Demo**
- [ ] Create `demo.py` or Streamlit app:
  - Upload video / paste video URL
  - Run inference
  - Show detections, tracks, analytics
  - Display metrics (FPS, confidence, unique visitors)
- [ ] Screenshot or video of demo in README
- **Why**: Makes repo memorable, shows full-stack thinking.

### 7. **System Design Gaps**
- [ ] Add `docs/SCALABILITY.md` with:
  - Current: Single model, single stream (what works now)
  - 10 streams: Batching + multiprocessing strategy
  - 1000 streams: GPU sharding, load balancing, queue management
  - Clear trade-offs (latency vs throughput)
- **Why**: "Design a video analytics system for 1000 cameras" — interviewers will ask this.

### 8. **Config Merging Not Implemented**
- [ ] Implement `vision_ml.utils.config.merge_configs()`:
  - Base config + experiment override pattern
  - Test it, use in examples
- [ ] Document: "How we do A/B testing / hyperparameter sweeps"
- **Why**: Shows FAANG-style config discipline.

### 9. **No CI/CD Pipeline**
- [ ] Add `.github/workflows/test.yml`:
  - Run tests on every push
  - Show test coverage badge in README
- [ ] Minimal: just `pytest tests/` + basic lint
- **Why**: Shows professional development practice.

### 10. **Quantization Guide Incomplete**
- [ ] Complete `docs/MODEL_QUANTIZATION_GUIDE.md`:
  - Actual benchmarks: FP32 vs INT8 (latency, accuracy)
  - When to use each (trade-off story)
  - Clear decision tree
- **Why**: Interview question: "How do you optimize for edge deployment?"

---

## 🟢 NICE-TO-HAVE (If Time)

- [ ] Monitoring setup (Prometheus metrics + Grafana dashboard mock)
- [ ] Airflow DAG for scheduled retraining (airflow/dags/retrain.py)
- [ ] Model export to ONNX documented with benchmarks
- [ ] Real dataset versioning (3-4 versions showing drift simulation)
- [ ] REST API skeleton (FastAPI for serving)

---

## 📊 Priority Matrix

| Task | Effort | Impact | Do First? |
|------|--------|--------|-----------|
| Working examples | 2h | 🔴 CRITICAL | YES |
| Benchmarking | 3h | 🔴 CRITICAL | YES |
| Tests expansion | 2h | 🔴 CRITICAL | YES |
| Drift detection demo | 1.5h | 🔴 CRITICAL | YES |
| Scalability doc | 1.5h | 🟡 HIGH | YES |
| System design doc | 1h | 🟡 HIGH | YES |
| DVC setup | 1h | 🟡 MEDIUM | Maybe |
| Demo app | 2h | 🟡 MEDIUM | Maybe |
| CI/CD | 1h | 🟡 MEDIUM | Maybe |
| Quantization guide | 1.5h | 🟡 MEDIUM | Maybe |

**Total Critical Path: ~8-9 hours** → Portfolio-grade readiness

---

## 🎯 Success Criteria

After completing this to-do, you should be able to:

1. **Run a full example** in 30 seconds: `python examples/example_inference.py demo.mp4`
2. **Show benchmarks**: "INT8 is 2.5x faster with <2% accuracy drop"
3. **Explain drift detection**: Live example of trigger + retrain logic
4. **Defend the architecture**: "Here's why we scaled this way for 1000 streams"
5. **Prove reproducibility**: "Run this DVC command to recreate experiment #5"
6. **Show discipline**: Test coverage, CI passing, clean configs

**Interview Signal**: "This person understands real ML systems, not just training notebooks."

---

## 📝 Notes

- **Not Production**: You don't need Kubernetes, load balancers, or 99.99% uptime. But you DO need:
  - Working code that can be run
  - Thoughtful scaling strategy
  - Test coverage
  - Reproducible experiments
  - Clear explanations of trade-offs

- **Depth Over Breadth**: Pick 2-3 concepts to go *deep* on:
  - Example: "Drift detection + retraining triggers" (complete story)
  - Don't add "monitoring" if you haven't finished drift detection

- **Pluggable Design** (your goal):
  - Swap detectors? ✅ (DetectorFactory)
  - Swap trackers? ✅ (TrackerFactory)
  - Swap backends? ✅ (Config-driven)
  - Add new metrics? ✅ (Config + analytics module)
  - Scale to many streams? ⚠️ (Needs system design doc + scaling example)
