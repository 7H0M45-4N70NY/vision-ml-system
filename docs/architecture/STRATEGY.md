# 🔥 Strategic Vision: Retail Analytics MVP

---

## The Decision: Controlled Ambition

**We are NOT building Amazon Go.**

We are building a **clean, engineered Retail Analytics MVP** with:

- **Small SKU set** (5–10 mock brand store products)
- **Controlled environment** (clean shelf layout, consistent lighting)
- **Synthetic or generated videos** (for demo and drift simulation)
- **Strong MLOps backbone** (MLflow, DVC, Airflow, monitoring)

This is **intelligent portfolio strategy**.

---

## Why This Scope Is Brilliant

### 1. Scope is Controlled
- 5–10 SKUs max (not thousands)
- Category-level detection sufficient
- Optional SKU-level only if dataset is clean and small

### 2. Dataset is Manageable
- Curated labeled frames for training
- Generated videos for demo + drift simulation
- No industrial chaos, no annotation nightmare

### 3. Retraining Pipeline is Realistic
- Drift simulation possible (lighting change, shelf rearrangement, camera angle shift)
- Metric-based retraining triggers
- Versioned model promotion logic

### 4. Monitoring is Meaningful
- Prometheus metrics (latency, confidence, interaction counts)
- Grafana dashboards (drift score, performance decay)
- Actionable alerts

### 5. Real Value Becomes Visible
The important bits:
- **MLflow** + **DVC** + **Airflow** + **Drift** + **Monitoring** + **Versioned Serving**

Not toy code. Not overengineered chaos. **Production thinking at portfolio scale.**

---

## Architecture Summary (Final Form)

### Vision Layer
- YOLO26 (person + product category detection)
- ByteTrack (persistent visitor IDs)
- Interaction rule engine

### Data Layer
- DVC dataset versioning
- Simulated drift dataset versions:
  - Lighting change
  - Shelf rearrangement
  - Camera angle shift

### Experiment Layer
- MLflow tracking
- Model registry
- Version promotion logic

### Automation Layer
- Airflow retraining DAG
- Drift-triggered retrain

### Monitoring Layer
- Prometheus metrics
- Grafana dashboards:
  - Latency
  - Detection confidence avg
  - Interaction counts
  - Drift score

---

## Important Reality Check

### Generating Store Videos
Video generation models are fine for:
- Concept demo
- UI visuals
- Presentation

**But training still needs:**
- Consistent labeled dataset

### Practical Approach
- **Use generated videos for**: Demo + drift simulation
- **Use curated/labeled frames for**: Training

Be pragmatic.

---

## What Makes This Project Powerful

Not that you detect products.

But that you can explain:

1. **Why small SKU set was chosen** — Scope management, realistic dataset size
2. **How drift was simulated** — Lighting, shelf rearrangement, camera angle
3. **How retraining triggers are defined** — Metric thresholds, scheduled jobs
4. **How model promotion works** — Registry, versioning, rollback
5. **How version rollback is handled** — Safety, reproducibility

**That's senior thinking.**

---

## Phase 1: Core System (This Phase)

### Detection + Tracking
- YOLO26 modular integration
- ByteTrack persistent tracking
- Config-driven inference

### Training Pipeline
- PyTorch training loop
- MLflow experiment tracking
- ONNX export
- Reproducible checkpointing

### Benchmarking
- Latency analysis (CPU vs GPU)
- Batch scaling experiments
- Resolution impact study
- Memory profiling

### Documentation
- Architecture diagrams
- Scaling strategy
- Bottleneck analysis
- Interview-ready explanations

---

## Phase 2: MLOps Automation (Future)

- DVC drift dataset versions
- Airflow retraining DAG
- Prometheus + Grafana monitoring
- Automated model promotion
- Drift detection (Evidently/Alibi-Detect)

---

## Phase 3: Business Logic (Future)

- Dwell time analytics
- Heatmap generation
- Conversion funnel tracking
- Peak hour analysis

---

## Definition of Done

Project is complete when:

- ✅ Detection + tracking pipeline stable
- ✅ Training pipeline reproducible
- ✅ ONNX export validated
- ✅ Benchmarks documented
- ✅ Scaling behavior analyzed
- ✅ Tests passing
- ✅ Architecture & scaling documentation written
- ✅ Interview-level system explanations prepared

---

## Execution Philosophy

- **Depth over chaos** — Master one system deeply
- **Structure over randomness** — Clear repo organization
- **Documentation over assumption** — Every decision explained
- **Benchmarking over guessing** — Measured, not intuitive
- **Consistency over intensity** — Sustainable pace

This is not a hobby project.

This is a deliberate transition toward **ML Systems Engineering excellence**.
