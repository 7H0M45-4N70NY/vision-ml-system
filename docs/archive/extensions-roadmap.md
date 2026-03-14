# 🔮 Vision ML System — Extensions & Future Roadmap

**⚠️ DEPRECATED**: This document has been superseded by [ROADMAP.md](ROADMAP.md)

Please refer to ROADMAP.md for the current phase-based development plan.

---

## Original Extensions Document (Reference Only)

This document captures features intentionally excluded from Phase 1 to prevent scope creep.

---

# Phase 2 — Platform Enhancements

## 1. DVC Integration
- Dataset version control
- Snapshot reproducibility
- Data lineage tracking

## 2. DagsHub Integration
- Remote MLflow tracking
- Artifact registry
- Model version management

## 3. Data Drift Detection
- Input distribution monitoring
- Mean/std comparison
- KL divergence metrics

## 4. Model Drift Detection
- Confidence score monitoring
- Performance decay tracking

## 5. Automated Retraining
- Metric-based retraining trigger
- Scheduled retraining jobs
- Versioned model promotion

---

# Phase 3 — Advanced Automation

## Automatic Data Annotation
- Pseudo-labeling
- Semi-supervised loops

## Streaming Data Collection
- Ingestion simulation
- Batch data pipeline

## Observability Layer

### Prometheus
- Latency metrics
- Throughput metrics
- Resource usage metrics

### Grafana
- Real-time dashboards
- Historical trend visualization

---

# Phase 4 — Business Logic Extensibility (Retail Analytics Example)

The core detection and tracking capabilities can serve as the foundation for complex business use cases.

## Attribute Pipeline
- Face cropping from tracked bounding boxes
- Gender and Age estimation models
- Emotion detection per zone
- **Optimization**: Run periodically per track, not per frame

## Re-Identification (Re-ID) & Embeddings
- Extract face/body embeddings for repeat visitor detection
- Vector database integration for cosine similarity matching
- Privacy compliance filtering (GDPR)

## Advanced Analytics Engine
- Dwell time estimation per zone
- Heatmap generation
- Conversion funnel analytics
- Peak hour engagement analysis

---

# Future Research Directions

- Lightweight DETR variants
- Sparse attention mechanisms
- Dynamic resolution scaling
- Custom detection architecture

---

# Design Rule

A feature moves to implementation only if:

- Core system stable
- Benchmarks documented
- Tests passing
- It improves scalability, reproducibility, or observability

No feature added impulsively.