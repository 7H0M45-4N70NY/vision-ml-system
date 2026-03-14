# 🚀 Vision ML System — Master Plan

**⚠️ DEPRECATED**: This document has been superseded by:
- [STRATEGY.md](STRATEGY.md) — Strategic vision and scope
- [REPO_ARCHITECTURE.md](REPO_ARCHITECTURE.md) — Repository structure
- [ROADMAP.md](ROADMAP.md) — Phase-by-phase development plan

Please refer to those documents for current strategic and architectural guidance.

---

## Original Master Plan (Reference Only)

---

# 1. Strategic Context

This project is designed as a structured transition from:

Feature-Level AI Engineering  
→  
ML Systems Engineering

The goal is not to build a demo, but to build deep understanding in:

- Detection model internals
- End-to-end inference pipelines
- Training lifecycle management
- Performance benchmarking
- Scaling constraints
- Reproducibility discipline
- System design tradeoffs
- Cost-performance optimization

This project is intentionally engineered to simulate real-world ML production thinking. It represents a shift in engineering maturity:

- **Beginner**: "How does the model work?"
- **Intermediate**: "How fast does it run?"
- **Advanced**: "How much does it cost per million frames?"
- **Senior**: "How do I redesign the pipeline to reduce required inference by 90%?"

---

# 2. Architectural Understanding of RFDETR

Initial research confirmed that:

`model.predict(image)` performs the following internally:

1. Resize image
2. Normalize (ImageNet mean/std)
3. Convert to tensor
4. Pad to ensure spatial dimensions divisible by 32
5. Forward pass (CNN backbone + Transformer attention)
6. Post-processing (confidence filtering, thresholding)
7. Rescale boxes to original image resolution

Key insight:

- Backbone stride = 32 (5 downsampling stages).
- Input height & width must be divisible by 32.
- Attention cost scales approximately O((H × W)^2).
- Resolution directly impacts latency.

This decomposition informs:

- ONNX export design
- Batch benchmarking strategy
- Scaling architecture
- Deployment considerations

---

# 3. Project Objective

Build a modular, scalable vision system including:

- RFDETR-based detection abstraction
- ByteTrack tracking integration
- Config-driven training pipeline
- MLflow experiment tracking
- ONNX export and optimization
- Performance benchmarking framework
- Multiprocessing scaling experiments
- Clean architecture and documentation

---

# 4. Core Scope (Phase 1)

## Detection Layer
- RFDETR modular integration
- Manual preprocessing replication
- Batch inference benchmarking
- CPU/GPU compatibility
- Resolution scaling experiments

## Tracking Layer
- ByteTrack integration
- Persistent object ID tracking
- Dwell-time computation
- Detection-to-tracker interface abstraction

## Training Pipeline
- Config-driven hyperparameters (YAML)
- Fine-tuning capability
- MLflow logging
- Checkpoint saving
- Reproducible experiment tracking
- ONNX export after training

## Benchmarking Suite
- Full pipeline timing (.predict())
- Pure forward pass timing
- Batch size scaling (1, 4, 8)
- Resolution scaling impact
- CPU vs GPU comparison
- Memory profiling (psutil / CUDA)

## Engineering Discipline
- SOLID principles
- Clean modular folder structure
- Unit tests (pytest)
- Deterministic seed control
- Documentation-first development

---

# 5. Non-Goals (Phase 1)

Intentionally excluded:

- Kubernetes
- Kafka
- ViT or alternate backbones
- Custom architecture research
- Enterprise automation
- Production cloud deployment

Focus: depth over breadth.

---

# 6. Research vs Engineering Boundary

Level 1: User → Calls `.predict()`  
Level 2: Engineer → Understands preprocessing, stride constraints, scaling  
Level 3: Researcher → Designs new architecture  

This project operates intentionally at Level 2.

Architecture research is reserved for future exploration.

---

# 7. Learning Outcomes

By completion, I must confidently explain:

- Why divisible-by-32 is required
- Why padding vs resizing matters
- Attention complexity scaling
- Where latency bottlenecks occur
- When batching improves throughput
- Why CPU saturates
- When GPU becomes mandatory
- ONNX optimization boundaries
- Quantization tradeoffs
- GIL and multiprocessing constraints

If I cannot explain it clearly, I do not understand it deeply enough.

---

# 8. Execution Philosophy

- Depth over tool count
- Structure over chaos
- Measured benchmarking over guessing
- Modularity over monolithic scripts
- Documentation over assumptions
- Consistency over intensity

---

# 9. Definition of Done

Project is complete when:

- Detection + tracking pipeline stable
- Training pipeline reproducible
- ONNX export validated
- Benchmarks documented
- Scaling behavior analyzed
- Tests passing
- Architecture & scaling documentation written
- Interview-level system explanations prepared