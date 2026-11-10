# Vision ML System Documentation Index

Welcome to the documentation for **VisionFlow / Vision ML System**. This is the single navigation entry point for understanding the system architecture, setting up your environment, running the active learning flywheel, and scaling inference.

---

## 🧭 Reader Pathways (Choose Your Goal)

- **🚀 Quickstart Track (Get it running in 5 minutes):**
  1. [Project Overview](../README.md)
  2. [Environment & Config Setup](./getting-started/RUN_SETUP_ENV_CONFIG.md)
  3. [Quick Start Guide](./getting-started/QUICKSTART.md)
  4. Hardware verification: run `python scripts/check_cuda.py` and `python scripts/env_check.py`

- **🎓 Portfolio / Interview Track (Architecture & Deep Dive):**
  1. [Master Interview & Systems Playbook](../INTERVIEW_PLAYBOOK.md)
  2. [System Design & Microservices](./architecture/SYSTEM_DESIGN.md)
  3. [Strategic Vision & Scope](./architecture/STRATEGY.md)
  4. [Repository Architecture & Contracts](./architecture/REPO_ARCHITECTURE.md)

- **⚙️ Engineering & MLOps Track (Pipelines & Infrastructure):**
  1. [Hybrid Dual-Detector Modes](./guides/HYBRID_DETECTOR_MODES.md)
  2. [Dataset Structure & DVC Versioning](./guides/DATASET_STRUCTURE.md)
  3. [Training Pipeline & Ingestion](./guides/TRAINING_PIPELINE.md)
  4. [DagsHub + MLflow + DVC Integration](./guides/DAGSHUB_MLFLOW_DVC_INTEGRATION.md)
  5. [Active Learning with Roboflow](./guides/ROBOFLOW_GUIDE.md)

---

## 1) Getting Started

- Project overview: [../README.md](../README.md)
- Quick setup & pipeline run: [getting-started/QUICKSTART.md](./getting-started/QUICKSTART.md)
- Environment variables & config reference: [getting-started/RUN_SETUP_ENV_CONFIG.md](./getting-started/RUN_SETUP_ENV_CONFIG.md)

## 2) Architecture & Strategy

- Strategy and scope: [architecture/STRATEGY.md](./architecture/STRATEGY.md)
- System design: [architecture/SYSTEM_DESIGN.md](./architecture/SYSTEM_DESIGN.md)
- Repository layout and import boundaries: [architecture/REPO_ARCHITECTURE.md](./architecture/REPO_ARCHITECTURE.md)
- Interview playbook: [../INTERVIEW_PLAYBOOK.md](../INTERVIEW_PLAYBOOK.md)

## 3) Hardware & Model Optimization

- CUDA, PyTorch & RTX 4050 Edge Guide: [guides/RTX4050_CUDA_TORCH_GUIDE.md](./guides/RTX4050_CUDA_TORCH_GUIDE.md)
- Model Quantization & Optimization (FP32, FP16, INT8): [guides/MODEL_QUANTIZATION_GUIDE.md](./guides/MODEL_QUANTIZATION_GUIDE.md)
- Scaling heuristics & bottlenecks: [guides/SCALING.md](./guides/SCALING.md)

## 4) Data, Training, and MLOps

- Dataset and DVC layout: [guides/DATASET_STRUCTURE.md](./guides/DATASET_STRUCTURE.md)
- Training pipeline: [guides/TRAINING_PIPELINE.md](./guides/TRAINING_PIPELINE.md)
- DagsHub + MLflow + DVC: [guides/DAGSHUB_MLFLOW_DVC_INTEGRATION.md](./guides/DAGSHUB_MLFLOW_DVC_INTEGRATION.md)
- Secrets management: [guides/SECRETS_INJECTION_PATTERN.md](./guides/SECRETS_INJECTION_PATTERN.md)
- Roboflow integration: [guides/ROBOFLOW_GUIDE.md](./guides/ROBOFLOW_GUIDE.md)
- Hybrid detector modes: [guides/HYBRID_DETECTOR_MODES.md](./guides/HYBRID_DETECTOR_MODES.md)

## 5) Engineering Standards

- Development guidelines: [DEVELOPMENT.md](./DEVELOPMENT.md)
- Active execution plan: [../WORKPLAN.md](../WORKPLAN.md)

---

## Documentation Lifecycle Policy

- Keep docs that describe current behavior and current architecture.
- Remove deprecated or duplicate docs instead of stacking alternatives.
- Prefer one canonical file per topic; link to it from here and README.
