# 📑 Vision ML System Documentation

Welcome to the official documentation for the **Vision ML System**. This project is a production-grade template for retail analytics, featuring person detection, tracking, and a full MLOps lifecycle.

---

## 🚀 Getting Started
*Essential guides to get the system running on your machine.*

- **[Quickstart](./getting-started/QUICKSTART.md)**: 5-minute setup and your first inference run.
- **[Environment Setup](./getting-started/RUN_SETUP_ENV_CONFIG.md)**: Detailed guide on configuring `.env` and API keys.
- **[System Walkthrough](./getting-started/WALKTHROUGH.md)**: A guided tour of the core features and UI.

---

## 🏗️ Architecture & Strategy
*How the system is built and why we made these decisions.*

- **[System Strategy](./architecture/STRATEGY.md)**: The "Flywheel" philosophy and privacy-first design.
- **[Technical Architecture](./architecture/architecture.md)**: High-level component diagrams and data flow.
- **[Repository Structure](./architecture/REPO_ARCHITECTURE.md)**: Module organization and coding conventions.
- **[Master Design Plan](./architecture/System%20Design.md)**: The original blueprint for the vision system.

---

## 📘 Developer & ML Guides
*Deep dives into specific modules and workflows.*

### Data & Training
- **[Dataset Structure](./guides/DATASET_STRUCTURE.md)**: DVC, YOLO formats, and data versioning.
- **[Training Pipeline](./guides/TRAINING_PIPELINE.md)**: How to train, tune, and register models.
- **[Roboflow Integration](./guides/ROBOFLOW_GUIDE.md)**: Active learning and cloud-based labeling.

### MLOps & Optimization
- **[MLflow & DagsHub](./guides/DAGSHUB_MLFLOW_DVC_INTEGRATION.md)**: Remote experiment tracking and metrics.
- **[Model Quantization](./guides/MODEL_QUANTIZATION_GUIDE.md)**: Optimizing for edge devices (ONNX, FP16).
- **[Hybrid Detector Modes](./guides/HYBRID_DETECTOR_MODES.md)**: Using the YOLO + RF-DETR dual setup.
- **[Performance Scaling](./guides/scaling.md)**: Latency benchmarks and throughput analysis.

---

## 🛠️ API Reference
*Detailed technical specifications for internal and external APIs.*

- **[Core Package (src/vision_ml)](./api/README.md)**: Package-level documentation. (Coming Soon)
- **[REST API Specification](./api/api_spec.md)**: FastAPI endpoints and WebSocket protocols. (Coming Soon)

---

## 📂 Project Governance
- **[WORKPLAN.md](../WORKPLAN.md)**: The active roadmap and task backlog.
- **[AGENTS.md](../AGENTS.md)**: Guidelines for AI-assisted development (Jules).
- **[Archive](./archive/)**: Historical design docs and wave summaries.

---

> **Looking for the code?** The source is located in the `src/` directory. All CLI scripts are in `scripts/`.
