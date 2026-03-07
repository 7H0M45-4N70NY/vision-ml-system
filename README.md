# Vision ML System — Retail Analytics MVP

A production-grade vision system for retail analytics with controlled scope, strong MLOps backbone, and interview-ready architecture.

## 🎯 Project Goal

Build a **clean, engineered Retail Analytics MVP** that demonstrates:
- Modular detection + tracking pipeline
- Reproducible training with MLflow
- Dataset versioning with DVC
- Drift simulation and monitoring
- Performance benchmarking
- Scaling analysis

**Not** building Amazon Go. Building a **portfolio-grade system** with 5–10 SKUs, controlled environment, and realistic MLOps practices.

## 📚 Documentation

- **[Strategic Vision](docs/STRATEGY.md)** — MVP scope, architecture decisions, why this approach
- **[Repository Architecture](docs/REPO_ARCHITECTURE.md)** — Directory structure, module organization
- **[Run Setup (Env + Config)](docs/RUN_SETUP_ENV_CONFIG.md)** — complete env vars and config map for app/scripts
- **[Dataset Structure](docs/DATASET_STRUCTURE.md)** — DVC layout, versioning strategy
- **[Training Pipeline](docs/TRAINING_PIPELINE.md)** — MLflow config, hyperparameters, reproducibility
- **[System Architecture](docs/ARCHITECTURE.md)** — Detection, tracking, inference pipeline
- **[Scaling Analysis](docs/SCALING.md)** — Performance benchmarks, bottleneck analysis
- **[Roadmap](docs/ROADMAP.md)** — Phase 2+ features (drift detection, Airflow, monitoring)

## 🚀 Quick Start

```bash
# Setup environment
conda activate ./venv
pip install -r requirements.txt

# Create env file from template
copy .env.example .env

# Run training
python scripts/train.py --config config/training/base.yaml --trigger manual

# Run inference
python scripts/inference.py --mode offline --source path/to/video.mp4

# View MLflow experiments in terminal
python scripts/mlflow_cli.py --action experiments

# Launch Streamlit dashboard
streamlit run home.py
```

## 🏗 Architecture Overview

```
Input Video
    ↓
Frame Extraction
    ↓
Person Detection (YOLO26)
    ↓
Multi-Object Tracking (ByteTrack)
    ↓
Interaction Rule Engine
    ↓
Analytics Output
```

## 🔧 Tech Stack

- **Detection**: YOLO26 (person + product category)
- **Tracking**: ByteTrack
- **Training**: PyTorch + MLflow
- **Data Versioning**: DVC
- **Monitoring**: Prometheus + Grafana (Phase 2)
- **Orchestration**: Airflow (Phase 2)

## 📖 Learning Outcomes

By completion, you'll understand:
- Why small SKU sets are strategically chosen
- How drift is simulated and detected
- How retraining pipelines are triggered
- Model promotion and version rollback
- Production inference scaling
- MLOps best practices

This is **senior-level portfolio material**.
