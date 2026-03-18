# Vision ML System — Retail Analytics MVP

A production-grade vision system for retail analytics with controlled scope, strong MLOps backbone, and interview-ready architecture.

## 🎯 Project Goal

Build a **clean, engineered Retail Analytics MVP** that demonstrates:
- ✅ Modular detection + tracking pipeline
- ✅ Reproducible training with MLflow
- ✅ Dataset versioning with DVC
- ✅ Production-grade secrets management (3-layer architecture)
- ✅ Multi-source data ingestion (local + Roboflow)
- ✅ Label schema validation (fail-fast)
- 🔄 Drift simulation and monitoring
- 🔄 Performance benchmarking
- 🔄 Scaling analysis

**Not** building Amazon Go. Building a **portfolio-grade system** with 5–10 SKUs, controlled environment, and realistic MLOps practices.

## 📚 Documentation

### Core Architecture
- **[Strategic Vision](docs/architecture/STRATEGY.md)** — MVP scope, architecture decisions, why this approach
- **[System Design](docs/architecture/System%20Design.md)** — Detection, tracking, inference pipeline
- **[Repository Architecture](docs/architecture/REPO_ARCHITECTURE.md)** — Directory structure, module organization

### Setup & Configuration
- **[Quick Start](docs/getting-started/QUICKSTART.md)** — Get running in 5 minutes
- **[Run Setup (Env + Config)](docs/getting-started/RUN_SETUP_ENV_CONFIG.md)** — Environment variables and config mapping
- **[Secrets Injection Pattern](docs/guides/SECRETS_INJECTION_PATTERN.md)** — Production-grade secrets management (3-layer: YAML → ENV → Code)

### Data & Training
- **[Dataset Structure](docs/guides/DATASET_STRUCTURE.md)** — DVC layout, versioning strategy
- **[Training Pipeline](docs/guides/TRAINING_PIPELINE.md)** — MLflow config, hyperparameters, reproducibility
- **[DagsHub + MLflow + DVC Integration](docs/guides/DAGSHUB_MLFLOW_DVC_INTEGRATION.md)** — End-to-end MLOps setup

### Advanced Topics
- **[Hybrid Detector Modes](docs/guides/HYBRID_DETECTOR_MODES.md)** — Dual-model setup for active learning
- **[Scaling Analysis](docs/guides/SCALING.md)** — Performance benchmarks, bottleneck analysis
- **[Development Guide](docs/DEVELOPMENT.md)** — Contributing, testing, CI/CD

## 🚀 Quick Start

### 🐳 Docker (Recommended)

Run the full system (Streamlit Dashboard + API) in a container:

```bash
# Build and start
docker-compose up --build

# Access Dashboard: http://localhost:8501
# Access API Docs: http://localhost:8000/docs
```

### 🐍 Local Development

```bash
# Setup environment
conda activate ./venv
pip install -r requirements.txt

# Create env file from template
copy .env.example .env

# Run Inference API
uvicorn src.vision_ml.api.main:app --reload

# Run Dashboard (Streamlit)
streamlit run home.py
```

### 🧪 Testing & CI/CD

This project uses **GitHub Actions** for automated testing and linting.

```bash
# Run unit tests
pytest tests/

# Run linting
ruff check .
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

## 🔐 Secrets Management (Production-Ready)

The system uses a **3-layer secrets architecture** (YAML → ENV → Code):

1. **YAML Configuration** (`config/training/base.yaml`): Structure and parameters only
2. **Environment Variables** (`.env` or CI/CD): API keys and credentials
3. **Code Logic** (`src/vision_ml/utils/config.py`): Centralized injection via `inject_secrets()`

**Benefits:**
- ✅ Secrets never committed to git
- ✅ Same code for dev/staging/production
- ✅ Safe logging (no credential leaks)
- ✅ Works with Docker, K8s, CI/CD

See **[Secrets Injection Pattern](docs/guides/SECRETS_INJECTION_PATTERN.md)** for details.

## 📊 Data Ingestion Pipeline

Multi-source ingestion with validation:

```
Local Sources (auto-labeled + low-confidence)
    ↓
Roboflow (cloud dataset)
    ↓
Label Schema Validation (fail-fast)
    ↓
Priority-based Deduplication
    ↓
Train/Val Split
    ↓
YOLO-format Dataset
```

**Features:**
- ✅ Validates label structure before processing
- ✅ Ensures `source` key exists for all samples
- ✅ Type checking for boxes, class_ids, image_path
- ✅ Detailed error logging with context
- ✅ Graceful handling of malformed data

## 🚀 MLOps Integration (In Progress)

- **DagsHub**: Remote model registry + experiment tracking
- **MLflow**: Reproducible training runs with parameter logging
- **DVC**: Dataset versioning and pipeline reproducibility
- **GitHub Actions**: Automated testing and linting

See **[DagsHub + MLflow + DVC Integration](docs/guides/DAGSHUB_MLFLOW_DVC_INTEGRATION.md)**.

## � Running Data Preparation

```bash
# Prepare from local sources (auto-labeled + low-confidence frames)
python scripts/prepare_data.py --config config/training/base.yaml --source local

# Download from Roboflow (requires ROBOFLOW_API_KEY in .env)
python scripts/prepare_data.py --config config/training/base.yaml --source roboflow

# Combine both sources
python scripts/prepare_data.py --config config/training/base.yaml --source both
```

Output: `data/prepared/dataset.yaml` (YOLO format, ready for training)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_analytics.py -v

# Run with coverage
pytest tests/ --cov=src/vision_ml
```

## 📦 Dependencies

See `requirements.txt` for full list. Key packages:
- `ultralytics` — YOLO11 detection
- `supervision` — Detection/tracking utilities
- `mlflow` — Experiment tracking
- `dvc` — Data versioning
- `pydantic` — Config validation
- `fastapi` — API framework
- `streamlit` — Dashboard

## 🤝 Contributing

1. Create feature branch: `git checkout -b feat/your-feature`
2. Make changes and test: `pytest tests/`
3. Commit with clear message: `git commit -m "feat: description"`
4. Push and open PR

See **[Development Guide](docs/DEVELOPMENT.md)** for details.

## 📄 License

MIT License — See LICENSE file for details.