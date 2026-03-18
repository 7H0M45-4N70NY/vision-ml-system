# VisionFlow — Retail Analytics System

A production-grade computer vision system for retail analytics. Combines real-time person detection and tracking with a full MLOps backbone: experiment tracking, dataset versioning, active learning, and a live web dashboard.

## Overview

VisionFlow ingests video from webcams, RTSP streams, or local files and outputs structured visitor analytics — dwell time, traffic patterns, detection confidence drift — all surfaced through a Next.js dashboard backed by a FastAPI inference server.

Key design principles:
- **Config-driven** — no hardcoded thresholds; everything lives in YAML
- **Pluggable detectors** — swap YOLO ↔ RF-DETR via a single config flag
- **Active learning loop** — low-confidence frames are auto-captured, triaged in the UI, and fed back into training
- **Privacy-first** — fully local, no cloud dependency for inference

## 🏗 Architecture Overview

```
Input Video
    ↓
Frame Extraction
    ↓
Person Detection (YOLO11n / RF-DETR)
    ↓
Multi-Object Tracking (ByteTrack)
    ↓
Interaction Rule Engine
    ↓
Visitor Analytics (dwell time, unique counts)
    ↓
Auto-Labeler + Drift Detector
    ↓
Dashboard (Next.js + FastAPI + SQLite)
```

## 🔧 Tech Stack

- **Detection**: YOLO11n + RF-DETR (dual detector mode)
- **Tracking**: ByteTrack
- **Training**: PyTorch + MLflow
- **Data Versioning**: DVC
- **Monitoring**: Prometheus + Grafana (Phase 2)
- **Orchestration**: Airflow (Phase 2)

## Quick Start

### Docker

```bash
docker-compose up --build
# Dashboard: http://localhost:3000
# API docs:  http://localhost:8000/docs
```

### Local Development

```bash
# Install dependencies (Python)
pip install -r requirements.txt

# Copy environment template and fill in credentials
cp .env.example .env

# Start the inference API
uvicorn src.vision_ml.api.main:app --reload

# Start the dashboard (separate terminal)
cd frontend && npm install && npm run dev
```

## Usage

```bash
# Offline inference on a video file
python scripts/inference.py --mode offline --source path/to/video.mp4

# Train with MLflow tracking
python scripts/train.py --config config/training/base.yaml --trigger manual

# Prepare training data (local + Roboflow)
python scripts/prepare_data.py --config config/training/base.yaml --source both

# Launch MLflow experiment UI
mlflow ui
```

## Documentation

| Topic | Link |
|---|---|
| System design | [docs/architecture/System Design.md](docs/architecture/System%20Design.md) |
| Repository layout | [docs/architecture/REPO_ARCHITECTURE.md](docs/architecture/REPO_ARCHITECTURE.md) |
| Quick start guide | [docs/getting-started/QUICKSTART.md](docs/getting-started/QUICKSTART.md) |
| Environment & config | [docs/getting-started/RUN_SETUP_ENV_CONFIG.md](docs/getting-started/RUN_SETUP_ENV_CONFIG.md) |
| Secrets management | [docs/guides/SECRETS_INJECTION_PATTERN.md](docs/guides/SECRETS_INJECTION_PATTERN.md) |
| Dataset structure | [docs/guides/DATASET_STRUCTURE.md](docs/guides/DATASET_STRUCTURE.md) |
| Training pipeline | [docs/guides/TRAINING_PIPELINE.md](docs/guides/TRAINING_PIPELINE.md) |
| MLOps integration | [docs/guides/DAGSHUB_MLFLOW_DVC_INTEGRATION.md](docs/guides/DAGSHUB_MLFLOW_DVC_INTEGRATION.md) |
| Dual detector modes | [docs/guides/HYBRID_DETECTOR_MODES.md](docs/guides/HYBRID_DETECTOR_MODES.md) |
| Scaling analysis | [docs/guides/SCALING.md](docs/guides/SCALING.md) |
| Development guide | [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) |

## Secrets Management

Three-layer secrets architecture keeps credentials out of version control:

1. **YAML** (`config/`) — structure and non-sensitive parameters only
2. **Environment variables** (`.env` / CI secrets) — API keys and credentials
3. **Code** (`src/vision_ml/utils/config.py`) — injection via `inject_secrets()`

Works with local `.env`, Docker environment, Kubernetes secrets, and CI/CD pipelines without code changes.

## Data Pipeline

```
Local auto-labeled frames
Local low-confidence captures
Roboflow cloud dataset
        ↓
  Label schema validation (fail-fast)
        ↓
  Priority-based deduplication
        ↓
  Train / val split
        ↓
  YOLO-format dataset  →  Training
```

## Testing

```bash
pytest tests/
pytest tests/test_analytics.py -v
pytest tests/ --cov=src/vision_ml
```

## Contributing

1. Branch: `git checkout -b feat/your-feature`
2. Test: `pytest tests/`
3. Commit: `git commit -m "feat: description"`
4. Open a pull request

## License

MIT — see [LICENSE](LICENSE) for details.
