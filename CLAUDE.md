# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Production-grade Retail Analytics MVP: person detection + tracking pipeline with MLOps backbone.
YOLO/RF-DETR detection → ByteTrack tracking → visitor analytics. Config-driven, MLflow-tracked, Streamlit dashboard.

## Strategic Vision & Utility
- **Production Template:** Position as a "forkable" CV system template for any detection task.
- **Privacy-First:** Target local, offline analytics for businesses (SQLite + YOLO) to avoid cloud costs/privacy concerns.
- **Active Learning Showcase:** Emphasize the `DualDetector` loop for automatic failure collection and self-improvement.
- **High-End UI:** Transition to a minimalist, "Google-life" aesthetic (Vanilla JS/CSS/HTML) for the production frontend.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run home.py

# Training
python scripts/train.py --config config/training/base.yaml --trigger manual

# Inference (offline)
python scripts/inference.py --mode offline --source path/to/video.mp4

# MLflow UI
mlflow ui

# Tests
pytest tests/
pytest tests/test_detection.py              # Single test file
pytest tests/ --cov=src/vision_ml           # With coverage
```

## Architecture

**Core pipeline flow:** Video → Frame Extraction → Detection → Tracking → Analytics

**Key patterns:**
- **Factory pattern** for detectors (`DetectorFactory.from_config(config)`) and trackers (`TrackerFactory.from_config(config)`). Models are singleton-cached via `ModelRegistry`; trackers are always new instances (stateful).
- **Base class interfaces:** `BaseDetector` and `BaseTracker` ABCs in their respective `base.py` files. All implementations follow these contracts.
- **DualDetector:** Ensemble mode — YOLO primary, RF-DETR fallback for low-confidence detections. Controlled via `detection.use_dual_detector` config flag.
- **InferencePipeline** (`src/vision_ml/inference/pipeline.py`) orchestrates: Detector → Tracker → Annotator → VisitorAnalytics → AutoLabeler → DriftDetector.

**Config system:** Two separate YAML config trees (`config/inference/` and `config/training/`) because inference is production-stable while training is experimental. Loaded via `src/vision_ml/utils/config.py` (`load_config`, `merge_configs`, `validate_config`). Design rule: never hardcode thresholds — everything goes in YAML.

**Data persistence:** `AnalyticsDB` uses SQLite (`data/analytics.db`) with tables for inference runs, visitor analytics, training events, and labeling events.

**Entry points:**
- `home.py` — Streamlit multi-page app (pages in `pages/`)
- `scripts/*.py` — CLI entry points for training, inference, MLflow, DVC, analytics
- `main.py` — subprocess launcher for Streamlit

## Source Layout

```
src/vision_ml/
├── detection/          # BaseDetector, YOLODetector, RFDETRDetector, DualDetector, DetectorFactory, ModelRegistry
├── tracking/           # BaseTracker, ByteTrack, TrackerFactory
├── inference/          # InferencePipeline (main orchestrator)
├── training/           # Trainer (MLflow callbacks), DriftDetector
├── analytics/          # AnalyticsDB (SQLite), VisitorAnalytics (dwell time)
├── annotation/         # Supervision-based frame annotator
├── labeling/           # AutoLabeler (pseudo-label collection)
├── utils/              # Config loading/merging/validation
└── mlflow_integration.py
```

## Conventions

- Python >=3.13, managed with uv (lock file: `uv.lock`)
- All detection outputs use `supervision.Detections` format
- Config keys use snake_case; detector types match model names (e.g., `yolo11n`, `rfdetr`)
- Environment variables from `.env` (copy `.env.example`): DagsHub, Roboflow, MLflow settings
