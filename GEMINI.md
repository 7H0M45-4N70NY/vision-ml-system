# Vision ML System: Project Context

## Overview
A production-grade retail analytics pipeline for person detection and tracking.
- **Primary stack:** YOLOv11 (Ultralytics), ByteTrack, Supervision.
- **MLOps:** MLflow + DagsHub for experiment tracking, DVC for data versioning.
- **Analytics:** SQLite backend for dwell time and visitor count persistence.

## Core Components
- `InferencePipeline`: Orchestrates the flow from raw video to annotated output and analytics.
- `DualDetector`: Ensemble logic (YOLO primary + RF-DETR fallback). Supports 'hot', 'inline', and 'batch' modes.
- `AutoLabeler`: Collects low-confidence frames for active learning and Roboflow upload.
- `DriftDetector`: Monitors rolling average confidence to trigger retraining alerts.

## Key Design Patterns
- **Factory Pattern:** `DetectorFactory` and `TrackerFactory` for component decoupling.
- **Singleton Caching:** `ModelRegistry` ensures expensive models are loaded only once.
- **Universal Contract:** `supervision.Detections` is the standard object passed between all pipeline stages.

## Strategic Vision & Utility
This project is positioned as a **Production-Grade CV System Template**, focusing on the "flywheel" around the model rather than just the model itself.

### Key Value Propositions
1. **Developer Boilerplate:** A "Project-in-a-Box" for CV applications. Modular factories allow for forking and swapping models (e.g., swapping YOLO for MediaPipe) while keeping the MLOps/Analytics infrastructure intact.
2. **Privacy-First "Edge" Analytics:** Local-only processing (SQLite + YOLO) enables professional foot-traffic analytics for small businesses with 100% data privacy and zero cloud costs.
3. **Active Learning Showcase:** The `DualDetector` (YOLO+RF-DETR) solves the "failure detection" problem by automatically collecting and labeling hard-to-detect frames.
4. **Professional UI:** Transitioning from Streamlit to a "Google-inspired" Minimalist Web UI (Vanilla JS/CSS) to provide a high-performance, polished user experience.

## Future Roadmap
- **UI Transition:** Migrate from Streamlit to a custom Minimalist Web UI (Vanilla JS/CSS/HTML).
  - *Goal:* High-performance, "Google-inspired" aesthetic (airy, clean, fast).
  - *Strategy:* Replace heavy Python-backed pages with a focused frontend that consumes the existing FastAPI/AnalyticsDB backend.

## Data Ingestion (Remote/Cloud)
For Kaggle/Colab environments, use one of the following:
- **Direct Roboflow Sync:** `python scripts/prepare_data.py --source roboflow --roboflow-version N`
- **DVC Remote:** `dvc pull` (requires configured remote storage).
- **Small Test Runs:** Place ~5-10 frames in `data/low_confidence_frames/` and run `prepare_data.py --source local`.

## Execution Requirements
- **PYTHONPATH**: Must include the project root (`.`) for imports to resolve correctly.
  - *Cloud Tip:* Use `export PYTHONPATH=$PYTHONPATH:$(pwd)` in notebook cells.
- **Venv**: Use `.venv\Scripts\activate` (managed via `uv`).
- **Environment**: `ROBOFLOW_API_KEY` and `DAGSHUB_USERNAME` are required for full MLOps functionality.

## Data Structure
- `data/low_confidence_frames/`: Raw frames and pseudo-labels for active learning.
- `data/auto_labeled/`: Exported JSON labels for training preparation.
- `data/analytics.db`: SQLite database for all persistent metrics.
