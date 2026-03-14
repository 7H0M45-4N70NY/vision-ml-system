# Vision ML System — Project Workplan

> A contributor-friendly breakdown of all remaining work.
> Each task is scoped for **one person or one AI agent to complete independently**.
>
> **AI Agent Compatible:** Tasks marked with `jules` are optimized for
> [Google Jules](https://jules.google.com/) — add the `jules` label to the
> GitHub issue and Jules will pick it up, plan the work, and open a PR.

---

## Project Status Overview

| Component | Completion | Status |
|-----------|-----------|--------|
| Core Inference Pipeline | 95% | Production-ready |
| Detection (YOLO/RF-DETR) | 90% | Fully functional |
| Tracking (ByteTrack) | 85% | Working with version compat |
| Visitor Analytics | 90% | Dwell time, counts, persistence |
| Drift Detection | 80% | Model drift works; data drift missing |
| Analytics DB (SQLite) | 90% | Schema + queries working |
| Training Pipeline | 60% | Architecture real, data missing |
| MLflow Integration | 70% | Callback works, remote needs setup |
| Dashboard (Streamlit) | 60% | UI exists, many features display-only |
| FastAPI Endpoints | 50% | Health + video predict work |
| DVC Pipeline | 20% | Defined but no data, no remote |
| CI/CD | 30% | Basic lint + test, no deploy |
| Events System | 10% | Broken — base.py missing |
| Test Coverage | 30% | Basic scaffolding, integration tests missing |

---

## How to Contribute

### Human Contributors

1. Pick a task from any workstream below
2. Check dependencies — make sure prerequisite tasks are done
3. Create a branch: `feature/<workstream>/<task-short-name>`
4. Follow conventions: Config in YAML, detections as `supervision.Detections`, logging via `vision_ml.logging`
5. Write tests for your changes
6. Submit PR referencing the task ID (e.g., `Closes #12`)

### Using Jules (AI Agent)

1. Create a GitHub issue using the task description below (copy the full task block)
2. Add the label **`jules`** to the issue
3. Jules will clone the repo, read `AGENTS.md`, plan the work, and open a PR
4. Review the PR, request changes if needed, merge when ready

> **Which tasks are Jules-friendly?** Look for the `jules` tag in the task header.
> Tasks requiring external credentials, UI testing, or hardware (GPU) are marked `human-only`.

### Difficulty & Size Guide

| Difficulty | Who | Size | Effort |
|-----------|-----|------|--------|
| `beginner` | First-time contributors / Jules | `S` | 1-3 hours |
| `intermediate` | Python + CV experience / Jules | `M` | 3-8 hours |
| `advanced` | Deep ML/systems experience | `L` | 8-16 hours |
| | | `XL` | 16+ hours |

---

## Workstream 1: Critical Fixes

> Unblock everything else. Do these first.

---

### FIX-01: Create `events/base.py` `jules` `beginner` `S` `P0-critical`

**Problem:** `src/vision_ml/events/in_memory.py` imports `Event`, `EventType`,
`Job`, `JobState` from a `base` module that does not exist. All imports fail.
CI will break.

**Files to create:**
- `src/vision_ml/events/base.py`
- `src/vision_ml/events/__init__.py`

**Files to read for context:**
- `src/vision_ml/events/in_memory.py` (see what classes it imports and extends)

**Requirements:**
- Define `EventType` as an enum with values: `INFERENCE_COMPLETE`, `DRIFT_DETECTED`, `TRAINING_TRIGGERED`, `LABELING_COMPLETE`
- Define `Event` dataclass with fields: `event_type: EventType`, `payload: dict`, `timestamp: datetime`
- Define `JobState` as an enum: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`
- Define `Job` dataclass with fields: `job_id: str`, `state: JobState`, `task: str`, `created_at: datetime`, `result: dict | None`
- Define abstract base classes: `EventPublisher`, `EventSubscriber`, `JobQueue`
- Ensure `in_memory.py` imports resolve without errors

**Verify:**
```bash
python -c "from src.vision_ml.events.base import Event, EventType, Job, JobState; print('OK')"
python -c "from src.vision_ml.events.in_memory import InMemoryEventPublisher; print('OK')"
pytest tests/ -x --tb=short
```

**Depends on:** Nothing

---

### FIX-02: Fix duplicate log output `jules` `beginner` `S`

**Problem:** Every log message prints twice. The root logger in
`src/vision_ml/logging/logger.py` has its own handler, and child loggers
propagate to it, causing duplication.

**Files to modify:**
- `src/vision_ml/logging/logger.py`

**Fix:** Remove the root logger setup block at the bottom of the file
(lines 112-123). Child loggers already have `propagate = False` and their
own handlers.

**Verify:**
```bash
python -c "
from vision_ml.logging import get_logger
logger = get_logger('test')
logger.info('This should print exactly once')
"
```

**Depends on:** Nothing

---

### FIX-03: Add `__init__.py` to all packages `jules` `beginner` `S`

**Problem:** `src/vision_ml/api/` and `src/vision_ml/events/` may be missing
`__init__.py` files, breaking package imports.

**Files to check/create:**
- `src/vision_ml/api/__init__.py`
- `src/vision_ml/events/__init__.py`

**Requirements:**
- `api/__init__.py` should export the FastAPI `app` from `main.py`
- `events/__init__.py` should export base classes and in-memory implementations

**Verify:**
```bash
python -c "from src.vision_ml.api import app; print('API OK')"
python -c "from src.vision_ml.events import InMemoryEventPublisher; print('Events OK')"
```

**Depends on:** FIX-01

---

### FIX-04: Resolve import path inconsistency `jules` `intermediate` `M`

**Problem:** Two conflicting import styles exist in the codebase:
- Streamlit pages: `from src.vision_ml.labeling.auto_labeler import AutoLabeler`
- CLI scripts: `from vision_ml.labeling.auto_labeler import AutoLabeler`
- Inside packages: mixed `from ..logging import get_logger` and absolute imports

**Files to modify:**
- All files under `src/vision_ml/` — use **relative imports** (e.g., `from ..logging import get_logger`)
- All files under `pages/` — use `from src.vision_ml.X` (absolute with `src.` prefix)
- All files under `scripts/` — use `sys.path.insert` + `from vision_ml.X`

**Requirements:**
- Every `src/vision_ml/**/*.py` file must use relative imports for sibling modules
- Document the import convention in a comment block at the top of `src/vision_ml/__init__.py`

**Verify:**
```bash
python scripts/inference.py --help
python scripts/auto_labeling_cli.py --help
python -c "from src.vision_ml.inference.pipeline import InferencePipeline; print('OK')"
pytest tests/ -x --tb=short
```

**Depends on:** Nothing

---

## Workstream 2: Data Pipeline

> No training, evaluation, or DVC without real data flowing through.

---

### DATA-01: End-to-end inference run with DB persistence `human-only` `beginner` `M`

**Problem:** `analytics.db` has 0 inference runs. The pipeline works but
results may not be persisted to the database when run via CLI.

**Steps:**
1. Obtain or generate a short test video (5-10 seconds, any people visible)
2. Run: `python scripts/inference.py --config config/inference/base.yaml --mode offline --source <video_path>`
3. Verify DB populated:
   ```bash
   python -c "
   import sqlite3
   conn = sqlite3.connect('data/analytics.db')
   c = conn.cursor()
   c.execute('SELECT COUNT(*) FROM inference_runs')
   print('Inference runs:', c.fetchone()[0])
   c.execute('SELECT COUNT(*) FROM visitor_analytics')
   print('Visitors:', c.fetchone()[0])
   "
   ```
4. If counts are 0, trace the save path in `scripts/inference.py` and
   `src/vision_ml/inference/pipeline.py` to find where DB write is missing.

**Why human-only:** Requires a real video file and visual verification.

**Depends on:** Nothing

---

### DATA-02: Validate Roboflow upload flow `human-only` `intermediate` `M`

**Problem:** Roboflow upload falls back to local export. Need to verify
credentials, workspace name, and upload API call.

**Steps:**
1. Set `ROBOFLOW_API_KEY` in `.env`
2. Verify workspace name matches Roboflow account URL slug
3. Run: `python scripts/auto_labeling_cli.py --config config/training/base.yaml --provider roboflow`
4. Check Roboflow UI for uploaded images

**Files to check:**
- `config/inference/base.yaml` — `labeling.roboflow_workspace` value
- `config/training/base.yaml` — `labeling.roboflow_workspace` value
- `src/vision_ml/labeling/auto_labeler.py` — `_upload_roboflow()` method

**Why human-only:** Requires Roboflow account credentials and UI verification.

**Depends on:** Nothing

---

### DATA-03: Wire DVC remote storage `human-only` `intermediate` `M`

**Problem:** `dvc.yaml` defines stages but no remote is configured.
`dvc push` / `dvc pull` will fail.

**Steps:**
1. Choose a remote: DagsHub (recommended, already have account), S3, or GCS
2. Run: `dvc remote add -d storage <remote-url>`
3. Run: `dvc push` to verify upload
4. Run: `dvc pull` in a clean checkout to verify download
5. Add setup instructions to README

**Files to modify:**
- `.dvc/config` (created by `dvc remote add`)
- `README.md` (add DVC setup section)

**Why human-only:** Requires cloud storage credentials.

**Depends on:** DATA-01

---

### DATA-04: Build sample dataset from inference `human-only` `intermediate` `L`

**Problem:** No labeled dataset exists for training. Need to generate one
from inference runs.

**Steps:**
1. Run inference on 3-5 short videos with `use_dual_detector: batch`
2. Verify `data/low_confidence_frames/` has 50-100 frames
3. Run: `python scripts/auto_labeling_cli.py --provider local`
4. Run: `python scripts/prepare_data.py --source local`
5. Verify `data/prepared/` has `dataset.yaml`, `train/`, `val/` in YOLO format

**Why human-only:** Requires video files and manual quality verification.

**Depends on:** DATA-01, DATA-02

---

### DATA-05: Validate `prepare_data.py` end-to-end `jules` `intermediate` `M`

**Problem:** `scripts/prepare_data.py` may have bugs in the data splitting
or YOLO format conversion logic.

**Files to read:**
- `scripts/prepare_data.py`
- `data/auto_labeled/auto_labels.json` (sample label format)

**Requirements:**
- Verify `--source local` reads from `data/auto_labeled/auto_labels.json`
- Verify it creates a valid YOLO `dataset.yaml` with paths to train/val splits
- Verify image files are copied to correct split directories
- Verify YOLO `.txt` annotation files are generated with normalized coordinates
- Add error handling for missing source files
- Add a `--dry-run` flag that prints what would be done without writing files

**Verify:**
```bash
python scripts/prepare_data.py --source local --dry-run
```

**Depends on:** DATA-04

---

## Workstream 3: Training & MLOps

> Make the training loop reproducible and tracked.

---

### TRAIN-01: Run training on coco8 (smoke test) `human-only` `beginner` `M`

**Problem:** Training pipeline has never been verified end-to-end.

**Steps:**
1. Run: `python scripts/train.py --config config/training/base.yaml --trigger manual`
2. Verify MLflow logs: `mlflow ui` → check experiment `visitor_analytics_training`
3. Verify model artifact saved under `runs/train/`

**Why human-only:** Requires MLflow UI verification and potentially GPU.

**Depends on:** Nothing

---

### TRAIN-02: Validate DagsHub MLflow remote tracking `human-only` `intermediate` `M`

**Problem:** DagsHub credentials in `.env` but remote tracking untested.

**Steps:**
1. Verify `.env` has `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `MLFLOW_TRACKING_URI`
2. Run training (TRAIN-01)
3. Check DagsHub UI for experiment run

**Why human-only:** Requires DagsHub account and UI verification.

**Depends on:** TRAIN-01

---

### TRAIN-03: Run full DVC pipeline `human-only` `intermediate` `M`

**Steps:**
1. Run: `dvc repro`
2. Verify all three stages complete: `prepare_data` → `train` → `evaluate`
3. Commit `dvc.lock`

**Why human-only:** Requires data and remote storage.

**Depends on:** DATA-05, TRAIN-01

---

### TRAIN-04: Implement data drift detection `jules` `advanced` `L`

**Problem:** `DriftDetector` only implements `confidence_drop` method.
`data_distribution` method is mentioned in docstrings but not implemented.

**Files to modify:**
- `src/vision_ml/training/drift_detector.py`
- `config/inference/base.yaml` (add config options)
- `tests/test_drift.py` (add tests)

**Requirements:**
- Add `data_distribution` method to `DriftDetector`
- Compare feature distributions between current window and a baseline:
  - Mean number of detections per frame
  - Mean bounding box area (normalized)
  - Detection count variance
- Use Kolmogorov-Smirnov test or simple threshold on distribution shift
- Add config key `drift.method: confidence_drop | data_distribution | both`
- Write at least 3 unit tests covering: normal data, drifted data, empty window
- Use the centralized logger: `from ..logging import get_logger`

**Verify:**
```bash
pytest tests/test_drift.py -v
```

**Depends on:** DATA-04

---

### TRAIN-05: Automated retraining trigger `jules` `advanced` `L`

**Problem:** When drift is detected, nothing happens automatically.
Need to wire `DriftDetector` → events → `Trainer`.

**Files to modify:**
- `src/vision_ml/inference/pipeline.py` (publish event on drift)
- `src/vision_ml/events/base.py` (use `DRIFT_DETECTED` event type)
- `src/vision_ml/training/trainer.py` (subscribe to drift events)

**Requirements:**
- When `DriftDetector.check()` returns `drift_detected=True`, publish a `DRIFT_DETECTED` event
- Create a `RetrainingSubscriber` that listens for drift events
- Subscriber should log the event and optionally trigger training via subprocess
- Add config key `drift.auto_retrain: true | false` (default false)
- Write integration test: simulate drift → verify event published

**Verify:**
```bash
pytest tests/test_drift.py -v
python -c "
from src.vision_ml.events.base import EventType
from src.vision_ml.events.in_memory import InMemoryEventPublisher
pub = InMemoryEventPublisher()
print('Event system OK')
"
```

**Depends on:** FIX-01, TRAIN-01

---

### TRAIN-06: Model versioning and promotion `jules` `advanced` `L`

**Problem:** No automated way to compare model versions and promote the best one.

**Files to modify:**
- `scripts/model_registry_cli.py`
- `scripts/evaluate.py`

**Requirements:**
- Add `compare` subcommand: loads two model checkpoints, runs evaluation on validation set, prints metrics comparison table
- Add `promote` subcommand: copies best model to a `models/production/` directory, updates a `models/production/metadata.json` with version info
- Use MLflow model registry if available, fallback to local file-based registry

**Verify:**
```bash
python scripts/model_registry_cli.py compare --help
python scripts/model_registry_cli.py promote --help
```

**Depends on:** TRAIN-03

---

## Workstream 4: Testing & Quality

> Expand from ~30% to 80%+ coverage.

---

### TEST-01: Unit tests for `VisitorAnalytics` `jules` `beginner` `M`

**Files to read:**
- `src/vision_ml/analytics/visitor_analytics.py`

**Files to create:**
- `tests/test_visitor_analytics.py`

**Requirements:**
- Test `update()` with synthetic tracker IDs across multiple frames
- Test `get_summary()` returns correct unique visitor count
- Test dwell time calculation: visitor present for 30 frames at 30fps = 1.0 second
- Test peak visitor count (max simultaneous visitors)
- Test empty input (no detections)
- Use `conftest.py` fixtures if helpful

**Verify:**
```bash
pytest tests/test_visitor_analytics.py -v
```

**Depends on:** Nothing

---

### TEST-02: Unit tests for `DualDetector` `jules` `intermediate` `M`

**Files to read:**
- `src/vision_ml/detection/dual_detector.py`
- `src/vision_ml/detection/base.py`

**Files to create:**
- `tests/test_dual_detector.py`

**Requirements:**
- Mock both primary and secondary detectors (do not load real models)
- Test hot path: only primary detector called
- Test inline mode: secondary called when primary confidence below threshold
- Test batch mode: low-confidence frames saved to disk
- Test frame-saving: verify `.jpg` and `.json` files written correctly
- Use `tmp_path` pytest fixture for file I/O tests

**Verify:**
```bash
pytest tests/test_dual_detector.py -v
```

**Depends on:** Nothing

---

### TEST-03: Unit tests for config system `jules` `beginner` `S`

**Files to read:**
- `src/vision_ml/utils/config.py`

**Files to create:**
- `tests/test_config.py`

**Requirements:**
- Test `load_config()` with a valid YAML file
- Test `load_config()` with a missing file (should raise or return default)
- Test `merge_configs()`: overlay values override base values
- Test `merge_configs()`: nested dict merging works correctly
- Test `validate_config()`: missing required keys flagged
- Use `tmp_path` to create temporary YAML files

**Verify:**
```bash
pytest tests/test_config.py -v
```

**Depends on:** Nothing

---

### TEST-04: Integration test — full inference pipeline `jules` `intermediate` `L`

**Files to create:**
- `tests/test_integration_pipeline.py`

**Requirements:**
- Create a synthetic 10-frame video using OpenCV (solid color + rectangles as fake people)
- Run through `InferencePipeline` with a test config
- Assert: detections returned are `supervision.Detections` objects
- Assert: analytics summary has `total_frames == 10`
- Assert: no crashes on empty frames (no detections)
- Mock the detector if loading real YOLO weights is too slow
- Use `tmp_path` for output directory

**Verify:**
```bash
pytest tests/test_integration_pipeline.py -v
```

**Depends on:** DATA-01

---

### TEST-05: Integration test — auto-labeling pipeline `jules` `intermediate` `M`

**Files to create:**
- `tests/test_auto_labeling.py`

**Requirements:**
- Create 3 fake low-confidence frame files (`.jpg` + `.json`) in `tmp_path`
- Load via `AutoLabeler.load_dual_detector_frames()`
- Export locally via `flush(output_dir=tmp_path / 'output')`
- Verify `auto_labels.json` contains 3 entries
- Verify each entry has `image_id`, `image_path`, `boxes`, `confidences`, `class_ids`
- Verify YOLO annotation format via `_write_yolo_annotation()`: normalized coordinates, correct class IDs

**Verify:**
```bash
pytest tests/test_auto_labeling.py -v
```

**Depends on:** Nothing

---

### TEST-06: API endpoint tests `jules` `intermediate` `M`

**Files to read:**
- `src/vision_ml/api/main.py`
- `src/vision_ml/api/schemas.py`

**Files to modify:**
- `tests/test_api.py`

**Requirements:**
- Use `fastapi.testclient.TestClient`
- Test `GET /health` returns `{"status": "healthy"}`
- Test `POST /predict/video` with a small synthetic video file
- Test `POST /predict/video` with invalid file returns 4xx error
- Mock the `InferencePipeline` to avoid loading real model weights

**Verify:**
```bash
pytest tests/test_api.py -v
```

**Depends on:** FIX-01

---

### TEST-07: Add pytest-cov to CI `jules` `beginner` `S`

**Files to modify:**
- `.github/workflows/ci.yaml`
- `README.md` (add coverage badge)

**Requirements:**
- Add `pytest-cov` to test dependencies
- Update CI step: `pytest tests/ --cov=src/vision_ml --cov-report=xml --cov-report=term-missing`
- Add coverage badge to README using shields.io or codecov

**Verify:**
```bash
pytest tests/ --cov=src/vision_ml --cov-report=term-missing
```

**Depends on:** Nothing

---

### TEST-08: Linting and type checking `jules` `intermediate` `L`

**Files to modify:**
- `.github/workflows/ci.yaml` (add mypy step)
- `pyproject.toml` (add mypy config)

**Requirements:**
- Add `mypy` to dev dependencies
- Configure mypy in `pyproject.toml` with `--ignore-missing-imports`
- Run mypy on `src/vision_ml/` — fix critical type errors only
- Add mypy step to CI pipeline after tests
- Do not add type annotations to code you did not change

**Verify:**
```bash
mypy src/vision_ml/ --ignore-missing-imports
```

**Depends on:** Nothing

---

## Workstream 5: API & Backend

> Build the REST interface for production deployment.

---

### API-01: Implement `/predict/stream` WebSocket endpoint `jules` `advanced` `L`

**Files to modify:**
- `src/vision_ml/api/main.py`

**Requirements:**
- Add WebSocket endpoint at `/predict/stream`
- Accept binary frame data (JPEG bytes)
- Run detection on each frame using `InferencePipeline.process_frame()`
- Return JSON: `{"detections": [...], "frame_idx": N, "tracker_ids": [...]}`
- Handle connection close gracefully
- Add connection timeout (30 seconds idle)

**Verify:**
```bash
python -c "
import asyncio, websockets, json
async def test():
    async with websockets.connect('ws://localhost:8000/predict/stream') as ws:
        print('Connected')
asyncio.run(test())
"
```

**Depends on:** Nothing

---

### API-02: Add `/models` endpoint `jules` `intermediate` `S`

**Files to modify:**
- `src/vision_ml/api/main.py`
- `src/vision_ml/api/schemas.py`

**Requirements:**
- `GET /models` returns list of available models
- Each model entry: `{"name": "yolo11n", "type": "yolo", "precision": "fp32", "cached": true}`
- Read from `ModelRegistry` singleton
- Add response schema to `schemas.py`

**Verify:**
```bash
curl http://localhost:8000/models
```

**Depends on:** Nothing

---

### API-03: Add `/config` endpoint `jules` `intermediate` `M`

**Files to modify:**
- `src/vision_ml/api/main.py`
- `src/vision_ml/api/schemas.py`

**Requirements:**
- `GET /config` returns current inference config as JSON
- `PUT /config` accepts partial config update, validates, merges with current
- Use `validate_config()` from `src/vision_ml/utils/config.py`
- Return 400 on invalid config keys
- Do NOT restart pipeline on config change (just update in memory)

**Verify:**
```bash
curl http://localhost:8000/config
curl -X PUT http://localhost:8000/config -H "Content-Type: application/json" -d '{"inference": {"confidence_threshold": 0.5}}'
```

**Depends on:** Nothing

---

### API-04: Add `/analytics` endpoints `jules` `beginner` `M`

**Files to modify:**
- `src/vision_ml/api/main.py`
- `src/vision_ml/api/schemas.py`

**Requirements:**
- `GET /analytics/summary` — returns `AnalyticsDB.get_analytics_summary()`
- `GET /analytics/runs` — returns `AnalyticsDB.get_inference_runs(limit=50)`
- `GET /analytics/visitors/{run_id}` — returns `AnalyticsDB.get_visitor_analytics(run_id)`
- Add Pydantic response schemas
- Return 404 if run_id not found

**Verify:**
```bash
curl http://localhost:8000/analytics/summary
curl http://localhost:8000/analytics/runs
```

**Depends on:** Nothing

---

### API-05: Add `/drift` endpoint `jules` `intermediate` `S`

**Files to modify:**
- `src/vision_ml/api/main.py`

**Requirements:**
- `GET /drift` returns current drift metrics from `DriftDetector.get_metrics()`
- Include `drift_detected`, `avg_confidence`, `drift_score`
- Include recommendation: `"action": "retrain"` if drift detected, `"action": "none"` otherwise

**Verify:**
```bash
curl http://localhost:8000/drift
```

**Depends on:** Nothing

---

### API-06: API authentication `jules` `intermediate` `M`

**Files to modify:**
- `src/vision_ml/api/main.py`

**Requirements:**
- Add middleware that checks `X-API-Key` header
- Valid keys stored in `API_KEYS` environment variable (comma-separated)
- Return 401 with `{"detail": "Invalid API key"}` on failure
- Exempt `/health` endpoint from auth
- Load keys via `os.environ.get('API_KEYS', '').split(',')`

**Verify:**
```bash
curl http://localhost:8000/health  # Should work without key
curl http://localhost:8000/models  # Should return 401
curl -H "X-API-Key: test123" http://localhost:8000/models  # Should work
```

**Depends on:** Nothing

---

### API-07: Dockerize the API `jules` `intermediate` `M`

**Files to create:**
- `Dockerfile`
- `docker-compose.yml`

**Requirements:**
- Multi-stage build: `python:3.13-slim` base
- Install only production dependencies
- Expose port 8000
- `CMD ["uvicorn", "src.vision_ml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]`
- `docker-compose.yml` with API service and optional volume mount for `data/`
- Add `.dockerignore` to exclude `.git`, `__pycache__`, `.env`, `data/`, `runs/`

**Verify:**
```bash
docker build -t vision-ml-api .
docker run -p 8000:8000 vision-ml-api
curl http://localhost:8000/health
```

**Depends on:** Nothing

---

## Workstream 6: Dashboard & UI

> Current: Streamlit. Future: Vanilla JS/CSS/HTML (Google-inspired).

### Phase 1: Fix Current Streamlit Dashboard

---

### UI-01: Wire training page to actual training `jules` `intermediate` `M`

**Files to modify:**
- `pages/4_training.py`

**Files to read for context:**
- `scripts/train.py` (see how training is triggered)
- `src/vision_ml/training/trainer.py`

**Requirements:**
- Add a "Start Training" button that runs `scripts/train.py` via `subprocess.Popen`
- Display training progress (epoch, loss) by reading stdout
- Show success/failure status
- Save training event to `AnalyticsDB`
- Do not block the Streamlit UI during training

**Depends on:** TRAIN-01

---

### UI-02: Wire MLflow experiments page `jules` `intermediate` `M`

**Files to modify:**
- `pages/5_mlflow_experiments.py`

**Requirements:**
- Connect to MLflow tracking server using `mlflow.MlflowClient()`
- List experiments and runs with metrics
- Display comparison chart (loss curves across runs)
- Handle connection errors gracefully (show message if MLflow unavailable)

**Depends on:** TRAIN-02

---

### UI-03: Wire model registry page `jules` `intermediate` `M`

**Files to modify:**
- `pages/6_model_registry.py`

**Requirements:**
- List registered models from MLflow or local `runs/train/` directory
- Show model metadata: name, version, metrics, creation date
- Add "Promote to Production" button
- Handle missing models gracefully

**Depends on:** TRAIN-06

---

### UI-04: Add drift visualization `jules` `intermediate` `M`

**Files to modify:**
- `pages/3_analytics.py` (add drift section) or create `pages/8_drift.py`

**Requirements:**
- Query `inference_runs` table for `avg_confidence` and `drift_score` columns
- Plot confidence trend over time (line chart)
- Plot drift score over time
- Show alert banner if latest drift_score indicates drift
- Use Plotly if available, fallback to `st.line_chart`

**Depends on:** DATA-01

---

### UI-05: Fix training pipeline page `jules` `advanced` `M`

**Files to modify:**
- `pages/7_training_pipeline.py`

**Requirements:**
- Display job queue status from events system
- Show running/completed/failed jobs
- Add "Cancel Job" button
- Wire to `InMemoryJobQueue` from events system

**Depends on:** FIX-01

---

### Phase 2: Vanilla JS/CSS/HTML Migration

---

### UI-10: Design system & component library `human-only` `intermediate` `L`

**Goal:** Define the "Google-inspired" minimalist design.

**Deliverables:**
- `frontend/css/design-tokens.css` — colors, typography, spacing, shadows
- `frontend/css/components.css` — cards, metric tiles, buttons, tables, nav
- Design reference document or Figma link

**Depends on:** Nothing

---

### UI-11: Static dashboard shell `jules` `intermediate` `M`

**Files to create:**
- `frontend/index.html`
- `frontend/css/style.css`
- `frontend/js/app.js`

**Requirements:**
- Responsive sidebar navigation (Dashboard, Inference, Training, Analytics)
- Header with project name
- Main content area with placeholder cards
- Clean, minimal aesthetic — no heavy frameworks
- Mobile-friendly

**Depends on:** UI-10

---

### UI-12: Analytics dashboard (JS) `jules` `intermediate` `L`

**Files to modify:**
- `frontend/js/analytics.js`

**Requirements:**
- Fetch data from `GET /analytics/summary` and `GET /analytics/runs`
- Render metric cards: total runs, total visitors, avg dwell time
- Render inference runs table
- Add Chart.js line chart for runs over time
- Handle API errors gracefully

**Depends on:** API-04, UI-11

---

### UI-13: Inference page (JS) `jules` `intermediate` `L`

**Files to modify:**
- `frontend/js/inference.js`

**Requirements:**
- File upload form (drag-and-drop + file picker)
- POST to `/predict/video`
- Display results: visitor count, dwell time, annotated video (if available)
- Show loading spinner during processing

**Depends on:** UI-11

---

### UI-14: Live stream page (JS) `human-only` `advanced` `XL`

**Requirements:**
- WebSocket connection to `/predict/stream`
- Capture frames from webcam via `getUserMedia()`
- Send frames to server, receive annotated results
- Render on `<canvas>` element
- Show live stats overlay

**Why human-only:** Requires webcam testing and real-time visual verification.

**Depends on:** API-01, UI-11

---

### UI-15: Training & MLflow pages (JS) `jules` `advanced` `L`

**Requirements:**
- Trigger training via `POST /train` endpoint (needs new API endpoint)
- Display experiment list from MLflow
- Compare runs with charts
- Model promotion UI

**Depends on:** API-03, UI-11

---

## Workstream 7: DevOps & Infrastructure

---

### OPS-01: Fix CI pipeline `jules` `beginner` `M`

**Files to modify:**
- `.github/workflows/ci.yaml`

**Requirements:**
- Ensure all imports resolve (depends on FIX-01)
- Add test data fixtures if needed (small JSON files)
- Verify `pytest tests/` passes
- Add step to install project: `pip install -e .` or `pip install -r requirements.txt`

**Verify:**
```bash
pytest tests/ -x --tb=short
```

**Depends on:** FIX-01

---

### OPS-02: Add Docker support `jules` `intermediate` `M`

Same as API-07. See that task for details.

**Depends on:** Nothing

---

### OPS-03: Add pre-commit hooks `jules` `beginner` `S`

**Files to create:**
- `.pre-commit-config.yaml`

**Requirements:**
- Ruff linting (match CI config)
- Ruff formatting
- Trailing whitespace removal
- End-of-file fixer
- YAML syntax check
- Do NOT add mypy (too slow for pre-commit)

**Verify:**
```bash
pip install pre-commit
pre-commit run --all-files
```

**Depends on:** Nothing

---

### OPS-04: GitHub Actions — build + test + coverage `jules` `intermediate` `M`

**Files to modify:**
- `.github/workflows/ci.yaml`

**Requirements:**
- Run tests with coverage: `pytest --cov=src/vision_ml --cov-report=xml`
- Upload coverage report as artifact
- Fail if coverage below 40%
- Add comment on PR with coverage summary (use codecov or similar)

**Depends on:** TEST-07

---

### OPS-05: GitHub Actions — Docker build + push `jules` `intermediate` `M`

**Files to create:**
- `.github/workflows/docker.yaml`

**Requirements:**
- Trigger on tag push (`v*`)
- Build Docker image
- Push to GitHub Container Registry (ghcr.io)
- Tag with version and `latest`

**Depends on:** OPS-02

---

### OPS-06: Add health monitoring `jules` `advanced` `L`

**Files to modify:**
- `src/vision_ml/api/main.py`

**Requirements:**
- Add `/metrics` endpoint returning Prometheus-format metrics
- Track: inference request count, latency histogram, detection count, memory usage
- Use `prometheus-client` library
- Add Grafana dashboard JSON (optional)

**Depends on:** API-01

---

## Workstream 8: Documentation & Benchmarks

---

### DOC-01: System design document `human-only` `advanced` `L`

**Deliverable:** `docs/SYSTEM_DESIGN.md`

**Requirements:**
- Architecture diagram (ASCII or Mermaid)
- Data flow: video → detection → tracking → analytics → storage
- Component interaction diagram
- Scaling strategy: 1 stream → 10 streams → 1000 streams
- Technology choices and trade-offs
- Target audience: interview panel

**Why human-only:** Requires architectural judgment and interview framing.

**Depends on:** Nothing

---

### DOC-02: Benchmarking — latency & throughput `human-only` `intermediate` `L`

**Deliverable:** `docs/BENCHMARKS.md`

**Requirements:**
- Measure FPS for: YOLO fp32 (CPU), YOLO fp32 (GPU), YOLO fp16 (GPU)
- Measure memory usage for each
- Test with 720p and 1080p input
- Create comparison table and charts
- Include hardware specs

**Why human-only:** Requires specific hardware and controlled measurement.

**Depends on:** Nothing

---

### DOC-03: Quantization analysis `human-only` `advanced` `L`

**Deliverable:** `docs/QUANTIZATION.md`

**Requirements:**
- Compare fp32 vs fp16 vs INT8 on mAP and FPS
- Document accuracy-speed trade-off curve
- Include export commands and runtime requirements

**Why human-only:** Requires GPU and careful measurement.

**Depends on:** DOC-02

---

### DOC-04: CONTRIBUTING.md `jules` `beginner` `M`

**Files to create:**
- `CONTRIBUTING.md`

**Requirements:**
- Development setup (uv, .env, dependencies)
- Import conventions (relative inside package, absolute for entry points)
- Testing: `pytest tests/ --cov=src/vision_ml`
- Config convention: all thresholds in YAML, never hardcoded
- Logging: use `from ..logging import get_logger`, never `print()`
- Detection output: always `supervision.Detections`
- Branch naming: `feature/<workstream>/<task-name>`
- PR template with checklist
- Code of conduct reference

**Depends on:** Nothing

---

### DOC-05: API documentation `jules` `beginner` `S`

**Files to modify:**
- `src/vision_ml/api/main.py` (add docstrings to all endpoints)

**Requirements:**
- Add FastAPI endpoint descriptions and response model documentation
- Verify Swagger UI at `/docs` renders correctly
- Add request/response examples in docstrings

**Verify:**
```bash
uvicorn src.vision_ml.api.main:app --reload
# Visit http://localhost:8000/docs
```

**Depends on:** API-04

---

### DOC-06: Update README `jules` `beginner` `M`

**Files to modify:**
- `README.md`

**Requirements:**
- Add quickstart section (3 commands: install, run inference, view dashboard)
- Add architecture diagram (Mermaid or ASCII)
- Add badges: CI status, coverage, Python version, license
- Add "Built With" section listing key dependencies
- Link to WORKPLAN.md for contributors
- Link to CONTRIBUTING.md

**Depends on:** Nothing

---

## Workstream 9: Performance & Optimization

---

### PERF-01: ONNX export and inference `jules` `advanced` `L`

**Files to create:**
- `src/vision_ml/detection/onnx_detector.py`
- `scripts/export_onnx.py`

**Requirements:**
- Create `ONNXDetector` implementing `BaseDetector` interface
- Load ONNX model via `onnxruntime.InferenceSession`
- Handle preprocessing (resize, normalize) and postprocessing (NMS, to `supervision.Detections`)
- Add `onnx` as detector type in `DetectorFactory`
- Add export script: `python scripts/export_onnx.py --model yolo11n --output models/yolo11n.onnx`
- Benchmark against PyTorch detector

**Verify:**
```bash
python scripts/export_onnx.py --model yolo11n --output models/yolo11n.onnx
python -c "
from src.vision_ml.detection.detector_factory import DetectorFactory
config = {'detection': {'detector_type': 'onnx'}, 'model': {'name': 'models/yolo11n.onnx'}}
det = DetectorFactory.from_config(config)
print('ONNX detector loaded')
"
```

**Depends on:** Nothing

---

### PERF-02: Batch frame processing `jules` `advanced` `M`

**Files to modify:**
- `src/vision_ml/detection/base.py` (add `detect_batch()` to interface)
- `src/vision_ml/detection/yolo_detector.py` (implement batch)

**Requirements:**
- Add `detect_batch(images: list[np.ndarray]) -> list[sv.Detections]` to `BaseDetector`
- Implement in `YOLODetector` using YOLO batch predict
- Benchmark: measure FPS improvement over sequential processing

**Depends on:** Nothing

---

### PERF-03: Async inference API `jules` `advanced` `M`

**Files to modify:**
- `src/vision_ml/api/main.py`

**Requirements:**
- Make `/predict/video` non-blocking using FastAPI `BackgroundTasks`
- Return `{"job_id": "uuid", "status": "processing"}` immediately
- Add `GET /jobs/{job_id}` to poll for results
- Store job results in memory (dict) with TTL

**Depends on:** API-01

---

### PERF-04: Memory profiling `jules` `intermediate` `M`

**Files to create:**
- `scripts/profile_memory.py`

**Requirements:**
- Use `tracemalloc` or `memory_profiler`
- Profile: model loading, 100-frame inference loop, analytics writes
- Report peak memory usage per component
- Identify any memory leaks (growing allocations)
- Output results as JSON + human-readable summary

**Verify:**
```bash
python scripts/profile_memory.py --frames 100
```

**Depends on:** Nothing

---

### PERF-05: GPU optimization `human-only` `advanced` `XL`

**Requirements:**
- CUDA stream pipelining for overlapping data transfer and compute
- Mixed precision inference (torch.cuda.amp)
- TensorRT integration (if available)
- Benchmark all optimizations

**Why human-only:** Requires GPU hardware and careful benchmarking.

**Depends on:** PERF-01

---

difficulty/intermediate
difficulty/advanced

size/S  size/M  size/L  size/XL

priority/P0-critical         (blocks other work)
priority/P1-high             (needed for MVP)
priority/P2-medium           (improves quality)
priority/P3-nice-to-have     (polish)
```

---

## References

- [Jules — Google's Async Coding Agent](https://jules.google.com/)
- [AGENTS.md — Standard for AI Coding Agent Instructions](https://agents.md/)
- [AGENTS.md Best Practices](https://agentsmd.io/agents-md-best-practices)
- [Jules Getting Started Docs](https://jules.google/docs/)
- [Jules CLI & API](https://developers.googleblog.com/en/meet-jules-tools-a-command-line-companion-for-googles-async-coding-agent/)
 & Frontend API                →jules
FE-01(Frontendscaffold)FE2UI CompontjlesFE04Live Monit)   