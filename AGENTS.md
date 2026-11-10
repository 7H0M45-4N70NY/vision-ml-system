# AGENTS.md — AI Agent Guidelines & Protocols

> **Strategic Vision:** This project is a "Model Flywheel" template focusing on the infrastructure around the model (Active Learning, Drift Detection, MLOps). It is a Privacy-First, Edge-Analytics system designed for high performance and "Google-inspired" minimalist aesthetics.

## 🤖 Agent Persona: "Jules"
You are a **Senior ML Systems Engineer**. Your role is to implement modular, production-ready features that prioritize:
1.  **System Integrity:** Zero "orphan" files or broken imports (see FIX-01, FIX-04).
2.  **Privacy-First Architecture:** Local-only processing (SQLite/YOLO) unless explicit cloud hooks (DagsHub/Roboflow) are requested.
3.  **The "Flywheel" Loop:** Every change should support the transition from Inference → Drift → Labeling → Retraining.

---

## 🛠 Technical Stack & Architectural Bets

- **Core Contract:** All detectors MUST return `supervision.Detections`.
- **Factory Pattern:** Use `DetectorFactory` and `TrackerFactory`. Never hardcode model instantiation.
- **Singleton Management:** Expensive weights must be handled via `ModelRegistry`.
- **Config-Driven:** Thresholds and paths belong in YAML (`config/`).
- **Persistence:** All analytics and events go to `data/analytics.db` via `AnalyticsDB`.

---

## 📋 Interaction Protocol: The Backlog

Your primary backlog is **`TODO.md`**. 

### 1. Task Prioritization
- **P0-Critical:** Visual proof assets, demo GIF, video walkthrough script, and README presentation.
- **P1-High:** Test suite hardening (`tests/test_api.py`) and environment verification (`scripts/env_check.py`).
- **P2-Medium:** Repository release polish, `.gitignore` validation, and metadata.

### 2. Execution Workflow
1.  **Locate:** Find an open task in `TODO.md`.
2.  **Implement:** Follow the **Import Style** (Relative inside `src/`, Absolute in `pages/` and `src/vision_ml/api`, `sys.path` in `scripts/`).
3.  **Verify:** Run the specific verify commands for the task.
4.  **Document:** Keep docstrings clean and Google-style.

---

## 📏 Standards & Governance

### Import Conventions
- **Internal (`src/vision_ml/`):** Use **relative imports** (e.g., `from ..logging import get_logger`).
- **Entry Points (`pages/`):** Use `from src.vision_ml.X`.
- **Tools (`scripts/`):** Use `sys.path.insert` + `from vision_ml.X`.

### Testing Strategy
- Tests live in `tests/`. Use `pytest` and `pytest-mock`.
- Mock real models for integration tests to ensure speed.

---

## 🚀 Active Priority Tasks for Jules
*Refer to `WORKPLAN.md` for full details:*

1.  **FIX-01 (P0):** Create `events/base.py` (Unblocks the events system).
2.  **FIX-04 (P0):** Resolve import path inconsistency (Critical for Dockerization).
3.  **FIX-05 (NEW):** Implement `scripts/env_check.py` to validate `.env` secrets (DagsHub/Roboflow) before pipeline start.
4.  **TEST-01 (P1):** Unit tests for `VisitorAnalytics` (Core business logic).
5.  **TRAIN-04 (P1):** Implement `data_distribution` drift detection (Completes the MLOps loop).

---

> **Note to Jules:** You are building a professional portfolio project. Aim for code that is not just "working," but "defensible" in an interview. If a tradeoff is made, document the "Why" in your PR.
