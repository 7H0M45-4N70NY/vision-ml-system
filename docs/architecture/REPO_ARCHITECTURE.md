# Repository Architecture (Current)

This document describes the **current high-level repository layout** and ownership boundaries.

## Top-Level Layout

```text
vision-ml-system/
├── config/                 # YAML configs for inference/training
├── data/                   # local datasets, prepared outputs, sample videos
├── docs/                   # documentation index + guides
├── examples/               # lightweight demos
├── frontend/               # Next.js dashboard app
├── notebooks/              # experimentation and cloud notebooks
├── pages/                  # Streamlit pages (legacy/ops UI)
├── scripts/                # CLI entrypoints and pipeline helpers
├── src/vision_ml/          # core application modules
├── tests/                  # pytest suite
├── dvc.yaml                # DVC pipeline stages
├── pyproject.toml          # tool config
└── README.md               # project entrypoint
```

## Core Python Package: `src/vision_ml`

```text
src/vision_ml/
├── analytics/              # visitor metrics and analytics DB logic
├── annotation/             # annotation adapters/utilities
├── api/                    # FastAPI app and endpoints
├── detection/              # detector interfaces + factory-backed implementations
├── events/                 # pipeline event publishers/base classes
├── inference/              # runtime inference orchestration
├── labeling/               # auto-labeling and Roboflow integration
├── logging/                # shared logging utilities
├── tracking/               # tracker interfaces + implementations
├── training/               # training logic and helpers
├── utils/                  # config + utility modules
└── mlflow_integration.py   # MLflow helper integrations
```

## Entry Points and Import Rules

- `src/vision_ml/**`: prefer **relative imports** inside the package
- `pages/**`: use absolute imports from `src.vision_ml...`
- `scripts/**`: use `sys.path` bootstrapping and import from `vision_ml...`

This keeps package boundaries clear and avoids path issues in Docker/CI.

## Configuration Ownership

- `config/inference/base.yaml`: runtime inference behavior
- `config/training/base.yaml`: data prep + training + MLflow settings
- secrets are **not** hardcoded in YAML; injected from env via `inject_secrets`

## Data and Pipeline Artifacts

- `data/auto_labeled/` and `data/low_confidence_frames/`: ingestion inputs
- `data/prepared/`: generated train/val YOLO dataset
- `dvc.yaml`: should orchestrate preparation and training stages

## Documentation Map

Use `docs/INDEX.md` as the canonical navigation entrypoint.
