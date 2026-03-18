# DagsHub + MLflow + DVC Integration (Current)

This guide covers a practical integration pattern for experiment tracking, model registry, and dataset/pipeline versioning.

## Prerequisites

- DagsHub repository created for this project
- `.env` configured with required credentials
- DVC and MLflow installed in your environment

## 1) Environment Variables

Set these in `.env` (or CI secrets):

```env
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_token
MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token
```

## 2) MLflow Tracking Setup

Training config (`config/training/base.yaml`) should include:

```yaml
mlflow:
  experiment_name: visitor_analytics_training
  run_name: yolo11n_baseline
  tracking_uri: https://dagshub.com/<owner>/<repo>.mlflow
  log_params: true
  log_metrics: true
  log_model: true
  register_model: true
  model_name: visitor-analytics-yolo11n
```

Recommended runtime flow:

1. Load YAML config
2. Inject secrets from environment
3. Set MLflow tracking URI
4. Start run and log params/metrics/artifacts

## 3) DVC Setup

Initialize and configure DVC remote for DagsHub:

```bash
dvc remote add -d dagshub https://dagshub.com/<owner>/<repo>.dvc
# store credentials securely (local config / env-based automation)
```

Then use standard workflow:

```bash
dvc repro
dvc push
```

## 4) Recommended Pipeline Stages

In `dvc.yaml`, keep these as canonical stages:

1. `prepare_data` → build `data/prepared/dataset.yaml`
2. `train` → train model and log metrics
3. `evaluate` (optional) → generate comparison artifacts

## 5) Dataset + Model Traceability

To maintain lineage:

- Commit code and config changes in git
- Track dataset/pipeline state with DVC
- Track runs and models with MLflow
- Use consistent run tags (`git_commit`, `dataset_version`, `trigger`)

## 6) Verification Checklist

- `python scripts/prepare_data.py ...` completes successfully
- MLflow run appears in DagsHub with metrics and artifacts
- DVC pipeline reproduces end-to-end on another machine
- Model registration metadata includes source run ID

## 7) Common Failure Points

### Authentication failures
- verify `DAGSHUB_TOKEN`
- verify tracking URI format and repository permissions

### Empty MLflow runs
- confirm training code actually calls `mlflow.log_*`
- confirm run context is active during training

### DVC push issues
- verify remote URL and auth
- verify large artifacts are not accidentally git-tracked

## Related Docs

- [Training Pipeline](./TRAINING_PIPELINE.md)
- [Dataset Structure](./DATASET_STRUCTURE.md)
- [Secrets Injection Pattern](./SECRETS_INJECTION_PATTERN.md)
- [Documentation Index](../INDEX.md)
