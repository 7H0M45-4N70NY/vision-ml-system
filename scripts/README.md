# Scripts — `scripts/`

Terminal-first entry points for training, MLflow, DVC, pipeline checks, and inference.

Before running scripts, create env file from template:

```bash
copy .env.example .env
```

## Script Matrix

| Capability | Script | Example |
|---|---|---|
| Train (manual/drift/auto) | `scripts/train.py` | `python scripts/train.py --trigger auto --device cuda --epochs 50` |
| MLflow experiments/runs/best | `scripts/mlflow_cli.py` | `python scripts/mlflow_cli.py --action experiments` |
| Model registry list/compare/promote | `scripts/model_registry_cli.py` | `python scripts/model_registry_cli.py --action promote --model-name visitor-analytics-yolo11n` |
| Pipeline status/drift checks/trigger | `scripts/pipeline_cli.py` | `python scripts/pipeline_cli.py --action drift-check --threshold 0.2` |
| DVC pull/add/push/status | `scripts/dvc_cli.py` | `python scripts/dvc_cli.py --action pull` |
| Auto-labeling export/upload | `scripts/auto_labeling_cli.py` | `python scripts/auto_labeling_cli.py --frame-dir data/low_confidence_frames` |
| Analytics snapshots | `scripts/analytics_cli.py` | `python scripts/analytics_cli.py --action summary` |
| Inference (offline/online) | `scripts/inference.py` | `python scripts/inference.py --mode offline --source demo.mp4` |

---

## `train.py` (expanded)

```bash
# manual training
python scripts/train.py --trigger manual --device cuda

# drift-triggered training
python scripts/train.py --trigger drift --run-name drift_retrain_20260307

# auto trigger resolution from config.schedule.mode
python scripts/train.py --trigger auto

# with DVC hooks
python scripts/train.py --enable-dvc --dvc-pull --dvc-add --dvc-push --dvc-targets runs/train,data/raw

# disable MLflow logging for local smoke test
python scripts/train.py --disable-mlflow
```

Key flags:
- `--trigger manual|drift|auto`
- `--dataset-yaml`, `--epochs`, `--batch-size`, `--learning-rate`, `--device`
- `--enable-dvc --dvc-pull --dvc-add --dvc-push --dvc-targets`
- `--disable-mlflow`

---

## `mlflow_cli.py`

```bash
# list experiments
python scripts/mlflow_cli.py --action experiments

# list runs in an experiment
python scripts/mlflow_cli.py --action runs --experiment-name visitor_analytics_training --limit 20

# best run by metric
python scripts/mlflow_cli.py --action best --experiment-name visitor_analytics_training --metric-name val_loss --mode min
```

---

## `model_registry_cli.py`

```bash
# list registered models
python scripts/model_registry_cli.py --action list

# compare staging vs production
python scripts/model_registry_cli.py --action compare --model-name visitor-analytics-yolo11n --metric-name val_loss

# promote staging -> production
python scripts/model_registry_cli.py --action promote --model-name visitor-analytics-yolo11n --from-stage Staging --to-stage Production
```

---

## `pipeline_cli.py`

```bash
# show recent training events
python scripts/pipeline_cli.py --action status --limit 30

# run drift check from inference history
python scripts/pipeline_cli.py --action drift-check --threshold 0.2 --lookback 50

# create a training event if drift is detected
python scripts/pipeline_cli.py --action trigger --threshold 0.2 --model-version v2
```

Use `--force` on trigger action to enqueue an event even without drift.

---

## `dvc_cli.py`

```bash
python scripts/dvc_cli.py --action status
python scripts/dvc_cli.py --action pull
python scripts/dvc_cli.py --action add --targets runs/train,data/raw
python scripts/dvc_cli.py --action push
```

---

## `auto_labeling_cli.py`

```bash
python scripts/auto_labeling_cli.py --frame-dir data/low_confidence_frames --output-dir data/auto_labeled
python scripts/auto_labeling_cli.py --provider roboflow
```

---

## `analytics_cli.py`

```bash
python scripts/analytics_cli.py --action summary
python scripts/analytics_cli.py --action inference-runs --limit 20
python scripts/analytics_cli.py --action training-events --limit 20
python scripts/analytics_cli.py --action labeling-events --limit 20
```

---

## `inference.py`

```bash
python scripts/inference.py --mode offline --source demo.mp4 --output annotated.mp4
python scripts/inference.py --mode online --source 0
```

---

## Notes

- All scripts are designed for terminal visibility and explicit progress output.
- This supports cloud GPU notebook usage by running scripts directly in notebook terminals.
- Streamlit remains available, but every major MLOps path now has a script entry point.
