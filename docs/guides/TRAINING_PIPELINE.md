# Training Pipeline (Current)

This document describes the **current** training flow used in this repository.

## Scope

- Data preparation from local and/or Roboflow sources
- Validation and deduplication of labels before training
- YOLO-format dataset generation (`data/prepared/dataset.yaml`)
- Training config via `config/training/base.yaml`
- MLflow + DagsHub integration hooks

## 1) Data Preparation

Use `scripts/prepare_data.py` to build the training dataset.

```bash
# local labels only
python scripts/prepare_data.py --config config/training/base.yaml --source local

# roboflow dataset only
python scripts/prepare_data.py --config config/training/base.yaml --source roboflow

# merge local + roboflow
python scripts/prepare_data.py --config config/training/base.yaml --source both
```

Output:

- `data/prepared/train/images`
- `data/prepared/train/labels`
- `data/prepared/val/images`
- `data/prepared/val/labels`
- `data/prepared/dataset.yaml`

## 2) Validation Guarantees in `prepare_data.py`

The ingestion pipeline validates sample quality before training:

- label must be a `dict`
- required keys must exist: `image_path`, `boxes`, `class_ids`, `source`
- `boxes` and `class_ids` must be lists
- missing class IDs are padded safely when needed
- invalid samples are dropped with warning logs
- source priority (`data.source_priority`) is applied before dedupe

This prevents malformed labels from reaching the training phase.

## 3) Training Configuration

Primary config file: `config/training/base.yaml`

Important sections:

- `model.name`: currently `yolo11n`
- `training.*`: epochs, batch size, image size, optimizer, device
- `mlflow.*`: experiment name, run name, registry options
- `detection.*`: primary/secondary detector choices
- `labeling.*`: Roboflow workspace/project and provider

## 4) Running Training

The training entrypoint depends on your selected workflow/script implementation.

Recommended pattern:

```bash
python scripts/train.py --config config/training/base.yaml
```

If your branch does not yet include `scripts/train.py`, complete data prep first and then wire training using the same config file.

## 5) MLflow / DagsHub Notes

For experiment tracking:

- configure `.env` with DagsHub credentials
- ensure `mlflow.tracking_uri` points to your DagsHub MLflow endpoint
- keep secrets out of YAML; use env injection (`inject_secrets`)

Related guide:

- [DAGSHUB_MLFLOW_DVC_INTEGRATION.md](./DAGSHUB_MLFLOW_DVC_INTEGRATION.md)

## 6) DVC Pipeline Integration

Typical DVC stages:

1. `prepare_data`
2. `train`
3. `evaluate` (optional)

Keep `dvc.yaml` as the source of truth for reproducible training runs.

## 7) Troubleshooting

### Roboflow credentials not configured
- set `ROBOFLOW_API_KEY` in `.env` (or environment)
- verify `labeling.provider` and workspace/project in training config

### No samples found
- run inference/auto-labeling first for local data
- verify Roboflow version/project exists

### Invalid label structure
- check logs for dropped samples
- ensure labels include required schema keys

## 8) Related Docs

- [Dataset Structure](./DATASET_STRUCTURE.md)
- [Secrets Injection Pattern](./SECRETS_INJECTION_PATTERN.md)
- [DagsHub + MLflow + DVC](./DAGSHUB_MLFLOW_DVC_INTEGRATION.md)
- [Documentation Index](../INDEX.md)
