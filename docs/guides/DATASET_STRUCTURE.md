# Dataset Structure and Versioning (Current)

This guide describes the data layout currently used by preparation and training flows.

## Data Inputs

### 1) Local sources

- `data/auto_labeled/auto_labels.json`
- `data/auto_labeled/images/*.jpg`
- `data/low_confidence_frames/*.json` + matching `.jpg`

### 2) Roboflow source

When `--source roboflow` or `--source both` is used, the script downloads a YOLOv8 export into:

- `data/prepared/_roboflow_download/`

## Prepared Output (Canonical Training Input)

After running `scripts/prepare_data.py`, the output is:

```text
data/prepared/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

`dataset.yaml` is the canonical artifact used by training.

## Label Schema Expectations

Each sample label dictionary must contain:

- `image_path: str`
- `boxes: list`
- `class_ids: list`
- `source: str` (`local` or `roboflow`)

The preparation pipeline validates schema and drops invalid samples.

## Source Priority and Dedup

Priority is configured in `config/training/base.yaml`:

```yaml
data:
  source_priority:
    roboflow: 0
    local: 1
```

Behavior:

1. Samples are sorted by source priority.
2. Dedup happens by normalized image filename key.
3. First sample wins after sorting (higher priority retained).

## Typical Commands

```bash
# Local only
python scripts/prepare_data.py --config config/training/base.yaml --source local

# Roboflow only
python scripts/prepare_data.py --config config/training/base.yaml --source roboflow

# Merge sources
python scripts/prepare_data.py --config config/training/base.yaml --source both
```

## DVC Notes

Recommended DVC tracking:

- Track source datasets (if project policy allows)
- Track preparation stage in `dvc.yaml`
- Treat `data/prepared/dataset.yaml` as the handoff artifact for training

## Related Docs

- [Training Pipeline](./TRAINING_PIPELINE.md)
- [DagsHub + MLflow + DVC Integration](./DAGSHUB_MLFLOW_DVC_INTEGRATION.md)
- [Secrets Injection Pattern](./SECRETS_INJECTION_PATTERN.md)
- [Documentation Index](../INDEX.md)
