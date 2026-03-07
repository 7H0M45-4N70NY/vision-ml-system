# Utils Module — `src/vision_ml/utils/`

## What This Module Does

This is the **foundation layer** of the entire system. Every other module depends on it. Right now it only has config loading, but this is intentional — in a FAANG-scale system, the utils layer must stay **thin and stable** because everything imports from it.

## Files

### `config.py`

Four functions that handle all YAML configuration management:

| Function | Purpose |
|---|---|
| `load_config(path)` | Reads a `.yaml` file from disk into a Python `dict`. Returns empty dict if file is empty. |
| `save_config(config, path)` | Writes a config dict back to YAML. Creates parent directories automatically. |
| `merge_configs(base, override)` | Deep-merges two config dicts. Override wins on conflict. This is critical for layered configs (base + experiment override). |
| `validate_config(config, required_keys)` | Returns a list of missing top-level keys. Empty list = config is valid. |

### Why `merge_configs` Matters

In a real training pipeline, you don't edit `base.yaml` directly. Instead you create an experiment override:

```python
base = load_config("config/training/base.yaml")
experiment = load_config("config/training/my_experiment.yaml")
final = merge_configs(base, experiment)
```

This is the same pattern used by Hydra (Facebook/Meta's config framework) and OmegaConf. We keep it simple here with plain dicts, but the concept scales to thousands of experiments.

### Why `validate_config` Exists

At FAANG scale, misconfigured training jobs waste GPU hours ($$). A quick validation at the start of every pipeline prevents:
- Missing `model` key → crash 10 minutes into training
- Missing `inference` key → silent failures in production

## System Design Thinking

```
Config Loading is a Singleton Concern
--------------------------------------
In a microservice architecture, each service loads its own config once at startup.
We do NOT pass config objects across service boundaries — that creates tight coupling.

Instead, each service (detection, tracking, training) loads its own copy of the config.
This means config changes require a service restart, which is the correct behavior
for ML pipelines (you don't want hyperparameters changing mid-training).

Future scaling:
- Replace YAML files with a config service (etcd, Consul, AWS Parameter Store)
- Add config versioning (every training run snapshots its config)
- Add schema validation (JSON Schema or Pydantic models)
```

## What I Learned Building This

1. **`yaml.safe_load` not `yaml.load`** — `yaml.load` can execute arbitrary Python code from YAML files (security risk). Always use `safe_load`.
2. **`copy.deepcopy` in merge** — Without deepcopy, merging mutates the original base config dict. This causes horrific bugs where Experiment 2's config bleeds into Experiment 1.
3. **Empty YAML files return `None`** — `yaml.safe_load` on an empty file returns `None`, not `{}`. The `if config is None: return {}` guard prevents downstream `TypeError: 'NoneType' object is not subscriptable`.
