import yaml
import os
import copy
from typing import Optional


def load_config(config_path: str) -> dict:
    """Load YAML configuration from disk."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    return config


def save_config(config: dict, output_path: str) -> None:
    """Persist a config dict back to YAML on disk."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge *override* into a copy of *base* (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_config(config: dict, required_keys: Optional[list] = None) -> list:
    """Return a list of missing top-level keys (empty list = valid)."""
    required = required_keys or ['model', 'inference']
    return [k for k in required if k not in config]
