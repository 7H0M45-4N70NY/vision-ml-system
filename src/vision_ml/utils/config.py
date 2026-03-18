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


def inject_secrets(config: dict) -> dict:
    """Inject environment variables into config (secrets layer).
    
    Maps environment variables to config keys following the pattern:
    - ROBOFLOW_API_KEY → labeling.roboflow_api_key
    - S3_BUCKET → storage.s3_bucket
    - DATABASE_URL → database.url
    
    Args:
        config: Configuration dict loaded from YAML
        
    Returns:
        Config dict with injected secrets
        
    Raises:
        ValueError: If required secrets are missing for enabled providers
    """
    secrets_map = {
        'roboflow_api_key': 'ROBOFLOW_API_KEY',
        's3_bucket': 'S3_BUCKET',
        'database_url': 'DATABASE_URL',
    }
    
    labeling = config.get('labeling', {})
    
    # Inject Roboflow API key if provider is roboflow
    if labeling.get('provider') == 'roboflow':
        rf_key = os.getenv('ROBOFLOW_API_KEY')
        if not rf_key:
            raise ValueError(
                "Roboflow provider configured but ROBOFLOW_API_KEY not set in environment. "
                "Set ROBOFLOW_API_KEY or use --source local"
            )
        labeling['roboflow_api_key'] = rf_key
    elif 'ROBOFLOW_API_KEY' in os.environ:
        # Inject if env var exists, even if not explicitly configured
        labeling['roboflow_api_key'] = os.getenv('ROBOFLOW_API_KEY')
    
    config['labeling'] = labeling
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
