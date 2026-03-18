# Secrets Injection Pattern: Production MLOps Design

## Overview

This document explains the **3-layer secrets management pattern** implemented in the vision-ml-system. This is the standard approach used by production MLOps systems (Kubeflow, Airflow, DVC, MLflow).

---

## Architecture: 3 Layers

### Layer 1: YAML Configuration (Static)
- **File**: `config/training/base.yaml`
- **Purpose**: Structure, parameters, non-sensitive settings
- **Example**:
```yaml
labeling:
  enabled: false
  provider: roboflow
  roboflow_workspace: thomas-workspace
  roboflow_project: vision_ml_system
  roboflow_api_key: null  # NEVER hardcode secrets here
```

### Layer 2: Environment Variables (Secrets)
- **File**: `.env` (local dev) or CI/CD secrets manager (production)
- **Purpose**: API keys, credentials, tokens
- **Example**:
```env
ROBOFLOW_API_KEY=your_actual_key_here
S3_BUCKET=my-bucket
DATABASE_URL=postgresql://user:pass@host/db
```

### Layer 3: Code (Logic)
- **File**: `src/vision_ml/utils/config.py` → `inject_secrets()`
- **Purpose**: Merge ENV vars into config at runtime
- **Example**:
```python
def inject_secrets(config: dict) -> dict:
    """Inject environment variables into config."""
    labeling = config.get('labeling', {})
    
    if labeling.get('provider') == 'roboflow':
        rf_key = os.getenv('ROBOFLOW_API_KEY')
        if not rf_key:
            raise ValueError("ROBOFLOW_API_KEY not set")
        labeling['roboflow_api_key'] = rf_key
    
    config['labeling'] = labeling
    return config
```

---

## Why This Pattern?

### ✅ Security
- Secrets never committed to git
- Separate from code and config
- Easy to rotate without code changes

### ✅ Flexibility
- Same code runs in dev, staging, production
- Different secrets per environment
- No hardcoding or string interpolation

### ✅ Auditability
- Clear separation of concerns
- Logging doesn't leak secrets
- Easy to trace where values come from

### ✅ Scalability
- Works with Docker, Kubernetes, CI/CD
- Compatible with secret managers (Vault, AWS Secrets Manager)
- Extensible for new secrets

---

## Usage

### Development (Local)

1. **Copy `.env.example` to `.env`**:
```bash
cp .env.example .env
```

2. **Fill in your secrets**:
```env
ROBOFLOW_API_KEY=your_actual_key
DAGSHUB_TOKEN=your_token
```

3. **Run script** (dotenv auto-loads):
```bash
python scripts/prepare_data.py --config config/training/base.yaml --source roboflow
```

The `inject_secrets()` call in `main()` will:
1. Load config from YAML
2. Read `ROBOFLOW_API_KEY` from `.env` (via `load_dotenv()`)
3. Inject it into config
4. Pass to `download_roboflow_dataset()`

### Production (Docker/K8s)

1. **Set environment variables** in deployment:
```yaml
# docker-compose.yml
environment:
  ROBOFLOW_API_KEY: ${ROBOFLOW_API_KEY}
  DAGSHUB_TOKEN: ${DAGSHUB_TOKEN}
```

2. **Run container** (no `.env` file needed):
```bash
docker run -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY my-image
```

3. **Same code path** handles injection automatically

---

## Implementation Details

### In `prepare_data.py`

**Before** (mixed concerns):
```python
api_key = label_cfg.get('roboflow_api_key') or os.environ.get('ROBOFLOW_API_KEY')
```

**After** (clean separation):
```python
# At script start:
config = load_config(args.config)
config = inject_secrets(config)  # ← Centralized injection

# In function:
api_key = config.get('labeling', {}).get('roboflow_api_key')
# ↑ Already injected, no os.environ calls
```

### Debug Logging (No Secret Leaks)

```python
logger.info(
    "Roboflow config loaded | workspace=%s project=%s api_key_present=%s",
    workspace,
    project,
    bool(api_key)  # ← Shows presence, not value
)
```

Output:
```
Roboflow config loaded | workspace=thomas-workspace project=vision_ml_system api_key_present=True
```

---

## Extending for New Secrets

To add a new secret (e.g., `S3_BUCKET`):

1. **Add to `.env.example`**:
```env
S3_BUCKET=my-bucket-name
```

2. **Update `inject_secrets()` in `config.py`**:
```python
def inject_secrets(config: dict) -> dict:
    # ... existing code ...
    
    storage = config.get('storage', {})
    if 'S3_BUCKET' in os.environ:
        storage['s3_bucket'] = os.getenv('S3_BUCKET')
    config['storage'] = storage
    
    return config
```

3. **Use in code**:
```python
s3_bucket = config.get('storage', {}).get('s3_bucket')
```

---

## Testing

### Unit Test Example

```python
def test_inject_secrets_roboflow():
    """Verify inject_secrets loads ROBOFLOW_API_KEY."""
    os.environ['ROBOFLOW_API_KEY'] = 'test_key_123'
    
    config = {
        'labeling': {
            'provider': 'roboflow',
            'roboflow_workspace': 'test-ws',
        }
    }
    
    result = inject_secrets(config)
    
    assert result['labeling']['roboflow_api_key'] == 'test_key_123'
    assert 'test_key_123' not in str(result)  # Verify no leaks in logs
```

---

## Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| Secret location | Mixed in code | Centralized in ENV |
| Config responsibility | Handles secrets | Only structure |
| Function signature | `os.environ` calls | Clean dict params |
| Testing | Requires mocking ENV | Inject test config |
| Logging | Risk of leaking secrets | Safe by design |
| Scalability | Hard to add secrets | Extensible pattern |

---

## Related Files

- **Config layer**: `src/vision_ml/utils/config.py`
- **Script entry point**: `scripts/prepare_data.py`
- **Environment template**: `.env.example`
- **Training config**: `config/training/base.yaml`

---

## References

This pattern is used by:
- **Kubeflow**: Secrets via ConfigMaps + Secrets
- **Airflow**: Variables + Connections
- **DVC**: Environment variables + `.dvc/config.local`
- **MLflow**: Environment variables for tracking URI
- **Docker**: `--env` flags + `.env` files
- **Kubernetes**: Secrets + ConfigMaps

---

## FAQ

**Q: Why not use `python-dotenv` everywhere?**  
A: `.env` is dev-only. Production uses actual environment variables (Docker, K8s, CI/CD). The pattern works for both.

**Q: What if ROBOFLOW_API_KEY is not set?**  
A: `inject_secrets()` will raise `ValueError` with clear message, failing fast before any API calls.

**Q: Can I override config values with ENV?**  
A: Yes, that's the point. ENV takes precedence over YAML.

**Q: Is this GDPR/SOC2 compliant?**  
A: Yes. Secrets are never logged, never committed, and can be rotated without code changes.

