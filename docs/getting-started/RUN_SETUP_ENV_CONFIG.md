# Run Setup: Environment Variables + Config Reference

This document is the single source of truth for running the application (Streamlit + scripts) with correct env vars and config files.

---

## 1) Where to put environment variables

Create a `.env` file in the project root:

```bash
copy .env.example .env
```

Then edit `.env` with your values.

Template file provided at:
- `.env.example`

---

## 2) Environment variables (complete list)

| Variable | Required | Used By | Purpose |
|---|---|---|---|
| `DAGSHUB_USERNAME` | Required for remote DagsHub tracking | `src/vision_ml/mlflow_integration.py` | DagsHub owner/account for MLflow initialization |
| `DAGSHUB_TOKEN` | Required for authenticated DagsHub access | DagsHub SDK / MLflow auth | Access token for DagsHub |
| `MLFLOW_TRACKING_USERNAME` | Recommended for remote MLflow auth | MLflow HTTP auth | Username for DagsHub-hosted MLflow |
| `MLFLOW_TRACKING_PASSWORD` | Recommended for remote MLflow auth | MLflow HTTP auth | Password/token for DagsHub-hosted MLflow |
| `MLFLOW_TRACKING_URI` | Optional override | MLflow client | Explicit tracking server URI |
| `ROBOFLOW_API_KEY` | Required only when `labeling.provider=roboflow` | `src/vision_ml/labeling/auto_labeler.py` | Roboflow upload/auth for auto-labeling |
| `ENV` | Optional | `src/vision_ml/mlflow_integration.py` | Run metadata tag (`environment`) |
| `USER` | Optional | `src/vision_ml/mlflow_integration.py` | Run metadata tag (`user`) |

### Notes
- If DagsHub credentials are missing, code may fall back to local MLflow (`./mlruns`) depending on path used.
- `ROBOFLOW_API_KEY` is not needed for local label export mode.

---

## 3) Config files used by application/scripts

| Config File | Used By | Main Purpose |
|---|---|---|
| `config/training/base.yaml` | `scripts/train.py`, `scripts/auto_labeling_cli.py`, training modules | Training hyperparameters, schedule, drift, MLflow, labeling |
| `config/inference/base.yaml` | `scripts/inference.py`, inference pipeline | Detection/tracking/inference runtime settings |

---

## 4) Script to config mapping

| Script | Config Input | Default |
|---|---|---|
| `scripts/train.py` | `--config` | `config/training/base.yaml` |
| `scripts/inference.py` | `--config` | `config/inference/base.yaml` |
| `scripts/auto_labeling_cli.py` | `--config` | `config/training/base.yaml` |
| `scripts/mlflow_cli.py` | N/A | Uses MLflow/DagsHub env + existing tracked experiments |
| `scripts/model_registry_cli.py` | N/A | Uses MLflow/DagsHub env + registry data |
| `scripts/pipeline_cli.py` | N/A | Reads analytics DB (`data/analytics.db`) |
| `scripts/dvc_cli.py` | N/A | Uses DVC project config (`dvc.yaml` / `.dvc/`) |
| `scripts/analytics_cli.py` | N/A | Reads analytics DB (`data/analytics.db`) |

---

## 5) Minimal run checklist

1. Activate environment
   ```bash
   conda activate ./venv
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Create and edit env file
   ```bash
   copy .env.example .env
   ```
4. Run training
   ```bash
   python scripts/train.py --trigger manual
   ```
5. Run inference
   ```bash
   python scripts/inference.py --mode offline --source demo.mp4
   ```
6. Run dashboard
   ```bash
   streamlit run home.py
   ```

---

## 6) Security

- Never commit `.env` with real tokens.
- Use `.env.example` as the shared template.
- Prefer tokens over passwords for third-party services.
