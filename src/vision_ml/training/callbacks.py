import sys
import os
import mlflow
import dagshub

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ..logging import get_logger

logger = get_logger(__name__)


class MLflowCallback:
    def __init__(self, config: dict):
        self.config = config
        self.mlflow_cfg = config.get('mlflow', {})
        self.run = None

        # Ensure MLflow file store or sqlite store never crashes on fallback
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

        # Synchronize credentials from .env for DagsHub and MLflow
        token = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN") or os.getenv("DAGSHUB_USER_TOKEN")
        user = os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("DAGSHUB_USERNAME")
        if token:
            os.environ["DAGSHUB_USER_TOKEN"] = token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            try:
                import dagshub.auth
                dagshub.auth.add_app_token(token)
            except Exception as auth_err:
                logger.debug(f"DagsHub token registration note: {auth_err}")

        if user:
            os.environ["DAGSHUB_USERNAME"] = user
            os.environ["MLFLOW_TRACKING_USERNAME"] = user

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or self.mlflow_cfg.get('tracking_uri', '')
        if not tracking_uri:
            local_db = os.path.abspath("data/mlflow.db")
            mlflow.set_tracking_uri(f"sqlite:///{local_db.replace(os.sep, '/')}")
            mlflow.set_experiment(self.mlflow_cfg.get('experiment_name', 'default_experiment'))
            return

        try:
            if 'dagshub.com' in tracking_uri:
                # Set up direct HTTP authentication for MLflow with DagsHub (avoids interactive OAuth prompts)
                repo_owner, _ = self._parse_dagshub_uri(tracking_uri)
                dagshub_user = user or repo_owner
                if dagshub_user:
                    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
                if token:
                    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.mlflow_cfg.get('experiment_name', 'default_experiment'))
            logger.info(f"Connected to remote MLflow: {tracking_uri}")

        except Exception as e:
            logger.warning(f"Could not connect to remote MLflow: {e}")

            # Check for auto-fallback environment variable or prompt user explicitly
            allow_fallback = os.getenv("MLFLOW_AUTO_FALLBACK", "").lower() in ("true", "1", "yes")

            if not allow_fallback:
                # Prompt the user interactively if in a terminal
                if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                    print("\n" + "=" * 65)
                    print(" [MLFLOW REMOTE TRACKING CONNECTION FAILED]")
                    print(f" Target URI : {tracking_uri}")
                    print(f" Error      : {e}")
                    print("=" * 65)
                    try:
                        choice = input("\n[?] Fall back to local SQLite tracking (sqlite:///data/mlflow.db)? [y/N]: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        choice = "n"

                    if choice in ("y", "yes"):
                        allow_fallback = True
                    else:
                        print("[!] Local fallback was declined by user. Aborting training.")
                        raise RuntimeError(
                            f"Remote MLflow connection to {tracking_uri} failed ({e}), and local fallback was declined by user."
                        ) from e
                else:
                    logger.error("Remote MLflow failed in non-interactive mode. Aborting (set MLFLOW_AUTO_FALLBACK=true to allow auto-fallback).")
                    raise

            logger.info("Falling back to local SQLite MLflow tracking...")
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            local_db = os.path.abspath("data/mlflow.db")
            mlflow.set_tracking_uri(f"sqlite:///{local_db.replace(os.sep, '/')}")
            try:
                mlflow.set_experiment(self.mlflow_cfg.get('experiment_name', 'default_experiment'))
            except Exception as exp_err:
                logger.warning(f"Local experiment set warning: {exp_err}")

    def _parse_dagshub_uri(self, uri: str):
        try:
            parts = uri.split('dagshub.com/')[1].split('.mlflow')[0].split('/')
            return parts[0], parts[1]
        except Exception:
            return None, None

    def start_run(self, run_name: str = None):
        name = run_name or self.mlflow_cfg.get('run_name', 'unnamed_run')
        self.run = mlflow.start_run(run_name=name)
        if self.mlflow_cfg.get('log_params', True):
            self._log_config()
        return self.run

    def _log_config(self):
        flat = self._flatten_dict(self.config)
        safe_params = {}
        for k, v in flat.items():
            str_val = str(v)
            if len(str_val) <= 500:
                safe_params[k] = str_val
        mlflow.log_params(safe_params)

    def log_metrics(self, metrics: dict, step: int = None):
        if not self.mlflow_cfg.get('log_metrics', True):
            return
        import re
        safe = {}
        for k, v in metrics.items():
            try:
                # Convert to float and ensure it's a finite number
                float_val = float(v)
                if float_val == float('inf') or float_val == float('-inf') or float_val != float_val:
                    continue  # Skip NaN and infinite values
                # Sanitize metric key for MLflow (replace invalid characters like parentheses with underscores)
                clean_key = re.sub(r'[^a-zA-Z0-9_\-\. /]', '_', str(k))
                safe[clean_key] = float_val
            except (TypeError, ValueError):
                continue
        if safe:
            try:
                mlflow.log_metrics(safe, step=step)
            except Exception as e:
                logger.warning(f"MLflow log_metrics note (skipped non-critical metric logging): {e}")

    def log_artifact(self, path: str, artifact_path: str = None):
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path)

    def log_model(self, model_path: str, artifact_path: str = "models"):
        if self.mlflow_cfg.get('log_model', True) and os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path)

    def register_model(self, model_path: str):
        if not self.mlflow_cfg.get('register_model', False):
            return
        model_name = self.mlflow_cfg.get('model_name', 'vision-ml-model')
        if self.run is not None and os.path.exists(model_path):
            run_id = self.run.info.run_id
            mlflow.log_artifact(model_path, "registered_model")
            artifact_uri = f"runs:/{run_id}/registered_model/{os.path.basename(model_path)}"
            try:
                mlflow.register_model(artifact_uri, model_name)
                logger.info(f"Model registered as '{model_name}'")
            except Exception as e:
                logger.warning(f"Model registration note: {e}")

    def set_tag(self, key: str, value: str):
        mlflow.set_tag(key, value)

    def end_run(self):
        mlflow.end_run()
        self.run = None

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
