import os
import mlflow
import dagshub


class MLflowCallback:
    def __init__(self, config: dict):
        self.config = config
        self.mlflow_cfg = config.get('mlflow', {})
        self.run = None

        tracking_uri = self.mlflow_cfg.get('tracking_uri', '')
        if 'dagshub.com' in tracking_uri:
            repo_owner, repo_name = self._parse_dagshub_uri(tracking_uri)
            if repo_owner and repo_name:
                dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.mlflow_cfg.get('experiment_name', 'default_experiment'))

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
        safe = {}
        for k, v in metrics.items():
            try:
                safe[k] = float(v)
            except (TypeError, ValueError):
                continue
        if safe:
            mlflow.log_metrics(safe, step=step)

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
                print(f"Model registered as '{model_name}'")
            except Exception as e:
                print(f"Model registration note: {e}")

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
