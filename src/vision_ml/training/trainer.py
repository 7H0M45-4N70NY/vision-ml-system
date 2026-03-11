import os
import json
from ultralytics import YOLO
from .callbacks import MLflowCallback


class Trainer:
    """Encapsulates the YOLO training lifecycle with MLflow integration.

    Supports manual and drift-triggered training through a single unified
    _run_training() method.  Callers use train() or train_on_drift() which
    only differ in the MLflow tag and default run name.
    """

    def __init__(self, config: dict):
        self.config = config
        self.mlflow_callback = MLflowCallback(config)

        model_cfg = config.get('model', {})
        self.model_name = model_cfg.get('name', 'yolo11n')
        if not self.model_name.endswith('.pt') and not self.model_name.endswith('.yaml'):
            self.model_name += '.pt'

        self.model = YOLO(self.model_name)

    # -- Public API ----------------------------------------------------------

    def train(self, run_name: str = None, dataset_yaml: str = None):
        name = run_name or self.config.get('mlflow', {}).get('run_name', 'exp')
        return self._run_training(name, trigger='manual', dataset_yaml=dataset_yaml)

    def train_on_drift(self, run_name: str = None, dataset_yaml: str = None):
        name = run_name or 'drift_retrain'
        return self._run_training(name, trigger='drift_detected', dataset_yaml=dataset_yaml)

    # -- Core training (single implementation) --------------------------------

    def _run_training(self, run_name: str, trigger: str, dataset_yaml: str = None):
        train_cfg = self.config.get('training', {})
        data_cfg = self.config.get('data', {})
        dataset = dataset_yaml or data_cfg.get('dataset_yaml') or 'coco8.yaml'

        self.mlflow_callback.start_run(run_name=run_name)
        self.mlflow_callback.set_tag('trigger', trigger)

        try:
            print(f"[Trainer] Starting training ({trigger}): {self.model_name} "
                  f"for {train_cfg.get('epochs', 10)} epochs")
            print(f"[Trainer] Dataset: {dataset}")

            results = self.model.train(
                data=dataset,
                epochs=train_cfg.get('epochs', 10),
                batch=train_cfg.get('batch_size', 16),
                imgsz=train_cfg.get('imgsz', 640),
                lr0=train_cfg.get('learning_rate', 0.01),
                optimizer=train_cfg.get('optimizer', 'auto'),
                device=train_cfg.get('device', 'cpu'),
                patience=train_cfg.get('patience', 5),
                project=train_cfg.get('project', 'runs/train'),
                name=run_name,
                exist_ok=True,
            )

            self._log_results(results, train_cfg, run_name)
            self._save_metrics(results, train_cfg, run_name)
            print(f"[Trainer] Training completed ({trigger}).")
            return results

        except Exception as e:
            self.mlflow_callback.set_tag('status', 'failed')
            print(f"[Trainer] Training failed: {e}")
            raise
        finally:
            self.mlflow_callback.end_run()

    # -- Helpers --------------------------------------------------------------

    def _log_results(self, results, train_cfg: dict, run_name: str):
        if hasattr(results, 'results_dict'):
            self.mlflow_callback.log_metrics(results.results_dict)

        project = train_cfg.get('project', 'runs/train')
        best_path = os.path.join(project, run_name, 'weights', 'best.pt')

        if os.path.exists(best_path):
            self.mlflow_callback.log_model(best_path)
            self.mlflow_callback.register_model(best_path)
        else:
            print(f"[Trainer] Best model not found at {best_path}, skipping artifact log.")

    def _save_metrics(self, results, train_cfg: dict, run_name: str):
        """Write a metrics.json alongside training output for DVC tracking."""
        metrics = {}
        if hasattr(results, 'results_dict'):
            for k, v in results.results_dict.items():
                try:
                    metrics[k] = float(v)
                except (TypeError, ValueError):
                    metrics[k] = str(v)

        project = train_cfg.get('project', 'runs/train')
        metrics_path = os.path.join(project, 'metrics.json')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[Trainer] Metrics saved to {metrics_path}")
