import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
from .callbacks import MLflowCallback
from ..logging import get_logger

logger = get_logger(__name__)


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
        raw_name = model_cfg.get('name', 'yolo11n')
        self.model_name = raw_name if raw_name.endswith(('.pt', '.yaml')) else f"{raw_name}.pt"

        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        original_path = repo_root / "models" / "original" / Path(self.model_name).name
        
        if original_path.is_file():
            logger.info(f"Trainer initializing from original base weights: {original_path}")
            self.model = YOLO(str(original_path))
        else:
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
            logger.info(f"Starting training ({trigger}): {self.model_name} "
                        f"for {train_cfg.get('epochs', 10)} epochs")
            logger.info(f"Dataset: {dataset}")

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
            logger.info(f"Training completed ({trigger}).")
            return results

        except Exception as e:
            self.mlflow_callback.set_tag('status', 'failed')
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.mlflow_callback.end_run()

    # -- Helpers --------------------------------------------------------------

    def _log_results(self, results, train_cfg: dict, run_name: str):
        save_dir = getattr(results, 'save_dir', None)
        if not save_dir:
            project = train_cfg.get('project', 'runs/train')
            for cand in [
                os.path.join(project, run_name),
                os.path.join('runs', 'detect', project, run_name),
            ]:
                if os.path.exists(cand):
                    save_dir = cand
                    break

        # 1. Log epoch-by-epoch loss & mAP curves from results.csv
        if save_dir:
            csv_path = Path(save_dir) / 'results.csv'
            if csv_path.is_file():
                try:
                    import csv
                    import re
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            epoch_str = row.get('epoch', '').strip()
                            if not epoch_str:
                                continue
                            epoch = int(epoch_str)
                            metrics = {}
                            for k, v in row.items():
                                k_clean = k.strip()
                                v_clean = v.strip()
                                if k_clean in ('epoch', 'time') or not v_clean:
                                    continue
                                try:
                                    clean_key = re.sub(r'[^a-zA-Z0-9_]', '_', k_clean.replace('/', '_').replace('-', '_'))
                                    clean_key = re.sub(r'_+', '_', clean_key).strip('_')
                                    metrics[clean_key] = float(v_clean)
                                except ValueError:
                                    pass
                            if metrics:
                                self.mlflow_callback.log_metrics(metrics, step=epoch)
                except Exception as e:
                    logger.warning(f"Could not parse results.csv for epoch curve logging: {e}")

            # 2. Upload plot artifacts to MLflow
            for plot_name in [
                'results.png',
                'BoxF1_curve.png',
                'BoxPR_curve.png',
                'BoxP_curve.png',
                'BoxR_curve.png',
                'confusion_matrix.png',
                'confusion_matrix_normalized.png',
            ]:
                p = Path(save_dir) / plot_name
                if p.is_file():
                    try:
                        self.mlflow_callback.log_artifact(str(p), artifact_path='plots')
                    except Exception as e:
                        logger.warning(f"Failed to log plot artifact {plot_name}: {e}")

        if hasattr(results, 'results_dict'):
            try:
                self.mlflow_callback.log_metrics(results.results_dict)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

        # Locate best model weights (checking Ultralytics save_dir first)
        best_path = None
        if save_dir:
            cand = Path(save_dir) / 'weights' / 'best.pt'
            if cand.is_file():
                best_path = str(cand)

        if not best_path:
            project = train_cfg.get('project', 'runs/train')
            for cand_path in [
                os.path.join(project, run_name, 'weights', 'best.pt'),
                os.path.join('runs', 'detect', project, run_name, 'weights', 'best.pt'),
            ]:
                if os.path.exists(cand_path):
                    best_path = cand_path
                    break

        if best_path and os.path.exists(best_path):
            try:
                self.mlflow_callback.log_model(best_path)
                self.mlflow_callback.register_model(best_path)
            except Exception as e:
                logger.warning(f"MLflow model log note: {e}")

            # Copy fine-tuned weights to models/fine_tuned/
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            fine_tuned_dir = repo_root / "models" / "fine_tuned"
            fine_tuned_dir.mkdir(parents=True, exist_ok=True)
            dest_best = fine_tuned_dir / f"{run_name}_best.pt"
            dest_latest = fine_tuned_dir / "best.pt"
            shutil.copy2(best_path, str(dest_best))
            shutil.copy2(best_path, str(dest_latest))
            logger.info(f"Fine-tuned weights saved to: {dest_best} and {dest_latest}")
        else:
            logger.warning("Best model not found, skipping artifact log.")

    def _save_metrics(self, results, train_cfg: dict, run_name: str):
        """Write a metrics.json alongside training output for DVC tracking."""
        metrics = {}
        if hasattr(results, 'results_dict'):
            for k, v in results.results_dict.items():
                try:
                    metrics[k] = float(v)
                except (TypeError, ValueError):
                    metrics[k] = str(v)

        save_dir = getattr(results, 'save_dir', None)
        if save_dir:
            metrics_path = os.path.join(str(save_dir), 'metrics.json')
        else:
            project = train_cfg.get('project', 'runs/train')
            metrics_path = os.path.join(project, run_name, 'metrics.json')

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
