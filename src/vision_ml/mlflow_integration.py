"""Production-grade MLflow integration with DagsHub for experiment tracking."""

import os
import json
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import dagshub

from .logging import get_logger

logger = get_logger(__name__)


class MLflowManager:
    """Centralized MLflow management for DagsHub integration."""

    def __init__(self, repo_name: str = "vision-ml-system", owner: Optional[str] = None):
        """Initialize MLflow with DagsHub.
        
        Args:
            repo_name: DagsHub repository name
            owner: DagsHub username (defaults to env var DAGSHUB_USERNAME)
        """
        self.repo_name = repo_name
        self.owner = owner or os.getenv('DAGSHUB_USERNAME', 'your_username')
        self.client = None
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow with DagsHub as remote tracking server."""
        try:
            # Initialize DagsHub (handles auth automatically)
            dagshub.init(
                repo=self.repo_name,
                owner=self.owner,
                mlflow=True
            )
            
            self.client = MlflowClient()
            tracking_uri = mlflow.get_tracking_uri()
            logger.info(f"✅ MLflow initialized with DagsHub")
            logger.info(f"   Tracking URI: {tracking_uri}")
            
        except Exception as e:
            logger.warning(f"⚠️ DagsHub initialization failed: {e}")
            logger.warning("   Falling back to local MLflow tracking")
            mlflow.set_tracking_uri("./mlruns")
            self.client = MlflowClient()

    def create_experiment(
        self,
        experiment_name: str,
        tags: Optional[Dict[str, str]] = None,
        artifact_location: Optional[str] = None
    ) -> str:
        """Create or get experiment.
        
        Args:
            experiment_name: Hierarchical name (e.g., "retail_analytics/person_detection/yolo_v2")
            tags: Experiment tags for filtering
            artifact_location: Custom artifact location
            
        Returns:
            Experiment ID
        """
        try:
            exp = self.client.get_experiment_by_name(experiment_name)
            if exp:
                return exp.experiment_id
        except:
            pass
        
        exp_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location
        )
        
        if tags:
            for key, value in tags.items():
                self.client.set_experiment_tag(exp_id, key, value)
        
        logger.info(f"✅ Created experiment: {experiment_name}")
        return exp_id

    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Start MLflow run with automatic tagging.
        
        Args:
            experiment_name: Experiment name
            run_name: Optional run name
            tags: Run tags (metadata)
            params: Hyperparameters
            
        Returns:
            MLflow run context
        """
        mlflow.set_experiment(experiment_name)
        
        run = mlflow.start_run(run_name=run_name)
        
        # Log standard tags
        standard_tags = {
            "timestamp": datetime.now().isoformat(),
            "environment": os.getenv('ENV', 'development'),
            "user": os.getenv('USER', 'unknown'),
        }
        
        if tags:
            standard_tags.update(tags)
        
        for key, value in standard_tags.items():
            mlflow.set_tag(key, value)
        
        # Log parameters
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        logger.info(f"✅ Started run: {run.info.run_id}")
        return run

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Training step/epoch
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of param_name -> value
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        framework: str = "pytorch"
    ):
        """Log model to MLflow.
        
        Args:
            model: PyTorch model
            artifact_path: Path in artifact store
            framework: Model framework (pytorch, sklearn, etc.)
        """
        if framework == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path)
        else:
            mlflow.sklearn.log_model(model, artifact_path)
        
        logger.info(f"✅ Logged model to {artifact_path}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact (file or directory) to MLflow.
        
        Args:
            local_path: Local file or directory path
            artifact_path: Destination path in artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"✅ Logged artifact: {local_path}")

    def log_dict(self, data: Dict[str, Any], filename: str):
        """Log dictionary as JSON artifact.
        
        Args:
            data: Dictionary to log
            filename: Output filename (e.g., "metrics.json")
        """
        fd, temp_path = tempfile.mkstemp(suffix=f"_{filename}")
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            mlflow.log_artifact(temp_path)
        finally:
            os.remove(temp_path)

    def end_run(self, status: str = "FINISHED"):
        """End current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)
        logger.info(f"✅ Ended run with status: {status}")

    def register_model(
        self,
        run_id: str,
        model_name: str,
        stage: str = "Staging"
    ) -> int:
        """Register model in MLflow Model Registry.
        
        Args:
            run_id: Run ID containing the model
            model_name: Model name in registry
            stage: Initial stage (Staging, Production, Archived)
            
        Returns:
            Model version number
        """
        model_uri = f"runs:/{run_id}/model"
        
        try:
            # Register model
            model_version = mlflow.register_model(model_uri, model_name)
            version = model_version.version
            logger.info(f"✅ Registered model: {model_name} v{version}")
        except Exception as e:
            logger.warning(f"Model already registered: {e}")
            # Get latest version
            versions = self.client.get_latest_versions(model_name)
            version = max([v.version for v in versions])
        
        # Transition to stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"✅ Transitioned model to {stage}")
        
        return version

    def compare_models(
        self,
        model_name: str,
        metric_name: str = "val_loss"
    ) -> Dict[str, Any]:
        """Compare models across stages.
        
        Args:
            model_name: Model name in registry
            metric_name: Metric to compare on
            
        Returns:
            Comparison dictionary
        """
        stages = ["Production", "Staging"]
        comparison = {}
        
        for stage in stages:
            try:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                if versions:
                    v = versions[0]
                    run = self.client.get_run(v.run_id)
                    metric_value = run.data.metrics.get(metric_name, None)
                    
                    comparison[stage] = {
                        "version": v.version,
                        "run_id": v.run_id,
                        metric_name: metric_value,
                        "created_at": v.creation_timestamp
                    }
            except:
                pass
        
        return comparison

    def promote_model(
        self,
        model_name: str,
        from_stage: str = "Staging",
        to_stage: str = "Production"
    ):
        """Promote model between stages.
        
        Args:
            model_name: Model name in registry
            from_stage: Source stage
            to_stage: Destination stage
        """
        versions = self.client.get_latest_versions(model_name, stages=[from_stage])
        
        if not versions:
            logger.error(f"No model found in {from_stage} stage")
            return
        
        version = versions[0].version
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=to_stage
        )
        
        logger.info(f"✅ Promoted {model_name} v{version} to {to_stage}")

    def get_experiment_runs(
        self,
        experiment_name: str,
        limit: int = 50,
        order_by: str = "metrics.val_loss ASC"
    ) -> List[Dict[str, Any]]:
        """Get runs from an experiment.
        
        Args:
            experiment_name: Experiment name
            limit: Maximum number of runs
            order_by: Order by clause (e.g., "metrics.val_loss ASC")
            
        Returns:
            List of run data
        """
        exp = self.client.get_experiment_by_name(experiment_name)
        
        if not exp:
            logger.warning(f"Experiment not found: {experiment_name}")
            return []
        
        runs = self.client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[order_by],
            max_results=limit
        )
        
        runs_data = []
        for run in runs:
            runs_data.append({
                "run_id": run.info.run_id,
                "status": run.info.status,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            })
        
        return runs_data

    def get_best_run(
        self,
        experiment_name: str,
        metric_name: str = "val_loss",
        mode: str = "min"
    ) -> Optional[Dict[str, Any]]:
        """Get best run from experiment.
        
        Args:
            experiment_name: Experiment name
            metric_name: Metric to optimize
            mode: "min" or "max"
            
        Returns:
            Best run data or None
        """
        runs = self.get_experiment_runs(experiment_name)
        
        if not runs:
            return None
        
        if mode == "min":
            best = min(runs, key=lambda r: r["metrics"].get(metric_name, float('inf')))
        else:
            best = max(runs, key=lambda r: r["metrics"].get(metric_name, float('-inf')))
        
        return best


class ExperimentTracker:
    """High-level experiment tracking context manager."""

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize experiment tracker.
        
        Args:
            experiment_name: Experiment name
            run_name: Optional run name
            tags: Run tags
            params: Hyperparameters
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.params = params or {}
        self.manager = MLflowManager()
        self.run = None

    def __enter__(self):
        """Start MLflow run."""
        self.run = self.manager.start_run(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            tags=self.tags,
            params=self.params,
        )
        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run."""
        status = "FAILED" if exc_type else "FINISHED"
        self.manager.end_run(status=status)
        
        if exc_type:
            logger.error(f"Run failed: {exc_val}")
            return False
        
        return True
