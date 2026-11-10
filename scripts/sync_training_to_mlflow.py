"""Sync training epoch metrics and artifacts from results.csv / results.png to MLflow/DagsHub.
"""

import os
import re
import csv
from pathlib import Path
import mlflow
from dotenv import load_dotenv

load_dotenv()

def sanitize_key(k: str) -> str:
    # Convert slashes and special chars to underscore
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', k.replace('/', '_').replace('-', '_'))
    return re.sub(r'_+', '_', clean).strip('_')

def sync_run(
    run_dir: str = "runs/detect/runs/train/yolo11n_roboflow_v1",
    experiment_name: str = "visitor_analytics_training",
    run_name: str = "yolo11n_roboflow_v1",
):
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    csv_path = run_path / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"results.csv not found in: {run_dir}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN") or os.getenv("DAGSHUB_USER_TOKEN")
    user = os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("DAGSHUB_USERNAME")

    if token:
        os.environ["DAGSHUB_USER_TOKEN"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    if user:
        os.environ["DAGSHUB_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_USERNAME"] = user

    print(f"[MLflow Sync] Connecting to tracking URI: {tracking_uri or 'local'}")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Find existing run or create new
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    existing_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    ) if experiment else []

    target_run_id = existing_runs[0].info.run_id if existing_runs else None

    with mlflow.start_run(run_id=target_run_id, run_name=run_name) as run:
        print(f"[MLflow Sync] Syncing metrics to run ID: {run.info.run_id}")

        # 1. Log epoch-by-epoch loss & mAP metrics
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                epoch_str = row.get("epoch", "").strip()
                if not epoch_str:
                    continue
                epoch = int(epoch_str)
                metrics = {}
                for k, v in row.items():
                    k_clean = k.strip()
                    v_clean = v.strip()
                    if k_clean in ("epoch", "time") or not v_clean:
                        continue
                    try:
                        metrics[sanitize_key(k_clean)] = float(v_clean)
                    except ValueError:
                        pass
                if metrics:
                    mlflow.log_metrics(metrics, step=epoch)
                    count += 1
            print(f"[MLflow Sync] Logged {count} epochs of loss & mAP metrics.")

        # 2. Log image artifacts (loss curves, confusion matrix, etc.)
        for img_name in [
            "results.png",
            "BoxF1_curve.png",
            "BoxPR_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
        ]:
            img_file = run_path / img_name
            if img_file.is_file():
                mlflow.log_artifact(str(img_file), artifact_path="plots")
                print(f"[MLflow Sync] Uploaded plot artifact: {img_name}")

        # 3. Log model weights
        best_pt = run_path / "weights" / "best.pt"
        if best_pt.is_file():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
            print(f"[MLflow Sync] Uploaded model weights: {best_pt}")

    print("[MLflow Sync] All loss curves and artifacts synced successfully!")

if __name__ == "__main__":
    sync_run()
