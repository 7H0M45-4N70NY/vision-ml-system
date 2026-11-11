"""Register the best fine-tuned model into the MLflow Model Registry on DagsHub.
"""

import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

def register_best_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN")
    user = os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("DAGSHUB_USERNAME")

    if token:
        os.environ["DAGSHUB_USER_TOKEN"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    if user:
        os.environ["DAGSHUB_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_USERNAME"] = user

    print(f"[Model Registry] Connecting to: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment_name = "visitor_analytics_training"
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.mlflow.runName = 'yolo11n_roboflow_v1'",
        max_results=1,
    )
    if not runs:
        # Fallback to the latest run in the experiment
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)

    if not runs:
        raise ValueError("No training runs found in MLflow")

    run = runs[0]
    run_id = run.info.run_id
    run_name = run.data.tags.get("mlflow.runName", "run")
    print(f"[Model Registry] Selected run: {run_name} (ID: {run_id})")

    model_name = "VisionFlow-YOLOv11-Edge"
    description = (
        "Fine-tuned YOLOv11 person detector produced by the VisionFlow Active Learning Flywheel. "
        "Trained locally on RTX 4050 GPU with RF-DETR teacher pseudo-labels (mAP50: 0.671)."
    )

    try:
        client.create_registered_model(model_name, description=description)
        print(f"[Model Registry] Created registered model: '{model_name}'")
    except Exception as e:
        print(f"[Model Registry] Registered model '{model_name}' already exists ({e})")

    source_uri = f"runs:/{run_id}/weights"
    print(f"[Model Registry] Creating model version from: {source_uri}")
    mv = client.create_model_version(
        name=model_name,
        source=source_uri,
        run_id=run_id,
        description="Version 1: Trained on 66 hard cases auto-captured and triaged with RF-DETR teacher.",
    )
    print(f"[Model Registry] Successfully registered '{model_name}' Version {mv.version}!")

    # Set champion / production alias
    try:
        client.set_registered_model_alias(model_name, "champion", mv.version)
        print(f"[Model Registry] Set alias '@champion' -> Version {mv.version}")
    except Exception as e:
        pass

    try:
        client.set_model_version_tag(model_name, mv.version, "framework", "ultralytics-yolo11")
        client.set_model_version_tag(model_name, mv.version, "precision", "fp16")
        client.set_model_version_tag(model_name, mv.version, "hardware_target", "rtx4050-ada")
        print(f"[Model Registry] Tagged version {mv.version} metadata.")
    except Exception:
        pass

    print(f"[Model Registry] Done! View Models at: {tracking_uri.replace('.mlflow', '')}/models")

if __name__ == "__main__":
    register_best_model()

