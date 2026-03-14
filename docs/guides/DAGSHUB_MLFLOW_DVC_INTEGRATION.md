# 🚀 DagsHub + MLflow + DVC Integration Guide

**Production-Grade ML Experiment Tracking & Data Versioning for Vision ML System**

---

## 📋 Overview

This document outlines how to integrate **DagsHub** as the central hub for:
- **MLflow Tracking**: Remote experiment tracking server
- **DVC Integration**: Data versioning and pipeline management
- **Model Registry**: Centralized model versioning and promotion
- **Team Collaboration**: Shared access control and audit trails

### Architecture
```
Local Development
    ├── Training Code (PyTorch + MLflow logging)
    ├── DVC Pipeline (data prep, training, evaluation)
    └── Git Repository
         ↓
DagsHub Remote (Cloud)
    ├── MLflow Tracking Server (https://dagshub.com/user/repo.mlflow)
    ├── DVC Remote Storage (S3-compatible)
    ├── Model Registry (versioning, staging, production)
    ├── Git Repository (code + .dvc files)
    └── Team Access Control
         ↓
Dashboard UI (Streamlit)
    ├── MLflow Experiments View
    ├── Training Pipeline Status
    ├── Model Comparison & Promotion
    └── DVC Pipeline Visualization
```

---

## 🔧 Setup Instructions

### **Step 1: Create DagsHub Account & Repository**

1. Go to https://dagshub.com
2. Sign up or log in
3. Create new repository: `vision-ml-system`
4. Copy repository URL: `https://dagshub.com/YOUR_USERNAME/vision-ml-system`

### **Step 2: Initialize DagsHub in Your Project**

```bash
# Install DagsHub Python client
pip install dagshub

# Initialize DagsHub in your project
dagshub init --repo vision-ml-system --owner YOUR_USERNAME
```

This creates a `.dagshub` directory with configuration.

### **Step 3: Configure MLflow with DagsHub**

Create `src/vision_ml/mlflow_config.py`:

```python
import os
import dagshub
import mlflow

def setup_mlflow_dagshub():
    """Initialize MLflow with DagsHub as remote tracking server."""
    
    # Initialize DagsHub (handles auth automatically)
    dagshub.init(
        repo='vision-ml-system',
        owner=os.getenv('DAGSHUB_USERNAME'),
        mlflow=True
    )
    
    # MLflow URI is automatically set by dagshub.init()
    tracking_uri = mlflow.get_tracking_uri()
    print(f"✅ MLflow Tracking URI: {tracking_uri}")
    
    return mlflow

def log_experiment(experiment_name, tags=None, params=None):
    """Start MLflow experiment with DagsHub."""
    
    mlflow.set_experiment(experiment_name)
    
    if tags:
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    if params:
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    return mlflow.active_run()

def log_metrics(metrics_dict, step=None):
    """Log metrics to MLflow."""
    for key, value in metrics_dict.items():
        mlflow.log_metric(key, value, step=step)

def log_model(model, artifact_path="model"):
    """Log PyTorch model to MLflow."""
    mlflow.pytorch.log_model(model, artifact_path)

def register_model(run_id, model_name, stage="Staging"):
    """Register model in DagsHub Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, model_name)
    
    # Transition to stage (Staging, Production, Archived)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage=stage
    )
```

### **Step 4: Configure DVC with DagsHub**

```bash
# Initialize DVC
dvc init

# Configure DVC to use DagsHub as remote storage
dvc remote add -d dagshub s3://dagshub/YOUR_USERNAME/vision-ml-system

# Set DagsHub credentials
dvc remote modify dagshub access_key_id YOUR_DAGSHUB_USERNAME
dvc remote modify dagshub secret_access_key YOUR_DAGSHUB_TOKEN
```

Or manually edit `.dvc/config`:

```ini
['remote "dagshub"']
    url = s3://dagshub/YOUR_USERNAME/vision-ml-system
    access_key_id = YOUR_DAGSHUB_USERNAME
    secret_access_key = YOUR_DAGSHUB_TOKEN
```

### **Step 5: Set Environment Variables**

Create `.env` file (add to `.gitignore`):

```bash
# DagsHub credentials
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_access_token
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_access_token

# MLflow configuration
MLFLOW_TRACKING_URI=https://dagshub.com/your_username/vision-ml-system.mlflow
```

Load in your training script:

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Credentials are now available
dagshub_username = os.getenv('DAGSHUB_USERNAME')
dagshub_token = os.getenv('DAGSHUB_TOKEN')
```

---

## 📊 MLflow Integration Patterns

### **Pattern 1: Hierarchical Experiment Organization**

```python
import mlflow
from src.vision_ml.mlflow_config import setup_mlflow_dagshub, log_experiment

setup_mlflow_dagshub()

# Experiment naming: project/task/variant
experiment_name = "retail_analytics/person_detection/yolo_v2_quantized"

with mlflow.start_run(experiment_name=experiment_name) as run:
    # Log metadata tags
    mlflow.set_tag("team", "ml-platform")
    mlflow.set_tag("project", "retail_analytics")
    mlflow.set_tag("model_type", "object_detection")
    mlflow.set_tag("dataset_version", "v2.3")
    mlflow.set_tag("production_ready", "false")
    
    # Log hyperparameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 100)
    mlflow.log_param("optimizer", "adam")
    
    # Training loop
    for epoch in range(100):
        loss = train_epoch(model, train_loader)
        val_loss = validate(model, val_loader)
        
        # Log metrics with step
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        # Log model checkpoint every 10 epochs
        if epoch % 10 == 0:
            mlflow.pytorch.log_model(model, f"checkpoints/epoch_{epoch}")
    
    # Log final model
    mlflow.pytorch.log_model(model, "model")
    
    # Log artifacts (config, plots, etc.)
    mlflow.log_artifact("config/training/base.yaml")
    mlflow.log_artifact("plots/training_curves.png")
    
    run_id = run.info.run_id
    print(f"✅ Run logged: {run_id}")
```

### **Pattern 2: Hyperparameter Sweep with Nested Runs**

```python
import mlflow
from itertools import product

setup_mlflow_dagshub()

# Parent run for the sweep
with mlflow.start_run(experiment_name="hyperparameter_sweep") as parent_run:
    mlflow.set_tag("sweep_type", "grid_search")
    
    # Hyperparameter grid
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64],
        "dropout": [0.2, 0.5]
    }
    
    best_val_loss = float('inf')
    best_params = None
    
    # Grid search
    for lr, bs, dropout in product(*param_grid.values()):
        with mlflow.start_run(nested=True) as child_run:
            # Log parameters
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", bs)
            mlflow.log_param("dropout", dropout)
            
            # Train with these parameters
            model = create_model(dropout=dropout)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            val_loss = train_and_evaluate(
                model, optimizer, train_loader, val_loader, batch_size=bs
            )
            
            mlflow.log_metric("val_loss", val_loss)
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {"lr": lr, "bs": bs, "dropout": dropout}
    
    # Log best parameters to parent run
    mlflow.log_param("best_learning_rate", best_params["lr"])
    mlflow.log_param("best_batch_size", best_params["bs"])
    mlflow.log_param("best_dropout", best_params["dropout"])
    mlflow.log_metric("best_val_loss", best_val_loss)
```

### **Pattern 3: Model Comparison & Promotion**

```python
import mlflow
from mlflow.tracking import MlflowClient

setup_mlflow_dagshub()

client = MlflowClient()

# Get all runs for an experiment
experiment = client.get_experiment_by_name("retail_analytics/person_detection")
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

# Find best model by val_loss
best_run = min(runs, key=lambda r: r.data.metrics.get("val_loss", float('inf')))

print(f"Best run: {best_run.info.run_id}")
print(f"Best val_loss: {best_run.data.metrics['val_loss']}")

# Register best model
model_name = "yolo_retail_detector"
model_uri = f"runs:/{best_run.info.run_id}/model"

try:
    mlflow.register_model(model_uri, model_name)
except Exception as e:
    print(f"Model already registered: {e}")

# Transition to Staging
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# Compare with Production model
try:
    prod_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    staging_version = client.get_latest_versions(model_name, stages=["Staging"])[0]
    
    print(f"Production: v{prod_version.version} - {prod_version.run_id}")
    print(f"Staging: v{staging_version.version} - {staging_version.run_id}")
    
    # If staging is better, promote
    staging_metrics = client.get_run(staging_version.run_id).data.metrics
    prod_metrics = client.get_run(prod_version.run_id).data.metrics
    
    if staging_metrics.get("val_loss") < prod_metrics.get("val_loss"):
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Production"
        )
        print("✅ Model promoted to Production!")
except:
    # First time, promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Production"
    )
    print("✅ Model registered and promoted to Production!")
```

---

## 📦 DVC Integration Patterns

### **Pattern 1: Data Versioning**

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python scripts/prepare_data.py
    deps:
      - data/raw/
    outs:
      - data/prepared/:
          cache: true
  
  train:
    cmd: python scripts/train.py
    deps:
      - data/prepared/
      - src/
    params:
      - training.learning_rate
      - training.batch_size
    outs:
      - models/model.pt:
          cache: true
    metrics:
      - metrics.json:
          cache: false
```

### **Pattern 2: Push/Pull Data to DagsHub**

```bash
# Add data to DVC
dvc add data/prepared/

# Push to DagsHub
dvc push

# Pull from DagsHub
dvc pull

# Check data status
dvc status
```

### **Pattern 3: Dataset Versioning**

```bash
# Tag dataset version
git tag -a dataset-v1.0 -m "Initial dataset"
dvc push

# Later, update dataset
dvc add data/prepared/
git add data/prepared/.gitignore
git commit -m "Update dataset with new samples"
git tag -a dataset-v1.1 -m "Added 100 new samples"
dvc push
```

---

## 🎯 Dashboard Integration

### **MLflow Experiments Page** (`pages/5_mlflow_experiments.py`)

```python
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

st.title("📈 MLflow Experiments")

client = MlflowClient()

# Get all experiments
experiments = client.search_experiments()

selected_exp = st.selectbox(
    "Select Experiment",
    options=[e.name for e in experiments]
)

if selected_exp:
    exp = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    
    # Display runs table
    runs_data = []
    for run in runs:
        runs_data.append({
            "Run ID": run.info.run_id[:8],
            "Status": run.info.status,
            "Train Loss": run.data.metrics.get("train_loss", "N/A"),
            "Val Loss": run.data.metrics.get("val_loss", "N/A"),
            "LR": run.data.params.get("learning_rate", "N/A"),
            "Batch Size": run.data.params.get("batch_size", "N/A"),
        })
    
    df = pd.DataFrame(runs_data)
    st.dataframe(df, use_container_width=True)
    
    # Best run
    best_run = min(runs, key=lambda r: r.data.metrics.get("val_loss", float('inf')))
    st.metric("Best Val Loss", best_run.data.metrics.get("val_loss"))
```

---

## 🔐 Security Best Practices

1. **Never commit credentials** - Use `.env` and `.gitignore`
2. **Use access tokens** - Create DagsHub token instead of password
3. **Rotate tokens regularly** - Update every 90 days
4. **Audit logs** - Check DagsHub audit trail for access
5. **Team access control** - Use DagsHub's role-based access

---

## 📚 References

- [DagsHub MLflow Integration](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
- [DagsHub DVC Integration](https://dagshub.com/docs/integration_guide/dvc/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [DVC Documentation](https://dvc.org/doc)

