# 🎯 Training Pipeline & MLflow Configuration

---

## Overview

The training pipeline is:
- **Config-driven** (YAML hyperparameters)
- **Reproducible** (seed management, version pinning)
- **Tracked** (MLflow experiment logging)
- **Checkpointed** (model snapshots)
- **Exportable** (ONNX format)

---

## Training Configuration

### `config/training/base.yaml`

```yaml
# Model Configuration
model:
  name: yolo26
  pretrained: true
  num_classes: 11
  backbone: yolov8n
  task: detect

# Training Hyperparameters
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  momentum: 0.937
  weight_decay: 0.0005
  optimizer: SGD
  scheduler: cosine
  warmup_epochs: 3
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1

# Data Configuration
data:
  train_path: data/raw/train
  val_path: data/raw/val
  test_path: data/raw/test
  image_size: 640
  augmentation: true
  
  # Augmentation parameters
  augment:
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4
    degrees: 10.0
    translate: 0.1
    scale: 0.5
    flipud: 0.0
    fliplr: 0.5
    mosaic: 1.0
    mixup: 0.0

# Reproducibility
reproducibility:
  seed: 42
  deterministic: true
  benchmark: false

# MLflow Configuration
mlflow:
  experiment_name: retail_analytics_v1
  run_name: yolo26_baseline
  tracking_uri: file:./mlruns
  registry_uri: file:./mlruns
  log_params: true
  log_metrics: true
  log_artifacts: true
  log_model: true

# Checkpoint Configuration
checkpoint:
  save_interval: 5
  save_best: true
  save_last: true
  checkpoint_dir: models/checkpoints

# Early Stopping
early_stopping:
  enabled: true
  patience: 10
  metric: val_loss
  mode: min

# Device Configuration
device:
  type: cuda
  device_id: 0
  mixed_precision: false
```

---

## Training Loop Architecture

### `src/vision_ml/training/trainer.py`

```python
class Trainer:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.device = self._setup_device()
        self.model = self._load_model()
        self.train_loader = self._load_data('train')
        self.val_loader = self._load_data('val')
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.mlflow_callback = MLflowCallback(self.config)
        
    def train(self):
        """Main training loop"""
        set_seed(self.config['reproducibility']['seed'])
        
        for epoch in range(self.config['training']['epochs']):
            # Training phase
            train_loss = self._train_epoch(epoch)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(epoch)
            
            # Logging
            self.mlflow_callback.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                **val_metrics
            }, epoch)
            
            # Checkpoint
            if (epoch + 1) % self.config['checkpoint']['save_interval'] == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if self._check_early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Scheduler step
            self.scheduler.step()
        
        # Export ONNX
        self._export_onnx()
        
        # Log final model
        self.mlflow_callback.log_model(self.model)
    
    def _train_epoch(self, epoch: int) -> float:
        """Single training epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self._compute_loss(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / len(self.train_loader)
    
    def _validate_epoch(self, epoch: int) -> Tuple[float, dict]:
        """Single validation epoch"""
        self.model.eval()
        total_loss = 0.0
        metrics = {}
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self._compute_loss(outputs, labels)
                total_loss += loss.item()
                
                # Compute metrics (mAP, precision, recall)
                batch_metrics = self._compute_metrics(outputs, labels)
                for key, value in batch_metrics.items():
                    metrics[key] = metrics.get(key, 0) + value
        
        # Average metrics
        num_batches = len(self.val_loader)
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return total_loss / num_batches, metrics
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_dir = self.config['checkpoint']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def _export_onnx(self):
        """Export model to ONNX format"""
        exporter = ONNXExporter(self.model, self.config)
        exporter.export(output_path="models/yolo26_best.onnx")
```

---

## MLflow Integration

### `src/vision_ml/training/callbacks.py`

```python
class MLflowCallback:
    def __init__(self, config: dict):
        self.config = config
        self.experiment_name = config['mlflow']['experiment_name']
        self.run_name = config['mlflow']['run_name']
        
        # Setup MLflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.experiment_name)
        
        # Start run
        self.run = mlflow.start_run(run_name=self.run_name)
        
        # Log config
        self._log_config()
    
    def _log_config(self):
        """Log all configuration parameters"""
        flat_config = self._flatten_dict(self.config)
        for key, value in flat_config.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics at each epoch"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model: torch.nn.Module):
        """Log trained model to MLflow"""
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="models",
            registered_model_name="yolo26_retail"
        )
    
    def _flatten_dict(self, d: dict, parent_key: str = '') -> dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()
```

---

## Metrics Computation

### `src/vision_ml/training/metrics.py`

```python
class MetricsComputer:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def compute_map(self, predictions: List[dict], ground_truth: List[dict]) -> float:
        """Compute mean Average Precision (mAP)"""
        # Implementation using torchvision.ops.box_iou
        pass
    
    def compute_precision(self, predictions: List[dict], ground_truth: List[dict]) -> float:
        """Compute precision"""
        pass
    
    def compute_recall(self, predictions: List[dict], ground_truth: List[dict]) -> float:
        """Compute recall"""
        pass
    
    def compute_f1(self, precision: float, recall: float) -> float:
        """Compute F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
```

---

## Training Entry Point

### `scripts/train.py`

```python
import argparse
from vision_ml.training.trainer import Trainer
from vision_ml.utils.config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create trainer
    trainer = Trainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    # End MLflow run
    trainer.mlflow_callback.end_run()

if __name__ == '__main__':
    main()
```

---

## Reproducibility Management

### `src/vision_ml/utils/reproducibility.py`

```python
def set_seed(seed: int):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_reproducibility_info() -> dict:
    """Get reproducibility metadata"""
    return {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
    }
```

---

## Configuration Loading

### `src/vision_ml/utils/config.py`

```python
import yaml

def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: dict, output_path: str):
    """Save configuration to YAML"""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Merge two configurations (override takes precedence)"""
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
```

---

## Training Workflow

### Step 1: Prepare Configuration
```bash
# Copy base config
cp config/training/base.yaml config/training/experiment_v1.yaml

# Edit hyperparameters
# vim config/training/experiment_v1.yaml
```

### Step 2: Run Training
```bash
python scripts/train.py --config config/training/experiment_v1.yaml
```

### Step 3: Monitor with MLflow
```bash
mlflow ui
# Open http://localhost:5000
```

### Step 4: Compare Experiments
```bash
# View all runs
mlflow runs list --experiment-name retail_analytics_v1

# Compare specific runs
mlflow runs compare <run_id_1> <run_id_2>
```

### Step 5: Register Best Model
```bash
# In MLflow UI, transition model to "Production"
# Or via API:
mlflow.register_model(
    model_uri="runs:/abc123/models",
    name="yolo26_retail"
)
```

---

## Hyperparameter Tuning

### `config/training/hyperparameters.yaml`

```yaml
# Hyperparameter ranges for grid/random search
search_space:
  learning_rate:
    type: loguniform
    min: 0.0001
    max: 0.01
  
  batch_size:
    type: choice
    values: [16, 32, 64]
  
  weight_decay:
    type: loguniform
    min: 0.0001
    max: 0.001
  
  warmup_epochs:
    type: choice
    values: [1, 3, 5]
  
  augment_scale:
    type: uniform
    min: 0.3
    max: 0.7

# Optimization strategy
optimization:
  method: random_search  # or grid_search, bayesian
  num_trials: 20
  timeout_hours: 48
```

---

## Model Registry & Versioning

### Model Promotion Workflow
```
Training Run
    ↓
Evaluate Metrics
    ↓
Register in MLflow
    ↓
Staging Environment
    ↓
Validate Performance
    ↓
Promote to Production
```

### MLflow Model Registry
```python
# Register model
mlflow.register_model(
    model_uri="runs:/abc123/models",
    name="yolo26_retail"
)

# Transition to staging
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="yolo26_retail",
    version=1,
    stage="Staging"
)

# Transition to production
client.transition_model_version_stage(
    name="yolo26_retail",
    version=1,
    stage="Production"
)
```

---

## Training Checkpoints

### Checkpoint Structure
```python
checkpoint = {
    'epoch': 25,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'config': config,
    'metrics': {
        'train_loss': 0.123,
        'val_loss': 0.145,
        'val_mAP': 0.87
    }
}
```

### Resume Training
```bash
python scripts/train.py \
  --config config/training/base.yaml \
  --resume models/checkpoints/checkpoint_epoch_25.pt
```

---

## Performance Monitoring

### Metrics to Track
- **train_loss**: Training loss per epoch
- **val_loss**: Validation loss per epoch
- **val_mAP**: Mean Average Precision
- **val_precision**: Detection precision
- **val_recall**: Detection recall
- **val_f1**: F1 score
- **learning_rate**: Current learning rate
- **epoch_time**: Time per epoch

### MLflow Dashboard
```
Experiment: retail_analytics_v1
├── Run 1: yolo26_baseline
│   ├── Params: lr=0.001, batch_size=32
│   ├── Metrics: val_mAP=0.87, val_loss=0.145
│   └── Artifacts: model.pt, config.yaml
├── Run 2: yolo26_augmented
│   ├── Params: lr=0.0005, batch_size=64
│   ├── Metrics: val_mAP=0.89, val_loss=0.132
│   └── Artifacts: model.pt, config.yaml
└── Run 3: yolo26_warmup
    ├── Params: lr=0.001, warmup_epochs=5
    ├── Metrics: val_mAP=0.88, val_loss=0.138
    └── Artifacts: model.pt, config.yaml
```

---

## Best Practices

### 1. Reproducibility
- ✅ Pin all dependency versions
- ✅ Set random seeds
- ✅ Use deterministic algorithms
- ✅ Log all hyperparameters

### 2. Experiment Tracking
- ✅ Log metrics at each epoch
- ✅ Save checkpoints regularly
- ✅ Track model artifacts
- ✅ Document decisions

### 3. Model Management
- ✅ Register best models
- ✅ Version models in registry
- ✅ Promote through stages
- ✅ Track model lineage

### 4. Code Quality
- ✅ Unit tests for training logic
- ✅ Integration tests for pipeline
- ✅ Validation of data loading
- ✅ Error handling and logging

---

## Troubleshooting

### Issue: Training Loss Not Decreasing
- **Check**: Learning rate (too high/low)
- **Check**: Data loading (corrupted labels)
- **Check**: Model initialization
- **Solution**: Reduce learning rate, validate data

### Issue: High Validation Loss
- **Check**: Overfitting (train loss << val loss)
- **Check**: Data distribution mismatch
- **Solution**: Add regularization, augmentation

### Issue: GPU Out of Memory
- **Solution**: Reduce batch size
- **Solution**: Reduce image resolution
- **Solution**: Use gradient accumulation

---

## Next Steps

1. ✅ Define training configuration
2. ✅ Implement training loop
3. ✅ Integrate MLflow
4. ✅ Set up checkpointing
5. ⏳ Run baseline training
6. ⏳ Hyperparameter tuning
7. ⏳ Model evaluation
8. ⏳ ONNX export
