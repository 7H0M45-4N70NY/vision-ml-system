# 🏗 Repository Architecture

---

## Directory Structure

```
vision-ml-system/
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata
│
├── config/                            # Configuration files
│   ├── training/
│   │   ├── base.yaml                  # Base training config
│   │   ├── yolo26.yaml                # YOLO26-specific config
│   │   └── hyperparameters.yaml       # Hyperparameter ranges
│   ├── inference/
│   │   ├── base.yaml                  # Base inference config
│   │   └── optimization.yaml          # ONNX/quantization config
│   └── mlflow/
│       └── config.yaml                # MLflow tracking config
│
├── data/                              # Data directory (DVC-managed)
│   ├── .dvc/                          # DVC metadata
│   ├── raw/
│   │   ├── train/
│   │   │   ├── images/                # Training images
│   │   │   └── labels/                # YOLO format annotations
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   ├── processed/
│   │   ├── train_augmented/           # Augmented training data
│   │   └── val_augmented/
│   └── drift/                         # Drift simulation datasets
│       ├── lighting_change/
│       ├── shelf_rearrangement/
│       └── camera_angle_shift/
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── vision_ml/
│   │   ├── __init__.py
│   │   ├── detection/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # BaseDetector interface
│   │   │   ├── yolo26.py              # YOLO26 implementation
│   │   │   └── preprocessing.py       # Image preprocessing
│   │   ├── tracking/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # BaseTracker interface
│   │   │   └── bytetrack.py           # ByteTrack implementation
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py             # PyTorch training loop
│   │   │   ├── losses.py              # Custom loss functions
│   │   │   ├── metrics.py             # Evaluation metrics
│   │   │   └── callbacks.py           # MLflow callbacks
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py            # End-to-end inference
│   │   │   ├── onnx_exporter.py       # ONNX export utilities
│   │   │   └── quantization.py        # Quantization strategies
│   │   ├── benchmarking/
│   │   │   ├── __init__.py
│   │   │   ├── latency.py             # Latency profiling
│   │   │   ├── throughput.py          # Throughput measurement
│   │   │   └── memory.py              # Memory profiling
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── config.py              # Config loading
│   │   │   ├── logging.py             # Structured logging
│   │   │   ├── reproducibility.py     # Seed management
│   │   │   └── data_utils.py          # Data loading helpers
│   │   └── monitoring/
│   │       ├── __init__.py
│   │       ├── drift.py               # Drift detection
│   │       └── metrics.py             # Prometheus metrics
│
├── scripts/                           # Executable scripts
│   ├── train.py                       # Training entry point
│   ├── inference.py                   # Inference entry point
│   ├── benchmark.py                   # Benchmarking suite
│   ├── export_onnx.py                 # ONNX export script
│   └── generate_drift_data.py         # Drift simulation
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_detection.py              # Detection tests
│   ├── test_tracking.py               # Tracking tests
│   ├── test_training.py               # Training pipeline tests
│   ├── test_inference.py              # Inference tests
│   ├── test_benchmarking.py           # Benchmark tests
│   └── fixtures/
│       ├── sample_images/             # Test images
│       └── sample_configs/            # Test configs
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_benchmarking.ipynb
│   └── 03_drift_simulation.ipynb
│
├── docs/                              # Documentation
│   ├── STRATEGY.md                    # Strategic vision
│   ├── REPO_ARCHITECTURE.md           # This file
│   ├── DATASET_STRUCTURE.md           # Data organization
│   ├── TRAINING_PIPELINE.md           # Training details
│   ├── ARCHITECTURE.md                # System architecture
│   ├── SCALING.md                     # Performance analysis
│   ├── ROADMAP.md                     # Phase 2+ features
│   └── DEVELOPMENT.md                 # Development guidelines
│
├── mlruns/                            # MLflow local storage
├── .dvc/                              # DVC configuration
├── .git/                              # Git repository
├── .gitignore                         # Git ignore rules
└── .python-version                    # Python version specification

```

---

## Module Organization Principles

### 1. Detection Module (`src/vision_ml/detection/`)

**Purpose**: Abstraction for object detection models

**Key Classes**:
- `BaseDetector` — Abstract interface
- `YOLO26Detector` — YOLO26 implementation
- `PreprocessingPipeline` — Image preprocessing

**Design**:
- Config-driven model loading
- Batch inference support
- ONNX export compatibility

---

### 2. Tracking Module (`src/vision_ml/tracking/`)

**Purpose**: Multi-object tracking with persistent IDs

**Key Classes**:
- `BaseTracker` — Abstract interface
- `ByteTrackTracker` — ByteTrack implementation

**Design**:
- Frame-to-frame association
- Dwell time computation
- Detection-to-tracker interface

---

### 3. Training Module (`src/vision_ml/training/`)

**Purpose**: PyTorch training pipeline with MLflow integration

**Key Classes**:
- `Trainer` — Main training loop
- `MetricsComputer` — Evaluation metrics
- `MLflowCallback` — Experiment tracking

**Design**:
- Config-driven hyperparameters
- Checkpoint management
- Reproducible training

---

### 4. Inference Module (`src/vision_ml/inference/`)

**Purpose**: Production inference pipeline

**Key Classes**:
- `InferencePipeline` — End-to-end pipeline
- `ONNXExporter` — ONNX export utilities
- `QuantizationStrategy` — Quantization options

**Design**:
- Batch processing
- ONNX optimization
- Latency optimization

---

### 5. Benchmarking Module (`src/vision_ml/benchmarking/`)

**Purpose**: Performance profiling and analysis

**Key Classes**:
- `LatencyProfiler` — Latency measurement
- `ThroughputMeasurer` — Throughput analysis
- `MemoryProfiler` — Memory usage tracking

**Design**:
- CPU vs GPU comparison
- Batch scaling analysis
- Resolution impact study

---

### 6. Utils Module (`src/vision_ml/utils/`)

**Purpose**: Shared utilities

**Key Functions**:
- `load_config()` — YAML config loading
- `set_seed()` — Reproducibility
- `setup_logging()` — Structured logging
- `DataLoader` — Data loading helpers

---

### 7. Monitoring Module (`src/vision_ml/monitoring/`)

**Purpose**: Drift detection and metrics

**Key Classes**:
- `DriftDetector` — Data/model drift detection
- `MetricsCollector` — Prometheus metrics

---

## Configuration Strategy

### Training Config (`config/training/base.yaml`)
```yaml
model:
  name: yolo26
  pretrained: true
  num_classes: 11  # 1 person + 10 product categories

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  scheduler: cosine

data:
  train_path: data/raw/train
  val_path: data/raw/val
  augmentation: true

mlflow:
  experiment_name: retail_analytics_v1
  tracking_uri: file:./mlruns
```

### Inference Config (`config/inference/base.yaml`)
```yaml
model:
  path: models/yolo26_best.pt
  device: cuda
  half_precision: false

inference:
  batch_size: 8
  confidence_threshold: 0.5
  iou_threshold: 0.45

onnx:
  export: true
  opset_version: 13
  optimize: true
```

---

## Dependency Management

### `requirements.txt`
```
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
opencv-python==4.8.0.74
mlflow==2.7.0
dvc==3.30.0
pyyaml==6.0
pytest==7.4.0
onnx==1.14.1
onnxruntime==1.16.0
```

---

## Development Workflow

### 1. Local Development
```bash
# Activate environment
conda activate ./venv

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run training
python scripts/train.py --config config/training/base.yaml
```

### 2. Experiment Tracking
```bash
# View MLflow UI
mlflow ui

# Check DVC status
dvc status
```

### 3. Benchmarking
```bash
# Run benchmarks
python scripts/benchmark.py --config config/inference/base.yaml
```

---

## Key Design Decisions

### 1. Modular Abstractions
- `BaseDetector`, `BaseTracker` interfaces
- Enables swapping implementations (YOLO → RF-DETR)
- Config-driven behavior

### 2. Separation of Concerns
- Detection ≠ Tracking ≠ Training ≠ Inference
- Each module has single responsibility
- Clear interfaces between modules

### 3. Config-Driven Architecture
- YAML configs for all hyperparameters
- No hardcoded values
- Reproducible experiments

### 4. MLflow Integration
- Automatic experiment tracking
- Model registry for versioning
- Artifact management

### 5. DVC for Data Versioning
- Dataset snapshots
- Drift simulation versions
- Reproducible data lineage

---

## Scalability Considerations

### Horizontal Scaling
- Multiprocessing for batch inference
- Worker pool architecture
- Frame batching strategies

### Vertical Scaling
- GPU utilization optimization
- ONNX export for inference
- Quantization for edge deployment

### Data Scaling
- DVC remote storage (S3, GCS)
- Distributed training (future)
- Streaming data ingestion (future)

---

## Testing Strategy

### Unit Tests
- Detection module tests
- Tracking module tests
- Utility function tests

### Integration Tests
- End-to-end pipeline tests
- Training + inference tests
- Config loading tests

### Performance Tests
- Benchmark regression tests
- Memory usage tests
- Latency SLA tests

---

## Documentation Standards

- **Code**: Docstrings for all public functions
- **Architecture**: Diagrams and explanations
- **Decisions**: ADRs (Architecture Decision Records)
- **Benchmarks**: Measured results with context
- **Reproducibility**: Seed management, version pinning

This structure ensures:
- ✅ Clarity and maintainability
- ✅ Scalability and extensibility
- ✅ Reproducibility and testability
- ✅ Interview-ready code quality
