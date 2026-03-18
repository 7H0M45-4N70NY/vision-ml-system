# 📖 Development Guidelines

---

## Code Organization Principles

### 1. Module Structure
- Each module has a clear responsibility
- Public API defined in `__init__.py`
- Implementation details in separate files
- Abstract base classes for interfaces

### 2. Naming Conventions
- Classes: `PascalCase` (e.g., `YOLO11Detector`)
- Functions: `snake_case` (e.g., `load_config`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_BATCH_SIZE`)
- Private methods: `_snake_case` (e.g., `_preprocess_image`)

### 3. Documentation Standards
- Docstrings for all public functions/classes
- Type hints for all parameters and returns
- Examples in docstrings where helpful
- Architecture decisions in comments

---

## Development Workflow

### 1. Setting Up Local Environment
```bash
# Clone repository
git clone <repo_url>
cd vision-ml-system

# Create virtual environment
conda create -p ./venv python=3.10
conda activate ./venv

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### 2. Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_detection.py

# Run with coverage
pytest tests/ --cov=src/vision_ml --cov-report=html

# Run specific test
pytest tests/test_detection.py::test_detector_initialization
```

### 3. Code Quality Checks
```bash
# Format code
black src/ tests/ scripts/

# Lint code
flake8 src/ tests/ scripts/

# Type checking
mypy src/
```

### 4. Running Training
```bash
# Train with base config
python scripts/train.py --config config/training/base.yaml

# Train with custom config
python scripts/train.py --config config/training/experiment_v1.yaml

# Resume from checkpoint
python scripts/train.py \
  --config config/training/base.yaml \
  --resume models/checkpoints/checkpoint_epoch_25.pt
```

### 5. Running Inference
```bash
# Inference on single image
python scripts/inference.py \
  --model models/yolo11n_best.pt \
  --image path/to/image.jpg

# Inference on video
python scripts/inference.py \
  --model models/yolo11n_best.pt \
  --video path/to/video.mp4

# Batch inference
python scripts/inference.py \
  --model models/yolo11n_best.pt \
  --image_dir path/to/images/
```

### 6. Benchmarking
```bash
# Run full benchmark suite
python scripts/benchmark.py --config config/inference/base.yaml

# Benchmark specific component
python scripts/benchmark.py \
  --config config/inference/base.yaml \
  --component detection

# Save benchmark results
python scripts/benchmark.py \
  --config config/inference/base.yaml \
  --output results/benchmark_results.json
```

---

## Git Workflow

### Branch Naming
- `feature/` — New features
- `bugfix/` — Bug fixes
- `refactor/` — Code refactoring
- `docs/` — Documentation updates
- `experiment/` — Experimental work

### Commit Messages
```
[TYPE] Brief description

Detailed explanation if needed.

- Bullet point 1
- Bullet point 2
```

**Types**:
- `[FEAT]` — New feature
- `[FIX]` — Bug fix
- `[REFACTOR]` — Code refactoring
- `[DOCS]` — Documentation
- `[TEST]` — Test additions
- `[PERF]` — Performance improvement

### Example Commits
```
[FEAT] Add YOLO11n detection module

- Implement BaseDetector interface
- Add YOLO11Detector class
- Add preprocessing pipeline
- Add unit tests

[FIX] Fix image normalization bug

Normalization was using wrong mean/std values.
Now using ImageNet standard values.

[DOCS] Update training pipeline documentation

- Add MLflow integration details
- Add checkpoint management guide
- Add hyperparameter tuning examples
```

---

## Testing Strategy

### Unit Tests
Test individual functions/classes in isolation.

```python
# tests/test_detection.py
def test_detector_initialization():
    detector = YOLO11Detector(model_name="yolo11n")
    assert detector.model_name == "yolo11n"

def test_preprocessing_pipeline():
    image = np.random.rand(480, 640, 3)
    preprocessor = PreprocessingPipeline(image_size=640)
    processed = preprocessor(image)
    assert processed.shape == (1, 3, 640, 640)
```

### Integration Tests
Test multiple components working together.

```python
# tests/test_training.py
def test_training_pipeline():
    config = load_config("config/training/test.yaml")
    trainer = Trainer(config)
    
    # Train for 1 epoch
    trainer.train_epoch(0)
    
    # Check metrics logged
    assert trainer.metrics['train_loss'] > 0
```

### Performance Tests
Test performance characteristics.

```python
# tests/test_benchmarking.py
def test_inference_latency():
    detector = YOLO11Detector()
    image = np.random.rand(480, 640, 3)
    
    start = time.time()
    for _ in range(100):
        detector.detect(image)
    elapsed = time.time() - start
    
    avg_latency = elapsed / 100
    assert avg_latency < 0.1  # < 100ms
```

### Test Configuration
```yaml
# config/training/test.yaml
model:
  name: yolo11n
  pretrained: false

training:
  epochs: 1
  batch_size: 4

data:
  train_path: tests/fixtures/data/train
  val_path: tests/fixtures/data/val
```

---

## Code Review Checklist

### Before Submitting PR
- ✅ Code follows style guide (black, flake8)
- ✅ Type hints added
- ✅ Docstrings written
- ✅ Tests passing
- ✅ No hardcoded values
- ✅ No debug prints
- ✅ Commit messages clear

### Reviewer Checklist
- ✅ Code is readable
- ✅ Logic is correct
- ✅ Tests are adequate
- ✅ No performance regressions
- ✅ Documentation updated
- ✅ Follows SOLID principles

---

## Debugging Tips

### 1. Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug(f"Variable value: {var}")
```

### 2. Use Breakpoints
```python
import pdb; pdb.set_trace()
```

### 3. Profile Code
```python
import cProfile
cProfile.run('trainer.train()')
```

### 4. Memory Profiling
```bash
pip install memory-profiler
python -m memory_profiler scripts/train.py
```

### 5. GPU Profiling
```bash
# NVIDIA GPU profiling
nvidia-smi
nvidia-smi dmon

# PyTorch profiler
from torch.profiler import profile, record_function
with profile(activities=[...]) as prof:
    model(input)
```

---

## Documentation Standards

### Docstring Format
```python
def detect(self, image: np.ndarray) -> List[Dict]:
    """
    Detect objects in image.
    
    Args:
        image: Input image (H, W, 3) in BGR format
    
    Returns:
        List of detections with keys:
        - class_id: int
        - confidence: float
        - bbox: [x1, y1, x2, y2]
    
    Example:
        >>> detector = YOLO11Detector()
        >>> image = cv2.imread("image.jpg")
        >>> detections = detector.detect(image)
    """
```

### README Format
- Project overview
- Quick start
- Architecture diagram
- Key features
- Documentation links
- Contributing guidelines
- License

### Architecture Decision Record (ADR)
```markdown
# ADR-001: Use YOLO11n for Detection

## Context
Need to choose object detection model for retail analytics.

## Decision
Use YOLO11n for detection.

## Rationale
- Fast inference (real-time capable)
- Accurate enough for retail use case
- Good community support
- Easy to fine-tune

## Consequences
- Need GPU for real-time inference
- Limited to 640x640 resolution
- May need quantization for edge deployment
```

---

## Performance Optimization Checklist

### Before Optimization
- ✅ Identify bottleneck (profiling)
- ✅ Measure baseline performance
- ✅ Set optimization target

### Optimization Strategies
1. **Batch Processing** — Process multiple images together
2. **ONNX Export** — Use ONNX Runtime for faster inference
3. **Quantization** — Reduce model precision (int8)
4. **Resolution Scaling** — Reduce input resolution
5. **Model Pruning** — Remove unnecessary weights
6. **Caching** — Cache preprocessing results

### After Optimization
- ✅ Measure new performance
- ✅ Verify accuracy not degraded
- ✅ Document optimization
- ✅ Update benchmarks

---

## Reproducibility Checklist

### Code Reproducibility
- ✅ Pin all dependency versions
- ✅ Set random seeds
- ✅ Use deterministic algorithms
- ✅ Document environment setup

### Data Reproducibility
- ✅ Version datasets with DVC
- ✅ Document data preprocessing
- ✅ Track data splits
- ✅ Validate data integrity

### Experiment Reproducibility
- ✅ Log all hyperparameters
- ✅ Save model checkpoints
- ✅ Track MLflow runs
- ✅ Document results

---

## Common Issues & Solutions

### Issue: CUDA Out of Memory
```python
# Solution 1: Reduce batch size
batch_size = 16  # instead of 32

# Solution 2: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
```

### Issue: Training Loss Not Decreasing
```python
# Check learning rate
print(optimizer.param_groups[0]['lr'])

# Reduce learning rate
optimizer.param_groups[0]['lr'] = 0.0001

# Check data loading
for images, labels in train_loader:
    print(images.shape, labels.shape)
    break
```

### Issue: Model Overfitting
```python
# Add regularization
weight_decay = 0.0005

# Add dropout
model.add_dropout(p=0.5)

# Add data augmentation
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
])
```

---

## Resources

### Official Documentation
- [PyTorch](https://pytorch.org/docs/)
- [MLflow](https://mlflow.org/docs/)
- [DVC](https://dvc.org/doc)
- [YOLO](https://docs.ultralytics.com/)

### Learning Materials
- [YOLO Paper](https://arxiv.org/abs/2301.11799)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [MLOps Best Practices](https://ml-ops.systems/)

### Tools
- [Weights & Biases](https://wandb.ai/) — Experiment tracking
- [Hugging Face](https://huggingface.co/) — Model hub
- [Kaggle](https://kaggle.com/) — Datasets

---

## Contact & Support

For questions or issues:
1. Check documentation
2. Search GitHub issues
3. Create new issue with details
4. Contact maintainers

---

This guide ensures:
- ✅ Consistent code quality
- ✅ Reproducible results
- ✅ Efficient development
- ✅ Easy collaboration
- ✅ Professional standards
