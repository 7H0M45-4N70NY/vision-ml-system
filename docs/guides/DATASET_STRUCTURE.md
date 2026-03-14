# 📊 Dataset Structure & DVC Layout

---

## Overview

The dataset is organized for:
- **Training**: Curated labeled frames (real or high-quality synthetic)
- **Validation**: Held-out test set
- **Drift Simulation**: Synthetic variations (lighting, shelf rearrangement, camera angle)
- **Versioning**: DVC snapshots for reproducibility

---

## Directory Layout

```
data/
├── .dvc/                              # DVC configuration
├── .gitignore                         # Ignore large files
│
├── raw/                               # Original data (DVC-tracked)
│   ├── train/
│   │   ├── images/
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   └── ... (N images)
│   │   └── labels/
│   │       ├── 0001.txt               # YOLO format
│   │       ├── 0002.txt
│   │       └── ... (N labels)
│   │
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── test/
│       ├── images/
│       └── labels/
│
├── processed/                         # Augmented data (DVC-tracked)
│   ├── train_augmented/
│   │   ├── images/
│   │   └── labels/
│   └── val_augmented/
│       ├── images/
│       └── labels/
│
├── drift/                             # Drift simulation datasets
│   ├── lighting_change/               # Simulated lighting variations
│   │   ├── v1/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── v2/
│   │       ├── images/
│   │       └── labels/
│   │
│   ├── shelf_rearrangement/           # Simulated shelf layout changes
│   │   ├── v1/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── v2/
│   │       ├── images/
│   │       └── labels/
│   │
│   └── camera_angle_shift/            # Simulated camera angle changes
│       ├── v1/
│       │   ├── images/
│       │   └── labels/
│       └── v2/
│           ├── images/
│           └── labels/
│
├── metadata/                          # Dataset metadata
│   ├── class_names.txt                # Class definitions
│   ├── train_split.txt                # Training set indices
│   ├── val_split.txt                  # Validation set indices
│   ├── test_split.txt                 # Test set indices
│   └── drift_versions.yaml            # Drift dataset versions
│
└── README.md                          # Dataset documentation
```

---

## YOLO Format Specification

Each image has a corresponding `.txt` label file with the same name.

### Label Format
```
<class_id> <x_center> <y_center> <width> <height>
```

**Example** (`0001.txt`):
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
10 0.8 0.7 0.1 0.15
```

**Explanation**:
- `0` = Person (class 0)
- `0.5 0.5` = Bounding box center at 50%, 50% of image
- `0.3 0.4` = Width 30%, height 40% of image
- Values are normalized to [0, 1]

---

## Class Definitions

### `data/metadata/class_names.txt`
```
person
product_category_1
product_category_2
product_category_3
product_category_4
product_category_5
product_category_6
product_category_7
product_category_8
product_category_9
product_category_10
```

**Total**: 11 classes (1 person + 10 product categories)

---

## Dataset Statistics

### Training Set
- **Images**: ~1,000 (curated, high-quality)
- **Annotations**: ~5,000 objects
- **Average objects per image**: 5
- **Resolution**: 640×640 (normalized)

### Validation Set
- **Images**: ~200
- **Annotations**: ~1,000 objects

### Test Set
- **Images**: ~200
- **Annotations**: ~1,000 objects

---

## DVC Configuration

### Initialize DVC
```bash
cd data/
dvc init
```

### Track Raw Data
```bash
dvc add raw/
dvc add processed/
dvc add drift/
```

### Create `.dvc` Files
```
raw.dvc
processed.dvc
drift.dvc
```

### Remote Storage (Optional)
```bash
# Configure S3 remote
dvc remote add -d myremote s3://my-bucket/vision-ml-data

# Push to remote
dvc push
```

---

## Drift Simulation Strategy

### 1. Lighting Change Dataset
**Simulation Method**: Adjust brightness, contrast, and color temperature

**Versions**:
- `v1`: 20% brightness reduction
- `v2`: 40% brightness reduction
- `v3`: Color temperature shift (warm/cool)

**Purpose**: Test model robustness to lighting variations

**Expected Behavior**: Confidence scores drop, detection rate decreases

---

### 2. Shelf Rearrangement Dataset
**Simulation Method**: Reposition product locations, change shelf layout

**Versions**:
- `v1`: 25% product position shift
- `v2`: 50% product position shift
- `v3`: Complete shelf reorganization

**Purpose**: Test model generalization to layout changes

**Expected Behavior**: Category detection stable, but localization accuracy decreases

---

### 3. Camera Angle Shift Dataset
**Simulation Method**: Apply perspective transformation, simulate camera tilt

**Versions**:
- `v1`: 5° camera tilt
- `v2`: 10° camera tilt
- `v3`: 15° camera tilt

**Purpose**: Test model robustness to viewpoint changes

**Expected Behavior**: Confidence scores drop, especially for occluded objects

---

## Data Versioning Workflow

### Create New Dataset Version
```bash
# Generate drift data
python scripts/generate_drift_data.py \
  --source data/raw/test \
  --output data/drift/lighting_change/v2 \
  --drift_type lighting \
  --intensity 0.4

# Track with DVC
dvc add data/drift/lighting_change/v2

# Commit to Git
git add data/drift/lighting_change/v2.dvc
git commit -m "Add lighting_change/v2 drift dataset"
```

### Switch Dataset Version
```bash
# Checkout specific version
dvc checkout data/drift/lighting_change/v1.dvc

# Or pull from remote
dvc pull data/drift/lighting_change/v2.dvc
```

---

## Metadata Files

### `data/metadata/drift_versions.yaml`
```yaml
drift_datasets:
  lighting_change:
    v1:
      intensity: 0.2
      description: "20% brightness reduction"
      created_date: "2024-02-01"
      num_images: 200
    v2:
      intensity: 0.4
      description: "40% brightness reduction"
      created_date: "2024-02-05"
      num_images: 200
    v3:
      intensity: 0.0
      color_temp_shift: 0.3
      description: "Color temperature shift (warm)"
      created_date: "2024-02-10"
      num_images: 200

  shelf_rearrangement:
    v1:
      position_shift: 0.25
      description: "25% product position shift"
      created_date: "2024-02-02"
      num_images: 200
    v2:
      position_shift: 0.5
      description: "50% product position shift"
      created_date: "2024-02-06"
      num_images: 200

  camera_angle_shift:
    v1:
      tilt_degrees: 5
      description: "5° camera tilt"
      created_date: "2024-02-03"
      num_images: 200
    v2:
      tilt_degrees: 10
      description: "10° camera tilt"
      created_date: "2024-02-07"
      num_images: 200
```

---

## Data Loading Utilities

### `src/vision_ml/utils/data_utils.py`
```python
class DataLoader:
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = data_dir
        self.split = split
        self.images = self._load_images()
        self.labels = self._load_labels()
    
    def _load_images(self) -> List[str]:
        """Load image paths from split directory"""
        image_dir = f"{self.data_dir}/{self.split}/images"
        return sorted(glob(f"{image_dir}/*.jpg"))
    
    def _load_labels(self) -> List[str]:
        """Load label paths from split directory"""
        label_dir = f"{self.data_dir}/{self.split}/labels"
        return sorted(glob(f"{label_dir}/*.txt"))
    
    def __getitem__(self, idx: int):
        """Get image and labels"""
        image = cv2.imread(self.images[idx])
        labels = self._parse_yolo_labels(self.labels[idx])
        return image, labels
    
    def _parse_yolo_labels(self, label_path: str) -> np.ndarray:
        """Parse YOLO format labels"""
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            boxes.append([class_id, x_center, y_center, width, height])
        
        return np.array(boxes)
```

---

## Data Augmentation Strategy

### Training Augmentation
- Random horizontal flip (50%)
- Random brightness adjustment (±20%)
- Random contrast adjustment (±20%)
- Random rotation (±10°)
- Random crop (80-100% of image)

### Validation/Test
- No augmentation (deterministic evaluation)

---

## Quality Assurance

### Data Validation Checklist
- ✅ All images exist and are readable
- ✅ All labels exist and match images
- ✅ All class IDs are valid (0-10)
- ✅ All bounding boxes are normalized [0, 1]
- ✅ No duplicate images
- ✅ No corrupted files

### Validation Script
```bash
python scripts/validate_dataset.py --data_dir data/raw
```

---

## Reproducibility

### Dataset Snapshot
```bash
# Create reproducible snapshot
dvc add data/raw/
git add data/raw.dvc
git commit -m "Dataset snapshot v1.0"

# Reproduce exact dataset
dvc pull data/raw.dvc
```

### Seed Management
```python
# Ensure deterministic data loading
np.random.seed(42)
torch.manual_seed(42)
```

---

## Storage Considerations

### Local Storage
- **Raw data**: ~2 GB
- **Processed data**: ~2 GB
- **Drift datasets**: ~1 GB
- **Total**: ~5 GB

### Remote Storage (S3/GCS)
- Use DVC remote for versioning
- Compress before upload
- Enable versioning on remote

---

## Future Enhancements

### Phase 2
- Automated data annotation
- Semi-supervised learning
- Active learning for hard examples

### Phase 3
- Streaming data ingestion
- Real-time data validation
- Automated quality checks

This structure ensures:
- ✅ Reproducible datasets
- ✅ Version control for data
- ✅ Drift simulation capability
- ✅ Scalable storage
- ✅ Clear data lineage
