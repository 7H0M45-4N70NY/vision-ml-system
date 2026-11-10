# Hybrid Dual-Detector Modes

The Vision ML System supports three detector modes to balance **inference speed** and **detection accuracy**:

## Modes

### 1. Hot Path (`use_dual_detector: false` or `'hot'`)
**Purpose**: Maximum speed, minimal overhead.

- **Primary detector only** (YOLO)
- No secondary detector loaded
- No frame saving
- **Latency**: ~30-50ms per frame (CPU)
- **Use case**: Real-time webcam, live dashboards, speed-critical applications

**Config**:
```yaml
detection:
  use_dual_detector: false  # or 'hot'
  save_low_confidence_frames: false
```

---

### 2. Inline Mode (`use_dual_detector: 'inline'`)
**Purpose**: Real-time dual detection with immediate feedback.

- **Primary detector** (YOLO) runs first
- **Secondary detector** (RF-DETR) runs on low-confidence detections
- Merges high-confidence primary + non-overlapping secondary results
- Saves low-confidence frames for offline training
- **Latency**: ~100-200ms per frame (CPU, slower due to secondary)
- **Use case**: Accuracy-critical real-time inference, immediate drift detection

**Config**:
```yaml
detection:
  use_dual_detector: inline
  dual_confidence_threshold: 0.5  # Primary < 0.5 → check secondary
  save_low_confidence_frames: true
  low_confidence_dir: data/low_confidence_frames
```

---

### 3. Batch Mode (`use_dual_detector: 'batch'`)
**Purpose**: Fast inference + deferred secondary analysis.

- **Primary detector only** during inference (YOLO)
- Saves low-confidence frames to disk
- Run secondary detector **offline** via batch job
- **Inference latency**: ~30-50ms per frame (same as hot path)
- **Analysis latency**: Deferred (run batch job when convenient)
- **Use case**: High-throughput inference with offline quality analysis

**Config**:
```yaml
detection:
  use_dual_detector: batch
  dual_confidence_threshold: 0.5
  save_low_confidence_frames: true
  low_confidence_dir: data/low_confidence_frames
```

**Offline batch processing**:
```bash
python scripts/secondary_detector_batch.py \
  --input data/low_confidence_frames \
  --output data/secondary_analysis \
  --config config/inference/base.yaml
```

---

## Comparison

| Aspect | Hot | Inline | Batch |
|--------|-----|--------|-------|
| **Inference Speed** | Fastest | Slowest | Fast |
| **Secondary Detector** | No | Yes (real-time) | No (offline) |
| **Frame Saving** | No | Yes | Yes |
| **Drift Detection** | Primary only | Real-time | Deferred |
| **Use Case** | Speed-critical | Accuracy-critical | Balanced |

---

## Workflow Examples

### Example 1: Real-Time Webcam (Hot Path)
```yaml
detection:
  use_dual_detector: false
mode:
  type: online
  source: 0  # Webcam
```
**Result**: Fast, snappy inference. No secondary analysis.

---

### Example 2: Batch Video with Real-Time Dual Detection (Inline)
```yaml
detection:
  use_dual_detector: inline
mode:
  type: offline
  source: video.mp4
```
**Result**: Slower but accurate. Saves low-confidence frames for training.

---

### Example 3: High-Throughput Inference + Offline Analysis (Batch)
```yaml
detection:
  use_dual_detector: batch
mode:
  type: online
  source: rtsp://camera.local/stream
```

**Step 1**: Run inference (fast)
```bash
python scripts/inference.py --config config/inference/base.yaml
```

**Step 2**: Later, run secondary detector on saved frames
```bash
python scripts/secondary_detector_batch.py \
  --input data/low_confidence_frames \
  --output data/secondary_analysis
```

---

## Configuration Reference

```yaml
detection:
  # Detector types
  primary_detector: yolo11n        # Fast primary
  secondary_detector: rfdetr       # Accurate fallback

  # Mode selection
  use_dual_detector: batch         # false | 'hot' | 'inline' | 'batch'

  # Dual detector parameters
  dual_confidence_threshold: 0.5   # Primary < 0.5 → candidate failure frame
  dual_min_confidence_floor: 0.15  # Discard pure noise/hallucinations below 0.15

  # Active Learning Frame Sampling Controls (prevents dataset bloating & redundancy)
  save_low_confidence_frames: true # Save frames for analysis/training
  save_cooldown_frames: 30         # Minimum interval between captures (30 frames ≈ 1s at 30 FPS)
  max_saved_frames: 25             # Maximum frames saved per stream/video run
  similarity_threshold: 0.95       # Visual diversity filter (drops near-duplicate static postures)
  low_confidence_dir: data/low_confidence_frames
```

---

## Active Learning Frame Sampling Controls

Saving every single low-confidence frame from a continuous video stream is counterproductive: a 7-second occlusion event at 30 FPS produces **~210 near-identical frames**. This causes **data leakage** across train/val splits and forces models to overfit to specific background patterns.

To prevent dataset bloat and maximize informational entropy, `DualDetector` applies four active learning controls:

1. **Confidence Floor (`dual_min_confidence_floor: 0.15`):**
   - Detections with confidence $< 0.15$ are usually background hallucinations or camera glare.
   - Only borderline candidate failures ($0.15 \le \text{confidence} < 0.50$) are collected.
2. **Temporal Cooldown (`save_cooldown_frames: 30`):**
   - Enforces a minimum interval between saved frames ($\sim 1\text{ to }1.5\text{s}$). During prolonged occlusions, captures $\sim 3\text{--}5$ diverse moments instead of hundreds of duplicates.
3. **Max Run Budget (`max_saved_frames: 25`):**
   - Hard quota per video/stream session. Guarantees no single camera clip dominates your active learning buffer.
4. **Visual Similarity Filter (`similarity_threshold: 0.95`):**
   - Compares candidate frame thumbnails against the previously saved frame. If the scene hasn't changed by at least 5% (e.g. static camera, subject stopped moving), the redundant duplicate is dropped.

---

## When to Use Each Mode

### Use **Hot** when:
- Inference speed is critical (webcam, live dashboards)
- You don't need secondary detection
- Single-detector accuracy is sufficient

### Use **Inline** when:
- You need real-time dual detection
- Accuracy is more important than speed
- You want immediate drift detection feedback
- Offline training on pseudo-labels is acceptable

### Use **Batch** when:
- You need fast inference (high throughput)
- Secondary analysis can be deferred
- You want to analyze low-confidence frames offline
- You're processing long video streams

---

## Implementation Details

### Hot Path
- `DualDetector.mode = False` or `'hot'`
- Secondary detector not loaded (saves memory)
- No frame I/O
- Returns primary detections only

### Inline Mode
- `DualDetector.mode = 'inline'`
- Secondary detector loaded and runs on low-confidence frames
- Merges results (high-conf primary + non-overlapping secondary)
- Saves frames for offline training

### Batch Mode
- `DualDetector.mode = 'batch'`
- Secondary detector not loaded during inference
- Saves low-confidence frames to disk
- Run `secondary_detector_batch.py` separately for offline analysis

---

## Future Extensions

- **Data Drift Detection**: Compare input feature distributions (brightness, contrast) against training baseline
- **Ensemble Methods**: Combine multiple secondary detectors
- **Adaptive Thresholding**: Dynamically adjust `dual_confidence_threshold` based on drift metrics
