import os
import json
import glob
import numpy as np
import supervision as sv


class AutoLabeler:
    """Collects high-confidence detections as pseudo-labels for training.

    Two label sources:
    1. collect() — from live inference (high-confidence detections)
    2. load_dual_detector_frames() — from DualDetector's saved low-confidence
       frames (data/low_confidence_frames/*.json), already labeled by RF-DETR
    """

    def __init__(self, config: dict):
        label_cfg = config.get('labeling', {})
        self.enabled = label_cfg.get('enabled', False)
        self.provider = label_cfg.get('provider', 'local')
        self.min_confidence = label_cfg.get('auto_label_confidence', 0.7)
        self.roboflow_api_key = label_cfg.get('roboflow_api_key') or os.environ.get('ROBOFLOW_API_KEY')
        self.roboflow_workspace = label_cfg.get('roboflow_workspace')
        self.roboflow_project = label_cfg.get('roboflow_project')
        self.pending_labels = []

    def collect(self, image: np.ndarray, detections: sv.Detections, image_id: str = None):
        if not self.enabled or detections.confidence is None or len(detections) == 0:
            return
        mask = detections.confidence >= self.min_confidence
        if not mask.any():
            return
        filtered = detections[mask]
        self.pending_labels.append({
            'image_id': image_id or f"frame_{len(self.pending_labels)}",
            'boxes': filtered.xyxy.tolist(),
            'confidences': filtered.confidence.tolist(),
            'class_ids': filtered.class_id.tolist() if filtered.class_id is not None else [],
        })

    def load_dual_detector_frames(self, frame_dir: str = 'data/low_confidence_frames'):
        """Import pseudo-labels saved by DualDetector for offline training."""
        if not os.path.isdir(frame_dir):
            print(f"[AutoLabeler] No saved frames at {frame_dir}")
            return 0
        label_files = sorted(glob.glob(os.path.join(frame_dir, '*.json')))
        for lf in label_files:
            with open(lf, 'r') as f:
                label = json.load(f)
            label['image_id'] = os.path.splitext(os.path.basename(lf))[0]
            label['image_path'] = lf.replace('.json', '.jpg')
            self.pending_labels.append(label)
        print(f"[AutoLabeler] Loaded {len(label_files)} pseudo-labels from {frame_dir}")
        return len(label_files)

    def flush(self, output_dir: str = 'data/auto_labeled'):
        if self.provider == 'roboflow' and self.roboflow_api_key:
            self._upload_roboflow()
        else:
            self._export_local(output_dir)
        self.pending_labels.clear()

    def _export_local(self, output_dir: str):
        if not self.pending_labels:
            return
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'auto_labels.json')
        with open(out_path, 'w') as f:
            json.dump(self.pending_labels, f, indent=2)
        print(f"[AutoLabeler] Exported {len(self.pending_labels)} labels to {out_path}")

    def _upload_roboflow(self):
        if not self.roboflow_api_key:
            print("[AutoLabeler] Roboflow API key not set.")
            return
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=self.roboflow_api_key)
            project = rf.workspace(self.roboflow_workspace).project(self.roboflow_project)
            print(f"[AutoLabeler] Connected to Roboflow: {project.name}")
        except ImportError:
            print("[AutoLabeler] pip install roboflow")
        except Exception as e:
            print(f"[AutoLabeler] Roboflow error: {e}")
