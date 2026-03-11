import os
import json
import glob
import tempfile
import cv2
import supervision as sv

from ..logging import get_logger

logger = get_logger(__name__)


class AutoLabeler:
    """Collects low-confidence frames labeled by RF-DETR for active learning.

    Active learning approach: focus on frames where model struggles (low-confidence).
    These are saved by DualDetector and labeled by RF-DETR fallback detector.

    Single label source:
    - load_dual_detector_frames() — from DualDetector's saved low-confidence
      frames (data/low_confidence_frames/*.json), already labeled by RF-DETR
    """

    def __init__(self, config: dict):
        label_cfg = config.get('labeling', {})
        self.enabled = label_cfg.get('enabled', False)
        self.provider = label_cfg.get('provider', 'local')
        self.roboflow_api_key = label_cfg.get('roboflow_api_key') or os.environ.get('ROBOFLOW_API_KEY')
        self.roboflow_workspace = label_cfg.get('roboflow_workspace')
        self.roboflow_project = label_cfg.get('roboflow_project')
        self.pending_labels = []

    def collect(self, image, detections, image_id: str = None):
        """No-op: high-confidence collection disabled. Only low-confidence frames via load_dual_detector_frames()."""
        pass

    def load_dual_detector_frames(self, frame_dir: str = 'data/low_confidence_frames'):
        """Import pseudo-labels saved by DualDetector for offline training."""
        if not os.path.isdir(frame_dir):
            logger.warning(f"No saved frames at {frame_dir}")
            return 0
        label_files = sorted(glob.glob(os.path.join(frame_dir, '*.json')))
        for lf in label_files:
            with open(lf, 'r') as f:
                label = json.load(f)
            label['image_id'] = os.path.splitext(os.path.basename(lf))[0]
            label['image_path'] = lf.replace('.json', '.jpg')
            self.pending_labels.append(label)
        logger.info(f"Loaded {len(label_files)} pseudo-labels from {frame_dir}")
        return len(label_files)

    def flush(self, output_dir: str = 'data/auto_labeled'):
        if self.provider == 'roboflow' and self.roboflow_api_key:
            self._upload_roboflow()
        else:
            self._export_local(output_dir)
        self.pending_labels.clear()

    def _export_local(self, output_dir: str):
        if not self.pending_labels:
            logger.debug("No pending labels to export")
            return
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'auto_labels.json')
        with open(out_path, 'w') as f:
            json.dump(self.pending_labels, f, indent=2)
        logger.info(f"Exported {len(self.pending_labels)} labels to {out_path}")

    def _upload_roboflow(self):
        if not self.roboflow_api_key:
            logger.error("Roboflow API key not set. Set ROBOFLOW_API_KEY environment variable.")
            return
        if not self.roboflow_workspace or not self.roboflow_project:
            logger.error("Roboflow workspace or project not configured. Update config file.")
            logger.info("Falling back to local export...")
            self._export_local('data/auto_labeled')
            return
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=self.roboflow_api_key)
            project = rf.workspace(self.roboflow_workspace).project(self.roboflow_project)
            logger.info(f"Connected to Roboflow project: {project.name}")

            uploaded = 0
            with tempfile.TemporaryDirectory() as tmp_dir:
                for label in self.pending_labels:
                    image_id = label.get('image_id', f'frame_{uploaded}')
                    image_path = label.get('image_path')

                    # All images come from low_confidence_frames (DualDetector saves them)
                    if not image_path or not os.path.isfile(image_path):
                        continue

                    # Build YOLO-format annotation file
                    # Use .yolov8.txt extension for better Roboflow format detection
                    annotation_path = os.path.join(tmp_dir, f"{image_id}.yolov8.txt")
                    self._write_yolo_annotation(label, annotation_path)

                    # Upload image + annotation to Roboflow
                    # Note: annotation_format is not supported by SDK; format is auto-detected from file extension
                    project.upload(
                        image_path=image_path,
                        annotation_path=annotation_path,
                    )
                    uploaded += 1

            logger.info(f"Successfully uploaded {uploaded}/{len(self.pending_labels)} labels to Roboflow")

        except ImportError:
            logger.error("roboflow package not installed. Run: pip install roboflow")
            logger.info("Falling back to local export...")
            self._export_local('data/auto_labeled')
        except Exception as e:
            logger.error(f"Roboflow upload failed: {e}")
            logger.info("Falling back to local export...")
            self._export_local('data/auto_labeled')

    @staticmethod
    def _write_yolo_annotation(label: dict, output_path: str):
        """Convert label dict (xyxy boxes) to YOLO format annotation file."""
        boxes = label.get('boxes', [])
        class_ids = label.get('class_ids', [])
        # Need image dimensions to normalize — get from image_path if available
        img_path = label.get('image_path')
        img_w, img_h = 640, 640  # fallback
        if img_path and os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_h, img_w = img.shape[:2]

        with open(output_path, 'w') as f:
            for i, box in enumerate(boxes):
                cls_id = class_ids[i] if i < len(class_ids) else 0
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
