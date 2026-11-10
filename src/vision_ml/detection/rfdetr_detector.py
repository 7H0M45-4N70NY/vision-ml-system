"""RF-DETR detector from Roboflow (pip install rfdetr).

Returns sv.Detections directly from model.predict() — no conversion needed.
Runs at native precision (RF-DETR manages its own weights).
Singleton caching handled by DetectorFactory.
"""

import numpy as np
import supervision as sv
from .base import BaseDetector
from ..logging import get_logger

logger = get_logger(__name__)


class RFDETRDetector(BaseDetector):
    # RF-DETR uses standard COCO IDs where person=1, unlike YOLO which
    # remaps filtered classes so person=0 when classes=[0] is passed.
    # This map translates config class IDs (YOLO-style) to COCO IDs.
    _YOLO_TO_COCO = {0: 1}  # YOLO person(0) → COCO person(1)

    def __init__(self, config: dict):
        self.model = None
        inf_cfg = config.get('inference', {})
        raw_conf = inf_cfg.get('confidence_threshold', 0.35)
        try:
            if isinstance(raw_conf, str):
                raw_conf = raw_conf.rstrip('s')
            self.confidence_threshold = float(raw_conf)
        except (ValueError, TypeError):
            self.confidence_threshold = 0.35

        yolo_classes = inf_cfg.get('classes', [0])
        # Convert YOLO-style class IDs to COCO IDs for RF-DETR filtering
        self.classes = [self._YOLO_TO_COCO.get(c, c) for c in yolo_classes]

        target_device = inf_cfg.get('device', 'cuda:0')
        try:
            import torch
            if 'cuda' in str(target_device).lower() and not torch.cuda.is_available():
                target_device = 'cpu'
            elif not target_device or target_device == 'cpu':
                if torch.cuda.is_available():
                    target_device = 'cuda:0'
        except ImportError:
            target_device = 'cpu'

        self.device = str(target_device)
        self.load_model()

    def load_model(self, config: dict = None) -> None:
        try:
            from rfdetr import RFDETRBase
            self.model = RFDETRBase(device=self.device)
            logger.info(f"Loaded RF-DETR on {self.device} (native precision)")
        except ImportError:
            raise ImportError("rfdetr not installed. pip install rfdetr")

    def detect(self, image: np.ndarray) -> sv.Detections:
        if self.model is None:
            raise RuntimeError("RF-DETR model not loaded.")

        from PIL import Image
        pil_image = Image.fromarray(image[..., ::-1])  # BGR→RGB→PIL

        detections = self.model.predict(pil_image, threshold=self.confidence_threshold)

        if self.classes and detections.class_id is not None:
            mask = np.isin(detections.class_id, self.classes)
            detections = detections[mask]
            # Remap COCO class IDs back to YOLO-style (person 1→0)
            coco_to_yolo = {v: k for k, v in self._YOLO_TO_COCO.items()}
            detections.class_id = np.array([
                coco_to_yolo.get(c, c) for c in detections.class_id
            ])

        return detections
