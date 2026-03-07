"""RF-DETR detector from Roboflow (pip install rfdetr).

Returns sv.Detections directly from model.predict() — no conversion needed.
Runs at native precision (RF-DETR manages its own weights).
Singleton caching handled by DetectorFactory.
"""

import numpy as np
import supervision as sv
from .base import BaseDetector


class RFDETRDetector(BaseDetector):

    def __init__(self, config: dict):
        self.model = None
        self.confidence_threshold = config.get('inference', {}).get('confidence_threshold', 0.35)
        self.classes = config.get('inference', {}).get('classes', [0])
        self.load_model()

    def load_model(self, config: dict = None) -> None:
        try:
            from rfdetr import RFDETRBase
            self.model = RFDETRBase()
            print("[RFDETRDetector] Loaded (native precision)")
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

        return detections
