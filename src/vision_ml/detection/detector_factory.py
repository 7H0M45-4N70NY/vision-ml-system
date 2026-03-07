from .base import BaseDetector
from .yolo_detector import YOLODetector
from .rfdetr_detector import RFDETRDetector
from .model_reistry import ModelRegistry


_REGISTRY = {
    'yolo11n': YOLODetector,
    'yolo11s': YOLODetector,
    'yolo11m': YOLODetector,
    'yolo11l': YOLODetector,
    'yolo11x': YOLODetector,
    'rfdetr': RFDETRDetector,
}


class DetectorFactory:
    """Factory for detectors. YOLO variants are singletons (via ModelRegistry).
    RF-DETR is also cached. Trackers are never cached (stateful).
    """

    @staticmethod
    def get(detector_type: str, config: dict) -> BaseDetector:
        if detector_type not in _REGISTRY:
            raise ValueError(f"Unknown detector: {detector_type}. Available: {list(_REGISTRY.keys())}")

        def _load(_key):
            return _REGISTRY[detector_type](config)

        return ModelRegistry.get_model(detector_type, _load)

    @staticmethod
    def from_config(config: dict) -> BaseDetector:
        detector_type = config.get('detection', {}).get('detector_type', 'yolo11n')
        return DetectorFactory.get(detector_type, config)

    @staticmethod
    def list_available() -> list:
        return list(_REGISTRY.keys())
