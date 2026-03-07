import numpy as np
import supervision as sv
from ultralytics import YOLO
from .base import BaseDetector


class YOLODetector(BaseDetector):
    """YOLO detector via Ultralytics with precision control.

    Precision options (set in config model.precision):
      - fp32: Default. Works on CPU + GPU. Full precision.
      - fp16: Half-precision. GPU-only (CUDA). ~2x faster, ~50% less VRAM.
              Falls back to FP32 on CPU.

    Output format is IDENTICAL across all precisions — sv.Detections
    with the same .xyxy, .confidence, .class_id fields.

    Models are automatically cached in ~/.cache/yolo/ by Ultralytics.
    Singleton caching handled by DetectorFactory.
    """

    def __init__(self, config: dict):
        inf = config.get('inference', {})
        self.confidence_threshold = inf.get('confidence_threshold', 0.35)
        self.iou_threshold = inf.get('iou_threshold', 0.45)
        self.classes = inf.get('classes', [0])
        self.device = inf.get('device', 'cpu')

        model_cfg = config.get('model', {})
        self.precision = model_cfg.get('precision', 'fp32')
        self._half = False  # set True only for fp16 on GPU

        self.model = None
        self._load(model_cfg)

    def _load(self, model_cfg: dict) -> None:
        """Load YOLO model with optional FP16 on GPU.

        Note: INT8 quantization requires TensorRT (format='engine'),
        which is GPU-only and complex for a portfolio project.
        For CPU, FP32 is the practical choice.
        """
        name = model_cfg.get('name', 'yolo11n')
        if not name.endswith(('.pt', '.yaml')):
            name += '.pt'

        is_gpu = 'cuda' in self.device.lower()

        # FP16 only on GPU; otherwise use FP32
        if self.precision == 'fp16' and is_gpu:
            self.model = YOLO(name)
            self._half = True
            print(f"[YOLODetector] FP16 on {self.device}")
        else:
            if self.precision == 'fp16' and not is_gpu:
                print("[YOLODetector] FP16 requires CUDA. Using FP32 on CPU.")
            self.model = YOLO(name)
            print(f"[YOLODetector] FP32 on {self.device} (cached in ~/.cache/yolo/)")

    def load_model(self, config: dict = None) -> None:
        if config:
            self._load(config.get('model', {}))

    def detect(self, image: np.ndarray) -> sv.Detections:
        results = self.model(
            image, conf=self.confidence_threshold, iou=self.iou_threshold,
            classes=self.classes, device=self.device, verbose=False,
            half=self._half,
        )
        return sv.Detections.from_ultralytics(results[0])
