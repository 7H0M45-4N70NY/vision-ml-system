import os
import shutil
from pathlib import Path
import numpy as np
import supervision as sv
from ultralytics import YOLO
from .base import BaseDetector
from ..logging import get_logger

logger = get_logger(__name__)


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
        raw_conf = inf.get('confidence_threshold', 0.35)
        try:
            if isinstance(raw_conf, str):
                raw_conf = raw_conf.rstrip('s')
            self.confidence_threshold = float(raw_conf)
        except (ValueError, TypeError):
            self.confidence_threshold = 0.35

        self.iou_threshold = float(inf.get('iou_threshold', 0.45))
        self.classes = inf.get('classes', [0])
        
        target_device = inf.get('device', 'cuda:0')
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

        model_cfg = config.get('model', {})
        self.precision = model_cfg.get('precision', 'fp32')
        self._half = False  # set True only for fp16 on GPU

        self.model = None
        self._load(model_cfg)

    def _load(self, model_cfg: dict) -> None:
        """Load YOLO model with optional FP16 on GPU.

        Original pre-trained models are isolated in models/original/.
        Fine-tuned models reside in models/fine_tuned/.
        """
        raw_name = model_cfg.get('name', 'yolo11n')
        filename = raw_name if raw_name.endswith(('.pt', '.yaml', '.onnx', '.engine')) else f"{raw_name}.pt"

        # Determine repository root (4 levels up from this file)
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        original_dir = repo_root / "models" / "original"
        fine_tuned_dir = repo_root / "models" / "fine_tuned"
        original_dir.mkdir(parents=True, exist_ok=True)
        fine_tuned_dir.mkdir(parents=True, exist_ok=True)

        target_model_path = None

        # Check explicit path first (if path components are provided)
        candidate_explicit = Path(filename)
        if candidate_explicit.is_file():
            target_model_path = candidate_explicit
        elif (repo_root / filename).is_file():
            target_model_path = repo_root / filename
        # Check models/fine_tuned/ next
        elif (fine_tuned_dir / candidate_explicit.name).is_file():
            target_model_path = fine_tuned_dir / candidate_explicit.name
        # Check models/original/
        elif (original_dir / candidate_explicit.name).is_file():
            target_model_path = original_dir / candidate_explicit.name

        is_gpu = 'cuda' in self.device.lower()
        use_half = self.precision == 'fp16' and is_gpu
        if self.precision == 'fp16' and not is_gpu:
            logger.warning("FP16 requires CUDA. Using FP32 on CPU.")

        if target_model_path is not None:
            logger.info(f"Loading model weights from: {target_model_path} ({'FP16' if use_half else 'FP32'} on {self.device})")
            self.model = YOLO(str(target_model_path))
            self._half = use_half
        else:
            # First-time download: download model into models/original/
            dest_path = original_dir / candidate_explicit.name
            logger.info(f"Model weights not found locally. Downloading '{candidate_explicit.name}' to models/original/...")
            self.model = YOLO(candidate_explicit.name)
            self._half = use_half
            
            # If Ultralytics downloaded into current working directory, relocate to models/original/
            for cwd_candidate in [Path.cwd() / candidate_explicit.name, repo_root / candidate_explicit.name]:
                if cwd_candidate.is_file() and cwd_candidate.resolve() != dest_path.resolve():
                    shutil.move(str(cwd_candidate), str(dest_path))
                    self.model = YOLO(str(dest_path))
                    logger.info(f"Relocated original model weights to: {dest_path}")
                    break

    def load_model(self, config: dict = None) -> None:
        if config:
            self._load(config.get('model', {}))

    def detect(self, image: np.ndarray) -> sv.Detections:
        kwargs = {
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "classes": self.classes,
            "device": self.device,
            "verbose": False,
        }
        if self._half:
            # Modern Ultralytics uses 'quantize=16' for FP16 inference on GPU.
            # Passing 'half=False' or 'half=True' triggers deprecation warnings.
            try:
                results = self.model(image, quantize=16, **kwargs)
            except (TypeError, ValueError):
                results = self.model(image, half=True, **kwargs)
        else:
            results = self.model(image, **kwargs)

        return sv.Detections.from_ultralytics(results[0])
