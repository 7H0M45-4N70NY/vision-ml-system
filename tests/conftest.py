import sys
import numpy as np
import pytest
from unittest.mock import MagicMock


class MockSupervision:
    class Detections:
        def __init__(self, xyxy, confidence, class_id, tracker_id=None):
            self.xyxy = np.array(xyxy) if not isinstance(xyxy, np.ndarray) and len(xyxy) > 0 else (xyxy if isinstance(xyxy, np.ndarray) else np.empty((0, 4)))
            self.confidence = np.array(confidence) if not isinstance(confidence, np.ndarray) and len(confidence) > 0 else (confidence if isinstance(confidence, np.ndarray) else np.empty(0))
            self.class_id = np.array(class_id) if not isinstance(class_id, np.ndarray) and len(class_id) > 0 else (class_id if isinstance(class_id, np.ndarray) else np.empty(0, dtype=int))
            self.tracker_id = tracker_id

        @classmethod
        def from_ultralytics(cls, results):
            return cls(
                xyxy=results.boxes.xyxy.cpu().numpy(),
                confidence=results.boxes.conf.cpu().numpy(),
                class_id=results.boxes.cls.cpu().numpy()
            )

        @classmethod
        def empty(cls):
            return cls(np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int))

        @classmethod
        def merge(cls, dets_list):
            if not dets_list or all(len(d) == 0 for d in dets_list):
                return cls.empty()
            xyxy = np.concatenate([d.xyxy for d in dets_list if len(d) > 0])
            conf = np.concatenate([d.confidence for d in dets_list if len(d) > 0])
            cid = np.concatenate([d.class_id for d in dets_list if len(d) > 0])
            return cls(xyxy, conf, cid)

        def __len__(self):
            return len(self.xyxy)

        def __getitem__(self, index):
            """Support boolean mask and integer slicing."""
            return MockSupervision.Detections(
                xyxy=self.xyxy[index],
                confidence=self.confidence[index],
                class_id=self.class_id[index],
                tracker_id=self.tracker_id[index] if self.tracker_id is not None else None,
            )

    class ByteTrack:
        def __init__(self, **kwargs):
            self._next_id = 1

        def update_with_detections(self, detections):
            ids = np.arange(self._next_id, self._next_id + len(detections))
            self._next_id += len(detections)
            detections.tracker_id = ids
            return detections

        def reset(self):
            self._next_id = 1

    class BoundingBoxAnnotator:
        def __init__(self, **kwargs):
            pass

        def annotate(self, scene, detections):
            return scene

    class LabelAnnotator:
        def __init__(self, **kwargs):
            pass

        def annotate(self, scene, detections, labels=None):
            return scene

    class TraceAnnotator:
        def __init__(self, **kwargs):
            pass

        def annotate(self, scene, detections):
            return scene

    @staticmethod
    def box_iou_batch(box1, box2):
        return np.array([[0.5]])


# Mock supervision if not installed
try:
    import supervision
except ImportError:
    sys.modules["supervision"] = MockSupervision
    sys.modules["supervision.Detections"] = MockSupervision.Detections

# Mock ultralytics if not installed
try:
    import ultralytics
except ImportError:
    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO = MagicMock()
    sys.modules["ultralytics"] = mock_ultralytics

# Mock rfdetr if not installed
try:
    import rfdetr
except ImportError:
    mock_rfdetr = MagicMock()
    sys.modules["rfdetr"] = mock_rfdetr

# Mock cv2 if not installed (headless environments)
try:
    import cv2
except ImportError:
    mock_cv2 = MagicMock()
    sys.modules["cv2"] = mock_cv2

# Mock dagshub if not installed
try:
    import dagshub
except ImportError:
    sys.modules["dagshub"] = MagicMock()

# Mock mlflow if not installed
try:
    import mlflow
except ImportError:
    sys.modules["mlflow"] = MagicMock()
    sys.modules["mlflow.pytorch"] = MagicMock()
    sys.modules["mlflow.tracking"] = MagicMock()

# Mock prometheus if not installed
try:
    import prometheus_fastapi_instrumentator
except ImportError:
    mock_prom = MagicMock()
    mock_prom.Instrumentator.return_value.instrument.return_value.expose = MagicMock()
    sys.modules["prometheus_fastapi_instrumentator"] = mock_prom


# ---- Fixtures ----

@pytest.fixture
def sample_config():
    """Minimal valid config dict covering all modules."""
    return {
        'model': {'name': 'yolo11n', 'precision': 'fp32'},
        'inference': {
            'confidence_threshold': 0.35,
            'iou_threshold': 0.45,
            'classes': [0],
            'device': 'cpu',
        },
        'detection': {
            'detector_type': 'yolo11n',
            'use_dual_detector': False,
        },
        'tracking': {
            'enabled': True,
            'tracker_type': 'bytetrack',
            'track_thresh': 0.25,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'frame_rate': 30,
        },
        'annotation': {
            'bounding_box': {'enabled': True, 'thickness': 2},
            'label': {'enabled': True, 'text_scale': 0.5},
            'trace': {'enabled': False},
        },
        'analytics': {
            'enabled': True,
            'compute_dwell_time': True,
            'dwell_time_fps': 30,
            'output_json': True,
        },
        'drift': {
            'enabled': False,
            'method': 'confidence_drop',
            'confidence_threshold': 0.3,
            'window_size': 500,
            'check_interval': 100,
        },
        'labeling': {'enabled': False, 'provider': 'local'},
        'mode': {'type': 'offline', 'output_dir': 'runs/test', 'show_live': False},
        'mlflow': {
            'tracking_uri': '',
            'experiment_name': 'test',
            'run_name': 'test_run',
            'log_params': False,
            'log_metrics': True,
            'log_model': False,
            'register_model': False,
        },
        'training': {
            'epochs': 1,
            'batch_size': 4,
            'imgsz': 640,
            'learning_rate': 0.01,
            'optimizer': 'auto',
            'device': 'cpu',
            'patience': 5,
            'project': 'runs/train',
        },
        'data': {'dataset_yaml': 'coco8.yaml'},
    }


@pytest.fixture
def mock_numpy_frame():
    """480x640 black BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _clear_model_registry():
    """Reset ModelRegistry singleton cache between tests."""
    from src.vision_ml.detection.model_registry import ModelRegistry
    ModelRegistry.clear_models()
    yield
    ModelRegistry.clear_models()
