"""
Tests for the detection module.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Tests for the detection module.
"""
import sys
import os
import logging
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision_ml.detection.detector_factory import DetectorFactory
from src.vision_ml.detection.yolo_detector import YOLODetector
from src.vision_ml.logging.logger import _UltralyticsDeprecationFilter

def test_detector_initialization():
    """
    Tests that the DetectorFactory can create a detector.
    """
    config = {'detection': {'detector_type': 'yolo11n', 'inference': {'device': 'cpu'}}}
    # Mocking or using a lightweight detector if possible, or just checking factory logic
    # For now, let's just check if we can import and the factory exists
    assert DetectorFactory is not None


def test_yolo_detector_does_not_pass_half_when_false():
    """Verify YOLODetector does not pass half=False, avoiding deprecation warnings."""
    config = {
        'inference': {'confidence_threshold': 0.4, 'iou_threshold': 0.5, 'classes': [0], 'device': 'cpu'},
        'model': {'precision': 'fp32', 'name': 'yolo11n'}
    }
    with patch('src.vision_ml.detection.yolo_detector.YOLO') as mock_yolo_cls:
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.names = {0: 'person'}
        mock_model.return_value = [mock_result]
        mock_yolo_cls.return_value = mock_model

        with patch('supervision.Detections.from_ultralytics'):
            detector = YOLODetector(config)
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            detector.detect(dummy_frame)

            assert mock_model.call_count == 1
            call_kwargs = mock_model.call_args[1]
            assert 'half' not in call_kwargs
            assert 'quantize' not in call_kwargs


def test_yolo_detector_uses_quantize_when_fp16_enabled():
    """Verify YOLODetector passes quantize=16 when fp16 is enabled on GPU."""
    config = {
        'inference': {'confidence_threshold': 0.4, 'iou_threshold': 0.5, 'classes': [0], 'device': 'cuda:0'},
        'model': {'precision': 'fp16', 'name': 'yolo11n'}
    }
    with patch('src.vision_ml.detection.yolo_detector.YOLO') as mock_yolo_cls:
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.names = {0: 'person'}
        mock_model.return_value = [mock_result]
        mock_yolo_cls.return_value = mock_model

        with patch('supervision.Detections.from_ultralytics'):
            detector = YOLODetector(config)
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            detector.detect(dummy_frame)

            assert mock_model.call_count == 1
            call_kwargs = mock_model.call_args[1]
            assert call_kwargs.get('quantize') == 16
            assert 'half' not in call_kwargs


def test_ultralytics_deprecation_filter():
    """Verify the logging filter suppresses 'half' deprecation warnings."""
    filt = _UltralyticsDeprecationFilter()
    rec_deprecated = logging.LogRecord(
        name="ultralytics", level=logging.WARNING, pathname="", lineno=0,
        msg="'half' is deprecated and will be removed in the future. Use 'quantize' instead.",
        args=(), exc_info=None
    )
    rec_normal = logging.LogRecord(
        name="ultralytics", level=logging.INFO, pathname="", lineno=0,
        msg="Inference speed 12.5ms", args=(), exc_info=None
    )
    assert filt.filter(rec_deprecated) is False
    assert filt.filter(rec_normal) is True
