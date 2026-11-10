"""Tests for DualDetector."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.vision_ml.detection.dual_detector import DualDetector


@pytest.fixture
def dual_config():
    return {
        'detection': {
            'detector_type': 'yolo11n',
            'use_dual_detector': 'batch',
            'primary_detector': 'yolo11n',
            'secondary_detector': 'rfdetr',
            'dual_confidence_threshold': 0.5,
            'save_low_confidence_frames': False,
            'low_confidence_dir': 'data/test_low_conf',
        },
        'inference': {'confidence_threshold': 0.35, 'iou_threshold': 0.45, 'classes': [0], 'device': 'cpu'},
        'model': {'name': 'yolo11n', 'precision': 'fp32'},
    }


def _make_detections(n, conf=0.9):
    """Helper to create mock detections."""
    try:
        import supervision as sv
        if hasattr(sv, 'Detections') and not isinstance(sv, MagicMock):
            det = sv.Detections(
                xyxy=np.array([[10, 10, 50, 50]] * n, dtype=float),
                confidence=np.array([conf] * n),
                class_id=np.array([0] * n, dtype=int),
            )
        else:
            det = sv.Detections(
                xyxy=np.array([[10, 10, 50, 50]] * n, dtype=float),
                confidence=np.array([conf] * n),
                class_id=np.array([0] * n, dtype=int),
            )
    except Exception:
        from tests.conftest import MockSupervision
        det = MockSupervision.Detections(
            xyxy=np.array([[10, 10, 50, 50]] * n, dtype=float),
            confidence=np.array([conf] * n),
            class_id=np.array([0] * n, dtype=int),
        )
    return det


class TestDualDetector:
    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_hot_mode(self, mock_factory, dual_config):
        dual_config['detection']['use_dual_detector'] = 'hot'
        mock_det = MagicMock()
        mock_det.detect.return_value = _make_detections(2)
        mock_factory.get.return_value = mock_det

        dd = DualDetector(dual_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = dd.detect(frame)
        assert len(result) == 2

    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_batch_mode_high_conf(self, mock_factory, dual_config):
        """Batch mode with high confidence — no frames saved."""
        mock_det = MagicMock()
        mock_det.detect.return_value = _make_detections(2, conf=0.9)
        mock_factory.get.return_value = mock_det

        dd = DualDetector(dual_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets, sources = dd.detect_with_source(frame)
        assert len(dets) == 2
        assert all(s == 'primary' for s in sources)

    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_batch_mode_low_conf_saves(self, mock_factory, dual_config, tmp_path):
        """Batch mode with low confidence — frame saving triggered."""
        dual_config['detection']['save_low_confidence_frames'] = True
        dual_config['detection']['low_confidence_dir'] = str(tmp_path / 'low_conf')
        mock_det = MagicMock()
        mock_det.detect.return_value = _make_detections(1, conf=0.2)
        mock_factory.get.return_value = mock_det

        dd = DualDetector(dual_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dd.detect(frame)
        assert dd._saved_count == 1

    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_stats(self, mock_factory, dual_config):
        mock_det = MagicMock()
        mock_det.detect.return_value = _make_detections(1)
        mock_factory.get.return_value = mock_det

        dd = DualDetector(dual_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dd.detect(frame)
        stats = dd.stats
        assert stats['total_frames'] == 1
        assert 'secondary_ratio' in stats

    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_inline_mode_high_conf(self, mock_factory, dual_config):
        """Inline mode with high conf — secondary not called."""
        dual_config['detection']['use_dual_detector'] = 'inline'
        mock_primary = MagicMock()
        mock_primary.detect.return_value = _make_detections(2, conf=0.9)
        mock_secondary = MagicMock()
        mock_factory.get.side_effect = [mock_primary, mock_secondary]

        dd = DualDetector(dual_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dd.detect(frame)
        mock_secondary.detect.assert_not_called()

    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_inline_mode_low_conf_calls_secondary(self, mock_factory, dual_config):
        """Inline mode with low conf — secondary detector called."""
        dual_config['detection']['use_dual_detector'] = 'inline'
        dual_config['detection']['save_low_confidence_frames'] = False
        mock_primary = MagicMock()
        mock_primary.detect.return_value = _make_detections(1, conf=0.2)
        mock_secondary = MagicMock()
        mock_secondary.detect.return_value = _make_detections(1, conf=0.8)
        mock_factory.get.side_effect = [mock_primary, mock_secondary]

        dd = DualDetector(dual_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dd.detect(frame)
        mock_secondary.detect.assert_called_once()
        assert dd._secondary_calls == 1

    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_cooldown_prevents_duplicate_saving(self, mock_factory, dual_config, tmp_path):
        """Consecutive frames within cooldown interval must not be saved."""
        dual_config['detection']['save_low_confidence_frames'] = True
        dual_config['detection']['low_confidence_dir'] = str(tmp_path / 'low_conf')
        dual_config['detection']['save_cooldown_frames'] = 10
        mock_det = MagicMock()
        mock_det.detect.return_value = _make_detections(1, conf=0.3)
        mock_factory.get.return_value = mock_det

        dd = DualDetector(dual_config)
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 50

        # Frame 1: saves
        dd.detect(frame1)
        assert dd._saved_count == 1

        # Frame 2 (too soon, within 10 frames): dropped by cooldown
        dd.detect(frame2)
        assert dd._saved_count == 1

    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_noise_floor_ignores_garbage_confidence(self, mock_factory, dual_config, tmp_path):
        """Detections below min_confidence_floor (e.g. 0.05 < 0.15) should be ignored as noise."""
        dual_config['detection']['save_low_confidence_frames'] = True
        dual_config['detection']['low_confidence_dir'] = str(tmp_path / 'low_conf')
        dual_config['detection']['dual_min_confidence_floor'] = 0.15
        mock_det = MagicMock()
        mock_det.detect.return_value = _make_detections(1, conf=0.05)
        mock_factory.get.return_value = mock_det

        dd = DualDetector(dual_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dd.detect(frame)
        assert dd._saved_count == 0

    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_max_frame_budget_caps_savings(self, mock_factory, dual_config, tmp_path):
        """Max frame budget should stop saving once quota is reached."""
        dual_config['detection']['save_low_confidence_frames'] = True
        dual_config['detection']['low_confidence_dir'] = str(tmp_path / 'low_conf')
        dual_config['detection']['save_cooldown_frames'] = 1
        dual_config['detection']['max_saved_frames'] = 2
        mock_det = MagicMock()
        mock_det.detect.return_value = _make_detections(1, conf=0.25)
        mock_factory.get.return_value = mock_det

        dd = DualDetector(dual_config)
        for i in range(5):
            # Pass different images so similarity check doesn't block
            frame = np.ones((480, 640, 3), dtype=np.uint8) * (i * 30 + 1)
            dd.detect(frame)

        # Even though 5 opportunities occurred, it was capped at max 2
        assert dd._saved_count == 2

    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_multi_source_isolation_prevents_overwrite(self, mock_factory, dual_config, tmp_path):
        """Sequential runs on multiple videos must use distinct prefixes and never overwrite frames."""
        import os
        save_dir = tmp_path / 'low_conf'
        dual_config['detection']['save_low_confidence_frames'] = True
        dual_config['detection']['low_confidence_dir'] = str(save_dir)
        dual_config['detection']['save_cooldown_frames'] = 1
        dual_config['detection']['max_saved_frames'] = 5
        mock_det = MagicMock()
        mock_det.detect.return_value = _make_detections(1, conf=0.25)
        mock_factory.get.return_value = mock_det

        dd = DualDetector(dual_config)

        # Video 1
        dd.set_source("data/sample_videos/video_1.mp4")
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 10
        dd.detect(frame1)
        assert os.path.exists(save_dir / "video_1_frame_000000.jpg")
        assert os.path.exists(save_dir / "video_1_frame_000000.json")

        # Video 2
        dd.set_source("data/sample_videos/video_2.mp4")
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 20
        dd.detect(frame2)
        assert os.path.exists(save_dir / "video_2_frame_000000.jpg")
        assert os.path.exists(save_dir / "video_2_frame_000000.json")

        # Video 1's frame must still exist untouched
        assert os.path.exists(save_dir / "video_1_frame_000000.jpg")

    @patch('src.vision_ml.detection.dual_detector.DetectorFactory')
    def test_no_overwrite_when_frame_already_exists(self, mock_factory, dual_config, tmp_path):
        """When an existing frame is present on disk, counter advances without overwriting."""
        import os
        save_dir = tmp_path / 'low_conf'
        os.makedirs(save_dir, exist_ok=True)
        # Pre-seed existing frame
        (save_dir / "video_1_frame_000000.jpg").write_text("existing")
        (save_dir / "video_1_frame_000000.json").write_text("{}")

        dual_config['detection']['save_low_confidence_frames'] = True
        dual_config['detection']['low_confidence_dir'] = str(save_dir)
        mock_det = MagicMock()
        mock_det.detect.return_value = _make_detections(1, conf=0.25)
        mock_factory.get.return_value = mock_det

        dd = DualDetector(dual_config)
        dd.set_source("data/sample_videos/video_1.mp4")
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 10
        dd.detect(frame)

        # Pre-seeded frame preserved
        assert (save_dir / "video_1_frame_000000.jpg").read_text() == "existing"
        # New frame allocated next non-colliding slot
        assert os.path.exists(save_dir / "video_1_frame_000001.jpg")
