"""Tests for FrameAnnotator."""
import numpy as np

from src.vision_ml.annotation.annotator import FrameAnnotator


class TestFrameAnnotator:
    def test_init_default(self, sample_config):
        ann = FrameAnnotator(sample_config)
        assert ann.box_annotator is not None

    def test_annotate_returns_frame(self, sample_config, mock_numpy_frame):
        import supervision as sv
        ann = FrameAnnotator(sample_config)
        det = sv.Detections(
            xyxy=np.array([[10, 10, 50, 50]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        result = ann.annotate(mock_numpy_frame, det, labels=["#1 0.90"])
        assert result.shape == mock_numpy_frame.shape

    def test_build_labels_with_tracker(self):
        import supervision as sv
        det = sv.Detections(
            xyxy=np.array([[10, 10, 50, 50]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        det.tracker_id = np.array([42])
        labels = FrameAnnotator.build_labels(det)
        assert len(labels) == 1
        assert "42" in labels[0]

    def test_build_labels_no_tracker(self):
        import supervision as sv
        det = sv.Detections(
            xyxy=np.array([[10, 10, 50, 50]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        det.tracker_id = None
        labels = FrameAnnotator.build_labels(det)
        assert "?" in labels[0]
