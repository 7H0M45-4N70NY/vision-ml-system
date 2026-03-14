"""Tests for InferencePipeline."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestInferencePipeline:
    @patch('src.vision_ml.inference.pipeline.DriftDetector')
    @patch('src.vision_ml.inference.pipeline.AutoLabeler')
    @patch('src.vision_ml.inference.pipeline.VisitorAnalytics')
    @patch('src.vision_ml.inference.pipeline.FrameAnnotator')
    @patch('src.vision_ml.inference.pipeline.TrackerFactory')
    @patch('src.vision_ml.inference.pipeline.DetectorFactory')
    def _make_pipeline(self, config, mock_det_factory, mock_tracker_factory,
                       mock_annotator_cls, mock_analytics_cls, mock_labeler_cls,
                       mock_drift_cls):
        """Helper to create a pipeline with all dependencies mocked."""
        import supervision as sv

        # Mock detector
        mock_detector = MagicMock()
        det = sv.Detections(
            xyxy=np.array([[10, 10, 50, 50]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        det.tracker_id = np.array([1])
        mock_detector.detect.return_value = det
        mock_det_factory.from_config.return_value = mock_detector

        # Mock tracker
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = det
        mock_tracker_factory.from_config.return_value = mock_tracker

        # Mock annotator
        mock_ann = MagicMock()
        mock_ann.annotate.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_annotator_cls.return_value = mock_ann
        mock_annotator_cls.build_labels = MagicMock(return_value=["#1 0.90"])

        # Mock analytics
        mock_analytics = MagicMock()
        mock_analytics_cls.return_value = mock_analytics

        # Mock labeler
        mock_labeler = MagicMock()
        mock_labeler.pending_labels = []
        mock_labeler_cls.return_value = mock_labeler

        # Mock drift detector
        mock_drift = MagicMock()
        mock_drift_cls.return_value = mock_drift

        from src.vision_ml.inference.pipeline import InferencePipeline
        pipeline = InferencePipeline(config)
        return pipeline, {
            'detector': mock_detector,
            'tracker': mock_tracker,
            'annotator': mock_ann,
            'analytics': mock_analytics,
            'labeler': mock_labeler,
            'drift': mock_drift,
        }

    def test_init(self, sample_config):
        pipeline, _ = self._make_pipeline(sample_config)
        assert pipeline is not None

    def test_process_frame(self, sample_config):
        pipeline, mocks = self._make_pipeline(sample_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections, annotated = pipeline.process_frame(frame, frame_idx=0)
        assert detections is not None
        assert annotated is not None
        mocks['detector'].detect.assert_called_once()

    def test_reset(self, sample_config):
        pipeline, mocks = self._make_pipeline(sample_config)
        pipeline.reset()
        mocks['tracker'].reset.assert_called_once()
        mocks['analytics'].reset.assert_called_once()
        mocks['drift'].reset.assert_called_once()

    def test_flush_labels_empty(self, sample_config):
        pipeline, mocks = self._make_pipeline(sample_config)
        count = pipeline.flush_labels()
        assert count == 0

    def test_tracker_disabled(self, sample_config):
        sample_config['tracking']['enabled'] = False
        pipeline, mocks = self._make_pipeline(sample_config)
        assert pipeline.tracker is None
