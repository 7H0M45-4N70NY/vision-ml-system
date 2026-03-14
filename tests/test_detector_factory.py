"""Tests for DetectorFactory."""
import pytest
from unittest.mock import MagicMock
from src.vision_ml.detection.detector_factory import DetectorFactory, _REGISTRY


class TestDetectorFactory:
    def test_list_available(self):
        available = DetectorFactory.list_available()
        assert 'yolo11n' in available
        assert 'rfdetr' in available

    def test_get_calls_registry_class(self, sample_config):
        """Verify factory returns an instance from the registry."""
        mock_cls = MagicMock(return_value=MagicMock())
        original = _REGISTRY['yolo11n']
        _REGISTRY['yolo11n'] = mock_cls
        try:
            detector = DetectorFactory.get('yolo11n', sample_config)
            assert detector is not None
            mock_cls.assert_called_once_with(sample_config)
        finally:
            _REGISTRY['yolo11n'] = original

    def test_from_config(self, sample_config):
        """from_config reads detector_type from config."""
        mock_cls = MagicMock(return_value=MagicMock())
        original = _REGISTRY['yolo11n']
        _REGISTRY['yolo11n'] = mock_cls
        try:
            detector = DetectorFactory.from_config(sample_config)
            assert detector is not None
        finally:
            _REGISTRY['yolo11n'] = original

    def test_singleton_caching(self, sample_config):
        """Same key returns same instance (ModelRegistry caching)."""
        sentinel = MagicMock()
        mock_cls = MagicMock(return_value=sentinel)
        original = _REGISTRY['yolo11n']
        _REGISTRY['yolo11n'] = mock_cls
        try:
            d1 = DetectorFactory.get('yolo11n', sample_config)
            d2 = DetectorFactory.get('yolo11n', sample_config)
            assert d1 is d2
            assert mock_cls.call_count == 1
        finally:
            _REGISTRY['yolo11n'] = original

    def test_invalid_type_raises(self, sample_config):
        with pytest.raises(ValueError, match="Unknown detector"):
            DetectorFactory.get('nonexistent', sample_config)
