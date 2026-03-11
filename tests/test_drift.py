"""
Tests for the drift detector module.
"""
import pytest
from src.vision_ml.training.drift_detector import DriftDetector

class MockConfig:
    def get(self, key, default=None):
        if key == 'drift':
            return {
                'enabled': True,
                'method': 'confidence_drop',
                'confidence_threshold': 0.5,
                'window_size': 10,
                'check_interval': 2
            }
        return default

def test_drift_detector_initialization():
    config = {'drift': {'enabled': True}}
    detector = DriftDetector(config)
    assert detector.enabled is True
    assert detector.method == 'confidence_drop'

def test_drift_record_and_check():
    config = {
        'drift': {
            'enabled': True,
            'confidence_threshold': 0.5,
            'window_size': 5,
            'check_interval': 1
        }
    }
    detector = DriftDetector(config)
    
    # High confidence - no drift
    detector.record([0.9, 0.8])
    assert detector.check() is False
    
    # Low confidence - drift
    detector.record([0.1, 0.2, 0.1])
    # Average should be (0.9+0.8+0.1+0.2+0.1)/5 = 0.42 < 0.5
    assert detector.check() is True

def test_drift_metrics():
    config = {'drift': {'enabled': True}}
    detector = DriftDetector(config)
    detector.record([0.8])
    metrics = detector.get_metrics()
    
    assert 'avg_confidence' in metrics
    assert 'drift_score' in metrics
    assert metrics['total_detections'] == 1
