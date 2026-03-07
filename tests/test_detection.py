"""
Tests for the detection module.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection import Detector

def test_detector_initialization():
    """
    Tests that the Detector class can be initialized.
    """
    detector = Detector()
    assert detector.model_name == "RF-DETR"
