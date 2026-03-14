# --- Vision ML System: Base Detector ---
"""Abstract Base Class for all object detectors in the system."""

from abc import ABC, abstractmethod
import numpy as np
import supervision as sv
from typing import List, Dict, Any


class BaseDetector(ABC):
    """Interface contract for object detectors.
    
    All custom detectors must inherit from this class and return
    `supervision.Detections` to ensure pipeline compatibility.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize detector with configuration."""
        self.config = config

    @abstractmethod
    def load_model(self) -> None:
        """Loads the model weights into memory."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> sv.Detections:
        """Performs inference on a single image.
        
        Args:
            image: A BGR image array (OpenCV format).
            
        Returns:
            sv.Detections object containing bounding boxes, confidences, and class IDs.
        """
        pass

    def detect_batch(self, images: List[np.ndarray]) -> List[sv.Detections]:
        """Performs inference on a batch of images.
        
        Default implementation falls back to sequential processing.
        Override this for optimized batch inference (e.g., YOLO batch).
        
        Args:
            images: List of BGR image arrays.
            
        Returns:
            List of sv.Detections objects.
        """
        return [self.detect(img) for img in images]
