import sys
from unittest.mock import MagicMock

class MockSupervision:
    class Detections:
        def __init__(self, xyxy, confidence, class_id, tracker_id=None):
            self.xyxy = xyxy
            self.confidence = confidence
            self.class_id = class_id
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
            return cls([], [], [])
        
        @classmethod
        def merge(cls, dets_list):
            return cls([], [], [])
            
        def __len__(self):
            return len(self.xyxy)

    def box_iou_batch(self, box1, box2):
        return [[0.5]]

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

# Mock cv2 if not installed (headless environments)
try:
    import cv2
except ImportError:
    mock_cv2 = MagicMock()
    sys.modules["cv2"] = mock_cv2
