from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class InferenceRequest(BaseModel):
    source: str
    source_type: str = "video"  # video, rtsp, webcam
    mode: str = "offline"       # online, offline

class DetectionResult(BaseModel):
    class_id: List[int]
    confidence: List[float]
    xyxy: List[List[float]]
    tracker_id: Optional[List[int]] = None

class InferenceResponse(BaseModel):
    frame_id: int
    detections: DetectionResult
    analytics: Dict[str, Any]
    drift_metrics: Optional[Dict[str, Any]] = None

class HealthCheck(BaseModel):
    status: str
    version: str
