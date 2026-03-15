from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import asyncio
from typing import Any, Dict, Optional

from ..inference.pipeline import InferencePipeline
from ..utils.config import load_config
from .schemas import InferenceRequest, InferenceResponse, HealthCheck, ConfigUpdate, TriageAction, StreamConfig, PipelineToggles
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import time
import cv2
import numpy as np
import os
from ..logging import get_logger

from ..analytics.analytics_db import AnalyticsDB

logger = get_logger(__name__)

db = AnalyticsDB() # Persistent analytics storage

app = FastAPI(
    title="Vision ML Inference API",
    description="Real-time object detection and tracking API",
    version="1.0.0"
)

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline and stream state
pipeline: Optional[InferencePipeline] = None
current_stream_source: str = "0"
pipeline_state = {
    "enable_detection": True,
    "enable_tracking": True,
    "show_annotations": True,
    "stream_active": True
}

current_telemetry = {
    "fps": 0,
    "latency": 0,
    "objectCount": 0,
    "gpuUtilization": 0,
    "modelLoaded": True,
    "isConnected": True
}

@app.on_event("startup")
async def startup_event():
    global pipeline
    config = load_config('config/inference/base.yaml')
    # Set to API mode (no visualization)
    config['mode']['show_live'] = False
    pipeline = InferencePipeline(config)
    logger.info("Inference Pipeline loaded")

# Mount data/low_confidence_frames for triage image serving
data_dir = os.path.abspath(os.path.join(os.getcwd(), "data", "low_confidence_frames"))
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
app.mount("/data/frames", StaticFiles(directory=data_dir), name="frames")

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {
        "status": "healthy", 
        "version": "1.0.0"
    }

@app.post("/predict/video", response_model=Dict[str, Any])
async def predict_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a video file for offline inference.
    Returns the analytics summary.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        summary = pipeline.run_offline(temp_path)

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict/stream")
async def start_stream(request: InferenceRequest):
    """
    Start processing a stream (RTSP/Camera).
    Note: This is a placeholder. In a real microservice, this would start a background worker.
    """
    return {"message": "Stream processing started (Not implemented in MVP API wrapper yet)"}

import threading

class ThreadedVideoCapture:
    """Threaded capture: isolates the camera driver from the inference loop."""

    # On Windows, DirectShow (DSHOW) is far more stable than MSMF for webcams.
    # Fall back to CAP_ANY for RTSP / file sources where DSHOW is irrelevant.
    _WEBCAM_BACKEND = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    _MAX_RECONNECT_DELAY = 30.0  # cap exponential backoff at 30 s

    def __init__(self, source):
        self.source = source
        self.cap = None
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.error_count = 0
        self.reconnect_delay = 1.0
        self._reconnect_attempts = 0

        self.start_capture()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _try_open(self, source_val, backend) -> bool:
        """Attempt to open a capture device; returns True on success."""
        try:
            cap = cv2.VideoCapture(source_val, backend)
            if cap.isOpened():
                self.cap = cap
                return True
            cap.release()
        except Exception:
            pass
        return False

    def start_capture(self):
        source_val = int(self.source) if str(self.source).isdigit() else self.source
        is_index   = isinstance(source_val, int)

        opened = False
        if is_index:
            # On Windows try DSHOW first (lower latency), fall back to MSMF
            for backend in ([cv2.CAP_DSHOW, cv2.CAP_MSMF] if os.name == 'nt' else [cv2.CAP_ANY]):
                if self._try_open(source_val, backend):
                    logger.info("Camera opened: source=%s backend=%s", self.source, backend)
                    opened = True
                    break
        else:
            opened = self._try_open(source_val, cv2.CAP_ANY)

        if opened:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._reconnect_attempts = 0
            self.reconnect_delay = 1.0
        else:
            self._reconnect_attempts += 1
            self.reconnect_delay = min(
                2 ** self._reconnect_attempts, self._MAX_RECONNECT_DELAY
            )
            if self._reconnect_attempts <= 3 or self._reconnect_attempts % 5 == 0:
                logger.warning(
                    "Camera unavailable (source=%s, attempt=%d). Retrying in %.0fs…",
                    self.source, self._reconnect_attempts, self.reconnect_delay,
                )

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(self.reconnect_delay)
                self.start_capture()
                continue

            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.frame = frame
                self.error_count = 0
            else:
                self.error_count += 1
                if self.error_count > 10:
                    # Only log on first invalidation, not every cycle
                    if self.error_count == 11:
                        logger.warning("Device read failing (source=%s). Resetting driver…", self.source)
                    self.cap.release()
                    self.error_count = 0
                    time.sleep(self.reconnect_delay)

            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame is not None, self.frame

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

def feed_generator():
    global pipeline, current_telemetry, current_stream_source, pipeline_state
    
    # Wait for pipeline to be initialized
    retry_count = 0
    while pipeline is None and retry_count < 30:
        time.sleep(1)
        retry_count += 1
        
    if pipeline is None:
        logger.error("Pipeline failed to initialize")
        return
        
    last_source = None
    threaded_cap = None
    frame_idx = 0
    
    try:
        while True:
            # Handle Source Change
            if last_source != current_stream_source:
                if threaded_cap is not None:
                    threaded_cap.release()
                
                logger.info(f"Connecting to source: {current_stream_source}...")
                threaded_cap = ThreadedVideoCapture(current_stream_source)
                last_source = current_stream_source

            if not pipeline_state.get("stream_active", True):
                if threaded_cap:
                    threaded_cap.release()
                    threaded_cap = None
                
                # Telemetry reset
                current_telemetry["fps"] = 0
                current_telemetry["latency"] = 0
                
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "OFFLINE", (260, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
                continue

            start_time = time.time()
            success, frame = threaded_cap.read() if threaded_cap else (False, None)
            
            if not success or frame is None:
               # Show 'reconnecting' UI if frame is missing
               fail_frame = np.zeros((480, 640, 3), dtype=np.uint8)
               cv2.putText(fail_frame, "RECONNECTING CAMERA...", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
               ret, buffer = cv2.imencode('.jpg', fail_frame)
               if ret:
                   yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
               time.sleep(0.5)
               continue
                
            # Copy frame to avoid thread race conditions during processing
            process_frame = frame.copy()
            
            # Apply pipeline toggles
            if pipeline_state["enable_detection"]:
                detections, annotated = pipeline.process_frame(process_frame, frame_idx)
                obj_len = len(detections) if detections is not None else 0
            else:
                annotated = process_frame
                obj_len = 0
                
            if not pipeline_state["show_annotations"]:
                annotated = process_frame

            frame_idx += 1
            
            # Update telemetry
            latency = int((time.time() - start_time) * 1000)
            current_telemetry["latency"] = latency
            current_telemetry["objectCount"] = obj_len
            current_telemetry["fps"] = int(1000 / latency) if latency > 0 else 0
            
            if pipeline_state["enable_detection"] and detections is not None and len(detections) > 0:
                current_telemetry["avgConfidence"] = float(np.mean(detections.confidence))
            else:
                current_telemetry["avgConfidence"] = 0.0
            
            ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Ensure we don't blast the network faster than the browser can render
            time.sleep(0.01)
            
    finally:
        if threaded_cap:
            threaded_cap.release()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(feed_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/live-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(0.5)
            await websocket.send_json(current_telemetry)
    except WebSocketDisconnect:
        logger.info("Client disconnected from live stream")

@app.get("/config")
async def get_config():
    if pipeline:
        return pipeline.config
    return load_config('config/inference/base.yaml')

@app.patch("/config")
async def update_config(update: ConfigUpdate):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    if update.confidence_threshold is not None:
        # Update YOLO detector threshold
        # Assuming the detector has a way to update its threshold
        if hasattr(pipeline.detector, 'model'):
            pipeline.detector.model.conf = update.confidence_threshold
            logger.info(f"Updated YOLO confidence threshold to {update.confidence_threshold}")
            
    if update.iou_threshold is not None:
         if hasattr(pipeline.detector, 'model'):
            pipeline.detector.model.iou = update.iou_threshold
            logger.info(f"Updated YOLO IoU threshold to {update.iou_threshold}")
            
    if update.dual_mode is not None:
        # Assuming pipeline has a way to toggle dual mode
        # In our implementation, DualDetector is initialized at factory level
        # but let's assume we can update a state in it
        if hasattr(pipeline.detector, 'dual_mode'):
            pipeline.detector.dual_mode = update.dual_mode
            logger.info(f"Updated DualDetector mode: {update.dual_mode}")

    return {"status": "Config updated successfully", "applied": update.dict(exclude_none=True)}

@app.post("/stream/switch")
async def switch_stream(config: StreamConfig):
    global current_stream_source
    current_stream_source = config.source
    return {"status": "Stream switching requested", "source": current_stream_source}

@app.patch("/config/toggles")
async def update_toggles(toggles: PipelineToggles):
    global pipeline_state
    if toggles.enable_detection is not None:
        pipeline_state["enable_detection"] = toggles.enable_detection
    if toggles.enable_tracking is not None:
        pipeline_state["enable_tracking"] = toggles.enable_tracking
    if toggles.show_annotations is not None:
        pipeline_state["show_annotations"] = toggles.show_annotations
    if toggles.stream_active is not None:
        pipeline_state["stream_active"] = toggles.stream_active
        
    return {"status": "Toggles updated", "state": pipeline_state}

@app.post("/analytics/reset")
async def reset_analytics():
    if pipeline:
        pipeline.reset()
    return {"status": "Analytics indices reset"}

@app.get("/triage/frames")
async def get_triage_frames():
    """
    Scans data/low_confidence_frames and returns metadata for all captured frames.
    """
    frames = []
    target_dir = os.path.join("data", "low_confidence_frames")
    if not os.path.exists(target_dir):
        return {"frames": []}
    
    # Sort by filename descending to show newest first
    all_files = sorted(os.listdir(target_dir), reverse=True)
    
    import json
    for f in all_files:
        if f.endswith(".json"):
            base_id = f.replace(".json", "")
            img_file = f"{base_id}.jpg"
            if img_file in all_files:
                try:
                    with open(os.path.join(target_dir, f), 'r') as jf:
                        meta = json.load(jf)
                    frames.append({
                        "id": base_id,
                        "timestamp": meta.get("timestamp", time.time()),
                        "reason": meta.get("reason", "Unknown"),
                        "confidence": meta.get("confidence", 0.0),
                        "class": meta.get("class", "unknown"),
                        "imageUrl": f"/data/frames/{img_file}"
                    })
                except Exception as e:
                    logger.error(f"Error reading metadata {f}: {e}")
                    
    return {"frames": frames[:50]} # Return latest 50

@app.post("/triage/action")
async def perform_triage_action(action: TriageAction):
    """
    Execute a triage action on one or more captured frames.
    - reject:  delete jpg + json (discard bad frames)
    - accept:  move jpg + json to data/auto_labeled/ (clean labels for training)
    - label:   same as accept, marks frame as pending human labeling
    """
    import json as _json

    src_dir   = os.path.join("data", "low_confidence_frames")
    auto_dir  = os.path.join("data", "auto_labeled")
    img_dir   = os.path.join(auto_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    processed, errors = 0, []

    for frame_id in action.frame_ids:
        jpg  = os.path.join(src_dir, f"{frame_id}.jpg")
        meta = os.path.join(src_dir, f"{frame_id}.json")

        try:
            if action.action == "reject":
                for fp in (jpg, meta):
                    if os.path.exists(fp):
                        os.remove(fp)

            elif action.action in ("accept", "label"):
                # Move image
                if os.path.exists(jpg):
                    shutil.move(jpg, os.path.join(img_dir, f"{frame_id}.jpg"))

                # Append metadata to auto_labels.json
                auto_labels_path = os.path.join(auto_dir, "auto_labels.json")
                existing: list = []
                if os.path.exists(auto_labels_path):
                    with open(auto_labels_path) as f:
                        try:
                            existing = _json.load(f)
                        except Exception:
                            existing = []

                if os.path.exists(meta):
                    with open(meta) as f:
                        lbl = _json.load(f)
                    lbl["image_path"] = os.path.join(img_dir, f"{frame_id}.jpg")
                    lbl["image_id"]   = frame_id
                    lbl["source"]     = action.action
                    existing.append(lbl)
                    os.remove(meta)

                with open(auto_labels_path, "w") as f:
                    _json.dump(existing, f, indent=2)

            processed += 1
        except Exception as e:
            errors.append({"frame_id": frame_id, "error": str(e)})
            logger.error("Triage action %s failed for %s: %s", action.action, frame_id, e)

    return {
        "status": "ok",
        "action": action.action,
        "processed": processed,
        "errors": errors,
    }

@app.get("/analytics/stats")
async def get_analytics_stats():
    return db.get_analytics_summary()

@app.get("/analytics/timeseries")
async def get_analytics_timeseries():
    runs = db.get_inference_runs(limit=30)
    # Convert to format suitable for charts
    drift_data = []
    visitor_data = []
    for run in reversed(runs):
        timestamp = run['timestamp']
        if isinstance(timestamp, str):
             # Try to simplify timestamp for display
             try:
                 timestamp = timestamp.split(' ')[1] # Just the time
             except: pass
             
        drift_data.append({"date": timestamp, "score": run.get('drift_score', 0)})
        visitor_data.append({"date": timestamp, "visitors": run.get('unique_visitors', 0)})
        
    return {"drift": drift_data, "visitors": visitor_data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
