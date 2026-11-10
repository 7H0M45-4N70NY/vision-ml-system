import warnings

# Suppress known non-actionable deprecation warnings from third-party ML dependencies
warnings.filterwarnings("ignore", message=r".*['\"]?half['\"]? is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ultralytics.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"ultralytics.*")

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
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Instrumentator = None
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import time
import cv2
import numpy as np
import os
import supervision as sv
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
if PROMETHEUS_AVAILABLE and Instrumentator is not None:
    Instrumentator().instrument(app).expose(app)
else:
    logger.warning("prometheus_fastapi_instrumentator is not installed. /metrics endpoint will not be exposed.")

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
    "uniqueVisitors": 0,
    "avgDwellTime": 0.0,
    "gpuUtilization": 0,
    "modelLoaded": True,
    "isConnected": True,
    "avgConfidence": 0.0
}

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        config = load_config('config/inference/base.yaml')
        # Set to API mode (no visualization)
        config['mode']['show_live'] = False
        pipeline = InferencePipeline(config)
        logger.info("Inference Pipeline loaded")
    except Exception as e:
        logger.warning(f"Could not initialize default inference pipeline on startup: {e}")

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

import threading as _threading
from vidgear.gears import CamGear

# ── Shared capture singleton (B-08 fix) ───────────────────────────────────────
# All /video_feed connections share one VidGearCapture.  DSHOW on Windows only
# grants exclusive access to one handle — multiple handles = second connection
# fails.  Ref-counting ensures the capture stays alive while any client is
# connected and is cleanly released when the last client disconnects.
_capture: Optional["VidGearCapture"] = None
_capture_source: str = ""
_capture_lock = _threading.Lock()
_capture_refs: int = 0


def _acquire_capture(source: str) -> "VidGearCapture":
    """Return the shared VidGearCapture, (re)opening it if the source changed."""
    global _capture, _capture_source, _capture_refs
    s_clean = str(source).strip()
    with _capture_lock:
        if _capture is None or _capture_source != s_clean:
            if _capture is not None:
                _capture.release()
            logger.info("Opening shared capture for source: %s", s_clean)
            _capture = VidGearCapture(s_clean)
            _capture_source = s_clean
        _capture_refs += 1
        return _capture


def _release_capture() -> None:
    """Decrement ref count; destroy the shared capture when no readers remain."""
    global _capture, _capture_refs
    with _capture_lock:
        _capture_refs = max(0, _capture_refs - 1)
        if _capture_refs == 0 and _capture is not None:
            logger.info("Releasing shared capture for source: %s", _capture_source)
            _capture.release()
            _capture = None


class VidGearCapture:
    """Video capture via VidGear CamGear.

    Replaces the hand-rolled ThreadedVideoCapture.  VidGear manages its own
    capture thread and handles platform quirks (Windows DSHOW/MSMF) internally.
    Exposes the same read() / release() interface so feed_generator() is unchanged.
    """

    _MAX_RECONNECT_DELAY = 15.0
    _NULL_FRAME_THRESHOLD = 15

    def __init__(self, source):
        self.source = str(source).strip()
        self.resolved_source = self._resolve_source(self.source)
        self._stream: Optional[CamGear] = None
        self._reconnect_attempts = 0
        self.reconnect_delay = 1.0
        self._last_attempt = 0.0
        self._null_count = 0
        self._open()

    @staticmethod
    def _resolve_source(source: str):
        s = str(source).strip()
        if s.isdigit():
            return int(s)
        if s.startswith(("rtsp://", "http://", "https://")):
            return s
        # Remove surrounding quotes
        s = s.strip('"\'')
        if os.path.exists(s):
            return os.path.abspath(s)
        # Check relative to cwd or data/sample_videos
        candidates = [
            os.path.join(os.getcwd(), s),
            os.path.join(os.getcwd(), "data", "sample_videos", os.path.basename(s)),
            os.path.join(os.getcwd(), "data", os.path.basename(s)),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return s

    def _open(self):
        source_val = self.resolved_source
        is_webcam = isinstance(source_val, int)

        options = {
            "CAP_PROP_BUFFERSIZE": 1,
        }
        if is_webcam:
            options["CAP_PROP_FRAME_WIDTH"] = 640
            options["CAP_PROP_FRAME_HEIGHT"] = 480
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        else:
            backend = cv2.CAP_ANY

        self._last_attempt = time.time()
        try:
            self._stream = CamGear(source=source_val, backend=backend, logging=False, **options).start()
            self._reconnect_attempts = 0
            self._null_count = 0
            self.reconnect_delay = 1.0
            logger.info("CamGear opened: source=%s (resolved=%s)", self.source, source_val)
        except Exception as exc:
            self._stream = None
            self._reconnect_attempts += 1
            self.reconnect_delay = min(2 ** self._reconnect_attempts, self._MAX_RECONNECT_DELAY)
            if self._reconnect_attempts <= 3 or self._reconnect_attempts % 5 == 0:
                logger.warning(
                    "CamGear unavailable (source=%s, attempt=%d): %s. Retry in %.0fs…",
                    self.source, self._reconnect_attempts, exc, self.reconnect_delay,
                )

    def read(self):
        source_val = self.resolved_source
        is_file = isinstance(source_val, str) and not source_val.startswith(("rtsp://", "http://", "https://"))

        # Attempt reconnect after backoff window
        if self._stream is None:
            if time.time() - self._last_attempt >= self.reconnect_delay:
                self._open()
            return False, None

        frame = self._stream.read()
        if frame is None:
            self._null_count += 1
            if is_file:
                # Video file reached EOF: seamlessly loop by restarting from frame 0
                self._null_count = 0
                self.release()
                self._open()
                if self._stream is not None:
                    loop_frame = self._stream.read()
                    if loop_frame is not None:
                        return True, loop_frame
                return False, None

            if self._null_count < self._NULL_FRAME_THRESHOLD:
                return False, None

            self._null_count = 0
            self.release()
            self._reconnect_attempts += 1
            self.reconnect_delay = min(2 ** self._reconnect_attempts, self._MAX_RECONNECT_DELAY)
            if self._reconnect_attempts <= 3 or self._reconnect_attempts % 5 == 0:
                logger.warning(
                    "CamGear read failed (source=%s, attempt=%d). Reconnecting in %.0fs…",
                    self.source, self._reconnect_attempts, self.reconnect_delay,
                )
            return False, None

        self._null_count = 0
        return True, frame

    def release(self):
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream = None

async def feed_generator(requested_source: Optional[str] = None):
    """Async MJPEG generator (B-07 fix).

    - All await asyncio.sleep() calls yield control back to the event loop
      so other requests/WebSocket messages are not starved.
    - CPU-bound inference runs in the default thread-pool executor so it
      never blocks the event loop thread.
    - Reads from the module-level singleton VidGearCapture (B-08 fix) so
      multiple simultaneous /video_feed connections share one camera handle.
    """
    global pipeline, current_telemetry, current_stream_source, pipeline_state

    # Wait for pipeline to be ready without blocking the event loop
    retry_count = 0
    while pipeline is None and retry_count < 30:
        await asyncio.sleep(1)
        retry_count += 1

    if pipeline is None:
        logger.error("Pipeline failed to initialize")
        return

    loop = asyncio.get_event_loop()
    last_source: str = ""
    cap: Optional[VidGearCapture] = None
    frame_idx = 0

    try:
        while True:
            target_source = requested_source if requested_source is not None else current_stream_source

            if not pipeline_state.get("stream_active", True):
                # When stream is turned off, release camera hardware so it's truly off
                if cap is not None:
                    _release_capture()
                    cap = None
                    last_source = ""
                current_telemetry["fps"] = 0
                current_telemetry["latency"] = 0
                current_telemetry["objectCount"] = 0
                black = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black, "STREAM TURNED OFF", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
                ret, buf = cv2.imencode(".jpg", black)
                if ret:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
                await asyncio.sleep(0.5)
                continue

            # Stream is active: acquire / reuse capture; reopen on source change
            if last_source != target_source or cap is None:
                if cap is not None:
                    _release_capture()
                cap = _acquire_capture(target_source)
                last_source = target_source

            start_time = time.time()
            success, frame = cap.read() if cap else (False, None)

            if not success or frame is None:
                fail = np.zeros((480, 640, 3), dtype=np.uint8)
                src_display = str(target_source)
                if src_display.isdigit():
                    msg = f"WEBCAM ({src_display}) NOT DETECTED"
                    hint = "Check camera or switch to Video File / RTSP"
                elif src_display.startswith(("rtsp://", "http://", "https://")):
                    msg = "RTSP STREAM CONNECTING / UNAVAILABLE"
                    hint = src_display[:45]
                else:
                    msg = "VIDEO SOURCE NOT FOUND"
                    hint = os.path.basename(src_display)

                cv2.putText(fail, msg, (120, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                cv2.putText(fail, hint, (100, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                ret, buf = cv2.imencode(".jpg", fail)
                if ret:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
                await asyncio.sleep(0.5)
                continue

            process_frame = frame.copy()

            # Run inference in thread pool — keeps event loop free (B-07)
            if pipeline_state["enable_detection"]:
                try:
                    detections, annotated = await loop.run_in_executor(
                        None, pipeline.process_frame, process_frame, frame_idx
                    )
                except Exception as exc:
                    logger.error(
                        "Unexpected pipeline error at frame %d (%s: %s); serving raw frame.",
                        frame_idx, type(exc).__name__, exc,
                    )
                    detections, annotated = sv.Detections.empty(), process_frame
                obj_len = len(detections) if detections is not None else 0
            else:
                detections, annotated, obj_len = None, process_frame, 0

            if not pipeline_state["show_annotations"]:
                annotated = process_frame

            frame_idx += 1

            latency = int((time.time() - start_time) * 1000)
            current_telemetry["latency"] = latency
            current_telemetry["objectCount"] = obj_len
            current_telemetry["fps"] = int(1000 / latency) if latency > 0 else 0
            current_telemetry["avgConfidence"] = (
                float(np.mean(detections.confidence))
                if detections is not None and len(detections) > 0 else 0.0
            )

            # Extract real-time visitor tracking and dwell time metrics
            if pipeline is not None and hasattr(pipeline, 'analytics'):
                current_telemetry["uniqueVisitors"] = pipeline.analytics.unique_visitor_count
                summary_metrics = pipeline.analytics.get_summary()
                current_telemetry["avgDwellTime"] = summary_metrics.get("avg_dwell_time_seconds", 0.0)

            ret, buf = cv2.imencode(".jpg", annotated,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ret:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")

            await asyncio.sleep(0.01)

    finally:
        if cap is not None:
            _release_capture()


@app.get("/video_feed")
async def video_feed(source: Optional[str] = None):
    return StreamingResponse(
        feed_generator(requested_source=source),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

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

@app.get("/stream/sources")
async def get_stream_sources():
    """List available sources (webcams, sample videos, rtsp presets) and current status."""
    global current_stream_source, pipeline_state
    
    sample_videos = []
    videos_dir = os.path.join(os.getcwd(), "data", "sample_videos")
    if os.path.exists(videos_dir):
        for f in sorted(os.listdir(videos_dir)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                sample_videos.append({
                    "name": f,
                    "path": f"data/sample_videos/{f}"
                })
                
    return {
        "current_source": current_stream_source,
        "stream_active": pipeline_state.get("stream_active", True),
        "available_sources": {
            "webcam": [
                {"name": "Camera 0 (Default)", "source": "0"},
                {"name": "Camera 1 (Secondary)", "source": "1"}
            ],
            "sample_videos": sample_videos,
            "rtsp": [
                {"name": "Local RTSP Stream", "source": "rtsp://localhost:8554/live"}
            ]
        }
    }

@app.get("/stream/status")
async def get_stream_status():
    """Get real-time status of stream and pipeline."""
    return {
        "current_source": current_stream_source,
        "stream_active": pipeline_state.get("stream_active", True),
        "pipeline_state": pipeline_state,
        "telemetry": current_telemetry
    }

@app.post("/stream/switch")
async def switch_stream(config: StreamConfig):
    global current_stream_source, pipeline_state, _capture, _capture_refs
    new_src = config.source.strip()
    current_stream_source = new_src
    pipeline_state["stream_active"] = True

    # Force release of current capture under lock so new source opens immediately
    with _capture_lock:
        if _capture is not None:
            _capture.release()
            _capture = None
            _capture_refs = 0

    logger.info("Switched stream source to: %s", current_stream_source)
    return {
        "status": "Stream switching requested",
        "source": current_stream_source,
        "stream_active": pipeline_state["stream_active"]
    }

@app.patch("/config/toggles")
async def update_toggles(toggles: PipelineToggles):
    global pipeline_state, _capture, _capture_refs
    if toggles.enable_detection is not None:
        pipeline_state["enable_detection"] = toggles.enable_detection
    if toggles.enable_tracking is not None:
        pipeline_state["enable_tracking"] = toggles.enable_tracking
    if toggles.show_annotations is not None:
        pipeline_state["show_annotations"] = toggles.show_annotations
    if toggles.stream_active is not None:
        pipeline_state["stream_active"] = toggles.stream_active
        if not toggles.stream_active:
            # When stream is turned off, release the capture device immediately
            with _capture_lock:
                if _capture is not None:
                    _capture.release()
                    _capture = None
                    _capture_refs = 0
            logger.info("Stream turned off, hardware capture released.")
        else:
            logger.info("Stream turned on.")
        
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

@app.get("/analytics/runs")
async def get_analytics_runs(limit: int = 50):
    """Retrieve list of recent inference runs from database."""
    runs = db.get_inference_runs(limit=limit)
    return {"runs": runs}

@app.get("/analytics/visitor-flow")
async def get_visitor_flow():
    """Retrieve hourly visitor flow distribution for heatmap / bar chart."""
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT strftime('%H:00', timestamp) as hour, SUM(unique_visitors) as visitors
            FROM inference_runs
            GROUP BY hour
            ORDER BY hour
        """)
        rows = cursor.fetchall()
        
    flow_map = {row['hour']: row['visitors'] for row in rows} if rows else {}
    standard_hours = ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00"]
    result = []
    for h in standard_hours:
        result.append({
            "hour": h,
            "visitors": flow_map.get(h, 0)
        })
    # If we have runs in hours outside standard_hours, append them
    for h, count in flow_map.items():
        if h not in standard_hours:
            result.append({"hour": h, "visitors": count})
    result.sort(key=lambda x: x["hour"])
    return {"flow": result}

@app.get("/analytics/dwell-distribution")
async def get_dwell_distribution():
    """Retrieve real engagement tier breakdown based on duration_seconds in visitor_analytics."""
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN duration_seconds < 2.0 THEN 1 ELSE 0 END) as quick_transit,
                SUM(CASE WHEN duration_seconds >= 2.0 AND duration_seconds < 5.0 THEN 1 ELSE 0 END) as short_stay,
                SUM(CASE WHEN duration_seconds >= 5.0 AND duration_seconds < 10.0 THEN 1 ELSE 0 END) as browsing,
                SUM(CASE WHEN duration_seconds >= 10.0 THEN 1 ELSE 0 END) as high_engagement,
                COUNT(*) as total
            FROM visitor_analytics
        """)
        row = cursor.fetchone()
        
    if row and row[4] and row[4] > 0:
        return {
            "distribution": [
                {"name": "Transit (< 2s)", "value": row[0] or 0},
                {"name": "Short Stay (2-5s)", "value": row[1] or 0},
                {"name": "Browsing (5-10s)", "value": row[2] or 0},
                {"name": "High Engagement (> 10s)", "value": row[3] or 0},
            ],
            "total": row[4]
        }
    return {
        "distribution": [
            {"name": "Transit (< 2s)", "value": 0},
            {"name": "Short Stay (2-5s)", "value": 0},
            {"name": "Browsing (5-10s)", "value": 0},
            {"name": "High Engagement (> 10s)", "value": 0},
        ],
        "total": 0
    }

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
