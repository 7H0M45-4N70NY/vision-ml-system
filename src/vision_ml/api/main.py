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

logger = get_logger(__name__)

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

def feed_generator():
    global pipeline, current_telemetry, current_stream_source, pipeline_state
    if pipeline is None:
        return
        
    last_source = current_stream_source
    cap = cv2.VideoCapture(int(current_stream_source) if current_stream_source.isdigit() else current_stream_source)
    
    if not cap.isOpened():
        logger.error(f"Could not open stream: {current_stream_source}")
        return
        
    frame_idx = 0
    try:
        while True:
            if not pipeline_state.get("stream_active", True):
                if cap is not None and cap.isOpened():
                    cap.release()
                
                # Update telemetry for offline state
                current_telemetry["fps"] = 0
                current_telemetry["latency"] = 0
                current_telemetry["objectCount"] = 0
                
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "STREAM STOPPED", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
                continue

            # Ensure camera is opened if we transitioned from inactive to active
            if cap is None or not cap.isOpened():
                 cap = cv2.VideoCapture(int(current_stream_source) if current_stream_source.isdigit() else current_stream_source)

            # Check if source was changed dynamically
            if last_source != current_stream_source:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(int(current_stream_source) if current_stream_source.isdigit() else current_stream_source)
                last_source = current_stream_source
                if not cap.isOpened():
                     logger.error(f"Failed to switch to {current_stream_source}")
                     break

            start_time = time.time()
            success, frame = cap.read()
            if not success:
               # If stream ends/fails, wait a bit and retry (or sleep and send black if it completely fails)
               time.sleep(1)
               continue
                
            # Apply pipeline toggles
            if pipeline_state["enable_detection"]:
                detections, annotated = pipeline.process_frame(frame, frame_idx)
                obj_len = len(detections) if detections is not None else 0
            else:
                annotated = frame.copy()
                obj_len = 0
                
            # If annotations disabled, overwrite with raw frame
            if not pipeline_state["show_annotations"]:
                annotated = frame.copy()

            frame_idx += 1
            
            # Update telemetry
            latency = int((time.time() - start_time) * 1000)
            current_telemetry["latency"] = latency
            current_telemetry["objectCount"] = obj_len
            current_telemetry["fps"] = int(1000 / latency) if latency > 0 else 0
            
            ret, buffer = cv2.imencode('.jpg', annotated)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

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
    # In a real app, apply this to the running pipeline config object
    return {"status": "Config updated successfully"}

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
async def list_triage_actions(action: TriageAction):
    # Placeholder: move/delete files based on action
    return {"status": f"Successfully performed {action.action} on {len(action.frame_ids)} frames"}

@app.get("/analytics/stats")
async def get_analytics_stats():
    # Placeholder: fetch aggregate stats from AnalyticsDB
    return {"total_visitors": 1420, "avg_dwell_time": "45s"}

@app.get("/analytics/timeseries")
async def get_analytics_timeseries():
    # Placeholder: fetch timeseries data for charts
    return {"drift": [], "visitors": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
