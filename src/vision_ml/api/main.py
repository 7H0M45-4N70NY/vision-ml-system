from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import asyncio
from typing import Any, Dict, Optional

from ..inference.pipeline import InferencePipeline
from ..utils.config import load_config
from .schemas import InferenceRequest, InferenceResponse, HealthCheck, ConfigUpdate, TriageAction
from prometheus_fastapi_instrumentator import Instrumentator
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

# Global pipeline instance (loaded on startup)
pipeline: Optional[InferencePipeline] = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    config = load_config('config/inference/base.yaml')
    # Set to API mode (no visualization)
    config['mode']['show_live'] = False
    pipeline = InferencePipeline(config)
    logger.info("Inference Pipeline loaded")

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

@app.websocket("/ws/live-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Placeholder for actual video stream and metadata
            # In production, this would read from the pipeline's output queue
            await asyncio.sleep(1)
            await websocket.send_json({"fps": 30, "latency": 45, "objects": 2})
    except WebSocketDisconnect:
        logger.info("Client disconnected from live stream")

@app.get("/config")
async def get_config():
    if pipeline:
        return pipeline.config
    return load_config('config/inference/base.yaml')

@app.patch("/config")
async def update_config(update: ConfigUpdate):
    # In a real app, apply this to the running pipeline
    return {"status": "Config updated successfully"}

@app.get("/triage/frames")
async def get_triage_frames():
    # Placeholder: List files in data/low_confidence_frames
    return {"frames": []}

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
