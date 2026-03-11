from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
from typing import Optional

from src.vision_ml.inference.pipeline import InferencePipeline
from src.vision_ml.utils.config import load_config
from .schemas import InferenceRequest, InferenceResponse, HealthCheck
from prometheus_fastapi_instrumentator import Instrumentator

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
    print("✅ Inference Pipeline loaded")

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
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run inference (blocking for simplicity in MVP, should be async task queue in prod)
        if pipeline is None:
             raise HTTPException(status_code=503, detail="Pipeline not initialized")
             
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
