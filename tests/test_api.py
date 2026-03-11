from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pytest
from src.vision_ml.api.main import app

client = TestClient(app)

@pytest.fixture
def mock_pipeline():
    with patch("src.vision_ml.api.main.pipeline") as mock:
        yield mock

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}

def test_predict_video_no_pipeline(mock_pipeline):
    # Simulate pipeline not initialized
    with patch("src.vision_ml.api.main.pipeline", None):
        # Create a dummy video file
        files = {'file': ('test.mp4', b'fake video content', 'video/mp4')}
        response = client.post("/predict/video", files=files)
        assert response.status_code == 503
        assert "Pipeline not initialized" in response.json()['detail']

def test_predict_video_success():
    # Mock the pipeline instance and its run_offline method
    mock_instance = MagicMock()
    mock_instance.run_offline.return_value = {
        "duration_seconds": 10,
        "total_frames": 300,
        "unique_visitors": 2
    }
    
    with patch("src.vision_ml.api.main.pipeline", mock_instance):
        files = {'file': ('test.mp4', b'fake video content', 'video/mp4')}
        response = client.post("/predict/video", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data['unique_visitors'] == 2
        assert data['total_frames'] == 300
