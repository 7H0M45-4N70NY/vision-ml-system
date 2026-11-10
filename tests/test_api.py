import io
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException, UploadFile
from src.vision_ml.api.main import health_check, predict_video


def run_sync(coro):
    """Run an async coroutine to completion synchronously without event loop socket overhead."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value


def test_health_check():
    response = run_sync(health_check())
    assert response == {"status": "healthy", "version": "1.0.0"}


def test_predict_video_no_pipeline():
    with patch("src.vision_ml.api.main.pipeline", None):
        mock_file = UploadFile(filename="test.mp4", file=io.BytesIO(b"fake video content"))
        with pytest.raises(HTTPException) as exc_info:
            run_sync(predict_video(file=mock_file))
        assert exc_info.value.status_code == 503
        assert "Pipeline not initialized" in exc_info.value.detail


def test_predict_video_success():
    mock_instance = MagicMock()
    mock_instance.run_offline.return_value = {
        "duration_seconds": 10,
        "total_frames": 300,
        "unique_visitors": 2,
    }
    with patch("src.vision_ml.api.main.pipeline", mock_instance):
        mock_file = UploadFile(filename="test.mp4", file=io.BytesIO(b"fake video content"))
        response = run_sync(predict_video(file=mock_file))
        assert response["unique_visitors"] == 2
        assert response["total_frames"] == 300
        mock_instance.run_offline.assert_called_once()



