"""
Tests for the analytics database module.
"""
import os
import pytest
import sqlite3
from src.vision_ml.analytics.analytics_db import AnalyticsDB

@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_analytics.db")

def test_db_initialization(temp_db_path):
    db = AnalyticsDB(temp_db_path)
    assert os.path.exists(temp_db_path)
    
    with sqlite3.connect(temp_db_path) as conn:
        cursor = conn.cursor()
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "inference_runs" in tables
        assert "visitor_analytics" in tables

def test_save_and_retrieve_run(temp_db_path):
    db = AnalyticsDB(temp_db_path)
    run_data = {
        'source_type': 'test',
        'duration_seconds': 10.0,
        'total_frames': 100,
        'unique_visitors': 5
    }
    
    run_id = db.save_inference_run(run_data)
    assert run_id.startswith("run_")
    
    runs = db.get_inference_runs(limit=1)
    assert len(runs) == 1
    assert runs[0]['run_id'] == run_id
    assert runs[0]['total_frames'] == 100
