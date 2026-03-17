import sqlite3

from src.vision_ml.analytics.analytics_db import AnalyticsDB


def test_training_event_lifecycle_transitions(tmp_path):
    db_path = str(tmp_path / "training_events.db")
    db = AnalyticsDB(db_path)

    event_id = db.save_training_event(
        {
            "trigger_type": "manual",
            "dataset_size": 10,
            "drift_score": 0.0,
            "model_version": "v1",
            "status": "pending",
        }
    )

    assert db.mark_training_event_running(event_id, dataset_size=25)
    assert db.mark_training_event_completed(event_id)

    events = db.get_training_events(limit=1)
    assert len(events) == 1
    assert events[0]["event_id"] == event_id
    assert events[0]["status"] == "completed"
    assert events[0]["dataset_size"] == 25


def test_update_training_event_status_missing_event(tmp_path):
    db_path = str(tmp_path / "missing_event.db")
    db = AnalyticsDB(db_path)

    assert not db.update_training_event_status("missing", "running")
