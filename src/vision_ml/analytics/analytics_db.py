"""SQLite analytics database for persistent data storage.

Stores inference runs, visitor analytics, and training events.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class AnalyticsDB:
    """SQLite database for Vision ML analytics."""
    
    def __init__(self, db_path: str = "data/analytics.db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Inference runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source_type TEXT,
                    duration_seconds REAL,
                    total_frames INTEGER,
                    unique_visitors INTEGER,
                    avg_dwell_time_seconds REAL,
                    use_dual_detector BOOLEAN,
                    secondary_ratio REAL,
                    frames_saved INTEGER,
                    avg_confidence REAL,
                    drift_score REAL,
                    status TEXT
                )
            """)
            # Add columns if upgrading from older schema
            for col, col_type in [('avg_confidence', 'REAL'), ('drift_score', 'REAL')]:
                try:
                    cursor.execute(f"ALTER TABLE inference_runs ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            
            # Visitor analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS visitor_analytics (
                    visitor_id TEXT,
                    run_id TEXT,
                    tracker_id INTEGER,
                    first_frame INTEGER,
                    last_frame INTEGER,
                    duration_frames INTEGER,
                    duration_seconds REAL,
                    PRIMARY KEY (visitor_id, run_id),
                    FOREIGN KEY (run_id) REFERENCES inference_runs(run_id)
                )
            """)
            
            # Training events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    trigger_type TEXT,
                    dataset_size INTEGER,
                    drift_score REAL,
                    status TEXT,
                    model_version TEXT
                )
            """)
            
            # Auto-labeling events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS labeling_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    frames_processed INTEGER,
                    labels_created INTEGER,
                    provider TEXT,
                    status TEXT
                )
            """)
            
            conn.commit()
    
    def save_inference_run(self, run_data: Dict) -> str:
        """Save inference run to database.
        
        Args:
            run_data: Dictionary with inference metrics
            
        Returns:
            run_id: Unique identifier for this run
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inference_runs (
                    run_id, source_type, duration_seconds, total_frames,
                    unique_visitors, avg_dwell_time_seconds, use_dual_detector,
                    secondary_ratio, frames_saved, avg_confidence, drift_score, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                run_data.get('source_type', 'unknown'),
                run_data.get('duration_seconds', 0),
                run_data.get('total_frames', 0),
                run_data.get('unique_visitors', 0),
                run_data.get('avg_dwell_time_seconds', 0),
                run_data.get('use_dual_detector', False),
                run_data.get('secondary_ratio', 0),
                run_data.get('frames_saved', 0),
                run_data.get('avg_confidence', 0),
                run_data.get('drift_score', 0),
                'completed'
            ))
            conn.commit()
        
        return run_id
    
    def save_visitor_analytics(self, run_id: str, dwell_times: Dict):
        """Save visitor dwell times for a run.
        
        Args:
            run_id: Inference run ID
            dwell_times: Dictionary of tracker_id -> dwell_info
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for tracker_id, dwell_info in dwell_times.items():
                visitor_id = f"{run_id}_person_{tracker_id}"
                cursor.execute("""
                    INSERT INTO visitor_analytics (
                        visitor_id, run_id, tracker_id, first_frame,
                        last_frame, duration_frames, duration_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    visitor_id,
                    run_id,
                    tracker_id,
                    dwell_info['first_frame'],
                    dwell_info['last_frame'],
                    dwell_info['duration_frames'],
                    dwell_info['duration_seconds']
                ))
            
            conn.commit()
    
    def save_labeling_event(self, event_data: Dict) -> str:
        """Save auto-labeling event.
        
        Args:
            event_data: Dictionary with labeling metrics
            
        Returns:
            event_id: Unique identifier for this event
        """
        event_id = f"label_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO labeling_events (
                    event_id, frames_processed, labels_created, provider, status
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                event_id,
                event_data.get('frames_processed', 0),
                event_data.get('labels_created', 0),
                event_data.get('provider', 'local'),
                'completed'
            ))
            conn.commit()
        
        return event_id
    
    def save_training_event(self, event_data: Dict) -> str:
        """Save training event.
        
        Args:
            event_data: Dictionary with training metrics
            
        Returns:
            event_id: Unique identifier for this event
        """
        event_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_events (
                    event_id, trigger_type, dataset_size, drift_score, status, model_version
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                event_data.get('trigger_type', 'manual'),
                event_data.get('dataset_size', 0),
                event_data.get('drift_score', 0),
                'pending',
                event_data.get('model_version', 'v1')
            ))
            conn.commit()
        
        return event_id
    
    def get_inference_runs(self, limit: int = 50) -> List[Dict]:
        """Get recent inference runs.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List of inference run records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM inference_runs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_visitor_analytics(self, run_id: str) -> List[Dict]:
        """Get visitor analytics for a specific run.
        
        Args:
            run_id: Inference run ID
            
        Returns:
            List of visitor records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM visitor_analytics
                WHERE run_id = ?
                ORDER BY duration_seconds DESC
            """, (run_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_labeling_events(self, limit: int = 20) -> List[Dict]:
        """Get recent labeling events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of labeling event records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM labeling_events
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_training_events(self, limit: int = 20) -> List[Dict]:
        """Get recent training events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of training event records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM training_events
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_analytics_summary(self) -> Dict:
        """Get overall analytics summary.
        
        Returns:
            Dictionary with summary statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total runs
            cursor.execute("SELECT COUNT(*) as count FROM inference_runs")
            total_runs = cursor.fetchone()[0]
            
            # Total visitors
            cursor.execute("SELECT COUNT(DISTINCT visitor_id) as count FROM visitor_analytics")
            total_visitors = cursor.fetchone()[0]
            
            # Avg dwell time
            cursor.execute("SELECT AVG(duration_seconds) as avg FROM visitor_analytics")
            avg_dwell = cursor.fetchone()[0] or 0
            
            # Total frames processed
            cursor.execute("SELECT SUM(total_frames) as total FROM inference_runs")
            total_frames = cursor.fetchone()[0] or 0
            
            # Labeling events
            cursor.execute("SELECT COUNT(*) as count FROM labeling_events")
            total_labeling_events = cursor.fetchone()[0]
            
            # Training events
            cursor.execute("SELECT COUNT(*) as count FROM training_events")
            total_training_events = cursor.fetchone()[0]
            
            return {
                'total_runs': total_runs,
                'total_visitors': total_visitors,
                'avg_dwell_time_seconds': round(avg_dwell, 2),
                'total_frames': total_frames,
                'total_labeling_events': total_labeling_events,
                'total_training_events': total_training_events,
            }
