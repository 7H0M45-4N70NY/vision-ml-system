# Analytics Module — `src/vision_ml/analytics/`

## What This Module Does

This is the **business logic layer**. Detection tells us WHERE people are, tracking tells us WHO they are across frames, and analytics answers the questions the business actually cares about:

- **How many unique visitors entered the store today?**
- **How long did each person stay (dwell time)?**
- **What was the peak crowd density?**
- **What's the average foot traffic per frame?**

This module is **stateful** — it accumulates data across all frames and produces a summary at the end.

## Files

### `visitor_analytics.py` — `VisitorAnalytics` Class

**Internal Data Structures:**

```python
self.track_first_seen = {}       # {track_id: first_frame_index}
self.track_last_seen = {}        # {track_id: last_frame_index}
self.track_frames = defaultdict(int)  # {track_id: total_frames_visible}
self.frame_counts = []           # [num_people_in_frame_0, num_people_in_frame_1, ...]
```

**Why these specific structures?**

- `track_first_seen` + `track_last_seen` → **Dwell time** = `(last - first + 1) / fps`
- `track_frames` → **Visibility ratio** = How many frames was this person actually detected vs. their total dwell span (useful for detecting tracking gaps)
- `frame_counts` → **Time series of crowd density** → feeds dashboards, peak detection, trend analysis

**The `update()` method — called once per frame:**

```python
def update(self, tracker_ids, frame_idx: int):
    # For each tracked person in this frame:
    #   - Record first_seen if new
    #   - Update last_seen
    #   - Increment frame count
    # Append total people count for this frame
```

This runs in O(P) where P = people in the current frame. Even with 100 people, this is microseconds. The bottleneck is always detection, never analytics.

**The `get_summary()` method — called once at the end:**

```python
{
    'unique_visitors': 47,
    'total_frames': 1800,
    'peak_visitors_per_frame': 12,
    'avg_visitors_per_frame': 4.3,
    'avg_dwell_time_seconds': 34.5,
    'dwell_times': {
        1: {'first_frame': 0, 'last_frame': 450, 'duration_seconds': 15.0},
        2: {'first_frame': 30, 'last_frame': 1200, 'duration_seconds': 39.0},
        ...
    }
}
```

## Dwell Time: The Key Metric

Dwell time = how long a person stays in the camera's view. This is THE metric for retail analytics.

```
Dwell Time Calculation:
  duration_frames = last_frame - first_frame + 1
  duration_seconds = duration_frames / fps

  Example: Person #5 first seen at frame 100, last seen at frame 400
           Video is 30fps
           Duration = (400 - 100 + 1) / 30 = 10.03 seconds
```

**Important caveat**: This is **apparent dwell time**, not true dwell time. If a person is occluded for 50 frames in the middle, we still count those frames because they were between first_seen and last_seen. The `track_frames` dict tells you how many frames they were ACTUALLY visible, which can detect this gap:

```
Visibility ratio = track_frames[id] / (last_seen - first_seen + 1)
If ratio < 0.5, the person was occluded more than half the time → noisy dwell estimate
```

## System Design: Scaling Analytics

```
Current (Project Level):
  In-memory Python dicts, single video, summary at end

FAANG Scale (Real-Time Multi-Store Analytics):
  ┌──────────────────────────────────────────────────────────┐
  │  Per-Camera Pipeline                                      │
  │    Detection → Tracking → Frame-level events              │
  │         |                                                  │
  │         v                                                  │
  │  Event Stream (Kafka / Redis Streams)                      │
  │    {"camera_id": "store1_cam3",                            │
  │     "frame": 1200,                                         │
  │     "track_ids": [1, 5, 7],                                │
  │     "timestamp": "2026-03-05T10:15:30Z"}                   │
  │         |                                                  │
  │         v                                                  │
  │  Stream Processor (Flink / Spark Streaming)                │
  │    - Real-time unique visitor counting (HyperLogLog)       │
  │    - Sliding window dwell time computation                 │
  │    - Zone-based analytics (entry, exit, hot spots)         │
  │         |                                                  │
  │         v                                                  │
  │  Time-Series DB (TimescaleDB / InfluxDB)                   │
  │    - Per-minute visitor counts                              │
  │    - Per-hour dwell time distributions                      │
  │    - Per-day unique visitor trends                          │
  │         |                                                  │
  │         v                                                  │
  │  Dashboard (Grafana / Custom React App)                    │
  │    - Live visitor counter                                   │
  │    - Heatmaps of movement                                   │
  │    - Historical trend charts                                │
  └──────────────────────────────────────────────────────────┘

  Key data structure upgrade at scale:
  - HyperLogLog for approximate unique counting (O(1) memory)
  - Sliding windows instead of full history (bounded memory)
  - Zone polygons for area-specific analytics
```

## What I Learned Building This

1. **`defaultdict(int)` is perfect for counters** — No need to check `if key in dict` before incrementing. Cleaner code, fewer bugs.

2. **FPS must be set correctly** — If `fps=30` but the actual video is 60fps, all dwell times will be 2x too long. The pipeline sets `self.analytics.fps = fps` from the video capture properties. Always propagate the actual FPS.

3. **`peak_visitors_per_frame` was missing initially** — I added it during self-review because it's a critical metric for capacity planning ("what's the max crowd density we saw?"). Simple `max(self.frame_counts)` but very useful.

4. **`reset()` is critical** — Without reset between videos, visitor counts accumulate. Person #1 from Video A and Person #1 from Video B would be counted as the same visitor (same track_id, different identity). Always reset between independent sessions.

5. **Dwell time is an approximation** — ByteTrack may lose and re-acquire a person with a NEW track ID. This splits one true visit into two shorter dwell times. Cross-camera Re-ID would help, but for a single fixed camera, ByteTrack's `track_buffer` (keep lost tracks for N frames) mitigates this well.
