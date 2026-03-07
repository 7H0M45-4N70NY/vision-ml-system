# Tracking Module — `src/vision_ml/tracking/`

## What This Module Does

This module takes **per-frame detections** (which have NO identity) and assigns **persistent track IDs** across frames. Without tracking, you know "there are 3 people in frame 1" and "there are 3 people in frame 2", but you don't know if they're the SAME 3 people. Tracking solves this.

This is the bridge between "detection" and "analytics" — you can't count unique visitors or compute dwell time without persistent IDs.

## Files

### `base.py` — Abstract Tracker Interface

```python
class BaseTracker(ABC):
    def update(self, detections: sv.Detections) -> sv.Detections: ...
    def reset(self) -> None: ...
```

**Why this interface?**

- `update()` takes detections WITHOUT tracker IDs and returns detections WITH tracker IDs (the `tracker_id` field gets populated).
- `reset()` clears all internal state. Critical when switching between videos in offline mode — without reset, the tracker thinks Video 2's people are continuations of Video 1's tracks.
- Input and output are both `sv.Detections`, so the tracker is a **transparent pass-through** in the pipeline. You can disable tracking and everything still works (just without IDs).

### `bytetrack.py` — ByteTrack Implementation via Supervision

```python
class ByteTrackTracker(BaseTracker):
    def __init__(self, config: dict):
        self.tracker = sv.ByteTrack(...)

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)
```

This is intentionally thin — the Supervision library's `sv.ByteTrack` does all the heavy lifting. Our wrapper exists for:
1. **Config-driven initialization** — thresholds come from YAML, not hardcoded
2. **Interface compliance** — implements `BaseTracker` so we can swap trackers
3. **Encapsulation** — rest of the system doesn't know about `sv.ByteTrack` internals

## How ByteTrack Works (The Algorithm)

ByteTrack is a multi-object tracker that uses a two-stage association strategy. Here's the mental model:

```
Frame N detections arrive
    |
    v
Split into HIGH confidence (>0.25) and LOW confidence detections
    |
    v
Stage 1: Match HIGH confidence detections to existing tracks
         using IoU (Intersection over Union) + Kalman Filter prediction
    |
    v
Stage 2: Match LOW confidence detections to UNMATCHED tracks
         (catches partially occluded people that other trackers miss)
    |
    v
Unmatched detections → Create NEW tracks
Unmatched tracks → Mark as LOST (keep for track_buffer frames)
Lost too long → DELETE track
    |
    v
Output: detections with tracker_id assigned
```

**Why ByteTrack over other trackers?**

| Tracker | Approach | Pros | Cons |
|---|---|---|---|
| SORT | Kalman + Hungarian matching | Fast, simple | Loses tracks on occlusion |
| DeepSORT | SORT + Re-ID features | Better re-identification | Needs separate Re-ID model (slow) |
| **ByteTrack** | **Two-stage IoU matching** | **Fast, handles occlusion, no Re-ID needed** | **Less robust for long-term re-identification** |
| BoT-SORT | ByteTrack + camera motion | Handles camera shake | More complex |

For visitor analytics in a fixed camera setup, ByteTrack is the sweet spot — fast enough for real-time, robust enough for occlusion, and doesn't need an expensive Re-ID network.

## Config Parameters Explained

```yaml
tracking:
  track_thresh: 0.25    # Detections below this → "low confidence" bucket
  track_buffer: 30      # Keep lost tracks alive for 30 frames (1 sec at 30fps)
  match_thresh: 0.8     # IoU threshold for matching detections to tracks
  frame_rate: 30        # Used to scale track_buffer to real time
```

- **`track_thresh`**: Lower = more aggressive tracking (catches faint detections), but more false positives. 0.25 is the ByteTrack paper default.
- **`track_buffer`**: If a person is occluded for < 1 second, their track survives. Set higher for crowds where occlusion is frequent.
- **`match_thresh`**: How much overlap is needed to say "this detection IS that track". 0.8 is strict (good for non-overlapping people).

## System Design: Scaling Tracking

```
Current (Project Level):
  Single-threaded, all tracks in memory, one video stream

FAANG Scale (Multi-Camera Retail Store):
  ┌─────────────────────────────────────────────────────┐
  │ Camera 1 ──→ Detector ──→ Tracker Instance 1        │
  │ Camera 2 ──→ Detector ──→ Tracker Instance 2        │
  │ Camera 3 ──→ Detector ──→ Tracker Instance 3        │
  │       |            |             |                   │
  │       └────────────┴─────────────┘                   │
  │                    |                                  │
  │        Cross-Camera Re-ID Service                    │
  │        (merges track IDs across cameras)              │
  │                    |                                  │
  │            Global Track Store                         │
  │            (Redis / TimescaleDB)                      │
  └─────────────────────────────────────────────────────┘

  Key insight: Each camera gets its OWN tracker instance (ByteTrack is
  stateful per-stream). Cross-camera identity is a SEPARATE problem
  solved by a Re-ID model or spatial overlap zones.

  Scaling levers:
  1. One tracker per camera stream (embarrassingly parallel)
  2. Tracker state is tiny (~KB per track) so it fits in memory
  3. Re-ID across cameras is the expensive part (run async, not real-time)
```

## What I Learned Building This

1. **Supervision's `update_with_detections()` is the key API** — It takes `sv.Detections` in, returns `sv.Detections` out with `tracker_id` populated. This is much cleaner than manually wrangling numpy arrays.

2. **Tracker is STATEFUL** — Unlike the detector (stateless per-frame), the tracker accumulates state across frames (Kalman filter states, lost track buffer). This means:
   - You MUST call `reset()` between different videos
   - You CANNOT parallelize tracking of a single stream (frames must be sequential)
   - Memory grows with number of active + lost tracks

3. **`track_buffer` is time-dependent** — A buffer of 30 frames means 1 second at 30fps but 0.5 seconds at 60fps. The `frame_rate` parameter normalizes this to real time. Always set it to match your actual video FPS.

4. **The wrapper pattern is worth it** — Even though `ByteTrackTracker` is only 20 lines, it decouples us from the Supervision library. If Supervision changes their API or we switch to the `trackers` package, only this file changes.
