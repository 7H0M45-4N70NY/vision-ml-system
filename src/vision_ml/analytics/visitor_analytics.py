from collections import defaultdict


class VisitorAnalytics:
    def __init__(self, config: dict):
        analytics_cfg = config.get('analytics', {})
        self.enabled = analytics_cfg.get('enabled', True)
        self.compute_dwell = analytics_cfg.get('compute_dwell_time', True)
        self.fps = analytics_cfg.get('dwell_time_fps', 30)

        self.track_first_seen = {}
        self.track_last_seen = {}
        self.track_frames = defaultdict(int)
        self.frame_counts = []

    def update(self, tracker_ids, frame_idx: int):
        if not self.enabled or tracker_ids is None:
            return

        current_ids = set()
        for tid in tracker_ids:
            tid = int(tid)
            current_ids.add(tid)
            self.track_frames[tid] += 1
            if tid not in self.track_first_seen:
                self.track_first_seen[tid] = frame_idx
            self.track_last_seen[tid] = frame_idx

        self.frame_counts.append(len(current_ids))

    @property
    def unique_visitor_count(self) -> int:
        return len(self.track_first_seen)

    @property
    def current_frame_count(self) -> int:
        return self.frame_counts[-1] if self.frame_counts else 0

    def get_dwell_times(self) -> dict:
        if not self.compute_dwell:
            return {}
        dwell = {}
        for tid in self.track_first_seen:
            first = self.track_first_seen[tid]
            last = self.track_last_seen[tid]
            duration_frames = last - first + 1
            duration_seconds = duration_frames / max(self.fps, 1)
            dwell[tid] = {
                'first_frame': first,
                'last_frame': last,
                'duration_frames': duration_frames,
                'duration_seconds': round(duration_seconds, 2),
            }
        return dwell

    def get_summary(self) -> dict:
        dwell_times = self.get_dwell_times()
        avg_dwell = 0.0
        if dwell_times:
            avg_dwell = sum(d['duration_seconds'] for d in dwell_times.values()) / len(dwell_times)

        return {
            'unique_visitors': self.unique_visitor_count,
            'total_frames': len(self.frame_counts),
            'peak_visitors_per_frame': max(self.frame_counts) if self.frame_counts else 0,
            'avg_visitors_per_frame': round(
                sum(self.frame_counts) / max(len(self.frame_counts), 1), 2
            ),
            'avg_dwell_time_seconds': round(avg_dwell, 2),
            'dwell_times': dwell_times,
        }

    def reset(self):
        self.track_first_seen.clear()
        self.track_last_seen.clear()
        self.track_frames.clear()
        self.frame_counts.clear()
