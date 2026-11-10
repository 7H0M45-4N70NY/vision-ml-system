import supervision as sv

try:
    import trackers as roboflow_trackers
    HAS_ROBOFLOW_TRACKERS = True
except ImportError:
    HAS_ROBOFLOW_TRACKERS = False

from .base import BaseTracker
from .bytetrack import ByteTrackTracker
from ..logging import get_logger

logger = get_logger(__name__)


def _create_sv_tracker(tracker_type: str, config: dict):
    """Create supervision or roboflow tracker with version compatibility.
    
    Prefers standalone `trackers` package if available, falling back
    to `supervision` for backwards compatibility.
    """
    tracker_cfg = config.get('tracking', {})
    
    if HAS_ROBOFLOW_TRACKERS:
        try:
            lost_buffer = tracker_cfg.get('track_buffer', 30)
            fps = float(tracker_cfg.get('frame_rate', 30))
            thresh = float(tracker_cfg.get('track_thresh', 0.25))

            if tracker_type == 'botsort':
                return roboflow_trackers.BoTSORTTracker(
                    lost_track_buffer=lost_buffer,
                    frame_rate=fps,
                    track_activation_threshold=thresh,
                )
            elif tracker_type == 'ocsort':
                return roboflow_trackers.OCSORTTracker(
                    lost_track_buffer=lost_buffer,
                    frame_rate=fps,
                )
            else:  # bytetrack
                return roboflow_trackers.ByteTrackTracker(
                    lost_track_buffer=lost_buffer,
                    frame_rate=fps,
                    track_activation_threshold=thresh,
                )
        except Exception as e:
            logger.warning(f"Could not initialize from `trackers`: {e}. Trying `supervision`...")

    # Fallback to supervision
    params = {
        'track_activation_threshold': tracker_cfg.get('track_thresh', 0.25),
        'lost_track_buffer': tracker_cfg.get('track_buffer', 30),
        'minimum_matching_threshold': tracker_cfg.get('match_thresh', 0.8),
        'frame_rate': tracker_cfg.get('frame_rate', 30),
    }
    
    try:
        if tracker_type == 'botsort' and hasattr(sv, 'BoTSORT'):
            return sv.BoTSORT(**params)
        elif tracker_type == 'ocsort' and hasattr(sv, 'OCSORT'):
            return sv.OCSORT(**params)
        elif hasattr(sv, 'ByteTrack'):
            return sv.ByteTrack(**params)
    except TypeError as e:
        logger.warning(f"Parameter error: {e}")
        logger.warning(f"Retrying {tracker_type} with minimal parameters...")
        if tracker_type == 'botsort' and hasattr(sv, 'BoTSORT'):
            return sv.BoTSORT()
        elif tracker_type == 'ocsort' and hasattr(sv, 'OCSORT'):
            return sv.OCSORT()
        elif hasattr(sv, 'ByteTrack'):
            return sv.ByteTrack()

    if hasattr(sv, 'ByteTrack'):
        return sv.ByteTrack()

    raise RuntimeError(
        f"Unable to instantiate tracker '{tracker_type}'. "
        "Please install 'trackers>=2.2.0'."
    )


class GenericSVTracker(BaseTracker):
    """Generic wrapper for any Supervision tracker (BoT-SORT, OC-SORT, etc).
    
    Trackers are stateful — never cached/singleton.
    Each pipeline gets its own tracker instance.
    """

    def __init__(self, tracker_type: str, config: dict):
        self.tracker = _create_sv_tracker(tracker_type, config)

    def update(self, detections: sv.Detections) -> sv.Detections:
        if hasattr(self.tracker, 'update_with_detections'):
            return self.tracker.update_with_detections(detections)
        return self.tracker.update(detections)

    def reset(self) -> None:
        self.tracker.reset()


_TRACKER_CLASSES = {
    'bytetrack': ByteTrackTracker,
    'botsort': lambda cfg: GenericSVTracker('botsort', cfg),
    'ocsort': lambda cfg: GenericSVTracker('ocsort', cfg),
}


class TrackerFactory:
    """Factory for creating tracker instances (never cached).
    
    Trackers are stateful and must NOT be singletons.
    Each inference pipeline gets its own tracker instance.
    
    Supports: bytetrack, botsort, ocsort
    
    Usage:
        tracker = TrackerFactory.create('bytetrack', config)
        detections = tracker.update(detections)
        
        # Each call creates a NEW instance (not cached)
        tracker2 = TrackerFactory.create('bytetrack', config)
        assert tracker is not tracker2  # Different objects
    """

    @staticmethod
    def create(tracker_type: str, config: dict) -> BaseTracker:
        """Create a NEW tracker instance (never cached).
        
        Args:
            tracker_type: Type of tracker ('bytetrack', 'botsort', 'ocsort')
            config: Configuration dict
            
        Returns:
            NEW BaseTracker instance
            
        Raises:
            ValueError: If tracker_type is not supported
        """
        if tracker_type not in _TRACKER_CLASSES:
            raise ValueError(
                f"Unknown tracker: {tracker_type}. "
                f"Available: {list(_TRACKER_CLASSES.keys())}"
            )

        creator = _TRACKER_CLASSES[tracker_type]
        return creator(config)

    @staticmethod
    def from_config(config: dict) -> BaseTracker:
        """Create tracker from config (reads tracking.tracker_type)."""
        tracker_type = config.get('tracking', {}).get('tracker_type', 'bytetrack')
        return TrackerFactory.create(tracker_type, config)

    @staticmethod
    def list_available() -> list:
        """List available tracker types."""
        return list(_TRACKER_CLASSES.keys())
