import supervision as sv

from .base import BaseTracker
from .bytetrack import ByteTrackTracker
from ..logging import get_logger

logger = get_logger(__name__)


def _create_sv_tracker(tracker_type: str, config: dict):
    """Create supervision tracker with version compatibility.
    
    Handles different Supervision versions which may have different parameter names.
    """
    tracker_cfg = config.get('tracking', {})
    
    # Try with full parameters first
    params = {
        'track_activation_threshold': tracker_cfg.get('track_thresh', 0.25),
        'lost_track_buffer': tracker_cfg.get('track_buffer', 30),
        'minimum_matching_threshold': tracker_cfg.get('match_thresh', 0.8),
        'frame_rate': tracker_cfg.get('frame_rate', 30),
    }
    
    try:
        if tracker_type == 'botsort':
            return sv.BoTSORT(**params)
        elif tracker_type == 'ocsort':
            return sv.OCSORT(**params)
        else:  # bytetrack
            return sv.ByteTrack(**params)
    except TypeError as e:
        # If parameter names don't match, try with minimal params
        logger.warning(f"Parameter error: {e}")
        logger.warning(f"Retrying {tracker_type} with minimal parameters...")
        try:
            if tracker_type == 'botsort':
                return sv.BoTSORT()
            elif tracker_type == 'ocsort':
                return sv.OCSORT()
            else:
                return sv.ByteTrack()
        except Exception as e2:
            logger.error(f"Failed to create {tracker_type}: {e2}")
            logger.warning("Falling back to ByteTrack with defaults")
            return sv.ByteTrack()
    except AttributeError:
        logger.warning(f"{tracker_type} not available, falling back to ByteTrack")
        return sv.ByteTrack()


class GenericSVTracker(BaseTracker):
    """Generic wrapper for any Supervision tracker (BoT-SORT, OC-SORT, etc).
    
    Trackers are stateful — never cached/singleton.
    Each pipeline gets its own tracker instance.
    """

    def __init__(self, tracker_type: str, config: dict):
        self.tracker = _create_sv_tracker(tracker_type, config)

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)

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
