"""ByteTrack wrapper — delegates to GenericSVTracker for compatibility."""

from .base import BaseTracker


class ByteTrackTracker(BaseTracker):
    """ByteTrack tracker wrapper.
    
    Delegates to GenericSVTracker to handle parameter passing correctly
    across different Supervision versions.
    """

    def __init__(self, config: dict):
        # Import here to avoid circular dependency
        from .tracker_factory import GenericSVTracker
        self._tracker = GenericSVTracker('bytetrack', config)

    def update(self, detections):
        return self._tracker.update(detections)

    def reset(self) -> None:
        self._tracker.reset()
