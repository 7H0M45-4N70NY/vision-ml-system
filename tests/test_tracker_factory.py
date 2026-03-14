"""Tests for TrackerFactory."""
import pytest
from unittest.mock import patch, MagicMock
from src.vision_ml.tracking.tracker_factory import TrackerFactory


class TestTrackerFactory:
    def test_list_available(self):
        available = TrackerFactory.list_available()
        assert 'bytetrack' in available
        assert 'botsort' in available
        assert 'ocsort' in available

    def test_create_bytetrack(self, sample_config):
        tracker = TrackerFactory.create('bytetrack', sample_config)
        assert tracker is not None

    def test_from_config(self, sample_config):
        tracker = TrackerFactory.from_config(sample_config)
        assert tracker is not None

    def test_new_instance_each_time(self, sample_config):
        t1 = TrackerFactory.create('bytetrack', sample_config)
        t2 = TrackerFactory.create('bytetrack', sample_config)
        assert t1 is not t2

    def test_invalid_type_raises(self, sample_config):
        with pytest.raises(ValueError, match="Unknown tracker"):
            TrackerFactory.create('nonexistent', sample_config)

    @pytest.mark.parametrize("tracker_type", ['bytetrack', 'botsort', 'ocsort'])
    def test_create_all_types(self, tracker_type, sample_config):
        tracker = TrackerFactory.create(tracker_type, sample_config)
        assert tracker is not None
