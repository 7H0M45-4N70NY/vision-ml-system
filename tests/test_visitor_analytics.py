"""Tests for VisitorAnalytics — pure logic, no mocks needed."""
from src.vision_ml.analytics.visitor_analytics import VisitorAnalytics


class TestVisitorAnalytics:
    def _make(self, **overrides):
        config = {'analytics': {'enabled': True, 'compute_dwell_time': True, 'dwell_time_fps': 30}}
        config['analytics'].update(overrides)
        return VisitorAnalytics(config)

    def test_empty_state(self):
        va = self._make()
        assert va.unique_visitor_count == 0
        assert va.current_frame_count == 0

    def test_update_counts_visitors(self):
        va = self._make()
        va.update([1, 2], frame_idx=0)
        va.update([2, 3], frame_idx=1)
        assert va.unique_visitor_count == 3

    def test_current_frame_count(self):
        va = self._make()
        va.update([1, 2, 3], frame_idx=0)
        assert va.current_frame_count == 3
        va.update([1], frame_idx=1)
        assert va.current_frame_count == 1

    def test_dwell_time_calculation(self):
        va = self._make(dwell_time_fps=10)
        va.update([1], frame_idx=0)
        va.update([1], frame_idx=9)
        dwell = va.get_dwell_times()
        assert dwell[1]['duration_frames'] == 10
        assert dwell[1]['duration_seconds'] == 1.0

    def test_get_summary(self):
        va = self._make()
        va.update([1, 2], frame_idx=0)
        va.update([2, 3], frame_idx=1)
        summary = va.get_summary()
        assert summary['unique_visitors'] == 3
        assert summary['total_frames'] == 2
        assert 'dwell_times' in summary

    def test_reset(self):
        va = self._make()
        va.update([1, 2], frame_idx=0)
        va.reset()
        assert va.unique_visitor_count == 0
        assert va.current_frame_count == 0

    def test_disabled(self):
        va = self._make(enabled=False)
        va.update([1, 2], frame_idx=0)
        assert va.unique_visitor_count == 0
