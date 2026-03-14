"""Model drift detection via inference confidence monitoring.

This module implements **model drift** detection — detecting when the deployed
model's prediction quality degrades over time.  It works by monitoring the
rolling average of prediction confidences and flagging a drift event when the
average drops below a configurable threshold.

Architecture
------------
- `DriftDetector` is the concrete implementation for model drift.
- It exposes a simple record → check → get_metrics cycle that the
  `InferencePipeline` calls on every frame.
- The detector is stateless across sessions — each inference run starts fresh.
  Cross-run drift is tracked via `avg_confidence` / `drift_score` columns
  persisted in `AnalyticsDB` and visualised on the Streamlit drift pages.

Future: Data Drift
------------------
Data drift (input distribution shift) is a separate concern that would compare
feature distributions of incoming data against a training-time reference.
To add it:
  1. Create a `DataDriftDetector` class alongside this one.
  2. Have it compute image-level features (brightness, contrast, resolution)
     and compare to a stored baseline via KS-test or PSI.
  3. Wire it into `InferencePipeline.process_frame()` the same way.
  4. Add a `drift.data_drift` config section.
Both detectors can coexist — model drift and data drift are orthogonal signals
that together give a complete degradation picture.
"""

from collections import deque
from typing import Dict

from ..logging import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Detects **model drift** by monitoring inference confidence scores.

    Operates on a sliding window of per-detection confidence values.
    When the rolling average drops below `confidence_threshold`, drift is
    flagged.  Callers should persist the metrics returned by `get_metrics()`
    into the analytics DB so the Streamlit dashboards can display cross-run
    drift trends.

    Config keys (under ``drift:`` in YAML):
        enabled:              bool   – master switch
        method:               str    – 'confidence_drop' (only model drift impl for now)
        confidence_threshold: float  – avg confidence below this = drift (default 0.3)
        window_size:          int    – max recent confidences to keep (default 500)
        check_interval:       int    – run check() logic every N record() calls (default 100)
    """

    def __init__(self, config: dict):
        drift_cfg = config.get('drift', {})
        self.enabled = drift_cfg.get('enabled', False)
        self.method = drift_cfg.get('method', 'confidence_drop')
        self.threshold = drift_cfg.get('confidence_threshold', 0.3)
        self.window_size = drift_cfg.get('window_size', 500)
        self.check_interval = drift_cfg.get('check_interval', 100)

        # Sliding window of per-detection confidence scores
        self.confidence_buffer = deque(maxlen=self.window_size)
        self.inference_count = 0
        self.drift_detected = False

        # Running stats for metrics
        self._total_detections = 0
        self._low_conf_count = 0  # detections below threshold

    # -- Public API ----------------------------------------------------------

    def record(self, confidences: list):
        """Append per-detection confidence scores from one frame."""
        if not self.enabled:
            return
        self.confidence_buffer.extend(confidences)
        self.inference_count += 1
        self._total_detections += len(confidences)
        self._low_conf_count += sum(1 for c in confidences if c < self.threshold)

    def check(self) -> bool:
        """Evaluate drift condition.  Returns True if drift is detected."""
        if not self.enabled:
            return False

        if self.inference_count % self.check_interval != 0:
            return self.drift_detected  # return last known state

        if len(self.confidence_buffer) < self.check_interval:
            return False

        if self.method == 'confidence_drop':
            avg_conf = sum(self.confidence_buffer) / len(self.confidence_buffer)
            self.drift_detected = avg_conf < self.threshold
            if self.drift_detected:
                logger.warning(
                    f"MODEL DRIFT DETECTED: "
                    f"avg confidence {avg_conf:.3f} < threshold {self.threshold}"
                )
            return self.drift_detected

        # TODO: Add 'data_drift' method here when DataDriftDetector is implemented.
        #       Data drift would compare input feature distributions against a
        #       training-time reference (e.g. KS-test on brightness/contrast).
        return False

    def get_metrics(self) -> Dict:
        """Return a snapshot of drift metrics for persistence / display.

        The pipeline and Streamlit pages use these values to populate the
        ``avg_confidence`` and ``drift_score`` columns in AnalyticsDB.
        """
        buf = self.confidence_buffer
        avg_conf = sum(buf) / len(buf) if buf else 0.0
        drift_score = 1.0 - avg_conf if avg_conf > 0 else 0.0
        low_ratio = (self._low_conf_count / self._total_detections
                     if self._total_detections > 0 else 0.0)

        return {
            'avg_confidence': round(avg_conf, 4),
            'drift_score': round(drift_score, 4),
            'drift_detected': self.drift_detected,
            'buffer_size': len(buf),
            'total_detections': self._total_detections,
            'low_confidence_ratio': round(low_ratio, 4),
            'method': self.method,
            'threshold': self.threshold,
        }

    def reset(self):
        """Clear all state.  Called between inference runs."""
        self.confidence_buffer.clear()
        self.inference_count = 0
        self.drift_detected = False
        self._total_detections = 0
        self._low_conf_count = 0
