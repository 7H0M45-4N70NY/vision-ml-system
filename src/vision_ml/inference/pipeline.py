# --- Vision ML System: Inference Pipeline ---
"""Orchestrator for the core video analytics flow.

Coordinates frame extraction, object detection, multi-object tracking,
visitor analytics, active learning collection, and drift monitoring.
"""

import os
import json
import cv2
import numpy as np
import supervision as sv
from typing import Tuple, Dict, Any, Optional

from ..detection.detector_factory import DetectorFactory
from ..detection.dual_detector import DualDetector
from ..tracking.tracker_factory import TrackerFactory
from ..annotation.annotator import FrameAnnotator
from ..analytics.visitor_analytics import VisitorAnalytics
from ..labeling.auto_labeler import AutoLabeler
from ..training.drift_detector import DriftDetector
from ..logging import get_logger
from ..utils.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)


class InferencePipeline:
    """Main entry point for running inference on video files or live streams.

    This class encapsulates the entire vision lifecycle:
    1. Detection (YOLO/RF-DETR)
    2. Tracking (ByteTrack)
    3. Analytics (Visitor counts, dwell time)
    4. Feedback Loops (Auto-labeling, drift detection)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the pipeline with a unified configuration dictionary.

        Args:
            config: Configuration dictionary containing sections for detection,
                tracking, analytics, and labeling.
        """
        self.config = config
        use_dual = config.get('detection', {}).get('use_dual_detector', False)
        detector_type = config.get('detection', {}).get('detector_type', 'yolo11n')

        logger.info(f"Initializing InferencePipeline (mode={use_dual}, detector={detector_type})")

        # Detector: single (cached singleton) or dual (with frame collection)
        if use_dual:
            logger.info(f"Using DualDetector (mode={use_dual})")
            self.detector = DualDetector(config)
        else:
            logger.info(f"Using primary detector: {detector_type}")
            self.detector = DetectorFactory.from_config(config)

        # Tracker: always a new stateful instance
        if config.get('tracking', {}).get('enabled', True):
            self.tracker = TrackerFactory.from_config(config)
            logger.info("Tracker enabled")
        else:
            self.tracker = None
            logger.info("Tracker disabled")

        self.annotator = FrameAnnotator(config)
        self.analytics = VisitorAnalytics(config)
        self.auto_labeler = AutoLabeler(config)
        self.drift_detector = DriftDetector(config)

        self.mode = config.get('mode', {}).get('type', 'offline')
        self.output_dir = config.get('mode', {}).get('output_dir', 'runs/inference')
        os.makedirs(self.output_dir, exist_ok=True)

        # One circuit breaker per pipeline stage — failure policy centralised in CircuitBreaker
        self._cb_detect   = CircuitBreaker("detect",   failure_threshold=3, recovery_frames=300)
        self._cb_track    = CircuitBreaker("track",    failure_threshold=5, recovery_frames=150)
        self._cb_analytics = CircuitBreaker("analytics")
        self._cb_labeler  = CircuitBreaker("labeler")
        self._cb_drift    = CircuitBreaker("drift")
        self._cb_annotate = CircuitBreaker("annotate", failure_threshold=5, recovery_frames=150)

        logger.info(f"Pipeline initialized successfully (output_dir={self.output_dir})")

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Tuple[Any, np.ndarray]:
        """Processes a single frame through the entire vision pipeline.

        Args:
            frame: Raw BGR image from OpenCV.
            frame_idx: The current frame sequence number.

        Returns:
            A tuple containing (supervision.Detections, annotated_frame).
        """
        # Stage 1 — Detection (hard failure: no detections → return raw frame immediately)
        detections = self._cb_detect.call(
            self.detector.detect, frame, frame_idx=frame_idx, fallback=None
        )
        if detections is None:
            return sv.Detections.empty(), frame

        # Stage 2 — Tracking (soft failure: keep untracked detections, no IDs)
        if self.tracker is not None:
            detections = self._cb_track.call(
                self.tracker.update, detections,
                frame_idx=frame_idx, fallback=detections,
            )

        # Stage 3 — Analytics (non-critical: skip on failure)
        self._cb_analytics.call(
            self.analytics.update,
            detections.tracker_id if detections.tracker_id is not None else [],
            frame_idx,
            frame_idx=frame_idx, fallback=None,
        )

        # Stage 4 — Auto-labeler (non-critical: skip on failure)
        self._cb_labeler.call(
            self.auto_labeler.collect, frame, detections,
            frame_idx=frame_idx, fallback=None,
            image_id=f"frame_{frame_idx}",
        )

        # Stage 5 — Drift detection (non-critical: skip on failure)
        if detections.confidence is not None and len(detections) > 0:
            self._cb_drift.call(
                self.drift_detector.record, detections.confidence.tolist(),
                frame_idx=frame_idx, fallback=None,
            )

        # Stage 6 — Annotation (soft failure: return raw frame)
        annotated = self._cb_annotate.call(
            self._annotate, frame, detections,
            frame_idx=frame_idx, fallback=frame,
        )

        return detections, annotated

    def _annotate(self, frame: np.ndarray, detections) -> np.ndarray:
        labels = FrameAnnotator.build_labels(detections)
        return self.annotator.annotate(frame, detections, labels)

    # ---- Offline mode: batch process a video file ----

    def run_offline(self, video_path: str, output_path: str = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path or self.config.get('mode', {}).get('save_video', False):
            out_file = output_path or os.path.join(self.output_dir, 'output.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

        self.analytics.fps = fps
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            _, annotated = self.process_frame(frame, frame_idx)

            if writer is not None:
                writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        summary = self.analytics.get_summary()
        self._save_analytics(summary)
        return summary

    # ---- Online mode: real-time stream (webcam / RTSP) ----

    def run_online(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open stream source: {source}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        self.analytics.fps = fps
        frame_idx = 0
        show_live = self.config.get('mode', {}).get('show_live', True)

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                detections, annotated = self.process_frame(frame, frame_idx)

                if show_live:
                    visitors = self.analytics.unique_visitor_count
                    current = self.analytics.current_frame_count
                    cv2.putText(
                        annotated,
                        f"Visitors: {visitors} | In-frame: {current}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Vision ML - Live", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()

        summary = self.analytics.get_summary()
        self._save_analytics(summary)
        return summary

    # ---- Dispatcher ----

    def run(self, source=None):
        src = source or self.config.get('mode', {}).get('source')

        if self.mode == 'online':
            stream_src = int(src) if src is not None and str(src).isdigit() else (src or 0)
            return self.run_online(stream_src)
        else:
            if src is None:
                raise ValueError("Offline mode requires a video source path in config or argument.")
            return self.run_offline(str(src))

    def flush_labels(self, output_dir: str = 'data/auto_labeled'):
        """Manually flush collected auto-labels to local or Roboflow."""
        count = len(self.auto_labeler.pending_labels)
        if count > 0:
            self.auto_labeler.flush(output_dir)
            logger.info(f"Flushed {count} auto-labels")
        else:
            logger.info("No pending labels to flush")
        return count

    def reset(self):
        if self.tracker is not None:
            self.tracker.reset()
        self.analytics.reset()
        self.drift_detector.reset()

    def _save_analytics(self, summary: dict):
        # Append dual-detector stats if available
        if isinstance(self.detector, DualDetector):
            summary['dual_detector'] = self.detector.stats

        # Append drift metrics
        self.drift_detector.check()
        summary['drift'] = self.drift_detector.get_metrics()

        if self.config.get('analytics', {}).get('output_json', True):
            out_path = os.path.join(self.output_dir, 'analytics.json')
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Analytics saved to {out_path}")
