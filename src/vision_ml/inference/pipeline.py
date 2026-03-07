import os
import json
import cv2
import numpy as np

from ..detection.detector_factory import DetectorFactory
from ..detection.dual_detector import DualDetector
from ..tracking.tracker_factory import TrackerFactory
from ..annotation.annotator import FrameAnnotator
from ..analytics.visitor_analytics import VisitorAnalytics


class InferencePipeline:
    def __init__(self, config: dict):
        self.config = config
        use_dual = config.get('detection', {}).get('use_dual_detector', False)

        # Detector: single (cached singleton) or dual (with frame collection)
        if use_dual:
            self.detector = DualDetector(config)
        else:
            self.detector = DetectorFactory.from_config(config)

        # Tracker: always a new stateful instance
        if config.get('tracking', {}).get('enabled', True):
            self.tracker = TrackerFactory.from_config(config)
        else:
            self.tracker = None

        self.annotator = FrameAnnotator(config)
        self.analytics = VisitorAnalytics(config)

        self.mode = config.get('mode', {}).get('type', 'offline')
        self.output_dir = config.get('mode', {}).get('output_dir', 'runs/inference')
        os.makedirs(self.output_dir, exist_ok=True)

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> tuple:
        detections = self.detector.detect(frame)

        if self.tracker is not None:
            detections = self.tracker.update(detections)

        self.analytics.update(
            detections.tracker_id if detections.tracker_id is not None else [],
            frame_idx,
        )

        labels = FrameAnnotator.build_labels(detections)
        annotated = self.annotator.annotate(frame, detections, labels)
        return detections, annotated

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

    def reset(self):
        if self.tracker is not None:
            self.tracker.reset()
        self.analytics.reset()

    def _save_analytics(self, summary: dict):
        # Append dual-detector stats if available
        if isinstance(self.detector, DualDetector):
            summary['dual_detector'] = self.detector.stats

        if self.config.get('analytics', {}).get('output_json', True):
            out_path = os.path.join(self.output_dir, 'analytics.json')
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Analytics saved to {out_path}")
