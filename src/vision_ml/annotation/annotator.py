import numpy as np
import supervision as sv

from ..logging import get_logger

logger = get_logger(__name__)


class FrameAnnotator:
    def __init__(self, config: dict):
        ann_cfg = config.get('annotation', {})

        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None

        if ann_cfg.get('bounding_box', {}).get('enabled', True):
            # Try BoundingBoxAnnotator (newer Supervision), fall back to BoxAnnotator (older)
            try:
                self.box_annotator = sv.BoundingBoxAnnotator(
                    thickness=ann_cfg.get('bounding_box', {}).get('thickness', 2),
                )
                logger.info("Using BoundingBoxAnnotator")
            except AttributeError:
                try:
                    self.box_annotator = sv.BoxAnnotator(
                        thickness=ann_cfg.get('bounding_box', {}).get('thickness', 2),
                    )
                    logger.info("Using BoxAnnotator (legacy)")
                except AttributeError:
                    logger.warning("No box annotator available")

        if ann_cfg.get('label', {}).get('enabled', True):
            try:
                self.label_annotator = sv.LabelAnnotator(
                    text_scale=ann_cfg.get('label', {}).get('text_scale', 0.5),
                    text_thickness=ann_cfg.get('label', {}).get('text_thickness', 1),
                )
            except Exception as e:
                logger.warning(f"LabelAnnotator error: {e}")

        if ann_cfg.get('trace', {}).get('enabled', False):
            try:
                self.trace_annotator = sv.TraceAnnotator(
                    trace_length=ann_cfg.get('trace', {}).get('trace_length', 60),
                )
            except Exception as e:
                logger.warning(f"TraceAnnotator error: {e}")

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: list = None,
    ) -> np.ndarray:
        annotated = frame.copy()

        if self.box_annotator is not None:
            annotated = self.box_annotator.annotate(
                scene=annotated, detections=detections
            )

        if self.label_annotator is not None and labels is not None:
            annotated = self.label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )

        if self.trace_annotator is not None:
            annotated = self.trace_annotator.annotate(
                scene=annotated, detections=detections
            )

        return annotated

    @staticmethod
    def build_labels(detections: sv.Detections) -> list:
        labels = []
        for i in range(len(detections)):
            tracker_id = (
                detections.tracker_id[i]
                if detections.tracker_id is not None
                else "?"
            )
            confidence = (
                f"{detections.confidence[i]:.2f}"
                if detections.confidence is not None
                else ""
            )
            labels.append(f"#{tracker_id} {confidence}")
        return labels
