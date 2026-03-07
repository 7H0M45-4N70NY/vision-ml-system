"""Dual-detector: YOLO (fast) + RF-DETR (accurate fallback).

When primary has low confidence, secondary re-detects and the frame + labels
are saved to disk for offline training. No Kafka — just a disk buffer.
"""

import os
import json
import cv2
import numpy as np
import supervision as sv
from typing import Tuple

from .detector_factory import DetectorFactory


class DualDetector:
    """Primary (YOLO) + secondary (RF-DETR) with frame collection for training.

    Low-confidence frames are saved to `data/low_confidence_frames/` so the
    training pipeline can use them as pseudo-labeled data to retrain the primary.
    """

    def __init__(self, config: dict):
        det_cfg = config.get('detection', {})
        self.primary_type = det_cfg.get('primary_detector', 'yolo11n')
        self.secondary_type = det_cfg.get('secondary_detector', 'rfdetr')
        self.confidence_threshold = det_cfg.get('dual_confidence_threshold', 0.5)
        self.frame_save_dir = det_cfg.get('frame_save_dir', 'data/low_confidence_frames')

        self.primary = DetectorFactory.get(self.primary_type, config)
        self.secondary = DetectorFactory.get(self.secondary_type, config)

        self._saved_count = 0
        self._secondary_calls = 0
        self._total_frames = 0

    def detect(self, image: np.ndarray) -> sv.Detections:
        dets, _ = self.detect_with_source(image)
        return dets

    def detect_with_source(self, image: np.ndarray) -> Tuple[sv.Detections, np.ndarray]:
        self._total_frames += 1
        primary_dets = self.primary.detect(image)

        has_low_conf = (
            primary_dets.confidence is not None
            and len(primary_dets) > 0
            and (primary_dets.confidence < self.confidence_threshold).any()
        )

        if not has_low_conf:
            return primary_dets, np.array(['primary'] * len(primary_dets))

        # Secondary re-detects the full frame
        self._secondary_calls += 1
        secondary_dets = self.secondary.detect(image)

        # Save frame + secondary labels to disk for offline training
        self._save_frame(image, secondary_dets)

        # Merge: keep high-conf primary + non-overlapping secondary
        high_mask = primary_dets.confidence >= self.confidence_threshold
        high_conf = primary_dets[high_mask]

        if len(high_conf) > 0 and len(secondary_dets) > 0:
            iou = sv.box_iou_batch(high_conf.xyxy, secondary_dets.xyxy)
            unmatched = iou.max(axis=0) < 0.3
            sec_only = secondary_dets[unmatched]
        else:
            sec_only = secondary_dets

        if len(high_conf) == 0 and len(sec_only) == 0:
            return sv.Detections.empty(), np.array([])

        merged = sv.Detections.merge([high_conf, sec_only])
        sources = np.concatenate([
            np.array(['primary'] * len(high_conf)),
            np.array(['secondary'] * len(sec_only)),
        ])
        return merged, sources

    def _save_frame(self, image: np.ndarray, detections: sv.Detections):
        """Save frame + labels to disk for offline training. No Kafka needed."""
        os.makedirs(self.frame_save_dir, exist_ok=True)
        frame_id = f"frame_{self._saved_count:06d}"

        cv2.imwrite(os.path.join(self.frame_save_dir, f"{frame_id}.jpg"), image)

        label = {
            'boxes': detections.xyxy.tolist(),
            'confidences': detections.confidence.tolist() if detections.confidence is not None else [],
            'class_ids': detections.class_id.tolist() if detections.class_id is not None else [],
        }
        with open(os.path.join(self.frame_save_dir, f"{frame_id}.json"), 'w') as f:
            json.dump(label, f)

        self._saved_count += 1

    @property
    def secondary_ratio(self) -> float:
        return self._secondary_calls / max(self._total_frames, 1)

    @property
    def stats(self) -> dict:
        return {
            'total_frames': self._total_frames,
            'secondary_calls': self._secondary_calls,
            'secondary_ratio': round(self.secondary_ratio, 3),
            'frames_saved': self._saved_count,
        }
