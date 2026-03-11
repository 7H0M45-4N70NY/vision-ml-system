"""Dual-detector: YOLO (fast) + RF-DETR (accurate fallback).

Three operational modes:
  - hot:    Primary detector only. Minimal overhead. Fastest inference.
  - inline: Dual detector runs during inference. Real-time feedback but slower.
  - batch:  Primary only during inference; save low-confidence frames for offline
            secondary detection via batch job. Fast inference + deferred analysis.

Low-confidence frames are saved to disk for offline training/analysis.
"""

import os
import json
import cv2
import numpy as np
import supervision as sv
from typing import Tuple

from .detector_factory import DetectorFactory


class DualDetector:
    """Primary (YOLO) + secondary (RF-DETR) with configurable modes.

    Modes:
      - hot:    Primary only (no secondary, no frame saving)
      - inline: Primary + secondary during inference (real-time, slower)
      - batch:  Primary only; save low-confidence frames for offline secondary detection

    Low-confidence frames saved to `data/low_confidence_frames/` for training/analysis.
    """

    def __init__(self, config: dict):
        det_cfg = config.get('detection', {})
        self.mode = det_cfg.get('use_dual_detector', 'batch')  # false | 'inline' | 'batch'
        self.primary_type = det_cfg.get('primary_detector', 'yolo11n')
        self.secondary_type = det_cfg.get('secondary_detector', 'rfdetr')
        self.confidence_threshold = det_cfg.get('dual_confidence_threshold', 0.5)
        self.save_frames = det_cfg.get('save_low_confidence_frames', True)
        self.frame_save_dir = det_cfg.get('low_confidence_dir', 'data/low_confidence_frames')

        self.primary = DetectorFactory.get(self.primary_type, config)
        # Only load secondary if mode is 'inline'
        self.secondary = (
            DetectorFactory.get(self.secondary_type, config)
            if self.mode == 'inline'
            else None
        )

        self._saved_count = 0
        self._secondary_calls = 0
        self._total_frames = 0

    def detect(self, image: np.ndarray) -> sv.Detections:
        dets, _ = self.detect_with_source(image)
        return dets

    def detect_with_source(self, image: np.ndarray) -> Tuple[sv.Detections, np.ndarray]:
        """Detect with mode-specific behavior.

        Modes:
          - hot:    Primary only. No secondary, no frame saving. Fastest.
          - inline: Primary + secondary (real-time). Slower but immediate feedback.
          - batch:  Primary only; save low-confidence frames for offline secondary.
        """
        self._total_frames += 1
        primary_dets = self.primary.detect(image)

        # Hot path: return primary detections only
        if self.mode is False or self.mode == 'hot':
            return primary_dets, np.array(['primary'] * len(primary_dets))

        # Check for low-confidence detections
        has_low_conf = (
            primary_dets.confidence is not None
            and len(primary_dets) > 0
            and (primary_dets.confidence < self.confidence_threshold).any()
        )

        # Batch mode: save low-confidence frames for offline secondary detection
        if self.mode == 'batch':
            if has_low_conf and self.save_frames:
                self._save_frame(image, primary_dets)
            return primary_dets, np.array(['primary'] * len(primary_dets))

        # Inline mode: run secondary detector during inference
        if self.mode == 'inline':
            if not has_low_conf:
                return primary_dets, np.array(['primary'] * len(primary_dets))

            # Secondary re-detects the full frame
            self._secondary_calls += 1
            secondary_dets = self.secondary.detect(image)

            # Save frame + best-available labels for offline training
            if self.save_frames:
                save_dets = secondary_dets if len(secondary_dets) > 0 else primary_dets
                self._save_frame(image, save_dets)

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

        # Fallback: return primary
        return primary_dets, np.array(['primary'] * len(primary_dets))

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
