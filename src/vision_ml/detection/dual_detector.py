"""Dual-detector: YOLO (fast) + RF-DETR (accurate fallback).

Three operational modes:
  - hot:    Primary detector only. Minimal overhead. Fastest inference.
  - inline: Dual detector runs during inference. Real-time feedback but slower.
  - batch:  Primary only during inference; save low-confidence frames for offline
            secondary detection via batch job. Fast inference + deferred analysis.

Low-confidence frames are saved to disk for offline training/analysis.
"""

import os
import re
import json
from pathlib import Path
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

    Low-confidence frames saved to `data/low_confidence_frames/` for training/analysis,
    governed by active learning sampling controls (temporal cooldown, noise floor, max budget).
    """

    def __init__(self, config: dict):
        det_cfg = config.get('detection', {})
        self.mode = det_cfg.get('use_dual_detector', 'batch')  # false | 'inline' | 'batch'
        self.primary_type = det_cfg.get('primary_detector', 'yolo11n')
        self.secondary_type = det_cfg.get('secondary_detector', 'rfdetr')
        self.confidence_threshold = det_cfg.get('dual_confidence_threshold', 0.5)
        self.min_confidence_floor = det_cfg.get('dual_min_confidence_floor', 0.15)
        self.save_frames = det_cfg.get('save_low_confidence_frames', True)
        self.frame_save_dir = det_cfg.get('low_confidence_dir', 'data/low_confidence_frames')

        # Active Learning Frame Sampling Controls (prevents dataset bloating / redundancy)
        self.save_cooldown_frames = det_cfg.get('save_cooldown_frames', 30)
        self.max_saved_frames = det_cfg.get('max_saved_frames', 25)
        self.similarity_threshold = det_cfg.get('similarity_threshold', 0.95)

        self.primary = DetectorFactory.get(self.primary_type, config)
        # Only load secondary if mode is 'inline'
        self.secondary = (
            DetectorFactory.get(self.secondary_type, config)
            if self.mode == 'inline'
            else None
        )

        self._saved_count = 0
        self._saved_in_stream = 0
        self._secondary_calls = 0
        self._total_frames = 0
        self._last_saved_frame_idx = -float('inf')
        self._last_saved_thumb = None
        self.source_prefix = 'frame'

        # Auto-configure source prefix if present in config
        source = config.get('mode', {}).get('source')
        if source:
            self.set_source(source)

    def set_source(self, source: str):
        """Set the video or stream source name for frame prefixing and reset stream state.

        Args:
            source: Path to video file or stream identifier (e.g., 'data/sample_videos/video_1.mp4').
        """
        if source:
            stem = Path(str(source)).stem
            clean_stem = re.sub(r'[^a-zA-Z0-9_\-]', '_', stem).strip('_')
            self.source_prefix = clean_stem or 'frame'
        else:
            self.source_prefix = 'frame'

        # Reset per-stream budget and diversity state
        self._saved_in_stream = 0
        self._total_frames = 0
        self._last_saved_frame_idx = -float('inf')
        self._last_saved_thumb = None

    def _is_valid_low_conf(self, detections: sv.Detections) -> bool:
        """Check if detections contain borderline failure candidates (above noise floor, below threshold)."""
        if detections.confidence is None or len(detections) == 0:
            return False
        return (
            (detections.confidence >= self.min_confidence_floor)
            & (detections.confidence < self.confidence_threshold)
        ).any()

    def _should_save_frame(self, image: np.ndarray, detections: sv.Detections) -> bool:
        """Determines whether a low-confidence candidate should be saved based on active learning controls."""
        if not self.save_frames:
            return False

        # 1. Quota / Budget check per stream
        if self.max_saved_frames is not None and self._saved_in_stream >= self.max_saved_frames:
            return False

        # 2. Temporal cooldown check
        if (self._total_frames - self._last_saved_frame_idx) < max(1, self.save_cooldown_frames):
            return False

        # 3. Visual diversity check (reject near-identical static frames)
        if image is not None and self.similarity_threshold < 1.0:
            try:
                gray_thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
                if self._last_saved_thumb is not None:
                    diff = np.mean(np.abs(gray_thumb.astype(float) - self._last_saved_thumb.astype(float))) / 255.0
                    if diff < (1.0 - self.similarity_threshold):
                        return False
                self._last_saved_thumb = gray_thumb
            except Exception:
                pass

        return True

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

        # Check for valid low-confidence detections (between noise floor and threshold)
        has_low_conf = self._is_valid_low_conf(primary_dets)

        # Batch mode: save low-confidence frames for offline secondary detection
        if self.mode == 'batch':
            if has_low_conf and self._should_save_frame(image, primary_dets):
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
            save_dets = secondary_dets if len(secondary_dets) > 0 else primary_dets
            if self._should_save_frame(image, save_dets):
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
        """Save frame + labels to disk for offline training. Non-colliding filenames."""
        os.makedirs(self.frame_save_dir, exist_ok=True)

        # Find non-colliding frame identifier within the configured directory
        idx = self._saved_in_stream
        while True:
            frame_id = f"{self.source_prefix}_frame_{idx:06d}"
            img_path = os.path.join(self.frame_save_dir, f"{frame_id}.jpg")
            json_path = os.path.join(self.frame_save_dir, f"{frame_id}.json")
            if not os.path.exists(img_path) and not os.path.exists(json_path):
                break
            idx += 1

        cv2.imwrite(img_path, image)

        label = {
            'image_id': frame_id,
            'source_prefix': self.source_prefix,
            'boxes': detections.xyxy.tolist(),
            'confidences': detections.confidence.tolist() if detections.confidence is not None else [],
            'class_ids': detections.class_id.tolist() if detections.class_id is not None else [],
        }
        with open(json_path, 'w') as f:
            json.dump(label, f, indent=2)

        self._saved_in_stream += 1
        self._saved_count += 1
        self._last_saved_frame_idx = self._total_frames

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
            'stream_frames_saved': self._saved_in_stream,
            'max_saved_frames': self.max_saved_frames,
            'source_prefix': self.source_prefix,
        }
