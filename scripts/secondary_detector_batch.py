"""Batch secondary detector — run offline on saved low-confidence frames.

This script processes frames saved by DualDetector in 'batch' mode, running
the secondary detector (RF-DETR) on low-confidence frames and comparing results
with the primary detector. Useful for deferred analysis without slowing inference.

Usage:
    python scripts/secondary_detector_batch.py --input data/low_confidence_frames \
        --output data/secondary_analysis --config config/inference/base.yaml
"""

import sys
import os
import argparse
import json
import glob
from pathlib import Path
from dotenv import load_dotenv

import cv2
import numpy as np

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.utils.config import load_config
from vision_ml.detection.detector_factory import DetectorFactory


def main():
    parser = argparse.ArgumentParser(
        description='Batch secondary detector for low-confidence frames'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/low_confidence_frames',
        help='Input directory with low-confidence frames (from batch mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/secondary_analysis',
        help='Output directory for secondary detector results'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/inference/base.yaml',
        help='Inference config'
    )
    parser.add_argument(
        '--secondary-detector',
        type=str,
        default=None,
        help='Override secondary detector type (e.g., rfdetr)'
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.secondary_detector:
        config.setdefault('detection', {})['secondary_detector'] = args.secondary_detector

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize detectors
    det_cfg = config.get('detection', {})
    primary_type = det_cfg.get('primary_detector', 'yolo11n')
    secondary_type = det_cfg.get('secondary_detector', 'rfdetr')

    print(f"[SecondaryBatch] Primary detector: {primary_type}")
    print(f"[SecondaryBatch] Secondary detector: {secondary_type}")

    primary = DetectorFactory.get(primary_type, config)
    secondary = DetectorFactory.get(secondary_type, config)

    # Find all low-confidence frames
    frame_files = sorted(glob.glob(os.path.join(args.input, '*.jpg')))
    if not frame_files:
        print(f"[SecondaryBatch] No frames found in {args.input}")
        return

    print(f"[SecondaryBatch] Processing {len(frame_files)} frames...")

    results = []
    agreement_count = 0
    disagreement_count = 0

    for frame_path in frame_files:
        frame_id = Path(frame_path).stem
        label_path = os.path.join(args.input, f"{frame_id}.json")

        # Load frame and original labels
        image = cv2.imread(frame_path)
        if image is None:
            print(f"[SecondaryBatch] Failed to load {frame_path}")
            continue

        with open(label_path, 'r') as f:
            original_label = json.load(f)

        # Run secondary detector
        secondary_dets = secondary.detect(image)

        # Compare with original (primary) detections
        primary_boxes = np.array(original_label.get('boxes', []))
        secondary_boxes = secondary_dets.xyxy.numpy() if len(secondary_dets) > 0 else np.array([])

        # Simple agreement check: IoU > 0.3
        if len(primary_boxes) > 0 and len(secondary_boxes) > 0:
            from supervision import box_iou_batch
            iou = box_iou_batch(primary_boxes, secondary_boxes)
            agreement = (iou.max(axis=1) > 0.3).sum()
            agreement_count += agreement
            disagreement_count += len(primary_boxes) - agreement
        elif len(primary_boxes) > 0:
            disagreement_count += len(primary_boxes)
        elif len(secondary_boxes) > 0:
            agreement_count += len(secondary_boxes)

        # Save analysis
        analysis = {
            'frame_id': frame_id,
            'primary_detections': len(primary_boxes),
            'secondary_detections': len(secondary_boxes),
            'primary_boxes': primary_boxes.tolist() if len(primary_boxes) > 0 else [],
            'secondary_boxes': secondary_boxes.tolist() if len(secondary_boxes) > 0 else [],
            'secondary_confidences': (
                secondary_dets.confidence.tolist()
                if secondary_dets.confidence is not None
                else []
            ),
        }
        results.append(analysis)

    # Save results
    output_file = os.path.join(args.output, 'secondary_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    total_detections = agreement_count + disagreement_count
    agreement_rate = (
        agreement_count / total_detections * 100
        if total_detections > 0
        else 0
    )

    print(f"\n[SecondaryBatch] Analysis Complete")
    print(f"  Frames processed: {len(results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Agreement rate: {agreement_rate:.1f}%")
    print(f"  Results saved to: {output_file}")


if __name__ == '__main__':
    main()
