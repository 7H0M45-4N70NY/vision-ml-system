#!/usr/bin/env python3
"""Drift detection demo: simulate confidence degradation and trigger retraining.

Demonstrates the DriftDetector module by simulating a scenario where model
confidence gradually degrades over time (e.g., lighting change, new objects).
Shows how the system detects drift and flags retraining.

No model or video required — uses synthetic confidence scores.

Usage:
    python examples/demo_drift.py
    python examples/demo_drift.py --threshold 0.4 --window-size 200
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision_ml.training.drift_detector import DriftDetector
from vision_ml.analytics.analytics_db import AnalyticsDB

import numpy as np


def simulate_confidence_stream(num_frames: int = 600, degradation_start: int = 300):
    """Simulate per-frame detection confidences with gradual degradation.

    First half: stable high confidence (0.7-0.95)
    Second half: gradual decline to 0.2-0.5 (simulating distribution shift)
    """
    rng = np.random.RandomState(42)
    confidences = []

    for i in range(num_frames):
        num_detections = rng.randint(1, 5)

        if i < degradation_start:
            # Stable period: high confidence
            confs = rng.uniform(0.7, 0.95, size=num_detections).tolist()
        else:
            # Degradation: confidence drops linearly
            progress = (i - degradation_start) / (num_frames - degradation_start)
            base = 0.8 - (0.5 * progress)  # 0.8 → 0.3
            noise = 0.1
            confs = rng.uniform(
                max(0.1, base - noise),
                min(1.0, base + noise),
                size=num_detections,
            ).tolist()

        confidences.append(confs)

    return confidences


def main():
    parser = argparse.ArgumentParser(description="Drift Detection Demo")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for drift detection')
    parser.add_argument('--window-size', type=int, default=200,
                        help='Sliding window size for confidence buffer')
    parser.add_argument('--check-interval', type=int, default=50,
                        help='Check drift every N frames')
    parser.add_argument('--num-frames', type=int, default=800,
                        help='Total frames to simulate')
    parser.add_argument('--degradation-start', type=int, default=300,
                        help='Frame where confidence degradation begins')
    args = parser.parse_args()

    config = {
        'drift': {
            'enabled': True,
            'method': 'confidence_drop',
            'confidence_threshold': args.threshold,
            'window_size': args.window_size,
            'check_interval': args.check_interval,
        }
    }

    print("=" * 60)
    print("  DRIFT DETECTION DEMO")
    print("=" * 60)
    print(f"  Threshold:          {args.threshold}")
    print(f"  Window size:        {args.window_size}")
    print(f"  Check interval:     {args.check_interval}")
    print(f"  Total frames:       {args.num_frames}")
    print(f"  Degradation start:  frame {args.degradation_start}")
    print("=" * 60)
    print()

    # Step 1: Generate synthetic confidence stream
    confidence_stream = simulate_confidence_stream(
        num_frames=args.num_frames,
        degradation_start=args.degradation_start,
    )

    # Step 2: Feed through DriftDetector
    detector = DriftDetector(config)
    drift_frame = None
    checkpoints = []

    print(f"{'Frame':>6}  {'Detections':>10}  {'Avg Conf':>8}  {'Drift Score':>11}  {'Status'}")
    print("-" * 60)

    for frame_idx, confs in enumerate(confidence_stream):
        detector.record(confs)
        was_detected = detector.drift_detected

        drift_now = detector.check()

        # Print at check intervals or when drift first detected
        if frame_idx % args.check_interval == 0 or (drift_now and not was_detected):
            metrics = detector.get_metrics()
            status = "DRIFT!" if metrics['drift_detected'] else "OK"
            print(f"{frame_idx:>6}  {len(confs):>10}  "
                  f"{metrics['avg_confidence']:>8.4f}  "
                  f"{metrics['drift_score']:>11.4f}  "
                  f"{status}")
            checkpoints.append({
                'frame': frame_idx,
                'avg_confidence': metrics['avg_confidence'],
                'drift_score': metrics['drift_score'],
                'drift_detected': metrics['drift_detected'],
            })

            if drift_now and drift_frame is None:
                drift_frame = frame_idx

    # Step 3: Final summary
    final = detector.get_metrics()
    print()
    print("=" * 60)
    print("  DRIFT DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Total detections:      {final['total_detections']}")
    print(f"  Final avg confidence:  {final['avg_confidence']:.4f}")
    print(f"  Final drift score:     {final['drift_score']:.4f}")
    print(f"  Low confidence ratio:  {final['low_confidence_ratio']:.4f}")
    print(f"  Drift detected:        {final['drift_detected']}")

    if drift_frame is not None:
        print(f"  First drift at frame:  {drift_frame}")
        print(f"  Degradation started:   frame {args.degradation_start}")
        detection_delay = drift_frame - args.degradation_start
        print(f"  Detection delay:       {detection_delay} frames "
              f"({detection_delay / 30:.1f}s @ 30fps)")
    else:
        print(f"  Drift NOT detected within {args.num_frames} frames")

    print("=" * 60)

    # Step 4: Persist metrics and trigger retraining event
    db = AnalyticsDB()
    run_id = db.save_inference_run({
        'source_type': 'demo_drift_simulation',
        'total_frames': args.num_frames,
        'unique_visitors': 0,
        'avg_confidence': final['avg_confidence'],
        'drift_score': final['drift_score'],
    })
    print(f"\n[Demo] Inference run saved: {run_id}")

    if final['drift_detected']:
        event_id = db.save_training_event({
            'trigger_type': 'drift',
            'dataset_size': 0,
            'drift_score': final['drift_score'],
            'model_version': 'v1',
        })
        print(f"[Demo] Retraining event triggered: {event_id}")
        print(f"\n[Demo] In production, this would kick off:")
        print(f"  python scripts/train.py --trigger drift")
    else:
        print(f"\n[Demo] No drift detected — model is healthy.")

    print(f"\n[Demo] Next steps:")
    print(f"  python scripts/pipeline_cli.py --action drift-check")
    print(f"  python scripts/analytics_cli.py --action summary")


if __name__ == '__main__':
    main()
