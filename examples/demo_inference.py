#!/usr/bin/env python3
"""End-to-end inference demo: generate video → detect → track → analytics.

Demonstrates the full InferencePipeline without any external data.
Generates a synthetic video, runs YOLO person detection + ByteTrack,
and prints visitor analytics.

Usage:
    python examples/demo_inference.py
    python examples/demo_inference.py --source path/to/your/video.mp4
    python examples/demo_inference.py --dual-detector batch
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from vision_ml.inference.pipeline import InferencePipeline
from vision_ml.utils.config import load_config
from vision_ml.analytics.analytics_db import AnalyticsDB


def main():
    parser = argparse.ArgumentParser(description="Inference Pipeline Demo")
    parser.add_argument('--source', type=str, default=None,
                        help='Video path. If omitted, generates a synthetic test video.')
    parser.add_argument('--config', type=str, default='config/inference/base.yaml')
    parser.add_argument('--output-dir', type=str, default='runs/demo_inference')
    parser.add_argument('--dual-detector', type=str, default='false',
                        choices=['false', 'batch', 'inline'],
                        help='Dual detector mode')
    parser.add_argument('--save-video', action='store_true', default=True,
                        help='Save annotated output video')
    args = parser.parse_args()

    # Step 1: Get or generate video
    if args.source and os.path.isfile(args.source):
        video_path = args.source
        print(f"[Demo] Using provided video: {video_path}")
    else:
        print("[Demo] No video provided — generating synthetic test scene...")
        from examples.generate_sample_video import generate_video
        video_path = generate_video(
            output_path='data/sample_videos/test_scene.mp4',
            duration=10, fps=30, num_people=3,
        )

    # Step 2: Configure pipeline
    config = load_config(args.config)
    config['mode'] = {
        'type': 'offline',
        'source': video_path,
        'output_dir': args.output_dir,
        'save_video': args.save_video,
        'show_live': False,
    }
    config['detection']['use_dual_detector'] = (
        args.dual_detector if args.dual_detector != 'false' else False
    )

    # Step 3: Run pipeline
    print(f"\n[Demo] Initializing pipeline...")
    print(f"  Detector:       {config['detection']['detector_type']}")
    print(f"  Dual mode:      {config['detection']['use_dual_detector']}")
    print(f"  Tracker:        {config['tracking']['tracker_type']}")
    print(f"  Drift enabled:  {config['drift']['enabled']}")
    print()

    pipeline = InferencePipeline(config)

    start = time.time()
    summary = pipeline.run_offline(video_path)
    elapsed = time.time() - start

    # Step 4: Display results
    total_frames = summary.get('total_frames', 0)
    fps = total_frames / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 55)
    print("  INFERENCE RESULTS")
    print("=" * 55)
    print(f"  Video:                 {video_path}")
    print(f"  Total frames:          {total_frames}")
    print(f"  Processing time:       {elapsed:.2f}s ({fps:.1f} fps)")
    print(f"  Unique visitors:       {summary.get('unique_visitors', 0)}")
    print(f"  Peak visitors/frame:   {summary.get('peak_visitors_per_frame', 0)}")
    print(f"  Avg visitors/frame:    {summary.get('avg_visitors_per_frame', 0)}")
    print(f"  Avg dwell time:        {summary.get('avg_dwell_time_seconds', 0):.2f}s")

    # Drift metrics
    drift = summary.get('drift', {})
    if drift:
        print(f"\n  --- Drift Metrics ---")
        print(f"  Avg confidence:        {drift.get('avg_confidence', 0):.4f}")
        print(f"  Drift score:           {drift.get('drift_score', 0):.4f}")
        print(f"  Drift detected:        {drift.get('drift_detected', False)}")

    # Dual detector stats
    dual = summary.get('dual_detector', {})
    if dual:
        print(f"\n  --- Dual Detector ---")
        print(f"  Secondary calls:       {dual.get('secondary_calls', 0)}")
        print(f"  Secondary ratio:       {dual.get('secondary_ratio', 0):.3f}")
        print(f"  Frames saved:          {dual.get('frames_saved', 0)}")

    print("=" * 55)

    # Step 5: Persist to analytics DB
    db = AnalyticsDB()
    run_id = db.save_inference_run({
        'source_type': 'demo_synthetic',
        'total_frames': total_frames,
        'unique_visitors': summary.get('unique_visitors', 0),
        'avg_dwell_time_seconds': summary.get('avg_dwell_time_seconds', 0),
        'use_dual_detector': bool(config['detection']['use_dual_detector']),
        'avg_confidence': drift.get('avg_confidence', 0),
        'drift_score': drift.get('drift_score', 0),
    })
    dwell_times = summary.get('dwell_times', {})
    if dwell_times:
        db.save_visitor_analytics(run_id, dwell_times)

    print(f"\n[Demo] Results saved to DB as: {run_id}")
    print(f"[Demo] Analytics JSON: {args.output_dir}/analytics.json")
    if args.save_video:
        print(f"[Demo] Annotated video: {args.output_dir}/output.mp4")
    print(f"\n[Demo] Next steps:")
    print(f"  python scripts/analytics_cli.py --action summary")
    print(f"  python examples/demo_drift.py")


if __name__ == '__main__':
    main()
