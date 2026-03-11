import sys
import os
import json
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.inference.pipeline import InferencePipeline
from vision_ml.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Vision ML Inference Pipeline")
    parser.add_argument('--config', type=str, default='config/inference/base.yaml')
    parser.add_argument('--source', type=str, default=None,
                        help='Video path (offline) or camera index/RTSP URL (online)')
    parser.add_argument('--mode', type=str, default=None, choices=['online', 'offline'],
                        help='Override mode from config')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode:
        config.setdefault('mode', {})['type'] = args.mode
    if args.source:
        config.setdefault('mode', {})['source'] = args.source

    pipeline = InferencePipeline(config)
    mode = config.get('mode', {}).get('type', 'offline')
    source = args.source or config.get('mode', {}).get('source')

    print(f"[Inference] Mode: {mode} | Source: {source}")

    if mode == 'online':
        summary = pipeline.run_online(int(source) if str(source).isdigit() else (source or 0))
    else:
        if source is None:
            print("[Inference] ERROR: Offline mode requires --source <video_path>")
            return
        summary = pipeline.run_offline(source, args.output)

    print("\n========= Visitor Analytics Summary =========")
    print(f"  Unique Visitors:       {summary.get('unique_visitors', 0)}")
    print(f"  Total Frames:          {summary.get('total_frames', 0)}")
    print(f"  Avg Visitors/Frame:    {summary.get('avg_visitors_per_frame', 0)}")
    print(f"  Avg Dwell Time (sec):  {summary.get('avg_dwell_time_seconds', 0)}")
    print("=============================================\n")


if __name__ == '__main__':
    main()
