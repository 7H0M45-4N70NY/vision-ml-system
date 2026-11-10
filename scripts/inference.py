import sys
import os
import glob
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.inference.pipeline import InferencePipeline
from vision_ml.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Vision ML Inference Pipeline (Zero-Friction Runner)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('video_files', nargs='*', default=[],
                        help='Path to video file(s), directory, or camera index')
    parser.add_argument('--source', type=str, default=None,
                        help='Explicit video path (offline) or camera index/RTSP URL (online)')
    parser.add_argument('--all', action='store_true',
                        help='Run inference on all sample videos in data/sample_videos/')
    parser.add_argument('--config', type=str, default='config/inference/base.yaml',
                        help='Path to inference configuration YAML')
    parser.add_argument('--mode', type=str, default=None, choices=['online', 'offline'],
                        help='Override mode (auto-detected if omitted)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output annotated video path or directory')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'cuda:0'],
                        help='Compute device (auto-detects RTX 4050 / CUDA if available)')
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Resolve Input Sources
    sources = []
    if args.all:
        sources = sorted(glob.glob("data/sample_videos/*.mp4"))
        print(f"[Inference] --all flag specified -> Found {len(sources)} sample video(s)")
    elif args.video_files:
        for item in args.video_files:
            if os.path.isdir(item):
                sources.extend(sorted(glob.glob(os.path.join(item, "*.mp4"))))
            elif os.path.isfile(item) or str(item).isdigit() or str(item).startswith(('rtsp://', 'http://', 'https://')):
                sources.append(item)
    elif args.source:
        sources = [args.source]
    elif config.get('mode', {}).get('source'):
        sources = [config['mode']['source']]
    else:
        sample_candidates = sorted(glob.glob("data/sample_videos/*.mp4"))
        if sample_candidates:
            sources = [sample_candidates[0]]
            print(f"[Inference] No source specified -> Auto-detected sample video: {sources[0]}")
        else:
            print("[Inference] ERROR: No video source provided and no sample videos found in data/sample_videos/.")
            print("  Usage: python scripts/inference.py path/to/video.mp4")
            return

    if not sources:
        print("[Inference] ERROR: No valid video sources found to process.")
        return

    # 2. Hardware Acceleration Auto-Detection
    if args.device:
        config.setdefault('inference', {})['device'] = args.device
    else:
        try:
            import torch
            if torch.cuda.is_available():
                config.setdefault('inference', {})['device'] = 'cuda:0'
                config.setdefault('model', {})['precision'] = 'fp16'
        except ImportError:
            pass

    active_device = config.get('inference', {}).get('device', 'cpu')
    print(f"\n========================================================")
    print(f" VISION ML INFERENCE PIPELINE")
    print(f"========================================================")
    print(f"  Total Sources: {len(sources)}")
    print(f"  Device:        {active_device.upper()}")
    print(f"========================================================\n")

    # 3. Instantiate pipeline once for high throughput
    pipeline = InferencePipeline(config)

    for idx, source in enumerate(sources, 1):
        is_stream = str(source).isdigit() or str(source).startswith(('rtsp://', 'http://', 'https://'))
        mode = args.mode or ('online' if is_stream else 'offline')

        # Resolve output path
        output_path = args.output
        if mode == 'offline':
            out_dir = Path("runs/inference")
            out_dir.mkdir(parents=True, exist_ok=True)
            if not output_path or len(sources) > 1 or os.path.isdir(str(output_path)):
                video_stem = Path(str(source)).stem
                output_path = str(out_dir / f"{video_stem}_annotated.mp4")

        print(f"\n--- [{idx}/{len(sources)}] Processing: {source} (Mode: {mode.upper()}) ---")
        if mode == 'online':
            summary = pipeline.run_online(int(source) if str(source).isdigit() else source)
        else:
            summary = pipeline.run_offline(str(source), output_path)

        # Display Analytics Summary per Video
        print("\n--------------------------------------------------------")
        print(f" SUMMARY FOR: {Path(str(source)).name}")
        print("--------------------------------------------------------")
        print(f"  Unique Visitors:       {summary.get('unique_visitors', 0)}")
        print(f"  Total Processed Frames:{summary.get('total_frames', 0)}")
        print(f"  Avg Visitors/Frame:    {summary.get('avg_visitors_per_frame', 0.0)}")
        print(f"  Avg Dwell Time (sec):  {summary.get('avg_dwell_time_seconds', 0.0):.2f}s")
        if mode == 'offline' and output_path and os.path.exists(output_path):
            print(f"  Annotated Video Saved: {output_path}")
        print("--------------------------------------------------------")

    print("\n[Inference] All video processing completed successfully.")


if __name__ == '__main__':
    main()
