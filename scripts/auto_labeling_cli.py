import sys
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.utils.config import load_config
from vision_ml.labeling.auto_labeler import AutoLabeler


def main():
    parser = argparse.ArgumentParser(description='Auto-labeling CLI')
    parser.add_argument('--config', type=str, default='config/training/base.yaml')
    parser.add_argument('--frame-dir', type=str, default='data/low_confidence_frames', help='Directory with saved dual-detector JSON frames')
    parser.add_argument('--output-dir', type=str, default='data/auto_labeled', help='Local output dir for pseudo labels')
    parser.add_argument('--provider', type=str, default=None, choices=['local', 'roboflow'], help='Override labeling provider from config')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.provider:
        config.setdefault('labeling', {})['provider'] = args.provider
        config.setdefault('labeling', {})['enabled'] = True

    labeler = AutoLabeler(config)
    loaded = labeler.load_dual_detector_frames(args.frame_dir)
    print(f'[AutoLabeling] Loaded pseudo-label files: {loaded}')

    if loaded == 0:
        print('[AutoLabeling] Nothing to export/upload.')
        return

    labeler.flush(output_dir=args.output_dir)
    print('[AutoLabeling] Flush complete.')


if __name__ == '__main__':
    main()
