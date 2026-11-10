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
    parser.add_argument('--method', type=str, default='zip', choices=['zip', 'per-image'], help='Upload method: "zip" (in-code zip to signed URL, fastest) or "per-image" (concurrent individual images)')
    parser.add_argument('--batch-name', type=str, default=None, help='Custom batch name for Roboflow upload')
    parser.add_argument('--workers', type=int, default=5, help='Concurrent upload worker threads for per-image flow (default: 5)')
    parser.add_argument('--no-compress', dest='compress', action='store_false', help='Do not compress/downscale images before upload')
    parser.add_argument('--max-dim', type=int, default=1280, help='Max dimension for uploaded images (default: 1280)')
    parser.add_argument('--export-zip', type=str, default=None, help='Export frames and YOLO annotations to a single .zip file for Roboflow web drag-and-drop')
    parser.add_argument('--purge-after-upload', action='store_true', help='Automatically delete processed low-confidence frames after successful upload')
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

    # Option 1: Export to a single zip file for manual web upload
    if args.export_zip:
        zip_path = labeler.export_zip(zip_path=args.export_zip, compress=args.compress, max_dim=args.max_dim)
        print(f'[AutoLabeling] Ready! You can drag-and-drop "{zip_path}" directly into Roboflow in the browser.')
        return

    # Option 2: Upload in-code (in-code zip flow or parallel per-image) or export locally
    use_zip = (args.method == 'zip')
    labeler.flush(
        output_dir=args.output_dir,
        workers=args.workers,
        compress=args.compress,
        max_dim=args.max_dim,
        use_zip=use_zip,
        batch_name=args.batch_name,
    )
    print('[AutoLabeling] Flush complete.')

    if args.purge_after_upload:
        import glob
        purged = 0
        for f in glob.glob(os.path.join(args.frame_dir, '*')):
            try:
                os.remove(f)
                purged += 1
            except OSError:
                pass
        print(f'[AutoLabeling] Purged {purged} processed files from {args.frame_dir}.')


if __name__ == '__main__':
    main()
