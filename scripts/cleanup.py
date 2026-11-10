"""Clean up and purge pipeline artifacts, temporary buffers, and runs.

Usage:
    python scripts/cleanup.py --target all
    python scripts/cleanup.py --target buffer
    python scripts/cleanup.py --target runs
    python scripts/cleanup.py --target prepared
    python scripts/cleanup.py --target db
    python scripts/cleanup.py --target all --archive
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'
RUNS_DIR = REPO_ROOT / 'runs'
MODELS_DIR = REPO_ROOT / 'models'

TARGET_PATHS = {
    'buffer': [
        DATA_DIR / 'low_confidence_frames',
        DATA_DIR / 'auto_labeled',
        DATA_DIR / 'secondary_analysis',
    ],
    'runs': [
        RUNS_DIR / 'inference',
        RUNS_DIR / 'detect',
        RUNS_DIR / 'train',
        RUNS_DIR / 'test',
        RUNS_DIR,
    ],
    'prepared': [
        DATA_DIR / 'prepared',
    ],
    'models': [
        MODELS_DIR / 'fine_tuned',
    ],
    'db': [
        DATA_DIR / 'analytics.db',
    ],
}


def _purge_or_archive_path(path: Path, archive_dir: Path = None):
    if not path.exists():
        return 0

    cleaned_count = 0

    if path.is_file():
        if archive_dir:
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(archive_dir / path.name))
        else:
            path.unlink(missing_ok=True)
        return 1

    if path.is_dir():
        for item in path.iterdir():
            # Never delete sample_videos or original base models
            if item.name in ('sample_videos', 'original'):
                continue

            cleaned_count += 1
            if archive_dir:
                dest = archive_dir / item.name
                archive_dir.mkdir(parents=True, exist_ok=True)
                if item.is_dir():
                    shutil.move(str(item), str(dest))
                else:
                    shutil.move(str(item), str(dest))
            else:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)

    return cleaned_count


def main():
    parser = argparse.ArgumentParser(description='Purge and reset vision pipeline artifacts')
    parser.add_argument(
        '--target',
        choices=['all', 'buffer', 'runs', 'prepared', 'models', 'db'],
        default='all',
        help='Which stage artifacts to clean (default: all)',
    )
    parser.add_argument(
        '--archive',
        action='store_true',
        help='Move items to data/archive/<timestamp>/ instead of permanently deleting them',
    )
    args = parser.parse_args()

    archive_base = None
    if args.archive:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_base = DATA_DIR / 'archive' / ts
        archive_base.mkdir(parents=True, exist_ok=True)
        print(f"[Cleanup] Archiving items to {archive_base}...")

    targets_to_clean = []
    if args.target == 'all':
        targets_to_clean = ['buffer', 'runs', 'prepared', 'models', 'db']
    else:
        targets_to_clean = [args.target]

    total_cleaned = 0
    for target in targets_to_clean:
        paths = TARGET_PATHS.get(target, [])
        for p in paths:
            sub_archive = archive_base / target if archive_base else None
            count = _purge_or_archive_path(p, archive_dir=sub_archive)
            total_cleaned += count

    # Ensure required empty buffer and run directories exist for future runs
    (DATA_DIR / 'low_confidence_frames').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'secondary_analysis').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'prepared').mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / 'inference').mkdir(parents=True, exist_ok=True)

    action_word = "Archived" if args.archive else "Purged"
    print(f"[Cleanup] {action_word} {total_cleaned} items across target: {args.target}")
    print("[Cleanup] Ready for fresh pipeline run (sample_videos and original models preserved).")


if __name__ == '__main__':
    main()
