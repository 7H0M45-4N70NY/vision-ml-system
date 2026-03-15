"""DVC CLI helper for Vision ML System.

Wraps common DVC commands with project-aware defaults and structured logging.

Usage:
    python scripts/dvc_cli.py --action repro
    python scripts/dvc_cli.py --action metrics
    python scripts/dvc_cli.py --action params
    python scripts/dvc_cli.py --action status
    python scripts/dvc_cli.py --action pull
    python scripts/dvc_cli.py --action push
    python scripts/dvc_cli.py --action add --targets runs/train
"""

import sys
import os
import subprocess
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.logging import get_logger

logger = get_logger(__name__)

ACTIONS = ['repro', 'status', 'pull', 'push', 'add', 'metrics', 'params', 'dag']

# Use python -m dvc so venv-installed DVC is found without PATH setup
DVC = [sys.executable, '-m', 'dvc']


def _run(cmd: list) -> int:
    logger.info("Running: %s", ' '.join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("Command failed with exit code %d", result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='DVC CLI helper for Vision ML System')
    parser.add_argument('--action', required=True, choices=ACTIONS,
                        help='DVC action to perform')
    parser.add_argument('--targets', type=str, default='runs/train',
                        help='Comma-separated paths for the add action')
    parser.add_argument('--stage', type=str, default=None,
                        help='Specific stage to reproduce (default: full pipeline)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run all stages (dvc repro --force)')
    args = parser.parse_args()

    if args.action == 'repro':
        cmd = [*DVC,'repro']
        if args.force:
            cmd.append('--force')
        if args.stage:
            cmd.append(args.stage)
        sys.exit(_run(cmd))

    elif args.action == 'status':
        sys.exit(_run([*DVC,'status']))

    elif args.action == 'pull':
        sys.exit(_run([*DVC,'pull']))

    elif args.action == 'push':
        sys.exit(_run([*DVC,'push']))

    elif args.action == 'add':
        targets = [t.strip() for t in args.targets.split(',') if t.strip()]
        for target in targets:
            rc = _run([*DVC,'add', target])
            if rc != 0:
                sys.exit(rc)

    elif args.action == 'metrics':
        # Show current metrics and diff from last commit
        rc = _run([*DVC,'metrics', 'show'])
        if rc == 0:
            _run([*DVC,'metrics', 'diff'])
        sys.exit(rc)

    elif args.action == 'params':
        # Show current params and diff from last commit
        rc = _run([*DVC,'params', 'diff'])
        sys.exit(rc)

    elif args.action == 'dag':
        sys.exit(_run([*DVC,'dag']))


if __name__ == '__main__':
    main()
