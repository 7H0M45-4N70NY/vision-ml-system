"""DVC Reproducibility Pipeline Demo.

Demonstrates the full DVC workflow for the vision-ml-system:

    Stage 1: prepare_data  → Collect auto-labeled frames, split train/val
    Stage 2: train         → YOLO fine-tuning with MLflow tracking
    Stage 3: evaluate      → Validate best.pt, emit eval_metrics.json

Run all three stages end-to-end:
    python examples/demo_dvc.py

Or reproduce selectively with DVC:
    dvc repro                   # runs only changed stages
    dvc repro train             # force only the train stage
    dvc repro --force           # force all stages

Compare metrics across git commits:
    dvc metrics show
    dvc metrics diff HEAD~1

Track parameter changes:
    dvc params diff HEAD~1

Visualise pipeline DAG:
    dvc dag

Push / pull artifacts with DagsHub remote:
    dvc push
    dvc pull
"""

import sys
import os
import subprocess
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use `python -m dvc` so the venv-installed DVC is always found,
# even when `dvc` is not on the system PATH.
DVC = [sys.executable, '-m', 'dvc']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, cwd: str = PROJECT_ROOT) -> int:
    """Run a subprocess command and return exit code."""
    logger.info("$ %s", ' '.join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def _section(title: str) -> None:
    width = 60
    logger.info("=" * width)
    logger.info("  %s", title)
    logger.info("=" * width)


def check_dvc_initialized() -> bool:
    dvc_dir = os.path.join(PROJECT_ROOT, '.dvc')
    if not os.path.isdir(dvc_dir):
        logger.error(".dvc/ not found. Run: dvc init")
        return False
    return True


def check_dvc_available() -> bool:
    result = subprocess.run([*DVC, 'version'], capture_output=True)
    if result.returncode != 0:
        logger.error("DVC not found. Run: pip install dvc")
        return False
    version = result.stdout.decode().strip().split('\n')[0]
    logger.info("DVC available: %s", version)
    return True


# ---------------------------------------------------------------------------
# Stage runners (can be called directly or via dvc repro)
# ---------------------------------------------------------------------------

def run_prepare_data(source: str = 'local') -> bool:
    """Stage 1: prepare training data from auto-labeled frames."""
    _section("Stage 1: prepare_data")
    rc = _run([
        sys.executable, 'scripts/prepare_data.py',
        '--config', 'config/training/base.yaml',
        '--source', source,
        '--output', 'data/prepared',
    ])
    if rc != 0:
        logger.warning("prepare_data exited with %d (no local samples is ok for demo)", rc)
    return True  # non-fatal if no samples — stage still writes output dir


def run_train(epochs: int = 3, disable_mlflow: bool = False) -> bool:
    """Stage 2: train YOLO with MLflow tracking."""
    _section("Stage 2: train")
    cmd = [
        sys.executable, 'scripts/train.py',
        '--config', 'config/training/base.yaml',
        '--trigger', 'manual',
        '--epochs', str(epochs),
    ]
    if disable_mlflow:
        cmd.append('--disable-mlflow')
    rc = _run(cmd)
    return rc == 0


def run_evaluate() -> bool:
    """Stage 3: evaluate best checkpoint, write eval_metrics.json."""
    _section("Stage 3: evaluate")
    rc = _run([
        sys.executable, 'scripts/evaluate.py',
        '--config', 'config/training/base.yaml',
        '--output', 'runs/train/eval_metrics.json',
    ])
    return rc == 0


# ---------------------------------------------------------------------------
# DVC commands demo
# ---------------------------------------------------------------------------

def show_pipeline_dag() -> None:
    _section("Pipeline DAG (dvc dag)")
    _run([*DVC, 'dag'])


def show_metrics() -> None:
    _section("Metrics (dvc metrics show)")
    # Check if any metrics file exists
    metrics_path = os.path.join(PROJECT_ROOT, 'metrics', 'eval_metrics.json')
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        logger.info("eval_metrics.json:")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info("  %-30s %.4f", k, v)
            else:
                logger.info("  %-30s %s", k, v)
    else:
        logger.info("No eval_metrics.json yet — run evaluate stage first")

    _run([*DVC, 'metrics', 'show'])


def show_params_diff() -> None:
    _section("Params diff (dvc params diff)")
    _run([*DVC, 'params', 'diff'])


def show_status() -> None:
    _section("Pipeline status (dvc status)")
    _run([*DVC, 'status'])


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='DVC reproducibility pipeline demo')
    parser.add_argument('--mode', choices=['run', 'repro', 'status', 'dag', 'metrics'],
                        default='run',
                        help=(
                            'run: execute stages directly via Python; '
                            'repro: use `dvc repro` (only re-runs changed stages); '
                            'status/dag/metrics: inspect without running'
                        ))
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs (only used in run/repro mode)')
    parser.add_argument('--disable-mlflow', action='store_true',
                        help='Skip MLflow logging (faster for demo)')
    args = parser.parse_args()

    start = time.time()

    if not check_dvc_initialized():
        sys.exit(1)

    if not check_dvc_available():
        sys.exit(1)

    if args.mode == 'status':
        show_status()

    elif args.mode == 'dag':
        show_pipeline_dag()

    elif args.mode == 'metrics':
        show_metrics()
        show_params_diff()

    elif args.mode == 'repro':
        # Let DVC decide which stages are stale and re-run only those
        _section("DVC Repro — smart re-run")
        logger.info("DVC checks each stage's inputs/outputs/params for changes.")
        logger.info("Only stale stages are re-executed.")
        rc = _run([*DVC, 'repro'])
        if rc != 0:
            logger.error("dvc repro failed with exit code %d", rc)
            sys.exit(rc)
        show_metrics()
        show_params_diff()

    else:  # mode == 'run'
        # Run all three stages directly (bypasses DVC caching — good for first-run demo)
        logger.info("Running full pipeline directly (bypasses DVC cache).")
        logger.info("After this, use --mode repro to benefit from DVC caching.")

        run_prepare_data()
        ok = run_train(epochs=args.epochs, disable_mlflow=args.disable_mlflow)
        if not ok:
            logger.error("Training failed. Skipping evaluate.")
            sys.exit(1)
        run_evaluate()
        show_metrics()

        logger.info("")
        logger.info("Pipeline complete in %.1fs", time.time() - start)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  dvc repro               # smart re-run (only changed stages)")
        logger.info("  dvc metrics diff        # compare metrics to last commit")
        logger.info("  dvc params diff         # compare hyperparams to last commit")
        logger.info("  dvc dag                 # visualise pipeline graph")
        logger.info("  dvc push                # push artifacts to DagsHub remote")


if __name__ == '__main__':
    main()
