#!/usr/bin/env python3
"""Training demo: train YOLO on COCO-8 mini dataset with MLflow tracking.

Uses the built-in coco8.yaml (8-image test dataset from Ultralytics) so no
external data download is needed. Demonstrates the full training lifecycle:
config loading → Trainer → MLflow logging → metrics export.

Usage:
    python examples/demo_training.py
    python examples/demo_training.py --epochs 5 --device cuda
    python examples/demo_training.py --disable-mlflow
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

from vision_ml.training.trainer import Trainer
from vision_ml.analytics.analytics_db import AnalyticsDB
from vision_ml.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Training Pipeline Demo")
    parser.add_argument('--config', type=str, default='config/training/base.yaml')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dataset', type=str, default='coco8.yaml',
                        help='Dataset YAML (default: coco8 built-in mini dataset)')
    parser.add_argument('--disable-mlflow', action='store_true',
                        help='Skip MLflow logging')
    args = parser.parse_args()

    # Step 1: Load and override config
    config = load_config(args.config)
    config['training']['epochs'] = args.epochs
    config['training']['device'] = args.device
    config['training']['batch_size'] = 4  # Small for demo
    config['data']['dataset_yaml'] = args.dataset

    if args.disable_mlflow:
        for key in ['log_params', 'log_metrics', 'log_model', 'register_model']:
            config.setdefault('mlflow', {})[key] = False

    print("=" * 55)
    print("  TRAINING DEMO")
    print("=" * 55)
    print(f"  Model:       {config['model']['name']}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {config['training']['batch_size']}")
    print(f"  Device:      {args.device}")
    print(f"  MLflow:      {'disabled' if args.disable_mlflow else 'enabled'}")
    print("=" * 55)
    print()

    # Step 2: Log training event to analytics DB
    db = AnalyticsDB()
    event_id = db.save_training_event({
        'trigger_type': 'manual',
        'dataset_size': 8,  # coco8 has 8 images
        'drift_score': 0.0,
        'model_version': config.get('mlflow', {}).get('model_name', 'demo'),
    })
    print(f"[Demo] Training event logged: {event_id}")

    # Step 3: Train
    trainer = Trainer(config)
    start = time.time()

    try:
        results = trainer.train(
            run_name='demo_training',
            dataset_yaml=args.dataset,
        )
        elapsed = time.time() - start

        # Step 4: Display results
        print(f"\n{'=' * 55}")
        print("  TRAINING RESULTS")
        print(f"{'=' * 55}")
        print(f"  Duration:     {elapsed:.1f}s")

        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"  mAP50:        {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP50-95:     {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"  Precision:    {metrics.get('metrics/precision(B)', 'N/A')}")
            print(f"  Recall:       {metrics.get('metrics/recall(B)', 'N/A')}")

        print(f"{'=' * 55}")

        # Check for saved weights
        best_path = os.path.join(
            config['training'].get('project', 'runs/train'),
            'demo_training', 'weights', 'best.pt'
        )
        if os.path.exists(best_path):
            size_mb = os.path.getsize(best_path) / (1024 * 1024)
            print(f"\n[Demo] Best weights: {best_path} ({size_mb:.1f} MB)")

        print(f"\n[Demo] Next steps:")
        print(f"  python scripts/evaluate.py                    # Evaluate model")
        print(f"  python scripts/mlflow_cli.py --action runs    # View MLflow runs")
        print(f"  python examples/demo_inference.py             # Run inference")

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[Demo] Training failed after {elapsed:.1f}s: {e}")
        print(f"[Demo] This is expected if YOLO cannot download coco8 (offline environment).")
        raise


if __name__ == '__main__':
    main()
