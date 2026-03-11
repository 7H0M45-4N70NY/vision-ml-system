"""Evaluate a trained model and output metrics as JSON.

DVC pipeline stage that:
1. Loads the best trained model from runs/train/
2. Runs validation on the prepared dataset
3. Writes eval_metrics.json for DVC metric tracking

Usage:
    python scripts/evaluate.py --config config/training/base.yaml
    python scripts/evaluate.py --weights runs/train/exp/weights/best.pt
"""

import sys
import os
import json
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.utils.config import load_config


def find_best_weights(project_dir: str) -> str:
    """Find the most recent best.pt in the training output directory."""
    candidates = sorted(
        glob.glob(os.path.join(project_dir, '**', 'weights', 'best.pt'), recursive=True),
        key=os.path.getmtime, reverse=True,
    )
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--config', default='config/training/base.yaml')
    parser.add_argument('--weights', default=None, help='Path to model weights (auto-detects if not set)')
    parser.add_argument('--data', default=None, help='Path to dataset.yaml (auto-detects if not set)')
    parser.add_argument('--output', default='runs/train/eval_metrics.json')
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get('training', {})

    # Find weights
    weights = args.weights
    if not weights:
        project = train_cfg.get('project', 'runs/train')
        weights = find_best_weights(project)
    if not weights or not os.path.isfile(weights):
        print("[Evaluate] No trained weights found. Run training first.")
        # Write empty metrics so DVC pipeline doesn't fail
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'status': 'no_weights', 'mAP50': 0, 'mAP50-95': 0}, f, indent=2)
        return

    # Find dataset
    data = args.data
    if not data:
        data = config.get('data', {}).get('dataset_yaml') or 'data/prepared/dataset.yaml'
    if not os.path.isfile(data):
        print(f"[Evaluate] Dataset not found at {data}. Using coco8.yaml fallback.")
        data = 'coco8.yaml'

    print(f"[Evaluate] Weights: {weights}")
    print(f"[Evaluate] Dataset: {data}")

    from ultralytics import YOLO
    model = YOLO(weights)

    results = model.val(
        data=data,
        imgsz=train_cfg.get('imgsz', 640),
        device=train_cfg.get('device', 'cpu'),
        batch=train_cfg.get('batch_size', 16),
    )

    # Extract metrics
    metrics = {}
    if hasattr(results, 'results_dict'):
        for k, v in results.results_dict.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                metrics[k] = str(v)

    # Add summary fields
    if hasattr(results, 'box'):
        metrics['mAP50'] = float(results.box.map50) if hasattr(results.box, 'map50') else 0
        metrics['mAP50-95'] = float(results.box.map) if hasattr(results.box, 'map') else 0

    metrics['weights'] = weights
    metrics['dataset'] = data
    metrics['status'] = 'completed'

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[Evaluate] Metrics saved to {args.output}")
    print(f"[Evaluate] mAP50={metrics.get('mAP50', 'N/A'):.4f}  "
          f"mAP50-95={metrics.get('mAP50-95', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
