"""Prepare training data from auto-labeled and low-confidence frames.

This script is a DVC pipeline stage that:
1. Collects labeled data from auto_labeled/ and low_confidence_frames/
2. Optionally downloads a Roboflow dataset version
3. Splits into train/val sets
4. Writes a YOLO-format dataset.yaml

Usage:
    python scripts/prepare_data.py --config config/training/base.yaml
    python scripts/prepare_data.py --source roboflow --roboflow-version 1
    python scripts/prepare_data.py --source local
"""

import sys
import os
import json
import glob
import shutil
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.utils.config import load_config
from vision_ml.logging import get_logger

logger = get_logger(__name__)


def collect_local_labels(auto_dir: str, low_conf_dir: str):
    """Gather (image_path, label_dict) pairs from local directories."""
    samples = []

    # From auto-labeled (warm-path collected frames)
    label_file = os.path.join(auto_dir, 'auto_labels.json')
    if os.path.isfile(label_file):
        with open(label_file) as f:
            labels = json.load(f)
        for lbl in labels:
            img = lbl.get('image_path')
            if img and os.path.isfile(img):
                samples.append((img, lbl))

    # From auto-labeled images dir (warm-path images without JSON envelope)
    img_dir = os.path.join(auto_dir, 'images')
    if os.path.isdir(img_dir):
        for img_path in sorted(glob.glob(os.path.join(img_dir, '*.jpg'))):
            frame_id = os.path.splitext(os.path.basename(img_path))[0]
            # Check if already captured above
            if any(s[0] == img_path for s in samples):
                continue
            samples.append((img_path, {
                'image_id': frame_id, 'image_path': img_path,
                'boxes': [], 'confidences': [], 'class_ids': [],
            }))

    # From low-confidence frames (cold-path DualDetector output)
    if os.path.isdir(low_conf_dir):
        for jf in sorted(glob.glob(os.path.join(low_conf_dir, '*.json'))):
            img = jf.replace('.json', '.jpg')
            if not os.path.isfile(img):
                continue
            with open(jf) as f:
                lbl = json.load(f)
            lbl['image_path'] = img
            lbl['image_id'] = os.path.splitext(os.path.basename(jf))[0]
            samples.append((img, lbl))

    return samples


def download_roboflow_dataset(config: dict, version: int, dest: str):
    """Download a specific Roboflow dataset version in YOLOv8 format."""
    label_cfg = config.get('labeling', {})
    api_key = label_cfg.get('roboflow_api_key') or os.environ.get('ROBOFLOW_API_KEY')
    workspace = label_cfg.get('roboflow_workspace')
    project = label_cfg.get('roboflow_project')

    if not api_key or not workspace or not project:
        logger.warning("Roboflow credentials not configured. Skipping download.")
        return None

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=dest)
        logger.info("Downloaded Roboflow dataset v%d to %s", version, dest)
        return dataset.location
    except ImportError:
        logger.warning("roboflow package not installed. pip install roboflow")
        return None
    except Exception as e:
        logger.error("Roboflow download error: %s", e)
        return None


def write_yolo_annotation(label: dict, output_path: str, img_w: int, img_h: int):
    """Convert label dict (xyxy boxes) to YOLO-format .txt file."""
    boxes = label.get('boxes', [])
    class_ids = label.get('class_ids', [])
    with open(output_path, 'w') as f:
        for i, box in enumerate(boxes):
            cls_id = class_ids[i] if i < len(class_ids) else 0
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def build_dataset(samples, output_dir: str, val_split: float = 0.2, num_classes: int = 1):
    """Split samples into train/val, copy images, write annotations, emit dataset.yaml."""
    train_img = os.path.join(output_dir, 'train', 'images')
    train_lbl = os.path.join(output_dir, 'train', 'labels')
    val_img = os.path.join(output_dir, 'val', 'images')
    val_lbl = os.path.join(output_dir, 'val', 'labels')
    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    random.shuffle(samples)
    split_idx = max(1, int(len(samples) * (1 - val_split)))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    import cv2

    for split_name, split_data, img_dir, lbl_dir in [
        ('train', train_samples, train_img, train_lbl),
        ('val', val_samples, val_img, val_lbl),
    ]:
        for img_path, label in split_data:
            fname = os.path.basename(img_path)
            dest_img = os.path.join(img_dir, fname)
            shutil.copy2(img_path, dest_img)

            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
            else:
                h, w = 640, 640

            txt_name = os.path.splitext(fname)[0] + '.txt'
            write_yolo_annotation(label, os.path.join(lbl_dir, txt_name), w, h)

    # Write dataset.yaml
    names = {0: 'person'}  # extend if num_classes > 1
    dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
    import yaml
    with open(dataset_yaml, 'w') as f:
        yaml.dump({
            'path': os.path.abspath(output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': num_classes,
            'names': names,
        }, f, default_flow_style=False)

    logger.info("Dataset ready: %d train, %d val", len(train_samples), len(val_samples))
    logger.info("dataset.yaml -> %s", dataset_yaml)
    return dataset_yaml


def main():
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--config', default='config/training/base.yaml')
    parser.add_argument('--source', default='local', choices=['local', 'roboflow', 'both'],
                        help="Data source: 'local' (auto-labeled+low-conf), 'roboflow' (download), 'both'")
    parser.add_argument('--roboflow-version', type=int, default=1,
                        help='Roboflow dataset version to download')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--output', default='data/prepared')
    args = parser.parse_args()

    config = load_config(args.config)
    num_classes = config.get('model', {}).get('num_classes', 1)

    samples = []

    if args.source in ('local', 'both'):
        auto_dir = 'data/auto_labeled'
        low_conf_dir = 'data/low_confidence_frames'
        local_samples = collect_local_labels(auto_dir, low_conf_dir)
        logger.info("Collected %d local samples", len(local_samples))
        samples.extend(local_samples)

    if args.source in ('roboflow', 'both'):
        rf_dest = os.path.join(args.output, '_roboflow_download')
        rf_loc = download_roboflow_dataset(config, args.roboflow_version, rf_dest)
        if rf_loc:
            logger.info("Roboflow dataset at %s", rf_loc)
            # Roboflow downloads in YOLO format already — just point dataset.yaml there
            if args.source == 'roboflow':
                dataset_yaml = os.path.join(rf_loc, 'data.yaml')
                if os.path.isfile(dataset_yaml):
                    shutil.copy2(dataset_yaml, os.path.join(args.output, 'dataset.yaml'))
                    logger.info("Using Roboflow dataset directly")
                    return

    if not samples:
        logger.warning("No samples found. Skipping dataset build.")
        logger.warning("Run inference first to collect auto-labels, or download from Roboflow.")
        return

    build_dataset(samples, args.output, args.val_split, num_classes)


if __name__ == '__main__':
    main()
