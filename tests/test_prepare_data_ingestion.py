import os

import cv2
import numpy as np

from scripts.prepare_data import _dedupe_samples, _validate_samples, collect_roboflow_samples


def test_dedupe_samples_by_filename():
    samples = [
        ("/tmp/a/frame_001.jpg", {"boxes": [], "class_ids": []}),
        ("/tmp/b/frame_001.jpg", {"boxes": [], "class_ids": []}),
        ("/tmp/b/frame_002.jpg", {"boxes": [], "class_ids": []}),
    ]

    deduped = _dedupe_samples(samples)
    assert len(deduped) == 2
    assert os.path.basename(deduped[0][0]) == "frame_001.jpg"
    assert os.path.basename(deduped[1][0]) == "frame_002.jpg"


def test_validate_samples_filters_missing_and_fills_class_ids(tmp_path):
    img_path = str(tmp_path / "img.jpg")
    cv2.imwrite(img_path, np.zeros((16, 16, 3), dtype=np.uint8))

    samples = [
        (img_path, {"boxes": [[1, 1, 8, 8]], "class_ids": []}),
        (str(tmp_path / "missing.jpg"), {"boxes": [], "class_ids": []}),
    ]

    validated = _validate_samples(samples)
    assert len(validated) == 1
    assert validated[0][1]["class_ids"] == [0]


def test_collect_roboflow_samples_reads_yolo_labels(tmp_path):
    dataset_root = tmp_path / "rf_ds"
    image_dir = dataset_root / "train" / "images"
    label_dir = dataset_root / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image_path = image_dir / "frame_001.jpg"
    label_path = label_dir / "frame_001.txt"

    cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
    label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    samples = collect_roboflow_samples(str(dataset_root))
    assert len(samples) == 1
    _, label = samples[0]
    assert label["class_ids"] == [0]
    assert len(label["boxes"]) == 1
