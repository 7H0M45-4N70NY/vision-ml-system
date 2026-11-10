"""Tests for AutoLabeler."""
import os
import json
import pytest

from src.vision_ml.labeling.auto_labeler import AutoLabeler


class TestAutoLabeler:
    def test_collect_is_noop(self, sample_config):
        al = AutoLabeler(sample_config)
        al.collect(None, None, image_id="test")
        assert len(al.pending_labels) == 0

    def test_load_no_dir(self, sample_config, tmp_path):
        al = AutoLabeler(sample_config)
        count = al.load_dual_detector_frames(str(tmp_path / "nonexistent"))
        assert count == 0

    def test_load_dual_detector_frames(self, sample_config, tmp_path):
        # Create fake label files
        for i in range(3):
            label = {'boxes': [[10, 10, 50, 50]], 'class_ids': [0], 'confidences': [0.9]}
            with open(tmp_path / f"frame_{i:06d}.json", 'w') as f:
                json.dump(label, f)

        al = AutoLabeler(sample_config)
        count = al.load_dual_detector_frames(str(tmp_path))
        assert count == 3
        assert len(al.pending_labels) == 3

    def test_export_local(self, sample_config, tmp_path):
        al = AutoLabeler(sample_config)
        al.pending_labels = [{'boxes': [[10, 10, 50, 50]], 'class_ids': [0]}]
        out_dir = str(tmp_path / "output")
        al.flush(out_dir)
        assert os.path.exists(os.path.join(out_dir, 'auto_labels.json'))
        assert len(al.pending_labels) == 0

    def test_flush_empty(self, sample_config, tmp_path):
        al = AutoLabeler(sample_config)
        out_dir = str(tmp_path / "output")
        al.flush(out_dir)
        # Should not create file when empty
        assert not os.path.exists(os.path.join(out_dir, 'auto_labels.json'))

    def test_export_zip(self, sample_config, tmp_path):
        al = AutoLabeler(sample_config)
        dummy_img = tmp_path / "frame_000001.jpg"
        dummy_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100) # dummy jpg
        al.pending_labels = [{
            'image_id': 'frame_000001',
            'image_path': str(dummy_img),
            'boxes': [[10, 10, 50, 50]],
            'class_ids': [0],
        }]
        zip_path = str(tmp_path / "test_batch.zip")
        res = al.export_zip(zip_path=zip_path, compress=False)
        assert res == zip_path
        assert os.path.exists(zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            namelist = zf.namelist()
            assert 'frame_000001.jpg' in namelist
            assert 'frame_000001.txt' in namelist
