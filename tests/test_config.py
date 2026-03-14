"""Tests for config loading, saving, merging, and validation."""
import os
import pytest
from src.vision_ml.utils.config import load_config, save_config, merge_configs, validate_config


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("model:\n  name: yolo11n\n")
        config = load_config(str(cfg_file))
        assert config['model']['name'] == 'yolo11n'

    def test_load_empty_yaml(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        assert load_config(str(cfg_file)) == {}

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestSaveConfig:
    def test_save_and_reload(self, tmp_path):
        config = {'model': {'name': 'yolo11n'}, 'epochs': 10}
        out = str(tmp_path / "out.yaml")
        save_config(config, out)
        reloaded = load_config(out)
        assert reloaded == config

    def test_save_creates_parent_dirs(self, tmp_path):
        out = str(tmp_path / "sub" / "dir" / "cfg.yaml")
        save_config({'a': 1}, out)
        assert os.path.exists(out)


class TestMergeConfigs:
    def test_shallow_override(self):
        base = {'a': 1, 'b': 2}
        override = {'b': 99}
        merged = merge_configs(base, override)
        assert merged == {'a': 1, 'b': 99}

    def test_deep_merge(self):
        base = {'model': {'name': 'yolo11n', 'precision': 'fp32'}}
        override = {'model': {'precision': 'fp16'}}
        merged = merge_configs(base, override)
        assert merged['model']['name'] == 'yolo11n'
        assert merged['model']['precision'] == 'fp16'

    def test_does_not_mutate_base(self):
        base = {'a': {'b': 1}}
        merge_configs(base, {'a': {'b': 2}})
        assert base['a']['b'] == 1


class TestValidateConfig:
    def test_valid_config(self):
        config = {'model': {}, 'inference': {}}
        assert validate_config(config) == []

    def test_missing_keys(self):
        config = {'model': {}}
        missing = validate_config(config)
        assert 'inference' in missing

    def test_custom_required_keys(self):
        config = {'a': 1}
        missing = validate_config(config, required_keys=['a', 'b', 'c'])
        assert missing == ['b', 'c']
