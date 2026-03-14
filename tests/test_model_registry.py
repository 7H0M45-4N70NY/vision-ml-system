"""Tests for ModelRegistry singleton cache."""
from src.vision_ml.detection.model_registry import ModelRegistry


class TestModelRegistry:
    def test_caches_model(self):
        loader = lambda key: f"model_{key}"
        m1 = ModelRegistry.get_model("test", loader)
        m2 = ModelRegistry.get_model("test", loader)
        assert m1 is m2

    def test_different_keys_different_models(self):
        loader = lambda key: f"model_{key}"
        m1 = ModelRegistry.get_model("a", loader)
        m2 = ModelRegistry.get_model("b", loader)
        assert m1 != m2

    def test_clear_models(self):
        loader = lambda key: object()
        m1 = ModelRegistry.get_model("x", loader)
        ModelRegistry.clear_models()
        m2 = ModelRegistry.get_model("x", loader)
        assert m1 is not m2

    def test_loader_called_once(self):
        call_count = 0

        def loader(key):
            nonlocal call_count
            call_count += 1
            return "model"

        ModelRegistry.get_model("once", loader)
        ModelRegistry.get_model("once", loader)
        assert call_count == 1
