from typing import Dict , Any

class ModelRegistry:
    __models : Dict[str,Any] = {}

    @classmethod
    def get_model(cls, model_key: str, loader_func) -> Any:
        if model_key not in cls.__models:
            cls.__models[model_key] = loader_func(model_key)
        return cls.__models[model_key]
    
    @classmethod
    def clear_models(cls) -> None:
        cls.__models.clear()

