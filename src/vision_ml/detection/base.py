from abc import ABC, abstractmethod
import numpy as np
import supervision as sv


class BaseDetector(ABC):
    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> sv.Detections:
        pass
