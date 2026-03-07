from collections import deque


class DriftDetector:
    def __init__(self, config: dict):
        drift_cfg = config.get('drift', {})
        self.enabled = drift_cfg.get('enabled', False)
        self.method = drift_cfg.get('method', 'confidence_drop')
        self.threshold = drift_cfg.get('confidence_threshold', 0.3)
        self.window_size = drift_cfg.get('window_size', 500)
        self.check_interval = drift_cfg.get('check_interval', 100)

        self.confidence_buffer = deque(maxlen=self.window_size)
        self.inference_count = 0
        self.drift_detected = False

    def record(self, confidences: list):
        if not self.enabled:
            return
        self.confidence_buffer.extend(confidences)
        self.inference_count += 1

    def check(self) -> bool:
        if not self.enabled:
            return False

        if self.inference_count % self.check_interval != 0:
            return False

        if len(self.confidence_buffer) < self.check_interval:
            return False

        if self.method == 'confidence_drop':
            avg_conf = sum(self.confidence_buffer) / len(self.confidence_buffer)
            self.drift_detected = avg_conf < self.threshold
            if self.drift_detected:
                print(
                    f"[DriftDetector] DRIFT DETECTED: avg confidence {avg_conf:.3f} "
                    f"< threshold {self.threshold}"
                )
            return self.drift_detected

        return False

    def reset(self):
        self.confidence_buffer.clear()
        self.inference_count = 0
        self.drift_detected = False
