from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class Detector:
    def __init__(self, weights: str, confidence: float = 0.4):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the model extra: uv sync --extra yolo") from exc
        self.model = YOLO(weights)
        self._lock = Lock()
        self.confidence = confidence
        self.labels = set(self.model.names.values())

    def detect(self, frame) -> list[Detection]:
        with self._lock:
            result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        return [Detection(str(result.names[int(box.cls.item())]), float(box.conf.item()),
                          tuple(int(v) for v in box.xyxy[0].tolist())) for box in result.boxes]
