from dataclasses import dataclass
from typing import Iterable

from ultralytics import YOLO


COCO_CAR_CLASS_ID = 2


@dataclass(frozen=True)
class Detection:
    frame_number: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class CarDetector:
    def __init__(self, weights_path: str, confidence_threshold: float) -> None:
        self._model = YOLO(weights_path)
        self._confidence_threshold = confidence_threshold

    def detect(self, frame, frame_number: int) -> list[Detection]:
        results = self._model.predict(
            source=frame,
            classes=[COCO_CAR_CLASS_ID],
            conf=self._confidence_threshold,
            verbose=False,
        )
        return list(self._extract_detections(results, frame_number))

    def _extract_detections(self, results: Iterable, frame_number: int) -> Iterable[Detection]:
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                yield Detection(
                    frame_number=frame_number,
                    class_name=self._model.names[class_id],
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
