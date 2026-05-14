import logging
from pathlib import Path

import cv2

from detector.yolo_detector import CarDetector, Detection


logger = logging.getLogger(__name__)


class VideoProcessingError(RuntimeError):
    """Raised when video frames cannot be read for detection."""


class VideoProcessor:
    def __init__(self, detector: CarDetector, frame_stride: int = 1) -> None:
        self._detector = detector
        self._frame_stride = frame_stride

    def process(self, video_path: Path) -> None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoProcessingError(f"Unable to open video file: {video_path}")

        try:
            frame_number = 0
            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    break

                frame_number += 1
                if frame_number % self._frame_stride != 0:
                    continue

                detections = self._detector.detect(frame, frame_number)
                self._print_detections(frame_number, detections)
        finally:
            capture.release()

    @staticmethod
    def _print_detections(frame_number: int, detections: list[Detection]) -> None:
        if not detections:
            print(f"frame={frame_number} cars=0")
            return

        for detection in detections:
            print(
                "frame={frame} class={class_name} confidence={confidence:.3f} "
                "bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})".format(
                    frame=detection.frame_number,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                )
            )
