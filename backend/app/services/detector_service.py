from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import Settings


BBox = Tuple[int, int, int, int]


@dataclass
class DetectionFrameResult:
    person_tracks: List[tuple[int, BBox]]
    cup_boxes: List[BBox]


class DetectorService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = YOLO(settings.yolo_model)

    def detect_and_track(self, frame: np.ndarray) -> DetectionFrameResult:
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.settings.yolo_conf,
            tracker=self.settings.tracker_cfg,
            verbose=False,
        )

        if not results:
            return DetectionFrameResult(person_tracks=[], cup_boxes=[])

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return DetectionFrameResult(person_tracks=[], cup_boxes=[])

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.array([], dtype=int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(xyxy), -1, dtype=int)

        person_tracks: List[tuple[int, BBox]] = []
        cup_boxes: List[BBox] = []

        for idx, box in enumerate(xyxy):
            class_id = int(cls[idx]) if idx < len(cls) else -1
            x1, y1, x2, y2 = map(int, box.tolist())
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            current_box: BBox = (x1, y1, x2, y2)

            if class_id == self.settings.person_class_id and idx < len(track_ids) and track_ids[idx] >= 0:
                person_tracks.append((int(track_ids[idx]), current_box))
            elif class_id == self.settings.cup_class_id:
                cup_boxes.append(current_box)

        return DetectionFrameResult(person_tracks=person_tracks, cup_boxes=cup_boxes)

    @staticmethod
    def draw_box(
        frame: np.ndarray,
        box: BBox,
        label: str,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
