from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import Settings

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None


BBox = Tuple[int, int, int, int]


@dataclass
class DetectionFrameResult:
    person_tracks: List[tuple[int, BBox]]
    cup_boxes: List[BBox]


class DetectorService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = YOLO(settings.yolo_model)
        if DeepSort:
            self.tracker = DeepSort(
                max_age=settings.max_age,
                n_init=settings.n_init,
                max_cosine_distance=settings.max_cosine_distance,
            )
        else:
            self.tracker = None
            print("Warning: deep-sort-realtime not installed. Tracking will be limited.")

    def detect_and_track(self, frame: np.ndarray) -> DetectionFrameResult:
        # Use predict instead of track for manual DeepSORT integration
        results = self.model.predict(
            source=frame,
            conf=self.settings.yolo_conf,
            verbose=False,
        )

        if not results:
            return DetectionFrameResult(person_tracks=[], cup_boxes=[])

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return DetectionFrameResult(person_tracks=[], cup_boxes=[])

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.array([], dtype=int)
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

        person_detections = []
        cup_boxes: List[BBox] = []

        for idx, box in enumerate(xyxy):
            class_id = int(cls[idx])
            x1, y1, x2, y2 = map(int, box.tolist())
            current_box: BBox = (x1, y1, x2, y2)

            if class_id == self.settings.person_class_id:
                # DeepSORT expects [left, top, w, h]
                person_detections.append(([x1, y1, x2 - x1, y2 - y1], conf[idx], class_id))
            elif class_id == self.settings.cup_class_id:
                cup_boxes.append(current_box)

        person_tracks: List[tuple[int, BBox]] = []
        if self.tracker:
            tracks = self.tracker.update_tracks(person_detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                # Track IDs are strings in deep-sort-realtime, try to make it int if it's numeric
                try:
                    track_id_int = int(track_id)
                except ValueError:
                    # hash it if not numeric
                    track_id_int = hash(track_id) % 10000
                
                ltrb = track.to_ltrb() # x1, y1, x2, y2
                person_tracks.append((track_id_int, tuple(map(int, ltrb))))
        else:
            # Fallback: just return detections as untracked (id -1)
            for det in person_detections:
                bbox_xywh, _, _ = det
                x, y, w, h = bbox_xywh
                person_tracks.append((-1, (int(x), int(y), int(x + w), int(y + h))))

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
