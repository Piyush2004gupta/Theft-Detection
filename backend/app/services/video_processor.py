from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config import Settings
from app.schemas import AnalyticsResponse
from app.services.analytics_service import AnalyticsAggregator
from app.services.classifier_service import BehaviorClassifierService
from app.services.detector_service import DetectorService


from app.services.hand_detector_service import HandDetectorService


BBox = Tuple[int, int, int, int]


class VideoProcessor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.detector = DetectorService(settings=settings)
        self.classifier = BehaviorClassifierService(settings=settings)
        self.hand_detector = HandDetectorService()

    def process_video(self, video_path: Path, save_video: bool = True) -> AnalyticsResponse:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        aggregator = AnalyticsAggregator()
        writer: Optional[cv2.VideoWriter] = None
        output_path: Optional[Path] = None

        if save_video:
            output_path = self._create_output_video_path(video_path)
            writer, output_path = self._create_video_writer(output_path, fps, (width, height))

        frame_index = 0
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            frame_index += 1
            result = self.detector.detect_and_track(frame)
            hands = self.hand_detector.detect_hands(frame, person_tracks=result.person_tracks)
            


            # Theft Logic: Check interactions
            self._analyze_theft_triggers(frame, result, hands, aggregator)

            for track_id, person_box in result.person_tracks:
                aggregator.update_track(track_id, frame_index)

                if self._is_holding_cup(person_box, result.cup_boxes):
                    aggregator.increment_holding_cup(track_id)

                if frame_index % max(1, self.settings.classify_every_n_frames) == 0:
                    crop = self._safe_crop(frame, person_box)
                    if crop.size > 0:
                        label = self.classifier.classify_crop(crop)
                        aggregator.vote_activity(track_id, label)

                if writer is not None:
                    self._draw_person_annotation(frame, track_id, person_box, aggregator)

            if writer is not None:
                for cup_box in result.cup_boxes:
                    DetectorService.draw_box(frame, cup_box, "Cup", (0, 255, 255), thickness=1)
                for hand in hands:
                    if hand.landmarks:
                        self._draw_hand_landmarks(frame, hand.landmarks)
                    else:
                        DetectorService.draw_box(frame, hand.bbox, "Hand", (255, 0, 255), thickness=1)
                writer.write(frame)

    def _draw_hand_landmarks(self, frame: np.ndarray, landmarks: list[tuple[float, float, float]]) -> None:
        h, w = frame.shape[:2]
        # Connections for MediaPipe Hands
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (5, 9), (9, 10), (10, 11), (11, 12), # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
        ]
        
        # Color palette for "Luxius" feel: Sleek Neon Pink/Cyan
        joint_color = (0, 255, 255) # Cyan joints
        line_color = (180, 105, 255) # Pinkish-purple lines
        
        points = []
        for lm in landmarks:
            px, py = int(lm[0] * w), int(lm[1] * h)
            points.append((px, py))
            cv2.circle(frame, (px, py), 3, joint_color, -1)

        for connection in connections:
            p1, p2 = points[connection[0]], points[connection[1]]
            cv2.line(frame, p1, p2, line_color, 2)

        capture.release()
        if writer is not None:
            writer.release()
        self.hand_detector.close()

        response = aggregator.finalize(fps=fps)
        if output_path is not None:
            if output_path.exists():
                response.processed_video_path = output_path.name
        return response

    def _analyze_theft_triggers(self, frame, result, hands, aggregator):
        for track_id, person_box in result.person_tracks:
            # 1. Hand + Object Interaction
            for hand in hands:
                # If hand belongs to this person track
                if self._box_overlap(hand.bbox, person_box) > 0.4:
                    for cup_box in result.cup_boxes:
                        if self._box_overlap(hand.bbox, cup_box) > self.settings.hand_object_overlap_threshold:
                            aggregator.mark_hand_interaction(track_id, cup_box)
            
            # 2. Object Disappeared
            state = aggregator.tracks.get(track_id)
            if state and state.hand_near_object and state.last_known_object_pos and not state.object_disappeared:
                obj_still_there = False
                for cup_box in result.cup_boxes:
                    if self._dist(self._center(cup_box), self._center(state.last_known_object_pos)) < self.settings.object_disappeared_threshold:
                        obj_still_there = True
                        break
                if not obj_still_there:
                    aggregator.mark_object_disappeared(track_id)
            
            # 3. Person Moves Away
            if state and state.object_disappeared and state.last_known_object_pos:
                dist_from_spot = self._dist(self._center(person_box), self._center(state.last_known_object_pos))
                if dist_from_spot > self.settings.move_away_threshold:
                    aggregator.mark_moved_away(track_id)

    @staticmethod
    def _dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _box_overlap(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1: return 0.0
        inter = (float(x2) - x1) * (y2 - y1)
        area1 = (float(box1[2]) - box1[0]) * (box1[3] - box1[1])
        return inter / area1 if area1 > 0 else 0

    def _draw_person_annotation(
        self,
        frame: np.ndarray,
        track_id: int,
        person_box: BBox,
        aggregator: AnalyticsAggregator,
    ) -> None:
        activity = aggregator.get_person_activity(track_id)
        is_theft = activity == "Theft"
        color = (0, 0, 255) if is_theft else (0, 255, 0)
        
        state = aggregator.tracks.get(track_id)
        status_text = "Normal"
        if state:
            if is_theft:
                status_text = "THEFT!!"
            elif state.object_disappeared:
                status_text = "Object Missing"
            elif state.hand_near_object:
                status_text = "Interacting"

        label = f"ID {track_id} [{status_text}]"
        DetectorService.draw_box(frame, person_box, label, color)

    @staticmethod
    def _safe_crop(frame: np.ndarray, box: BBox) -> np.ndarray:
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(1, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _is_holding_cup(person_box: BBox, cup_boxes: list[BBox], distance_threshold: float = 120.0) -> bool:
        px1, py1, px2, py2 = person_box
        person_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)

        for cup in cup_boxes:
            cx1, cy1, cx2, cy2 = cup
            cup_center = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
            distance = ((person_center[0] - cup_center[0]) ** 2 + (person_center[1] - cup_center[1]) ** 2) ** 0.5
            if distance <= distance_threshold:
                return True
        return False

    def _create_output_video_path(self, input_path: Path) -> Path:
        outputs_dir = self.settings.resolved_outputs_dir
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{timestamp}_{input_path.stem}.mp4"
        return outputs_dir / filename

    @staticmethod
    def _create_video_writer(
        preferred_output_path: Path,
        fps: float,
        frame_size: Tuple[int, int],
    ) -> Tuple[Optional[cv2.VideoWriter], Path]:
        candidates = [
            (preferred_output_path.with_suffix(".mp4"), "avc1"),
            (preferred_output_path.with_suffix(".mp4"), "H264"),
            (preferred_output_path.with_suffix(".webm"), "VP80"),
            (preferred_output_path.with_suffix(".avi"), "MJPG"),
            (preferred_output_path.with_suffix(".mp4"), "mp4v"),
        ]

        for output_path, codec in candidates:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                frame_size,
            )
            if writer.isOpened():
                return writer, output_path
            writer.release()

        return None, preferred_output_path
