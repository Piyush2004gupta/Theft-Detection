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


BBox = Tuple[int, int, int, int]


class VideoProcessor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.detector = DetectorService(settings=settings)
        self.classifier = BehaviorClassifierService(settings=settings)

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
            aggregator.update_cup_count(len(result.cup_boxes))

            for track_id, person_box in result.person_tracks:
                aggregator.update_track(track_id, frame_index)

                if self._is_holding_cup(person_box, result.cup_boxes):
                    aggregator.increment_holding_cup(track_id)

                if frame_index % max(1, self.settings.classify_every_n_frames) == 0:
                    crop = self._safe_crop(frame, person_box)
                    label = self.classifier.classify_crop(crop)
                    aggregator.vote_activity(track_id, label)

                if writer is not None:
                    self._draw_person_annotation(frame, track_id, person_box, aggregator)

            if writer is not None:
                for cup_box in result.cup_boxes:
                    DetectorService.draw_box(frame, cup_box, "Cup", (0, 255, 255), thickness=2)
                writer.write(frame)

        capture.release()
        if writer is not None:
            writer.release()

        response = aggregator.finalize(fps=fps)
        if output_path is not None:
            if output_path.exists():
                response.processed_video_path = output_path.name
            else:
                print(f"Warning: Output video file not created at {output_path}")
        return response

    def _draw_person_annotation(
        self,
        frame: np.ndarray,
        track_id: int,
        person_box: BBox,
        aggregator: AnalyticsAggregator,
    ) -> None:
        is_suspicious = track_id in aggregator.suspicious_ids
        color = (0, 0, 255) if is_suspicious else (0, 255, 0)
        label = f"ID {track_id} {'Theft' if is_suspicious else 'Normal'}"
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
