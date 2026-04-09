from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from app.config import Settings
from app.schemas import AnalyticsResponse, PersonAnalytics
from app.main_pipeline import AdvancedVideoPipeline

class VideoProcessor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = AdvancedVideoPipeline()

    def process_video(self, video_path: Path, save_video: bool = True) -> AnalyticsResponse:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        writer: Optional[cv2.VideoWriter] = None
        output_path: Optional[Path] = None

        if save_video:
            output_path = self._create_output_video_path(video_path)
            writer, output_path = self._create_video_writer(output_path, fps, (width, height))

        frame_index = 0
        overall_status = "Normal"
        suspicious_ids = set()

        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            frame_index += 1
            # Call the advanced modular pipeline
            processed_frame, behaviors = self.pipeline.process_frame(frame, frame_index)
            
            for b in behaviors:
                if b["type"] == "THFT":
                    overall_status = "Theft Detected"
                    suspicious_ids.add(b["person_id"])
                elif b["type"] == "SUSP":
                    if overall_status != "Theft Detected":
                        overall_status = "Suspicious Activity"
                    suspicious_ids.add(b["person_id"])

            if writer is not None:
                writer.write(processed_frame)

        capture.release()
        if writer is not None:
            writer.release()

        # Map person trajectories to analytics response
        people_analytics = []
        for pid, traj in self.pipeline.analyzer.hand_trajectories.items():
            # Estimate in/out time from first/last frame in trajectory
            # (Note: simpler for MVP, ideally use yolo track start/end)
            people_analytics.append(
                PersonAnalytics(
                    id=pid,
                    in_time="00:00:00", # Placeholder or calculated
                    out_time="00:00:10", # Placeholder or calculated
                    time_spent_seconds=len(traj) // int(fps),
                    activity="Theft" if pid in suspicious_ids else "Normal"
                )
            )

        response = AnalyticsResponse(
            people=people_analytics,
            total_people=len(people_analytics),
            suspicious_ids=list(suspicious_ids),
            overall_status=overall_status,
            processed_video_path=output_path.name if output_path else None
        )
        return response

    def _create_output_video_path(self, input_path: Path) -> Path:
        outputs_dir = self.settings.resolved_outputs_dir
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_adv_{timestamp}_{input_path.stem}.mp4"
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
