from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from app.schemas import AnalyticsResponse, PersonAnalytics
from app.utils.time_utils import frame_to_seconds, seconds_to_hhmmss


BBox = Tuple[int, int, int, int]


@dataclass
class TrackState:
    first_frame: int
    last_frame: int
    activity_votes: List[str] = field(default_factory=list)
    holds_cup_frames: int = 0


class AnalyticsAggregator:
    def __init__(self) -> None:
        self.tracks: Dict[int, TrackState] = {}
        self.total_cups_detected: int = 0
        self.suspicious_ids: Set[int] = set()

    def update_track(self, track_id: int, frame_index: int) -> None:
        state = self.tracks.get(track_id)
        if state is None:
            self.tracks[track_id] = TrackState(first_frame=frame_index, last_frame=frame_index)
            return
        state.last_frame = frame_index

    def update_cup_count(self, cups_seen_in_frame: int) -> None:
        self.total_cups_detected += max(0, cups_seen_in_frame)

    def vote_activity(self, track_id: int, label: str) -> None:
        state = self.tracks.get(track_id)
        if state is None:
            return
        state.activity_votes.append(label)
        if label == "Theft":
            self.suspicious_ids.add(track_id)

    def increment_holding_cup(self, track_id: int) -> None:
        state = self.tracks.get(track_id)
        if state is None:
            return
        state.holds_cup_frames += 1

    def finalize(self, fps: float) -> AnalyticsResponse:
        people: List[PersonAnalytics] = []

        for track_id, state in sorted(self.tracks.items(), key=lambda item: item[0]):
            in_seconds = frame_to_seconds(state.first_frame, fps)
            out_seconds = frame_to_seconds(state.last_frame, fps)
            total_seconds = max(0, out_seconds - in_seconds)

            theft_votes = sum(1 for vote in state.activity_votes if vote == "Theft")
            normal_votes = sum(1 for vote in state.activity_votes if vote == "Normal")
            activity = "Theft" if theft_votes > normal_votes else "Normal"

            if state.holds_cup_frames > 0 and theft_votes > 0:
                activity = "Theft"

            if activity == "Theft":
                self.suspicious_ids.add(track_id)

            people.append(
                PersonAnalytics(
                    id=track_id,
                    in_time=seconds_to_hhmmss(in_seconds),
                    out_time=seconds_to_hhmmss(out_seconds),
                    time_spent_seconds=total_seconds,
                    activity=activity,
                )
            )

        overall_status = "Theft Detected" if self.suspicious_ids else "Normal"

        return AnalyticsResponse(
            people=people,
            total_people=len(people),
            total_cups_detected=self.total_cups_detected,
            suspicious_ids=sorted(self.suspicious_ids),
            overall_status=overall_status,
            processed_video_path=None,
        )
