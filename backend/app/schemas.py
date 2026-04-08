from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class PersonAnalytics(BaseModel):
    id: int
    in_time: str
    out_time: str
    time_spent_seconds: int
    activity: str


class AnalyticsResponse(BaseModel):
    people: List[PersonAnalytics]
    total_people: int

    suspicious_ids: List[int]
    overall_status: str
    processed_video_path: Optional[str] = None
