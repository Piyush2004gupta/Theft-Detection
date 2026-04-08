from __future__ import annotations


def frame_to_seconds(frame_index: int, fps: float) -> int:
    if fps <= 0:
        return 0
    return int(frame_index / fps)


def seconds_to_hhmmss(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
