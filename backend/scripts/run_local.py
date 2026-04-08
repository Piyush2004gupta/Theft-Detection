from __future__ import annotations

import argparse
from pathlib import Path

from app.config import settings
from app.services.video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local AI video analytics")
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument("--no-save-video", action="store_true", help="Disable saving annotated output video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Input video not found: {args.video}")

    processor = VideoProcessor(settings=settings)
    response = processor.process_video(video_path=args.video, save_video=not args.no_save_video)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
