from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Settings:
    # -----------------------------------------------------------------------
    # Paths – default to sibling folders inside the backend/ directory
    # -----------------------------------------------------------------------
    base_dir: Path = Path(os.getenv("THEFT_BASE_DIR", Path(__file__).resolve().parents[1]))
    outputs_dir: Path = Path(os.getenv("THEFT_OUTPUTS_DIR", "outputs"))
    model_path: Path = Path(os.getenv("THEFT_BEHAVIOR_MODEL", "models/theft_modelss.keras"))

    # -----------------------------------------------------------------------
    # YOLO / tracker
    # -----------------------------------------------------------------------
    yolo_model: str = os.getenv("THEFT_YOLO_MODEL", "yolov8n.pt")
    tracker_cfg: str = os.getenv("THEFT_TRACKER_CFG", "bytetrack.yaml")

    # -----------------------------------------------------------------------
    # Detection class IDs (COCO)
    # -----------------------------------------------------------------------
    person_class_id: int = int(os.getenv("THEFT_PERSON_CLASS_ID", "0"))
    cup_class_id: int = int(os.getenv("THEFT_CUP_CLASS_ID", "41"))

    # -----------------------------------------------------------------------
    # Inference settings
    # -----------------------------------------------------------------------
    yolo_conf: float = float(os.getenv("THEFT_YOLO_CONF", "0.25"))
    classify_every_n_frames: int = int(os.getenv("THEFT_CLASSIFY_EVERY_N_FRAMES", "15"))
    save_video_default: bool = os.getenv("THEFT_SAVE_VIDEO_DEFAULT", "true").lower() == "true"

    # -----------------------------------------------------------------------
    # CORS – comma-separated origins, e.g. "http://localhost:5500,https://myapp.com"
    # Use "*" to allow all (development only).
    # -----------------------------------------------------------------------
    cors_origins: List[str] = field(default_factory=lambda: [
        o.strip()
        for o in os.getenv("THEFT_CORS_ORIGINS", "*").split(",")
        if o.strip()
    ])

    # -----------------------------------------------------------------------
    # Theft Detection Thresholds
    # -----------------------------------------------------------------------
    hand_object_overlap_threshold: float = float(os.getenv("THEFT_HAND_OVERLAP", "0.15"))
    object_disappeared_threshold: float = float(os.getenv("THEFT_OBJ_MISSING_DIST", "80.0"))
    move_away_threshold: float = float(os.getenv("THEFT_MOVE_AWAY_DIST", "120.0"))

    # -----------------------------------------------------------------------
    # DeepSORT Settings
    # -----------------------------------------------------------------------
    max_age: int = int(os.getenv("THEFT_MAX_AGE", "100"))
    n_init: int = int(os.getenv("THEFT_N_INIT", "5"))

    max_cosine_distance: float = float(os.getenv("THEFT_MAX_COS_DIST", "0.2"))

    # -----------------------------------------------------------------------
    # Resolved paths
    # -----------------------------------------------------------------------
    @property
    def resolved_outputs_dir(self) -> Path:
        path = self.outputs_dir
        if not path.is_absolute():
            path = self.base_dir / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resolved_model_path(self) -> Path:
        path = self.model_path
        if not path.is_absolute():
            path = self.base_dir / path
        return path


settings = Settings()
