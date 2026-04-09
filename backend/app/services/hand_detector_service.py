from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


@dataclass
class HandResult:
    bbox: BBox
    landmarks: List[Tuple[float, float, float]]


# ---------------------------------------------------------------------------
# Try MediaPipe Tasks API (v0.10+), then fall back to legacy solutions API
# ---------------------------------------------------------------------------
_MP_TASKS_AVAILABLE = False
_MP_SOLUTIONS_AVAILABLE = False

try:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )
    _MP_TASKS_AVAILABLE = True
except ImportError:
    pass

if not _MP_TASKS_AVAILABLE:
    try:
        import mediapipe as mp
        _hands_module = mp.solutions.hands
        _MP_SOLUTIONS_AVAILABLE = True
    except (ImportError, AttributeError):
        pass


class HandDetectorService:
    """Detects hands using MediaPipe (Tasks or Solutions API) with a
    pseudo-hand fallback when neither is available."""

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self._landmarker = None
        self._legacy_hands = None

        # Resolve model path for Tasks API
        if model_dir is None:
            model_dir = str(Path(__file__).resolve().parents[2])  # backend/
        task_model = os.path.join(model_dir, "models", "hand_landmarker.task")

        if _MP_TASKS_AVAILABLE and os.path.isfile(task_model):
            try:
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=task_model),
                    running_mode=RunningMode.IMAGE,
                    num_hands=4,
                    min_hand_detection_confidence=0.35,
                    min_hand_presence_confidence=0.35,
                    min_tracking_confidence=0.35,
                )
                self._landmarker = HandLandmarker.create_from_options(options)
                print("Successfully initialized MediaPipe Hands (Tasks API).")
                return
            except Exception as e:
                print(f"Warning: Tasks API init failed ({e}). Trying fallback.")

        if _MP_SOLUTIONS_AVAILABLE:
            try:
                self._legacy_hands = _hands_module.Hands(
                    static_image_mode=False,
                    max_num_hands=4,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("Successfully initialized MediaPipe Hands (Solutions API).")
                return
            except Exception as e:
                print(f"Warning: Solutions API init failed ({e}).")

        print("Warning: MediaPipe not available. Using pseudo-hand fallback.")

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def detect_hands(
        self,
        frame: np.ndarray,
        person_tracks: Optional[List[Tuple[int, BBox]]] = None,
    ) -> List[HandResult]:
        hand_results: List[HandResult] = []

        if self._landmarker is not None:
            hand_results = self._detect_tasks(frame)
        elif self._legacy_hands is not None:
            hand_results = self._detect_solutions(frame)

        # Fallback to pseudo-hands
        if not hand_results and person_tracks:
            hand_results = self._detect_pseudo(person_tracks)

        return hand_results

    # --- Tasks API path ---
    def _detect_tasks(self, frame: np.ndarray) -> List[HandResult]:
        try:
            import mediapipe as _mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            hands: List[HandResult] = []
            h, w = frame.shape[:2]
            for hand_lms in result.hand_landmarks:
                xs = [lm.x for lm in hand_lms]
                ys = [lm.y for lm in hand_lms]
                x1, x2 = int(min(xs) * w), int(max(xs) * w)
                y1, y2 = int(min(ys) * h), int(max(ys) * h)
                pad = 10
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                hands.append(HandResult(bbox=(x1, y1, x2, y2), landmarks=landmarks))
            return hands
        except Exception as e:
            print(f"MediaPipe Tasks processing error: {e}")
            return []

    # --- Legacy Solutions API path ---
    def _detect_solutions(self, frame: np.ndarray) -> List[HandResult]:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._legacy_hands.process(rgb)
            hands: List[HandResult] = []
            if results.multi_hand_landmarks:
                h, w = frame.shape[:2]
                for hand_lms in results.multi_hand_landmarks:
                    xs = [lm.x for lm in hand_lms.landmark]
                    ys = [lm.y for lm in hand_lms.landmark]
                    x1, x2 = int(min(xs) * w), int(max(xs) * w)
                    y1, y2 = int(min(ys) * h), int(max(ys) * h)
                    pad = 10
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
                    hands.append(HandResult(bbox=(x1, y1, x2, y2), landmarks=landmarks))
            return hands
        except Exception as e:
            print(f"MediaPipe Solutions processing error: {e}")
            return []

    # --- Pseudo-hand fallback ---
    @staticmethod
    def _detect_pseudo(person_tracks: List[Tuple[int, BBox]]) -> List[HandResult]:
        pseudo: List[HandResult] = []
        for _, (x1, y1, x2, y2) in person_tracks:
            pw, ph = x2 - x1, y2 - y1
            # Left hand region
            lx1, ly1 = x1 + int(pw * 0.1), y1 + int(ph * 0.4)
            lx2, ly2 = x1 + int(pw * 0.35), y1 + int(ph * 0.7)
            pseudo.append(HandResult(bbox=(lx1, ly1, lx2, ly2), landmarks=[]))
            # Right hand region
            rx1, ry1 = x1 + int(pw * 0.65), y1 + int(ph * 0.4)
            rx2, ry2 = x1 + int(pw * 0.9), y1 + int(ph * 0.7)
            pseudo.append(HandResult(bbox=(rx1, ry1, rx2, ry2), landmarks=[]))
        return pseudo

    def close(self) -> None:
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:
                pass
        if self._legacy_hands:
            try:
                self._legacy_hands.close()
            except Exception:
                pass
