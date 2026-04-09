from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from typing import List, Tuple, Optional, Dict
from collections import deque

class HandTracker:
    def __init__(self, model_path: str = "models/hand_landmarker.task"):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        # Trajectories: Dictionary mapping hand index (approximate) or region to path
        # Since MediaPipe Hands doesn't have native IDs across frames easily without a tracker,
        # we'll use spatial mapping in the pipeline.
        self.trajectories: Dict[int, deque] = {} 

    def get_landmarks(self, frame: np.ndarray) -> List[Dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        
        hands_data = []
        h, w = frame.shape[:2]
        
        for idx, hand_lms in enumerate(result.hand_landmarks):
            landmarks = []
            for lm in hand_lms:
                landmarks.append((lm.x, lm.y, lm.z))
            
            # Simple bbox for hand based on landmarks
            xs = [lm[0] for lm in landmarks]
            ys = [lm[1] for lm in landmarks]
            bbox = (int(min(xs)*w), int(min(ys)*h), int(max(xs)*w), int(max(ys)*h))
            
            hands_data.append({
                "id": idx,
                "bbox": bbox,
                "landmarks": landmarks,
                "wrist": landmarks[0] # Landmarks[0] is typically the wrist
            })
            
        return hands_data

    @staticmethod
    def draw_hand_skeleton(frame: np.ndarray, landmarks: List[Tuple[float, float, float]]):
        h, w = frame.shape[:2]
        connections = mp.solutions.hands.HAND_CONNECTIONS
        
        points = []
        for lm in landmarks:
            px, py = int(lm[0] * w), int(lm[1] * h)
            points.append((px, py))
            cv2.circle(frame, (px, py), 3, (0, 210, 255), -1)

        for connection in connections:
            p1 = points[connection[0]]
            p2 = points[connection[1]]
            cv2.line(frame, p1, p2, (180, 105, 255), 2)

    def close(self):
        self.landmarker.close()
