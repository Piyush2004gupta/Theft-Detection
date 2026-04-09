from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from typing import List, Tuple, Optional, Dict
from collections import deque

class HandTracker:
    def __init__(self, model_path: str = "models/hand_landmarker.task"):
        # We use the new Tasks API because mp.solutions is not consistently available
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.1, # Increase sensitivity
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.1
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.trajectories: Dict[int, deque] = {} 

    def get_landmarks(self, frame: np.ndarray) -> List[Dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        
        hands_data = []
        h, w = frame.shape[:2]
        
        if result.hand_landmarks:
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
                    "wrist": landmarks[0] 
                })
            
        return hands_data

    @staticmethod
    def draw_hand_skeleton(frame: np.ndarray, landmarks: List[Tuple[float, float, float]]):
        h, w = frame.shape[:2]
        # Connections for MediaPipe Hands
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (5, 9), (9, 10), (10, 11), (11, 12), # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Little
        ]
        
        points = []
        for lm in landmarks:
            px, py = int(lm[0] * w), int(lm[1] * h)
            points.append((px, py))
            cv2.circle(frame, (px, py), 4, (0, 210, 255), -1)

        for p1_idx, p2_idx in connections:
            if p1_idx < len(points) and p2_idx < len(points):
                p1 = points[p1_idx]
                p2 = points[p2_idx]
                cv2.line(frame, p1, p2, (180, 105, 255), 2)

    def close(self):
        self.landmarker.close()
