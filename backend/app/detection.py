from __future__ import annotations
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class DetectionResult:
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    confidence: float
    track_id: int = -1

class Detector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.25, valuable_classes: List[int] = None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.person_class = 0
        self.valuable_classes = valuable_classes or [41, 63, 67, 73]

    def detect_and_track(self, frame: np.ndarray) -> Tuple[List[DetectionResult], List[DetectionResult]]:
        results = self.model.track(frame, persist=True, conf=self.conf, verbose=False)
        
        people = []
        objects = []
        
        if not results or not results[0].boxes:
            return people, objects

        boxes = results[0].boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            tid = int(box.id[0]) if box.id is not None else -1
            coords = box.xyxy[0].cpu().numpy().astype(int)
            res = DetectionResult(box=tuple(coords), class_id=cls, confidence=conf, track_id=tid)
            
            if cls == self.person_class:
                people.append(res)
            elif cls in self.valuable_classes:
                objects.append(res)
                
        return people, objects

    @staticmethod
    def draw_box(frame: np.ndarray, res: DetectionResult, label: str, color: Tuple[int, int, int]):
        x1, y1, x2, y2 = res.box
        cv2 = __import__("cv2")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
