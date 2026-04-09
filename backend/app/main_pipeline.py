from __future__ import annotations
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional
from app.detection import Detector, DetectionResult
from app.hand_tracking import HandTracker
from app.behavior_logic import BehaviorAnalyzer
from app.services.classifier_service import BehaviorClassifierService
from app.config import settings

class AdvancedVideoPipeline:
    def __init__(self, model_dir: Optional[str] = None):
        # Use absolute path to the models directory relative to this file
        base_dir = Path(__file__).parent.parent
        self.hand_model_path = base_dir / "models" / "hand_landmarker.task"
        
        self.detector = Detector(valuable_classes=settings.theft_object_class_ids)
        self.hand_tracker = HandTracker(model_path=str(self.hand_model_path))
        self.analyzer = BehaviorAnalyzer()
        self.classifier = BehaviorClassifierService(settings)
        self.events = []

    def process_frame(self, frame: np.ndarray, frame_idx: int):
        # 1. Detect People and Objects
        people, objects = self.detector.detect_and_track(frame)
        
        # 2. Track Hands
        hand_results = self.hand_tracker.get_landmarks(frame)
        
        # 3. Behavior Classification (Model-based)
        h, w = frame.shape[:2]
        person_behaviors = {}
        for p in people:
            x1, y1, x2, y2 = p.box
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size > 0:
                # Classify every N frames to save compute
                if frame_idx % settings.classify_every_n_frames == 0:
                    behavior = self.classifier.classify_crop(crop)
                    person_behaviors[p.track_id] = behavior

        # 4. Analyze Behavior (Heuristic + Model)
        behaviors = self.analyzer.analyze(frame_idx, people, objects, hand_results, person_behaviors, w, h)
        
        # 5. Visualization & Logging
        self._visualize(frame, people, objects, hand_results, person_behaviors, behaviors)
        
        for b in behaviors:
            self.events.append({
                "frame": frame_idx,
                "timestamp": time.strftime("%H:%M:%S"),
                "event": b["description"],
                "confidence": b["confidence"]
            })
            
        return frame, behaviors

    def _visualize(self, frame, people, objects, hands, person_behaviors, behaviors):
        h, w = frame.shape[:2]
        
        # Draw People
        for p in people:
            behavior_label = person_behaviors.get(p.track_id, "Normal")
            color = (0, 0, 255) if behavior_label == "Theft" else (0, 255, 0)
            self.detector.draw_box(frame, p, f"P{p.track_id} [{behavior_label}]", color)
            
            # Draw Trajectory Line
            traj = self.analyzer.hand_trajectories.get(p.track_id)
            if traj and len(traj) > 1:
                pts = [ (int(pt[0]*w), int(pt[1]*h)) for pt in traj ]
                for i in range(len(pts)-1):
                    cv2.line(frame, pts[i], pts[i+1], (0, 210, 255), 2)
        
        # Draw Objects & Zones
        for obj in objects:
            color = (255, 255, 0)
            status = "Protected"
            
            # Show Original Position (Ghost Box)
            orig = self.analyzer.object_origins.get(obj.track_id)
            if orig:
                cv2.rectangle(frame, (orig[0], orig[1]), (orig[2], orig[3]), (100, 100, 100), 1, cv2.LINE_AA)
            
            self.detector.draw_box(frame, obj, f"Object {obj.track_id} [{status}]", color)

        # Draw Hands
        for hand in hands:
            self.hand_tracker.draw_hand_skeleton(frame, hand["landmarks"])

        # Draw Global Status
        overall_status = "Normal Activity"
        status_color = (0, 255, 0)
        
        for b in behaviors:
            if b["type"] == "THFT":
                overall_status = "THEFT DETECTED!"
                status_color = (0, 0, 255)
                break
            elif b["type"] == "SUSP":
                overall_status = "Suspicious Activity"
                status_color = (0, 165, 255)
        
        cv2.putText(frame, overall_status, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, status_color, 3)

