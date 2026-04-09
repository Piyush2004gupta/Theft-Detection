from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import time
from app.detection import DetectionResult

class BehaviorAnalyzer:
    def __init__(self):
        # person_id -> history of hand positions
        self.hand_trajectories: Dict[int, deque] = {}
        # object_id -> original position
        self.object_origins: Dict[int, Tuple[int, int, int, int]] = {}
        # object_id -> current owner_id (first person detected near it for a while)
        self.object_owners: Dict[int, int] = {}
        # track interactions over time
        self.history_window = 60 # ~2 seconds at 30fps
        
        # Thresholds
        self.displacement_threshold = 50 
        self.grab_speed_threshold = 0.05 # change in normalized coords per frame
        self.interaction_dist = 120

    def analyze(self, frame_idx: int, people: List[DetectionResult], objects: List[DetectionResult], hands: List[Dict], person_behaviors: Dict[int, str], width: int, height: int):
        results = []
        self.w = width
        self.h = height
        
        # 0. Persistence factor for model predictions
        # If the model says "Theft", boost confidence
        
        # 1. Update Object Registry & Detect Displacement
        for obj in objects:
            if obj.track_id not in self.object_origins:
                self.object_origins[obj.track_id] = obj.box
            
            # Check displacement
            orig = self.object_origins[obj.track_id]
            curr = obj.box
            dist = self._dist(self._center(orig), self._center(curr))
            is_displaced = dist > self.displacement_threshold
            
            # 2. Assign/Verify Owner
            # Simple heuristic: person closest to object in first 2 seconds is owner
            if obj.track_id not in self.object_owners:
                best_p, min_d = self._find_closest_person(obj, people)
                if best_p and min_d < self.interaction_dist:
                    self.object_owners[obj.track_id] = best_p.track_id

            # 3. Analyze Hand Interactions
            for p in people:
                # Add to trajectory
                if p.track_id not in self.hand_trajectories:
                    self.hand_trajectories[p.track_id] = deque(maxlen=self.history_window)
                
                # Match hand to person (MediaPipe hands are anonymous, so match by proximity to person bbox)
                associated_hands = self._match_hands_to_person(p, hands)
                
                for h in associated_hands:
                    wrist = h["wrist"]
                    self.hand_trajectories[p.track_id].append(wrist)
                    
                    # Check "Restricted Zone" Entry
                    if self._point_in_box(wrist, obj.box):
                        # Analyze Intent
                        speed = self._calculate_hand_speed(p.track_id)
                        is_owner = self.object_owners.get(obj.track_id) == p.track_id
                        model_says_theft = person_behaviors.get(p.track_id) == "Theft"
                        
                        if (not is_owner and is_displaced) or model_says_theft:
                            # Quick Grab + Displacement by non-owner -> THEFT
                            conf = self._calculate_confidence(speed, dist)
                            if model_says_theft: conf = max(conf, 0.85)
                            
                            results.append({
                                "type": "THFT",
                                "person_id": p.track_id,
                                "object_id": obj.track_id, 
                                "confidence": conf,
                                "description": f"Theft: Person {p.track_id} snatched object {obj.track_id}"
                            })
                        elif not is_owner and self._is_reaching_fast(speed):
                            results.append({
                                "type": "SUSP",
                                "person_id": p.track_id,
                                "object_id": obj.track_id,
                                "confidence": 0.6,
                                "description": "Suspicious: Rapid reach toward object"
                            })
        
        return results

    def _dist(self, p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    def _center(self, box):
        return ((box[0]+box[2])/2, (box[1]+box[3])/2)

    def _point_in_box(self, pt, box):
        # pt is (x, y, z) normalized. scale by internal self.w/self.h
        return box[0] <= pt[0]*self.w <= box[2] and box[1] <= pt[1]*self.h <= box[3]

    def _find_closest_person(self, obj, people):
        min_d = 9999
        best_p = None
        oc = self._center(obj.box)
        for p in people:
            pc = self._center(p.box)
            d = self._dist(oc, pc)
            if d < min_d:
                min_d = d
                best_p = p
        return best_p, min_d

    def _match_hands_to_person(self, person, hands):
        matches = []
        for h in hands:
            # Check if hand wrist is inside or near person bbox
            if self._point_in_box(h["wrist"], person.box):
                matches.append(h)
        return matches

    def _calculate_hand_speed(self, person_id):
        traj = self.hand_trajectories.get(person_id)
        if not traj or len(traj) < 2: return 0
        # Speed as distance between last two points
        p1, p2 = traj[-2], traj[-1]
        return self._dist(p1, p2)

    def _is_reaching_fast(self, speed):
        return speed > self.grab_speed_threshold

    def _calculate_confidence(self, speed, displacement):
        # More speed and more displacement = higher confidence
        s_score = min(1.0, speed / 0.1)
        d_score = min(1.0, displacement / 200)
        return (s_score + d_score) / 2.0
