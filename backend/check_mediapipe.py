import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

def test_mediapipe():
    print("--- MediaPipe Diagnostic ---")
    model_path = os.path.join("models", "hand_landmarker.task")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    print(f"Found model file: {model_path}")
    
    try:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE
        )
        with HandLandmarker.create_from_options(options) as landmarker:
            print("SUCCESS: MediaPipe HandLandmarker initialized successfully.")
            
            # Test with a dummy black image
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_image)
            
            result = landmarker.detect(mp_image)
            print("SUCCESS: Inference completed (no hands expected in black image).")
            
    except Exception as e:
        print(f"FAILURE: MediaPipe initialization or inference failed: {e}")

if __name__ == "__main__":
    test_mediapipe()
