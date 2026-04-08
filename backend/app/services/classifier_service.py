from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from app.config import Settings


class BehaviorClassifierService:
    """Supports both Keras (.keras/.h5) and PyTorch (.pt/.pth) models."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_type: Optional[str] = None  # "keras" or "torch"
        self._load_model(settings.resolved_model_path)

    def _load_model(self, model_path: Path) -> None:
        if not model_path.exists():
            print(f"Warning: Model path {model_path} does not exist.")
            return

        suffix = model_path.suffix.lower()

        # --- Try Keras first for .keras / .h5 files ---
        if suffix in (".keras", ".h5"):
            try:
                import keras
                self.model = keras.saving.load_model(str(model_path))
                self.model_type = "keras"
                print(f"Successfully loaded Keras model from {model_path}")
                return
            except Exception as e:
                print(f"Failed to load Keras model: {e}")
                # Fall through to PyTorch loaders

        # --- PyTorch loaders ---
        try:
            try:
                scripted = torch.jit.load(str(model_path), map_location=self.device)
                scripted.eval()
                self.model = scripted
                self.model_type = "torch"
                print(f"Successfully loaded TorchScript model from {model_path}")
                return
            except Exception:
                pass

            loaded = torch.load(str(model_path), map_location=self.device)
            if isinstance(loaded, torch.nn.Module):
                loaded.eval()
                self.model = loaded
                self.model_type = "torch"
            elif isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], torch.nn.Module):
                model = loaded["model"]
                model.eval()
                self.model = model
                self.model_type = "torch"

            if self.model:
                print(f"Successfully loaded PyTorch model from {model_path}")
            else:
                print(f"Warning: Could not load model from {model_path}")

        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("The backend will start, but behavior classification will be disabled.")
            self.model = None

    # --------------------------------------------------------------------- #
    # Inference
    # --------------------------------------------------------------------- #
    def classify_crop(self, crop: np.ndarray) -> str:
        if crop.size == 0 or self.model is None:
            return "Normal"

        try:
            if self.model_type == "keras":
                return self._classify_keras(crop)
            else:
                return self._classify_torch(crop)
        except Exception:
            return "Normal"

    # --- Keras inference path ---
    def _classify_keras(self, crop: np.ndarray) -> str:
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = rgb.astype(np.float32) / 255.0
        batch = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)

        prediction = self.model.predict(batch, verbose=0)

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        prediction = np.squeeze(prediction)

        # Binary sigmoid output
        if prediction.ndim == 0 or (prediction.ndim == 1 and prediction.shape[0] == 1):
            prob = float(prediction) if prediction.ndim == 0 else float(prediction[0])
            return "Theft" if prob >= 0.5 else "Normal"

        # Multi-class softmax output
        if prediction.ndim >= 1 and prediction.shape[0] >= 2:
            class_idx = int(np.argmax(prediction))
            return "Theft" if class_idx == 1 else "Normal"

        return "Normal"

    # --- PyTorch inference path ---
    def _classify_torch(self, crop: np.ndarray) -> str:
        tensor = self._preprocess_torch(crop)
        with torch.no_grad():
            output = self.model(tensor)
        return self._predict_label_torch(output)

    def _preprocess_torch(self, crop: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = rgb.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (image - mean) / std
        chw = np.transpose(normalized, (2, 0, 1))

        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        return tensor

    @staticmethod
    def _predict_label_torch(output: torch.Tensor | tuple | list) -> str:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            return "Normal"

        if output.ndim == 1:
            output = output.unsqueeze(0)

        if output.shape[-1] == 1:
            probability = torch.sigmoid(output)[0, 0].item()
            return "Theft" if probability >= 0.5 else "Normal"

        if output.shape[-1] >= 2:
            class_idx = int(torch.argmax(output, dim=1)[0].item())
            return "Theft" if class_idx == 1 else "Normal"

        return "Normal"
