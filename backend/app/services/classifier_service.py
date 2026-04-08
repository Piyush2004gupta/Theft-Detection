from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from app.config import Settings


class BehaviorClassifierService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[torch.nn.Module] = None
        self._load_model(settings.resolved_model_path)

    def _load_model(self, model_path: Path) -> None:
        if not model_path.exists():
            self.model = None
            return

        try:
            scripted_model = torch.jit.load(str(model_path), map_location=self.device)
            scripted_model.eval()
            self.model = scripted_model
            return
        except Exception:
            pass

        loaded = torch.load(str(model_path), map_location=self.device)
        if isinstance(loaded, torch.nn.Module):
            loaded.eval()
            self.model = loaded
        elif isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], torch.nn.Module):
            model = loaded["model"]
            model.eval()
            self.model = model
        else:
            self.model = None

    def classify_crop(self, crop: np.ndarray) -> str:
        if crop.size == 0 or self.model is None:
            return "Normal"

        try:
            tensor = self._preprocess(crop)
            with torch.no_grad():
                output = self.model(tensor)

            predicted_label = self._predict_label(output)
            return predicted_label
        except Exception:
            return "Normal"

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
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
    def _predict_label(output: torch.Tensor | tuple | list) -> str:
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
