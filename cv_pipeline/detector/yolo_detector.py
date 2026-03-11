from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO

VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}


@dataclass
class YoloDetector:
    model_path: str = "yolov8n.pt"
    confidence: float = 0.35

    def __post_init__(self) -> None:
        self.model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run YOLO inference and return normalized vehicle detections."""
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        detections: list[dict[str, Any]] = []

        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = result.names[class_id]
            if label not in VEHICLE_LABELS:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "label": label,
                    "confidence": float(box.conf[0].item()),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

        return detections
