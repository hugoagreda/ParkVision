from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
from ultralytics import YOLO


@dataclass
class YoloDetector:
    model_path: str = "yolov8n.pt"
    confidence: float = 0.35

    def __post_init__(self) -> None:
        self.model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> List[dict[str, Any]]:
        """Run YOLO and return normalized vehicle detections for a single frame."""
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        detections: List[dict[str, Any]] = []

        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0].item())
            label = result.names[class_id]
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": class_id,
                    "label": label,
                    "confidence": confidence,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

        return detections
