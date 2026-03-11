from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class SpotManager:
    spot_config_path: str

    def __post_init__(self) -> None:
        self.spots = self._load_spots(self.spot_config_path)

    def _load_spots(self, path: str) -> list[dict[str, Any]]:
        content = Path(path).read_text(encoding="utf-8")
        data = json.loads(content)
        return data.get("spots", [])

    def evaluate(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return occupancy state per configured parking spot."""
        states: list[dict[str, Any]] = []
        for spot in self.spots:
            polygon = np.array(spot["polygon"], dtype=np.int32)
            occupied = self._spot_has_vehicle(polygon, detections)
            states.append({"spot_id": spot["id"], "occupied": occupied})
        return states

    def _spot_has_vehicle(self, polygon: np.ndarray, detections: list[dict[str, Any]]) -> bool:
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if cv2.pointPolygonTest(polygon, center, False) >= 0:
                return True
        return False
