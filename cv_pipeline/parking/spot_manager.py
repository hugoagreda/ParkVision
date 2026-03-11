from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import cv2
import numpy as np

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class SpotManager:
    spot_config_path: str

    def __post_init__(self) -> None:
        self.spots = self._load_spots(self.spot_config_path)

    def _load_spots(self, path: str) -> List[dict[str, Any]]:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        return config["spots"]

    def evaluate(self, detections: Iterable[dict[str, Any]]) -> List[dict[str, Any]]:
        spot_status = []
        for spot in self.spots:
            polygon = np.array(spot["polygon"], dtype=np.int32)
            occupied = self._is_occupied(polygon, detections)
            spot_status.append(
                {
                    "spot_id": spot["id"],
                    "occupied": occupied,
                }
            )
        return spot_status

    def _is_occupied(self, polygon: np.ndarray, detections: Iterable[dict[str, Any]]) -> bool:
        for detection in detections:
            if detection["label"] not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = detection["bbox"]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            inside = cv2.pointPolygonTest(polygon, center, False)
            if inside >= 0:
                return True

        return False
