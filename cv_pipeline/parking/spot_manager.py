from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv_pipeline.parking.pixel_detector import PixelOccupancyDetector


@dataclass
class SpotManager:
    spot_config_path: str
    bbox_overlap_threshold: float = 0.40
    bbox_anchor_y_ratio: float = 0.85
    occupied_hold_frames: int = 10
    # Pixel validator — activo por defecto
    pixel_diff_threshold: float = 18.0
    pixel_min_empty_frames: int = 25
    use_pixel_validator: bool = True

    def __post_init__(self) -> None:
        self.spots = self._load_spots(self.spot_config_path)
        # Keep a short occupancy memory to avoid false "free" states during brief occlusions.
        self._hold_counters: dict[str, int] = {spot["id"]: 0 for spot in self.spots}
        self._pixel_detector: PixelOccupancyDetector | None = (
            PixelOccupancyDetector(
                diff_threshold=self.pixel_diff_threshold,
                min_empty_frames=self.pixel_min_empty_frames,
            )
            if self.use_pixel_validator
            else None
        )

    def _load_spots(self, path: str) -> list[dict[str, Any]]:
        content = Path(path).read_text(encoding="utf-8")
        data = json.loads(content)
        return data.get("spots", [])

    def evaluate(
        self,
        detections: list[dict[str, Any]],
        frame: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Return occupancy state per configured parking spot."""
        states: list[dict[str, Any]] = []
        for spot in self.spots:
            spot_id = spot["id"]
            polygon = np.array(spot["polygon"], dtype=np.int32)
            yolo_occupied = self._spot_has_vehicle(polygon, detections)

            # --- Pixel validator -------------------------------------------
            occupied_now = yolo_occupied
            if self._pixel_detector is not None and frame is not None:
                roi = self._pixel_detector.extract_roi(frame, spot["polygon"])
                if roi is not None:
                    # Actualizar referencia solo con frames donde YOLO ve vacío
                    self._pixel_detector.update_reference(spot_id, roi, not yolo_occupied)
                    pixel_occupied, has_ref = self._pixel_detector.is_occupied(spot_id, roi)
                    if has_ref:
                        # El pixel detector sustituye a YOLO una vez calentado:
                        # - Si YOLO se equivoca (bbox vecino), los píxeles reales
                        #   siguen mostrando el suelo → plaza libre.
                        # - Si hay coche aunque YOLO lo pierda, los píxeles lo
                        #   detectan igualmente.
                        occupied_now = pixel_occupied
            # ----------------------------------------------------------------

            if occupied_now:
                self._hold_counters[spot_id] = self.occupied_hold_frames
                occupied = True
            else:
                remaining = self._hold_counters.get(spot_id, 0)
                occupied = remaining > 0
                if remaining > 0:
                    self._hold_counters[spot_id] = remaining - 1

            states.append({"spot_id": spot_id, "occupied": occupied})
        return states

    def _spot_has_vehicle(self, polygon: np.ndarray, detections: list[dict[str, Any]]) -> bool:
        polygon_area = cv2.contourArea(polygon.astype(np.float32))
        if polygon_area <= 0:
            return False

        polygon_f = polygon.astype(np.float32)
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            if x2 <= x1 or y2 <= y1:
                continue

            rect = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            )
            intersect_area, _ = cv2.intersectConvexConvex(polygon_f, rect)
            if intersect_area <= 0:
                continue

            overlap_ratio = float(intersect_area / polygon_area)
            if overlap_ratio < self.bbox_overlap_threshold:
                continue

            anchor_x = float((x1 + x2) / 2.0)
            anchor_y = float(y1 + self.bbox_anchor_y_ratio * (y2 - y1))
            anchor_inside = cv2.pointPolygonTest(polygon_f, (anchor_x, anchor_y), False) >= 0
            if anchor_inside:
                return True
        return False
