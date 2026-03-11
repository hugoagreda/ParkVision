from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel


class SpotStatus(BaseModel):
    spot_id: str
    occupied: bool


class OccupancyResponse(BaseModel):
    timestamp: str
    frame_index: int
    total_spots: int
    occupied_spots: int
    free_spots: int
    spots: List[SpotStatus]
    detections: List[dict[str, Any]]
