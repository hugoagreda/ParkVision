from __future__ import annotations

from fastapi import APIRouter

from backend.models.schemas import OccupancyResponse
from backend.services.occupancy_service import occupancy_service

router = APIRouter(prefix="/occupancy", tags=["occupancy"])


@router.get("/latest", response_model=OccupancyResponse)
def get_latest_occupancy() -> OccupancyResponse:
    data = occupancy_service.latest()
    return OccupancyResponse(**data)
