from __future__ import annotations

from fastapi import FastAPI

from backend.api.routes.occupancy import router as occupancy_router
from backend.services.occupancy_service import occupancy_service

app = FastAPI(title="ParkVision API", version="0.1.0")


@app.on_event("startup")
def on_startup() -> None:
    occupancy_service.start()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(occupancy_router)
