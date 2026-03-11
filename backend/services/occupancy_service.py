from __future__ import annotations

import logging
import threading
from typing import Any

from backend.core.config import settings
from cv_pipeline.pipeline import ParkingPipeline

logger = logging.getLogger(__name__)


class OccupancyService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {
            "timestamp": "",
            "frame_index": -1,
            "total_spots": 0,
            "occupied_spots": 0,
            "free_spots": 0,
            "spots": [],
            "detections": [],
        }
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()

    def _run_loop(self) -> None:
        pipeline = ParkingPipeline(
            source=settings.parsed_video_source,
            model_path=settings.yolo_model_path,
            spot_config_path=settings.spot_config_path,
            confidence=settings.yolo_confidence,
        )
        try:
            for state in pipeline.stream():
                with self._lock:
                    self._latest = state
        except Exception as exc:
            logger.exception("Pipeline failed: %s", exc)

    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)


occupancy_service = OccupancyService()
