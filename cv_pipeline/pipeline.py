from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from cv_pipeline.detector.yolo_detector import YoloDetector
from cv_pipeline.ingestion.video_source import VideoSource
from cv_pipeline.parking.spot_manager import SpotManager


@dataclass
class ParkingPipeline:
    source: str | int
    model_path: str
    spot_config_path: str
    confidence: float = 0.35

    def __post_init__(self) -> None:
        self.video_source = VideoSource(self.source)
        self.detector = YoloDetector(self.model_path, self.confidence)
        self.spot_manager = SpotManager(self.spot_config_path)

    def run_once(self, frame: Any) -> dict[str, Any]:
        detections = self.detector.detect(frame)
        spots = self.spot_manager.evaluate(detections)
        occupied_count = sum(1 for s in spots if s["occupied"])

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_spots": len(spots),
            "occupied_spots": occupied_count,
            "free_spots": len(spots) - occupied_count,
            "spots": spots,
            "detections": detections,
        }

    def stream(self):
        for frame, frame_index in self.video_source.frames():
            state = self.run_once(frame)
            state["frame_index"] = frame_index
            yield state
