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
        self.detector = YoloDetector(model_path=self.model_path, confidence=self.confidence)
        self.spot_manager = SpotManager(spot_config_path=self.spot_config_path)

    def process_frame(self, frame: Any) -> dict[str, Any]:
        detections = self.detector.detect(frame)
        spots = self.spot_manager.evaluate(detections)
        occupied = sum(1 for spot in spots if spot["occupied"])

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_spots": len(spots),
            "occupied_spots": occupied,
            "free_spots": len(spots) - occupied,
            "spots": spots,
            "detections": detections,
        }

    def stream(self, include_frame: bool = False):
        for frame, frame_index in self.video_source.frames():
            state = self.process_frame(frame)
            state["frame_index"] = frame_index
            if include_frame:
                state["_frame"] = frame
            yield state
