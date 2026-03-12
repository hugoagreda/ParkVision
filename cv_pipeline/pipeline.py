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
    bbox_overlap_threshold: float = 0.40
    occupied_hold_frames: int = 10
    pixel_diff_threshold: float = 18.0
    pixel_min_empty_frames: int = 25
    use_pixel_validator: bool = True

    def __post_init__(self) -> None:
        self.video_source = VideoSource(self.source)
        self.detector = YoloDetector(model_path=self.model_path, confidence=self.confidence)
        self.spot_manager = SpotManager(
            spot_config_path=self.spot_config_path,
            bbox_overlap_threshold=self.bbox_overlap_threshold,
            occupied_hold_frames=self.occupied_hold_frames,
            pixel_diff_threshold=self.pixel_diff_threshold,
            pixel_min_empty_frames=self.pixel_min_empty_frames,
            use_pixel_validator=self.use_pixel_validator,
        )

    def process_frame(self, frame: Any) -> dict[str, Any]:
        detections = self.detector.detect(frame)
        spots = self.spot_manager.evaluate(detections, frame=frame)
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
