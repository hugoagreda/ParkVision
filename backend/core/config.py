from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    video_source: str = os.getenv("VIDEO_SOURCE", "0")
    yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    yolo_confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    spot_config_path: str = os.getenv("SPOT_CONFIG_PATH", "config/parking_spots.example.json")

    @property
    def parsed_video_source(self) -> str | int:
        return int(self.video_source) if self.video_source.isdigit() else self.video_source


settings = Settings()
