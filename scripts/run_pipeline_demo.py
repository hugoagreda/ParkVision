from __future__ import annotations

import argparse
import json

from cv_pipeline.pipeline import ParkingPipeline


def parse_source(raw_source: str) -> str | int:
    return int(raw_source) if raw_source.isdigit() else raw_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ParkVision CV pipeline demo")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument(
        "--spots",
        default="config/parking_spots.example.json",
        help="Parking spots json config path",
    )
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO confidence")
    parser.add_argument("--max-frames", type=int, default=30, help="Frames to process")
    args = parser.parse_args()

    pipeline = ParkingPipeline(
        source=parse_source(args.source),
        model_path=args.model,
        spot_config_path=args.spots,
        confidence=args.confidence,
    )

    for state in pipeline.stream():
        print(json.dumps(state, ensure_ascii=True))
        if state["frame_index"] + 1 >= args.max_frames:
            break


if __name__ == "__main__":
    main()
