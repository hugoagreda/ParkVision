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
    parser.add_argument(
        "--spot-overlap-threshold",
        type=float,
        default=0.40,
        help="Min bbox/polygon overlap ratio to mark spot as occupied",
    )
    parser.add_argument(
        "--occupied-hold-frames",
        type=int,
        default=10,
        help="Frames to keep a spot occupied after temporary detection loss",
    )
    parser.add_argument(
        "--pixel-diff-threshold",
        type=float,
        default=18.0,
        help="MAD pixel diff (0-255) to consider a spot occupied by pixel analysis",
    )
    parser.add_argument(
        "--pixel-min-empty-frames",
        type=int,
        default=25,
        help="Frames of YOLO-empty needed before pixel reference is trusted",
    )
    parser.add_argument(
        "--no-pixel-validator",
        action="store_true",
        help="Disable pixel-level occupancy validator (use YOLO only)",
    )
    parser.add_argument("--max-frames", type=int, default=30, help="Frames to process")
    args = parser.parse_args()

    pipeline = ParkingPipeline(
        source=parse_source(args.source),
        model_path=args.model,
        spot_config_path=args.spots,
        confidence=args.confidence,
        bbox_overlap_threshold=args.spot_overlap_threshold,
        occupied_hold_frames=args.occupied_hold_frames,
        pixel_diff_threshold=args.pixel_diff_threshold,
        pixel_min_empty_frames=args.pixel_min_empty_frames,
        use_pixel_validator=not args.no_pixel_validator,
    )

    for state in pipeline.stream():
        print(json.dumps(state, ensure_ascii=True))
        if state["frame_index"] + 1 >= args.max_frames:
            break


if __name__ == "__main__":
    main()
