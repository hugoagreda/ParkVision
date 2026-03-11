from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_pipeline.pipeline import ParkingPipeline


def run_video(
    video_path: str,
    model_path: str,
    spots_path: str,
    confidence: float,
    max_frames: int,
    annotated_video_output: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    pipeline = ParkingPipeline(
        source=video_path,
        model_path=model_path,
        spot_config_path=spots_path,
        confidence=confidence,
    )

    spot_polygons = {
        spot["id"]: spot["polygon"]
        for spot in pipeline.spot_manager.spots
    }

    writer: cv2.VideoWriter | None = None
    frames: list[dict[str, Any]] = []
    for state in pipeline.stream(include_frame=annotated_video_output is not None):
        frame = state.pop("_frame", None)

        if annotated_video_output is not None and frame is not None:
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    annotated_video_output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (width, height),
                )

            # Draw parking zones first so YOLO boxes remain visible on top.
            for spot_state in state["spots"]:
                spot_id = spot_state["spot_id"]
                polygon = spot_polygons.get(spot_id)
                if not polygon:
                    continue
                poly = np.array(polygon, dtype=np.int32)
                color = (0, 0, 255) if spot_state["occupied"] else (0, 200, 0)
                cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
                px, py = polygon[0]
                cv2.putText(frame, spot_id, (int(px), int(py) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for detection in state["detections"]:
                x1, y1, x2, y2 = detection["bbox"]
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, (255, 170, 0), 2)
                label = f"{detection['label']} {detection['confidence']:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (p1[0], max(18, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 170, 0),
                    2,
                )

            summary_text = f"Occ: {state['occupied_spots']}/{state['total_spots']}"
            cv2.putText(
                frame,
                summary_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            writer.write(frame)

        frames.append(state)
        if max_frames > 0 and state["frame_index"] + 1 >= max_frames:
            break

    if writer is not None:
        writer.release()

    if not frames:
        summary = {
            "video": video_path,
            "frames_processed": 0,
            "avg_occupied_spots": 0.0,
            "max_occupied_spots": 0,
            "min_occupied_spots": 0,
            "final_state": {},
        }
        return frames, summary, annotated_video_output

    occupied_values = [frame["occupied_spots"] for frame in frames]
    summary = {
        "video": video_path,
        "frames_processed": len(frames),
        "avg_occupied_spots": round(mean(occupied_values), 3),
        "max_occupied_spots": max(occupied_values),
        "min_occupied_spots": min(occupied_values),
        "final_state": frames[-1],
    }
    return frames, summary, annotated_video_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ParkVision offline batch on video files")
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Video file paths to process. Example: --videos data/video1.mp4 data/video2.mp4",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--spots", default="config/parking_spots.example.json", help="Spot config path")
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames per video (0 means full video)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/offline",
        help="Directory for generated JSON results",
    )
    parser.add_argument(
        "--no-render-video",
        action="store_true",
        help="Disable annotated output video generation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_summary: list[dict[str, Any]] = []

    for video in args.videos:
        if not Path(video).exists():
            raise FileNotFoundError(f"Video not found: {video}")

        print(f"[ParkVision] Processing video: {video}")
        annotated_video_path = None if args.no_render_video else str(output_dir / f"{Path(video).stem}_annotated.mp4")
        frames, summary, produced_video_path = run_video(
            video_path=video,
            model_path=args.model,
            spots_path=args.spots,
            confidence=args.confidence,
            max_frames=args.max_frames,
            annotated_video_output=annotated_video_path,
        )

        stem = Path(video).stem
        frame_output = output_dir / f"{stem}_frames.jsonl"
        summary_output = output_dir / f"{stem}_summary.json"

        with frame_output.open("w", encoding="utf-8") as handle:
            for frame in frames:
                handle.write(json.dumps(frame, ensure_ascii=True) + "\n")

        summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        batch_summary.append(summary)

        print(f"[ParkVision] Done: {video}")
        print(f"  - Frames file: {frame_output}")
        print(f"  - Summary file: {summary_output}")
        if produced_video_path is not None:
            print(f"  - Annotated video: {produced_video_path}")

    batch_output = output_dir / "batch_summary.json"
    batch_output.write_text(json.dumps(batch_summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[ParkVision] Batch summary: {batch_output}")


if __name__ == "__main__":
    main()
