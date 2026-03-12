from __future__ import annotations

import argparse
import csv
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
    spot_overlap_threshold: float,
    occupied_hold_frames: int,
    max_frames: int,
    annotated_video_output: str | None = None,
    pixel_diff_threshold: float = 18.0,
    pixel_min_empty_frames: int = 25,
    use_pixel_validator: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    cap_probe = cv2.VideoCapture(video_path)
    source_fps: float = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    cap_probe.release()

    pipeline = ParkingPipeline(
        source=video_path,
        model_path=model_path,
        spot_config_path=spots_path,
        confidence=confidence,
        bbox_overlap_threshold=spot_overlap_threshold,
        occupied_hold_frames=occupied_hold_frames,
        pixel_diff_threshold=pixel_diff_threshold,
        pixel_min_empty_frames=pixel_min_empty_frames,
        use_pixel_validator=use_pixel_validator,
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
                    source_fps,
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

    spot_ids = [s["spot_id"] for s in frames[0]["spots"]]
    spots_stats: dict[str, Any] = {}
    for spot_id in spot_ids:
        occ_seq = [int(s["occupied"]) for f in frames for s in f["spots"] if s["spot_id"] == spot_id]
        changes = sum(1 for i in range(1, len(occ_seq)) if occ_seq[i] != occ_seq[i - 1])
        spots_stats[spot_id] = {
            "occupancy_rate_pct": round(mean(occ_seq) * 100, 1),
            "state_changes": changes,
        }

    summary = {
        "video": video_path,
        "frames_processed": len(frames),
        "avg_occupied_spots": round(mean(occupied_values), 3),
        "max_occupied_spots": max(occupied_values),
        "min_occupied_spots": min(occupied_values),
        "spots_stats": spots_stats,
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
        "--spot-overlap-threshold",
        type=float,
        default=0.40,
        help="Min bbox/polygon overlap ratio to mark spot as occupied",
    )
    parser.add_argument(
        "--occupied-hold-frames",
        type=int,
        default=10,
        help="Frames to keep spot occupied after temporary occlusion",
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
            spot_overlap_threshold=args.spot_overlap_threshold,
            occupied_hold_frames=args.occupied_hold_frames,
            max_frames=args.max_frames,
            annotated_video_output=annotated_video_path,
            pixel_diff_threshold=args.pixel_diff_threshold,
            pixel_min_empty_frames=args.pixel_min_empty_frames,
            use_pixel_validator=not args.no_pixel_validator,
        )

        stem = Path(video).stem
        frame_output = output_dir / f"{stem}_frames.jsonl"
        summary_output = output_dir / f"{stem}_summary.json"

        with frame_output.open("w", encoding="utf-8") as handle:
            for frame in frames:
                handle.write(json.dumps(frame, ensure_ascii=True) + "\n")

        summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        batch_summary.append(summary)

        if frames:
            csv_output = output_dir / f"{stem}_timeline.csv"
            spot_ids = [s["spot_id"] for s in frames[0]["spots"]]
            with csv_output.open("w", newline="", encoding="utf-8") as csv_handle:
                writer_csv = csv.writer(csv_handle)
                writer_csv.writerow(["frame_index", "timestamp"] + spot_ids)
                for f in frames:
                    spot_map = {s["spot_id"]: int(s["occupied"]) for s in f["spots"]}
                    writer_csv.writerow([f["frame_index"], f["timestamp"]] + [spot_map.get(sid, 0) for sid in spot_ids])

        print(f"[ParkVision] Done: {video}")
        print(f"  - Frames file: {frame_output}")
        print(f"  - Summary file: {summary_output}")
        if frames:
            print(f"  - Timeline CSV: {csv_output}")
        if produced_video_path is not None:
            print(f"  - Annotated video: {produced_video_path}")

    batch_output = output_dir / "batch_summary.json"
    batch_output.write_text(json.dumps(batch_summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[ParkVision] Batch summary: {batch_output}")


if __name__ == "__main__":
    main()
