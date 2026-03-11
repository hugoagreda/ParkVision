from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select YOLO ROI from first frame of a video and save config"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output",
        default="config/yolo_roi.json",
        help="Output JSON path for ROI configuration",
    )
    parser.add_argument(
        "--window-name",
        default="ParkVision ROI Selector",
        help="OpenCV window title",
    )
    return parser.parse_args()


def read_first_frame(video_path: str):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read first frame from video: {video_path}")

    return frame


ZONE_COLORS = [
    (0, 220, 255), (0, 255, 100), (255, 180, 0), (200, 0, 255),
    (0, 160, 255), (255, 80, 80), (80, 255, 200), (255, 255, 0),
]


class PolygonEditor:
    def __init__(self, frame: np.ndarray, window_name: str) -> None:
        self.frame = frame
        self.window_name = window_name
        # Each zone is a list of 4 [x, y] points
        self.zones: list[list[list[int]]] = []
        self.active_zone: int = -1      # index of zone being edited
        self.active_corner: int = -1    # corner index inside active zone
        self.drawing = False
        self.drag_start: Optional[tuple[int, int]] = None
        self.drag_current: Optional[tuple[int, int]] = None
        self.dragging_corner = False
        # History stores snapshots of the full zones list
        self.history: list[list[list[list[int]]]] = [[]]
        self.future: list[list[list[list[int]]]] = []
        self.handle_radius = 6

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _clone_zones(self) -> list[list[list[int]]]:
        return [[[p[0], p[1]] for p in zone] for zone in self.zones]

    def _push_history(self) -> None:
        self.history.append(self._clone_zones())
        self.future = []

    def _nearest_handle(self, x: int, y: int) -> tuple[int, int] | tuple[None, None]:
        """Return (zone_idx, corner_idx) of the nearest handle, or (None, None)."""
        best = (None, None)
        best_dist = float("inf")
        threshold = (self.handle_radius * 3) ** 2
        for zi, zone in enumerate(self.zones):
            for ci, (px, py) in enumerate(zone):
                dist = (px - x) ** 2 + (py - y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = (zi, ci)
        if best_dist <= threshold:
            return best
        return (None, None)

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        h, w = self.frame.shape[:2]
        return max(0, min(x, w - 1)), max(0, min(y, h - 1))

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            zi, ci = self._nearest_handle(x, y)
            if zi is not None:
                # Grab corner of an existing zone
                self.active_zone = zi
                self.active_corner = ci
                self.dragging_corner = True
                self._push_history()
            else:
                # Start drawing a new zone
                self.drawing = True
                self.drag_start = self._clamp(x, y)
                self.drag_current = self.drag_start

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_corner:
                nx, ny = self._clamp(x, y)
                self.zones[self.active_zone][self.active_corner] = [nx, ny]
            elif self.drawing:
                self.drag_current = self._clamp(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_corner:
                self.dragging_corner = False
            elif self.drawing:
                self.drawing = False
                x0, y0 = self.drag_start
                x1, y1 = self._clamp(x, y)
                if abs(x1 - x0) > 5 and abs(y1 - y0) > 5:
                    self._push_history()
                    new_zone = [
                        [min(x0, x1), min(y0, y1)],
                        [max(x0, x1), min(y0, y1)],
                        [max(x0, x1), max(y0, y1)],
                        [min(x0, x1), max(y0, y1)],
                    ]
                    self.zones.append(new_zone)
                    self.active_zone = len(self.zones) - 1
                self.drag_start = None
                self.drag_current = None

    def _draw_text(self, canvas: np.ndarray, text: str, x: int, y: int) -> None:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw(self) -> np.ndarray:
        canvas = self.frame.copy()
        overlay = canvas.copy()

        # All confirmed zones
        for zi, zone in enumerate(self.zones):
            color = ZONE_COLORS[zi % len(ZONE_COLORS)]
            pts = np.array(zone, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)

        canvas = cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0)

        for zi, zone in enumerate(self.zones):
            color = ZONE_COLORS[zi % len(ZONE_COLORS)]
            pts = np.array(zone, dtype=np.int32)
            thickness = 3 if zi == self.active_zone else 1
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
            # Label
            cx = sum(p[0] for p in zone) // 4
            cy = sum(p[1] for p in zone) // 4
            self._draw_text(canvas, f"Z{zi + 1}", cx - 10, cy + 8)
            for ci, (px, py) in enumerate(zone):
                is_active = (zi == self.active_zone and ci == self.active_corner)
                c = (0, 255, 0) if is_active else color
                cv2.circle(canvas, (px, py), self.handle_radius, c, -1)

        # Rectangle being drawn live
        if self.drawing and self.drag_start and self.drag_current:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_current
            next_color = ZONE_COLORS[len(self.zones) % len(ZONE_COLORS)]
            cv2.rectangle(canvas, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), next_color, 2)

        # HUD
        zone_count = f"  ({len(self.zones)} zona{'s' if len(self.zones) != 1 else ''})"
        lines = [
            "Click + arrastra: nueva zona" + zone_count,
            "Arrastra esquina: ajustar",
            "d: borrar zona activa  u/Ctrl+Z: deshacer  r: rehacer",
            "s / Enter: guardar  q / Esc: salir",
        ]
        for i, text in enumerate(lines):
            self._draw_text(canvas, text, 15, 28 + i * 28)

        return canvas

    def _undo(self) -> None:
        if len(self.history) <= 1:
            return
        self.future.append(self._clone_zones())
        self.history.pop()
        self.zones = self._clone_zones_from(self.history[-1])
        self.active_zone = min(self.active_zone, len(self.zones) - 1)

    def _redo(self) -> None:
        if not self.future:
            return
        self.history.append(self._clone_zones())
        self.zones = self._clone_zones_from(self.future.pop())
        self.active_zone = min(self.active_zone, len(self.zones) - 1)

    def _clone_zones_from(self, src: list[list[list[int]]]) -> list[list[list[int]]]:
        return [[[p[0], p[1]] for p in zone] for zone in src]

    def _delete_active(self) -> None:
        if 0 <= self.active_zone < len(self.zones):
            self._push_history()
            self.zones.pop(self.active_zone)
            self.active_zone = max(0, self.active_zone - 1) if self.zones else -1

    def edit(self) -> list[list[list[int]]]:
        while True:
            cv2.imshow(self.window_name, self._draw())
            key = cv2.waitKey(20) & 0xFF

            if key in (13, ord("s")):  # Enter or s
                cv2.destroyAllWindows()
                return self.zones
            if key in (27, ord("q")):  # Esc or q
                cv2.destroyAllWindows()
                return []
            if key in (ord("u"), 26):  # u or Ctrl+Z
                self._undo()
            if key == ord("r"):
                self._redo()
            if key == ord("d"):
                self._delete_active()


def main() -> None:
    args = parse_args()
    frame = read_first_frame(args.video)

    editor = PolygonEditor(frame=frame, window_name=args.window_name)
    zones = editor.edit()

    if not zones:
        print("Sin zonas guardadas.")
        return

    spots = [
        {"id": f"Z{i + 1}", "polygon": zone}
        for i, zone in enumerate(zones)
    ]
    output_data = {"spots": spots}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[ParkVision] {len(spots)} zona(s) guardadas en: {output_path}")
    print(json.dumps(output_data, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
