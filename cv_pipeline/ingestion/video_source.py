from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple

import cv2
import numpy as np


@dataclass
class VideoSource:
    source: str | int

    def frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yield frames and frame index from a camera index or video file path."""
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame, frame_index
            frame_index += 1

        capture.release()
