from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Normalized size for all spot ROI comparisons (w, h).
_ROI_SIZE = (64, 32)


@dataclass
class PixelOccupancyDetector:
    """Detecta ocupación comparando los píxeles actuales del polígono de plaza
    contra una imagen de referencia del suelo vacío.

    Cómo funciona:
    - Cada vez que YOLO reporta una plaza como libre, se usa ese frame para
      actualizar incrementalmente una imagen de referencia del suelo vacío.
    - Una vez acumulados ``min_empty_frames`` frames vacíos, la referencia
      se considera estable y se usa como validador.
    - En cada frame se computa la diferencia absoluta media (MAD) entre el ROI
      actual y la referencia. Si MAD >= ``diff_threshold`` → hay coche.

    Ventajas sobre bounding-box puro:
    - Inmune a falsos positivos de YOLO por solapamiento de bbox vecino.
    - Funciona aunque el coche no sea detectado por YOLO (coche pequeño,
      oclusión parcial, etc.).
    - No requiere etiquetas manuales.
    """

    diff_threshold: float = 18.0
    """MAD mínima (0-255) para considerar la plaza como ocupada."""

    min_empty_frames: int = 25
    """Frames YOLO-libre necesarios antes de que la referencia sea fiable."""

    def __post_init__(self) -> None:
        # Referencia float32 por plaza (spot_id → ndarray 32x64)
        self._references: dict[str, np.ndarray] = {}
        # Contador de frames vacíos acumulados por plaza
        self._empty_counts: dict[str, int] = {}
        # Acumulador de media incremental para construir la referencia
        self._accumulators: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Extracción de ROI
    # ------------------------------------------------------------------

    def extract_roi(
        self, frame: np.ndarray, polygon: list[list[int]]
    ) -> np.ndarray | None:
        """Extrae y normaliza el ROI en escala de grises del polígono dado.

        Returns ``None`` si el polígono cae fuera del frame.
        """
        pts = np.array(polygon, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
        return cv2.resize(gray, _ROI_SIZE, interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Actualización de referencia de fondo
    # ------------------------------------------------------------------

    def update_reference(
        self, spot_id: str, roi: np.ndarray, yolo_empty: bool
    ) -> None:
        """Actualiza la referencia del suelo vacío con media incremental.

        Solo se actualiza cuando YOLO dice que la plaza está libre, de modo
        que nunca incorporamos un coche a la referencia.
        """
        if not yolo_empty:
            return  # No tocar la referencia mientras YOLO ve coche

        roi_f = roi.astype(np.float32)
        cnt = self._empty_counts.get(spot_id, 0) + 1
        self._empty_counts[spot_id] = cnt

        if spot_id not in self._accumulators:
            self._accumulators[spot_id] = roi_f.copy()
        else:
            # Media incremental: evita acumular todos los frames en RAM
            alpha = 1.0 / cnt
            self._accumulators[spot_id] = (
                (1.0 - alpha) * self._accumulators[spot_id] + alpha * roi_f
            )

        if cnt >= self.min_empty_frames:
            self._references[spot_id] = self._accumulators[spot_id]

    # ------------------------------------------------------------------
    # Decisión de ocupación
    # ------------------------------------------------------------------

    def is_occupied(
        self, spot_id: str, roi: np.ndarray
    ) -> tuple[bool, bool]:
        """Devuelve ``(occupied, has_reference)``.

        ``occupied`` solo es significativo cuando ``has_reference`` es True.
        Mientras no hay referencia estable, el llamador debe confiar en YOLO.
        """
        ref = self._references.get(spot_id)
        if ref is None:
            return False, False
        mad = float(np.mean(np.abs(roi.astype(np.float32) - ref)))
        return mad >= self.diff_threshold, True

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------

    def debug_info(self, spot_id: str) -> dict:
        """Info de diagnóstico para un spot (útil en visualización)."""
        return {
            "spot_id": spot_id,
            "empty_frames_collected": self._empty_counts.get(spot_id, 0),
            "has_reference": spot_id in self._references,
            "min_empty_frames": self.min_empty_frames,
        }
