from __future__ import annotations

import cv2
import numpy as np

from .config import AnonymizerConfig, AnonymizationType
from .datatypes import Track


class Anonymizer:
    def __init__(self, config: AnonymizerConfig | None = None):
        self._config = config or AnonymizerConfig()

    def anonymize(self, image: np.ndarray, tracks: list[Track]) -> np.ndarray:
        output = image.copy()
        img_h, img_w = output.shape[:2]

        for track in tracks:
            x1, y1, x2, y2 = self._expand_bbox(track.bbox, img_h, img_w)
            if x2 <= x1 or y2 <= y1:
                continue

            if self._config.method == AnonymizationType.GAUSSIAN_BLUR:
                self._apply_gaussian_blur(output, x1, y1, x2, y2)
            else:
                self._apply_mosaic(output, x1, y1, x2, y2)

        return output

    def _apply_gaussian_blur(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> None:
        roi = image[y1:y2, x1:x2]
        face_size = max(roi.shape[0], roi.shape[1])
        ksize = int(self._config.intensity * face_size)
        ksize = max(3, min(ksize, 99))
        if ksize % 2 == 0:
            ksize += 1
        image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)

    def _apply_mosaic(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> None:
        roi = image[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        block = max(2, int((1.0 - self._config.intensity) * min(w, h) * 0.5) + 2)
        small_w = max(1, w // block)
        small_h = max(1, h // block)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        image[y1:y2, x1:x2] = cv2.resize(
            small, (w, h), interpolation=cv2.INTER_NEAREST
        )

    def _expand_bbox(
        self, bbox: np.ndarray, img_h: int, img_w: int
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        ew = w * self._config.expand_ratio
        eh = h * self._config.expand_ratio
        x1 = int(max(0, x1 - ew))
        y1 = int(max(0, y1 - eh))
        x2 = int(min(img_w, x2 + ew))
        y2 = int(min(img_h, y2 + eh))
        return x1, y1, x2, y2
