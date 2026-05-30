from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    landmarks: Optional[np.ndarray] = None  # (5, 2)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    state: str  # 'tentative' | 'confirmed' | 'lost'
    age: int = 0
    hits: int = 0
    time_since_update: int = 0


@dataclass
class FrameResult:
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    anonymized_frame: Optional[np.ndarray] = None
