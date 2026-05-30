from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AnonymizationType(Enum):
    GAUSSIAN_BLUR = "gaussian_blur"
    MOSAIC = "mosaic"


@dataclass
class DetectorConfig:
    model_path: str = "models/centerface.onnx"
    score_threshold: float = 0.5
    nms_threshold: float = 0.3
    input_size: tuple[int, int] = (640, 640)  # (W, H)


@dataclass
class TrackerConfig:
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass
class AnonymizerConfig:
    method: AnonymizationType = AnonymizationType.GAUSSIAN_BLUR
    intensity: float = 0.8  # 0.0 - 1.0
    expand_ratio: float = 0.1


@dataclass
class EngineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    anonymizer: AnonymizerConfig = field(default_factory=AnonymizerConfig)
    detect_every_n: int = 1
    use_gpu: bool = False
