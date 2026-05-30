from __future__ import annotations

from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .anonymizer import Anonymizer
from .config import EngineConfig
from .datatypes import Detection, FrameResult, Track
from .detector import CenterFaceDetector
from .tracker import SORTTracker


class FaceAnonEngine:
    def __init__(self, config: EngineConfig | None = None):
        self._config = config or EngineConfig()
        self._detector = CenterFaceDetector(
            self._config.detector, use_gpu=self._config.use_gpu
        )
        self._tracker = SORTTracker(self._config.tracker)
        self._anonymizer = Anonymizer(self._config.anonymizer)

    def process_image(self, image: np.ndarray) -> FrameResult:
        detections = self._detector.detect(image)
        pseudo_tracks = [
            Track(
                track_id=i,
                bbox=d.bbox,
                state="confirmed",
                hits=self._config.tracker.min_hits,
            )
            for i, d in enumerate(detections)
        ]
        anonymized = self._anonymizer.anonymize(image, pseudo_tracks)
        return FrameResult(
            frame_index=0,
            detections=detections,
            tracks=pseudo_tracks,
            anonymized_frame=anonymized,
        )

    def process_video(
        self,
        input_path: str,
        output_path: str,
        callback: Optional[Callable[[int, int, FrameResult], None]] = None,
    ) -> list[FrameResult]:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self._tracker.reset()
        results: list[FrameResult] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self._process_single_frame(frame, frame_idx)
            results.append(result)

            if result.anonymized_frame is not None:
                writer.write(result.anonymized_frame)

            if callback is not None:
                callback(frame_idx, total, result)

            frame_idx += 1

        cap.release()
        writer.release()
        return results

    def process_video_frames(
        self, frames: Iterator[np.ndarray]
    ) -> Iterator[FrameResult]:
        self._tracker.reset()
        for idx, frame in enumerate(frames):
            yield self._process_single_frame(frame, idx)

    def _process_single_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        if frame_idx % self._config.detect_every_n == 0:
            detections = self._detector.detect(frame)
            tracks = self._tracker.update(detections)
        else:
            detections = []
            tracks = self._tracker.predict_only()

        anonymized = self._anonymizer.anonymize(frame, tracks)
        return FrameResult(
            frame_index=frame_idx,
            detections=detections,
            tracks=tracks,
            anonymized_frame=anonymized,
        )

    @property
    def config(self) -> EngineConfig:
        return self._config
