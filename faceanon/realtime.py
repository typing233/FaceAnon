from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from .engine import FaceAnonEngine


@dataclass
class RealtimeConfig:
    source: int | str = 0
    output_path: str | None = None
    display: bool = True
    window_name: str = "FaceAnon - Realtime"
    max_fps: float = 0.0


class RealtimeProcessor:
    def __init__(self, engine: FaceAnonEngine, config: RealtimeConfig | None = None):
        self._engine = engine
        self._config = config or RealtimeConfig()
        self._stop_event = threading.Event()
        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        cfg = self._config
        self._cap = cv2.VideoCapture(cfg.source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open source: {cfg.source}")

        if isinstance(cfg.source, str):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        if cfg.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                cfg.output_path, fourcc, fps, (width, height)
            )

        min_frame_time = 1.0 / cfg.max_fps if cfg.max_fps > 0 else 0.0
        fps_counter = _FPSCounter()

        try:
            for result in self._engine.process_video_frames(self._frame_generator()):
                frame = result.anonymized_frame
                if frame is None:
                    continue

                fps_counter.tick()

                if cfg.display:
                    display_frame = self._draw_overlay(
                        frame, fps_counter.fps, len(result.tracks)
                    )
                    cv2.imshow(cfg.window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                if self._writer is not None:
                    self._writer.write(frame)

                if min_frame_time > 0:
                    elapsed = fps_counter.last_elapsed
                    if elapsed < min_frame_time:
                        time.sleep(min_frame_time - elapsed)
        finally:
            self._cleanup()

    def _frame_generator(self) -> Iterator[np.ndarray]:
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def _draw_overlay(
        self, frame: np.ndarray, fps: float, face_count: int
    ) -> np.ndarray:
        display = frame.copy()
        text = f"FPS: {fps:.1f} | Faces: {face_count}"
        cv2.putText(
            display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        return display

    def _cleanup(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._config.display:
            cv2.destroyAllWindows()


class _FPSCounter:
    def __init__(self, window: int = 30):
        self._window = window
        self._times: list[float] = []
        self._last_time: float = time.perf_counter()
        self.fps: float = 0.0
        self.last_elapsed: float = 0.0

    def tick(self) -> None:
        now = time.perf_counter()
        self.last_elapsed = now - self._last_time
        self._last_time = now
        self._times.append(now)
        if len(self._times) > self._window:
            self._times = self._times[-self._window:]
        if len(self._times) >= 2:
            self.fps = (len(self._times) - 1) / (self._times[-1] - self._times[0])
