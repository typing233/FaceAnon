from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .anonymizer import Anonymizer
from .config import EngineConfig
from .detector import CenterFaceDetector
from .tracker import SORTTracker


@dataclass
class RealtimeConfig:
    source: int | str = 0
    output_path: str | None = None
    display: bool = True
    window_name: str = "FaceAnon - Realtime"
    max_fps: float = 0.0


class RealtimeProcessor:
    def __init__(self, engine_config: EngineConfig | None = None, config: RealtimeConfig | None = None):
        self._engine_config = engine_config or EngineConfig()
        self._config = config or RealtimeConfig()
        self._stop_event = threading.Event()
        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None

        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._capture_thread: threading.Thread | None = None

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
        src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        if cfg.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = cfg.max_fps if cfg.max_fps > 0 else src_fps
            self._writer = cv2.VideoWriter(
                cfg.output_path, fourcc, out_fps, (width, height)
            )

        detector = CenterFaceDetector(self._engine_config.detector, use_gpu=self._engine_config.use_gpu)
        tracker = SORTTracker(self._engine_config.tracker)
        anonymizer = Anonymizer(self._engine_config.anonymizer)
        detect_every_n = self._engine_config.detect_every_n

        min_frame_time = 1.0 / cfg.max_fps if cfg.max_fps > 0 else 0.0
        fps_counter = _FPSCounter()

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                if not self._frame_ready.wait(timeout=1.0):
                    if self._stop_event.is_set():
                        break
                    continue

                with self._frame_lock:
                    frame = self._latest_frame
                    self._latest_frame = None
                    self._frame_ready.clear()

                if frame is None:
                    break

                loop_start = time.perf_counter()

                if frame_idx % detect_every_n == 0:
                    detections = detector.detect(frame)
                    tracks = tracker.update(detections)
                else:
                    tracks = tracker.predict_only()

                anonymized = anonymizer.anonymize(frame, tracks)
                frame_idx += 1
                fps_counter.tick()

                if cfg.display:
                    display_frame = self._draw_overlay(
                        anonymized, fps_counter.fps, len(tracks)
                    )
                    cv2.imshow(cfg.window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                if self._writer is not None:
                    self._writer.write(anonymized)

                if min_frame_time > 0:
                    elapsed = time.perf_counter() - loop_start
                    if elapsed < min_frame_time:
                        time.sleep(min_frame_time - elapsed)
        finally:
            self._stop_event.set()
            self._cleanup()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._stop_event.set()
                with self._frame_lock:
                    self._latest_frame = None
                self._frame_ready.set()
                break
            with self._frame_lock:
                self._latest_frame = frame
            self._frame_ready.set()

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
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
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
