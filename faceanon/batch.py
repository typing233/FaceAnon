from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .engine import FaceAnonEngine

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class BatchConfig:
    input_dir: str
    output_dir: str
    recursive: bool = False
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov")
    skip_existing: bool = False


@dataclass
class BatchResult:
    total_files: int = 0
    successes: int = 0
    failures: int = 0
    skipped: int = 0
    elapsed_seconds: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Batch Processing Summary",
            "=" * 50,
            f"  Total files:  {self.total_files}",
            f"  Successes:    {self.successes}",
            f"  Failures:     {self.failures}",
            f"  Skipped:      {self.skipped}",
            f"  Elapsed:      {self.elapsed_seconds:.1f}s",
            "=" * 50,
        ]
        if self.errors:
            lines.append("Errors:")
            for filepath, msg in self.errors:
                lines.append(f"  {filepath}: {msg}")
        return "\n".join(lines)


class BatchProcessor:
    def __init__(self, engine: FaceAnonEngine, config: BatchConfig):
        self._engine = engine
        self._config = config

    def run(self) -> BatchResult:
        result = BatchResult()
        start = time.time()

        files = self._discover_files()
        result.total_files = len(files)

        if not files:
            result.elapsed_seconds = time.time() - start
            return result

        output_base = Path(self._config.output_dir)
        output_base.mkdir(parents=True, exist_ok=True)
        input_base = Path(self._config.input_dir)

        iterator = tqdm(files, desc="Processing", unit="file") if tqdm else files

        for filepath in iterator:
            rel = filepath.relative_to(input_base)
            out_path = output_base / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if self._config.skip_existing and out_path.exists():
                result.skipped += 1
                continue

            ext = filepath.suffix.lower()
            try:
                if ext in self._config.image_extensions:
                    self._process_image(filepath, out_path)
                elif ext in self._config.video_extensions:
                    self._process_video(filepath, out_path)
                result.successes += 1
            except Exception as e:
                result.failures += 1
                result.errors.append((str(filepath), str(e)))

        result.elapsed_seconds = time.time() - start
        return result

    def _discover_files(self) -> list[Path]:
        input_path = Path(self._config.input_dir)
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self._config.input_dir}")

        all_extensions = set(
            self._config.image_extensions + self._config.video_extensions
        )
        files: list[Path] = []

        entries = input_path.rglob("*") if self._config.recursive else input_path.iterdir()
        for p in sorted(entries):
            if p.is_file() and p.suffix.lower() in all_extensions:
                files.append(p)

        return files

    def _process_image(self, input_path: Path, output_path: Path) -> None:
        image = cv2.imread(str(input_path))
        if image is None:
            raise IOError(f"Cannot read image: {input_path}")
        result = self._engine.process_image(image)
        cv2.imwrite(str(output_path), result.anonymized_frame)

    def _process_video(self, input_path: Path, output_path: Path) -> None:
        progress_bar = None

        def _callback(frame_idx: int, total: int, _result) -> None:
            nonlocal progress_bar
            if tqdm and progress_bar is None and total > 0:
                progress_bar = tqdm(
                    total=total,
                    desc=f"  {input_path.name}",
                    unit="frame",
                    leave=False,
                )
            if progress_bar is not None:
                progress_bar.update(1)

        try:
            self._engine.process_video(str(input_path), str(output_path), _callback)
        finally:
            if progress_bar is not None:
                progress_bar.close()
