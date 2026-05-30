from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .batch import BatchConfig, BatchProcessor
from .config import (
    AnonymizerConfig,
    AnonymizationType,
    DetectorConfig,
    EngineConfig,
    TrackerConfig,
)
from .engine import FaceAnonEngine
from .realtime import RealtimeConfig, RealtimeProcessor


def resolve_model_path(given: str) -> str:
    if os.path.isabs(given) and os.path.exists(given):
        return given
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent.parent))
    resolved = base / given
    if resolved.exists():
        return str(resolved)
    return given


def build_engine_config(args: argparse.Namespace) -> EngineConfig:
    model_path = resolve_model_path(args.model)

    w, h = 640, 640
    if args.input_size:
        parts = args.input_size.lower().split("x")
        if len(parts) == 2:
            w, h = int(parts[0]), int(parts[1])

    detector = DetectorConfig(
        model_path=model_path,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        input_size=(w, h),
    )
    tracker = TrackerConfig()
    anonymizer = AnonymizerConfig(
        method=AnonymizationType(args.method),
        intensity=args.intensity,
        expand_ratio=args.expand_ratio,
    )
    return EngineConfig(
        detector=detector,
        tracker=tracker,
        anonymizer=anonymizer,
        detect_every_n=args.detect_every_n,
        use_gpu=args.gpu,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faceanon",
        description="Face anonymization tool - detect, track and blur faces in images and videos",
    )

    common = parser.add_argument_group("engine options")
    common.add_argument(
        "--model", default="models/centerface.onnx", help="path to ONNX model"
    )
    common.add_argument(
        "--score-threshold", type=float, default=0.5, help="detection confidence threshold"
    )
    common.add_argument(
        "--nms-threshold", type=float, default=0.3, help="NMS IoU threshold"
    )
    common.add_argument(
        "--input-size", default="640x640", help="detector input size (WxH)"
    )
    common.add_argument(
        "--method",
        choices=["gaussian_blur", "mosaic"],
        default="gaussian_blur",
        help="anonymization method",
    )
    common.add_argument(
        "--intensity", type=float, default=0.8, help="anonymization intensity (0-1)"
    )
    common.add_argument(
        "--expand-ratio", type=float, default=0.1, help="bbox expansion ratio"
    )
    common.add_argument(
        "--detect-every-n", type=int, default=1, help="run detection every N frames"
    )
    common.add_argument("--gpu", action="store_true", help="enable GPU acceleration")

    sub = parser.add_subparsers(dest="command")

    # realtime
    rt = sub.add_parser("realtime", help="real-time camera/stream processing")
    rt.add_argument(
        "--source", default="0", help="camera index (int) or stream URL (default: 0)"
    )
    rt.add_argument("--output", default=None, help="optional output video file")
    rt.add_argument("--no-display", action="store_true", help="disable display window")
    rt.add_argument("--max-fps", type=float, default=0.0, help="cap framerate (0=unlimited)")

    # batch
    bt = sub.add_parser("batch", help="batch process a directory of media files")
    bt.add_argument("--input-dir", required=True, help="input directory")
    bt.add_argument("--output-dir", required=True, help="output directory")
    bt.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    bt.add_argument("--skip-existing", action="store_true", help="skip already processed files")

    # image
    img = sub.add_parser("image", help="process a single image")
    img.add_argument("--input", required=True, help="input image path")
    img.add_argument("--output", required=True, help="output image path")

    # video
    vid = sub.add_parser("video", help="process a single video")
    vid.add_argument("--input", required=True, help="input video path")
    vid.add_argument("--output", required=True, help="output video path")

    return parser


def cmd_realtime(args: argparse.Namespace, engine_config: EngineConfig) -> int:
    source: int | str = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    config = RealtimeConfig(
        source=source,
        output_path=args.output,
        display=not args.no_display,
        max_fps=args.max_fps,
    )
    processor = RealtimeProcessor(engine_config, config)
    try:
        processor.run()
    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_batch(args: argparse.Namespace, engine: FaceAnonEngine) -> int:
    config = BatchConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        skip_existing=args.skip_existing,
    )
    processor = BatchProcessor(engine, config)
    try:
        result = processor.run()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(result.summary())
    return 0 if result.failures == 0 else 1


def cmd_image(args: argparse.Namespace, engine: FaceAnonEngine) -> int:
    import cv2

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: cannot read image: {args.input}", file=sys.stderr)
        return 1

    result = engine.process_image(image)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, result.anonymized_frame)
    print(f"Saved: {args.output} ({len(result.detections)} faces detected)")
    return 0


def cmd_video(args: argparse.Namespace, engine: FaceAnonEngine) -> int:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    progress_bar = None

    def _callback(frame_idx: int, total: int, _result) -> None:
        nonlocal progress_bar
        if tqdm and progress_bar is None and total > 0:
            progress_bar = tqdm(total=total, desc="Processing", unit="frame")
        if progress_bar is not None:
            progress_bar.update(1)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        engine.process_video(args.input, args.output, _callback)
    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(f"Saved: {args.output}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    engine_config = build_engine_config(args)

    if args.command == "realtime":
        return cmd_realtime(args, engine_config)

    engine = FaceAnonEngine(engine_config)

    commands = {
        "batch": cmd_batch,
        "image": cmd_image,
        "video": cmd_video,
    }

    return commands[args.command](args, engine)

    return commands[args.command](args, engine)


if __name__ == "__main__":
    sys.exit(main())
