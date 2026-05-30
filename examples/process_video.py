"""Example: process a video file."""
import sys

import cv2

import faceanon


def main():
    if len(sys.argv) < 3:
        print("Usage: python process_video.py <input.mp4> <output.mp4>")
        return

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    config = faceanon.EngineConfig(
        anonymizer=faceanon.AnonymizerConfig(
            method=faceanon.AnonymizationType.MOSAIC,
            intensity=0.9,
        ),
        detect_every_n=3,
    )
    engine = faceanon.FaceAnonEngine(config)

    def progress(idx, total, result):
        if total > 0:
            pct = (idx + 1) / total * 100
            n_tracks = sum(1 for t in result.tracks if t.state == "confirmed")
            print(f"\r[{pct:5.1f}%] frame {idx+1}/{total}, tracking {n_tracks} face(s)", end="")

    engine.process_video(input_path, output_path, callback=progress)
    print("\nDone.")


if __name__ == "__main__":
    main()
