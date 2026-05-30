"""Example: process a single image."""
import cv2

import faceanon


def main():
    engine = faceanon.FaceAnonEngine()
    image = cv2.imread("image.png")
    if image is None:
        print("Error: cannot read image.png")
        return

    result = engine.process_image(image)
    print(f"Detected {len(result.detections)} face(s)")
    for det in result.detections:
        print(f"  score={det.score:.3f} bbox={det.bbox.astype(int).tolist()}")

    cv2.imwrite("output_image.png", result.anonymized_frame)
    print("Saved to output_image.png")


if __name__ == "__main__":
    main()
