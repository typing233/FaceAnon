"""Tests for detector - require the ONNX model, so mark as integration."""
import pytest
import numpy as np


@pytest.fixture
def detector():
    """Create detector - skips if model cannot be downloaded."""
    try:
        from faceanon.detector import CenterFaceDetector
        from faceanon.config import DetectorConfig
        return CenterFaceDetector(DetectorConfig())
    except Exception as e:
        pytest.skip(f"Cannot load detector: {e}")


class TestCenterFaceDetector:
    def test_detect_returns_list(self, detector):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(img)
        assert isinstance(result, list)

    def test_detect_on_blank_image(self, detector):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(img)
        # blank image should have no faces
        assert len(result) == 0

    def test_detection_fields(self, detector):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(img)
        for det in result:
            assert det.bbox.shape == (4,)
            assert 0 <= det.score <= 1.0
