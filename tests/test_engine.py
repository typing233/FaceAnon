"""Tests for the FaceAnonEngine integration."""
import pytest
import numpy as np


@pytest.fixture
def engine():
    try:
        from faceanon import FaceAnonEngine
        return FaceAnonEngine()
    except Exception as e:
        pytest.skip(f"Cannot create engine: {e}")


class TestFaceAnonEngine:
    def test_process_image(self, engine):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = engine.process_image(img)
        assert result.anonymized_frame is not None
        assert result.anonymized_frame.shape == img.shape
        assert result.frame_index == 0

    def test_process_video_frames(self, engine):
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
        results = list(engine.process_video_frames(iter(frames)))
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.frame_index == i
            assert r.anonymized_frame is not None
