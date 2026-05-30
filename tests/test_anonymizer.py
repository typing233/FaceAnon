import numpy as np

from faceanon.anonymizer import Anonymizer
from faceanon.config import AnonymizerConfig, AnonymizationType
from faceanon.datatypes import Track


def _make_image(h=200, w=300):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_track(x1=50, y1=50, x2=100, y2=100, state="confirmed"):
    return Track(
        track_id=1,
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        state=state,
        hits=3,
    )


class TestAnonymizer:
    def test_gaussian_blur(self):
        config = AnonymizerConfig(method=AnonymizationType.GAUSSIAN_BLUR, intensity=0.8)
        anon = Anonymizer(config)
        img = _make_image()
        track = _make_track()
        result = anon.anonymize(img, [track])
        assert result.shape == img.shape
        # the ROI should be different from original
        assert not np.array_equal(result[50:100, 50:100], img[50:100, 50:100])

    def test_mosaic(self):
        config = AnonymizerConfig(method=AnonymizationType.MOSAIC, intensity=0.9)
        anon = Anonymizer(config)
        img = _make_image()
        track = _make_track()
        result = anon.anonymize(img, [track])
        assert result.shape == img.shape
        assert not np.array_equal(result[50:100, 50:100], img[50:100, 50:100])

    def test_skips_non_confirmed(self):
        anon = Anonymizer()
        img = _make_image()
        track = _make_track(state="tentative")
        result = anon.anonymize(img, [track])
        assert np.array_equal(result, img)

    def test_intensity_affects_output(self):
        img = _make_image()
        track = _make_track()
        result_low = Anonymizer(AnonymizerConfig(intensity=0.2)).anonymize(img, [track])
        result_high = Anonymizer(AnonymizerConfig(intensity=0.9)).anonymize(img, [track])
        # both should differ from original
        assert not np.array_equal(result_low[50:100, 50:100], img[50:100, 50:100])
        assert not np.array_equal(result_high[50:100, 50:100], img[50:100, 50:100])

    def test_does_not_modify_input(self):
        anon = Anonymizer()
        img = _make_image()
        original = img.copy()
        anon.anonymize(img, [_make_track()])
        assert np.array_equal(img, original)

    def test_expand_ratio(self):
        config = AnonymizerConfig(expand_ratio=0.2)
        anon = Anonymizer(config)
        img = _make_image()
        track = _make_track(x1=50, y1=50, x2=100, y2=100)
        result = anon.anonymize(img, [track])
        # pixels outside original bbox but within expand should also be affected
        assert not np.array_equal(result[45:105, 45:105], img[45:105, 45:105])
