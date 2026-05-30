import numpy as np

from faceanon.datatypes import Detection
from faceanon.tracker import KalmanBoxTracker, SORTTracker
from faceanon.config import TrackerConfig


class TestKalmanBoxTracker:
    def setup_method(self):
        KalmanBoxTracker.reset_counter()

    def test_init_and_predict(self):
        bbox = np.array([10, 10, 50, 50], dtype=np.float32)
        t = KalmanBoxTracker(bbox)
        assert t.id == 1
        pred = t.predict()
        assert pred.shape == (4,)
        # predicted box should be close to initial
        assert abs(pred[0] - 10) < 5
        assert abs(pred[2] - 50) < 5

    def test_update(self):
        bbox = np.array([10, 10, 50, 50], dtype=np.float32)
        t = KalmanBoxTracker(bbox)
        t.predict()
        t.update(np.array([12, 12, 52, 52], dtype=np.float32))
        assert t.time_since_update == 0
        assert t.hits == 2


class TestSORTTracker:
    def setup_method(self):
        KalmanBoxTracker.reset_counter()

    def test_new_detections_create_tracks(self):
        tracker = SORTTracker(TrackerConfig(min_hits=1, max_age=5))
        dets = [
            Detection(bbox=np.array([10, 10, 50, 50], dtype=np.float32), score=0.9),
            Detection(bbox=np.array([100, 100, 150, 150], dtype=np.float32), score=0.8),
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == 2

    def test_tracking_continuity(self):
        tracker = SORTTracker(TrackerConfig(min_hits=2, max_age=5, iou_threshold=0.2))
        det1 = [Detection(bbox=np.array([10, 10, 50, 50], dtype=np.float32), score=0.9)]
        tracker.update(det1)

        det2 = [Detection(bbox=np.array([12, 12, 52, 52], dtype=np.float32), score=0.9)]
        tracks = tracker.update(det2)
        confirmed = [t for t in tracks if t.state == "confirmed"]
        assert len(confirmed) == 1
        assert confirmed[0].hits == 2

    def test_predict_only(self):
        tracker = SORTTracker(TrackerConfig(min_hits=1, max_age=5))
        det = [Detection(bbox=np.array([10, 10, 50, 50], dtype=np.float32), score=0.9)]
        tracker.update(det)
        tracks = tracker.predict_only()
        assert len(tracks) == 1
        assert tracks[0].time_since_update == 1

    def test_track_deletion(self):
        tracker = SORTTracker(TrackerConfig(min_hits=1, max_age=2))
        det = [Detection(bbox=np.array([10, 10, 50, 50], dtype=np.float32), score=0.9)]
        tracker.update(det)
        # no detections for max_age + 1 frames
        for _ in range(3):
            tracks = tracker.update([])
        # track should be deleted
        assert len(tracks) == 0

    def test_reset(self):
        tracker = SORTTracker()
        det = [Detection(bbox=np.array([10, 10, 50, 50], dtype=np.float32), score=0.9)]
        tracker.update(det)
        tracker.reset()
        tracks = tracker.predict_only()
        assert len(tracks) == 0
