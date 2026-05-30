from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackerConfig
from .datatypes import Detection, Track
from .utils import iou_matrix


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.array([cx, cy, w, h], dtype=np.float32)


def _z_to_bbox(z: np.ndarray) -> np.ndarray:
    cx, cy, w, h = z[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class KalmanBoxTracker:
    _id_counter: int = 0

    def __init__(self, bbox: np.ndarray):
        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter

        self._kf = cv2.KalmanFilter(8, 4, 0)
        self._kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._kf.transitionMatrix[i, i + 4] = 1.0

        self._kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self._kf.measurementMatrix[i, i] = 1.0

        self._kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self._kf.processNoiseCov[4:, 4:] *= 10.0
        self._kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self._kf.errorCovPost = np.eye(8, dtype=np.float32)
        self._kf.errorCovPost[4:, 4:] *= 100.0

        z = _bbox_to_z(bbox)
        state = np.zeros((8, 1), dtype=np.float32)
        state[:4, 0] = z
        self._kf.statePost = state

        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        state = self._kf.predict()
        self.age += 1
        self.time_since_update += 1
        return _z_to_bbox(state[:, 0])

    def update(self, bbox: np.ndarray) -> None:
        z = _bbox_to_z(bbox).reshape(4, 1)
        self._kf.correct(z)
        self.hits += 1
        self.time_since_update = 0

    def get_bbox(self) -> np.ndarray:
        return _z_to_bbox(self._kf.statePost[:, 0])

    @classmethod
    def reset_counter(cls) -> None:
        cls._id_counter = 0


class SORTTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self._config = config or TrackerConfig()
        self._trackers: list[KalmanBoxTracker] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        predicted_boxes = np.array(
            [t.predict() for t in self._trackers], dtype=np.float32
        )

        det_boxes = (
            np.array([d.bbox for d in detections], dtype=np.float32)
            if detections
            else np.empty((0, 4), dtype=np.float32)
        )

        matched, unmatched_dets, unmatched_trks = self._assign(
            predicted_boxes, det_boxes
        )

        for t_idx, d_idx in matched:
            self._trackers[t_idx].update(det_boxes[d_idx])

        for d_idx in unmatched_dets:
            self._trackers.append(KalmanBoxTracker(det_boxes[d_idx]))

        tracks: list[Track] = []
        alive: list[KalmanBoxTracker] = []
        for t in self._trackers:
            if t.time_since_update > self._config.max_age:
                continue
            alive.append(t)
            state = "confirmed" if t.hits >= self._config.min_hits else "tentative"
            if t.time_since_update > 0:
                state = "lost"
            tracks.append(
                Track(
                    track_id=t.id,
                    bbox=t.get_bbox(),
                    state=state,
                    age=t.age,
                    hits=t.hits,
                    time_since_update=t.time_since_update,
                )
            )
        self._trackers = alive
        return tracks

    def predict_only(self) -> list[Track]:
        tracks: list[Track] = []
        alive: list[KalmanBoxTracker] = []
        for t in self._trackers:
            t.predict()
            if t.time_since_update > self._config.max_age:
                continue
            alive.append(t)
            state = "confirmed" if t.hits >= self._config.min_hits else "tentative"
            if t.time_since_update > 0:
                state = "lost"
            tracks.append(
                Track(
                    track_id=t.id,
                    bbox=t.get_bbox(),
                    state=state,
                    age=t.age,
                    hits=t.hits,
                    time_since_update=t.time_since_update,
                )
            )
        self._trackers = alive
        return tracks

    def reset(self) -> None:
        self._trackers.clear()
        KalmanBoxTracker.reset_counter()

    def _assign(
        self, predicted: np.ndarray, detections: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if len(predicted) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(predicted)))

        iou_mat = iou_matrix(predicted, detections)
        cost = 1.0 - iou_mat
        row_indices, col_indices = linear_sum_assignment(cost)

        matched: list[tuple[int, int]] = []
        unmatched_dets = set(range(len(detections)))
        unmatched_trks = set(range(len(predicted)))

        for r, c in zip(row_indices, col_indices):
            if iou_mat[r, c] < self._config.iou_threshold:
                continue
            matched.append((r, c))
            unmatched_dets.discard(c)
            unmatched_trks.discard(r)

        return matched, sorted(unmatched_dets), sorted(unmatched_trks)
