import numpy as np

from faceanon.utils import iou_matrix, nms


class TestNMS:
    def test_empty(self):
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty(0, dtype=np.float32)
        assert nms(boxes, scores, 0.5) == []

    def test_single_box(self):
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        assert nms(boxes, scores, 0.5) == [0]

    def test_overlapping_boxes(self):
        boxes = np.array(
            [[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = nms(boxes, scores, 0.3)
        assert 0 in keep
        assert 2 in keep
        assert 1 not in keep

    def test_non_overlapping(self):
        boxes = np.array(
            [[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        assert len(nms(boxes, scores, 0.5)) == 2


class TestIOUMatrix:
    def test_empty(self):
        a = np.empty((0, 4), dtype=np.float32)
        b = np.array([[0, 0, 10, 10]], dtype=np.float32)
        result = iou_matrix(a, b)
        assert result.shape == (0, 1)

    def test_identical(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        result = iou_matrix(a, a)
        assert abs(result[0, 0] - 1.0) < 1e-5

    def test_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 20, 30, 30]], dtype=np.float32)
        result = iou_matrix(a, b)
        assert result[0, 0] < 1e-5

    def test_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[5, 5, 15, 15]], dtype=np.float32)
        result = iou_matrix(a, b)
        expected = 25.0 / (100 + 100 - 25)
        assert abs(result[0, 0] - expected) < 1e-5
