from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .config import DetectorConfig
from .datatypes import Detection
from .utils import ensure_model, nms


class CenterFaceDetector:
    def __init__(self, config: DetectorConfig | None = None, use_gpu: bool = False):
        self._config = config or DetectorConfig()
        model_path = ensure_model(self._config.model_path)

        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers.insert(0, "CUDAExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.enable_mem_pattern = False
        self._session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._model_input_shape = self._session.get_inputs()[0].shape

    def detect(self, image: np.ndarray) -> list[Detection]:
        orig_h, orig_w = image.shape[:2]
        blob, scale_w, scale_h = self._preprocess(image)
        outputs = self._session.run(self._output_names, {self._input_name: blob})
        return self._postprocess(outputs, scale_w, scale_h, orig_h, orig_w)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        target_w, target_h = self._config.input_size
        orig_h, orig_w = image.shape[:2]

        import cv2

        resized = cv2.resize(image, (target_w, target_h))
        blob = resized.astype(np.float32)
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        return blob, scale_w, scale_h

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        scale_w: float,
        scale_h: float,
        orig_h: int,
        orig_w: int,
    ) -> list[Detection]:
        heatmap = outputs[0][0, 0]  # (H/4, W/4)
        scale = outputs[1][0]  # (2, H/4, W/4)
        offset = outputs[2][0]  # (2, H/4, W/4)
        landmarks_raw = outputs[3][0]  # (10, H/4, W/4)

        feat_h, feat_w = heatmap.shape
        stride = self._config.input_size[1] / feat_h

        mask = heatmap > self._config.score_threshold
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return []

        scores = heatmap[rows, cols]
        ox = offset[0, rows, cols]
        oy = offset[1, rows, cols]
        sh = scale[0, rows, cols]  # height (log-space)
        sw = scale[1, rows, cols]  # width (log-space)

        cx = (cols + oy + 0.5) * stride
        cy = (rows + ox + 0.5) * stride
        w = np.exp(sw) * stride
        h = np.exp(sh) * stride

        x1 = (cx - w / 2) / scale_w
        y1 = (cy - h / 2) / scale_h
        x2 = (cx + w / 2) / scale_w
        y2 = (cy + h / 2) / scale_h

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(boxes, scores, self._config.nms_threshold)

        detections: list[Detection] = []
        for i in keep:
            lms = None
            if landmarks_raw is not None:
                lm_x = landmarks_raw[0:10:2, rows[i], cols[i]]
                lm_y = landmarks_raw[1:10:2, rows[i], cols[i]]
                lm_x = (cols[i] + lm_x + 0.5) * stride / scale_w
                lm_y = (rows[i] + lm_y + 0.5) * stride / scale_h
                lms = np.stack([lm_x, lm_y], axis=1)  # (5, 2)

            detections.append(
                Detection(bbox=boxes[i], score=float(scores[i]), landmarks=lms)
            )
        return detections
