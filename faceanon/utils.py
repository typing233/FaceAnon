from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://raw.githubusercontent.com/Star-Clouds/CenterFace/"
    "master/models/onnx/centerface_bnmerged.onnx"
)


def ensure_model(model_path: str) -> str:
    path = Path(model_path)
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CenterFace model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, str(path))
    _fix_model_dynamic_input(str(path))
    print("Download complete.")
    return str(path)


def _fix_model_dynamic_input(model_path: str) -> None:
    try:
        import onnx
    except ImportError:
        return

    model = onnx.load(model_path)
    needs_fix = False
    for inp in model.graph.input:
        if inp.name == "input.1":
            dims = inp.type.tensor_type.shape.dim
            if len(dims) >= 4 and dims[2].dim_value > 0:
                needs_fix = True
            break

    if not needs_fix:
        return

    for inp in model.graph.input:
        if inp.name == "input.1":
            dims = inp.type.tensor_type.shape.dim
            dims[0].dim_param = "batch"
            dims[0].ClearField("dim_value")
            dims[2].dim_param = "height"
            dims[2].ClearField("dim_value")
            dims[3].dim_param = "width"
            dims[3].ClearField("dim_value")
            break

    for out in model.graph.output:
        for dim in out.type.tensor_type.shape.dim:
            dim.dim_param = "dynamic"
            dim.ClearField("dim_value")

    initializer_names = {init.name for init in model.graph.initializer}
    inputs_to_keep = [
        inp for inp in model.graph.input if inp.name not in initializer_names
    ]
    del model.graph.input[:]
    model.graph.input.extend(inputs_to_keep)

    onnx.save(model, model_path)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    return keep


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)
