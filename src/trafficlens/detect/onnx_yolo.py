"""ONNX Runtime YOLO11 detector adapter -- runs the exact ONNX graph the
browser engine (TypeScript + onnxruntime-web, Task 20) will also run,
through the exact same shared ``letterbox``/``decode_yolo`` path as
``trafficlens.detect.ultralytics_yolo.UltralyticsDetector`` (see that
module's docstring for the full "both adapters agree" argument).

``onnxruntime`` is imported lazily, INSIDE ``OnnxDetector.__init__`` --
never at module import time -- so that
``import trafficlens.detect.onnx_yolo`` does not pull onnxruntime onto a
machine that only has the core install. See ``tests/test_detect.py::
test_importing_onnx_adapter_module_does_not_import_onnxruntime_at_top_level``.

A YOLO11 ONNX export (``ultralytics``'s own ``model.export(format="onnx")``,
no ``nms=True``) takes a ``(1, 3, size, size)`` float32 input and returns a
single ``(1, 4 + n_classes, N)`` output -- no built-in NMS, no
preprocessing baked into the graph -- so this adapter supplies both, using
the exact same ``trafficlens.detect.base`` functions the ultralytics
adapter and the TypeScript mirror use.
"""

from __future__ import annotations

import numpy as np

from trafficlens.core.classes import VEHICLE_CLASSES, class_ids
from trafficlens.core.constants import (
    DETECT_DEFAULT_CONF,
    DETECT_DEFAULT_INPUT_SIZE,
    DETECT_DEFAULT_NMS_IOU,
)
from trafficlens.detect.base import Detection, decode_yolo, letterbox


class OnnxDetector:
    """Detector backed by an onnxruntime session running a YOLO11 ONNX
    export, decoded through the shared ``letterbox``/``decode_yolo``
    path."""

    def __init__(
        self,
        weights: str,
        *,
        size: int = DETECT_DEFAULT_INPUT_SIZE,
        conf: float = DETECT_DEFAULT_CONF,
        iou: float = DETECT_DEFAULT_NMS_IOU,
        classes: tuple[str, ...] = VEHICLE_CLASSES,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            weights, providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.size = size
        self.conf = conf
        self.iou = iou
        self.keep_class_ids = class_ids(classes)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        chw, scale, pad_x, pad_y = letterbox(frame, self.size)
        (preds_np,) = self._session.run(None, {self._input_name: chw})
        return decode_yolo(
            preds_np, scale, pad_x, pad_y,
            conf=self.conf, iou=self.iou, keep_class_ids=self.keep_class_ids,
        )
