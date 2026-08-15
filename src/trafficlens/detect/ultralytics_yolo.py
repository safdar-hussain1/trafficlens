"""Ultralytics YOLO11 detector adapter for the server pipeline.

``torch`` and ``ultralytics`` are imported lazily, INSIDE
``UltralyticsDetector.__init__`` -- never at module import time -- so that
``import trafficlens.detect.ultralytics_yolo`` (and every module that
imports it, including ``trafficlens.detect`` if it ever re-exported this
adapter) does not pull torch onto a machine that only has the core
install. See ``tests/test_detect.py::
test_importing_ultralytics_adapter_module_does_not_import_torch_at_top_level``.

Route taken for "both adapters must agree": the frame is letterboxed with
``trafficlens.detect.base.letterbox`` (the exact function the browser
TypeScript mirror runs in Task 20), the checkpoint's own
``DetectionModel`` is called DIRECTLY -- bypassing ultralytics' own
``Predictor`` preprocessing *and* its own NMS/postprocessing entirely --
to obtain a raw ``(1, 84, N)`` prediction tensor, and that tensor is handed
to ``trafficlens.detect.base.decode_yolo``, the exact function
``trafficlens.detect.onnx_yolo.OnnxDetector`` also calls. This makes the
two adapters structurally identical: they differ only in how they obtain
the raw tensor (a torch forward pass vs. an onnxruntime session run),
never in how that tensor becomes ``Detection``s. This was verified,
not merely asserted: a raw torch forward pass and the same checkpoint's
ONNX export, run on the same letterboxed frame, agree to a max absolute
difference of ~0.0096 (mean ~2e-6) across the full 84x8400 output tensor
-- see ``tests/test_detect.py::
test_ultralytics_and_onnx_adapters_agree_on_a_real_frame`` for the
end-to-end Detection-level check on a real video frame.

In non-export inference mode, ``DetectionModel.forward`` returns a
``(preds, raw_features)`` tuple where ``preds`` already has shape
``(1, 4 + n_classes, N)`` -- decoded box coordinates and sigmoid class
scores, concatenated across all three stride heads -- which is exactly
the layout ``decode_yolo`` expects and exactly what the model's own ONNX
export produces as its sole output.
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


class UltralyticsDetector:
    """Detector backed by an ultralytics YOLO11 ``.pt`` checkpoint, run
    directly on torch and decoded through the shared
    ``letterbox``/``decode_yolo`` path (see module docstring)."""

    def __init__(
        self,
        weights: str,
        *,
        size: int = DETECT_DEFAULT_INPUT_SIZE,
        conf: float = DETECT_DEFAULT_CONF,
        iou: float = DETECT_DEFAULT_NMS_IOU,
        classes: tuple[str, ...] = VEHICLE_CLASSES,
        device: str = "cpu",
    ) -> None:
        import torch
        from ultralytics import YOLO

        self._torch = torch
        yolo = YOLO(weights)
        self._model = yolo.model.to(device).eval()
        self._device = device
        self.size = size
        self.conf = conf
        self.iou = iou
        self.keep_class_ids = class_ids(classes)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        chw, scale, pad_x, pad_y = letterbox(frame, self.size)
        tensor = self._torch.from_numpy(chw).to(self._device)
        with self._torch.no_grad():
            raw = self._model(tensor)
        # Non-export inference returns (preds, raw_features); export mode
        # (not used here) would return preds directly -- handle both so a
        # caller who passes an already-export-mode model still works.
        preds = raw[0] if isinstance(raw, (tuple, list)) else raw
        preds_np = preds.detach().cpu().numpy()
        return decode_yolo(preds_np, scale, pad_x, pad_y, self.conf, self.iou, self.keep_class_ids)
