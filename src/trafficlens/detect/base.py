"""Shared, torch-free preprocessing, decoding and NMS for YOLO11 detectors.

Imports only the standard library, numpy and OpenCV -- no torch, no
ultralytics, no onnxruntime -- so this module (and the core test suite)
can be exercised on a machine with none of those installed. Both detector
adapters (``trafficlens.detect.ultralytics_yolo``,
``trafficlens.detect.onnx_yolo``) obtain a raw ``(1, 84, N)`` model output
tensor by whatever means their own backend provides, and hand it to
``decode_yolo`` defined here -- so the two adapters, and later the
TypeScript browser engine, run the exact same math and can only ever
differ in how they obtain the raw tensor.

``letterbox`` is parity-critical: a TypeScript mirror (Task 20) is later
asserted to produce byte-identical tensors from this function's exact
documented rules, so its docstring below is the source of truth for that
port, not just documentation of this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from trafficlens.core.classes import COCO_CLASSES
from trafficlens.core.constants import (
    DETECT_DEFAULT_CONF,
    DETECT_DEFAULT_INPUT_SIZE,
    DETECT_DEFAULT_NMS_IOU,
    LETTERBOX_PAD_VALUE,
)


@dataclass(frozen=True)
class Detection:
    """One decoded, post-NMS detection in ORIGINAL image pixel coordinates
    (never letterboxed/model coordinates -- decode_yolo always undoes the
    letterbox transform before returning)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


class Detector(Protocol):
    """Structural interface every detector adapter satisfies: hand it a
    BGR uint8 frame (the raw cv2.VideoCapture read), get back Detections
    in original-image coordinates."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def letterbox(
    frame: np.ndarray, size: int = DETECT_DEFAULT_INPUT_SIZE
) -> tuple[np.ndarray, float, float, float]:
    """Resize ``frame`` (a BGR uint8 HxWx3 array) to fit inside a
    ``size`` x ``size`` square, preserving aspect ratio, and centre-pad the
    remainder with a constant grey so the whole square is filled.

    Exact rules (this is the specification the TypeScript mirror in Task
    20 is written from -- every detail here matters for byte-identical
    output):

    1. ``h, w = frame.shape[:2]``.
    2. ``scale = min(size / w, size / h)`` -- the single scale factor that
       makes the LARGER of the two resized dimensions exactly ``size``,
       so the resized image fits inside the square without cropping.
    3. The resized dimensions are ``new_w = int(round(w * scale))`` and
       ``new_h = int(round(h * scale))``, using Python's built-in
       ``round()`` -- ROUND-HALF-TO-EVEN ("banker's rounding"), not
       truncation and, critically, NOT ``Math.round``-style round-half-up.
       The two disagree exactly on values ending in ``.5``: e.g.
       ``round(358.5) == 358`` in Python but ``Math.round(358.5) === 359``
       in JavaScript (a real case -- a 1280x717 frame at ``size=640`` gives
       ``scale = 0.5`` and ``717 * 0.5 = 358.5`` exactly). A TypeScript
       mirror MUST implement round-half-to-even explicitly (e.g. check for
       the exact ``.5`` case and round to the nearest even integer) rather
       than calling ``Math.round``, or it will produce an off-by-one pixel
       letterbox on inputs like this one and fail the byte-identical
       parity assertion.
    4. The frame is resized to ``(new_w, new_h)`` with BIT-EXACT bilinear
       interpolation (``cv2.INTER_LINEAR_EXACT``), never plain
       ``cv2.INTER_LINEAR``. The distinction is the whole reason a
       TypeScript mirror is possible at all, and it is not a preference:

       ``cv2.resize(..., INTER_LINEAR)`` is not one algorithm. OpenCV's
       ``hal::resize`` opens with ``CALL_HAL(resize, cv_hal_resize, ...)``,
       so a vendor HAL gets first refusal before any OpenCV code runs, and
       the wheel this project is developed against reports ``Custom HAL:
       YES (carotene, KleidiCV)``. Measured on that build: a faithful port
       of OpenCV's own documented fixed-point ``INTER_LINEAR`` is
       byte-identical to ``cv2`` on 36 of 40 random shapes and on 0/31518
       pixels of each single-axis sweep, but diverges wholesale in a sharp
       band -- both axes downscaling with ``scale_x < 3`` -- and that band
       contains this product's own shape (1280x720 to 480x270 is
       ``scale = 2.667`` on both axes; 34% of pixels differ on a noise
       frame, by up to 2 grey levels). No port can match that, because
       what it would have to match is a vendor's NEON kernel, not a
       specification, and a different wheel is a different answer again.

       ``INTER_LINEAR_EXACT`` is OpenCV's answer to exactly this problem:
       pure integer Q8.8 arithmetic with coefficients computed in
       ``softdouble`` (a software float64, deterministic by construction),
       and the HAL does not claim it. A pure-integer reimplementation
       reproduces it on 0 differing pixels out of 989706 across 240 random
       shapes at 1 and 3 channels, including 0 of 388800 at this product's
       own 1280x720 to 480x270 shape. So this is the interpolation that
       makes "the visitor's browser runs the same detector" a statement
       that survives being checked.
    5. Padding is centred using INTEGER FLOOR division:
       ``pad_x = (size - new_w) // 2`` and ``pad_y = (size - new_h) // 2``.
       When ``size - new_w`` (or ``_h``) is odd, the single extra pixel of
       padding lands on the right/bottom, never the left/top, because only
       the left/top pad is computed and the resized image is placed
       starting exactly at ``(pad_x, pad_y)``.
    6. The ``size`` x ``size`` canvas is filled with the constant
       ``LETTERBOX_PAD_VALUE`` (114, the conventional YOLO letterbox grey)
       in all three channels BEFORE the resized image is written into it
       at ``[pad_y:pad_y+new_h, pad_x:pad_x+new_w]``.
    7. The canvas (BGR, as OpenCV read/produced it) is converted to RGB
       channel order.
    8. The result is transposed from HWC to CHW, cast to float32, and
       divided by 255.0 -- values in ``[0, 1]``.
    9. A leading batch dimension of 1 is added: final shape
       ``(1, 3, size, size)``.

    Returns ``(chw_float32, scale, pad_x, pad_y)``. To map a point in
    model/letterboxed space back to original image space:
    ``x_orig = (x_model - pad_x) / scale``,
    ``y_orig = (y_model - pad_y) / scale``.
    """
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR_EXACT
    )

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    canvas = np.full((size, size, 3), LETTERBOX_PAD_VALUE, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    chw = chw[np.newaxis, ...]

    return chw, float(scale), float(pad_x), float(pad_y)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy non-maximum suppression. ``boxes`` is ``(N, 4)`` in
    ``x1, y1, x2, y2`` order (any coordinate space; NMS is scale-agnostic),
    ``scores`` is ``(N,)``. Returns the KEPT indices into the original
    arrays, ordered by descending score.

    Determinism / tie-break rule (so a TypeScript mirror can match this
    exactly): boxes are visited in order of descending score, with ties
    broken by ascending ORIGINAL index -- i.e. a numpy STABLE sort
    (``kind="stable"``) of ``-scores``. Two boxes with an identical score
    therefore always resolve the same way regardless of array layout,
    hashing, or platform: the earlier (lower-index) one is considered
    first and, if the two overlap above ``iou``, is the one kept.
    """
    n = len(scores)
    if n == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(n, dtype=bool)
    keep: list[int] = []

    for pos in range(len(order)):
        i = order[pos]
        if suppressed[i]:
            continue
        keep.append(int(i))

        rest = order[pos + 1 :]
        rest = rest[~suppressed[rest]]
        if len(rest) == 0:
            continue

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou_vals = np.where(union > 0, inter / union, 0.0)
        suppressed[rest[iou_vals > iou]] = True

    return keep


def decode_yolo(
    output: np.ndarray,
    scale: float,
    pad_x: float,
    pad_y: float,
    *,
    conf: float = DETECT_DEFAULT_CONF,
    iou: float = DETECT_DEFAULT_NMS_IOU,
    keep_class_ids: set[int],
) -> list[Detection]:
    """Decode a raw YOLO11 output tensor into ``Detection``s in ORIGINAL
    image coordinates.

    ``output`` has shape ``(1, 4 + n_classes, N)``: rows 0-3 are
    ``cx, cy, w, h`` in letterboxed MODEL coordinates (the same
    ``size`` x ``size`` square ``letterbox()`` produced -- ``scale``,
    ``pad_x``, ``pad_y`` must be the exact values that call returned), rows
    ``4..4+n_classes`` are per-class scores already through a sigmoid
    (YOLO11 has no separate objectness row, unlike YOLOv5/v7). Class index
    ``k`` (0-based within the score rows) names ``COCO_CLASSES[k]``.

    Steps: transpose to ``(N, 4 + n_classes)``, take each row's max class
    score/id, threshold at ``conf`` INCLUSIVE (``score >= conf`` survives;
    a score exactly equal to ``conf`` is kept, mirroring ``nms``'s
    documented strict ``>`` for suppression), filter to ``keep_class_ids``, convert
    the surviving boxes from letterboxed ``cx, cy, w, h`` to
    ``x1, y1, x2, y2`` in ORIGINAL image coordinates by undoing the pad
    then the scale (``(v - pad) / scale``), then run class-wise ``nms``
    (each class suppressed independently, so an overlapping car and bus
    never suppress each other).

    Returns detections sorted by ascending ``class_id``, then by each
    class's own NMS keep-order (descending score, ties broken by
    ascending original column index -- see ``nms``'s docstring).
    """
    preds = output[0]  # (4 + n_classes, N)
    n_classes = preds.shape[0] - 4

    boxes_cxcywh = preds[:4, :].T  # (N, 4)
    class_scores = preds[4 : 4 + n_classes, :].T  # (N, n_classes)

    class_ids_all = np.argmax(class_scores, axis=1)
    scores_all = class_scores[np.arange(class_scores.shape[0]), class_ids_all]

    keep_mask = scores_all >= conf
    # keep_class_ids is always applied, even when empty (an empty set
    # legitimately means "keep nothing" -- a caller-side decision, never
    # silently ignored).
    allowed_ids = np.fromiter(keep_class_ids, dtype=np.int64, count=len(keep_class_ids))
    keep_mask &= np.isin(class_ids_all, allowed_ids)

    if not np.any(keep_mask):
        return []

    cx = boxes_cxcywh[keep_mask, 0]
    cy = boxes_cxcywh[keep_mask, 1]
    bw = boxes_cxcywh[keep_mask, 2]
    bh = boxes_cxcywh[keep_mask, 3]
    scores = scores_all[keep_mask]
    cls_ids = class_ids_all[keep_mask]

    x1_model = cx - bw / 2.0
    y1_model = cy - bh / 2.0
    x2_model = cx + bw / 2.0
    y2_model = cy + bh / 2.0

    x1 = (x1_model - pad_x) / scale
    y1 = (y1_model - pad_y) / scale
    x2 = (x2_model - pad_x) / scale
    y2 = (y2_model - pad_y) / scale

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    detections: list[Detection] = []
    for class_id in sorted(set(cls_ids.tolist())):
        idxs = np.where(cls_ids == class_id)[0]
        kept_local = nms(boxes_xyxy[idxs], scores[idxs], iou)
        for local_i in kept_local:
            global_i = idxs[local_i]
            detections.append(
                Detection(
                    x1=float(boxes_xyxy[global_i, 0]),
                    y1=float(boxes_xyxy[global_i, 1]),
                    x2=float(boxes_xyxy[global_i, 2]),
                    y2=float(boxes_xyxy[global_i, 3]),
                    score=float(scores[global_i]),
                    class_id=int(class_id),
                    class_name=COCO_CLASSES[int(class_id)],
                )
            )
    return detections
