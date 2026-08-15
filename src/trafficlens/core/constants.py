"""Machine-parseable constants shared across the Python core and, later, a
generated TypeScript mirror for the browser engine.

This file is parsed with Python's ``ast`` module by a separate script, so it
must contain only module-level ``UPPER_CASE = <literal>`` assignments (int,
float, or str literals) plus comments and this docstring. No imports, no
computed expressions, no tuples, no f-strings. Anything that needs to be
computed belongs at its point of use, not here.
"""

# Tolerance for geometric comparisons (cross-product sign, denominator
# magnitude) in trafficlens.core.geometry. Coordinates are pixel positions,
# so 1e-9 is far below any sub-pixel noise a detector or tracker could ever
# produce -- it only guards against floating-point rounding, not real
# ambiguity in the input.
GEOMETRY_EPS = 1e-9

# Maximum acceptable condition number of the (Hartley-normalized) DLT design
# matrix used by trafficlens.core.homography.RoadPlane.validate() to detect
# a degenerate or near-degenerate correspondence set (collinear points, a
# duplicated point, or any configuration close to one). Measured on real
# synthetic configurations: a healthy trapezoid of 4 surveyed road points
# gives a condition number of about 8; nudging one point 1mm toward
# collinear with the other three pushes it to about 56000; exactly collinear
# or duplicated points push it above 1e16. 10000.0 sits comfortably between
# the healthy and near-degenerate measurements (three orders of magnitude of
# margin on each side), so it rejects genuine near-degeneracy without ever
# rejecting a well-spread survey.
HOMOGRAPHY_MAX_CONDITION_NUMBER = 10000.0

# Default maximum acceptable mean reprojection error, in metres, for
# RoadPlane.validate() to accept a calibration. 0.5m mirrors the threshold a
# corrupted single-point survey error is expected to exceed: a mean error
# this large means an object's reported plane position -- and therefore its
# speed -- cannot be trusted to better than half a metre, which is too
# coarse for confident lane-level speed enforcement.
HOMOGRAPHY_MAX_MEAN_ERROR_M = 0.5

# Default minimum per-class confidence score trafficlens.detect.base.decode_yolo
# keeps a raw prediction at, before NMS runs. 0.25 is the long-standing YOLO
# default (YOLOv3 through ultralytics' own predict CLI), low enough to leave
# real-but-uncertain detections in for NMS and downstream tracking to sort
# out, high enough to drop the long tail of near-zero background scores.
DETECT_DEFAULT_CONF = 0.25

# Default IoU threshold trafficlens.detect.base.nms uses to suppress a
# lower-score duplicate box in favour of a higher-score one for the same
# class. 0.45 is the classic YOLO NMS default (YOLOv5 through v8): tight
# enough to collapse near-duplicate boxes on one vehicle, loose enough not
# to merge two genuinely distinct, closely-spaced vehicles into one.
DETECT_DEFAULT_NMS_IOU = 0.45

# Default square side, in pixels, trafficlens.detect.base.letterbox resizes
# and pads a frame to before it is handed to a YOLO11 model. 640 is the
# input resolution yolo11n.pt/yolo11s.pt/yolo11m.pt in this repo were
# trained and exported at -- confirmed by the ONNX export's own reported
# output shape (1, 84, 8400) for imgsz=640, since 8400 = (640/8)^2 +
# (640/16)^2 + (640/32)^2, the three detection-head stride grids summed.
DETECT_DEFAULT_INPUT_SIZE = 640

# Grey value trafficlens.detect.base.letterbox pads a frame's borders with,
# so padding never biases a detection toward any class. 114 is the value
# YOLOv5 introduced and ultralytics (and therefore YOLO11) has used ever
# since -- the same value every checkpoint in this repo was trained
# against, so padding with anything else would shift the input distribution
# the model actually saw during training.
LETTERBOX_PAD_VALUE = 114
