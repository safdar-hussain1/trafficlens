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
# gives a condition number of about 7.93; nudging one point 1mm toward
# collinear with the other three pushes it to about 56216; exactly collinear
# or duplicated points push it above 1e16. 10000.0 sits below the
# near-degenerate measurement (56216 / 10000 = ~5.6x) and far below the
# healthy measurement (10000 / 7.93 = ~1261x) -- the margin is not
# symmetric: comfortably wide on the healthy side, narrower but still clear
# on the near-degenerate side, which is why the near-degenerate fixture in
# tests/test_homography.py nudges a point by a full 1mm rather than a
# smaller amount that might land closer to the threshold.
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

# --- Kalman filter (trafficlens.track.kalman) --------------------------------
# All tunables of the constant-velocity box filter live here so the later
# TypeScript mirror reads the exact same values; kalman.py itself contains
# no numeric tunables.

# Base standard deviation of POSITION-like state components (cx, cy, h) as a
# fraction of the box height h. Height is a proxy for distance from the
# camera: a taller box is a nearer object, which moves more pixels per frame,
# so its uncertainty in pixels should be proportionally larger. 1/20 = 0.05
# is the weight the original DeepSORT filter shipped with and has been the
# de-facto standard for xyah box filters since.
KALMAN_STD_WEIGHT_POSITION = 0.05

# Base standard deviation of VELOCITY-like state components (vx, vy, vh) as a
# fraction of the box height, per frame. 1/160 = 0.00625 (DeepSORT's value):
# velocities change far more slowly than positions jitter, so the process
# trusts the constant-velocity model 8x more (0.05 / 0.00625) than it trusts
# any single position.
KALMAN_STD_WEIGHT_VELOCITY = 0.00625

# Multiplier on the position weight for the INITIAL position variance in
# initiate(): the first box comes from a single unconfirmed detection, so it
# is trusted less (2x the running std) than a measurement arriving during
# steady-state tracking.
KALMAN_INIT_POSITION_STD_FACTOR = 2.0

# Multiplier on the velocity weight for the INITIAL velocity variance in
# initiate(): velocities start at exactly 0 with no evidence at all, so
# their std is inflated 10x -- the first two or three measurements then set
# the velocity almost entirely on their own.
KALMAN_INIT_VELOCITY_STD_FACTOR = 10.0

# Standard deviation of the aspect-ratio (a = w/h) state used for the
# initial variance and the per-step process noise. Aspect ratio is
# dimensionless and near-constant for a rigid vehicle, so its uncertainty
# is a small absolute value rather than height-scaled. 1e-2 per DeepSORT.
KALMAN_ASPECT_STD = 0.01

# Standard deviation of the aspect-ratio VELOCITY (va) process noise per
# step. Aspect ratio should barely drift at all frame-to-frame, hence three
# orders of magnitude below KALMAN_ASPECT_STD. 1e-5 per DeepSORT.
KALMAN_ASPECT_VELOCITY_STD = 1e-05

# Standard deviation of the aspect-ratio MEASUREMENT noise in update().
# Detector aspect ratios jitter much more than the true aspect drifts
# (partial occlusion clips box width), so the measurement is trusted 10x
# less (1e-1) than the state's own process noise (1e-2). Per DeepSORT.
KALMAN_ASPECT_MEASUREMENT_STD = 0.1

# 95% quantile of the chi-square distribution with 4 degrees of freedom
# (one DOF per measured component: cx, cy, a, h). A squared Mahalanobis
# gating distance above this value means the measurement has under a 5%
# chance of belonging to the track under Gaussian assumptions; the tracker
# (Task 7) refuses to associate such pairs. scipy.stats.chi2.ppf(0.95, 4)
# = 9.487729036781154, conventionally quoted as 9.4877.
KALMAN_GATING_CHI2_95_4DOF = 9.4877
