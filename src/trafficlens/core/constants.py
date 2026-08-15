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

# Maximum acceptable value of the rank/uniqueness diagnostic
# trafficlens.core.homography._dlt_condition_number computes from the
# Hartley-normalized DLT design matrix, used by RoadPlane.validate() to
# detect a geometrically degenerate correspondence set (collinear points, a
# duplicated point, or any configuration close to one) without penalising a
# precise survey. The diagnostic is sigma_1/sigma_8 of the design matrix's
# singular values for BOTH the exactly-4-point case (8x9 matrix, 8 singular
# values total, sigma_8 is the last one) and the 5-or-more-point case (2Nx9
# matrix, 9 singular values, sigma_9 -- the smallest -- is excluded because
# it measures fit residual/noise, not geometry; see that function's
# docstring for why excluding it matters).
#
# Measured on real synthetic configurations, both sides of the threshold,
# for both point counts (name kept "CONDITION_NUMBER" despite the sigma_8
# vs sigma_9 distinction: it is still literally sigma_1 divided by another
# singular value of the same matrix, just the correct one):
#   - 4-point healthy trapezoid: ~7.93. Threshold / healthy = ~1261x margin.
#   - 4-point near-degenerate (one point nudged 1mm toward collinear with
#     the other three): ~56216. Near-degenerate / threshold = ~5.6x margin
#     -- narrower than the healthy side, but still clearly separated.
#   - 5-point healthy, well-spread survey, Gaussian pixel noise sigma in
#     {0.1, 0.25, 0.5, 1.0, 2.0}px, 30 trials each (see
#     tests/test_homography.py::test_precise_surveys_are_never_rejected_as_degenerate):
#     ~4.6-4.7 across every noise level tested, essentially flat -- this
#     diagnostic does not scale with survey precision the way the excluded
#     sigma_9 did. Threshold / worst observed (~4.66) = ~2146x margin.
#   - 5-point genuinely degenerate (4 of the 5 points collinear, 1 off the
#     line -- the minimum degeneracy that leaves no non-degenerate 4-point
#     subset among the 5): ~1.27e16. Degenerate / threshold = ~1.27e12x
#     margin.
# 10000.0 sits comfortably inside every one of these gaps; the narrowest
# margin on either point count is the 4-point near-degenerate case (~5.6x),
# which is why that fixture nudges a point by a full 1mm rather than a
# smaller amount that might land closer to the threshold.
HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER = 10000.0

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

# --- Multi-object tracker (trafficlens.track.tracker) ------------------------
# All tunables of the two-stage tracker live here so the later TypeScript
# mirror reads the exact same values; tracker.py itself contains no numeric
# tunables. Each is the default of the corresponding Tracker() constructor
# parameter.

# Detection score at or above which (inclusive) a detection counts as HIGH
# confidence: eligible for the first association stage against every live
# track, and the only kind of detection allowed to start a new track. 0.6
# follows ByteTrack's published track_thresh for vehicle-scale objects:
# high enough that a track is only ever born from a detection the model is
# genuinely sure about, low enough not to starve the tracker on ordinary
# footage.
TRACK_HIGH_CONF = 0.6

# Detection score at or above which (inclusive) a detection enters the LOW
# confidence band [TRACK_LOW_CONF, TRACK_HIGH_CONF) used by the second
# association stage to keep an occluded, already-confirmed track alive.
# Detections below this are discarded entirely. 0.1 is ByteTrack's floor:
# under it, boxes are overwhelmingly background noise that would only feed
# false re-associations.
TRACK_LOW_CONF = 0.1

# IoU floor for detection-to-track association, in BOTH stages: a pair is
# eligible only when IoU(predicted box, detection box) >= this value,
# implemented as an assignment cost of (1 - IoU) capped at max_cost
# = (1 - TRACK_MATCH_IOU). 0.8 is strict by design: with a per-frame
# Kalman prediction supplying the motion, a genuine continuation overlaps
# its predicted box almost entirely, so demanding 80% overlap rejects
# lane-neighbour confusions that a looser floor would let through.
TRACK_MATCH_IOU = 0.8

# Number of consecutive frames a confirmed track may go without a matched
# detection before it is dropped. A track whose time_since_update exceeds
# this dies; a gap of up to exactly this many frames survives on Kalman
# prediction alone and can re-associate. 30 frames is one second at the
# 30 fps footage this project targets -- the longest occlusion (an
# overtaking lorry, a sign gantry) worth bridging before the motion
# extrapolation itself becomes untrustworthy.
TRACK_MAX_AGE = 30

# Consecutive matched frames (hits) a new track needs before it is
# CONFIRMED and appears in Tracker.update() output; until then it is
# tentative and internal, and a single missed frame kills it. 3 is the
# SORT/DeepSORT convention: two frames of agreement can still be a
# double-counted NMS artefact, three in a row almost never are.
TRACK_MIN_HITS = 3
