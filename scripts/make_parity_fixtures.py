#!/usr/bin/env python3
"""Generate the cross-surface parity fixture: what the PYTHON engine decides.

The product's central claim is that the visitor's own GPU runs the same
engine that produced the published accuracy numbers. ``web/src/parity.test.ts``
turns that into a test by replaying the inputs written here through the
TypeScript engine and demanding the same decisions. Every expected value in
the fixture therefore comes from the Python engine -- ``Tracker``,
``GateCounter``, ``SpeedEstimator``, ``RoadPlane`` and ``decode_yolo`` -- run
over inputs this script also writes. A fixture the TypeScript side produced
would only prove that the TypeScript side agrees with itself.

What goes in it, and why each part is there
-------------------------------------------
**A real clip window.** 150 consecutive frames of the motorway sample, with
the detections the exported ONNX graph actually produces on them, tracked and
counted with the shipped constants and the surveyed road plane. This is the
distribution the product meets: near-duplicate boxes stacked on one vehicle,
detections flickering in and out of the confidence band, tracks born and
reaped continuously.

**Deliberately constructed boundary cases.** Uniform random inputs, and real
clips, essentially never land ON a decision boundary -- so the real window
above proves the two engines agree in the easy interior and says nothing
about the surface where they would actually diverge. Every straddle case here
is placed by hand and then asserted to have had the effect it was built for,
so it cannot quietly stop testing anything:

- ``iouExactlyAtMatchThresh`` -- an IoU of EXACTLY ``TRACK_MATCH_IOU``, paired
  with a control one ULP below it whose Mahalanobis gating distance is
  IDENTICAL, so the IoU comparison is provably the only thing separating them.
- ``scoreExactlyAtHighThresh`` -- a detection whose score is exactly
  ``TRACK_HIGH_CONF`` (inclusive: it may start a track) beside one a single
  ULP below (it may not), and the same pair on ``TRACK_LOW_CONF``.
- ``anchorExactlyOnGate`` -- a position whose cross product against the gate
  is IDENTICALLY zero, not merely inside ``GEOMETRY_EPS``: exact zero is the
  only form of "on the line" both languages are guaranteed to agree on.
- ``deferredOnLineUsesLastOffLinePoint`` -- a crossing deferred by an on-line
  frame whose bounded-segment check only passes when it is made from the
  STORED last off-line point. The counterfactual is computed too and recorded:
  resolving it from ``prev`` yields no event at all.
- ``assignmentCostExactTie`` -- one track flanked by two detections at a
  bit-for-bit identical IoU, so the optimal match SET is non-unique and only
  ``assign``'s canonical reconstruction decides it. This is the case that puts
  a number on replacing scipy's solver with a hand-written one in the browser.
- ``argmaxFloat32ClassTie`` -- two kept classes tied at the identical float32
  score, so the emitted class id depends entirely on the argmax tie rule. It
  is replayed through the tracker as well as the decoder, because a tie
  resolved the other way becomes a track's permanent class name and the
  tracker bars association across classes.

Two hazards this script is written around
-----------------------------------------
1. The fixture serialises the FITTED 3x3 plane matrix, never the surveyed
   correspondences: ``web/src/engine/homography.ts`` deliberately omits
   ``fromCorrespondences``, because the browser has no cv2 and no SVD.
2. The Kalman path is NOT pinned bit-for-bit. numpy here dispatches to
   Accelerate, whose kernels fuse multiply-add, so covariances and gating
   distances agree only to ~1e-14 across the two languages. Boundary cases
   are therefore built so that no DECISION sits inside that noise: the IoU
   straddle's gating distance is 1.08 against a 9.4877 gate, and the on-line
   gate cases use explicit positions rather than filtered anchors.

Usage:

    PYTHONPATH=src .venv/bin/python scripts/make_parity_fixtures.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trafficlens.analytics.speed import SpeedEstimator  # noqa: E402
from trafficlens.config import load_config  # noqa: E402
from trafficlens.core.classes import VEHICLE_CLASSES, class_ids  # noqa: E402
from trafficlens.core.constants import (  # noqa: E402
    KALMAN_GATING_CHI2_95_4DOF,
    TRACK_HIGH_CONF,
    TRACK_LOW_CONF,
    TRACK_MATCH_IOU,
    TRACK_MAX_AGE,
    TRACK_MIN_HITS,
)
from trafficlens.core.classes import COCO_CLASSES  # noqa: E402
from trafficlens.core.gate import Gate, GateCounter  # noqa: E402
from trafficlens.detect.base import Detection, decode_yolo, letterbox  # noqa: E402
from trafficlens.track.associate import iou_matrix  # noqa: E402
from trafficlens.track.kalman import KalmanBoxFilter, xyxy_to_xyah  # noqa: E402
from trafficlens.track.tracker import Tracker  # noqa: E402

SCHEMA_VERSION = 1

CLIP = ROOT / "data" / "samples" / "motorway-a40.webm"
MODEL = ROOT / "web" / "public" / "models" / "yolo11n-480.onnx"
CONFIG = ROOT / "configs" / "motorway.yaml"

#: The clip's own geometry, asserted against the container below rather than
#: assumed: the plane matrix and the gates are both derived from it.
WIDTH, HEIGHT, FPS = 1280, 720, 30.0

#: Detector settings, matching the browser runtime's own.
INPUT_SIZE, CONF, IOU = 480, 0.35, 0.5

#: The replayed window. 150 frames from 120 was chosen by measurement, not by
#: taste: it is the shortest window in this clip that produces more than one
#: counted crossing on more than one gate while still holding tracks that
#: never reach ``SPEED_MIN_SAMPLES``, so the null-speed path is exercised too.
CLIP_START, CLIP_FRAMES = 120, 150

#: No limit is set for this clip (see configs/motorway.yaml: the sign is not
#: legible at this resolution and an invented limit would flag invented
#: violations), so every crossing's ``is_violation`` is False by policy.
SPEED_LIMIT_KMH = None

STRADDLE_KINDS = [
    "anchorExactlyOnGate",
    "iouExactlyAtMatchThresh",
    "scoreExactlyAtHighThresh",
    "assignmentCostExactTie",
    "argmaxFloat32ClassTie",
    "deferredOnLineUsesLastOffLinePoint",
]


# -- serialisation helpers ----------------------------------------------------


def _detection(det: Detection, role: str | None = None) -> dict:
    out = {
        "x1": float(det.x1),
        "y1": float(det.y1),
        "x2": float(det.x2),
        "y2": float(det.y2),
        "score": float(det.score),
        "classId": int(det.class_id),
        "className": det.class_name,
    }
    if role is not None:
        out["role"] = role
    return out


def _event(event) -> dict:
    return {
        "trackId": int(event.track_id),
        "className": event.class_name,
        "gate": event.gate,
        "direction": event.direction,
        "signedDirection": int(event.signed_direction),
        "frameIndex": int(event.frame_index),
        "timestamp": float(event.timestamp),
        "crossingX": float(event.crossing_x),
        "crossingY": float(event.crossing_y),
        "speedKmh": None if event.speed_kmh is None else float(event.speed_kmh),
        "isViolation": bool(event.is_violation),
    }


def _gate_spec(gate: Gate) -> dict:
    return {
        "name": gate.name,
        "start": [float(gate.start[0]), float(gate.start[1])],
        "end": [float(gate.end[0]), float(gate.end[1])],
        "labelPositive": gate.label_positive,
        "labelNegative": gate.label_negative,
    }


def _counts(counters: dict[str, GateCounter]) -> dict:
    return {
        name: {
            class_name: dict(directions)
            for class_name, directions in counter.totals.items()
        }
        for name, counter in counters.items()
        if counter.totals
    }


# -- the engine loop the fixture records --------------------------------------


def run_engine(
    frames: list[dict],
    gates: list[Gate],
    plane,
    roles: dict[float, str] | None = None,
) -> dict:
    """Run the pipeline's per-frame loop over scripted detections.

    This is ``trafficlens.pipeline.run_session``'s body reduced to the parts a
    parity comparison can see -- track, observe, count, reap -- so that the
    fixture can be built from scripted detections with no video decoding,
    annotation or timing in the way. The reaping rule is the pipeline's own,
    strictly ``> max_age``: a confirmed track may re-associate at exactly
    ``max_age``, so forgetting one frame earlier would drop a gate counter's
    ``_counted`` entry while the track could still return, and the same
    vehicle would be counted twice.
    """
    tracker = Tracker()
    counters = {gate.name: GateCounter(gate) for gate in gates}
    speed = SpeedEstimator(plane, FPS)
    previous_anchor: dict[int, tuple[float, float]] = {}
    last_seen: dict[int, int] = {}
    out_frames: list[dict] = []
    events: list[dict] = []
    allocated = 0

    for frame in frames:
        detections = [
            Detection(
                x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
                score=d["score"], class_id=d["classId"], class_name=d["className"],
            )
            for d in frame["detections"]
        ]
        tracks = tracker.update(detections, frame["frameIndex"])
        row: list[dict] = []

        for track in tracks:
            anchor = track.anchor
            last_seen[track.track_id] = frame["frameIndex"]
            allocated = max(allocated, track.track_id)

            speed.observe(track.track_id, anchor, frame["timestamp"])
            speed_kmh = speed.speed_kmh(track.track_id)
            record = {
                "trackId": int(track.track_id),
                "className": track.class_name,
                "speedKmh": None if speed_kmh is None else float(speed_kmh),
            }
            if roles is not None:
                # Roles are matched by the track's box centre, which on the
                # creation frame is exactly the detection's own centre; the
                # boundary cases place their objects hundreds of pixels apart
                # so the nearest centre is never ambiguous.
                centre = (track.box[0] + track.box[2]) / 2.0
                record["role"] = roles[min(roles, key=lambda x: abs(x - centre))]
            row.append(record)

            previous = previous_anchor.get(track.track_id)
            if previous is not None:
                for gate in gates:
                    event = counters[gate.name].update(
                        track.track_id, track.class_name, previous, anchor,
                        frame["frameIndex"], frame["timestamp"],
                        speed_kmh=speed_kmh, speed_limit_kmh=SPEED_LIMIT_KMH,
                    )
                    if event is not None:
                        events.append(_event(event))
            previous_anchor[track.track_id] = anchor

        for track_id, seen in sorted(last_seen.items()):
            if frame["frameIndex"] - seen > TRACK_MAX_AGE:
                del last_seen[track_id]
                for counter in counters.values():
                    counter.forget(track_id)
                speed.forget(track_id)
                previous_anchor.pop(track_id, None)

        out_frames.append({"frameIndex": frame["frameIndex"], "tracks": row})

    return {
        "frames": out_frames,
        "events": events,
        "counts": _counts(counters),
        "tracksAllocated": int(allocated),
    }


def _frame(index: int, detections: list[dict]) -> dict:
    return {
        "frameIndex": index,
        "timestamp": float(index) / FPS,
        "detections": detections,
    }


def _det(box, score, role, class_id=2, class_name="car") -> dict:
    return {
        "x1": float(box[0]), "y1": float(box[1]),
        "x2": float(box[2]), "y2": float(box[3]),
        "score": float(score), "classId": int(class_id),
        "className": class_name, "role": role,
    }


# -- straddle case: IoU exactly at the match threshold ------------------------

#: The track's box on the frame the straddle is measured. A freshly created
#: track's Kalman mean is the measurement itself with zero velocity, and one
#: predict step over zero velocity leaves it untouched, so this box is EXACT
#: on both sides -- no BLAS is involved, and the xyah round trip is exact
#: because the aspect ratio is 0.5.
_AT_TRACK_BOX = (100.0, 100.0, 150.0, 200.0)
#: Overlaps it at an IoU of exactly 4000/5000 = 0.8 == TRACK_MATCH_IOU.
_AT_DET_BOX = (110.0, 100.0, 150.0, 200.0)
#: The control, 500 px away so the two objects never interact. Its top edge is
#: nudged by one ULP, which moves the IoU to nextafter(0.8, 0) -- one ULP
#: below the floor -- while leaving the gating distance within 1.2e-15 of the
#: at-threshold pair's (1.0846114503110150 against 1.0846114503110138), and
#: both at 11% of the chi-square gate. The IoU comparison is therefore
#: provably the only thing that separates the two.
_BELOW_TRACK_BOX = (600.0, 100.0, 650.0, 200.0)
_BELOW_DET_BOX = (610.0, math.nextafter(100.0, 200.0), 650.0, 200.0)


def build_iou_case(plane) -> dict:
    frames = [
        _frame(0, [
            _det(_AT_TRACK_BOX, 0.9, "at_threshold"),
            _det(_BELOW_TRACK_BOX, 0.9, "below_threshold"),
        ]),
        _frame(1, [
            _det(_AT_DET_BOX, 0.9, "at_threshold"),
            _det(_BELOW_DET_BOX, 0.9, "below_threshold"),
        ]),
        _frame(2, [
            _det(_AT_DET_BOX, 0.9, "at_threshold"),
            _det(_BELOW_DET_BOX, 0.9, "below_threshold"),
        ]),
        _frame(3, [
            _det(_AT_DET_BOX, 0.9, "at_threshold"),
            _det(_BELOW_DET_BOX, 0.9, "below_threshold"),
        ]),
    ]
    roles = {
        (_AT_TRACK_BOX[0] + _AT_TRACK_BOX[2]) / 2.0: "at_threshold",
        (_BELOW_TRACK_BOX[0] + _BELOW_TRACK_BOX[2]) / 2.0: "below_threshold",
    }
    expected = run_engine(frames, [], plane, roles=roles)

    measured_iou = _iou(_AT_TRACK_BOX, _AT_DET_BOX)
    control_iou = _iou(_BELOW_TRACK_BOX, _BELOW_DET_BOX)
    gating_at = _gating(_AT_TRACK_BOX, _AT_DET_BOX)
    gating_below = _gating(_BELOW_TRACK_BOX, _BELOW_DET_BOX)

    # The case must have had its effect, or it has silently stopped testing.
    assert measured_iou == TRACK_MATCH_IOU, measured_iou
    assert control_iou == math.nextafter(TRACK_MATCH_IOU, 0.0), control_iou
    # The two pairs' gating distances agree to 1.2e-15 -- they are the same
    # geometry 500 px apart with one endpoint nudged by an ULP -- and both sit
    # at 11% of the chi-square gate. The gate is therefore provably not what
    # separates them; the IoU is.
    assert abs(gating_at - gating_below) < 1e-12, (gating_at, gating_below)
    assert max(gating_at, gating_below) < 0.25 * KALMAN_GATING_CHI2_95_4DOF, gating_at
    # The at-threshold object keeps track 1 across the straddle frame; the
    # control's tentative track dies on its first miss and a NEW id is
    # allocated for it, which is the whole visible difference.
    at_ids = {
        t["trackId"] for f in expected["frames"] for t in f["tracks"]
        if t["role"] == "at_threshold"
    }
    below_ids = {
        t["trackId"] for f in expected["frames"] for t in f["tracks"]
        if t["role"] == "below_threshold"
    }
    assert at_ids == {1}, ["the at-threshold pair failed to match", at_ids]
    assert below_ids == {3}, [
        "the one-ULP-below pair matched anyway, or died differently", below_ids
    ]
    assert expected["tracksAllocated"] == 3, expected["tracksAllocated"]

    return {
        "name": "tracker_iou_at_match_thresh",
        "straddles": ["iouExactlyAtMatchThresh"],
        "gates": [],
        "frames": frames,
        "expected": expected,
        "straddleTrackBox": list(_AT_TRACK_BOX),
        "straddleDetectionBox": list(_AT_DET_BOX),
        "straddleIou": measured_iou,
        "straddleGatingDistance": gating_at,
        "straddleGatingChi2": KALMAN_GATING_CHI2_95_4DOF,
        "controlTrackBox": list(_BELOW_TRACK_BOX),
        "controlDetectionBox": list(_BELOW_DET_BOX),
        "controlIou": control_iou,
    }


def _iou(a, b) -> float:
    return float(iou_matrix(np.array([a]), np.array([b]))[0, 0])


def _gating(track_box, det_box) -> float:
    """Squared Mahalanobis distance from a track created at ``track_box`` and
    predicted one step, to a measurement at ``det_box``."""
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(xyxy_to_xyah(np.array(track_box)))
    mean, cov = kf.predict(mean, cov)
    measurement = np.array([xyxy_to_xyah(np.array(det_box))])
    return float(kf.gating_distance(mean, cov, measurement)[0])


# -- straddle case: an exact tie in the assignment cost -----------------------

#: One track flanked by two detections at IDENTICAL IoU -- 9000/11000 on both
#: sides, bit-for-bit equal, and both above the match floor. Tied costs make
#: the optimal match SET itself non-unique, so which optimum a solver returns
#: is implementation-internal; ``assign`` therefore uses the solver only as an
#: oracle for the optimal TOTAL and reconstructs the lexicographically-least
#: optimum. That reconstruction is the load-bearing claim behind replacing
#: scipy's ``linear_sum_assignment`` with a hand-written solver in the
#: browser, and this is the case that puts a number on it: Python picks the
#: LEFT detection, and the two engines' whole subsequent track histories
#: diverge if the browser picks the right one.
_TIE_TRACK_BOX = (100.0, 100.0, 200.0, 200.0)
_TIE_LEFT_BOX = (90.0, 100.0, 190.0, 200.0)
_TIE_RIGHT_BOX = (110.0, 100.0, 210.0, 200.0)


def build_assignment_tie_case(plane) -> dict:
    frames = [
        _frame(0, [_det(_TIE_TRACK_BOX, 0.9, "tie_seed")]),
        # The tie. Left is column 0, right is column 1.
        _frame(1, [
            _det(_TIE_LEFT_BOX, 0.9, "tie_left"),
            _det(_TIE_RIGHT_BOX, 0.9, "tie_right"),
        ]),
        # From here the object continues LEFT, so the track that won the tie
        # keeps matching and confirms while the loser's tentative track dies
        # on its first miss. Had the tie gone the other way, the confirmed
        # track would carry id 2 and appear a frame later.
        _frame(2, [_det((80.0, 100.0, 180.0, 200.0), 0.9, "tie_left")]),
        _frame(3, [_det((70.0, 100.0, 170.0, 200.0), 0.9, "tie_left")]),
    ]
    expected = run_engine(frames, [], plane)

    left_iou = _iou(_TIE_TRACK_BOX, _TIE_LEFT_BOX)
    right_iou = _iou(_TIE_TRACK_BOX, _TIE_RIGHT_BOX)
    assert left_iou == right_iou, ["the tie is not exact", left_iou, right_iou]
    assert left_iou >= TRACK_MATCH_IOU, ["both tied pairs must clear the floor", left_iou]
    ids = [[t["trackId"] for t in f["tracks"]] for f in expected["frames"]]
    assert ids == [[], [], [1], [1]], [
        "the cost tie no longer resolves to the lexicographically-least "
        "optimum: a solver that took the right-hand detection confirms track "
        "2 one frame later instead", ids
    ]
    return {
        "name": "tracker_assignment_cost_tie",
        "straddles": ["assignmentCostExactTie"],
        "gates": [],
        "frames": frames,
        "expected": expected,
        "tieTrackBox": list(_TIE_TRACK_BOX),
        "tieDetectionBoxes": [list(_TIE_LEFT_BOX), list(_TIE_RIGHT_BOX)],
        "tieIou": left_iou,
    }


# -- straddle case: score exactly at the confidence thresholds ----------------


def build_score_case(plane) -> dict:
    scores = {
        "at_high": TRACK_HIGH_CONF,
        "below_high": math.nextafter(TRACK_HIGH_CONF, 0.0),
        "at_low": TRACK_LOW_CONF,
        "below_low": math.nextafter(TRACK_LOW_CONF, 0.0),
    }
    boxes = {
        "at_high": (100.0, 100.0, 150.0, 200.0),
        "below_high": (400.0, 100.0, 450.0, 200.0),
        "at_low": (700.0, 100.0, 750.0, 200.0),
        "below_low": (1000.0, 100.0, 1050.0, 200.0),
    }
    detections = [_det(boxes[r], scores[r], r) for r in boxes]
    frames = [_frame(i, detections) for i in range(TRACK_MIN_HITS + 1)]
    roles = {(b[0] + b[2]) / 2.0: r for r, b in boxes.items()}
    expected = run_engine(frames, [], plane, roles=roles)

    surviving = {t["role"] for f in expected["frames"] for t in f["tracks"]}
    assert surviving == {"at_high"}, [
        "the confidence band no longer resolves as measured: exactly the "
        "at-threshold detection may start a track", surviving
    ]
    assert expected["tracksAllocated"] == 1, expected["tracksAllocated"]
    assert scores["below_high"] < TRACK_HIGH_CONF < 1.0
    assert TRACK_LOW_CONF <= scores["below_high"] < TRACK_HIGH_CONF
    assert scores["below_low"] < TRACK_LOW_CONF

    return {
        "name": "tracker_score_at_high_thresh",
        "straddles": ["scoreExactlyAtHighThresh"],
        "gates": [],
        "frames": frames,
        "expected": expected,
    }


# -- the real clip window -----------------------------------------------------


def build_real_case(plane, gates: list[Gate]) -> dict:
    import cv2
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    keep = class_ids(VEHICLE_CLASSES)

    capture = cv2.VideoCapture(str(CLIP))
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == WIDTH
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == HEIGHT
    assert capture.get(cv2.CAP_PROP_FPS) == FPS
    capture.set(cv2.CAP_PROP_POS_FRAMES, CLIP_START)

    frames: list[dict] = []
    for offset in range(CLIP_FRAMES):
        ok, image = capture.read()
        if not ok:
            raise SystemExit(f"{CLIP} ran out of frames at offset {offset}")
        chw, scale, pad_x, pad_y = letterbox(image, INPUT_SIZE)
        (raw,) = session.run(None, {input_name: chw})
        detections = decode_yolo(
            raw, scale, pad_x, pad_y, conf=CONF, iou=IOU, keep_class_ids=keep
        )
        frames.append(_frame(CLIP_START + offset, [_detection(d) for d in detections]))
    capture.release()

    expected = run_engine(frames, gates, plane)

    speeds = [t["speedKmh"] for f in expected["frames"] for t in f["tracks"]]
    assert len(expected["events"]) >= 2, expected["events"]
    assert len({e["gate"] for e in expected["events"]}) >= 2, expected["events"]
    assert any(s is None for s in speeds), "the null-speed path is not exercised"
    assert sum(1 for s in speeds if s is not None) >= 100
    return {
        "name": "tracker_real_clip_window",
        "straddles": [],
        "gates": [_gate_spec(g) for g in gates],
        "frames": frames,
        "expected": expected,
    }


# -- gate cases ---------------------------------------------------------------

#: A horizontal gate running left to right, so ``side_of_line``'s cross
#: product is ``(gate_end_x - gate_start_x) * (gate_y - p_y)``: it is
#: IDENTICALLY zero for any point at y = 500, whatever its x. That exactness
#: is the point -- a position that is merely within GEOMETRY_EPS of the line
#: would be on the line by a tolerance rather than by construction.
_GATE = Gate("median", (100.0, 500.0), (300.0, 500.0), "away", "toward")


def _step(track_id, prev, curr, index) -> dict:
    return {
        "trackId": int(track_id),
        "className": "car",
        "prev": [float(prev[0]), float(prev[1])],
        "curr": [float(curr[0]), float(curr[1])],
        "frameIndex": int(index),
        "timestamp": float(index) / FPS,
    }


def _run_gate(steps: list[dict], gate: Gate = _GATE) -> tuple[list[dict], dict]:
    counter = GateCounter(gate)
    events = [
        _event(event)
        for step in steps
        if (event := counter.update(
            step["trackId"], step["className"],
            tuple(step["prev"]), tuple(step["curr"]),
            step["frameIndex"], step["timestamp"],
            speed_kmh=None, speed_limit_kmh=SPEED_LIMIT_KMH,
        )) is not None
    ]
    return events, _counts({gate.name: counter})


def build_gate_on_line_case() -> dict:
    """A track whose anchor lands EXACTLY on the gate line for one frame.

    ``crossing_direction`` returns 0 for an on-line endpoint rather than
    firing twice, so the crossing must be deferred to the following frame and
    resolved against the remembered side. A mirror that dropped the deferral
    would lose the crossing entirely; one that fired on the on-line frame
    would count it a frame early.
    """
    steps = [
        _step(1, (150.0, 480.0), (150.0, 490.0), 1),
        _step(1, (150.0, 490.0), (150.0, 500.0), 2),   # exactly on the line
        _step(1, (150.0, 500.0), (150.0, 520.0), 3),   # resolves here
        # The control, on a different axis from the deferral: a track that
        # approaches the gate and turns back without ever reaching it.
        _step(2, (200.0, 460.0), (200.0, 470.0), 1),
        _step(2, (200.0, 470.0), (200.0, 460.0), 2),
    ]
    events, counts = _run_gate(steps)
    assert len(events) == 1, events
    assert events[0]["frameIndex"] == 3, ["the deferral resolved on the wrong frame", events]
    assert events[0]["direction"] == "toward", events
    assert events[0]["crossingY"] == 500.0, events
    return {
        "name": "gate_anchor_exactly_on_line",
        "straddles": ["anchorExactlyOnGate"],
        "gate": _gate_spec(_GATE),
        "steps": steps,
        "expected": {"events": events, "counts": counts},
    }


def build_gate_deferred_origin_case() -> dict:
    """The deferred crossing whose bounds check needs the STORED off-line
    point.

    The on-line frame lands at x = 400, well past the gate's right end, so
    the swept segment from ``prev`` misses the bounded gate entirely and the
    crossing would be dropped. From the remembered last off-line position at
    x = 150 the segment cuts the gate at x = 233.3 and the crossing counts.
    The counterfactual is computed below, not merely asserted about: a case
    whose two branches agree pins nothing.
    """
    steps = [
        _step(1, (150.0, 480.0), (150.0, 490.0), 1),
        _step(1, (150.0, 490.0), (400.0, 500.0), 2),   # on the line, out of bounds
        _step(1, (400.0, 500.0), (400.0, 520.0), 3),
    ]
    events, counts = _run_gate(steps)

    # The counterfactual: the same three steps through a counter that resolves
    # a deferred crossing from `prev` instead of the stored off-line point.
    counterfactual = _run_prev_origin_counterfactual(steps)

    assert len(events) == 1, events
    assert events[0]["frameIndex"] == 3, events
    assert 100.0 <= events[0]["crossingX"] <= 300.0, events
    assert counterfactual == [], [
        "resolving from prev produced the event too: this case pins nothing",
        counterfactual,
    ]
    return {
        "name": "gate_deferred_off_line_origin",
        "straddles": ["deferredOnLineUsesLastOffLinePoint", "anchorExactlyOnGate"],
        "gate": _gate_spec(_GATE),
        "steps": steps,
        "expected": {"events": events, "counts": counts},
        "counterfactualPrevOrigin": {"events": counterfactual},
    }


def _run_prev_origin_counterfactual(steps: list[dict]) -> list[dict]:
    """Replay ``steps`` through a GateCounter whose deferred resolution uses
    ``prev`` as the bounds-check origin -- the implementation this project
    does NOT have -- so the difference can be recorded as a measurement
    rather than asserted from the code."""

    class _PrevOriginCounter(GateCounter):
        def update(self, track_id, class_name, prev, curr, *args, **kwargs):
            self._last_off_line_point[track_id] = prev
            return super().update(track_id, class_name, prev, curr, *args, **kwargs)

    counter = _PrevOriginCounter(_GATE)
    return [
        _event(event)
        for step in steps
        if (event := counter.update(
            step["trackId"], step["className"],
            tuple(step["prev"]), tuple(step["curr"]),
            step["frameIndex"], step["timestamp"],
            speed_kmh=None, speed_limit_kmh=SPEED_LIMIT_KMH,
        )) is not None
    ]


def build_gate_bounds_case() -> dict:
    """The bounded-segment rule, with its must-count and must-not-count halves.

    Track 1 crosses exactly through the gate's own right ENDPOINT: bounds are
    inclusive, so it counts. Track 2 crosses the gate's infinite line 100 px
    past that endpoint -- a parallel carriageway -- so it does not. Track 3
    crosses and comes straight back: a gate counts a track once, ever.
    """
    steps = [
        _step(1, (300.0, 480.0), (300.0, 520.0), 1),   # exactly the endpoint
        _step(2, (400.0, 480.0), (400.0, 520.0), 1),   # past the end: no count
        _step(3, (200.0, 480.0), (200.0, 520.0), 1),   # counts
        _step(3, (200.0, 520.0), (200.0, 480.0), 2),   # already counted
    ]
    events, counts = _run_gate(steps)
    assert [e["trackId"] for e in events] == [1, 3], events
    assert counts == {"median": {"car": {"toward": 2}}}, counts
    return {
        "name": "gate_bounded_segment",
        "straddles": [],
        "gate": _gate_spec(_GATE),
        "steps": steps,
        "expected": {"events": events, "counts": counts},
    }


# -- decode case: the float32 argmax tie --------------------------------------

N_CLASSES = 80
_TIE_SCALE, _TIE_PAD_X, _TIE_PAD_Y = 0.375, 0.0, 105.0


def build_decode_tie_case() -> dict:
    """Two kept classes tied at the identical float32 score.

    numpy's ``argmax`` returns the FIRST maximum, so the column is a car; a
    mirror scanning with ``>=`` would take the LAST and call it a truck. Both
    classes are kept, so the column survives either way -- the disagreement is
    a wrong class id, not a missing detection, and it is invisible without a
    tie to test.

    The tie is replayed through the tracker as well as the decoder because
    the class it resolves to becomes the track's PERMANENT class name and the
    tracker bars association across classes, so a tie decided differently
    changes which detections may ever match which tracks.
    """
    car, truck = 2, 7
    columns = 2
    raw = np.zeros((1, 4 + N_CLASSES, columns), dtype=np.float32)
    tie_score = np.float32(0.66)
    raw[0, 0:4, 0] = np.array((1000.0, 700.0, 30.0, 30.0), dtype=np.float32)
    raw[0, 4 + car, 0] = tie_score
    raw[0, 4 + truck, 0] = tie_score
    # The control, on a different axis: an untied column whose single best
    # class is the HIGHER id, so a mirror that simply always preferred the
    # lower id would fail here while passing the tie.
    raw[0, 0:4, 1] = np.array((200.0, 700.0, 30.0, 30.0), dtype=np.float32)
    raw[0, 4 + truck, 1] = np.float32(0.70)

    keep = sorted(class_ids(VEHICLE_CLASSES))
    detections = decode_yolo(
        raw, _TIE_SCALE, _TIE_PAD_X, _TIE_PAD_Y,
        conf=CONF, iou=IOU, keep_class_ids=class_ids(VEHICLE_CLASSES),
    )
    assert [d.class_id for d in detections] == [car, truck], [
        "the argmax tie no longer resolves to the FIRST maximum",
        [(d.class_id, d.score) for d in detections],
    ]
    assert raw[0, 4 + car, 0] == raw[0, 4 + truck, 0], "the tie is not exact"

    frames = [
        _frame(i, [_detection(d) for d in detections]) for i in range(TRACK_MIN_HITS)
    ]
    expected = run_engine(frames, [], None)
    tracks = expected["frames"][-1]["tracks"]
    assert [t["className"] for t in tracks] == ["car", "truck"], tracks

    return {
        "name": "decode_argmax_class_tie",
        "straddles": ["argmaxFloat32ClassTie"],
        "dims": [1, 4 + N_CLASSES, columns],
        "raw": [float(v) for v in raw.reshape(-1)],
        "scale": _TIE_SCALE,
        "padX": _TIE_PAD_X,
        "padY": _TIE_PAD_Y,
        "conf": CONF,
        "iou": IOU,
        "keepClasses": [
            {"classId": int(c), "className": COCO_CLASSES[c]} for c in keep
        ],
        "tieColumn": 0,
        "tieClassIds": [car, truck],
        "expectedDetections": [_detection(d) for d in detections],
        "replayFrames": TRACK_MIN_HITS,
        "expectedTracks": [
            {"trackId": t["trackId"], "className": t["className"], "speedKmh": None}
            for t in tracks
        ],
    }


# -- assembly -----------------------------------------------------------------


def build_fixture() -> tuple[dict, dict]:
    config = load_config(CONFIG)
    gates = [gate_config.to_gate(WIDTH, HEIGHT) for gate_config in config.gates]
    plane = config.calibration.to_plane(WIDTH, HEIGHT)
    # The FITTED matrix, never the correspondences: the browser has no cv2
    # and no SVD, and web/src/engine/homography.ts deliberately omits the fit.
    image_to_world = [[float(v) for v in row] for row in np.asarray(plane._h)]

    tracker_cases = [
        build_iou_case(plane),
        build_assignment_tie_case(plane),
        build_score_case(plane),
        build_real_case(plane, gates),
    ]
    gate_cases = [
        build_gate_on_line_case(),
        build_gate_deferred_origin_case(),
        build_gate_bounds_case(),
    ]
    decode_cases = [build_decode_tie_case()]

    fixture = {
        "schemaVersion": SCHEMA_VERSION,
        "source": {
            "clip": CLIP.name,
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "model": MODEL.name,
            "inputSize": INPUT_SIZE,
            "conf": CONF,
            "iou": IOU,
            "startFrame": CLIP_START,
        },
        "plane": {"imageToWorld": image_to_world},
        "tracker": {
            "highThresh": TRACK_HIGH_CONF,
            "lowThresh": TRACK_LOW_CONF,
            "matchThresh": TRACK_MATCH_IOU,
            "maxAge": TRACK_MAX_AGE,
            "minHits": TRACK_MIN_HITS,
        },
        "speedLimitKmh": SPEED_LIMIT_KMH,
        "straddleKinds": STRADDLE_KINDS,
        "trackerCases": tracker_cases,
        "gateCases": gate_cases,
        "decodeCases": decode_cases,
    }

    real = tracker_cases[-1]
    speeds = [
        t["speedKmh"] for f in real["expected"]["frames"] for t in f["tracks"]
    ]
    iou_case = tracker_cases[0]
    report = {
        "caseCount": len(tracker_cases) + len(gate_cases) + len(decode_cases),
        "straddleKinds": STRADDLE_KINDS,
        "straddleCases": {
            kind: sorted(
                case["name"]
                for case in tracker_cases + gate_cases + decode_cases
                if kind in case["straddles"]
            )
            for kind in STRADDLE_KINDS
        },
        "realClip": {
            "clip": CLIP.name,
            "startFrame": CLIP_START,
            "frames": len(real["frames"]),
            "detections": sum(len(f["detections"]) for f in real["frames"]),
            "tracksAllocated": real["expected"]["tracksAllocated"],
            "events": len(real["expected"]["events"]),
            "counts": real["expected"]["counts"],
            "speedsReported": sum(1 for s in speeds if s is not None),
            "speedsWithheld": sum(1 for s in speeds if s is None),
            "maxSpeedKmh": max(s for s in speeds if s is not None),
        },
        "iouStraddle": {
            "iou": iou_case["straddleIou"],
            "matchThresh": TRACK_MATCH_IOU,
            "controlIou": iou_case["controlIou"],
            "controlIsOneUlpBelow": (
                iou_case["controlIou"] == math.nextafter(TRACK_MATCH_IOU, 0.0)
            ),
            "gatingDistance": iou_case["straddleGatingDistance"],
            "gatingChi2": KALMAN_GATING_CHI2_95_4DOF,
        },
        "speedToleranceKmh": 1e-06,
    }
    return fixture, report


def write_fixtures(out_root: Path) -> dict:
    """Write both artefacts under ``out_root`` and return the report, so a
    caller (and the regeneration test) needs only one pass over the clip."""
    fixture, report = build_fixture()
    fixture_dir = out_root / "web" / "src" / "fixtures"
    report_dir = out_root / "reports"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "parity.json").write_text(
        json.dumps(fixture, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    (report_dir / "parity.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> int:
    report = write_fixtures(ROOT)
    print(f"parity fixture: {report['caseCount']} cases, "
          f"{report['realClip']['frames']} real frames, "
          f"{report['realClip']['detections']} detections, "
          f"{report['realClip']['events']} crossings, "
          f"{report['realClip']['tracksAllocated']} tracks allocated")
    for kind, names in report["straddleCases"].items():
        print(f"  {kind}: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
