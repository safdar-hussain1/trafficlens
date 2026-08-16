"""Spoiling the input four independent ways, and scoring what survives.

Task 14 measured the engine against hand-labelled ground truth on clean
30 fps footage. This module measures what happens when the input stops
being clean: the frame rate falls, frames go missing, detections drop out
under occlusion, and boxes jitter. Each protocol is seeded, reproducible,
and reduces exactly to the undegraded run at its identity level -- which
is the only thing that makes the degraded rows interpretable at all.

What this module may not know
-----------------------------
The same seam ``scoring`` is held to. ``scoring`` decides what a match
IS and must not know how a prediction was produced; this module decides
how the INPUT is spoiled and must not know what consumes it. It imports
no tracker, no counting rule and no pipeline: methods arrive as opaque
callables and labels as data. A degradation protocol holding a handle on
the engine could be shaped -- deliberately or by drift -- against the
engine's behaviour, and no downstream number would ever show it.

The match window MUST widen with the sampling interval
------------------------------------------------------
Ground-truth frames are indexed in the ORIGINAL 30 fps stream. A method
that only sees every Delta-th frame physically cannot report a crossing
before the next sampled frame at or after the true one, so scoring a
decimated run against the undegraded ``[label - 1, label + 4]`` window
charges every method for the sampling grid rather than for its own
behaviour. Scoring therefore happens in original frame indices
throughout -- ``map_events_to_source`` maps every decimated index back --
and only the LATE side of the window moves:

    window at realised gap G  =  [label - 1,  label + 4 + (G - 1)]

``G`` is the largest inter-sample gap the retained pattern actually
realises, measured in original frames, and it is computable from the
pattern before any scoring happens. For a uniform stride it is the
stride, so the formula reduces to ``+ (Delta - 1)``; for the irregular
grids that 25 fps and the dropped-frame protocol produce it is the
conservative choice, and it is published beside every figure it scored.
The early side never moves: widening it would start matching predictions
to a vehicle that had not yet arrived.

``window_resolution`` states the cost of that widening rather than
hiding it. On the shipped label set the closest crossing pair is five
frames apart, so their windows already share one frame undegraded, and
every widening enlarges the shared region. Figures scored with a widened
window are resolution-limited, and the report says which.

The stream handed to the engine is renumbered
---------------------------------------------
A resampled stream is renumbered from zero before the engine sees it, as
though the clip itself had been recorded that way. That is deliberate:
``Tracker.update`` ages a track once per CALL, so handing it the original
indices would age tracks on the 30 fps clock while the tracker's own
``time_since_update`` still ticked once per sample -- two clocks
disagreeing inside one run. The consequence is stated rather than hidden:
``max_age`` is expressed in frames, not seconds, so decimation
implicitly lengthens the engine's memory in wall-clock terms.

Seeding
-------
Every level derives its own stream from the seed, the protocol name and
the level VALUE -- never from one sequence consumed in sweep order.
Otherwise adding a sweep point silently moves every published number
after it, and the re-run would look like a measurement.

numpy + stdlib + ``trafficlens.bench.scoring`` / ``.slitscan`` /
``trafficlens.core.gate`` / ``trafficlens.detect.base`` only.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Callable, Iterable, Sequence

import numpy as np

from trafficlens.bench.scoring import (
    DEFAULT_MATCH_WINDOW,
    MatchResult,
    MatchWindow,
    match_crossings,
    max_cardinality_true_positives,
    restrict_to_adjudicated,
)
from trafficlens.bench.slitscan import Crossing
from trafficlens.core.gate import CrossingEvent
from trafficlens.detect.base import Detection

#: One frame of a detection stream: ``(frame_index, timestamp,
#: detections)``. Structurally identical to
#: ``trafficlens.bench.harness.FrameDetections``; spelled out here rather
#: than imported because importing ``harness`` would drag the tracker in
#: and break the seam the module docstring describes.
FrameDetections = tuple[int, float, list[Detection]]

#: The one seed every protocol in this family derives from. Recorded in
#: the report so a reader can reproduce any row.
ROBUSTNESS_SEED = 20260815

PROTOCOL_FRAME_RATE = "frame_rate"
PROTOCOL_DROPPED_FRAMES = "dropped_frames"
PROTOCOL_DETECTION_DROPOUT = "detection_dropout"
PROTOCOL_BOX_JITTER = "box_jitter"

#: ``z`` such that ``P(|Z| <= z) = 0.95`` for a standard normal, used to
#: convert a measured p95 of an absolute residual into a standard
#: deviation. Hard-coded rather than taken from scipy so this module keeps
#: its numpy-and-stdlib dependency floor.
NORMAL_P95_ABS_Z = 1.959963984540054

_SQRT2 = math.sqrt(2.0)

#: Above this many false alarms the per-level record publishes the count
#: alone and sets ``false_positive_frames`` to null. The per-frame rule
#: emits hundreds of them; listing every frame would bury the report's
#: real content under one method's known failure mode.
FALSE_POSITIVE_FRAME_LIMIT = 25

__all__ = [
    "FALSE_POSITIVE_FRAME_LIMIT",
    "FrameDetections",
    "NORMAL_P95_ABS_Z",
    "PROTOCOL_BOX_JITTER",
    "PROTOCOL_DETECTION_DROPOUT",
    "PROTOCOL_DROPPED_FRAMES",
    "PROTOCOL_FRAME_RATE",
    "ROBUSTNESS_SEED",
    "DegradedStream",
    "corner_sigma_equivalents",
    "decimate",
    "drop_detections",
    "drop_frames",
    "dropout_streams",
    "dropped_frame_streams",
    "frame_rate_streams",
    "jitter_boxes",
    "jitter_streams",
    "map_events_to_source",
    "run_protocol",
    "run_stream",
    "stream_rng",
    "widen_for_gap",
    "window_resolution",
]


# --- seeding ----------------------------------------------------------------


def stream_rng(seed: int, protocol: str, level: float) -> np.random.Generator:
    """The generator for one (protocol, level) pair.

    Derived from the seed together with a stable digest of the protocol
    name and the level VALUE, so that every level is independent of every
    other and of the order the sweep happens to visit them in. A single
    generator consumed across a sweep would make each level's numbers
    depend on which levels preceded it, and inserting one point would
    silently rewrite the rest of the published table.

    ``blake2b`` rather than ``hash()``: the built-in hash of a string is
    randomised per process, so a seed derived from it would not reproduce
    across runs at all.
    """
    tag = hashlib.blake2b(
        f"{protocol}:{float(level):.10g}".encode("utf-8"), digest_size=8
    ).digest()
    return np.random.default_rng([int(seed), int.from_bytes(tag, "big")])


# --- one degraded stream ----------------------------------------------------


@dataclass(frozen=True)
class DegradedStream:
    """One degradation level's stream, and everything needed to score it.

    ``frames`` is renumbered from zero -- see the module docstring --
    while ``source_frames`` holds each retained frame's ORIGINAL index, so
    predictions can be mapped back onto the ground truth's clock.
    ``max_gap`` is the largest inter-sample gap the retained pattern
    realises, in original frames, and is what the match window widens by.
    """

    protocol: str
    level: float
    level_label: str
    frames: tuple[FrameDetections, ...]
    source_frames: tuple[int, ...]
    max_gap: int
    seed: int | None
    frames_total: int
    detections_kept: int
    detections_total: int
    detail: dict

    @property
    def frames_kept(self) -> int:
        return len(self.frames)

    def as_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "level": self.level,
            "level_label": self.level_label,
            "seed": self.seed,
            "frames_kept": self.frames_kept,
            "frames_total": self.frames_total,
            "detections_kept": self.detections_kept,
            "detections_total": self.detections_total,
            "max_gap_frames": self.max_gap,
            "detail": dict(self.detail),
        }


def _max_gap(source_frames: Sequence[int]) -> int:
    """The largest step between consecutive retained frames, in original
    frames. A stream that retained everything contiguous reports 1."""
    if len(source_frames) < 2:
        return 1
    return int(max(np.diff(np.asarray(source_frames, dtype=np.int64))))


def _assemble(
    source: Sequence[FrameDetections],
    positions: Sequence[int],
    per_frame: Sequence[list[Detection]],
    *,
    protocol: str,
    level: float,
    level_label: str,
    seed: int | None,
    detail: dict,
) -> DegradedStream:
    """Build a stream from the retained ``positions`` of ``source``, using
    the (possibly already degraded) detection lists in ``per_frame``.

    Every protocol goes through here, including at its identity level:
    there is no short-circuit for "no degradation", so the reduction test
    proves the transform rather than proving a branch.
    """
    if not positions:
        raise ValueError(
            f"{protocol} at {level_label} retained no frames at all; a stream "
            f"with nothing in it cannot be scored against labels"
        )
    frames = tuple(
        (output_index, source[position][1], list(per_frame[position]))
        for output_index, position in enumerate(positions)
    )
    source_frames = tuple(source[position][0] for position in positions)
    return DegradedStream(
        protocol=protocol,
        level=float(level),
        level_label=level_label,
        frames=frames,
        source_frames=source_frames,
        max_gap=_max_gap(source_frames),
        seed=seed,
        frames_total=len(source),
        detections_kept=sum(len(detections) for _i, _t, detections in frames),
        detections_total=sum(len(detections) for _i, _t, detections in source),
        detail=dict(detail),
    )


# --- protocol 1: frame-rate decimation --------------------------------------


def decimate(
    detections: Sequence[FrameDetections], *, source_fps: float, target_fps: float
) -> DegradedStream:
    """Keep the frames a ``target_fps`` recording of the same clip would
    have held.

    Output slot ``j`` takes source position ``floor(j * source_fps /
    target_fps)``. The stride is computed in exact rational arithmetic, so
    a non-integer ratio such as 30 -> 25 fps produces the same grid on
    every platform instead of drifting on the last bit of a float: five
    frames in every six, gaps of 1 and 2, and a realised worst gap of 2
    that the match window is widened by. A nominal stride of 1.2 is not a
    number of frames and is never used as one.
    """
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if target_fps > source_fps:
        raise ValueError(
            f"cannot decimate {source_fps} fps footage to {target_fps} fps: "
            f"this protocol removes frames, it does not invent them"
        )
    stride = Fraction(source_fps).limit_denominator(10_000) / Fraction(
        target_fps
    ).limit_denominator(10_000)

    positions: list[int] = []
    slot = 0
    while True:
        position = math.floor(stride * slot)
        if position >= len(detections):
            break
        if not positions or position > positions[-1]:
            positions.append(position)
        slot += 1

    return _assemble(
        detections,
        positions,
        [list(frame[2]) for frame in detections],
        protocol=PROTOCOL_FRAME_RATE,
        level=float(target_fps),
        level_label=f"{float(target_fps):g} fps",
        seed=None,
        detail={
            "source_fps": float(source_fps),
            "target_fps": float(target_fps),
            "nominal_stride": float(stride),
        },
    )


def frame_rate_streams(
    detections: Sequence[FrameDetections],
    *,
    source_fps: float,
    target_rates: Iterable[float],
) -> list[DegradedStream]:
    """The frame-rate sweep. Deterministic and unseeded: which frames a
    lower rate keeps is a property of the grid, not of a draw."""
    return [
        decimate(detections, source_fps=source_fps, target_fps=float(rate))
        for rate in target_rates
    ]


# --- protocol 2: dropped frames ---------------------------------------------


def drop_frames(
    detections: Sequence[FrameDetections],
    *,
    fraction: float,
    seed: int = ROBUSTNESS_SEED,
) -> DegradedStream:
    """Delete a seeded random ``fraction`` of the frames outright.

    Unlike decimation the surviving grid is irregular, so the window is
    widened by the realised worst gap rather than by a stride. That gap is
    computed from the drop pattern, before any scoring happens, so it can
    never be tuned against an engine's output.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(
            f"fraction must lie in [0, 1) -- dropping every frame leaves "
            f"nothing to score -- got {fraction}"
        )
    total = len(detections)
    rng = stream_rng(seed, PROTOCOL_DROPPED_FRAMES, fraction)
    to_drop = int(round(total * fraction))
    dropped = set(rng.choice(total, size=to_drop, replace=False).tolist())
    positions = [index for index in range(total) if index not in dropped]

    return _assemble(
        detections,
        positions,
        [list(frame[2]) for frame in detections],
        protocol=PROTOCOL_DROPPED_FRAMES,
        level=float(fraction),
        level_label=f"{fraction:.0%} dropped",
        seed=seed,
        detail={
            "fraction_requested": float(fraction),
            "frames_dropped": len(dropped),
            "fraction_realised": (len(dropped) / total) if total else 0.0,
        },
    )


def dropped_frame_streams(
    detections: Sequence[FrameDetections],
    *,
    fractions: Iterable[float],
    seed: int = ROBUSTNESS_SEED,
) -> list[DegradedStream]:
    return [
        drop_frames(detections, fraction=float(fraction), seed=seed)
        for fraction in fractions
    ]


# --- protocol 3: detection dropout ------------------------------------------


def drop_detections(
    detections: Sequence[FrameDetections],
    *,
    probability: float,
    seed: int = ROBUSTNESS_SEED,
) -> DegradedStream:
    """Drop each detection independently with ``probability``, simulating
    occlusion.

    Every frame is kept, so the sampling grid -- and therefore the match
    window -- is untouched: what changes is how often a track has nothing
    to associate with. Detections are only ever removed, never reordered,
    never moved between frames and never invented; a transform that
    shuffled would hit the same count and change every tracker's
    association.

    At ``probability = 0`` the keep test ``uniform >= 0`` is satisfied by
    every draw, so the identity falls out of the general path rather than
    out of a special case.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must lie in [0, 1], got {probability}")
    rng = stream_rng(seed, PROTOCOL_DETECTION_DROPOUT, probability)

    per_frame: list[list[Detection]] = []
    for _index, _timestamp, frame_detections in detections:
        if not frame_detections:
            per_frame.append([])
            continue
        keep = rng.random(len(frame_detections)) >= probability
        per_frame.append(
            [
                detection
                for detection, keeping in zip(frame_detections, keep)
                if keeping
            ]
        )

    total = sum(len(frame[2]) for frame in detections)
    kept = sum(len(frame) for frame in per_frame)
    return _assemble(
        detections,
        list(range(len(detections))),
        per_frame,
        protocol=PROTOCOL_DETECTION_DROPOUT,
        level=float(probability),
        level_label=f"p={probability:.2f}",
        seed=seed,
        detail={
            "probability_requested": float(probability),
            "detections_dropped": total - kept,
            "fraction_realised": ((total - kept) / total) if total else 0.0,
        },
    )


def dropout_streams(
    detections: Sequence[FrameDetections],
    *,
    probabilities: Iterable[float],
    seed: int = ROBUSTNESS_SEED,
) -> list[DegradedStream]:
    return [
        drop_detections(detections, probability=float(probability), seed=seed)
        for probability in probabilities
    ]


# --- protocol 4: box jitter -------------------------------------------------


def jitter_boxes(
    detections: Sequence[FrameDetections],
    *,
    sigma_px: float,
    seed: int = ROBUSTNESS_SEED,
) -> DegradedStream:
    """Add independent Gaussian noise of standard deviation ``sigma_px``
    to each of a box's four corner coordinates.

    Four independent draws, not one shared offset. A translation would
    leave every box exactly the size it was, so the detector's largest
    measured residual -- box width -- would never be perturbed and the
    sweep would be calibrated against a quantity it does not touch. Per
    corner noise of ``sigma`` inflates the width residual by
    ``sigma * sqrt(2)`` and the centre residual by ``sigma / sqrt(2)``;
    ``corner_sigma_equivalents`` is that algebra inverted, and is how the
    measured clip is placed on this sweep.

    Only geometry is degraded. Confidence and class are a different
    failure and mixing them in here would make this curve unattributable.

    A corner that overshoots its opposite number is re-ordered rather than
    clamped, so a box is always well formed; the count of boxes that
    needed it is published, because a sweep in which most boxes are
    inverted is measuring the re-ordering rule and not the noise.

    At ``sigma_px = 0`` every draw is exactly zero, so the identity falls
    out of the general path rather than out of a special case.
    """
    if sigma_px < 0.0:
        raise ValueError(
            f"sigma_px is a standard deviation and cannot be negative, got "
            f"{sigma_px}"
        )
    rng = stream_rng(seed, PROTOCOL_BOX_JITTER, sigma_px)

    per_frame: list[list[Detection]] = []
    reordered = 0
    for _index, _timestamp, frame_detections in detections:
        if not frame_detections:
            per_frame.append([])
            continue
        noise = rng.normal(0.0, float(sigma_px), size=(len(frame_detections), 4))
        jittered: list[Detection] = []
        for detection, (dx1, dy1, dx2, dy2) in zip(frame_detections, noise):
            left = detection.x1 + float(dx1)
            top = detection.y1 + float(dy1)
            right = detection.x2 + float(dx2)
            bottom = detection.y2 + float(dy2)
            if left > right or top > bottom:
                reordered += 1
            jittered.append(
                replace(
                    detection,
                    x1=min(left, right),
                    y1=min(top, bottom),
                    x2=max(left, right),
                    y2=max(top, bottom),
                )
            )
        per_frame.append(jittered)

    return _assemble(
        detections,
        list(range(len(detections))),
        per_frame,
        protocol=PROTOCOL_BOX_JITTER,
        level=float(sigma_px),
        level_label=f"sigma={float(sigma_px):g} px",
        seed=seed,
        detail={
            "sigma_px": float(sigma_px),
            "noise": "independent Gaussian per corner coordinate",
            "boxes_reordered": reordered,
        },
    )


def jitter_streams(
    detections: Sequence[FrameDetections],
    *,
    sigmas_px: Iterable[float],
    seed: int = ROBUSTNESS_SEED,
) -> list[DegradedStream]:
    return [
        jitter_boxes(detections, sigma_px=float(sigma), seed=seed)
        for sigma in sigmas_px
    ]


# --- calibrating the jitter sweep against a measurement ---------------------


def corner_sigma_equivalents(measured: dict) -> dict:
    """The per-corner ``sigma`` this module's jitter would have to use to
    reproduce a measured residual, for every quantity the measurement
    reports.

    The sweep's knob is a per-corner standard deviation; the measurement
    in ``reports/detection_noise.json`` is a residual of box width, box
    height and box centre. This is the bridge between them:

    - width and height are differences of two independent corners, so
      their residual is inflated by ``sqrt(2)``  ->  ``sigma = std / sqrt(2)``
    - a centre is the mean of two independent corners, so its residual is
      damped by ``sqrt(2)``  ->  ``sigma = std * sqrt(2)``

    Both statistics are converted, and they are expected to DISAGREE: the
    measured distribution is heavy-tailed, with a standard deviation above
    its own p95 for the centres, so a Gaussian read off the tail and a
    Gaussian read off the bulk are two different Gaussians. Publishing the
    range rather than a single number is the honest form, and neither end
    of it is a measurement of the detector -- the source report says why.
    """
    residuals = measured.get("residuals", {})
    factors = {
        "box_width": 1.0 / _SQRT2,
        "box_height": 1.0 / _SQRT2,
        "centre_x": _SQRT2,
        "centre_y": _SQRT2,
    }
    from_std: dict[str, float] = {}
    from_p95: dict[str, float] = {}
    for name, factor in factors.items():
        record = residuals.get(name)
        if not record:
            continue
        from_std[name] = float(record["std_px"]) * factor
        from_p95[name] = (
            float(record["p95_abs_px"]) / NORMAL_P95_ABS_Z
        ) * factor

    values = list(from_std.values()) + list(from_p95.values())
    return {
        "method": (
            "Per-corner sigma reproducing each measured residual: width and "
            "height are differences of two independent corners (sigma = std / "
            "sqrt(2)), a centre is their mean (sigma = std * sqrt(2)). p95 of "
            "an absolute residual is converted with the standard normal's "
            f"{NORMAL_P95_ABS_Z:.6f}. The two disagree because the measured "
            "distribution is heavy-tailed, not Gaussian; the spread between "
            "them is the honest uncertainty and neither end is a measurement "
            "of the detector."
        ),
        "from_std": from_std,
        "from_p95": from_p95,
        "min_px": min(values) if values else 0.0,
        "max_px": max(values) if values else 0.0,
    }


# --- the match window, widened by the realised sampling gap -----------------


def widen_for_gap(window: MatchWindow, max_gap: int) -> MatchWindow:
    """``window`` with its LATE side widened by ``max_gap - 1`` frames.

    Derived a priori from the sampling grid and never fitted to output: a
    crossing at original frame ``f`` cannot be reported before the next
    retained frame, which lies in ``[f, f + max_gap - 1]``. The anchor lag
    the undegraded window already encodes is added on top, unchanged. The
    early side never moves.
    """
    if max_gap < 1:
        raise ValueError(
            f"a sampling gap is at least one frame, got {max_gap}; a gap "
            f"below one would narrow the window rather than widen it"
        )
    return MatchWindow(
        frames_before=window.frames_before,
        frames_after=window.frames_after + max_gap - 1,
        reason=(
            f"Widened on the late side only, by a realised maximum "
            f"inter-sample gap of {max_gap} original frames. A method that "
            f"sees this sampling grid cannot report a crossing before the "
            f"next retained frame, so the late side absorbs the "
            f"quantisation; the gap is computed from the retained pattern "
            f"before any scoring, never fitted to an engine's output, and "
            f"the early side is untouched. Underneath it the undegraded "
            f"window is unchanged: {window.reason}"
        ),
    )


def window_resolution(
    labels: Sequence[Crossing],
    window: MatchWindow,
    *,
    baseline_window: MatchWindow = DEFAULT_MATCH_WINDOW,
) -> dict:
    """How much separating power the label set retains under ``window``.

    A widened window is not free. Two labels whose windows intersect can
    have a prediction that is eligible for either, and one-to-one matching
    bounds the damage without removing it. On the shipped label set the
    closest pair is five frames apart, so their windows already share one
    frame undegraded -- the widening enlarges an overlap that was always
    there rather than creating one.

    ``resolution_limited`` is therefore defined against the UNDEGRADED
    overlap, not against zero: it is true when this window shares more
    frames between some pair of labels than the baseline window did.
    """
    frames = sorted(label.frame for label in labels)
    width = window.frames_before + window.frames_after + 1

    def overlaps(active: MatchWindow) -> tuple[int, int]:
        pairs = 0
        worst = 0
        for i in range(len(frames)):
            low_i = frames[i] - active.frames_before
            high_i = frames[i] + active.frames_after
            for j in range(i + 1, len(frames)):
                low_j = frames[j] - active.frames_before
                high_j = frames[j] + active.frames_after
                shared = min(high_i, high_j) - max(low_i, low_j) + 1
                if shared > 0:
                    pairs += 1
                    worst = max(worst, shared)
        return pairs, worst

    pairs, worst = overlaps(window)
    _baseline_pairs, baseline_worst = overlaps(baseline_window)

    separations = [b - a for a, b in zip(frames, frames[1:])]
    closest = int(np.argmin(separations)) if separations else None

    return {
        "window_frames": width,
        "min_separation_frames": min(separations) if separations else None,
        "closest_pair_frames": (
            [frames[closest], frames[closest + 1]] if closest is not None else None
        ),
        "overlapping_label_pairs": pairs,
        "max_overlap_frames": worst,
        "max_overlap_fraction": worst / width if width else 0.0,
        "resolution_limited": worst > baseline_worst,
    }


# --- mapping predictions back onto the ground truth's clock -----------------


def map_events_to_source(
    events: Sequence[CrossingEvent], source_frames: Sequence[int]
) -> list[CrossingEvent]:
    """Re-index ``events`` from the degraded stream's numbering back into
    the original 30 fps stream the labels live in.

    Without this the labels and the predictions are on two different
    clocks and every score is noise. Only the frame index moves; the
    timestamp was never renumbered because decimating a clip does not
    change when anything happened.
    """
    return [
        replace(event, frame_index=source_frames[event.frame_index])
        for event in events
    ]


# --- scoring one degraded stream --------------------------------------------


def _record(
    joint: MatchResult,
    adjudicated: MatchResult,
    ignored: int,
    events: Sequence[CrossingEvent],
    labels: Sequence[Crossing],
    window: MatchWindow,
    gate_name: str,
) -> dict:
    """One method's figures at one degradation level.

    Crossing-level precision, recall and F1 come FIRST and the count error
    comes alongside, never instead: a count error alone is the metric this
    benchmark exists to discredit -- Task 14's band rule predicted 18
    crossings against 17 real ones, a near-perfect total, while landing
    one or two of them on the right frame.
    """
    deltas = [delta for _prediction, _label, delta in joint.matches]
    false_positive_frames = [
        joint.predictions[index].frame_index for index in joint.false_positives
    ]
    return {
        "n_predicted": joint.n_predicted,
        "n_ground_truth": joint.n_ground_truth,
        "true_positives": joint.true_positives,
        "false_positives": len(joint.false_positives),
        "misses": len(joint.misses),
        "precision": joint.precision,
        "recall": joint.recall,
        "f1": joint.f1,
        "count_error": joint.count_error,
        "signed_bias": joint.signed_bias,
        "miss_rate": joint.miss_rate,
        "phantom_rate": joint.phantom_rate,
        "max_cardinality_true_positives": max_cardinality_true_positives(
            events, labels, window, gate_name=gate_name
        ),
        "miss_frames": [joint.labels[index].frame for index in joint.misses],
        # Bounded on purpose: the per-frame rule emits hundreds of these
        # and the count is the finding, not the list.
        "false_positive_frames": (
            false_positive_frames
            if len(false_positive_frames) <= FALSE_POSITIVE_FRAME_LIMIT
            else None
        ),
        "matched_frame_delta": {
            "mean": (sum(deltas) / len(deltas)) if deltas else None,
            "min": min(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
        },
        "certain_only": {
            "n_predicted": adjudicated.n_predicted,
            "n_ground_truth": adjudicated.n_ground_truth,
            "true_positives": adjudicated.true_positives,
            "precision": adjudicated.precision,
            "recall": adjudicated.recall,
            "f1": adjudicated.f1,
            "predictions_moved_to_ignore": ignored,
        },
    }


def run_stream(
    stream: DegradedStream,
    methods: dict[str, Callable[[Sequence[FrameDetections]], list[CrossingEvent]]],
    labels: Sequence[Crossing],
    *,
    gate_name: str,
    base_window: MatchWindow = DEFAULT_MATCH_WINDOW,
) -> dict:
    """Run every method over ONE degraded stream and score each against
    ``labels`` in original frame indices.

    The stream is a value, built once and handed to every method
    unchanged. That is the same structural guarantee the counting
    benchmark makes about its detections, for the same reason: a method
    that saw its own draw of noise would differ from another by the draw
    rather than by the method, and no downstream number would say so.
    """
    window = widen_for_gap(base_window, stream.max_gap)
    records: dict[str, dict] = {}
    for name, method in methods.items():
        events = map_events_to_source(method(stream.frames), stream.source_frames)
        joint = match_crossings(events, labels, window, gate_name=gate_name)
        adjudicated, ignored = restrict_to_adjudicated(joint)
        records[name] = _record(
            joint, adjudicated, ignored, events, labels, window, gate_name
        )

    entry = stream.as_dict()
    entry["match_window"] = window.as_dict()
    entry["window_widened_by_frames"] = window.frames_after - base_window.frames_after
    entry["resolution"] = window_resolution(
        labels, window, baseline_window=base_window
    )
    entry["methods"] = records
    return entry


def run_protocol(
    streams: Sequence[DegradedStream],
    methods: dict[str, Callable[[Sequence[FrameDetections]], list[CrossingEvent]]],
    labels: Sequence[Crossing],
    *,
    gate_name: str,
    base_window: MatchWindow = DEFAULT_MATCH_WINDOW,
) -> list[dict]:
    """Score a whole sweep, one entry per level, in the order given."""
    return [
        run_stream(
            stream, methods, labels, gate_name=gate_name, base_window=base_window
        )
        for stream in streams
    ]
