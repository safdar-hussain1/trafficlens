#!/usr/bin/env python3
"""Measure tracking quality only where an identity error CHANGES AN OUTPUT.

There is no frame-by-frame identity label set for this clip and this task
does not produce one, so nothing here is an ID-switch count, an IDF1, a
MOTA or an identity-preservation figure -- see ``claims_not_made`` in the
report, which states the refusal and the reason so an absence cannot later
be read as a claim. A tracker cannot grade its own identities, and a proxy
derived from the tracker's own output would be exactly that.

What CAN be measured against the Task 12 ground truth is what identity
errors cost the two products the gate actually emits: how many distinct
predicted identities the clip's labelled vehicles are spent on, and whether
a matched crossing carries the class the labeller recorded. Both are scored
against the same 17 hand-labelled crossings, for the engine's tracker and
for both baseline trackers, with the counting rule held fixed at the gate
rule so a difference is attributable to the TRACKER.

Two readings of "fragmentation", and why both are published
-----------------------------------------------------------
The obvious reading -- count the distinct track IDs that EMITTED a gate
crossing -- cannot measure fragmentation on this clip, and that is a fact
about ``GateCounter`` rather than an opinion. It fires once per identity, on
a side change, so for a trajectory that crosses the gate ONCE only one half
of a split track can change side: the first half emits the crossing while
the second begins with no previous anchor to cross from, and the count goes
DOWN or stays put. Every labelled crossing on this clip is that
single-traversal case. It is NOT a universal impossibility -- a vehicle that
traverses the gate twice, split between the traversals, doubles this count
from 1 to 2, because each identity may latch once -- and
``tests/test_bench_tracking.py`` pins both cases rather than letting the
narrower claim stand as the general one. The reading is published here as
``crossing_id_ratio`` precisely because it is the measurement that shows why
it cannot carry the claim.

The reading that can is ``fragmentation_ratio``: distinct predicted
identities that REACH THE GATE, over the labelled vehicle count. A split
track puts two identities at the gate where the clip has one vehicle, and
the ratio reads 2.0. Its exact definition, its denominator and its edge
cases are in ``metric_definitions`` in the report and in
``gate_region_definition`` below, fixed before any output was looked at.

Where the numbers come from
---------------------------
One git-ignored detection cache, the same one ``scripts/bench_counting.py``
writes and ``scripts/bench_robustness.py`` reads. The detector is never
constructed here: a tracking report whose stream came from a different
detector run could not be compared with ``counting_accuracy.json``, and
the reduction to it is how this family proves its protocols correct.

The clean clip can barely discriminate between trackers -- Task 14 measured
all three scoring within one event of each other, and one event is the
smallest step a 17-label set can express -- so the same metrics are also
swept across the four degradation protocols of
``trafficlens.bench.degrade``, at exactly the levels and seed
``reports/robustness.json`` used. Whether the clean clip discriminates at
all on THESE metrics is a measurement, published in
``questions.clean_clip_degeneracy`` and derived there rather than predicted
here. Every protocol's identity level reproduces the clean-clip record
exactly, through the general code path, and
``tests/test_bench_tracking.py`` asserts it field by field.

The gate region has a half-width, and the report's answers are not equally
robust to it, so ``gate_region_sweep`` publishes what every answer does
across 5-60 px. The published width stays ``BASELINE_BAND_PX`` -- a
constant with independent provenance in the counting benchmark -- because
choosing the width that produces the friendlier verdict would be tuning the
instrument to the answer. The sweep is published BECAUSE an answer depends
on it, not in order to pick one.

No wall-clock timing is published: per-method cost already lives in
``counting_accuracy.json``, and a timing column would make the property
this report most needs -- byte reproducibility -- untestable.

Writes ``reports/tracking.json``. Usage:

    PYTHONPATH=src .venv/bin/python scripts/bench_tracking.py \
        --config configs/motorway.yaml --gate inbound \
        --gt data/groundtruth/motorway_inbound_gt.json
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trafficlens.bench.baselines import (  # noqa: E402
    CentroidTracker,
    GreedyIoUTracker,
    _band_offset,
)
from trafficlens.bench.degrade import (  # noqa: E402
    PROTOCOL_BOX_JITTER,
    PROTOCOL_DETECTION_DROPOUT,
    PROTOCOL_DROPPED_FRAMES,
    PROTOCOL_FRAME_RATE,
    ROBUSTNESS_SEED,
    dropout_streams,
    dropped_frame_streams,
    frame_rate_streams,
    jitter_streams,
    map_events_to_source,
    widen_for_gap,
)
from trafficlens.bench.harness import (  # noqa: E402
    DetectionCacheError,
    read_detection_cache,
    run_counting,
    write_report,
)
from trafficlens.bench.scoring import (  # noqa: E402
    DEFAULT_MATCH_WINDOW,
    match_crossings,
)
from trafficlens.bench.slitscan import GroundTruth  # noqa: E402
from trafficlens.config import load_config  # noqa: E402
from trafficlens.core.constants import BASELINE_BAND_PX  # noqa: E402
from trafficlens.core.gate import GateCounter  # noqa: E402
from trafficlens.io.video import VideoSource  # noqa: E402
from trafficlens.track.tracker import Tracker  # noqa: E402

#: The schema version of ``reports/tracking.json``.
REPORT_SCHEMA_VERSION = 1

#: Half-width, in pixels, of the band around the gate SEGMENT inside which
#: a track counts as having reached the gate. ``BASELINE_BAND_PX`` rather
#: than a fresh number: it is the band the counting benchmark's band rule
#: already uses, so the region this metric is measured over is a value with
#: a stated provenance elsewhere in the project rather than one chosen here
#: to make a ratio come out. It is a knob, and ``gate_region_sweep``
#: publishes what every answer in the report does across a range of it; it
#: is NOT moved to whichever value produces a friendlier verdict.
DEFAULT_GATE_REGION_PX = BASELINE_BAND_PX

#: Half-widths the report publishes every answer at, so a reader can see
#: which conclusions survive the knob and which are conditional on it. The
#: published width is in the middle of the range, not at an end of it.
DEFAULT_REGION_SWEEP_PX = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0)

#: The three trackers compared, with the counting rule held fixed at the
#: gate rule. Holding the rule fixed and swapping the tracker is what makes
#: a difference attributable to the association step.
TRACKER_FACTORIES = {
    "engine": Tracker,
    "centroid": CentroidTracker,
    "greedy-iou": GreedyIoUTracker,
}

#: The counting rule every figure here is measured through.
COUNTING_RULE = "gate"


def gate_region_definition(region_px: float) -> str:
    """What "reaches the gate" means, in one place, for the report and for
    the code that implements it."""
    return (
        f"A predicted identity REACHES THE GATE when either of two things "
        f"is true of the track the tracker actually returned: its anchor "
        f"was observed inside the closed band of half-width "
        f"{float(region_px):g} px around the gate SEGMENT (the same band "
        f"geometry trafficlens.bench.baselines uses, so the foot of "
        f"perpendicular must lie between the gate's endpoints -- a vehicle "
        f"on another carriageway crossing the gate's infinite line past its "
        f"end does not count), or its swept path crossed the bounded gate "
        f"segment and the gate rule emitted a crossing for it. The second "
        f"clause is not decoration: under decimation a vehicle can move "
        f"further than the band's whole width between samples, and a "
        f"containment test alone would report that it never reached the "
        f"gate, collapsing the ratio for reasons that have nothing to do "
        f"with identity."
    )


@dataclass(frozen=True)
class GateApproach:
    """How close each predicted identity ever came to the gate, and which
    identities the gate rule counted.

    Recording the CLOSEST APPROACH rather than a yes/no band test is what
    lets the region's half-width be swept after the fact from ONE run of the
    trackers: band containment at half-width ``h`` is exactly
    ``closest_approach_px[track_id] <= h``, so every width in
    ``gate_region_sweep`` scores the identical association rather than a
    re-run of it. That is not an optimisation -- it is what makes the sweep
    a measurement of the KNOB and of nothing else.
    """

    #: Smallest absolute perpendicular distance, in pixels, at which each
    #: identity was observed while its foot of perpendicular lay within the
    #: gate segment. Identities never seen within the segment's span do not
    #: appear at all.
    closest_approach_px: dict[int, float]
    #: Identities the gate rule emitted a crossing for.
    crossing_track_ids: frozenset[int]

    def reaching(self, half_width_px: float = DEFAULT_GATE_REGION_PX) -> set[int]:
        """The identities that REACH THE GATE at this half-width -- see
        ``gate_region_definition`` for the two clauses."""
        return {
            track_id
            for track_id, distance in self.closest_approach_px.items()
            if distance <= float(half_width_px)
        } | set(self.crossing_track_ids)


class GateApproachObserver:
    """A transparent wrapper that records how close each track came to the
    gate, and changes nothing else.

    It exists so this is collected from inside the SAME
    ``trafficlens.bench.harness.run_counting`` every other benchmark in this
    repo drives -- the same track lifecycle, the same reaping, the same
    anchor bookkeeping -- rather than from a second loop that could drift
    from it. ``update`` returns the wrapped tracker's list unchanged and
    ``max_age`` is read off the wrapped tracker, so a baseline tracker with
    a different lifetime is still reaped on its own clock.

    The span test is not reimplemented here: ``_band_offset`` with an
    unbounded half-width IS the gate-segment span test with the distance
    test removed, so the one place in the project that decides whether a
    point is in front of the gate segment still decides it.

    No per-ID state is ever forgotten, because none needs to be: every
    tracker in this project allocates IDs strictly ascending and never
    recycles one, so an ID seen at the gate belongs to exactly one identity
    for the whole clip.
    """

    def __init__(self, tracker, gate):
        self._tracker = tracker
        self._gate = gate
        self.max_age = tracker.max_age
        self.closest_approach_px: dict[int, float] = {}

    def update(self, detections, frame_index):
        tracks = self._tracker.update(detections, frame_index)
        for track in tracks:
            offset = _band_offset(self._gate, track.anchor, math.inf)
            if offset is None:
                continue  # never in front of the gate segment at all
            distance = abs(offset)
            previous = self.closest_approach_px.get(track.track_id)
            if previous is None or distance < previous:
                self.closest_approach_px[track.track_id] = distance
        return tracks


def run_tracking(frames, tracker, gate) -> tuple[list, GateApproach]:
    """Play ``frames`` through ``tracker`` and the gate rule, returning
    ``(crossing_events, GateApproach)``.

    The events are exactly what ``run_counting`` produces for the same
    tracker and the same gate -- the observer is transparent, and a test
    asserts that on both a clean and a degraded stream.
    """
    observer = GateApproachObserver(tracker, gate)
    events = run_counting(frames, observer, GateCounter(gate))
    return events, GateApproach(
        closest_approach_px=dict(observer.closest_approach_px),
        crossing_track_ids=frozenset(event.track_id for event in events),
    )


def _ratio(numerator: int, denominator: int) -> float:
    """The scorer's own no-denominator convention: 0.0 rather than a raise
    or a NaN, with the counts always published alongside so a 0.0 that
    means "nothing to divide by" is never read as a measurement."""
    return numerator / denominator if denominator else 0.0


def fold_deviation(ratio: float) -> float | None:
    """How far a RATIO is from 1, measured multiplicatively:
    ``max(ratio, 1 / ratio)``, floor 1.0, read as "N times away from one
    predicted identity per labelled vehicle".

    The argument for it is a priori and has nothing to do with which answer
    it produces. The quantity is a ratio, so its natural distance from 1 is
    multiplicative: halving and doubling are the same size of error, and any
    honest distance must satisfy ``d(r) == d(1/r)``. The obvious
    ``abs(ratio - 1)`` does not -- it is bounded by 1.0 below (a ratio of 0
    scores exactly 1.0) and unbounded above, so under it a tracker that puts
    NO identity at the gate can never rank worse than one that fragments to
    2.0, and total association collapse is systematically flattered. This
    project's sweep reaches that regime, so the asymmetry is not academic.

    Monotone-equivalent to ``abs(log(ratio))`` -- ``max(r, 1/r) ==
    exp(abs(log r))`` -- so the two produce the identical ranking; the fold
    form is published for legibility, not for a different answer.

    ``None`` where the ratio is zero: the fold is infinite, because not one
    identity reached the gate. It ranks as the WORST possible value, which
    is what total collapse deserves and what ``abs(ratio - 1)`` could not
    express. Published as null rather than as a large number, so nobody
    averages it.
    """
    if ratio <= 0.0:
        return None
    return max(ratio, 1.0 / ratio)


def _absolute_deviation(ratio: float) -> float:
    """The superseded measure, kept only so the report can publish what the
    correction moved and by how much. See ``fold_deviation`` for why it was
    replaced."""
    return abs(ratio - 1.0)


def _worst(value: float | None) -> float:
    """A deviation as a sort key, with ``None`` ranking worst of all."""
    return math.inf if value is None else value


def score_tracking(events, gate_region_track_ids, labels, *, window, gate_name) -> dict:
    """One tracker's identity figures over one stream.

    Matching is ``trafficlens.bench.scoring.match_crossings`` -- the one
    matcher this project has, one-to-one, greedy nearest-frame-first,
    class-blind and direction-aware -- so class consistency here is the same
    quantity ``counting_accuracy.json`` publishes, over the same pairing,
    and the two can be compared directly.

    ``events`` must already be indexed in the ground truth's own frame
    numbering; ``map_events_to_source`` does that for a degraded stream.
    """
    result = match_crossings(events, labels, window, gate_name=gate_name)
    n_ground_truth = len(labels)
    region = len(set(gate_region_track_ids))
    crossing = len({event.track_id for event in events if event.gate == gate_name})
    fragmentation = _ratio(region, n_ground_truth)
    return {
        "gate_region_track_ids": region,
        "crossing_track_ids": crossing,
        "n_predicted": result.n_predicted,
        "true_positives": result.true_positives,
        "n_ground_truth": n_ground_truth,
        "fragmentation_ratio": fragmentation,
        # Multiplicative distance from one identity per labelled vehicle.
        # BOTH directions are errors -- above 1.0 is extra identities at the
        # gate, below 1.0 is identities that never got there, which is what
        # total association collapse looks like -- and a RATIO's distance
        # from 1 has to be a fold, or the two directions are not comparable.
        # See fold_deviation.
        "identity_deviation": fold_deviation(fragmentation),
        "crossing_id_ratio": _ratio(crossing, n_ground_truth),
        "class_consistency": result.as_dict()["class_consistency"],
    }


def _score_level(
    frames,
    source_frames,
    gate,
    labels,
    *,
    gate_name: str,
    window,
    region_px: float,
) -> tuple[dict, dict]:
    """Every tracker over one stream: ``(records, approaches)``.

    The single path both the clean block and every degraded level go
    through, so the published record and the half-width sweep are derived
    from the same run of the same trackers and cannot disagree.
    """
    records: dict[str, dict] = {}
    approaches: dict[str, GateApproach] = {}
    for name, factory in TRACKER_FACTORIES.items():
        events, approach = run_tracking(frames, factory(), gate)
        if source_frames is not None:
            events = map_events_to_source(events, source_frames)
        records[name] = score_tracking(
            events,
            approach.reaching(region_px),
            labels,
            window=window,
            gate_name=gate_name,
        )
        approaches[name] = approach
    return records, approaches


def score_clean(
    detections,
    gate,
    labels,
    *,
    gate_name: str,
    window=DEFAULT_MATCH_WINDOW,
    region_px: float = DEFAULT_GATE_REGION_PX,
) -> dict:
    """Every tracker's record on the UNDEGRADED stream, scored with the
    undegraded window. This is what every protocol's identity level must
    reproduce exactly."""
    return _score_level(
        detections,
        None,
        gate,
        labels,
        gate_name=gate_name,
        window=window,
        region_px=region_px,
    )[0]


def score_stream(
    stream,
    gate,
    labels,
    *,
    gate_name: str,
    base_window=DEFAULT_MATCH_WINDOW,
    region_px: float = DEFAULT_GATE_REGION_PX,
) -> dict:
    """Every tracker's record over ONE degraded stream, scored in original
    frame indices with the window the realised sampling gap requires.

    The stream is a value, built once and handed to every tracker
    unchanged -- the same structural guarantee the robustness family makes,
    for the same reason: a tracker that saw its own draw of noise would
    differ from another by the draw rather than by the tracker.
    """
    return _score_stream(
        stream,
        gate,
        labels,
        gate_name=gate_name,
        base_window=base_window,
        region_px=region_px,
    )[0]


def _score_stream(
    stream, gate, labels, *, gate_name: str, base_window, region_px: float
) -> tuple[dict, dict]:
    window = widen_for_gap(base_window, stream.max_gap)
    records, approaches = _score_level(
        stream.frames,
        stream.source_frames,
        gate,
        labels,
        gate_name=gate_name,
        window=window,
        region_px=region_px,
    )
    entry = stream.as_dict()
    entry["match_window"] = window.as_dict()
    entry["window_widened_by_frames"] = window.frames_after - base_window.frames_after
    entry["trackers"] = records
    return entry, approaches


# --- the metric definitions, fixed before any output is looked at -----------


def metric_definitions(n_ground_truth: int, region_px: float) -> dict:
    """Every published quantity's exact definition, its denominator and its
    edge cases.

    In the report as data rather than in a commit message, because a number
    whose definition is decided after seeing the output is a number fitted
    to the output, and a definition a reader cannot find is one they will
    guess at.
    """
    return {
        "fragmentation_ratio": {
            "definition": (
                "Distinct predicted track IDs that reach the gate, divided "
                "by the labelled vehicle count. 1.0 is one predicted "
                "identity per labelled vehicle; 2.0 is two, which is what a "
                "track split at the gate produces."
            ),
            "numerator": "distinct predicted track IDs reaching the gate",
            "denominator": n_ground_truth,
            "denominator_definition": (
                "ALL labelled crossings, certain and probable alike -- not "
                "only the matched ones. Conditioning the denominator on the "
                "match would make this ratio move whenever the COUNTING "
                "RULE's success moved, and the whole point of holding the "
                "gate rule fixed is that a difference here is attributable "
                "to the tracker."
            ),
            "edge_cases": [
                "A track that reaches the gate twice counts ONCE: the "
                "numerator is a count of distinct identities, not of "
                "visits. On this clip GateCounter agrees -- it fires once "
                "per identity -- so the two readings do not diverge here.",
                "An identity matched to no label still counts in the "
                "numerator. The numerator counts predicted identities at "
                "the gate and the denominator counts labelled vehicles; "
                "neither is conditioned on the pairing. A ratio above 1.0 "
                "therefore does not distinguish a fragmented track from an "
                "unlabelled vehicle in the region, and with no identity "
                "labels for this clip nothing here can.",
                "A ratio BELOW 1.0 is an error too, not a good score: it "
                "means identities that never reached the gate at all, which "
                "is what total association collapse looks like. That is why "
                "identity_deviation, not the ratio, is what the comparison "
                "against crossing F1 is made on.",
            ],
            "not_a_claim": (
                "This is not an ID-switch count and not a proxy for one. It "
                "counts identities at the gate; it cannot say which vehicle "
                "an identity belonged to, because that would need the "
                "frame-by-frame identity labels this clip does not have."
            ),
        },
        "crossing_id_ratio": {
            "definition": (
                "Distinct predicted track IDs that EMITTED a gate crossing, "
                "divided by the labelled vehicle count. Published as the "
                "measurement that shows why it cannot carry the "
                "fragmentation claim on its own."
            ),
            "numerator": "distinct predicted track IDs emitting a crossing",
            "denominator": n_ground_truth,
            "why_it_cannot_measure_fragmentation": (
                "For a trajectory that crosses the gate ONCE, GateCounter "
                "fires once per identity on a side change, so only one half "
                "of a split track can change side: the first half emits the "
                "crossing while the second starts with no previous anchor to "
                "cross from, and a split moves this count DOWN or leaves it "
                "alone. Every labelled crossing on this clip is that "
                "single-traversal case, and a synthetic clip whose every "
                "track is deliberately split in half reads 2.0 on "
                "fragmentation_ratio and exactly its undegraded value here."
            ),
            "not_a_universal_impossibility": (
                "The clause above is about single-traversal trajectories and "
                "is NOT a general claim about splitting. A vehicle that "
                "traverses the gate TWICE, split between the traversals, "
                "doubles this count from 1 to 2: GateCounter latches once "
                "per track ID, so the second identity is free to fire. "
                "tests/test_bench_tracking.py pins both cases -- the "
                "single-traversal split that cannot double and the "
                "two-traversal split that does -- so the narrow claim cannot "
                "be read as the general one."
            ),
            "not_an_independent_measurement": (
                "In every record published here this numerator EQUALS "
                "n_predicted, because each identity emits at most one "
                "crossing and identities are never recycled. So "
                "crossing_id_ratio is exactly n_predicted / "
                f"{n_ground_truth} -- the count ratio the counting benchmark "
                "already publishes as signed_bias -- and not a second, "
                "independent view of it. Its value here is the CONTRAST with "
                "fragmentation_ratio, not the number itself."
            ),
        },
        "identity_deviation": {
            "definition": (
                "max(fragmentation_ratio, 1 / fragmentation_ratio): the "
                "MULTIPLICATIVE distance from one predicted identity per "
                "labelled vehicle, floor 1.0, read as 'N times away from "
                "one'. null where the ratio is zero -- an infinite fold, "
                "because not one identity reached the gate -- and null ranks "
                "as the worst possible value."
            ),
            "why_a_fold_and_not_abs": (
                "The quantity is a RATIO, so its natural distance from 1 is "
                "multiplicative: halving and doubling are the same size of "
                "error, and an honest distance must satisfy d(r) = d(1/r). "
                "abs(r - 1) does not. It is bounded by 1.0 below -- a ratio "
                "of 0 scores exactly 1.0 -- and unbounded above, so under it "
                "a tracker that puts NO identity at the gate can never rank "
                "worse than one that fragments to 2.0, and total association "
                "collapse is systematically flattered. This sweep reaches "
                "both regimes, so the asymmetry is not academic. The "
                "argument is a priori and independent of which answer the "
                "measure produces, which is what makes the change a "
                "correction rather than a retune; what it moved is published "
                "in questions.agreement_with_crossing_f1."
            ),
            "equivalent_to": (
                "abs(log(fragmentation_ratio)), monotonically: max(r, 1/r) = "
                "exp(abs(log r)). The two rank identically; the fold form is "
                "published because '2.83x away from one' is legible and a "
                "log is not."
            ),
        },
        "class_consistency": {
            "definition": (
                "Among MATCHED crossings only: how many carry the class the "
                "labeller recorded, and which confusions the rest are. "
                "Matching is class-blind by ruling, so a detector class "
                "error surfaces here once, as a confusion, instead of twice "
                "as a miss plus a false alarm -- which is why this is a "
                "separate axis from precision, recall and F1 and must never "
                "be folded into them."
            ),
            "denominator": "matched pairs at this level, per tracker",
            "matched_by": (
                "trafficlens.bench.scoring.match_crossings: one-to-one, "
                "greedy nearest-frame-first, class-blind, direction-aware, "
                "over the asymmetric window published beside every level. "
                "The same matcher and the same pairing counting_accuracy."
                "json publishes, so the clean figures here are directly "
                "comparable with it."
            ),
        },
        "gate_region": {
            # Stated ONCE, at the top level. A long definition published
            # verbatim twice is two definitions the moment one is edited.
            "definition": (
                "The region this ratio's numerator counts identities in. "
                "Stated in full, once, in the report's own top-level "
                "gate_region block -- see stated_in."
            ),
            "stated_in": "gate_region.definition",
            "half_width_px": float(region_px),
            "swept_in": "gate_region_sweep",
        },
    }


# --- the questions, answered from the published series ----------------------


def _degeneracy_answer(clean: dict) -> dict:
    """Whether these metrics can tell the three trackers apart on the clean
    clip at all.

    Task 14 measured all three within one event of each other with the gate
    rule, and one event is the smallest step a 17-label set can express, so
    a degenerate answer here is the EXPECTED result rather than a failure --
    but it has to be measured and stated, not assumed, and the verdict is
    derived from the numbers so it cannot keep claiming a tie that has
    stopped being true.
    """
    ratios = {name: record["fragmentation_ratio"] for name, record in clean.items()}
    consistency = {
        name: record["class_consistency"]["rate"] for name, record in clean.items()
    }
    spread = max(ratios.values()) - min(ratios.values())
    consistency_spread = max(consistency.values()) - min(consistency.values())
    discriminates = spread > 0.0

    listed = ", ".join(
        f"{name} {value:.4f}" for name, value in sorted(ratios.items())
    )
    if discriminates:
        verdict = (
            f"On the undegraded clip the fragmentation ratio DOES separate "
            f"the three trackers, by {spread:.4f}: "
            f"{listed}"
            f". That is a change from Task 14's finding that all three "
            f"scored identically with the gate rule, so it is the ratio -- "
            f"not the crossing count -- that sees the difference."
        )
    else:
        verdict = (
            f"On the undegraded clip this metric cannot tell the three "
            f"trackers apart: all three spend "
            f"{next(iter(ratios.values())):.4f} predicted identities per "
            f"labelled vehicle at the gate, and the class-consistency "
            f"spread is {consistency_spread:.4f}. That is the finding, not "
            f"a result: the clean clip was already known to be unable to "
            f"discriminate -- Task 14 measured all three trackers within "
            f"one event of each other with the gate rule, and one event is "
            f"the smallest step a "
            f"{next(iter(clean.values()))['n_ground_truth']}-label set can "
            f"express. Three identical numbers here measure the clip, not "
            f"the trackers, which is why the same metrics are swept across "
            f"the four degradation protocols below."
        )
    return {
        "question": (
            "Does either metric discriminate between the engine's tracker "
            "and the two baselines on the clean clip?"
        ),
        "fragmentation_ratio_by_tracker": ratios,
        "class_consistency_rate_by_tracker": consistency,
        "spread": spread,
        "class_consistency_spread": consistency_spread,
        "discriminates": discriminates,
        "verdict": verdict,
    }


def ratio_series(approaches_by_level: dict, n_ground_truth: int, half_width: float):
    """``{level: {tracker: fragmentation_ratio}}`` at one half-width.

    Everything the report answers about fragmentation is a function of this
    one structure, so the headline and the half-width sweep are computed by
    the same code from the same runs; a sweep with its own arithmetic could
    disagree with the figure it is supposed to be sweeping.
    """
    return {
        key: {
            name: _ratio(len(approach.reaching(half_width)), n_ground_truth)
            for name, approach in by_tracker.items()
        }
        for key, by_tracker in approaches_by_level.items()
    }


def _spreads(ratios: dict) -> dict:
    return {
        key: max(scores.values()) - min(scores.values())
        for key, scores in ratios.items()
    }


def _furthest_from_one(ratios: dict, tracker: str, deviation=fold_deviation):
    """Levels where ``tracker`` is (or ties for) the largest deviation.

    A TIE counts: with three trackers a level where all three deviate
    equally lists every one of them, so this is "is not beaten", not "is
    strictly worst". The tie rule is published beside the list.
    """
    return sorted(
        key
        for key, scores in ratios.items()
        if _worst(deviation(scores[tracker]))
        == max(_worst(deviation(value)) for value in scores.values())
    )


def _agreement_lists(
    ratios: dict, crossing_f1_by_level: dict, deviation=fold_deviation
):
    """Where this metric's worst tracker is also crossing F1's worst.

    Returns ``(compared, separating, agree, disagree, detail)``.
    """
    compared = sorted(set(ratios) & set(crossing_f1_by_level))
    separating = sorted(
        key
        for key in compared
        if max(crossing_f1_by_level[key].values())
        - min(crossing_f1_by_level[key].values())
        > 0.0
    )
    agree: list[str] = []
    disagree: list[str] = []
    detail: dict[str, dict] = {}
    for key in separating:
        f1 = crossing_f1_by_level[key]
        deviations = {name: deviation(value) for name, value in ratios[key].items()}
        worst_f1 = sorted(
            name for name, value in f1.items() if value == min(f1.values())
        )
        worst_identity = sorted(
            name
            for name, value in deviations.items()
            if _worst(value) == max(_worst(other) for other in deviations.values())
        )
        agrees = bool(set(worst_f1) & set(worst_identity))
        (agree if agrees else disagree).append(key)
        detail[key] = {
            "lowest_crossing_f1": worst_f1,
            "largest_identity_deviation": worst_identity,
            "agrees": agrees,
        }
    return compared, separating, agree, disagree, detail


def _separation_answer(ratios: dict, trackers, *, clean_spread: float) -> dict:
    """Whether degradation separates the trackers on the fragmentation
    ratio, and in which direction."""
    spreads = _spreads(ratios)
    widest = max(spreads.values()) if spreads else 0.0
    differing = sorted(key for key, spread in spreads.items() if spread > 0.0)
    engine_furthest = _furthest_from_one(ratios, "engine")
    engine_worst_where_it_matters = sorted(set(engine_furthest) & set(differing))

    if widest > 0.0:
        argmax = max(spreads, key=lambda key: spreads[key])
        verdict = (
            f"Degradation DOES separate the three trackers on this metric: "
            f"the widest fragmentation spread is {widest:.4f}, at {argmax}, "
            f"and they differ at {len(differing)} of the {len(spreads)} "
            f"levels measured. The engine's tracker sits furthest from one "
            f"identity per vehicle at {len(engine_worst_where_it_matters)} "
            f"of those {len(differing)} levels."
        )
    else:
        verdict = (
            f"Degradation does NOT separate the three trackers on this "
            f"metric at any of the {len(spreads)} levels measured, which is "
            f"the finding. Every level's fragmentation ratio is identical "
            f"across the engine's tracker and both baselines, so on this "
            f"clip, at this gate, the metric has no discriminating power "
            f"even where crossing F1 does."
        )
    # Derived from the CLEAN measurement, never predicted: the controller's
    # own note expected the clean clip to be unable to discriminate, and the
    # measurement falsified it while this sentence went on asserting it.
    opening = (
        f"The clean clip separates the trackers by only {clean_spread:.4f}. "
        if clean_spread > 0.0
        else "The clean clip cannot discriminate. "
    )
    return {
        "question": (
            f"{opening}Does degradation -- the condition a motion model "
            f"exists for -- separate them further on the fragmentation ratio?"
        ),
        "metric": "fragmentation_ratio",
        "fragmentation_ratio_by_level": ratios,
        "spread_by_level": spreads,
        "max_spread": widest,
        "levels_measured": len(ratios),
        "trackers_separate": widest > 0.0,
        "levels_where_trackers_differ": differing,
        "levels_where_the_engine_is_furthest_from_one": engine_furthest,
        "levels_where_the_engine_is_furthest_and_they_differ": (
            engine_worst_where_it_matters
        ),
        "furthest_from_one_measured_by": (
            "identity_deviation -- the multiplicative fold max(r, 1/r), so a "
            "tracker whose identities never reached the gate is not "
            "flattered against one that fragmented. See "
            "metric_definitions.identity_deviation."
        ),
        "furthest_from_one_tie_rule": (
            "A TIE counts as furthest. A level where two or three trackers "
            "deviate by exactly the same fold lists every one of them, so "
            "this list means 'the engine is not beaten', not 'the engine is "
            "strictly worst'. On the published width two of its entries are "
            "three-way ties at spread 0.0, which is why the intersection "
            "with levels_where_trackers_differ is published beside it and is "
            "the figure the verdict quotes."
        ),
        "verdict": verdict,
    }


def _agreement_answer(
    ratios: dict,
    crossing_f1_by_level: dict,
    source: str,
    *,
    sweep: dict,
    region_px: float,
) -> dict:
    """Whether this benchmark's identity metric ranks the trackers the way
    crossing F1 does.

    The session's standing result is that the engine's tracker scores LOWEST
    on crossing F1 at every degradation level where the three differ, while
    costing about 21x a baseline tracker's CPU. If a second, independent
    metric agrees, that is worth saying plainly. If it disagrees, the
    disagreement is the most valuable thing this task produces and must be
    published rather than reconciled away -- so both lists are here, and
    the verdict is derived from them.

    This answer is also the one that turned out to DEPEND on the gate
    region's half-width, so the verdict is explicitly conditional on it and
    names what survives the knob. That dependence is why ``sweep`` is
    published at all; it is not a menu to choose a verdict from.
    """
    compared, separating, agree, disagree, detail = _agreement_lists(
        ratios, crossing_f1_by_level
    )
    # The measure this one replaced, recomputed live from the same data
    # rather than recalled, so what the correction moved is on the record as
    # a measurement. See metric_definitions.identity_deviation.
    _c, _s, was_agree, was_disagree, _d = _agreement_lists(
        ratios, crossing_f1_by_level, deviation=_absolute_deviation
    )

    complete = [
        row["half_width_px"]
        for row in sweep["by_half_width"]
        if row["agreement"]["agrees_everywhere"]
    ]
    incomplete = [
        row["half_width_px"]
        for row in sweep["by_half_width"]
        if not row["agreement"]["agrees_everywhere"]
    ]
    named = sorted(
        {
            level
            for row in sweep["by_half_width"]
            for level in row["agreement"]["levels_where_the_two_disagree"]
        }
    )
    conditional = (
        f"This answer is CONDITIONAL on the gate region's half-width, and "
        f"gate_region_sweep is published because of it, not in order to "
        f"choose from it. Over the swept "
        f"{min(row['half_width_px'] for row in sweep['by_half_width']):g}-"
        f"{max(row['half_width_px'] for row in sweep['by_half_width']):g} px "
        f"the agreement is COMPLETE at "
        f"{', '.join(f'{value:g}' for value in complete) or 'no'} px and "
        f"incomplete at "
        f"{', '.join(f'{value:g}' for value in incomplete) or 'no'} px, and "
        f"the disagreeing level is not the same one at every width -- the "
        f"union across the sweep is {', '.join(named) or 'empty'}. The "
        f"published width stays {region_px:g} px because it is "
        f"BASELINE_BAND_PX, a constant with independent provenance in the "
        f"counting benchmark; moving it to the width that produces the "
        f"friendlier answer would be tuning the instrument to the answer. "
    )
    if disagree:
        verdict = (
            f"The two metrics agree at {len(agree)} of the "
            f"{len(separating)} levels where crossing F1 separates the "
            f"trackers and DISAGREE at {len(disagree)}. A disagreement is "
            f"reported, not reconciled: at those levels the tracker with "
            f"the lowest crossing F1 is not the one furthest from one "
            f"identity per labelled vehicle, which means the two are "
            f"measuring different failures. Neither is thereby wrong -- "
            f"crossing F1 charges a mistimed or missing crossing, while "
            f"this metric charges identities spent at the gate, and a "
            f"tracker can lose crossings without splitting a single track. "
            f"The disagreeing levels are named above so the claim can be "
            f"checked rather than taken. " + conditional + sweep["invariants_sentence"]
        )
    else:
        verdict = (
            f"The two metrics agree at all {len(agree)} levels where "
            f"crossing F1 separates the trackers: wherever a tracker scores "
            f"lowest on crossing F1 it is also the one furthest from one "
            f"predicted identity per labelled vehicle. That is a second, "
            f"independent measurement pointing the same way as "
            f"{source}'s -- not a stronger version of the same one, and it "
            f"does not add an identity claim this clip's labels cannot "
            f"support. " + conditional + sweep["invariants_sentence"]
        )
    return {
        "superseded_measure": {
            "was": "abs(fragmentation_ratio - 1.0)",
            "now": "max(fragmentation_ratio, 1 / fragmentation_ratio)",
            "agreements_under_the_superseded_measure": len(was_agree),
            "disagreements_under_the_superseded_measure": len(was_disagree),
            "levels_it_called_disagreements": was_disagree,
            "agreements_now": len(agree),
            "disagreements_now": len(disagree),
            "why": (
                "abs(r - 1) is bounded by 1.0 below and unbounded above, so "
                "it cannot rank a tracker whose identities never reached the "
                "gate as worse than one that fragmented -- and that "
                "asymmetry, not a mechanism, produced one of the "
                "disagreements it reported. The replacement is scale-"
                "symmetric because the quantity is a ratio; the argument is "
                "a priori and does not depend on which answer it gives. Both "
                "counts above are recomputed from the same data by this "
                "report, not recalled."
            ),
        },
        "question": (
            f"{source} finds the three trackers separating against the "
            f"engine on crossing F1 at every degradation level where they "
            f"differ, while the engine's Kalman plus second association "
            f"stage costs about 21x a baseline tracker's CPU. Does this "
            f"benchmark's identity metric rank them the same way?"
        ),
        "source": source,
        "criterion": (
            "At a level where crossing F1 separates the trackers, the two "
            "AGREE when the tracker with the lowest crossing F1 is also one "
            "of those with the largest identity_deviation -- the largest "
            "distance from one predicted identity per labelled vehicle. "
            "Deviation, not the signed ratio, because a tracker that loses "
            "every track scores a ratio below 1.0 and a signed comparison "
            "would rank that as the best identity behaviour on the sweep; "
            "and a FOLD deviation rather than abs(r - 1), because the "
            "quantity is a ratio -- see "
            "metric_definitions.identity_deviation and superseded_measure. "
            "A null deviation (no identity reached the gate at all) ranks "
            "worst. Ties on either side are compared as SETS, so a level "
            "where two trackers tie for worst counts as agreement when "
            "either of them is the identity metric's worst."
        ),
        "half_width_px": float(region_px),
        "swept_in": "gate_region_sweep",
        "levels_compared": compared,
        "levels_where_crossing_f1_separates": separating,
        "levels_where_the_two_agree": agree,
        "levels_where_the_two_disagree": disagree,
        "agrees_everywhere": not disagree,
        "detail": detail,
        "verdict": verdict,
    }


def _region_sweep(
    clean_approaches: dict,
    level_approaches: dict,
    n_ground_truth: int,
    crossing_f1_by_level: dict,
    *,
    region_px: float,
    half_widths=DEFAULT_REGION_SWEEP_PX,
) -> dict:
    """Every answer in this report, recomputed at a range of gate-region
    half-widths.

    The half-width is the one free parameter of the fragmentation metric, and
    the report's answers are NOT equally robust to it: the agreement question
    changes branch across this range. Publishing the sweep is how that gets
    said out loud instead of resting under a single number. It costs nothing
    -- every width scores the same recorded ``GateApproach`` values, so the
    trackers are run once and the sweep varies the knob and nothing else.

    The published width is not chosen from this table. It is
    ``BASELINE_BAND_PX``, fixed by the counting benchmark, and the row for it
    must reproduce the report's own headline exactly -- which a test asserts,
    so the sweep can never become a separate measurement that merely sits
    beside the figures it claims to sweep.
    """
    rows = []
    for half_width in half_widths:
        clean_ratios = {
            name: _ratio(len(approach.reaching(half_width)), n_ground_truth)
            for name, approach in clean_approaches.items()
        }
        ratios = ratio_series(level_approaches, n_ground_truth, half_width)
        spreads = _spreads(ratios)
        widest = max(spreads.values()) if spreads else 0.0
        differing = sorted(key for key, spread in spreads.items() if spread > 0.0)
        furthest = _furthest_from_one(ratios, "engine")
        _c, separating, agree, disagree, _d = _agreement_lists(
            ratios, crossing_f1_by_level
        )
        engine = [scores["engine"] for scores in ratios.values()]
        rows.append(
            {
                "half_width_px": float(half_width),
                "published": float(half_width) == float(region_px),
                "clean": {
                    "fragmentation_ratio_by_tracker": clean_ratios,
                    "spread": max(clean_ratios.values()) - min(clean_ratios.values()),
                },
                "max_spread": widest,
                "widest_spread_level": (
                    max(spreads, key=lambda key: spreads[key]) if spreads else None
                ),
                "levels_where_trackers_differ": len(differing),
                "levels_where_the_engine_is_furthest_and_they_differ": len(
                    set(furthest) & set(differing)
                ),
                "engine_fragmentation_ratio_min": min(engine) if engine else None,
                "engine_fragmentation_ratio_max": max(engine) if engine else None,
                "agreement": {
                    "levels_where_crossing_f1_separates": len(separating),
                    "levels_where_the_two_agree": len(agree),
                    "levels_where_the_two_disagree": disagree,
                    "agrees_everywhere": not disagree,
                },
            }
        )

    clean_spreads = {row["clean"]["spread"] for row in rows}
    shares = [
        (
            row["levels_where_the_engine_is_furthest_and_they_differ"],
            row["levels_where_trackers_differ"],
        )
        for row in rows
    ]
    invariants = (
        f"What does NOT depend on the half-width, across the whole swept "
        f"range: the engine is furthest from one identity per labelled "
        f"vehicle at {min(share for share, _total in shares)}-"
        f"{max(share for share, _total in shares)} of the "
        f"{min(total for _share, total in shares)}-"
        f"{max(total for _share, total in shares)} levels where the trackers "
        f"differ, at every width; the trackers separate under degradation at "
        f"every width; and the clean-clip spread takes "
        f"{len(clean_spreads)} distinct value(s) over the sweep "
        f"({', '.join(f'{value:.4f}' for value in sorted(clean_spreads))}). "
        f"Those are the claims this report rests on. The agreement branch is "
        f"not one of them, and is labelled conditional."
    )
    return {
        "why": (
            "The gate region's half-width is the fragmentation metric's one "
            "free parameter, and the agreement question's ANSWER changes "
            "branch across this range. The sweep is published so that "
            "dependence is visible rather than hidden under a single number. "
            "The published width is BASELINE_BAND_PX and is NOT selected "
            "from this table: choosing the width that produces the friendlier "
            "verdict would be tuning the instrument to the answer."
        ),
        "cost": (
            "Free. Each tracker is run once per level and its closest "
            "approach to the gate recorded per identity, so every width "
            "scores the identical association -- the sweep varies the knob "
            "and nothing else. Band containment at half-width h is exactly "
            "closest_approach_px <= h."
        ),
        "published_half_width_px": float(region_px),
        "half_widths_px": [float(value) for value in half_widths],
        "by_half_width": rows,
        "invariants_sentence": invariants,
    }


# --- assembling the report --------------------------------------------------


def claims_not_made() -> list[dict]:
    """What this report refuses to say, and why.

    The field exists so an absence cannot later be read -- or written up --
    as a claim, and ``tests/test_bench_tracking.py`` asserts both halves:
    that the refusal is here, and that no such figure appears under any key
    anywhere in the document.
    """
    return [
        {
            "claim": (
                "No ID-switch count, IDF1, MOTA, MOTP or any other "
                "identity-preservation figure is published, here or on any "
                "other surface of this project."
            ),
            "reason": (
                "All of them need a frame-by-frame identity label set -- "
                "which vehicle each box belongs to, over the whole labelled "
                "window -- and this clip has none. Producing one is a human "
                "adjudication of the same kind Task 12's 17 crossings "
                "required, at a far larger scale, and it was not done. No "
                "proxy is offered in its place: a proxy derived from the "
                "tracker's own output would be the tracker grading its own "
                "identities, which is not evidence."
            ),
        },
        {
            "claim": (
                "The fragmentation ratio is not a count of identity errors "
                "and must not be quoted as one."
            ),
            "reason": (
                "It counts predicted identities at the gate against "
                "labelled vehicles. A ratio above 1.0 is consistent with a "
                "split track AND with an unlabelled vehicle in the region, "
                "and without identity labels nothing here can separate the "
                "two. It measures what identity behaviour COSTS at the "
                "gate, not how many switches happened."
            ),
        },
        {
            "claim": (
                "Class consistency is not a measurement of the classifier."
            ),
            "reason": (
                "It is scored over matched crossings only, so it sees one "
                "class decision per matched vehicle at one moment, on the "
                "17 labelled crossings of one gate of one clip. A "
                "confusion here is a detector class error the counting "
                "rule happened to carry into an event; the population is "
                "far too small, and far too selected, to be a class "
                "accuracy."
            ),
        },
        {
            "claim": (
                "Nothing here is evidence about tracking on the far "
                "carriageway, in the distance, or in a queue."
            ),
            "reason": (
                "The labelling gate was chosen in the near field for label "
                "RELIABILITY. Its traffic is the largest, best separated "
                "and least foreshortened in the frame, which is the "
                "easiest case for association as well as for labelling, so "
                "every figure here is an UPPER bound."
            ),
        },
    ]


def build_report(
    detections,
    truth: GroundTruth,
    gate,
    *,
    detector: dict | None,
    target_rates,
    drop_fractions,
    dropout_probabilities,
    jitter_sigmas,
    crossing_f1_by_level: dict,
    crossing_f1_source: str,
    seed: int = ROBUSTNESS_SEED,
    region_px: float = DEFAULT_GATE_REGION_PX,
    base_window=DEFAULT_MATCH_WINDOW,
) -> dict:
    """The whole tracking-quality report, scored and assembled.

    Deterministic end to end: no timing is measured, and every level draws
    its stream from the seed together with the protocol name and the level
    VALUE, so two runs over the same detections produce byte-identical
    JSON and adding a sweep point cannot move another point's numbers.
    """
    labels = list(truth.crossings)
    target_rates = [float(rate) for rate in target_rates]
    if not target_rates or target_rates[0] != float(truth.fps):
        raise ValueError(
            f"the decimation sweep must lead with the clip's own rate "
            f"({truth.fps} fps), because that is the identity level the "
            f"whole family reduces to; got {target_rates[:1]}"
        )

    clean, clean_approaches = _score_level(
        detections,
        None,
        gate,
        labels,
        gate_name=truth.gate_name,
        window=base_window,
        region_px=region_px,
    )

    protocols = {
        PROTOCOL_FRAME_RATE: {
            "knob": "target_fps",
            "levels": target_rates,
            "seeded": False,
            "streams": frame_rate_streams(
                detections, source_fps=truth.fps, target_rates=target_rates
            ),
        },
        PROTOCOL_DROPPED_FRAMES: {
            "knob": "fraction_dropped",
            "levels": [float(value) for value in drop_fractions],
            "seeded": True,
            "streams": dropped_frame_streams(
                detections, fractions=drop_fractions, seed=seed
            ),
        },
        PROTOCOL_DETECTION_DROPOUT: {
            "knob": "dropout_probability",
            "levels": [float(value) for value in dropout_probabilities],
            "seeded": True,
            "streams": dropout_streams(
                detections, probabilities=dropout_probabilities, seed=seed
            ),
        },
        PROTOCOL_BOX_JITTER: {
            "knob": "corner_sigma_px",
            "levels": [float(value) for value in jitter_sigmas],
            "seeded": True,
            "streams": jitter_streams(detections, sigmas_px=jitter_sigmas, seed=seed),
        },
    }
    level_approaches: dict[str, dict] = {}
    for protocol, block in protocols.items():
        entries = []
        for stream in block.pop("streams"):
            entry, approaches = _score_stream(
                stream,
                gate,
                labels,
                gate_name=truth.gate_name,
                base_window=base_window,
                region_px=region_px,
            )
            entries.append(entry)
            level_approaches[f"{protocol}@{entry['level_label']}"] = approaches
        block["entries"] = entries

    trackers = tuple(TRACKER_FACTORIES)
    degeneracy = _degeneracy_answer(clean)
    # The published series and the sweep are derived from the SAME recorded
    # approaches, so the headline row of the sweep is the headline itself
    # rather than a second measurement of it.
    ratios = ratio_series(level_approaches, len(labels), region_px)
    sweep = _region_sweep(
        clean_approaches,
        level_approaches,
        len(labels),
        crossing_f1_by_level,
        region_px=region_px,
    )
    return {
        "schema": REPORT_SCHEMA_VERSION,
        "seed": seed,
        "protocol": truth.protocol,
        # The label file's NAME, never its path: this report is tracked and
        # the same label set lives at a different absolute path on every
        # machine.
        "ground_truth": truth.path.name,
        "baseline_reports": ["counting_accuracy.json", crossing_f1_source],
        "clip": truth.clip,
        "fps": truth.fps,
        "frames": len(detections),
        "window": {"start_frame": truth.start_frame, "end_frame": truth.end_frame},
        "gate": {
            "name": gate.name,
            "start": [float(gate.start[0]), float(gate.start[1])],
            "end": [float(gate.end[0]), float(gate.end[1])],
            "label_positive": gate.label_positive,
            "label_negative": gate.label_negative,
        },
        "detector": detector,
        "labels": {
            "total": len(labels),
            "certain": sum(1 for label in labels if label.confidence == "certain"),
            "probable": sum(1 for label in labels if label.confidence != "certain"),
        },
        "trackers_compared": list(trackers),
        "counting_rule": COUNTING_RULE,
        "counting_rule_note": (
            "Held fixed at the gate rule for every figure, so a difference "
            "between two rows is attributable to the TRACKER. The rule "
            "comparison is a different measurement and lives in "
            "counting_accuracy.json."
        ),
        "base_match_window": base_window.as_dict(),
        "gate_region": {
            "half_width_px": float(region_px),
            "definition": gate_region_definition(region_px),
            "source_of_the_half_width": (
                "BASELINE_BAND_PX, the band half-width the counting "
                "benchmark's band rule already uses, so the region is a "
                "value with a stated provenance elsewhere in the project "
                "rather than one chosen here to make a ratio come out."
            ),
        },
        "metric_definitions": metric_definitions(len(labels), region_px),
        "clean": {
            "level_label": "undegraded",
            "match_window": base_window.as_dict(),
            "trackers": clean,
        },
        "reduction": {
            "claim": (
                "Every protocol's identity level -- 30 fps, 0 % dropped, "
                "p = 0, sigma = 0 -- reproduces the clean block EXACTLY, "
                "field by field, for all three trackers, scored with the "
                "undegraded window. The identity runs the general code "
                "path: there is no short-circuit branch, so the reduction "
                "proves the transform rather than proving a branch. The "
                "clean block itself reduces one step further -- its "
                "crossing counts and class consistency are the figures "
                "counting_accuracy.json publishes for the same tracker with "
                "the same rule -- which is what says this report read the "
                "same cached detection stream as the ones beside it."
            ),
            "identity_levels": {
                PROTOCOL_FRAME_RATE: float(truth.fps),
                PROTOCOL_DROPPED_FRAMES: 0.0,
                PROTOCOL_DETECTION_DROPOUT: 0.0,
                PROTOCOL_BOX_JITTER: 0.0,
            },
        },
        "protocols": protocols,
        "gate_region_sweep": sweep,
        "questions": {
            "clean_clip_degeneracy": degeneracy,
            "tracker_separation": _separation_answer(
                ratios, trackers, clean_spread=degeneracy["spread"]
            ),
            "agreement_with_crossing_f1": _agreement_answer(
                ratios,
                crossing_f1_by_level,
                crossing_f1_source,
                sweep=sweep,
                region_px=region_px,
            ),
        },
        "claims_not_made": claims_not_made(),
        "caveats": [
            "Every figure here is scored against the 17 hand-labelled "
            "crossings of ONE gate of ONE clip. The gate was chosen in the "
            "near field for label RELIABILITY, not for the engine's "
            "convenience, so these are UPPER bounds on the same trackers "
            "anywhere harder.",
            "The fragmentation ratio's numerator counts predicted "
            "identities, not identity ERRORS. Above 1.0 is consistent with "
            "a split track and with an unlabelled vehicle in the gate "
            "region alike, and no figure here separates the two.",
            "The two resampling protocols are scored with a match window "
            "widened on the LATE side by the realised sampling gap, derived "
            "from the retained pattern a priori and published beside every "
            "level it scored. The gate region's second clause exists for "
            "the same reason: a vehicle can step clean over the band "
            "between two samples, and charging that to identity would be "
            "charging the sampling grid.",
            "No wall-clock timing is published. Per-method cost is in "
            "counting_accuracy.json, and a timing column would make two "
            "runs of this report differ -- byte reproducibility is the "
            "property it most needs to be able to prove.",
            "Timing aside, this whole report is a function of the cached "
            "detections and the seed. Two runs are byte-identical, and "
            "every level draws from a stream derived from the protocol name "
            "and the level value, so adding a sweep point cannot move "
            "another point's numbers.",
            "The gate region's half-width is a free parameter, and not every "
            "answer here survives it: gate_region_sweep publishes what each "
            "one does across 5-60 px, and the agreement question's verdict is "
            "explicitly conditional on the width. The published width is "
            "BASELINE_BAND_PX, fixed by the counting benchmark, and is not "
            "selected from that table.",
            "certain_only figures are deliberately absent. This report's "
            "denominators are the full label set throughout; the ignore-"
            "region treatment of probable labels, and its two standing "
            "limitations, are stated in full in counting_accuracy.json and "
            "apply to any subset a reader derives from these rows.",
        ],
    }


# --- the command line -------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure tracking quality where an identity error changes an "
            "output: identities spent at the gate, and class consistency of "
            "matched crossings, for three trackers."
        )
    )
    parser.add_argument("--config", default="configs/motorway.yaml")
    parser.add_argument("--gate", default="inbound")
    parser.add_argument("--gt", default="data/groundtruth/motorway_inbound_gt.json")
    parser.add_argument(
        "--cache",
        default="private/bench",
        help="directory holding the shared detection cache; must be git-ignored",
    )
    parser.add_argument("--out", default="reports/tracking.json")
    parser.add_argument(
        "--robustness",
        default="reports/robustness.json",
        help=(
            "the robustness report this sweep takes its levels, its seed "
            "and its crossing-F1 comparison from"
        ),
    )
    parser.add_argument(
        "--region-px",
        type=float,
        default=DEFAULT_GATE_REGION_PX,
        help="half-width of the gate region, in pixels",
    )
    return parser.parse_args(argv)


def resolve(path_like: str) -> str:
    """A repo-relative path made absolute, so the script runs from
    anywhere."""
    path = Path(path_like)
    return str(path if path.is_absolute() else ROOT / path)


def load_detections(config, truth, cache_dir: Path):
    """Read the shared detections the counting benchmark cached. This
    script never constructs a detector: a tracking report whose stream came
    from a different detector run could not reduce to
    ``counting_accuracy.json``, and the reduction is how this family proves
    its protocols correct."""
    key = {
        "clip": truth.clip,
        "model": config.detector.model,
        "confidence": config.detector.confidence,
        "imgsz": config.detector.imgsz,
        "classes": list(config.detector.classes),
        "start_frame": truth.start_frame,
        "end_frame": truth.end_frame,
    }
    cache_path = cache_dir / f"{Path(truth.clip).stem}_detections.json"
    try:
        return read_detection_cache(cache_path, key=key), key
    except DetectionCacheError as error:
        raise SystemExit(
            f"no usable detection cache ({error}).\n"
            f"Run scripts/bench_counting.py first: it detects the labelled "
            f"window once and writes the cache this script reads. Detecting "
            f"here instead would risk scoring the identity level against a "
            f"different detector run than counting_accuracy.json used, and "
            f"the reduction test exists precisely to catch that."
        ) from error


def load_sweep(path: str) -> tuple[dict, dict]:
    """The levels, the seed and the crossing-F1 series this report is
    compared against, read from the committed robustness report.

    Taken from that report rather than re-declared here on purpose: the two
    sweeps must be the same levels at the same seed or the comparison
    between them is between different measurements, and one source of truth
    cannot drift from itself.
    """
    report_path = Path(path)
    if not report_path.is_file():
        raise SystemExit(
            f"no robustness report at {report_path}.\n"
            f"Run scripts/bench_robustness.py first: this report takes its "
            f"sweep levels, its seed and its crossing-F1 comparison from "
            f"it, so that the two are level-for-level comparable."
        )
    report = json.loads(report_path.read_text())
    crossing_f1 = {
        f"{protocol}@{entry['level_label']}": {
            name: entry["methods"][f"{name}+{COUNTING_RULE}"]["f1"]
            for name in TRACKER_FACTORIES
        }
        for protocol, block in report["protocols"].items()
        for entry in block["entries"]
    }
    return report, crossing_f1


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(resolve(args.config))

    gate_configs = {gate.name: gate for gate in config.gates}
    if args.gate not in gate_configs:
        raise SystemExit(
            f"config {args.config} has no gate named {args.gate!r}; it has: "
            f"{', '.join(sorted(gate_configs)) or '(none)'}"
        )

    source_spec = resolve(config.source)
    with VideoSource.open(source_spec) as probe:
        width, height = probe.width, probe.height
    gate = gate_configs[args.gate].to_gate(width, height)

    truth = GroundTruth.load(
        resolve(args.gt), gate=gate, clip_path=Path(source_spec)
    )
    detections, key = load_detections(config, truth, Path(resolve(args.cache)))
    robustness, crossing_f1 = load_sweep(resolve(args.robustness))
    print(
        f"ground truth {truth.path.name}: {len(truth.crossings)} crossings; "
        f"{len(detections)} cached detection frames"
    )

    levels = {name: block["levels"] for name, block in robustness["protocols"].items()}
    report = build_report(
        detections,
        truth,
        gate,
        detector={
            "model": key["model"],
            "confidence": key["confidence"],
            "imgsz": key["imgsz"],
            "classes": key["classes"],
        },
        target_rates=levels[PROTOCOL_FRAME_RATE],
        drop_fractions=levels[PROTOCOL_DROPPED_FRAMES],
        dropout_probabilities=levels[PROTOCOL_DETECTION_DROPOUT],
        jitter_sigmas=levels[PROTOCOL_BOX_JITTER],
        crossing_f1_by_level=crossing_f1,
        crossing_f1_source=Path(args.robustness).name,
        seed=robustness["seed"],
        region_px=args.region_px,
    )

    out_path = write_report(resolve(args.out), report)
    print(f"wrote {out_path.relative_to(ROOT)}")

    print("\nundegraded, gate rule:")
    print(
        f"  {'tracker':12s} {'ids at gate':>11s} {'frag':>7s} "
        f"{'class ok':>9s} {'pred':>6s}"
    )
    for name, record in report["clean"]["trackers"].items():
        consistency = record["class_consistency"]
        print(
            f"  {name:12s} {record['gate_region_track_ids']:11d} "
            f"{record['fragmentation_ratio']:7.4f} "
            f"{consistency['same_class']:4d}/{consistency['matched']:<4d} "
            f"{record['n_predicted']:6d}"
        )

    for protocol, block in report["protocols"].items():
        print(f"\n{protocol} ({block['knob']}), fragmentation ratio:")
        print(
            f"  {'level':16s} "
            + " ".join(f"{name:>12s}" for name in report["trackers_compared"])
        )
        for entry in block["entries"]:
            cells = " ".join(
                f"{entry['trackers'][name]['fragmentation_ratio']:12.4f}"
                for name in report["trackers_compared"]
            )
            print(f"  {entry['level_label']:16s} {cells}")

    for question in report["questions"].values():
        print("\n" + question["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
