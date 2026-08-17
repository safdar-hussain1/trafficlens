"""Tests for the tracking-quality benchmark (``scripts/bench_tracking.py``).

This benchmark scores tracking only where an identity error CHANGES AN
OUTPUT, so the ways it can quietly stop measuring anything are specific:

- **A fragmentation ratio that cannot see a split track.** The obvious
  reading of "distinct predicted track IDs crossing the gate" -- count the
  IDs that emitted a gate crossing -- is structurally incapable of it:
  ``GateCounter`` fires once per ID on a side change, so splitting a track
  at the gate DESTROYS the crossing rather than duplicating it, and the
  ratio moves down or not at all. Both readings are published and the
  fixture below pins the difference, because a metric that anchors at 1.0
  and can never reach 2.0 would pass a test that only checked the 1.0 end.
- **An observer that changes what it observes.** Gate-region IDs are
  collected by wrapping the tracker inside the SAME ``run_counting`` the
  counting benchmark uses. A wrapper that perturbed one association would
  make every figure here a measurement of the wrapper.
- **A protocol that does not reduce to the clean clip.** The identity
  level of each degradation must reproduce the undegraded record exactly,
  field by field, through the general code path.
- **A published rate nobody recomputed from its own counts.** Every ratio
  in the report is recomputed here from the integers beside it.
- **A stated absence that later gets quietly spun into a claim.** No
  identity label set exists for this clip, so no ID-switch, IDF1 or MOTA
  figure may be published. ``claims_not_made`` says so, and the tests
  below assert both halves: that the field carries the refusal, and that
  no such figure appears anywhere in the document.

Tests that read the COMMITTED report rather than a synthetic fixture are
deliberate, exactly as in ``test_bench_robustness``: a schema asserted
only against a dict built in the test proves the test's dict is
well-formed, not that the published numbers are.
"""

import importlib.util
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from trafficlens.bench.baselines import CentroidTracker
from trafficlens.bench.degrade import (
    PROTOCOL_BOX_JITTER,
    PROTOCOL_DETECTION_DROPOUT,
    PROTOCOL_DROPPED_FRAMES,
    PROTOCOL_FRAME_RATE,
    ROBUSTNESS_SEED,
    decimate,
    dropout_streams,
    dropped_frame_streams,
    frame_rate_streams,
    jitter_streams,
)
from trafficlens.bench.harness import run_counting
from trafficlens.bench.scoring import DEFAULT_MATCH_WINDOW
from trafficlens.bench.slitscan import Crossing, GroundTruth
from trafficlens.core.gate import Gate, GateCounter
from trafficlens.detect.base import Detection
from trafficlens.track.tracker import Tracker

ROOT = Path(__file__).resolve().parents[1]
COUNTING_REPORT = ROOT / "reports" / "counting_accuracy.json"
ROBUSTNESS_REPORT = ROOT / "reports" / "robustness.json"
TRACKING_REPORT = ROOT / "reports" / "tracking.json"

GATE_NAME = "inbound"
GATE_Y = 300.0
LAST_FRAME = 200

#: Half the number of frames each synthetic vehicle is visible for, on
#: each side of its own crossing frame. Centring a vehicle's visibility on
#: its crossing is what makes "split every track in half" land AT the
#: gate, which is the only place a split can change an output.
HALF_LIFE_FRAMES = 30

#: Three lanes far enough apart that an 80 px box never overlaps its
#: neighbour, each crossing at the centre of its own visible span.
SYNTHETIC_CROSSINGS = ((40, 60.0), (100, 160.0), (160, 260.0))


# -- fixtures ----------------------------------------------------------------


def _gate(name: str = GATE_NAME) -> Gate:
    """The same left-to-right gate at a constant image y the counting and
    robustness tests use: +1 (up the frame) is ``away``, -1 (down the
    frame, toward the camera) is ``toward``."""
    return Gate(
        name,
        (0.0, GATE_Y),
        (400.0, GATE_Y),
        label_positive="away",
        label_negative="toward",
        expected_direction="toward",
    )


def _det(
    x: float,
    y: float,
    score: float = 0.9,
    class_name: str = "car",
    width: float = 80.0,
    height: float = 80.0,
) -> Detection:
    """A detection whose bottom-centre anchor is exactly ``(x, y)``."""
    class_id = {"car": 2, "truck": 7}[class_name]
    return Detection(
        x1=x - width / 2.0,
        y1=y - height,
        x2=x + width / 2.0,
        y2=y,
        score=score,
        class_id=class_id,
        class_name=class_name,
    )


def _traffic(
    crossings=SYNTHETIC_CROSSINGS,
    last_frame: int = LAST_FRAME,
    *,
    classes=None,
    half_life: int = HALF_LIFE_FRAMES,
    speed_px: float = 1.0,
) -> list[tuple[int, float, list[Detection]]]:
    """A ``(frame_index, timestamp, detections)`` stream carrying one
    descending vehicle per ``(crossing_frame, lane_x)`` pair, each visible
    for exactly ``half_life`` frames either side of its own crossing.

    Half a step of offset keeps the anchor off the gate line itself, so the
    engine's swept-segment test decides the crossing frame unambiguously.
    The symmetric visibility window is what makes the halving fixture below
    split each track AT the gate rather than somewhere harmless.
    """
    classes = classes or {}
    per_frame: dict[int, list[Detection]] = {
        frame: [] for frame in range(last_frame + 1)
    }
    for crossing_frame, lane_x in crossings:
        for frame in range(
            max(0, crossing_frame - half_life),
            min(last_frame, crossing_frame + half_life) + 1,
        ):
            per_frame[frame].append(
                _det(
                    lane_x,
                    GATE_Y + (frame - crossing_frame) * speed_px + speed_px / 2.0,
                    class_name=classes.get(crossing_frame, "car"),
                )
            )
    return [
        (frame, frame / 30.0, per_frame[frame]) for frame in range(last_frame + 1)
    ]


#: The frame the two-traversal fixture below descends through the gate on,
#: and the frame it climbs back through it on.
TWO_TRAVERSAL_DOWN = 41
TWO_TRAVERSAL_UP = 120


def _two_traversals(last_frame: int = 160):
    """One vehicle starting well above the gate, descending through it and
    climbing back through it again: TWO genuine traversals, one identity.

    The case that separates "a split cannot double the crossing count" from
    "a split cannot double the crossing count on a single-traversal
    trajectory". Every labelled crossing on the real clip is the latter; this
    is the former, and it exists so the narrower claim cannot be published as
    the general one.
    """
    return [
        (
            frame,
            frame / 30.0,
            [_det(60.0, GATE_Y - 40.5 + (80 - abs(frame - 80)))],
        )
        for frame in range(0, last_frame + 1)
    ]


def _labels(classes=None) -> list[Crossing]:
    classes = classes or {}
    return [
        Crossing(
            index + 1,
            crossing_frame,
            classes.get(crossing_frame, "car"),
            "toward",
            "certain",
        )
        for index, (crossing_frame, _lane_x) in enumerate(SYNTHETIC_CROSSINGS)
    ]


def _load_script(name: str = "bench_tracking"):
    """Import a file under ``scripts/`` by path: the directory is not a
    package, so the report-assembling code is otherwise untestable."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observed_spans(frames, factory) -> dict[int, tuple[int, int]]:
    """The first and last frame each track ID was RETURNED on.

    Read from a real run rather than assumed, so the halving fixture below
    splits the tracks the tracker actually produced.
    """
    tracker = factory()
    spans: dict[int, tuple[int, int]] = {}
    for index, _timestamp, detections in frames:
        for track in tracker.update(detections, index):
            first, last = spans.get(track.track_id, (index, index))
            spans[track.track_id] = (min(first, index), max(last, index))
    return spans


#: Added to a track ID to make the second half of its life a different
#: identity. Larger than any ID these fixtures allocate, so the two halves
#: can never collide.
_SPLIT_ID_OFFSET = 1000


class _SplitAfterTracker:
    """A tracker whose every track is re-issued under a fresh ID after a
    chosen frame. Used to split a track at a place the caller picks, where
    ``_HalvingTracker`` splits it at its own midpoint."""

    def __init__(self, inner, split_after_frame: int) -> None:
        self._inner = inner
        self._split_after = split_after_frame
        self.max_age = inner.max_age

    def update(self, detections, frame_index):
        tracks = self._inner.update(detections, frame_index)
        if frame_index <= self._split_after:
            return tracks
        return [
            replace(track, track_id=track.track_id + _SPLIT_ID_OFFSET)
            for track in tracks
        ]


class _HalvingTracker:
    """A tracker whose every track is deliberately split in half: the
    second half of each track's observed lifetime is re-issued under a
    fresh ID.

    This is the fragmentation failure mode applied on purpose, and it is
    produced by SPLITTING real tracks -- the spans come from a real run of
    the wrapped tracker -- rather than by writing the expected number down.
    The association itself is untouched: only the identity a downstream
    consumer sees changes, which is exactly the error the report is
    measuring the output cost of.
    """

    def __init__(self, inner, spans: dict[int, tuple[int, int]]) -> None:
        self._inner = inner
        self._spans = spans
        self.max_age = inner.max_age

    def update(self, detections, frame_index):
        out = []
        for track in self._inner.update(detections, frame_index):
            first, last = self._spans[track.track_id]
            if frame_index > (first + last) // 2:
                track = replace(track, track_id=track.track_id + _SPLIT_ID_OFFSET)
            out.append(track)
        return out


def _report() -> dict:
    if not TRACKING_REPORT.is_file():
        pytest.fail(
            f"{TRACKING_REPORT.relative_to(ROOT)} is missing: the tracked "
            f"report is the artefact these tests check, so it must be "
            f"regenerated by scripts/bench_tracking.py whenever the "
            f"benchmark changes"
        )
    return json.loads(TRACKING_REPORT.read_text())


def _keys(node, found: set[str]) -> set[str]:
    """Every dict key anywhere in a JSON document."""
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            _keys(value, found)
    elif isinstance(node, list):
        for value in node:
            _keys(value, found)
    return found


# -- the fragmentation ratio's two anchor points -----------------------------


def test_the_fragmentation_ratio_is_exactly_one_when_every_vehicle_is_tracked_cleanly():
    """The lower anchor: one predicted identity per labelled vehicle.

    Three vehicles, three labels, no association error, so the ratio is
    exactly 1.0 -- not approximately. The count beside it is asserted too,
    because a ratio of 1.0 could also be reached by a numerator and a
    denominator that are both wrong.
    """
    script = _load_script()
    frames = _traffic()
    labels = _labels()
    gate = _gate()

    events, approach = script.run_tracking(frames, Tracker(), gate)
    region_ids = approach.reaching()
    record = script.score_tracking(
        events,
        region_ids,
        labels,
        window=DEFAULT_MATCH_WINDOW,
        gate_name=GATE_NAME,
    )

    assert record["n_ground_truth"] == 3
    assert record["gate_region_track_ids"] == 3
    assert record["fragmentation_ratio"] == 1.0
    # The fold deviation's floor: exactly one identity per vehicle is 1.0x
    # away from one identity per vehicle.
    assert record["identity_deviation"] == 1.0


def test_the_fragmentation_ratio_is_exactly_two_when_every_track_is_split_in_half():
    """The upper anchor: two predicted identities per labelled vehicle.

    The split is PRODUCED, not written down -- ``_HalvingTracker`` re-issues
    the second half of every real track's life under a new ID, and each
    vehicle's visible span is centred on its own crossing so the split
    lands at the gate.

    The control varies a different axis: the CROSSING-EVENT reading of the
    same fragmentation idea, computed over the identical run, does not move
    at all. That is the whole reason this report publishes the gate-region
    reading as its headline -- ``GateCounter`` fires once per ID on a side
    change, so a split at the gate leaves the first half to emit the
    crossing and the second half with no previous anchor to cross from. The
    obvious metric is blind here, and a suite that checked only the 1.0
    anchor would never see that.
    """
    script = _load_script()
    frames = _traffic()
    labels = _labels()
    gate = _gate()

    clean_events, clean_approach = script.run_tracking(frames, Tracker(), gate)
    clean_ids = clean_approach.reaching()
    clean = script.score_tracking(
        clean_events,
        clean_ids,
        labels,
        window=DEFAULT_MATCH_WINDOW,
        gate_name=GATE_NAME,
    )

    spans = _observed_spans(frames, Tracker)
    assert len(spans) == 3, spans
    split_events, split_approach = script.run_tracking(
        frames, _HalvingTracker(Tracker(), spans), gate
    )
    split_ids = split_approach.reaching()
    split = script.score_tracking(
        split_events,
        split_ids,
        labels,
        window=DEFAULT_MATCH_WINDOW,
        gate_name=GATE_NAME,
    )

    assert split["gate_region_track_ids"] == 6
    assert split["fragmentation_ratio"] == 2.0
    assert split["identity_deviation"] == 2.0

    # The control, on a different axis: the crossing-event reading of the
    # same run is unmoved by the split it is supposed to be about.
    assert clean["crossing_id_ratio"] == 1.0
    assert split["crossing_id_ratio"] == clean["crossing_id_ratio"]
    assert split["crossing_track_ids"] == clean["crossing_track_ids"]


def test_the_gate_region_reading_counts_a_track_that_crosses_twice_once():
    """The edge case the definition names, checked rather than asserted in
    prose alone: the numerator counts distinct IDs, so an identity that
    reaches the gate region twice contributes one, not two."""
    script = _load_script()
    gate = _gate()
    frames = _two_traversals()

    events, approach = script.run_tracking(frames, Tracker(), gate)
    region_ids = approach.reaching()
    assert len(region_ids) == 1, region_ids
    # GateCounter fires once per identity, so the second traverse is not a
    # second event either -- the two readings agree on this case.
    assert len(events) == 1
    record = script.score_tracking(
        events,
        region_ids,
        [Crossing(1, 41, "car", "toward", "certain")],
        window=DEFAULT_MATCH_WINDOW,
        gate_name=GATE_NAME,
    )
    assert record["true_positives"] == 1
    assert record["gate_region_track_ids"] == 1
    assert record["crossing_track_ids"] == 1
    assert record["fragmentation_ratio"] == 1.0


def test_splitting_a_two_traversal_track_DOES_double_the_crossing_count():
    """The limit of the claim that justifies substituting the brief's metric,
    pinned rather than described.

    "A split cannot double the crossing-emitting count" is true of a
    SINGLE-traversal trajectory -- only one half of a monotone track can
    change side -- and every labelled crossing on this clip is that case. It
    is NOT universally true. Split the two-traversal track between its
    traversals and both identities latch: the crossing count and the distinct
    crossing-ID count both go 1 -> 2, because ``GateCounter`` remembers per
    track ID and the second identity has never been counted.

    The must-survive control is on a different axis -- the same splitting
    applied to the single-traversal anchor fixture, where the crossing count
    does NOT move (asserted in
    ``test_the_fragmentation_ratio_is_exactly_two_when_every_track_is_split_in_half``).
    Together the two say exactly what the artefact now claims and nothing
    wider.
    """
    script = _load_script()
    gate = _gate()
    frames = _two_traversals()
    labels = [Crossing(1, TWO_TRAVERSAL_DOWN, "car", "toward", "certain")]

    whole, whole_approach = script.run_tracking(frames, Tracker(), gate)
    assert len(whole) == 1, whole
    intact = script.score_tracking(
        whole,
        whole_approach.reaching(),
        labels,
        window=DEFAULT_MATCH_WINDOW,
        gate_name=GATE_NAME,
    )
    assert intact["crossing_track_ids"] == 1
    assert intact["crossing_id_ratio"] == 1.0

    # Split BETWEEN the traversals, so each half carries one of them.
    split_after = (TWO_TRAVERSAL_DOWN + TWO_TRAVERSAL_UP) // 2
    events, approach = script.run_tracking(
        frames, _SplitAfterTracker(Tracker(), split_after), gate
    )
    assert len(events) == 2, events
    assert {event.frame_index for event in events} == {
        TWO_TRAVERSAL_DOWN,
        TWO_TRAVERSAL_UP,
    }
    split = script.score_tracking(
        events,
        approach.reaching(),
        labels,
        window=DEFAULT_MATCH_WINDOW,
        gate_name=GATE_NAME,
    )
    assert split["crossing_track_ids"] == 2
    assert split["crossing_id_ratio"] == 2.0
    # ... and the artefact must not claim the impossibility unqualified.
    definitions = _load_script().metric_definitions(17, 20.0)["crossing_id_ratio"]
    assert "can never double" not in json.dumps(definitions)
    assert "single-traversal" in definitions["why_it_cannot_measure_fragmentation"]
    assert definitions["not_a_universal_impossibility"].strip()


def test_a_gate_region_identity_matched_to_no_label_still_counts():
    """The second edge case the definition names.

    The numerator is a count of predicted identities at the gate and the
    denominator a count of labelled vehicles; neither is conditioned on the
    match. A vehicle the labeller never recorded therefore RAISES the ratio,
    which is the honest behaviour -- with no identity labels this metric
    cannot tell a fragmented track from an extra vehicle, and the report
    says so rather than hiding the case by restricting to matched pairs.
    """
    script = _load_script()
    frames = _traffic()
    gate = _gate()
    events, approach = script.run_tracking(frames, Tracker(), gate)
    region_ids = approach.reaching()

    # Two of the three crossings labelled; the third vehicle is at the gate
    # and matched to nothing.
    labels = _labels()[:2]
    record = script.score_tracking(
        events, region_ids, labels, window=DEFAULT_MATCH_WINDOW, gate_name=GATE_NAME
    )
    assert record["n_ground_truth"] == 2
    assert record["gate_region_track_ids"] == 3
    assert record["fragmentation_ratio"] == 1.5
    assert record["identity_deviation"] == 1.5
    assert record["true_positives"] == 2

    # ... and the other side of 1.0. FEWER identities at the gate than
    # labelled vehicles is an error too -- it is what association collapse
    # looks like -- so the deviation is a DISTANCE from one, not a signed
    # gap. Without a below-1.0 case nothing would tell the two apart, and
    # the sweep's engine rows are mostly below 1.0 at the lower frame rates.
    padded = _labels() + [
        Crossing(index, frame, "car", "toward", "certain")
        for index, frame in ((4, 20), (5, 60), (6, 180))
    ]
    sparse = script.score_tracking(
        events, region_ids, padded, window=DEFAULT_MATCH_WINDOW, gate_name=GATE_NAME
    )
    assert sparse["n_ground_truth"] == 6
    assert sparse["gate_region_track_ids"] == 3
    assert sparse["fragmentation_ratio"] == 0.5
    # 0.5 and 1.5 are NOT the same size of error under abs(r - 1) -- they
    # both read 0.5 -- and they are not the same error either: this one is
    # a 2x shortfall. A fold says so, and that asymmetry is why the measure
    # is multiplicative.
    assert sparse["identity_deviation"] == 2.0
    assert sparse["true_positives"] == 3


# -- class consistency -------------------------------------------------------


def test_class_consistency_is_one_when_classes_agree_and_drops_when_they_do_not():
    """A discriminating pair varying exactly one axis: the LABEL's class.

    The stream, the tracker, the gate, the window and the matching are
    identical in both halves -- matching is class-blind, so the same three
    pairs are made either way -- and only the class recorded against the
    middle label changes. A metric that had stopped reading the classes
    would pass the first half alone.
    """
    script = _load_script()
    gate = _gate()
    frames = _traffic()
    events, approach = script.run_tracking(frames, Tracker(), gate)
    region_ids = approach.reaching()

    agreeing = script.score_tracking(
        events, region_ids, _labels(), window=DEFAULT_MATCH_WINDOW, gate_name=GATE_NAME
    )["class_consistency"]
    assert agreeing["matched"] == 3
    assert agreeing["same_class"] == 3
    assert agreeing["rate"] == 1.0
    assert agreeing["confusions"] == {}

    disagreeing = script.score_tracking(
        events,
        region_ids,
        _labels(classes={100: "truck"}),
        window=DEFAULT_MATCH_WINDOW,
        gate_name=GATE_NAME,
    )["class_consistency"]
    # Class-blind matching: the SAME three pairs, one of them a confusion.
    assert disagreeing["matched"] == 3
    assert disagreeing["same_class"] == 2
    assert disagreeing["rate"] == pytest.approx(2 / 3)
    assert disagreeing["confusions"] == {"truck -> car": 1}


# -- the observer must not change what it observes ---------------------------


def test_collecting_gate_region_identities_changes_no_association():
    """Gate-region IDs are collected by wrapping the tracker inside the
    same ``run_counting`` every other benchmark in this repo drives, so the
    lifecycle, the reaping and the anchor bookkeeping cannot drift from the
    shipped pipeline. The wrapper must be transparent: if it perturbed one
    association, every figure in this report would be a measurement of the
    wrapper.

    The must-survive control varies a different axis -- the same comparison
    on a DEGRADED stream, where associations are genuinely being lost --
    so a wrapper that were transparent only on easy input would still fail.
    """
    script = _load_script()
    gate = _gate()
    frames = _traffic()

    plain = run_counting(frames, Tracker(), GateCounter(gate))
    wrapped, approach = script.run_tracking(frames, Tracker(), gate)
    region_ids = approach.reaching()
    assert wrapped == plain
    assert plain, "the fixture must produce crossings for this to compare"
    assert region_ids

    degraded = decimate(frames, source_fps=30.0, target_fps=5.0)
    plain_slow = run_counting(list(degraded.frames), Tracker(), GateCounter(gate))
    wrapped_slow, _slow = script.run_tracking(list(degraded.frames), Tracker(), gate)
    assert wrapped_slow == plain_slow


def test_a_track_that_steps_clean_over_the_band_still_reaches_the_gate_region():
    """The gate region is a property of the TRAJECTORY, not of the sampling
    grid.

    A decimated stream can move a vehicle further than the band's width in
    one sample, so a containment test alone would report that the vehicle
    never reached the gate at all, and the ratio would collapse for reasons
    that have nothing to do with identity. The swept path is therefore the
    second half of the definition, and this is the case that needs it: a
    vehicle sampled every sixth frame whose every sample lands outside the
    band while its path crosses the gate.
    """
    script = _load_script()
    gate = _gate()
    # 9 px per frame at a stride of 6 is 54 px per sample, so the two
    # samples either side of the gate sit 31.5 px above and 22.5 px below
    # it -- both outside the 20 px band.
    frames = _traffic(((100, 60.0),), last_frame=200, speed_px=9.0, half_life=40)
    degraded = decimate(frames, source_fps=30.0, target_fps=5.0)
    assert degraded.max_gap == 6

    anchors = [
        detection.y2
        for _index, _timestamp, detections in degraded.frames
        for detection in detections
    ]
    assert anchors, "the decimated fixture must still carry detections"
    assert all(abs(y - GATE_Y) > script.DEFAULT_GATE_REGION_PX for y in anchors), (
        "the fixture must never place a sample inside the band, or the "
        "swept-path clause is not what this test exercises"
    )

    # The centroid tracker, whose 60 px association radius still follows a
    # 54 px step: the engine's 0.8 IoU floor loses this stream entirely, and
    # a tracker that produced no track at all could not exercise anything.
    events, approach = script.run_tracking(
        list(degraded.frames), CentroidTracker(), gate
    )
    region_ids = approach.reaching()
    assert len(events) == 1, events
    assert len(region_ids) == 1, region_ids


def test_an_identity_that_never_reaches_the_gate_is_not_counted():
    """The restriction the fragmentation ratio's numerator depends on, and
    the one a clean-clip anchor cannot see.

    Both anchor fixtures have every track spend time inside the band AND
    outside it, so they read 1.0 and 2.0 whichever way round the band test
    points -- a mutation that inverted it survived both. On the real clip
    that inversion would sweep in the far carriageway and the ratio would
    stop being about the labelled gate at all, so the two ways a track can
    fail to reach the gate are exercised here directly:

    - too far from the line, though square in front of it;
    - close to the line, but past the end of the gate SEGMENT -- the
      other-carriageway case both the band geometry and the engine's own
      bounded-segment test exist to exclude.
    """
    script = _load_script()
    gate = _gate()
    frames = []
    for frame in range(0, 121):
        y = GATE_Y + (frame - 60) + 0.5
        frames.append(
            (
                frame,
                frame / 30.0,
                [
                    # Reaches the gate: descends through the segment.
                    _det(60.0, y),
                    # Never reaches it: 200 px in front of the line, parked.
                    _det(160.0, GATE_Y - 200.0),
                    # Crosses the gate's infinite LINE, 100 px past the
                    # segment's own end at x = 400.
                    _det(500.0, y),
                ],
            )
        )

    events, approach = script.run_tracking(frames, Tracker(), gate)
    region_ids = approach.reaching()
    assert len(events) == 1, events
    assert len(region_ids) == 1, region_ids
    record = script.score_tracking(
        events,
        region_ids,
        [Crossing(1, 60, "car", "toward", "certain")],
        window=DEFAULT_MATCH_WINDOW,
        gate_name=GATE_NAME,
    )
    assert record["true_positives"] == 1
    assert record["gate_region_track_ids"] == 1
    assert record["fragmentation_ratio"] == 1.0


# -- the reduction: every protocol's identity level --------------------------


def test_every_protocol_at_its_identity_level_reproduces_the_clean_clip_record():
    """The protocol's own correctness proof, for these metrics.

    30 fps, 0 % dropped frames, p = 0 and sigma = 0 must each reproduce the
    undegraded record EXACTLY, field by field, for all three trackers, and
    through the general code path -- there is no short-circuit branch, so
    this proves the transform rather than proving a branch.

    The fixture must be able to SEE a difference: all three trackers score
    three true positives undegraded, so any transform that moved a
    prediction off its label would drop that and the equalities would have
    something to disagree about.
    """
    script = _load_script()
    gate = _gate()
    frames = _traffic()
    labels = _labels()

    clean = script.score_clean(frames, gate, labels, gate_name=GATE_NAME)
    for name, record in clean.items():
        assert record["true_positives"] == 3, name
        assert record["fragmentation_ratio"] == 1.0, name

    identity = {
        PROTOCOL_FRAME_RATE: frame_rate_streams(
            frames, source_fps=30.0, target_rates=(30.0,)
        ),
        PROTOCOL_DROPPED_FRAMES: dropped_frame_streams(frames, fractions=(0.0,)),
        PROTOCOL_DETECTION_DROPOUT: dropout_streams(frames, probabilities=(0.0,)),
        PROTOCOL_BOX_JITTER: jitter_streams(frames, sigmas_px=(0.0,)),
    }
    fields = (
        "gate_region_track_ids",
        "crossing_track_ids",
        "n_predicted",
        "true_positives",
        "n_ground_truth",
        "fragmentation_ratio",
        "identity_deviation",
        "crossing_id_ratio",
        "class_consistency",
    )
    for protocol, streams in identity.items():
        assert len(streams) == 1
        entry = script.score_stream(streams[0], gate, labels, gate_name=GATE_NAME)
        assert entry["match_window"]["frames_before"] == 1, protocol
        assert entry["match_window"]["frames_after"] == 4, protocol
        assert set(entry["trackers"]) == set(clean)
        for name, record in entry["trackers"].items():
            for field in fields:
                assert record[field] == clean[name][field], (protocol, name, field)


# -- the published report ----------------------------------------------------


def test_the_published_report_states_its_metric_definitions_before_its_numbers():
    """A number whose definition is decided after the output is a number
    fitted to the output, so each metric's definition, its denominator and
    its edge cases travel with the report itself."""
    report = _report()
    # "Before its numbers" is an ordering claim about the published file, so
    # it is asserted on the file's own key order. The byte-reproducibility
    # test sees order STABILITY; this sees order CONTENT, and neither
    # substitutes for the other.
    keys = list(report)
    for name in ("metric_definitions", "gate_region", "base_match_window"):
        assert keys.index(name) < keys.index("protocols"), name
    definitions = report["metric_definitions"]
    assert set(definitions) >= {
        "fragmentation_ratio",
        "crossing_id_ratio",
        "identity_deviation",
        "class_consistency",
        "gate_region",
    }
    for name, block in definitions.items():
        assert block["definition"].strip(), name
    fragmentation = definitions["fragmentation_ratio"]
    assert fragmentation["denominator"] == report["labels"]["total"]
    edges = " ".join(fragmentation["edge_cases"]).lower()
    assert "twice" in edges
    assert "matched to no label" in edges
    assert fragmentation["edge_cases"], "the edge cases must be stated"
    assert report["gate_region"]["half_width_px"] == 20.0

    # The gate region is DEFINED once. Published verbatim in two places, it
    # is two definitions the moment one of them is edited.
    phrase = "REACHES THE GATE when either of two things"
    assert phrase in report["gate_region"]["definition"]
    assert json.dumps(report).count(phrase) == 1, (
        "the gate-region definition appears more than once in the document"
    )
    assert definitions["gate_region"]["stated_in"] == "gate_region.definition"

    # The fold deviation's a-priori justification must be published with it,
    # since the change of measure is what moved a published answer.
    fold = definitions["identity_deviation"]
    assert "d(r) = d(1/r)" in fold["why_a_fold_and_not_abs"]
    assert "abs(log" in fold["equivalent_to"]


def test_the_published_report_sweeps_the_gate_regions_half_width():
    """The half-width is the metric's one free parameter and one published
    answer depends on it, so the sweep is part of the artefact.

    Three things are asserted, and the third is what stops the sweep being
    decoration: the published width's row must BE the report's own headline
    (not a second measurement of it); the ratios must be monotone in the
    half-width, which is structural (a wider region can only add
    identities); and the knob must actually change something across the
    range, or a sweep of one answer repeated seven times would pass.
    """
    report = _report()
    sweep = report["gate_region_sweep"]
    rows = sweep["by_half_width"]
    widths = [row["half_width_px"] for row in rows]

    assert widths == sorted(widths), widths
    assert sweep["published_half_width_px"] == report["gate_region"]["half_width_px"]
    published = [row for row in rows if row["published"]]
    assert len(published) == 1, widths
    assert published[0]["half_width_px"] == sweep["published_half_width_px"]

    # The published row IS the headline.
    separation = report["questions"]["tracker_separation"]
    agreement = report["questions"]["agreement_with_crossing_f1"]
    degeneracy = report["questions"]["clean_clip_degeneracy"]
    row = published[0]
    assert row["clean"]["fragmentation_ratio_by_tracker"] == (
        degeneracy["fragmentation_ratio_by_tracker"]
    )
    assert row["clean"]["spread"] == pytest.approx(degeneracy["spread"])
    assert row["max_spread"] == pytest.approx(separation["max_spread"])
    assert row["levels_where_trackers_differ"] == len(
        separation["levels_where_trackers_differ"]
    )
    assert row["levels_where_the_engine_is_furthest_and_they_differ"] == len(
        separation["levels_where_the_engine_is_furthest_and_they_differ"]
    )
    assert row["agreement"]["levels_where_the_two_disagree"] == (
        agreement["levels_where_the_two_disagree"]
    )
    assert row["agreement"]["agrees_everywhere"] == agreement["agrees_everywhere"]
    assert row["agreement"]["levels_where_the_two_agree"] == len(
        agreement["levels_where_the_two_agree"]
    )

    # Monotone in the half-width: a wider region can only add identities, so
    # no tracker's clean ratio may ever fall as the width grows. A sweep that
    # had stopped varying the knob would still pass this; the next assertion
    # is what stops that.
    for earlier, later in zip(rows, rows[1:]):
        for name, value in earlier["clean"]["fragmentation_ratio_by_tracker"].items():
            assert later["clean"]["fragmentation_ratio_by_tracker"][name] >= value, (
                earlier["half_width_px"],
                later["half_width_px"],
                name,
            )

    # The knob is live: the widest and narrowest widths must not agree.
    assert (
        rows[0]["clean"]["fragmentation_ratio_by_tracker"]
        != rows[-1]["clean"]["fragmentation_ratio_by_tracker"]
    ), "the sweep produces the same ratios at every width, so it varies nothing"

    # And the verdict must say the answer is conditional on it, naming the
    # widths on each side rather than gesturing at a range.
    verdict = agreement["verdict"]
    assert "CONDITIONAL on the gate region's half-width" in verdict
    complete = [
        row["half_width_px"] for row in rows if row["agreement"]["agrees_everywhere"]
    ]
    incomplete = [
        row["half_width_px"]
        for row in rows
        if not row["agreement"]["agrees_everywhere"]
    ]
    assert complete and incomplete, (
        "every swept width gives the same agreement branch, so the verdict's "
        "conditional framing describes a dependence the sweep does not show"
    )
    assert f"COMPLETE at {', '.join(f'{v:g}' for v in complete)} px" in verdict
    assert f"incomplete at {', '.join(f'{v:g}' for v in incomplete)} px" in verdict
    # The published width is not the one that flatters the answer.
    assert sweep["published_half_width_px"] in incomplete
    assert "BASELINE_BAND_PX" in verdict


def test_the_swept_summary_fields_are_recomputed_from_the_reports_own_series():
    """The four swept fields the test above leaves free, recomputed.

    ``test_the_published_report_sweeps_the_gate_regions_half_width`` pins the
    published row's ratios, spreads, counts and agreement branch against the
    report's own headline, and every ratio is recomputed from its own integers
    elsewhere in this module. Four fields were in neither, and the mutation
    battery found all four hand-falsifiable with the whole suite green:
    ``widest_spread_level``, ``engine_fragmentation_ratio_min``,
    ``engine_fragmentation_ratio_max`` and ``invariants_sentence``.

    None of them is decoration. ``widest_spread_level`` names WHERE degradation
    separates the trackers most; the two extremes bracket how far the engine gets
    from one identity per labelled vehicle; and ``invariants_sentence`` is the
    report's own statement of which of its claims do NOT depend on the metric's
    one free parameter, which is the part a reader is entitled to rely on.

    Each is recomputed from its own definition over series the report publishes:
    the per-level fragmentation ratios for the published half-width, and the
    swept rows themselves for the sentence. The sentence's WORDING is restated
    here so a change has to be made deliberately on both sides; every NUMBER in
    it is computed from the rows rather than copied.
    """
    report = _report()
    sweep = report["gate_region_sweep"]
    rows = sweep["by_half_width"]
    ratios = report["questions"]["tracker_separation"][
        "fragmentation_ratio_by_level"
    ]
    assert ratios, "the per-level series is empty, so nothing below is checked"
    assert len(rows) >= 3, rows

    # -- the published row, cell by cell against its own per-level series ------
    spreads = {
        level: max(by_tracker.values()) - min(by_tracker.values())
        for level, by_tracker in ratios.items()
    }
    engine = [by_tracker["engine"] for by_tracker in ratios.values()]

    published = [row for row in rows if row["published"]]
    assert len(published) == 1, [row["half_width_px"] for row in published]
    row = published[0]

    named = row["widest_spread_level"]
    assert named in spreads, (
        f"widest_spread_level names {named!r}, which is not a level this report "
        f"measured"
    )
    # Stated as "no level spreads wider" rather than "argmax equals", so a tie
    # between two levels is not a failure -- but a level that is not widest is.
    assert spreads[named] == pytest.approx(max(spreads.values())), (
        f"widest_spread_level names {named!r} at spread {spreads[named]:.4f}, "
        f"but {max(spreads, key=lambda k: spreads[k])!r} spreads "
        f"{max(spreads.values()):.4f}"
    )
    assert row["engine_fragmentation_ratio_min"] == pytest.approx(min(engine))
    assert row["engine_fragmentation_ratio_max"] == pytest.approx(max(engine))

    # -- every row: what holds without a per-level series to compare against ---
    for earlier, later in zip(rows, rows[1:]):
        for field in (
            "engine_fragmentation_ratio_min", "engine_fragmentation_ratio_max"
        ):
            # A wider region can only ADD identities, so neither extreme may
            # fall as the half-width grows.
            assert later[field] >= earlier[field] - 1e-12, (
                field, earlier["half_width_px"], later["half_width_px"]
            )
    for each in rows:
        assert each["engine_fragmentation_ratio_min"] <= (
            each["engine_fragmentation_ratio_max"]
        ), each["half_width_px"]
        assert each["widest_spread_level"] in spreads, each["half_width_px"]

    # -- the invariants sentence, number for number ----------------------------
    shares = [
        (
            each["levels_where_the_engine_is_furthest_and_they_differ"],
            each["levels_where_trackers_differ"],
        )
        for each in rows
    ]
    # The two claims the sentence makes in words rather than figures, checked
    # against the rows rather than taken on trust.
    for share, total in shares:
        assert total > 0, "the sentence claims the trackers separate at EVERY width"
        assert share > 0, (
            "the sentence claims the engine is furthest from one identity per "
            "labelled vehicle at every width"
        )
    clean_spreads = {each["clean"]["spread"] for each in rows}
    expected_sentence = (
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
    assert sweep["invariants_sentence"] == expected_sentence, (
        "the published invariants sentence is not what the swept rows say:\n"
        f"published: {sweep['invariants_sentence']}\n"
        f"rows say : {expected_sentence}"
    )


def test_the_published_report_recomputes_every_ratio_from_its_own_counts():
    """The layer under the headline: no published ratio may be impossible.

    Fabricating a fragmentation ratio while leaving the ID counts correct
    is the cheapest way to invert this report's finding, so every ratio at
    every level for every tracker is recomputed here from the integers
    published beside it.
    """
    report = _report()
    total = report["labels"]["total"]
    checked = 0

    def check(record, where):
        assert record["n_ground_truth"] == total, where
        assert record["fragmentation_ratio"] == pytest.approx(
            record["gate_region_track_ids"] / total
        ), where
        assert record["crossing_id_ratio"] == pytest.approx(
            record["crossing_track_ids"] / total
        ), where
        ratio = record["fragmentation_ratio"]
        if ratio == 0.0:
            # An infinite fold: not one identity reached the gate. Published
            # as null so nobody averages it, and it ranks worst of all.
            assert record["identity_deviation"] is None, where
        else:
            assert record["identity_deviation"] == pytest.approx(
                max(ratio, 1.0 / ratio)
            ), where
            assert record["identity_deviation"] >= 1.0, where
        # EQUALITY, not an inequality. Each identity emits at most one
        # crossing and IDs are never recycled, so this is structural -- and
        # pinning it as equality is what would catch a lost crossing-ID
        # count, which an inequality would let through.
        assert record["crossing_track_ids"] == record["n_predicted"], where
        assert record["true_positives"] <= min(total, record["n_predicted"]), where
        consistency = record["class_consistency"]
        assert consistency["matched"] == record["true_positives"], where
        assert consistency["same_class"] <= consistency["matched"], where
        expected = (
            consistency["same_class"] / consistency["matched"]
            if consistency["matched"]
            else 0.0
        )
        assert consistency["rate"] == pytest.approx(expected), where
        assert (
            sum(consistency["confusions"].values())
            == consistency["matched"] - consistency["same_class"]
        ), where

    for name, record in report["clean"]["trackers"].items():
        check(record, f"clean/{name}")
        checked += 1
    for protocol, block in report["protocols"].items():
        for entry in block["entries"]:
            for name, record in entry["trackers"].items():
                check(record, f"{protocol}@{entry['level_label']}/{name}")
                checked += 1

    # 1 clean + 21 levels, three trackers each.
    assert checked == (1 + 21) * 3, checked


def test_the_published_report_reduces_exactly_to_its_own_clean_clip_block():
    """Every protocol's identity level must reproduce the undegraded
    record, on the real clip, to the last digit."""
    report = _report()
    clean = report["clean"]["trackers"]
    identity = report["reduction"]["identity_levels"]
    assert set(identity) == set(report["protocols"])
    for protocol, level in identity.items():
        entry = next(
            e for e in report["protocols"][protocol]["entries"] if e["level"] == level
        )
        assert entry["match_window"]["frames_before"] == 1, protocol
        assert entry["match_window"]["frames_after"] == 4, protocol
        assert set(entry["trackers"]) == set(clean)
        for name, record in entry["trackers"].items():
            assert record == clean[name], (protocol, name)


def test_the_published_report_shares_the_counting_benchmarks_detection_stream():
    """Ruling: the same cached detections, or this report could not be
    compared with the ones beside it.

    The clean block's gate-rule crossings must be exactly the ones
    ``counting_accuracy.json`` published for the same tracker. A tracking
    report whose stream came from a different detector run would score
    identities against traffic the other reports never saw.
    """
    report = _report()
    baseline = json.loads(COUNTING_REPORT.read_text())["methods"]
    for name, record in report["clean"]["trackers"].items():
        full = baseline[f"{name}+gate"]["full"]
        assert record["n_predicted"] == full["n_predicted"], name
        assert record["true_positives"] == full["true_positives"], name
        assert record["class_consistency"] == full["class_consistency"], name


def test_the_published_report_sweeps_the_same_levels_the_robustness_report_did():
    """The two reports must be comparable level by level, or the agreement
    question below is comparing different measurements."""
    report = _report()
    robustness = json.loads(ROBUSTNESS_REPORT.read_text())
    assert report["seed"] == robustness["seed"] == ROBUSTNESS_SEED
    assert set(report["protocols"]) == set(robustness["protocols"])
    for protocol, block in report["protocols"].items():
        mine = [entry["level_label"] for entry in block["entries"]]
        theirs = [
            entry["level_label"]
            for entry in robustness["protocols"][protocol]["entries"]
        ]
        assert mine == theirs, protocol


def _fold(ratio: float) -> float:
    """The report's own multiplicative deviation, restated independently
    here so the check is not the implementation checking itself.

    ``inf`` where the ratio is zero -- no identity reached the gate -- which
    is how a null deviation ranks worst without a special case at every
    comparison. Same reason ``test_bench_robustness`` restates the scorer's
    no-denominator convention rather than importing it.
    """
    return math.inf if ratio <= 0.0 else max(ratio, 1.0 / ratio)


# -- the questions, recomputed from the series under them --------------------


def test_the_published_report_answers_the_degeneracy_question_from_its_series():
    """The expected result on this clip is that the metric cannot tell the
    three trackers apart undegraded. Whether it can is recomputed here from
    the clean block rather than trusted, in both directions: if a later
    change makes the metric discriminate, the verdict has to stop saying it
    does not."""
    report = _report()
    question = report["questions"]["clean_clip_degeneracy"]
    ratios = {
        name: record["fragmentation_ratio"]
        for name, record in report["clean"]["trackers"].items()
    }
    assert question["fragmentation_ratio_by_tracker"] == ratios
    spread = max(ratios.values()) - min(ratios.values())
    assert question["spread"] == pytest.approx(spread)
    assert question["discriminates"] == (spread > 0.0)
    verdict = question["verdict"]
    assert verdict.strip()
    if spread > 0.0:
        assert "cannot tell the three trackers apart" not in verdict
        # The figure the prose quotes must be the recomputed one. A verdict
        # is what a reader takes away, so a number appearing only there is
        # the cheapest place to launder this result.
        assert f"by {spread:.4f}" in verdict
    else:
        assert "cannot tell the three trackers apart" in verdict
        assert f"{next(iter(ratios.values())):.4f} predicted identities" in verdict


def test_the_published_report_answers_the_separation_question_from_its_series():
    """Whether degradation separates the trackers on THIS metric, and in
    which direction, recomputed from the per-level records -- never from
    the question block's own copy of them."""
    report = _report()
    question = report["questions"]["tracker_separation"]
    trackers = tuple(report["trackers_compared"])

    ratios: dict[str, dict[str, float]] = {}
    for protocol, block in report["protocols"].items():
        for entry in block["entries"]:
            key = f"{protocol}@{entry['level_label']}"
            ratios[key] = {
                name: entry["trackers"][name]["fragmentation_ratio"]
                for name in trackers
            }

    assert set(question["fragmentation_ratio_by_level"]) == set(ratios)
    for key, scores in ratios.items():
        assert question["fragmentation_ratio_by_level"][key] == scores, key

    spreads = {
        key: max(scores.values()) - min(scores.values())
        for key, scores in ratios.items()
    }
    assert question["spread_by_level"] == pytest.approx(spreads)
    assert question["levels_measured"] == len(ratios)
    assert question["max_spread"] == pytest.approx(max(spreads.values()))
    assert question["trackers_separate"] == (max(spreads.values()) > 0.0)
    differing = sorted(key for key, spread in spreads.items() if spread > 0.0)
    assert question["levels_where_trackers_differ"] == differing

    # Separating is not the same as separating against the engine, and the
    # DIRECTION is the half a reader acts on -- so it is recomputed from the
    # records, not read out of the question block's own summary of them.
    # A FOLD distance from 1.0: a signed comparison would rank a tracker that
    # lost every track at the gate as the best identity behaviour here, and
    # abs(r - 1) would still flatter it, being capped at 1.0 below while
    # unbounded above.
    furthest = sorted(
        key
        for key, scores in ratios.items()
        if _fold(scores["engine"]) == max(_fold(value) for value in scores.values())
    )
    assert question["levels_where_the_engine_is_furthest_from_one"] == furthest
    where_it_matters = sorted(set(furthest) & set(differing))
    assert (
        question["levels_where_the_engine_is_furthest_and_they_differ"]
        == where_it_matters
    )
    # The tie rule has to be published, because the list above counts ties as
    # "furthest" and two of its entries are three-way ties at spread 0.0.
    tie_rule = question["furthest_from_one_tie_rule"]
    assert "TIE" in tie_rule
    assert question["furthest_from_one_measured_by"].strip()
    assert set(furthest) - set(differing), (
        "no level is 'furthest' without the trackers differing, so the tie "
        "rule the report documents describes nothing on this artefact"
    )

    # ... and the prose must quote the recomputed counts, not its own.
    verdict = question["verdict"]
    assert verdict.strip()
    if differing:
        assert f"differ at {len(differing)} of the {len(spreads)} levels" in verdict
        assert (
            f"at {len(where_it_matters)} of those {len(differing)} levels" in verdict
        )
    else:
        assert f"at any of the {len(spreads)} levels measured" in verdict


def test_the_published_report_compares_its_ranking_against_the_crossing_f1_one():
    """The session's standing result is that the engine's tracker scores
    LOWEST on crossing F1 at every degradation level where the three
    differ, at about 21x a baseline's cost. Whether this benchmark's
    identity metric agrees is the question worth asking, and a disagreement
    is a finding rather than something to reconcile -- so both the
    agreements and the disagreements are published, and both are recomputed
    here from the two reports.
    """
    report = _report()
    robustness = json.loads(ROBUSTNESS_REPORT.read_text())
    question = report["questions"]["agreement_with_crossing_f1"]
    assert question["source"] == "robustness.json"
    trackers = tuple(report["trackers_compared"])

    f1_by_level = {}
    for protocol, block in robustness["protocols"].items():
        for entry in block["entries"]:
            f1_by_level[f"{protocol}@{entry['level_label']}"] = {
                name: entry["methods"][f"{name}+gate"]["f1"] for name in trackers
            }
    # Recomputed from the RATIOS, not read out of the published deviation
    # field: the comparison must be the one the definition implies, so a
    # fabricated deviation cannot agree with itself here.
    ratio_by_level = {}
    deviation = {}
    for protocol, block in report["protocols"].items():
        for entry in block["entries"]:
            key = f"{protocol}@{entry['level_label']}"
            ratio_by_level[key] = {
                name: entry["trackers"][name]["fragmentation_ratio"]
                for name in trackers
            }
            deviation[key] = {
                name: _fold(value) for name, value in ratio_by_level[key].items()
            }
            for name in trackers:
                published = entry["trackers"][name]["identity_deviation"]
                recomputed = deviation[key][name]
                assert (published is None) == math.isinf(recomputed), (key, name)
                if published is not None:
                    assert published == pytest.approx(recomputed), (key, name)

    separating = sorted(
        key
        for key, scores in f1_by_level.items()
        if max(scores.values()) - min(scores.values()) > 0.0
    )
    assert question["levels_where_crossing_f1_separates"] == separating
    assert separating, (
        "crossing F1 separates the trackers at no level at all, so the "
        "agreement question is vacuous -- that would be a change to the "
        "published answer, not to this test"
    )

    agree = []
    disagree = []
    detail = {}
    for key in separating:
        worst_f1 = sorted(
            name
            for name, value in f1_by_level[key].items()
            if value == min(f1_by_level[key].values())
        )
        worst_identity = sorted(
            name
            for name, value in deviation[key].items()
            if value == max(deviation[key].values())
        )
        agrees = bool(set(worst_f1) & set(worst_identity))
        (agree if agrees else disagree).append(key)
        detail[key] = {
            "lowest_crossing_f1": worst_f1,
            "largest_identity_deviation": worst_identity,
            "agrees": agrees,
        }

    assert question["levels_where_the_two_agree"] == agree
    assert question["levels_where_the_two_disagree"] == disagree
    assert question["agrees_everywhere"] == (not disagree)
    assert question["detail"] == detail

    # The superseded measure's counts, recomputed the same way from the same
    # ratios. It is published so the correction is a measurement rather than
    # a recalled number, which means it has to be checked like one.
    superseded = question["superseded_measure"]
    was_agree, was_disagree = [], []
    for key in separating:
        absolute = {
            name: abs(value - 1.0) for name, value in ratio_by_level[key].items()
        }
        worst_f1 = sorted(
            name
            for name, value in f1_by_level[key].items()
            if value == min(f1_by_level[key].values())
        )
        worst_absolute = sorted(
            name
            for name, value in absolute.items()
            if value == max(absolute.values())
        )
        (was_agree if set(worst_f1) & set(worst_absolute) else was_disagree).append(key)
    assert superseded["agreements_under_the_superseded_measure"] == len(was_agree)
    assert superseded["disagreements_under_the_superseded_measure"] == len(
        was_disagree
    )
    assert superseded["levels_it_called_disagreements"] == was_disagree
    assert superseded["agreements_now"] == len(agree)
    assert superseded["disagreements_now"] == len(disagree)
    # The correction must have MOVED something, or publishing it says nothing.
    assert len(was_disagree) != len(disagree), (
        "the fold deviation and abs(r - 1) report the same agreement count, "
        "so superseded_measure documents a change that did not happen"
    )
    verdict = question["verdict"]
    assert verdict.strip()
    if disagree:
        assert "disagree" in verdict.lower()
        # The counts the prose quotes, recomputed: a verdict that shrank the
        # disagreement while the lists above stayed honest is exactly the
        # reconciling-away this field exists to prevent.
        assert f"agree at {len(agree)} of the" in verdict
        assert f"{len(separating)} levels where crossing F1 separates" in verdict
        assert f"DISAGREE at {len(disagree)}" in verdict
    else:
        assert f"all {len(agree)}" in verdict


# -- what this report does not claim -----------------------------------------


def test_the_published_report_says_what_it_does_not_claim():
    """The field exists so an absence cannot later be quietly spun into a
    claim. Blanking it must fail here."""
    report = _report()
    claims = report["claims_not_made"]
    assert claims, "claims_not_made must be present and non-empty"
    for entry in claims:
        assert entry["claim"].strip(), entry
        assert entry["reason"].strip(), entry
    text = json.dumps(claims).lower()
    for name in ("id-switch", "idf1", "mota", "identity label"):
        assert name in text, name


def test_the_published_report_publishes_no_identity_metric_anywhere():
    """The other half of the refusal: no ID-switch, IDF1, MOTA or
    identity-preservation FIGURE may appear, under any key, anywhere in the
    document -- including inside the fields that explain why not."""
    report = _report()
    banned = re.compile(r"id_switch|idf1|mota|identity_preserv", re.IGNORECASE)
    offending = sorted(key for key in _keys(report, set()) if banned.search(key))
    assert offending == [], offending


def test_the_published_report_carries_no_wall_clock_timing():
    """A timing column would make the one property this report most needs
    -- byte reproducibility -- untestable. Per-method cost is published in
    counting_accuracy.json."""
    report = _report()
    keys = _keys(report, set())
    for banned in ("timing", "seconds", "ms_per_frame", "elapsed"):
        assert banned not in keys, banned


# -- the assembled report ----------------------------------------------------


def _synthetic_ground_truth(labels) -> GroundTruth:
    return GroundTruth(
        path=Path("data/groundtruth/motorway_inbound_gt.json"),
        clip="motorway-a40.webm",
        fps=30.0,
        start_frame=0,
        end_frame=LAST_FRAME,
        gate_name=GATE_NAME,
        gate_start=(0.06, 0.80),
        gate_end=(0.46, 0.80),
        protocol="data/groundtruth/PROTOCOL.md",
        labeller="Safdar Hussain",
        labelled_on="2026-08-15",
        crossings=tuple(labels),
    )


def _built_report(script=None) -> dict:
    script = script or _load_script()
    labels = _labels()
    return script.build_report(
        _traffic(),
        _synthetic_ground_truth(labels),
        _gate(),
        detector={"model": "yolo11s.pt"},
        target_rates=(30.0, 5.0),
        drop_fractions=(0.0, 0.2),
        dropout_probabilities=(0.0, 0.2),
        jitter_sigmas=(0.0, 4.0),
        crossing_f1_by_level={
            f"{PROTOCOL_FRAME_RATE}@30 fps": {
                "engine": 1.0,
                "centroid": 1.0,
                "greedy-iou": 1.0,
            },
            f"{PROTOCOL_FRAME_RATE}@5 fps": {
                "engine": 0.5,
                "centroid": 1.0,
                "greedy-iou": 1.0,
            },
        },
        crossing_f1_source="robustness.json",
    )


def test_the_assembled_report_is_byte_identical_across_two_builds(tmp_path):
    """No timing anywhere, every level seeded from its own value: two runs
    over the same detections must produce the same bytes.

    Compared as the BYTES ``write_report`` writes, not as a sort_keys dump.
    A canonicalising dump cannot see key-order instability -- which is
    precisely what would break real byte reproducibility, and what would
    quietly undo "the definitions come before the numbers" in the published
    file.
    """
    script = _load_script()
    first = script.write_report(tmp_path / "first.json", _built_report(script))
    second = script.write_report(tmp_path / "second.json", _built_report(script))
    assert first.read_bytes() == second.read_bytes()
    # ... and the comparison must be capable of failing: the same report with
    # two keys transposed is the same sort_keys dump and different bytes.
    report = _built_report(script)
    reordered = {key: report[key] for key in reversed(list(report))}
    third = script.write_report(tmp_path / "third.json", reordered)
    assert json.dumps(reordered, sort_keys=True) == json.dumps(
        report, sort_keys=True
    ), "the control must change ONLY the key order"
    assert third.read_bytes() != first.read_bytes(), (
        "a byte comparison that cannot see key order is a sort_keys dump in "
        "disguise"
    )


def test_the_assembled_report_names_the_label_file_and_never_a_filesystem_path():
    """``reports/`` is tracked, and the same label set lives at a different
    absolute path on every machine."""
    report = _built_report()
    assert report["ground_truth"] == "motorway_inbound_gt.json"
    text = json.dumps(report)
    assert "/Us" + "ers/" not in text
    assert str(ROOT) not in text
