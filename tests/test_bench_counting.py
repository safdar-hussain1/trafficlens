"""Tests for the counting benchmark (``trafficlens.bench.harness``).

This is the one benchmark scored against ground truth the pipeline did
not produce, so the tests here are about the SCORER, not about the
engine. A scorer that cannot embarrass the engine is worthless, and the
ways a crossing-level scorer quietly stops being able to are all
represented below: a match window that absorbs a systematic offset, a
many-to-one match that lets a burst of predictions all score against one
label, a class-equality requirement that double-charges a detector
mistake, a timing column that is secretly a sum, and a report that
publishes one flattering subset without the other.

Two tests read the COMMITTED reports rather than a synthetic fixture.
That is deliberate: a schema asserted only against a dict built in the
test proves the test's dict is well-formed, not that the published
numbers are. The reports are tracked files; if the harness changes they
must be regenerated, and these tests are what says so.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from trafficlens.bench.baselines import GreedyIoUTracker
from trafficlens.bench.harness import (
    build_methods,
    measure_detection_noise,
    median_gate_approach_px_per_frame,
    run_counting,
    run_counting_benchmark,
    sweep_band_px,
    write_report,
)
from trafficlens.bench.scoring import (
    DEFAULT_MATCH_WINDOW,
    MatchWindow,
    match_crossings,
    max_cardinality_true_positives,
)
from trafficlens.bench.slitscan import Crossing, GroundTruth
from trafficlens.core.gate import CrossingEvent, Gate, GateCounter
from trafficlens.detect.base import Detection

ROOT = Path(__file__).resolve().parents[1]
COUNTING_REPORT = ROOT / "reports" / "counting_accuracy.json"
NOISE_REPORT = ROOT / "reports" / "detection_noise.json"
PROTOCOL = ROOT / "data" / "groundtruth" / "PROTOCOL.md"

GATE_NAME = "inbound"
GATE_Y = 300.0


def _gate(name: str = GATE_NAME) -> Gate:
    """A gate drawn left to right at a constant image y, exactly like the
    shipped motorway gates: +1 (up the frame) is ``away``, -1 (down the
    frame, toward the camera) is ``toward``."""
    return Gate(
        name,
        (0.0, GATE_Y),
        (400.0, GATE_Y),
        label_positive="away",
        label_negative="toward",
        expected_direction="toward",
    )


def _label(
    crossing_id: int,
    frame: int,
    class_name: str = "car",
    direction: str = "toward",
    confidence: str = "certain",
) -> Crossing:
    return Crossing(crossing_id, frame, class_name, direction, confidence)


def _prediction(
    frame: int,
    class_name: str = "car",
    direction: str = "toward",
    gate: str = GATE_NAME,
    track_id: int = 1,
) -> CrossingEvent:
    return CrossingEvent(
        track_id=track_id,
        class_name=class_name,
        gate=gate,
        direction=direction,
        signed_direction=-1 if direction == "toward" else 1,
        frame_index=frame,
        timestamp=frame / 30.0,
        crossing_x=100.0,
        crossing_y=GATE_Y,
        speed_kmh=None,
        is_violation=False,
    )


def _ground_truth(labels, *, end_frame: int = 734) -> GroundTruth:
    return GroundTruth(
        path=Path("data/groundtruth/motorway_inbound_gt.json"),
        clip="motorway-a40.webm",
        fps=30.0,
        start_frame=0,
        end_frame=end_frame,
        gate_name=GATE_NAME,
        gate_start=(0.06, 0.80),
        gate_end=(0.46, 0.80),
        protocol="data/groundtruth/PROTOCOL.md",
        labeller="Safdar Hussain",
        labelled_on="2026-08-15",
        crossings=tuple(labels),
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


def _stream(frames: dict[int, list[Detection]], last_frame: int):
    """``(frame_index, timestamp, detections)`` triples for every frame up
    to ``last_frame``, empty where ``frames`` has no entry."""
    return [
        (index, index / 30.0, frames.get(index, []))
        for index in range(last_frame + 1)
    ]


# -- the scorer / runner seam ------------------------------------------------


def test_the_scorer_imports_no_tracker_no_detector_and_no_counting_rule():
    """``scoring`` decides what a match IS; it must not know how a
    prediction was produced.

    The split exists so the scorer can be pointed at any source of
    crossings, and so that nothing under test can reach into the thing
    scoring it. An import creeping back in would undo that silently, so
    the module's own import graph is asserted rather than trusted.
    """
    import ast

    source = Path("src/trafficlens/bench/scoring.py")
    tree = ast.parse((ROOT / source).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    forbidden = {
        "trafficlens.track.tracker",
        "trafficlens.bench.baselines",
        "trafficlens.bench.harness",
        "trafficlens.detect.base",
        "trafficlens.pipeline",
        "torch",
        "ultralytics",
    }
    assert imported & forbidden == set(), sorted(imported & forbidden)
    assert "trafficlens.bench.slitscan" in imported  # the labels
    assert "trafficlens.core.gate" in imported  # the CrossingEvent type


# -- the match window --------------------------------------------------------


def test_the_default_match_window_is_asymmetric_one_before_four_after():
    assert DEFAULT_MATCH_WINDOW.frames_before == 1
    assert DEFAULT_MATCH_WINDOW.frames_after == 4
    assert DEFAULT_MATCH_WINDOW.reason.strip() != ""


def test_the_window_boundaries_are_closed_and_asymmetric():
    """The whole point of the ruling: a 4-frames-LATE prediction matches
    and a 2-frames-EARLY one does not. A symmetric window fails this in
    one direction or the other whatever its width."""
    window = DEFAULT_MATCH_WINDOW
    labels = [_label(1, 100)]

    for offset, expected in ((-2, 0), (-1, 1), (0, 1), (4, 1), (5, 0)):
        result = match_crossings(
            [_prediction(100 + offset)], labels, window, gate_name=GATE_NAME
        )
        assert result.true_positives == expected, f"offset {offset:+d}"


def test_the_asymmetry_shrinks_the_ambiguous_region_without_removing_it():
    """The closest real label pair is 411 and 416, five frames apart.

    Asymmetric windows share ONE frame where a symmetric +-4 would share
    four -- a four-fold reduction, and the whole of what the asymmetry
    buys. It does not make the overlap harmless, and the second half of
    this test is the counterexample: predictions that sit ON the shared
    frame are mis-assigned. An earlier version asserted "harmless" with
    predictions at +1, deltas that cannot reach the shared frame and so
    could never have falsified the claim.
    """
    window = DEFAULT_MATCH_WINDOW
    early = set(range(411 - window.frames_before, 411 + window.frames_after + 1))
    late = set(range(416 - window.frames_before, 416 + window.frames_after + 1))
    assert early & late == {415}

    wide_early = set(range(411 - 4, 411 + 4 + 1))
    wide_late = set(range(416 - 4, 416 + 4 + 1))
    assert len(wide_early & wide_late) == 4

    labels = [_label(10, 411), _label(11, 416, class_name="truck")]

    # Deltas of +1: neither prediction can reach frame 415, so both pair
    # off correctly. This is the easy case.
    clean = match_crossings(
        [_prediction(412), _prediction(417, class_name="truck")],
        labels,
        window,
        gate_name=GATE_NAME,
    )
    assert clean.true_positives == 2
    assert clean.misses == ()

    # Deltas of +4 on both: two CORRECT predictions, and the shared frame
    # 415 is mis-assigned. Label 416 takes it at distance 1 before label
    # 411 can take it at distance 4.
    ambiguous = match_crossings(
        [_prediction(415), _prediction(420, class_name="truck")],
        labels,
        window,
        gate_name=GATE_NAME,
    )
    assert ambiguous.true_positives == 1
    assert [labels[i].frame for i in ambiguous.misses] == [411]
    assert [
        ambiguous.predictions[i].frame_index for i in ambiguous.false_positives
    ] == [420]
    # The matched pair is 415 -> label 416, a delta of -1.
    assert ambiguous.matches == ((0, 1, -1),)

    # And that is a GREEDY shortfall, not an impossible one: a
    # maximum-cardinality assignment over the same eligible pairs gets 2.
    assert (
        max_cardinality_true_positives(
            [_prediction(415), _prediction(420, class_name="truck")],
            labels,
            window,
            gate_name=GATE_NAME,
        )
        == 2
    )


def test_greedy_matching_never_beats_maximum_cardinality():
    """Greedy errs only in the conservative direction. It is allowed to
    under-report the engine; it must never over-report it."""
    labels = [_label(1, 411), _label(2, 416)]
    for frames in ((415, 420), (412, 417), (411, 416), (415, 416), (410, 415)):
        predictions = [_prediction(frame) for frame in frames]
        greedy = match_crossings(
            predictions, labels, DEFAULT_MATCH_WINDOW, gate_name=GATE_NAME
        )
        best = max_cardinality_true_positives(
            predictions, labels, DEFAULT_MATCH_WINDOW, gate_name=GATE_NAME
        )
        assert greedy.true_positives <= best, frames


def test_equidistant_ties_break_canonically_not_by_input_order():
    """A prediction equidistant from two labels must go to the same one
    however the label list is ordered.

    Ranking by list position instead is deterministic only because
    ``GroundTruth`` happens to validate frame order -- nothing asserts
    that here, and this project has already been bitten twice by
    determinism that held by coincidence.
    """
    early, late = _label(1, 99), _label(2, 101)
    prediction = [_prediction(100)]  # +1 from 99, -1 from 101: a true tie

    forward = match_crossings(
        prediction, [early, late], gate_name=GATE_NAME
    )
    reversed_order = match_crossings(
        prediction, [late, early], gate_name=GATE_NAME
    )

    assert forward.true_positives == reversed_order.true_positives == 1
    matched_forward = forward.labels[forward.matches[0][1]]
    matched_reversed = reversed_order.labels[reversed_order.matches[0][1]]
    assert matched_forward.frame == matched_reversed.frame == 99
    assert matched_forward.id == matched_reversed.id == 1


# -- one-to-one matching -----------------------------------------------------


def test_two_predictions_near_one_label_yield_one_hit_and_one_false_alarm():
    labels = [_label(1, 100)]
    result = match_crossings(
        [_prediction(101), _prediction(103)], labels, gate_name=GATE_NAME
    )
    assert result.true_positives == 1
    assert len(result.false_positives) == 1
    assert result.misses == ()
    # The NEARER prediction is the one that scores.
    matched_pred_index = result.matches[0][0]
    assert matched_pred_index == 0


def test_matching_is_nearest_frame_first_not_first_come_first_served():
    """Leverage: a first-come pass over the predictions gives prediction 0
    to label 0 and leaves prediction 1 unmatched. Nearest-first pairs both
    off, which is a strictly better -- and the correct -- reading."""
    labels = [_label(1, 100), _label(2, 104)]
    predictions = [_prediction(104), _prediction(100)]
    result = match_crossings(predictions, labels, gate_name=GATE_NAME)

    assert result.true_positives == 2
    assert sorted((pred, gt) for pred, gt, _ in result.matches) == [(0, 1), (1, 0)]


def test_a_perfect_prediction_set_scores_one_across_the_board():
    labels = [_label(i + 1, frame) for i, frame in enumerate((10, 50, 200))]
    result = match_crossings(
        [_prediction(frame) for frame in (10, 50, 200)],
        labels,
        gate_name=GATE_NAME,
    )
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.count_error == 0
    assert result.signed_bias == 0


def test_an_empty_prediction_set_scores_recall_zero_without_dividing_by_zero():
    labels = [_label(1, 10), _label(2, 50)]
    result = match_crossings([], labels, gate_name=GATE_NAME)
    assert result.recall == 0.0
    assert result.precision == 0.0
    assert result.f1 == 0.0
    assert result.signed_bias == -2
    assert result.count_error == 2
    assert result.miss_rate == 1.0


def test_an_empty_label_set_does_not_divide_by_zero():
    result = match_crossings([_prediction(10)], [], gate_name=GATE_NAME)
    assert result.recall == 0.0
    assert result.precision == 0.0
    assert result.miss_rate == 0.0
    assert result.phantom_rate == 0.0
    assert result.signed_bias == 1


def test_count_error_is_absolute_and_bias_is_signed():
    labels = [_label(1, 10)]
    over = match_crossings(
        [_prediction(10), _prediction(500)], labels, gate_name=GATE_NAME
    )
    assert (over.signed_bias, over.count_error) == (1, 1)

    under = match_crossings([], labels, gate_name=GATE_NAME)
    assert (under.signed_bias, under.count_error) == (-1, 1)


# -- what matching is and is not sensitive to --------------------------------


def test_matching_is_class_blind_and_reports_the_class_error_separately():
    """A car detected as a truck must not be charged twice. It matches --
    and shows up in class consistency, and in the per-class counts."""
    labels = [_label(1, 100, class_name="car")]
    result = match_crossings(
        [_prediction(100, class_name="truck")], labels, gate_name=GATE_NAME
    )
    assert result.true_positives == 1
    assert result.false_positives == ()
    assert result.misses == ()

    consistency = result.as_dict()["class_consistency"]
    assert consistency["matched"] == 1
    assert consistency["same_class"] == 0
    assert consistency["rate"] == 0.0
    assert consistency["confusions"] == {"car -> truck": 1}

    per_class = result.as_dict()["per_class"]
    assert per_class["car"] == {"predicted": 0, "ground_truth": 1}
    assert per_class["truck"] == {"predicted": 1, "ground_truth": 0}


def test_matching_is_direction_aware():
    """A wrong-direction prediction at this gate is a genuine counting
    error: it is a false alarm AND the label it sits on is a miss."""
    labels = [_label(1, 100, direction="toward")]
    result = match_crossings(
        [_prediction(100, direction="away")], labels, gate_name=GATE_NAME
    )
    assert result.true_positives == 0
    assert len(result.false_positives) == 1
    assert len(result.misses) == 1

    per_direction = result.as_dict()["per_direction"]
    assert per_direction["toward"]["ground_truth"] == 1
    assert per_direction["away"]["predicted"] == 1


def test_a_prediction_at_another_gate_never_matches():
    labels = [_label(1, 100)]
    result = match_crossings(
        [_prediction(100, gate="outbound")], labels, gate_name=GATE_NAME
    )
    assert result.true_positives == 0
    assert len(result.false_positives) == 1
    assert len(result.misses) == 1


def test_matched_frame_deltas_are_reported_so_a_systematic_offset_is_visible():
    """The asymmetric window only stays honest while the offset it
    encodes is published. A scorer that hides the deltas could widen the
    window forever and nobody would see it."""
    labels = [_label(1, 100), _label(2, 200)]
    result = match_crossings(
        [_prediction(103), _prediction(203)], labels, gate_name=GATE_NAME
    )
    delta = result.as_dict()["matched_frame_delta"]
    assert delta["mean"] == 3.0
    assert delta["min"] == 3
    assert delta["max"] == 3
    assert delta["histogram"] == {"3": 2}


# -- driving a tracker and a counting rule over cached detections ------------


def test_a_track_seen_for_the_first_time_cannot_cross():
    """No substitute origin, ever. A track whose first observed anchor is
    already past the gate has no swept segment, and inventing one would
    fabricate a crossing spanning half the frame."""
    stream = _stream(
        {index: [_det(100.0, 400.0 + 5.0 * index)] for index in range(6)}, 5
    )
    events = run_counting(stream, GreedyIoUTracker(), GateCounter(_gate()))
    assert events == []


class _RecordingCounter(GateCounter):
    """A gate counter that records the frame each ``forget`` arrived on,
    so reaping can be observed at the interface instead of guessed at."""

    def __init__(self, gate, clock: list[int]) -> None:
        super().__init__(gate)
        self.clock = clock
        self.forgotten: list[tuple[int, int]] = []

    def forget(self, track_id: int) -> None:
        self.forgotten.append((track_id, self.clock[0]))
        super().forget(track_id)


def test_run_counting_reaps_a_dead_track_mid_clip_on_the_tracker_s_clock():
    """The tracker never announces a death, so tracks are reaped on a
    clock. This pins the exact boundary: a confirmed track survives while
    ``time_since_update <= max_age`` and may still re-associate at exactly
    ``max_age``, so ``last_seen + max_age`` must reap NOTHING and
    ``last_seen + max_age + 1`` is the first frame it is provably gone.

    A fixture whose track is still alive on the final frame cannot tell
    mid-clip reaping apart from the end-of-session drain -- delete the
    reap and it still passes -- so the vehicle here disappears at frame 7
    and the clip runs well past its death.
    """
    tracker = GreedyIoUTracker()
    max_age = tracker.max_age
    last_seen = 7
    clock = [-1]

    class Clocked(GreedyIoUTracker):
        def update(self, detections, frame_index):
            clock[0] = frame_index
            return super().update(detections, frame_index)

    # Detections for frames 0-7 only; frames 8 onward are empty, so the
    # track is last seen on frame 7 and then dies where we can watch it.
    stream = _stream(
        {index: [_det(100.0, 100.0 + 8.0 * index)] for index in range(last_seen + 1)},
        last_seen + max_age + 6,
    )
    counter = _RecordingCounter(_gate(), clock)
    run_counting(stream, Clocked(), counter)

    assert counter.forgotten == [(1, last_seen + max_age + 1)]

    # Both sides of the boundary, stated as the frames themselves.
    reaped_on = counter.forgotten[0][1]
    assert reaped_on == last_seen + max_age + 1
    assert reaped_on != last_seen + max_age


def test_run_counting_releases_a_track_still_alive_at_the_end_of_the_clip():
    """The end-of-session drain, which is a different code path from
    mid-clip reaping and would otherwise leave the last tracks alive
    holding state forever."""
    clock = [-1]

    class Clocked(GreedyIoUTracker):
        def update(self, detections, frame_index):
            clock[0] = frame_index
            return super().update(detections, frame_index)

    stream = _stream(
        {index: [_det(100.0, 100.0 + 8.0 * index)] for index in range(8)}, 7
    )
    counter = _RecordingCounter(_gate(), clock)
    run_counting(stream, Clocked(), counter)
    assert [track_id for track_id, _frame in counter.forgotten] == [1]


def test_run_counting_counts_a_crossing_the_way_the_pipeline_does():
    stream = _stream(
        {index: [_det(100.0, 261.0 + 8.0 * index)] for index in range(10)}, 9
    )
    events = run_counting(stream, GreedyIoUTracker(), GateCounter(_gate()))
    assert [event.direction for event in events] == ["toward"]
    # 261 + 8*i is 293 at i = 4 and 301 at i = 5, so the swept segment
    # first cuts the gate on frame 5.
    assert events[0].frame_index == 5


# -- the benchmark report ----------------------------------------------------


def _benchmark(methods, labels=None):
    truth = _ground_truth(labels if labels is not None else [_label(1, 100)])
    return run_counting_benchmark(
        _stream({}, 3),
        truth,
        methods,
        gate=_gate(),
    )


def test_the_report_carries_the_match_window_object_and_no_scalar_tolerance():
    report = _benchmark({"only": lambda detections: []})

    assert isinstance(report["protocol"], str) and report["protocol"]
    window = report["match_window"]
    assert window["frames_before"] == 1
    assert window["frames_after"] == 4
    assert window["reason"].strip() != ""

    text = json.dumps(report)
    assert "tolerance_frames" not in text


#: One-sided slack allowed when a method's measured time is checked
#: against the cost injected into it. ``time.sleep`` can only overshoot,
#: never undershoot, so the bound is ``[cost, cost + SLEEP_SLACK_S]``.
#: Measured on this machine over 40 trials at each of 0.01/0.05/0.09 s:
#: worst overshoot 10.1 ms, median 3.8-7.9 ms. 25 ms is ~2.5x the worst
#: observed, which leaves room for CPU contention while staying far
#: below the 60 ms gap a cumulative timing column would introduce.
SLEEP_SLACK_S = 0.025


def test_the_timing_block_holds_one_measurement_per_method_never_a_sum():
    """Each method is timed in its OWN bracket.

    Every entry is bounded against ITS OWN injected cost, both sides.
    Bounding each method against the NEXT method's cost instead is not
    enough: a cumulative column (each entry the running total 0.01 / 0.06
    / 0.15) satisfies relative bounds by construction, and leaves the
    most expensive method -- the one whose number gets published -- with
    no upper bound at all.
    """
    costs = {"cheap": 0.01, "middling": 0.05, "dear": 0.09}

    def sleeper(seconds):
        def method(detections):
            time.sleep(seconds)
            return []

        return method

    report = _benchmark({name: sleeper(cost) for name, cost in costs.items()})
    timing = report["timing"]

    assert sorted(timing) == sorted(costs)
    assert len(set(entry["seconds"] for entry in timing.values())) == 3

    for name, cost in costs.items():
        measured = timing[name]["seconds"]
        assert cost <= measured <= cost + SLEEP_SLACK_S, (
            f"{name}: measured {measured:.4f}s against an injected "
            f"{cost:.4f}s -- a method's timing must contain its own work "
            f"and nothing else"
        )


def test_every_method_is_handed_the_identical_detection_stream():
    seen: list[int] = []

    def method(detections):
        seen.append(id(detections))
        return []

    report = _benchmark({"a": method, "b": method, "c": method})
    assert len(seen) == 3
    assert len(set(seen)) == 1
    assert report["frames"] == 4


def test_both_confidence_subsets_are_published_together():
    labels = [
        _label(1, 100, confidence="certain"),
        _label(2, 200, confidence="probable"),
    ]
    report = _benchmark(
        {"engine": lambda detections: [_prediction(100)]}, labels=labels
    )
    scores = report["methods"]["engine"]

    assert scores["full"]["n_ground_truth"] == 2
    assert scores["full"]["recall"] == 0.5
    assert scores["certain_only"]["n_ground_truth"] == 1
    assert scores["certain_only"]["recall"] == 1.0


def test_certain_only_treats_probable_labels_as_ignore_regions():
    """A prediction landing on a `probable` label is neither credited nor
    charged: it leaves both the numerator and the denominator.

    Naive subsetting scores this same input at precision 0.5, because the
    correct prediction of the probable crossing becomes a false alarm.
    Under ignore semantics it is 1.0, and both are published."""
    labels = [
        _label(1, 100, confidence="certain"),
        _label(2, 200, confidence="probable"),
    ]
    report = _benchmark(
        {"engine": lambda detections: [_prediction(100), _prediction(200)]},
        labels=labels,
    )
    scores = report["methods"]["engine"]

    ignored = scores["certain_only"]
    assert ignored["n_ground_truth"] == 1
    assert ignored["n_predicted"] == 1  # the probable-matched one is gone
    assert ignored["true_positives"] == 1
    assert ignored["precision"] == 1.0
    assert ignored["recall"] == 1.0
    assert ignored["predictions_moved_to_ignore"] == 1
    assert ignored["labels_ignored"] == 1

    naive = scores["certain_only_naive"]
    assert naive["precision"] == 0.5
    assert naive["false_positives_matching_probable_labels"] == 1


def test_a_phantom_outside_every_ignore_region_is_still_charged():
    """Ignore semantics must not swallow genuine false alarms. Only a
    prediction the JOINT match paired with a probable label is ignored;
    one that matched nothing stays chargeable."""
    labels = [
        _label(1, 100, confidence="certain"),
        _label(2, 200, confidence="probable"),
    ]
    report = _benchmark(
        {
            "engine": lambda detections: [
                _prediction(100),
                _prediction(200),
                _prediction(600),  # nothing near it
            ]
        },
        labels=labels,
    )
    ignored = report["methods"]["engine"]["certain_only"]
    assert ignored["n_predicted"] == 2
    assert ignored["true_positives"] == 1
    assert ignored["false_positives"] == [600]
    assert ignored["precision"] == 0.5


def test_a_probable_label_can_claim_a_prediction_a_certain_label_needed():
    """The joint match is authoritative. A prediction taken by a probable
    label is removed from the certain subset even though a certain label
    was also within window -- which turns that certain label into a miss.

    Stated as a test because it is the one way ignore semantics can LOWER
    recall, and an unexplained recall gap would otherwise look like
    evidence the certain crossings are harder."""
    labels = [
        _label(1, 101, confidence="probable"),
        _label(2, 103, confidence="certain"),
    ]
    # Frame 102: +1 from the probable label, -1 from the certain one, and
    # the canonical tie-break gives it to the earlier-framed probable one.
    report = _benchmark(
        {"engine": lambda detections: [_prediction(102)]}, labels=labels
    )
    ignored = report["methods"]["engine"]["certain_only"]

    assert ignored["predictions_moved_to_ignore"] == 1
    assert ignored["n_predicted"] == 0
    assert ignored["true_positives"] == 0
    assert ignored["misses"] == [103]
    assert ignored["recall"] == 0.0
    # No denominator, so no precision -- not a claim the engine was wrong.
    assert ignored["precision"] == 0.0


def test_certain_only_is_a_restriction_of_the_full_match_not_a_rematch():
    """Every certain-only true positive must be a pair the full-set match
    already made, with the same frame delta. Re-matching against the
    surviving subset could pair a prediction differently and make the two
    published figures incomparable."""
    labels = [
        _label(1, 100, confidence="certain"),
        _label(2, 140, confidence="probable"),
        _label(3, 180, confidence="certain"),
    ]
    predictions = [_prediction(102), _prediction(143), _prediction(182)]
    report = _benchmark(
        {"engine": lambda detections: predictions}, labels=labels
    )
    scores = report["methods"]["engine"]

    full_deltas = scores["full"]["matched_frame_delta"]["histogram"]
    certain_deltas = scores["certain_only"]["matched_frame_delta"]["histogram"]
    # 102-100 = +2, 143-140 = +3, 182-180 = +2.
    assert full_deltas == {"2": 2, "3": 1}
    # The probable pair (the +3) is removed; the two certain pairs keep
    # their own deltas exactly as the joint match assigned them.
    assert certain_deltas == {"2": 2}
    assert scores["certain_only"]["predictions_moved_to_ignore"] == 1


def test_the_report_publishes_a_computed_max_cardinality_not_the_greedy_count():
    """The max-cardinality field is the published check that greedy cost
    the engine nothing. Echoing the greedy count into it would make the
    check assert itself, so this pins a case where the two genuinely
    diverge: the 411/416 shared frame with two +4 predictions."""
    labels = [_label(1, 411), _label(2, 416)]
    report = _benchmark(
        {"engine": lambda detections: [_prediction(415), _prediction(420)]},
        labels=labels,
    )
    full = report["methods"]["engine"]["full"]

    assert full["true_positives"] == 1  # greedy mis-assigns the shared frame
    assert full["max_cardinality_true_positives"] == 2  # a better pairing exists
    assert report["matching"]["greedy_equals_max_cardinality"] is False


def test_the_report_publishes_no_speed_figure():
    """Speed validation is gated on an independently anchored along-road
    scale; this task publishes counting only."""
    report = _benchmark({"only": lambda detections: []})
    text = json.dumps(report).lower()
    assert "kmh" not in text
    assert "km/h" not in text


# -- the band_px trade-off ---------------------------------------------------


def _band_sweep_stream():
    """One crossing the band misses when narrow, and one approach that
    phantoms when the band is wide, far enough apart in time and space
    that neither tracker confuses them.

    Vehicle A crosses: its anchor steps 40 px a frame and its closest
    approach to the gate is 15 px, at frame 7; it is genuinely past the
    gate at frame 8.

    Vehicle B never crosses: it closes to 25 px of the gate at frame 24
    and retreats.
    """
    frames: dict[int, list[Detection]] = {}
    for i in range(11):
        frames[i] = [_det(100.0, 5.0 + 40.0 * i)]
    approach = [175.0, 200.0, 225.0, 250.0, 275.0, 250.0, 225.0, 200.0, 175.0]
    for offset, y in enumerate(approach):
        frames[20 + offset] = [_det(300.0, y)]
    return _stream(frames, 30)


def test_the_band_sweep_trades_misses_against_phantoms():
    entries = sweep_band_px(
        _band_sweep_stream(),
        [_label(1, 8)],
        gate=_gate(),
        band_values=(10.0, 20.0, 30.0),
        tracker_factory=GreedyIoUTracker,
    )
    by_band = {entry["band_px"]: entry for entry in entries}
    assert sorted(by_band) == [10.0, 20.0, 30.0]

    assert (by_band[10.0]["misses"], by_band[10.0]["false_positives"]) == (1, 0)
    assert (by_band[20.0]["misses"], by_band[20.0]["false_positives"]) == (0, 0)
    assert (by_band[30.0]["misses"], by_band[30.0]["false_positives"]) == (0, 1)

    # The two error modes are separate series, never one netted number.
    assert by_band[10.0]["miss_rate"] == 1.0
    assert by_band[10.0]["phantom_rate"] == 0.0
    assert by_band[30.0]["miss_rate"] == 0.0
    assert by_band[30.0]["phantom_rate"] == 1.0


def test_the_gate_approach_speed_is_measured_not_assumed():
    """A band result is unreadable without the pixel speed at the gate:
    a band of half-width b fires about b/v frames early. Vehicle A steps
    40 px a frame and vehicle B 25, and A supplies three of the five
    in-range steps."""
    speed = median_gate_approach_px_per_frame(
        _band_sweep_stream(),
        _gate(),
        within_px=30.0,
        tracker_factory=GreedyIoUTracker,
    )
    assert speed == pytest.approx(40.0)


def test_the_gate_approach_speed_is_none_when_nothing_comes_near():
    stream = _stream(
        {index: [_det(100.0, 50.0 + 4.0 * index)] for index in range(10)}, 9
    )
    assert (
        median_gate_approach_px_per_frame(
            stream, _gate(), within_px=30.0, tracker_factory=GreedyIoUTracker
        )
        is None
    )


# -- the composed method set -------------------------------------------------


def test_build_methods_composes_every_tracker_with_every_counting_rule():
    methods = build_methods(_gate())
    assert len(methods) == 9
    for tracker in ("engine", "centroid", "greedy-iou"):
        for rule in ("gate", "band", "per-frame"):
            assert f"{tracker}+{rule}" in methods


def test_each_composed_method_builds_fresh_state_every_call():
    """A method that reused one tracker instance would score its second
    run against the first run's leftover track ids."""
    method = build_methods(_gate())["greedy-iou+gate"]
    stream = _stream(
        {index: [_det(100.0, 261.0 + 8.0 * index)] for index in range(10)}, 9
    )
    first = method(stream)
    second = method(stream)
    assert len(first) == 1  # or the comparison below would be vacuous
    assert [event.track_id for event in first] == [event.track_id for event in second]
    assert [event.frame_index for event in first] == [
        event.frame_index for event in second
    ]


# -- detection noise ---------------------------------------------------------


def _noise_stream(width_noise: np.ndarray, frames: int = 40):
    detections: dict[int, list[Detection]] = {}
    for index in range(frames):
        detections[index] = [
            _det(
                100.0,
                50.0 + 6.0 * index,
                width=120.0 + float(width_noise[index]),
                height=120.0,
            )
        ]
    return _stream(detections, frames - 1)


def test_a_noiseless_trajectory_measures_zero_detection_noise():
    zeros = np.zeros(40)
    noise = measure_detection_noise(_noise_stream(zeros))
    for quantity in ("box_width", "box_height", "centre_x", "centre_y"):
        assert noise["residuals"][quantity]["std_px"] == pytest.approx(0.0, abs=1e-9)
    assert noise["residuals"]["box_width"]["n"] > 0


def test_measured_noise_scales_with_the_noise_actually_present():
    rng = np.random.default_rng(20260815)
    base = rng.normal(0.0, 1.0, 40)
    quiet = measure_detection_noise(_noise_stream(base))
    loud = measure_detection_noise(_noise_stream(base * 4.0))

    quiet_std = quiet["residuals"]["box_width"]["std_px"]
    loud_std = loud["residuals"]["box_width"]["std_px"]
    assert quiet_std > 0.1
    assert loud_std / quiet_std == pytest.approx(4.0, rel=0.05)

    # Only the width was jittered, so the height residual stays at zero.
    assert loud["residuals"]["box_height"]["std_px"] == pytest.approx(0.0, abs=1e-9)


def test_the_detection_noise_report_calls_itself_a_proxy():
    noise = measure_detection_noise(_noise_stream(np.zeros(40)))
    assert "proxy" in noise["caveat"].lower()
    assert noise["median_filter_frames"] >= 3


def test_writing_a_report_refuses_a_non_finite_figure(tmp_path):
    """A NaN accuracy figure is not a number a reader can act on, and the
    tokens ``NaN``/``Infinity`` are not valid JSON however many parsers
    accept them. Nothing partial is left behind either."""
    target = tmp_path / "broken.json"
    with pytest.raises(ValueError):
        write_report(target, {"precision": float("nan")})
    assert not target.exists()

    write_report(target, {"precision": 1.0})
    assert json.loads(target.read_text()) == {"precision": 1.0}


# -- the documents and the committed reports ---------------------------------


def test_the_protocol_states_the_same_match_window_the_scorer_uses():
    """PROTOCOL.md and the scorer must not drift apart. The protocol used
    to fix a symmetric +-2 window; if that sentence comes back while the
    scorer stays asymmetric, the label set's own rules are a lie."""
    text = PROTOCOL.read_text()
    assert "[label - 1, label + 4]" in text
    assert "within **2 frames**" not in text
    assert "+0/-4" in text
    # The closest-pair evidence the asymmetry rests on.
    assert "411" in text and "416" in text


def test_the_protocol_states_the_ignore_region_rule_the_scorer_implements():
    """PROTOCOL.md's scoring-RULE prose, pinned to the scorer's behaviour.

    Only the match window was pinned before this. The rest of the document's
    scoring section -- how the `certain`-only figure is computed, and that
    matching is greedy rather than maximum-cardinality -- was held up by
    nothing, which is exactly the drift a written-down-first protocol exists to
    prevent: the mutation battery replaced "neither credited nor charged" with
    "charged as a false positive", inverting the published rule, and the whole
    suite stayed green.

    Two halves, and both are needed. The first pins the sentences. The second
    demonstrates that the sentences describe THIS scorer, so the pin cannot be
    satisfied by prose that has quietly stopped being true -- and the expected
    numbers below are worked out from the protocol's own three steps, not read
    off the scorer:

        labels: certain @100, probable @200; predictions @100, @200
        step 1  match once against ALL rows      -> 100<->100, 200<->200
        step 2  remove, from BOTH sides, every pair matched to a probable row
                                                -> label 200 and prediction 200
                                                   both leave
        step 3  score the certain rows against what remains
                                                -> 1 label, 1 prediction, 1 hit
                                                -> precision 1.0, recall 1.0

    Under the inverted rule the surviving prediction would be charged, giving
    precision 0.5 -- which is what the document calls the artefact that motivates
    ignore semantics, and what the report publishes separately as
    ``certain_only_naive``. Asserting both is what makes this a discriminating
    pair rather than a single number that could be reached either way.
    """
    # Whitespace-collapsed, because the document is hand-wrapped and which
    # word a sentence breaks after is the document's business, not a claim.
    text = " ".join(PROTOCOL.read_text().split())

    # -- the sentences that state the rule ------------------------------------
    for sentence in [
        # The treatment itself, and the phrase the whole rule turns on.
        "treated as **ignore regions**",
        "neither credited nor charged",
        # The three steps, in order.
        "Match once against **all** rows",
        "Remove, from both sides, every pair matched to a `probable` row",
        "Score the `certain` rows against the predictions that remain",
        # Why it is a restriction and not a second scoring run.
        "Scoring by restriction of the single joint match",
        "never by re-matching against the surviving subset",
        # The limitation the rule carries, which the report must publish.
        "count of predictions moved to the ignore set must be published",
        # The matching rule, and the obligation it puts on any scorer.
        "Matching is greedy nearest-first, not maximum-cardinality.",
        "**publish the maximum-cardinality count alongside its own**",
        "signed** frame offsets of its matched pairs",
    ]:
        assert sentence in text, (
            f"PROTOCOL.md no longer states {sentence!r}. The scoring rules are "
            f"fixed in this document, and a rule that lives only in the scorer "
            f"is a rule nobody can audit -- restore the sentence or change the "
            f"scorer to match, but never quietly drop it"
        )

    # -- and the scorer does what those sentences say --------------------------
    labels = [
        _label(1, 100, confidence="certain"),
        _label(2, 200, confidence="probable"),
    ]
    report = _benchmark(
        {"engine": lambda detections: [_prediction(100), _prediction(200)]},
        labels=labels,
    )
    scores = report["methods"]["engine"]

    ignored = scores["certain_only"]
    assert ignored["n_ground_truth"] == 1, "step 2 did not remove the probable row"
    assert ignored["n_predicted"] == 1, (
        "the prediction the probable row claimed was not removed from the "
        "denominator: PROTOCOL.md says it is neither credited NOR charged"
    )
    assert ignored["true_positives"] == 1
    assert ignored["precision"] == 1.0
    assert ignored["recall"] == 1.0
    # The absorbed mass the document requires to be visible.
    assert ignored["predictions_moved_to_ignore"] == 1

    # The artefact the document names, published beside it rather than instead.
    assert scores["certain_only_naive"]["precision"] == 0.5

    # A restriction of the joint match, not a rematch: the surviving pair keeps
    # the delta the joint match gave it.
    assert scores["full"]["matched_frame_delta"]["histogram"] == {"0": 2}
    assert ignored["matched_frame_delta"]["histogram"] == {"0": 1}

    # Greedy nearest-first, with the maximum-cardinality count published beside
    # it so the size of the gap is measured rather than assumed.
    assert scores["full"]["max_cardinality_true_positives"] >= (
        scores["full"]["true_positives"]
    )


def _ratio(numerator: int, denominator: int) -> float:
    """The scorer's own no-denominator convention, restated independently
    here so the check is not the implementation checking itself."""
    return numerator / denominator if denominator else 0.0


def test_the_committed_counting_report_is_self_consistent():
    assert COUNTING_REPORT.is_file(), (
        f"{COUNTING_REPORT} is missing: run scripts/bench_counting.py"
    )
    report = json.loads(COUNTING_REPORT.read_text())

    assert report["match_window"] == {
        "frames_before": 1,
        "frames_after": 4,
        "reason": DEFAULT_MATCH_WINDOW.reason,
    }
    assert "tolerance_frames" not in json.dumps(report)

    # A tracked report may carry no machine-specific path. tests/test_guards.py
    # cannot catch this one until the file is staged, so it is caught here.
    assert "/" not in report["ground_truth"]
    assert str(ROOT) not in json.dumps(report)

    assert report["labels"]["total"] == 17
    assert report["labels"]["certain"] == 7

    methods = report["methods"]
    assert len(methods) == 9
    assert sorted(report["timing"]) == sorted(methods)
    assert len(set(entry["seconds"] for entry in report["timing"].values())) == len(
        methods
    )

    for name, scores in methods.items():
        for subset, result in scores.items():
            where = f"{name}/{subset}"
            true_positives = result["true_positives"]
            misses = len(result["misses"])
            false_positives = len(result["false_positives"])

            # Cardinality: the arrays account for every label and every
            # prediction.
            assert true_positives + misses == result["n_ground_truth"], where
            assert (
                true_positives + false_positives == result["n_predicted"]
            ), where

            # Every PUBLISHED RATE recomputed from those same arrays.
            # Without this a report can carry cardinality-consistent
            # arrays and a fabricated headline -- 16/1/1 alongside a
            # claimed precision of 1.000 -- and pass.
            expected_precision = _ratio(true_positives, result["n_predicted"])
            expected_recall = _ratio(true_positives, result["n_ground_truth"])
            expected_f1 = (
                2.0
                * expected_precision
                * expected_recall
                / (expected_precision + expected_recall)
                if (expected_precision + expected_recall)
                else 0.0
            )
            assert result["precision"] == pytest.approx(expected_precision), where
            assert result["recall"] == pytest.approx(expected_recall), where
            assert result["f1"] == pytest.approx(expected_f1), where
            assert result["miss_rate"] == pytest.approx(
                _ratio(misses, result["n_ground_truth"])
            ), where
            assert result["phantom_rate"] == pytest.approx(
                _ratio(false_positives, result["n_ground_truth"])
            ), where
            assert result["count_error"] == abs(result["signed_bias"]), where
            assert (
                result["signed_bias"]
                == result["n_predicted"] - result["n_ground_truth"]
            ), where

            # The frame-delta histogram is the report's empirical
            # vindication of the asymmetric window, so it is pinned to the
            # window it claims to justify rather than left free.
            delta = result["matched_frame_delta"]
            assert sum(delta["histogram"].values()) == true_positives, where
            if true_positives:
                offsets = [int(key) for key in delta["histogram"]]
                assert min(offsets) == delta["min"], where
                assert max(offsets) == delta["max"], where
                assert delta["min"] >= -DEFAULT_MATCH_WINDOW.frames_before, where
                assert delta["max"] <= DEFAULT_MATCH_WINDOW.frames_after, where

    # The headline that lands on the site, pinned explicitly.
    #
    # These moved once, in Task 20, and the reason is recorded here rather than
    # only in a report: `detect.base.letterbox` changed from `cv2.INTER_LINEAR`
    # to `cv2.INTER_LINEAR_EXACT`, because plain INTER_LINEAR is intercepted by
    # a vendor resize HAL on some builds and so cannot be mirrored in the
    # browser (see that function's docstring). Preprocessing therefore shifted
    # by at most one grey level per pixel, which moved 8901 of the clip's 8914
    # cached detections by a median of 0.055 px, and one of those shifts was
    # enough to produce a second phantom crossing at frame 417.
    #
    # Recall did not move: still 16 of 17, still missing only frame 192. The
    # cost is one false positive, so precision and F1 dropped from 16/17.
    # Before: n_predicted 17, precision = recall = f1 = 16/17 = 0.9411764706.
    engine = methods["engine+gate"]["full"]
    assert engine["n_predicted"] == 18
    assert engine["n_ground_truth"] == 17
    assert engine["true_positives"] == 16
    assert engine["precision"] == pytest.approx(16 / 18)
    assert engine["recall"] == pytest.approx(16 / 17)
    assert engine["f1"] == pytest.approx(2 * (16 / 18) * (16 / 17) / (16 / 18 + 16 / 17))
    assert engine["matched_frame_delta"]["max"] <= 4
    # Greedy matching is conservative; the report says by how much.
    assert engine["max_cardinality_true_positives"] >= engine["true_positives"]

    text = json.dumps(report).lower()
    assert "kmh" not in text and "km/h" not in text
    assert any("upper bound" in caveat.lower() for caveat in report["caveats"])


def test_the_committed_detection_noise_report_is_labelled_a_proxy():
    assert NOISE_REPORT.is_file(), (
        f"{NOISE_REPORT} is missing: run scripts/bench_counting.py"
    )
    noise = json.loads(NOISE_REPORT.read_text())
    assert "proxy" in noise["caveat"].lower()
    for quantity in ("box_width", "box_height", "centre_x", "centre_y"):
        entry = noise["residuals"][quantity]
        assert entry["n"] > 0
        assert math.isfinite(entry["std_px"])
