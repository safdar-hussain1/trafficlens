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
    DEFAULT_MATCH_WINDOW,
    build_methods,
    match_crossings,
    measure_detection_noise,
    median_gate_approach_px_per_frame,
    run_counting,
    run_counting_benchmark,
    sweep_band_px,
    write_report,
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


def test_the_real_label_sets_closest_pair_stays_effectively_disjoint():
    """The 5-frame gap between labels 411 and 416 is what makes the
    asymmetric window safe, so it is asserted here on the real numbers.
    The windows touch at exactly one frame (415) and nearest-first
    matching keeps that harmless."""
    window = DEFAULT_MATCH_WINDOW
    early = set(range(411 - window.frames_before, 411 + window.frames_after + 1))
    late = set(range(416 - window.frames_before, 416 + window.frames_after + 1))
    assert early & late == {415}

    labels = [_label(10, 411), _label(11, 416, class_name="truck")]
    result = match_crossings(
        [_prediction(412), _prediction(417, class_name="truck")],
        labels,
        window,
        gate_name=GATE_NAME,
    )
    assert result.true_positives == 2
    assert result.false_positives == ()
    assert result.misses == ()


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


def test_run_counting_releases_every_track_it_saw():
    """The pipeline reaps on a clock because the tracker never announces
    a death; a harness that skips it leaks per-track state across a clip."""
    forgotten: list[int] = []

    class RecordingCounter(GateCounter):
        def forget(self, track_id: int) -> None:
            forgotten.append(track_id)
            super().forget(track_id)

    stream = _stream(
        {index: [_det(100.0, 100.0 + 8.0 * index)] for index in range(8)}, 7
    )
    run_counting(stream, GreedyIoUTracker(), RecordingCounter(_gate()))
    assert forgotten == [1]


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


def test_the_timing_block_holds_one_measurement_per_method_never_a_sum():
    """Each method is timed in its OWN bracket. A shared loop that timed
    every method and indexed one out would report the total (0.15 s) for
    all three, which the upper bounds below reject."""
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

    assert timing["cheap"]["seconds"] >= costs["cheap"]
    assert timing["cheap"]["seconds"] < costs["middling"]
    assert timing["middling"]["seconds"] >= costs["middling"]
    assert timing["middling"]["seconds"] < costs["dear"]
    assert timing["dear"]["seconds"] >= costs["dear"]


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


def test_the_certain_only_subset_declares_the_false_alarms_it_manufactures():
    """Dropping the probable labels turns their correct predictions into
    false alarms, so certain-only precision is NOT comparable with the
    full-set figure. The report has to say how many of its own false
    alarms it made that way."""
    labels = [
        _label(1, 100, confidence="certain"),
        _label(2, 200, confidence="probable"),
    ]
    report = _benchmark(
        {"engine": lambda detections: [_prediction(100), _prediction(200)]},
        labels=labels,
    )
    certain_only = report["methods"]["engine"]["certain_only"]

    assert certain_only["precision"] == 0.5
    assert certain_only["false_positives_matching_probable_labels"] == 1


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
            assert (
                result["true_positives"] + len(result["misses"])
                == result["n_ground_truth"]
            ), f"{name}/{subset}"
            assert (
                result["true_positives"] + len(result["false_positives"])
                == result["n_predicted"]
            ), f"{name}/{subset}"

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
