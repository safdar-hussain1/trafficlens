"""Tests for the end-to-end analysis pipeline (trafficlens.pipeline) and the
annotation renderer (trafficlens.annotate).

Everything here runs on a SCRIPTED detector -- a tiny object implementing the
``Detector`` protocol that returns canned ``Detection``s per frame index -- so
no model weights, no torch and no onnxruntime are involved. The video the
pipeline reads is a synthetic MJPG clip written by ``write_clip`` below: its
pixel content is irrelevant (the scripted detector ignores the frame), but it
exercises the real ``VideoSource`` open/iterate path rather than a stub.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from trafficlens import pipeline
from trafficlens.analytics.incidents import Incident, IncidentDetector
from trafficlens.analytics.speed import SpeedEstimator
from trafficlens.config import AppConfig
from trafficlens.core.gate import CrossingEvent, Gate, GateCounter
from trafficlens.detect.base import Detection
from trafficlens.io.export import validate_session_dict
from trafficlens.pipeline import SessionResult, build_session, run_session
from trafficlens.track.tracker import Tracker

ROOT = Path(__file__).resolve().parents[1]

WIDTH, HEIGHT = 640, 480

# A single horizontal gate across the middle of the frame, well inside both
# edges. With side_of_line's convention (y grows down, so smaller y is the
# LEFT/+1 side of a left-to-right gate), a vehicle moving DOWN the frame ends
# on the -1 side and is labelled "out"; one moving UP is labelled "in".
MID_GATE = {"name": "mid", "start": (0.1, 0.5), "end": (0.9, 0.5)}

# An exact pixel -> metre scaling (1 px = 0.1 m at this frame size), surveyed
# as five well-spread correspondences so the homography is over-determined and
# validates without a holdout set.
SCALE_CALIBRATION = {
    "image_points": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.5, 0.5)],
    "world_points": [
        (6.4, 4.8),
        (57.6, 4.8),
        (57.6, 43.2),
        (6.4, 43.2),
        (32.0, 24.0),
    ],
}


# --- fixtures / helpers -------------------------------------------------------


def write_clip(path: Path, frames: int, fps: float = 30.0) -> Path:
    """Write a ``frames``-long MJPG AVI at ``fps``. The content is a plain
    grey field: the scripted detector never looks at it, but VideoSource must
    genuinely decode it."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (WIDTH, HEIGHT)
    )
    assert writer.isOpened(), "OpenCV could not open an MJPG writer"
    blank = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    for _ in range(frames):
        writer.write(blank)
    writer.release()
    return path


def spin(seconds: float) -> None:
    """Burn at least ``seconds`` of wall clock, then return.

    "At least" is the important word, and it is why the timing assertions
    are ONE-SIDED. Neither this nor ``time.sleep`` can ever return early;
    both can be delayed. Measured over 60 calls per cell, 10 physical
    cores, loaded = 14 spinner processes:

        target 5 ms   unloaded            loaded (1.4x oversubscribed)
        sleep         mean 7.140 (+43%)   mean 6.752 (+35%)
                      min  5.040          min  5.075
        spin          mean 5.018 (+0.4%)  mean 5.288 (+5.8%)
                      min  5.000          min  5.000

    The ``min`` row is the point: no condition produced an undershoot, so
    a lower bound is the assertion that holds under any load, while an
    upper bound is the one that flakes.

    Busy-wait rather than ``time.sleep`` for a narrower reason than the
    first version of this docstring claimed. Sleep's overshoot is a
    roughly CONSTANT wake-up latency, not a proportional one (~2.1 ms
    here at both the 5 ms and 4 ms targets; a reviewer measured ~1.1 ms on
    other hardware), so its relative error is large AND machine-dependent
    -- a tolerance calibrated on one machine does not port. The spin's
    overshoot stays inside 6% even at 1.4x oversubscription, which is what
    lets the lower bound sit tight against the injected cost instead of
    being loosened until it stops testing anything.

    The trade, stated plainly: a SLEEPING stage would read ~0 ms if the
    pipeline's brackets ever moved to ``process_time``/``thread_time``,
    failing loudly and correctly; a spinning stage reads its full cost and
    would pass. That is real coverage the sleep gave for free and this
    gives up.
    """
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


class ScriptedDetector:
    """A ``Detector`` that returns canned detections per frame index.

    ``detect`` takes only a frame, so this keeps its own frame counter -- one
    ``detect`` call is one frame, exactly as the pipeline drives it. ``calls``
    records how many frames were actually handed to it, which is how the
    max_frames tests observe that the pipeline really stopped early.

    ``cost_s`` injects a KNOWN per-call duration (see ``spin``). That is what
    makes the timing tests able to fail: a pipeline that measured the frame
    once and split the total across three stages by some fixed ratio can
    satisfy any partition check, but it cannot make ``timings["detect"]``
    come out at an independently chosen 5 ms.
    """

    def __init__(self, script, cost_s: float = 0.0) -> None:
        self._script = script
        self._cost_s = cost_s
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._cost_s:
            spin(self._cost_s)
        index = self.calls
        self.calls += 1
        if callable(self._script):
            return self._script(index)
        return list(self._script.get(index, ()))


def box(cx: float, bottom: float, class_name: str, score: float = 0.9) -> Detection:
    """A 60x80 box whose bottom-centre anchor is ``(cx, bottom)``.

    60x80 with a 4 px/frame step keeps frame-to-frame IoU around 0.9, clear of
    the tracker's strict 0.8 association floor even on the first step, when the
    Kalman filter still predicts zero velocity.
    """
    ids = {"car": 2, "truck": 7, "bus": 5}
    return Detection(
        x1=cx - 30.0,
        y1=bottom - 80.0,
        x2=cx + 30.0,
        y2=bottom,
        score=score,
        class_id=ids[class_name],
        class_name=class_name,
    )


def two_vehicles(index: int) -> list[Detection]:
    """One car descending the frame (crosses "out") and one truck climbing it
    (crosses "in"), both inside the gate's x-span."""
    return [
        box(200.0, 120.0 + 4.0 * index, "car"),
        box(400.0, 360.0 - 4.0 * index, "truck"),
    ]


def make_config(source: Path, **overrides) -> AppConfig:
    data = {
        "source": str(source),
        "detector": {
            "model": "yolo11s.pt",
            "classes": ["car", "truck", "bus"],
            "confidence": 0.25,
            "imgsz": 640,
        },
        "gates": [dict(MID_GATE)],
    }
    data.update(overrides)
    return AppConfig.model_validate(data)


@pytest.fixture
def clip(tmp_path) -> Path:
    return write_clip(tmp_path / "clip.avi", frames=60)


# Known per-frame costs injected into two different stages so the timing
# tests have an independent yardstick per stage rather than only a
# self-consistent partition. Chosen to be the same order of magnitude as
# each other -- if one stage dominated, a mis-nested total could hide
# inside the partition check's 25% band.
DETECT_COST_MS = 5.0
TRACK_COST_MS = 4.0
TIMED_FRAMES = 15

# Floor as a fraction of the injected cost. `spin` never returns early
# (measured min == target under load and unloaded), so the true floor is
# 1.0; 0.9 leaves room for clock granularity on hardware whose
# perf_counter is coarser than this one's, and still sits an order of
# magnitude above any fabricated share of a whole-frame measurement.
COST_FLOOR = 0.9
# Ceiling as a multiple of the injected cost. Loose on purpose: the
# quantity can only overshoot, and a busy CI runner overshoots a lot.
COST_CEILING = 6.0


@pytest.fixture
def timed_session(tmp_path, monkeypatch):
    """A short session in which the detector and the tracker each cost a
    KNOWN amount of time, so every stage can be checked against a number
    the pipeline did not choose."""
    clip = write_clip(tmp_path / "timed.avi", frames=TIMED_FRAMES)

    class SlowTracker(Tracker):
        def update(self, detections, frame_index):
            spin(TRACK_COST_MS / 1000.0)
            return super().update(detections, frame_index)

    monkeypatch.setattr(pipeline, "Tracker", SlowTracker)
    detector = ScriptedDetector(two_vehicles, cost_s=DETECT_COST_MS / 1000.0)
    return run_session(make_config(clip), detector)


# --- counting: the scripted trajectory ----------------------------------------


def test_scripted_trajectory_produces_exactly_the_expected_counts(clip):
    result = run_session(make_config(clip), ScriptedDetector(two_vehicles))

    assert result.counts == {"mid": {"car": {"out": 1}, "truck": {"in": 1}}}
    assert len(result.events) == 2
    assert all(isinstance(event, CrossingEvent) for event in result.events)

    by_class = {event.class_name: event for event in result.events}
    assert by_class["car"].direction == "out"
    assert by_class["car"].signed_direction == -1
    assert by_class["truck"].direction == "in"
    assert by_class["truck"].signed_direction == 1
    for event in result.events:
        assert event.gate == "mid"
        # Both vehicles cross the gate line at y = 240 in pixel space.
        assert event.crossing_y == pytest.approx(240.0, abs=1e-6)


def test_a_vehicle_crossing_the_gate_line_outside_the_segment_is_not_counted(tmp_path):
    """The gate spans x in [64, 576]; a vehicle at x = 610 crosses the gate's
    infinite line but never the bounded segment."""
    clip = write_clip(tmp_path / "outside.avi", frames=60)
    script = lambda i: [box(610.0, 120.0 + 4.0 * i, "car")]  # noqa: E731
    result = run_session(make_config(clip), ScriptedDetector(script))
    assert result.counts == {"mid": {}}
    assert result.events == []


def test_a_tracks_first_frame_has_no_previous_anchor_and_cannot_cross(tmp_path):
    """A track appearing for the first time already BELOW the gate must not
    count. If the pipeline seeded a missing previous anchor with anything (the
    origin, the frame centre), the swept segment from that seed to this first
    anchor would cut the gate segment and fabricate a crossing."""
    clip = write_clip(tmp_path / "below.avi", frames=40)
    # First anchor is at (200, 250): just past the gate at y = 240, and the
    # segment (0, 0) -> (200, 250) does cross the gate segment.
    script = lambda i: [box(200.0, 250.0 + 4.0 * i, "car")]  # noqa: E731
    result = run_session(make_config(clip), ScriptedDetector(script))
    assert result.events == []
    assert result.counts == {"mid": {}}


def test_multiple_gates_are_each_counted_independently(tmp_path):
    clip = write_clip(tmp_path / "two_gates.avi", frames=60)
    config = make_config(
        clip,
        gates=[
            {"name": "upper", "start": (0.1, 0.375), "end": (0.9, 0.375)},
            {"name": "lower", "start": (0.1, 0.625), "end": (0.9, 0.625)},
        ],
    )
    # One car descending from y = 120 to y = 356 crosses y = 180 and y = 300.
    script = lambda i: [box(200.0, 120.0 + 4.0 * i, "car")]  # noqa: E731
    result = run_session(config, ScriptedDetector(script))
    assert result.counts == {
        "upper": {"car": {"out": 1}},
        "lower": {"car": {"out": 1}},
    }
    assert {event.gate for event in result.events} == {"upper", "lower"}


# --- max_frames ---------------------------------------------------------------


def test_max_frames_stops_the_session_early(clip):
    detector = ScriptedDetector(two_vehicles)
    result = run_session(make_config(clip), detector, max_frames=10)

    assert result.meta["frames_processed"] == 10
    assert detector.calls == 10, "the detector must not be run past max_frames"
    # The crossings happen around frame 31, so a 10-frame session sees none.
    assert result.events == []


def test_max_frames_beyond_the_clip_processes_every_frame(clip):
    detector = ScriptedDetector(two_vehicles)
    result = run_session(make_config(clip), detector, max_frames=10_000)
    assert result.meta["frames_processed"] == 60
    assert detector.calls == 60


def test_max_frames_of_zero_processes_nothing(clip):
    detector = ScriptedDetector(two_vehicles)
    result = run_session(make_config(clip), detector, max_frames=0)
    assert result.meta["frames_processed"] == 0
    assert detector.calls == 0
    assert result.events == []


# --- timings ------------------------------------------------------------------


def test_timings_carry_one_entry_per_stage(clip):
    result = run_session(make_config(clip), ScriptedDetector(two_vehicles))

    for stage in pipeline.STAGES:
        assert stage in result.timings, stage
        entry = result.timings[stage]
        assert set(entry) == {"mean_ms", "p95_ms", "n"}
        assert entry["n"] == result.meta["frames_processed"]
        assert entry["mean_ms"] > 0.0
        assert entry["p95_ms"] >= entry["mean_ms"] * 0.5


def test_each_stage_reports_its_own_independently_known_cost(timed_session):
    """The load-bearing timing test.

    A partition check alone cannot prove the stages were measured
    separately: ANY three numbers summing to the total satisfy it, so a
    pipeline that timed the whole frame body once and split the result by a
    fixed ratio (detect 2%, track 94%, analytics 4%) passes it with a ratio
    of exactly 1.0000. What such a pipeline cannot do is make two different
    stages come out at two independently chosen numbers.

    So the detector burns a known DETECT_COST_MS and the tracker a known
    TRACK_COST_MS, and each stage is asserted against its own injected cost.
    The only way to satisfy both is to have actually bracketed each stage.

    Assertion shape: ONE-SIDED, floor-first. An injected cost can only ever
    be overshot -- neither `spin` nor `time.sleep` returns early, and the
    `track` bracket legitimately contains the real `Tracker.update` work
    (~0.25 ms) on top of its injected 4.0 ms. A two-sided tolerance
    therefore spends most of its budget before the test even starts, and
    any scheduling delay pushes it out: the earlier `approx(rel=0.3)`
    version failed 7 times in 25 runs on `track` under CPU contention.
    All of the anti-fabrication power lives in the FLOOR anyway -- a split
    of a single whole-frame measurement cannot reach an independently
    chosen floor -- so the ceiling is only a loose sanity bound.
    """
    result = timed_session
    total = result.timings[pipeline.TOTAL]["mean_ms"]

    for stage, injected in (("detect", DETECT_COST_MS), ("track", TRACK_COST_MS)):
        measured = result.timings[stage]["mean_ms"]
        # The floor: this is the assertion with the teeth.
        assert measured >= injected * COST_FLOOR, (
            f"the {stage} bracket reports {measured:.3f} ms, below the "
            f"{injected} ms this stage was made to cost -- it cannot be "
            f"measuring that stage's own work"
        )
        # Exact structural invariant, not a tolerance: each stage's bracket
        # nests inside the frame bracket, so no stage mean can exceed the
        # frame mean. Load-independent, so it never flakes.
        assert measured <= total, (
            f"{stage} ({measured:.3f} ms) exceeds the whole frame "
            f"({total:.3f} ms), so it is counting work outside itself"
        )
        # Loose ceiling: catches an order-of-magnitude misattribution that
        # the partition check somehow let through. Deliberately generous --
        # under 1.4x CPU oversubscription a 5 ms spin was measured at up to
        # 11.8 ms, so anything tighter would flake on a busy CI runner.
        assert measured <= injected * COST_CEILING

    # Analytics has no injected cost, so it must stay far below the stages
    # that do.
    assert result.timings["analytics"]["mean_ms"] < TRACK_COST_MS / 2.0


def test_stage_means_sum_to_the_measured_total_frame_time(timed_session):
    """Each stage must be measured by its OWN perf_counter bracket, and the
    three together must account for the frame.

    This is the over-counting half of the guarantee (a stage that contains
    another's work makes the sum exceed the total) and the wrong-nesting
    half (a total that starts after a stage the parts still include makes
    the sum exceed it too). It runs on the same injected-cost profile as the
    test above deliberately: with detect and track both large, neither
    mistake can hide inside the 25% band the way it could when one stage was
    94% of the frame.
    """
    result = timed_session

    stage_sum = sum(result.timings[stage]["mean_ms"] for stage in pipeline.STAGES)
    total = result.timings[pipeline.TOTAL]["mean_ms"]
    assert total > 0.0
    assert abs(stage_sum - total) <= 0.25 * total, (
        f"stage means sum to {stage_sum:.4f} ms but the measured frame total "
        f"is {total:.4f} ms"
    )


def test_timings_of_an_empty_session_are_reported_as_zero_samples(clip):
    result = run_session(make_config(clip), ScriptedDetector({}), max_frames=0)
    for stage in (*pipeline.STAGES, pipeline.TOTAL):
        assert result.timings[stage] == {"mean_ms": 0.0, "p95_ms": 0.0, "n": 0}


# --- meta ---------------------------------------------------------------------


def test_meta_describes_the_session(clip):
    config = make_config(clip)
    result = run_session(config, ScriptedDetector(two_vehicles))

    meta = result.meta
    assert meta["source"] == str(clip)
    assert meta["fps"] == pytest.approx(30.0)
    assert meta["width"] == WIDTH
    assert meta["height"] == HEIGHT
    assert meta["frames_processed"] == 60
    assert meta["model"] == "yolo11s.pt"
    assert meta["classes"] == ["car", "truck", "bus"]
    assert meta["calibrated"] is False


def test_meta_reports_calibrated_when_a_calibration_block_is_present(clip):
    config = make_config(clip, calibration=dict(SCALE_CALIBRATION))
    result = run_session(config, ScriptedDetector(two_vehicles))
    assert result.meta["calibrated"] is True


# --- speeds -------------------------------------------------------------------


def test_an_uncalibrated_session_reports_no_speed_for_any_track(clip):
    result = run_session(make_config(clip), ScriptedDetector(two_vehicles))
    assert result.speeds  # tracks were seen
    assert set(result.speeds.values()) == {None}
    assert all(event.speed_kmh is None for event in result.events)


def test_a_calibrated_session_recovers_the_scripted_speed(clip):
    """4 px/frame at 30 fps is 120 px/s; at 1 px = 0.1 m that is 12 m/s =
    43.2 km/h."""
    config = make_config(clip, calibration=dict(SCALE_CALIBRATION))
    result = run_session(config, ScriptedDetector(two_vehicles))

    assert result.speeds
    for speed in result.speeds.values():
        assert speed == pytest.approx(43.2, abs=0.5)
    for event in result.events:
        assert event.speed_kmh == pytest.approx(43.2, abs=0.5)


def test_a_speed_limit_marks_the_crossing_as_a_violation(clip):
    config = make_config(
        clip,
        calibration=dict(SCALE_CALIBRATION),
        speed={"unit": "kmh", "limit": 20.0},
    )
    result = run_session(config, ScriptedDetector(two_vehicles))
    assert result.events
    assert all(event.is_violation for event in result.events)


def test_a_speed_under_the_limit_is_not_a_violation(clip):
    config = make_config(
        clip,
        calibration=dict(SCALE_CALIBRATION),
        speed={"unit": "kmh", "limit": 90.0},
    )
    result = run_session(config, ScriptedDetector(two_vehicles))
    assert result.events
    assert not any(event.is_violation for event in result.events)


def test_violation_snapshots_are_written_when_a_directory_is_given(clip, tmp_path):
    config = make_config(
        clip,
        calibration=dict(SCALE_CALIBRATION),
        speed={"unit": "kmh", "limit": 20.0},
    )
    snapshots = tmp_path / "violations"
    result = run_session(
        config, ScriptedDetector(two_vehicles), snapshot_dir=snapshots
    )
    written = sorted(p.name for p in snapshots.glob("*.jpg"))
    assert len(written) == len(result.events) == 2
    assert all(name.startswith("violation_mid_track") for name in written)


def test_no_snapshots_are_written_without_a_violation(clip, tmp_path):
    config = make_config(clip, calibration=dict(SCALE_CALIBRATION))
    snapshots = tmp_path / "violations"
    run_session(config, ScriptedDetector(two_vehicles), snapshot_dir=snapshots)
    assert not snapshots.exists() or list(snapshots.glob("*.jpg")) == []


# --- incidents ----------------------------------------------------------------


def test_a_stationary_vehicle_fires_one_stopped_incident(tmp_path):
    """At 10 fps, 130 frames is 13 s -- past the 10 s stopped threshold, with
    room to prove the incident fires exactly once, not once per frame."""
    clip = write_clip(tmp_path / "stopped.avi", frames=130, fps=10.0)
    config = make_config(clip, calibration=dict(SCALE_CALIBRATION))
    script = lambda i: [box(200.0, 300.0, "car")]  # noqa: E731
    result = run_session(config, ScriptedDetector(script))

    stopped = [i for i in result.incidents if i.kind == "stopped"]
    assert len(stopped) == 1
    assert isinstance(stopped[0], Incident)
    assert stopped[0].class_name == "car"
    assert stopped[0].timestamp == pytest.approx(10.0, abs=0.6)


def test_an_uncalibrated_stationary_vehicle_is_unknown_not_stopped(tmp_path):
    clip = write_clip(tmp_path / "stopped_uncal.avi", frames=130, fps=10.0)
    script = lambda i: [box(200.0, 300.0, "car")]  # noqa: E731
    result = run_session(make_config(clip), ScriptedDetector(script))
    assert result.incidents == []


def test_a_crossing_against_the_expected_direction_is_a_wrong_way_incident(tmp_path):
    clip = write_clip(tmp_path / "wrongway.avi", frames=60)
    config = make_config(
        clip,
        gates=[{**MID_GATE, "expected_direction": "in"}],
    )
    # The descending car crosses as "out", against the expected "in".
    script = lambda i: [box(200.0, 120.0 + 4.0 * i, "car")]  # noqa: E731
    result = run_session(config, ScriptedDetector(script))

    wrong = [i for i in result.incidents if i.kind == "wrong_way"]
    assert len(wrong) == 1
    assert wrong[0].class_name == "car"
    assert "mid" in wrong[0].detail


def test_a_crossing_with_the_expected_direction_is_not_an_incident(tmp_path):
    clip = write_clip(tmp_path / "rightway.avi", frames=60)
    config = make_config(clip, gates=[{**MID_GATE, "expected_direction": "out"}])
    script = lambda i: [box(200.0, 120.0 + 4.0 * i, "car")]  # noqa: E731
    result = run_session(config, ScriptedDetector(script))
    assert result.incidents == []


# --- per-track reaping --------------------------------------------------------


def test_the_reaper_holds_a_track_for_exactly_max_age_frames():
    reaper = pipeline.TrackReaper(max_age=30)
    reaper.saw(7, 100)
    # A confirmed track survives a gap of up to exactly max_age frames and may
    # still re-associate, so it must not be reaped inside that window.
    assert reaper.reap(130) == []
    assert reaper.reap(131) == [7]
    assert reaper.reap(200) == []  # already gone, never reaped twice


def test_the_reaper_forgets_a_track_that_reappears():
    reaper = pipeline.TrackReaper(max_age=5)
    reaper.saw(1, 0)
    assert reaper.reap(5) == []
    reaper.saw(1, 5)
    assert reaper.reap(10) == []
    assert reaper.reap(11) == [1]


# A convoy of vehicles crossing one after another, each vanishing well
# before the next appears -- so at most one track is ever alive, and any
# per-track state that is not forgotten grows to CONVOY_VEHICLES.
CONVOY_VEHICLES = 20
CONVOY_VISIBLE = 40  # frames each vehicle is on screen
CONVOY_GAP = 40  # frames of empty footage between vehicles (> max_age)


def convoy_script(index: int) -> list[Detection]:
    slot, offset = divmod(index, CONVOY_VISIBLE + CONVOY_GAP)
    if offset >= CONVOY_VISIBLE or slot >= CONVOY_VEHICLES:
        return []
    # Descends from y = 160 to y = 316, crossing the gate at y = 240.
    return [box(200.0, 160.0 + 4.0 * offset, "car")]


def write_convoy_clip(path: Path) -> Path:
    return write_clip(
        path,
        frames=CONVOY_VEHICLES * (CONVOY_VISIBLE + CONVOY_GAP),
        fps=30.0,
    )


def test_dead_tracks_are_reaped_so_gate_state_stays_bounded(tmp_path, monkeypatch):
    """A long session in which 20 vehicles cross one after another, each
    vanishing before the next appears. The GateCounter remembers every track it
    has counted, so without reaping its internal sets would grow to 20; with
    reaping at most one track's state is ever live at once."""
    vehicles = CONVOY_VEHICLES
    clip = write_convoy_clip(tmp_path / "long.avi")
    script = convoy_script

    created: list["SpyCounter"] = []

    class SpyCounter(GateCounter):
        """Records the HIGH-WATER MARK of the counter's per-track state.

        Measuring after the session would prove nothing: the session drains
        its remaining tracks on the way out, so even a pipeline that never
        reaped anything would finish with empty sets. The leak, if there is
        one, is only visible while the session is running.
        """

        def __init__(self, gate: Gate) -> None:
            super().__init__(gate)
            self.peak = 0
            created.append(self)

        def update(self, *args, **kwargs):
            event = super().update(*args, **kwargs)
            self.peak = max(
                self.peak,
                len(self._counted),
                len(self._last_side),
                len(self._last_off_line_point),
            )
            return event

    monkeypatch.setattr(pipeline, "GateCounter", SpyCounter)
    result = run_session(make_config(clip), ScriptedDetector(script))

    assert len(created) == 1
    counter = created[0]
    assert counter.total() == vehicles, "every vehicle must actually be counted"
    # Only one vehicle is ever on screen at a time, so the counter should
    # never have been remembering more than one track. Without reaping this
    # peak would climb to 20 -- one permanent record per vehicle ever seen.
    assert counter.peak <= 2, f"per-track state peaked at {counter.peak}"
    # And the session drains what is left, so nothing outlives it.
    assert counter._counted == set()
    assert counter._last_side == {}
    assert counter._last_off_line_point == {}
    # The counts themselves are unaffected by all that forgetting.
    assert len(result.counts["mid"]["car"]) == 1
    assert result.counts["mid"]["car"]["out"] == vehicles


def test_dead_tracks_are_reaped_so_speed_estimator_state_stays_bounded(
    tmp_path, monkeypatch
):
    """The gate counter is not the only per-track state holder, and guarding
    only it left the other two ``forget`` calls unprotected: deleting
    ``speed_estimator.forget(dead_id)`` used to pass every test. The speed
    estimator keeps a sample deque per track, so this pins it with the same
    high-water-mark technique.

    The session must be CALIBRATED: an uncalibrated estimator short-circuits
    in ``observe`` and buffers nothing, so it would have no state to leak and
    the test would have no teeth.
    """
    clip = write_convoy_clip(tmp_path / "long_speed.avi")
    created: list["SpyEstimator"] = []

    class SpyEstimator(SpeedEstimator):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.peak = 0
            created.append(self)

        def observe(self, *args, **kwargs):
            result = super().observe(*args, **kwargs)
            self.peak = max(self.peak, len(self._tracks))
            return result

    monkeypatch.setattr(pipeline, "SpeedEstimator", SpyEstimator)
    config = make_config(clip, calibration=dict(SCALE_CALIBRATION))
    result = run_session(config, ScriptedDetector(convoy_script))

    assert len(created) == 1
    estimator = created[0]
    assert result.counts["mid"]["car"]["out"] == CONVOY_VEHICLES
    # Sanity: the estimator really was buffering, so a peak of 1 means
    # "forgotten as they died", not "never used".
    assert estimator.peak >= 1
    assert estimator.peak <= 2, (
        f"per-track speed buffers peaked at {estimator.peak}; without "
        f"forgetting this climbs to {CONVOY_VEHICLES}"
    )
    assert estimator._tracks == {}


def test_dead_tracks_are_reaped_so_incident_state_stays_bounded(
    tmp_path, monkeypatch
):
    """Same guarantee for the incident detector, whose ``_stops`` map gains an
    entry for every track it is ever shown. Deleting
    ``incident_detector.forget(dead_id)`` used to pass every test."""
    clip = write_convoy_clip(tmp_path / "long_incident.avi")
    created: list["SpyIncidents"] = []

    class SpyIncidents(IncidentDetector):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.peak = 0
            created.append(self)

        def update(self, *args, **kwargs):
            incident = super().update(*args, **kwargs)
            self.peak = max(
                self.peak, len(self._stops), len(self._wrong_way_fired)
            )
            return incident

    monkeypatch.setattr(pipeline, "IncidentDetector", SpyIncidents)
    # expected_direction makes every crossing a wrong_way, so the
    # (track, gate) memo is exercised alongside the stopped-vehicle state.
    config = make_config(clip, gates=[{**MID_GATE, "expected_direction": "in"}])
    result = run_session(config, ScriptedDetector(convoy_script))

    assert len(created) == 1
    detector = created[0]
    assert len(result.incidents) == CONVOY_VEHICLES
    assert detector.peak >= 1
    assert detector.peak <= 2, (
        f"per-track incident state peaked at {detector.peak}; without "
        f"forgetting this climbs to {CONVOY_VEHICLES}"
    )
    assert detector._stops == {}
    assert detector._wrong_way_fired == set()


def test_speeds_keep_the_last_known_value_after_a_track_is_reaped(tmp_path):
    """Reaping frees the estimator's buffers but must not erase the session's
    reported speed for a track that has already been seen."""
    clip = write_clip(tmp_path / "reap_speed.avi", frames=140, fps=30.0)

    def script(index: int) -> list[Detection]:
        if index >= 40:
            return []  # the vehicle leaves and never returns
        return [box(200.0, 120.0 + 4.0 * index, "car")]

    config = make_config(clip, calibration=dict(SCALE_CALIBRATION))
    result = run_session(config, ScriptedDetector(script))
    assert set(result.speeds) == {1}
    assert result.speeds[1] == pytest.approx(43.2, abs=0.5)


# --- progress, frames and session export --------------------------------------


def test_progress_is_reported_once_per_processed_frame(clip):
    seen = []
    run_session(
        make_config(clip),
        ScriptedDetector(two_vehicles),
        progress=lambda processed, total: seen.append((processed, total)),
        max_frames=5,
    )
    assert seen == [(1, 60), (2, 60), (3, 60), (4, 60), (5, 60)]


def test_frames_are_not_recorded_unless_asked(clip):
    result = run_session(make_config(clip), ScriptedDetector(two_vehicles))
    assert result.frames == []


def test_recorded_frames_build_a_valid_schema_1_session(clip):
    config = make_config(clip, calibration=dict(SCALE_CALIBRATION))
    result = run_session(
        config, ScriptedDetector(two_vehicles), record_frames=True
    )

    assert len(result.frames) == 60
    session = build_session(result)
    validate_session_dict(session)  # raises if the contract is broken

    assert session["schema"] == 1
    assert session["clip"] == clip.name
    assert session["width"] == WIDTH and session["height"] == HEIGHT
    assert [gate["name"] for gate in session["gates"]] == ["mid"]
    assert session["gates"][0]["start"] == [64.0, 240.0]
    assert len(session["events"]) == 2
    # The first frames precede track confirmation, so they carry no tracks;
    # by mid-clip both vehicles are tracked with a known speed.
    mid = session["frames"][30]
    assert len(mid["tracks"]) == 2
    assert {t["class_name"] for t in mid["tracks"]} == {"car", "truck"}
    assert all(t["speed_kmh"] is not None for t in mid["tracks"])
    assert all(len(t["box"]) == 4 for t in mid["tracks"])


def test_a_session_result_is_the_documented_dataclass(clip):
    result = run_session(make_config(clip), ScriptedDetector(two_vehicles))
    assert isinstance(result, SessionResult)
    for field in ("counts", "events", "incidents", "speeds", "timings", "meta"):
        assert hasattr(result, field), field


# --- annotation ---------------------------------------------------------------


def test_draw_frame_returns_a_new_array_of_the_same_shape():
    from trafficlens import annotate
    from trafficlens.track.tracker import Track

    frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    gate = Gate.from_normalized("mid", (0.1, 0.5), (0.9, 0.5), WIDTH, HEIGHT)
    counter = GateCounter(gate)
    counter.totals = {"car": {"out": 3}}
    track = Track(
        track_id=1,
        class_name="car",
        box=(170.0, 160.0, 230.0, 240.0),
        score=0.9,
        age=5,
        hits=5,
        time_since_update=0,
        state="confirmed",
    )

    out = annotate.draw_frame(frame, [track], [gate], {"mid": counter}, {1: 51.5})
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert out is not frame
    assert np.array_equal(frame, np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8))
    assert not np.array_equal(out, frame), "something must have been drawn"


def test_draw_frame_marks_an_unknown_speed_explicitly():
    """An uncalibrated session must SAY it has no speed, not silently omit the
    label (which would read as "not measured yet" rather than "never")."""
    from trafficlens import annotate
    from trafficlens.track.tracker import Track

    frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    gate = Gate.from_normalized("mid", (0.1, 0.5), (0.9, 0.5), WIDTH, HEIGHT)
    track = Track(
        track_id=1,
        class_name="car",
        box=(170.0, 160.0, 230.0, 240.0),
        score=0.9,
        age=5,
        hits=5,
        time_since_update=0,
        state="confirmed",
    )
    labelled = annotate.draw_frame(
        frame, [track], [gate], {"mid": GateCounter(gate)}, {1: None}
    )
    known = annotate.draw_frame(
        frame, [track], [gate], {"mid": GateCounter(gate)}, {1: 51.5}
    )
    assert annotate.speed_label(None) == annotate.NO_SPEED_LABEL
    assert annotate.speed_label(51.5) == "51.5 km/h"
    assert not np.array_equal(labelled, known)


def test_draw_frame_accepts_incidents_and_highlights_them():
    from trafficlens import annotate
    from trafficlens.track.tracker import Track

    frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    gate = Gate.from_normalized("mid", (0.1, 0.5), (0.9, 0.5), WIDTH, HEIGHT)
    track = Track(
        track_id=1,
        class_name="car",
        box=(170.0, 160.0, 230.0, 240.0),
        score=0.9,
        age=5,
        hits=5,
        time_since_update=0,
        state="confirmed",
    )
    incident = Incident(
        kind="stopped",
        track_id=1,
        class_name="car",
        frame_index=5,
        timestamp=10.0,
        detail="stationary for 10.0 s",
    )
    plain = annotate.draw_frame(frame, [track], [gate], {}, {1: None})
    flagged = annotate.draw_frame(
        frame, [track], [gate], {}, {1: None}, incidents=[incident]
    )
    assert not np.array_equal(plain, flagged)


# --- dependency layering ------------------------------------------------------


def _leaked_heavy_modules(module: str) -> list[str]:
    code = (
        "import sys\n"
        f"import {module}\n"
        "leaked = sorted(m for m in ('torch', 'ultralytics', 'onnxruntime') "
        "if m in sys.modules)\n"
        "print(','.join(leaked))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    return out.split(",") if out else []


def test_importing_the_pipeline_never_pulls_in_a_detector_backend():
    """The pipeline takes an INJECTED detector; it must never import torch,
    ultralytics or onnxruntime itself."""
    assert _leaked_heavy_modules("trafficlens.pipeline") == []


def test_importing_the_annotator_never_pulls_in_a_detector_backend():
    assert _leaked_heavy_modules("trafficlens.annotate") == []
