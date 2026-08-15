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
from pathlib import Path

import cv2
import numpy as np
import pytest

from trafficlens import pipeline
from trafficlens.analytics.incidents import Incident
from trafficlens.config import AppConfig
from trafficlens.core.gate import CrossingEvent, Gate, GateCounter
from trafficlens.detect.base import Detection
from trafficlens.io.export import validate_session_dict
from trafficlens.pipeline import SessionResult, build_session, run_session

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


class ScriptedDetector:
    """A ``Detector`` that returns canned detections per frame index.

    ``detect`` takes only a frame, so this keeps its own frame counter -- one
    ``detect`` call is one frame, exactly as the pipeline drives it. ``calls``
    records how many frames were actually handed to it, which is how the
    max_frames tests observe that the pipeline really stopped early.
    """

    def __init__(self, script) -> None:
        self._script = script
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
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


def test_stage_means_sum_to_the_measured_total_frame_time(clip):
    """Each stage must be measured by its OWN perf_counter bracket. A pipeline
    that timed one region running everything and attributed the total to a
    single stage would make this sum roughly N times the real frame time; this
    asserts the three stage means genuinely partition the frame."""
    result = run_session(make_config(clip), ScriptedDetector(two_vehicles))

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


def test_dead_tracks_are_reaped_so_gate_state_stays_bounded(tmp_path, monkeypatch):
    """A long session in which 20 vehicles cross one after another, each
    vanishing before the next appears. The GateCounter remembers every track it
    has counted, so without reaping its internal sets would grow to 20; with
    reaping at most one track's state is ever live at once."""
    per_vehicle = 40  # frames each vehicle is visible for
    gap = 40  # frames of empty video between vehicles
    vehicles = 20
    frames = vehicles * (per_vehicle + gap)
    clip = write_clip(tmp_path / "long.avi", frames=frames, fps=30.0)

    def script(index: int) -> list[Detection]:
        slot, offset = divmod(index, per_vehicle + gap)
        if offset >= per_vehicle or slot >= vehicles:
            return []
        # Descends from y = 160 to y = 316, crossing the gate at y = 240.
        return [box(200.0, 160.0 + 4.0 * offset, "car")]

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
