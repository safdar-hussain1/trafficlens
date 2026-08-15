"""Tests for incident detection (stopped vehicles, wrong-way crossings)
and the speed-limit violation policy with snapshots."""

import pytest

from trafficlens.core.constants import (
    INCIDENT_MIN_STOPPED_S,
    INCIDENT_STOPPED_SPEED_KMH,
)
from trafficlens.core.gate import CrossingEvent, Gate, GateCounter
from trafficlens.analytics.incidents import Incident, IncidentDetector
from trafficlens.analytics.violations import ViolationPolicy


def make_gate(expected: str | None = "in", name: str = "main") -> Gate:
    """A vertical gate from (0, 0) to (0, 10): with side_of_line's
    orientation, ending at x > 0 is the +1 ("in") side and ending at
    x < 0 is the -1 ("out") side."""
    return Gate(
        name,
        (0.0, 0.0),
        (0.0, 10.0),
        label_positive="in",
        label_negative="out",
        expected_direction=expected,
    )


def make_event(
    track_id: int = 1,
    gate: str = "main",
    direction: str = "out",
    signed_direction: int = -1,
    frame_index: int = 10,
    timestamp: float = 1.0,
    speed_kmh: float | None = 50.0,
    class_name: str = "car",
    is_violation: bool = False,
) -> CrossingEvent:
    return CrossingEvent(
        track_id=track_id,
        class_name=class_name,
        gate=gate,
        direction=direction,
        signed_direction=signed_direction,
        frame_index=frame_index,
        timestamp=timestamp,
        crossing_x=0.0,
        crossing_y=5.0,
        speed_kmh=speed_kmh,
        is_violation=is_violation,
    )


def feed(det, speeds_and_times, track_id=7, class_name="car"):
    """Run det.update over (speed, timestamp, frame_index) triples and
    return the incidents that fired."""
    out = []
    for speed, t, i in speeds_and_times:
        inc = det.update(track_id, class_name, speed, t, i)
        if inc is not None:
            out.append(inc)
    return out


# --- stopped-vehicle detection ------------------------------------------------


def test_sustained_stop_fires_exactly_one_incident():
    det = IncidentDetector(min_stopped_s=2.0, stopped_speed_kmh=3.0)
    # 0.5 km/h at 10 fps for 4 seconds: well past the 2 s threshold, and
    # 20 further sub-threshold frames after the moment it first fires.
    incidents = feed(det, [(0.5, i * 0.1, i) for i in range(41)])
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.kind == "stopped"
    assert inc.track_id == 7
    assert inc.class_name == "car"
    # It fires at the first frame where the continuous run reaches 2.0 s.
    assert inc.timestamp == pytest.approx(2.0)
    assert inc.frame_index == 20
    assert "2.0" in inc.detail


def test_brief_stop_below_min_duration_fires_nothing():
    det = IncidentDetector(min_stopped_s=2.0, stopped_speed_kmh=3.0)
    # Slow for 1 s, then moving again: never fires.
    samples = [(0.5, i * 0.1, i) for i in range(11)]
    samples += [(30.0, 1.1 + i * 0.1, 11 + i) for i in range(30)]
    assert feed(det, samples) == []


def test_stopped_never_fires_without_calibrated_speed():
    # THE pinned policy test: speed_kmh=None means uncalibrated, and a
    # stopped-vehicle incident requires calibrated speed -- no amount of
    # None frames, over any duration, may ever fire one.
    det = IncidentDetector(min_stopped_s=2.0, stopped_speed_kmh=3.0)
    assert feed(det, [(None, i * 0.1, i) for i in range(101)]) == []


def test_none_speed_neither_fires_nor_accumulates_stopped_time():
    det = IncidentDetector(min_stopped_s=2.0, stopped_speed_kmh=3.0)
    incidents = []
    # Sub-threshold from t=0.0 to t=1.0 ...
    samples = [(0.5, i * 0.1, i) for i in range(11)]
    # ... then an uncalibrated gap t=1.1..1.5 (unknown, not stopped) ...
    samples += [(None, 1.1 + i * 0.1, 11 + i) for i in range(5)]
    # ... then sub-threshold again from t=1.6 onward. The continuous run
    # restarts at 1.6, so nothing may fire before t=3.6 even though the
    # track first read sub-threshold at t=0.0.
    samples += [(0.5, 1.6 + i * 0.1, 16 + i) for i in range(21)]
    for speed, t, i in samples:
        inc = det.update(7, "car", speed, t, i)
        if inc is not None:
            incidents.append((inc, t))
    assert len(incidents) == 1
    inc, fired_at = incidents[0]
    assert fired_at == pytest.approx(3.6)


def test_none_gap_after_firing_does_not_rearm():
    # Losing calibration is not evidence the vehicle moved: a track that
    # fires, goes through a None-speed gap, and is STILL sub-threshold
    # afterwards for another full min_stopped_s is one continuing stop
    # episode -- exactly one incident across the whole sequence. Only a
    # genuine at-or-above-threshold speed re-arms.
    det = IncidentDetector(min_stopped_s=2.0, stopped_speed_kmh=3.0)
    samples = [(0.5, i * 0.1, i) for i in range(21)]  # t=0.0..2.0, fires
    samples += [(None, i * 0.1, i) for i in range(21, 26)]  # uncalibrated gap
    # Sub-threshold again from t=2.6 to t=5.0: a fresh continuous run far
    # longer than min_stopped_s, which must NOT fire a second time.
    samples += [(0.5, i * 0.1, i) for i in range(26, 51)]
    incidents = feed(det, samples)
    assert len(incidents) == 1
    assert incidents[0].timestamp == pytest.approx(2.0)


def test_rearm_stop_move_stop_fires_twice():
    # A car that stops twice fires twice: the detector re-arms only after
    # the track moves above the threshold again.
    det = IncidentDetector(min_stopped_s=2.0, stopped_speed_kmh=3.0)
    samples = [(0.5, i * 0.1, i) for i in range(21)]  # t=0.0..2.0, fires
    samples += [(30.0, i * 0.1, i) for i in range(21, 26)]  # moving
    samples += [(0.5, i * 0.1, i) for i in range(26, 48)]  # fires again
    incidents = feed(det, samples)
    assert len(incidents) == 2
    assert [inc.kind for inc in incidents] == ["stopped", "stopped"]
    # The second run starts at t=2.6; it reaches 2.0 continuous seconds at
    # t=4.6 (within one 0.1 s frame step of float accumulation).
    assert incidents[1].timestamp == pytest.approx(4.6, abs=0.11)


def test_tracks_are_independent():
    det = IncidentDetector(min_stopped_s=2.0, stopped_speed_kmh=3.0)
    for i in range(21):
        stopped = det.update(1, "car", 0.5, i * 0.1, i)
        moving = det.update(2, "truck", 40.0, i * 0.1, i)
        assert moving is None
        if i < 20:
            assert stopped is None
    # Only track 1 ever fires, and its identity is its own.
    inc = det.update(1, "car", 0.5, 2.1, 21)
    assert inc is None  # already fired at i=20 above... verify below
    # Re-run cleanly to assert the single firing carried track 1's fields.
    det2 = IncidentDetector(min_stopped_s=2.0, stopped_speed_kmh=3.0)
    incidents = feed(det2, [(0.5, i * 0.1, i) for i in range(21)], track_id=1)
    assert len(incidents) == 1
    assert incidents[0].track_id == 1


def test_forget_clears_stopped_state():
    det = IncidentDetector(min_stopped_s=2.0, stopped_speed_kmh=3.0)
    assert feed(det, [(0.5, i * 0.1, i) for i in range(20)]) == []  # 1.9 s
    det.forget(7)
    # A fresh run starts at t=2.0; nothing fires before t=4.0.
    samples = [(0.5, 2.0 + i * 0.1, 20 + i) for i in range(21)]
    incidents = feed(det, samples)
    assert len(incidents) == 1
    assert incidents[0].timestamp == pytest.approx(4.0)


def test_detector_defaults_come_from_constants():
    det = IncidentDetector()
    assert det.min_stopped_s == INCIDENT_MIN_STOPPED_S
    assert det.stopped_speed_kmh == INCIDENT_STOPPED_SPEED_KMH


def test_detector_rejects_non_positive_tunables():
    with pytest.raises(ValueError):
        IncidentDetector(min_stopped_s=0.0)
    with pytest.raises(ValueError):
        IncidentDetector(stopped_speed_kmh=-1.0)


# --- wrong-way detection ------------------------------------------------------


def test_wrong_way_fires_from_a_real_gate_counter_event():
    # Drive the real GateCounter across the real Gate API end to end: a
    # track moving left-to-right across the (0,0)->(0,10) gate ends on
    # the +1 side, labelled "in", against expected_direction="out".
    gate = make_gate(expected="out")
    counter = GateCounter(gate)
    event = counter.update(3, "car", (-5.0, 5.0), (5.0, 5.0), 12, 0.4)
    assert event is not None
    assert event.direction == "in"

    det = IncidentDetector()
    inc = det.note_crossing(event, gate)
    assert isinstance(inc, Incident)
    assert inc.kind == "wrong_way"
    assert inc.track_id == 3
    assert inc.class_name == "car"
    assert inc.frame_index == 12
    assert inc.timestamp == 0.4
    assert "out" in inc.detail and "in" in inc.detail


def test_expected_direction_crossing_is_not_wrong_way():
    gate = make_gate(expected="out")
    det = IncidentDetector()
    assert det.note_crossing(make_event(direction="out"), gate) is None


def test_gate_without_expected_direction_never_fires():
    gate = make_gate(expected=None)
    det = IncidentDetector()
    assert det.note_crossing(make_event(direction="out"), gate) is None
    assert det.note_crossing(make_event(direction="in", signed_direction=1), gate) is None


def test_wrong_way_fires_once_per_track_and_gate():
    gate = make_gate(expected="in")
    det = IncidentDetector()
    event = make_event(track_id=5, direction="out")
    assert det.note_crossing(event, gate) is not None
    # A re-notified identical crossing must not duplicate.
    assert det.note_crossing(event, gate) is None
    # A different gate is separate state for the same track.
    other = make_gate(expected="in", name="side")
    assert det.note_crossing(make_event(track_id=5, gate="side", direction="out"), other) is not None
    # A different track is separate state on the same gate.
    assert det.note_crossing(make_event(track_id=6, direction="out"), gate) is not None


def test_forget_clears_wrong_way_state():
    gate = make_gate(expected="in")
    det = IncidentDetector()
    assert det.note_crossing(make_event(track_id=5, direction="out"), gate) is not None
    det.forget(5)
    assert det.note_crossing(make_event(track_id=5, direction="out"), gate) is not None


def test_misconfigured_expected_direction_fails_fast():
    # expected_direction must be one of the gate's two labels; anything
    # else is a configuration bug, not a wrong-way vehicle.
    gate = make_gate(expected="north")
    det = IncidentDetector()
    with pytest.raises(ValueError):
        det.note_crossing(make_event(direction="out"), gate)


# --- speed-limit violations ---------------------------------------------------


def test_check_is_strictly_over_limit():
    policy = ViolationPolicy(limit_kmh=50.0)
    assert policy.check(make_event(speed_kmh=50.1)) is True
    assert policy.check(make_event(speed_kmh=50.0)) is False  # at the limit
    assert policy.check(make_event(speed_kmh=49.9)) is False
    assert policy.check(make_event(speed_kmh=None)) is False  # uncalibrated


def test_check_with_no_limit_configured_is_always_false():
    # Limits are user-set; the product never assumes a posted limit.
    policy = ViolationPolicy(limit_kmh=None)
    assert policy.check(make_event(speed_kmh=200.0)) is False


def test_snapshot_path_is_deterministic_and_does_no_io(tmp_path):
    policy = ViolationPolicy(limit_kmh=50.0)
    out_dir = tmp_path / "snapshots"  # deliberately never created
    event = make_event(track_id=4, frame_index=120, speed_kmh=80.0)
    first = policy.snapshot_path(event, out_dir)
    second = policy.snapshot_path(event, out_dir)
    assert first == second
    assert first.parent == out_dir
    assert first.suffix == ".jpg"
    assert not out_dir.exists()  # pure computation, no filesystem writes


def test_snapshot_path_is_collision_free_across_gate_track_frame():
    policy = ViolationPolicy(limit_kmh=50.0)
    from pathlib import Path

    out = Path("snaps")
    paths = {
        policy.snapshot_path(make_event(track_id=1, frame_index=10), out),
        policy.snapshot_path(make_event(track_id=2, frame_index=10), out),
        policy.snapshot_path(make_event(track_id=1, frame_index=11), out),
        policy.snapshot_path(make_event(track_id=1, frame_index=10, gate="side"), out),
    }
    assert len(paths) == 4


def test_snapshot_path_sanitises_gate_name(tmp_path):
    policy = ViolationPolicy(limit_kmh=50.0)
    event = make_event(gate="M40 J3 / north exit")
    path = policy.snapshot_path(event, tmp_path)
    assert path.parent == tmp_path  # the slash must not create a subdirectory
    assert " " not in path.name
    assert "/" not in path.name
    assert path.suffix == ".jpg"


def test_violations_module_does_not_import_cv2_at_module_level():
    import trafficlens.analytics.violations as violations

    assert "cv2" not in vars(violations)


def test_save_snapshot_writes_annotated_jpeg(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    policy = ViolationPolicy(limit_kmh=50.0)
    event = make_event(track_id=4, frame_index=120, speed_kmh=80.0)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    out_dir = tmp_path / "snapshots"
    path = policy.save_snapshot(frame, event, out_dir)
    assert path == policy.snapshot_path(event, out_dir)
    assert path.exists()
    written = cv2.imread(str(path))
    assert written is not None
    assert written.shape == frame.shape
    assert written.any()  # the annotation left visible marks
    assert not frame.any()  # the caller's frame was not mutated
