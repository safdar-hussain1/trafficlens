"""Incident detection: stopped vehicles and wrong-way crossings."""

import pytest
from pydantic import ValidationError

from trafficlens.config import GateConfig, IncidentsConfig
from trafficlens.counting import CrossingEvent
from trafficlens.incidents import Incident, StoppedVehicleDetector, wrong_way_incident

FPS = 30.0


def drive(det: StoppedVehicleDetector, speeds: list[float | None], track_id: int = 1):
    """Feed a speed sequence at 30 fps; return the incidents raised."""
    out = []
    for i, s in enumerate(speeds):
        inc = det.update(track_id, "car", s, (100.0, 200.0), i, i / FPS)
        if inc:
            out.append(inc)
    return out


class TestStoppedVehicleDetector:
    def make(self) -> StoppedVehicleDetector:
        return StoppedVehicleDetector(speed_threshold=3.0, min_duration_s=2.0)

    def test_sustained_stop_fires_once(self):
        det = self.make()
        incidents = drive(det, [1.0] * 90)  # 3 s stationary
        assert len(incidents) == 1
        assert incidents[0].kind == "stopped"
        assert "stationary" in incidents[0].detail
        assert det.is_stopped(1)

    def test_brief_slowdown_does_not_fire(self):
        det = self.make()
        assert drive(det, [1.0] * 30 + [20.0] * 30) == []  # only 1 s below

    def test_moving_traffic_never_fires(self):
        det = self.make()
        assert drive(det, [40.0] * 120) == []

    def test_unknown_speed_makes_no_claim(self):
        det = self.make()
        assert drive(det, [None] * 300) == []

    def test_refires_after_moving_again(self):
        det = self.make()
        seq = [1.0] * 90 + [30.0] * 30 + [1.0] * 90  # stop, drive off, stop again
        incidents = drive(det, seq)
        assert len(incidents) == 2

    def test_creeping_queue_below_hysteresis_stays_flagged(self):
        det = self.make()
        # stops, then creeps at 4 km/h (below the 2x re-arm threshold): still
        # the same stop event, not a second incident
        incidents = drive(det, [1.0] * 90 + [4.0] * 30 + [1.0] * 90)
        assert len(incidents) == 1

    def test_reset_tracks_rearms_for_new_stream(self):
        det = self.make()
        drive(det, [1.0] * 90)
        det.reset_tracks()
        assert not det.is_stopped(1)
        assert len(drive(det, [1.0] * 90)) == 1


def make_event(direction: str) -> CrossingEvent:
    return CrossingEvent(
        track_id=7, class_name="car", gate="inbound", direction=direction,
        signed_direction=1 if direction == "in" else -1,
        frame_index=42, timestamp=1.4, speed=50.0,
    )


class TestWrongWay:
    def test_expected_direction_passes(self):
        assert wrong_way_incident(make_event("in"), "in", (0.0, 0.0)) is None

    def test_opposite_direction_raises_incident(self):
        inc = wrong_way_incident(make_event("out"), "in", (0.0, 0.0))
        assert isinstance(inc, Incident)
        assert inc.kind == "wrong_way"
        assert "expected in" in inc.detail

    def test_no_expectation_means_no_incident(self):
        assert wrong_way_incident(make_event("out"), None, (0.0, 0.0)) is None

    def test_gate_config_rejects_unknown_expected_direction(self):
        with pytest.raises(ValidationError, match="expected_direction"):
            GateConfig(name="g", start=(0.1, 0.5), end=(0.9, 0.5),
                       expected_direction="north")

    def test_gate_config_accepts_custom_label(self):
        g = GateConfig(name="g", start=(0.1, 0.5), end=(0.9, 0.5),
                       label_positive="entering", label_negative="leaving",
                       expected_direction="leaving")
        assert g.expected_direction == "leaving"


def test_incidents_config_bounds():
    with pytest.raises(ValidationError):
        IncidentsConfig(stopped_speed_threshold=0)
    with pytest.raises(ValidationError):
        IncidentsConfig(stopped_min_duration_s=-1)
