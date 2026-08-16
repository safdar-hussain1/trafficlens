"""The shipped motorway gates must label ordinary traffic as ordinary.

A gate's direction labels are pure geometry: ``side_of_line`` puts +1 on
the LEFT of the gate's start -> end travel direction, and for a gate drawn
left to right at a constant image y -- which both motorway gates are --
that left side is UP the frame, away from the camera. So a vehicle
approaching the camera moves down the frame, ends on the -1 side, and is
labelled with ``label_negative``; one receding is labelled with
``label_positive``. Getting that backwards in a config does not fail any
schema check: it silently turns every ordinary crossing into a wrong-way
incident, which is exactly the defect this file pins.

Everything here runs the REAL ``configs/motorway.yaml`` -- its real gate
endpoints, labels and expected directions -- through the real pipeline,
with only ``source`` swapped for a synthetic grey clip and a scripted
detector standing in for the model. No weights, no torch, no sample
footage needed.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from trafficlens.config import CalibrationConfig, load_config
from trafficlens.core.constants import INCIDENT_MIN_STOPPED_S
from trafficlens.detect.base import Detection
from trafficlens.pipeline import run_session

ROOT = Path(__file__).resolve().parents[1]
MOTORWAY = ROOT / "configs" / "motorway.yaml"

WIDTH, HEIGHT = 640, 480
FRAMES = 40

# Both gates sit at y = 0.80 -> 384 px on this frame. In normalized x the
# inbound gate spans 0.06-0.46 and the outbound 0.52-0.69, so these two
# anchor columns each fall inside exactly one gate's x-span.
GATE_Y = 0.80 * HEIGHT
INBOUND_X = 0.20 * WIDTH
OUTBOUND_X = 0.60 * WIDTH

# 4 px per frame keeps frame-to-frame IoU near 0.9 on a 60x80 box, clear
# of the tracker's 0.8 association floor from the very first step.
STEP = 4.0
NEAR, FAR = 460.0, 300.0  # bottom-anchor y either side of the gate line


def write_clip(path: Path, frames: int = FRAMES, fps: float = 30.0) -> Path:
    """A plain grey MJPG AVI. The scripted detector ignores its content,
    but ``VideoSource`` genuinely decodes it."""
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
    """Returns canned detections per frame index; one ``detect`` call is
    one frame, exactly as the pipeline drives it."""

    def __init__(self, script) -> None:
        self._script = script
        self.calls = 0

    def detect(self, frame):
        index = self.calls
        self.calls += 1
        return list(self._script(index))


def car(cx: float, bottom: float) -> Detection:
    """A 60x80 car box whose bottom-centre anchor is ``(cx, bottom)``."""
    return Detection(
        x1=cx - 30.0,
        y1=bottom - 80.0,
        x2=cx + 30.0,
        y2=bottom,
        score=0.9,
        class_id=2,
        class_name="car",
    )


def toward_camera(cx: float):
    """A vehicle driving toward the camera: down the frame, +y."""
    return lambda i: [car(cx, FAR + STEP * i)]


def away_from_camera(cx: float):
    """A vehicle driving away from the camera: up the frame, -y."""
    return lambda i: [car(cx, NEAR - STEP * i)]


def both_carriageways(i):
    """Ordinary flow: inbound approaches, outbound recedes, together."""
    return [car(INBOUND_X, FAR + STEP * i), car(OUTBOUND_X, NEAR - STEP * i)]


@pytest.fixture
def motorway_session(tmp_path):
    """Run the shipped motorway config against a scripted detector,
    returning the finished ``SessionResult``."""
    clip = write_clip(tmp_path / "synthetic.avi")
    config = load_config(MOTORWAY).model_copy(update={"source": str(clip)})

    def run(script):
        return run_session(config, ScriptedDetector(script))

    return run


def wrong_ways(result):
    return [i for i in result.incidents if i.kind == "wrong_way"]


def crossed_gates(result):
    return sorted({event.gate for event in result.events})


# --- the fix: ordinary flow is not an incident --------------------------------


def test_the_gate_span_assumptions_this_file_rests_on():
    """Pins the geometry the scripted tracks are aimed at, so a future
    edit to the gate endpoints fails loudly here rather than making the
    tests below silently vacuous."""
    gates = {g.name: g.to_gate(WIDTH, HEIGHT) for g in load_config(MOTORWAY).gates}
    for gate in gates.values():
        assert gate.start[1] == gate.end[1] == pytest.approx(GATE_Y)
        assert gate.start[0] < gate.end[0], "gates must run left to right"
        assert FAR < GATE_Y < NEAR, "scripted tracks must straddle the gate"
    inbound, outbound = gates["inbound"], gates["outbound"]
    assert inbound.start[0] < INBOUND_X < inbound.end[0]
    assert outbound.start[0] < OUTBOUND_X < outbound.end[0]


def test_ordinary_flow_on_both_carriageways_fires_no_wrong_way_incident(
    motorway_session,
):
    result = motorway_session(both_carriageways)

    # Both gates were genuinely crossed -- otherwise "zero incidents"
    # would be true of a session where nothing happened at all.
    assert crossed_gates(result) == ["inbound", "outbound"]
    assert wrong_ways(result) == []


@pytest.mark.parametrize(
    "gate_name, script, expected_label",
    [
        ("inbound", toward_camera(INBOUND_X), "toward"),
        ("outbound", away_from_camera(OUTBOUND_X), "away"),
    ],
)
def test_each_carriageway_alone_is_labelled_its_expected_direction(
    motorway_session, gate_name, script, expected_label
):
    result = motorway_session(script)

    assert [e.gate for e in result.events] == [gate_name]
    event = result.events[0]
    assert event.direction == expected_label
    gates = {g["name"]: g for g in result.meta["gates"]}
    assert event.direction == gates[gate_name]["expected_direction"]
    assert wrong_ways(result) == []


# --- the other direction: a genuinely reversed track DOES fire -----------------


@pytest.mark.parametrize(
    "gate_name, script",
    [
        # Driving away from the camera on the inbound carriageway, and
        # toward it on the outbound one: real wrong-way traffic.
        ("inbound", away_from_camera(INBOUND_X)),
        ("outbound", toward_camera(OUTBOUND_X)),
    ],
)
def test_a_reversed_track_on_either_carriageway_is_a_wrong_way_incident(
    motorway_session, gate_name, script
):
    result = motorway_session(script)

    assert [e.gate for e in result.events] == [gate_name]
    incidents = wrong_ways(result)
    assert len(incidents) == 1
    assert gate_name in incidents[0].detail


# --- the shipped config is UNCALIBRATED, and the pipeline honours that ---------


#: A calibration for the synthetic 640x480 clip used above, in the config's
#: own normalized coordinates. Nothing to do with the real motorway view --
#: its only job is to be the CONTROL for the tests below, so "no speed" is
#: shown to come from the shipped config's missing calibration rather than
#: from a scripted track that could never have produced one anyway.
_CONTROL_CALIBRATION_POINTS = {
    "image_points": [
        [0.20, 0.98], [0.20, 0.72], [0.20, 0.60],
        [0.60, 0.98], [0.60, 0.72], [0.60, 0.60],
    ],
    "world_points": [
        [0.0, 0.0], [0.0, 18.0], [0.0, 36.0],
        [3.75, 0.0], [3.75, 18.0], [3.75, 36.0],
    ],
}
CONTROL_CALIBRATION = CalibrationConfig(**_CONTROL_CALIBRATION_POINTS)


def test_the_shipped_motorway_config_reports_no_speed_at_all(motorway_session):
    """The flagship clip is uncalibrated, so every speed is None.

    Asserted alongside evidence that the session was not simply empty:
    both gates were crossed and both tracks reached the speed dictionary.
    "No speeds" on a session where nothing was tracked would be true of a
    broken pipeline too.
    """
    result = motorway_session(both_carriageways)

    assert result.meta["calibrated"] is False
    assert crossed_gates(result) == ["inbound", "outbound"]
    assert len(result.speeds) == 2, result.speeds
    assert set(result.speeds.values()) == {None}


def test_a_calibrated_control_on_the_same_clip_does_report_speeds(
    tmp_path, motorway_session
):
    """The must-succeed half of the pair, varying exactly one axis: the
    presence of a calibration block. Without this, the test above would
    pass just as well against a pipeline that had stopped estimating speed
    for any reason at all."""
    clip = write_clip(tmp_path / "control.avi")
    config = load_config(MOTORWAY).model_copy(
        update={"source": str(clip), "calibration": CONTROL_CALIBRATION}
    )
    result = run_session(config, ScriptedDetector(both_carriageways))

    assert result.meta["calibrated"] is True
    assert len(result.speeds) == 2, result.speeds
    assert all(speed is not None for speed in result.speeds.values()), result.speeds


def test_stopped_vehicle_detection_never_fires_on_the_uncalibrated_config(
    tmp_path,
):
    """A stated consequence, pinned rather than discovered later: stopped
    detection requires calibrated speed by design, so it cannot fire on
    this clip. The calibrated control on the identical stationary script
    is what shows the feature is otherwise alive."""
    def stationary(i):
        return [car(INBOUND_X, 460.0)]

    # INCIDENT_MIN_STOPPED_S is 10 s of continuous sub-3 km/h speed, so the
    # clip must be longer than that or neither side could fire and the
    # control would prove nothing.
    frames = int((INCIDENT_MIN_STOPPED_S + 2.0) * 30.0)

    def run(calibration):
        clip = write_clip(tmp_path / f"stopped-{bool(calibration)}.avi", frames=frames)
        update = {"source": str(clip)}
        if calibration is not None:
            update["calibration"] = calibration
        config = load_config(MOTORWAY).model_copy(update=update)
        return run_session(config, ScriptedDetector(stationary))

    shipped = run(None)
    control = run(CONTROL_CALIBRATION)

    assert [i for i in shipped.incidents if i.kind == "stopped"] == []
    assert [i for i in control.incidents if i.kind == "stopped"] != []
