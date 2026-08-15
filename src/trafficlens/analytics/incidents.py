"""Incident detection: stopped vehicles and wrong-way gate crossings.

Two independent detections share one per-track state store:

**Stopped vehicles.** A track whose *calibrated* speed stays strictly below
``stopped_speed_kmh`` for at least ``min_stopped_s`` CONTINUOUS seconds
fires exactly one ``Incident(kind="stopped")`` -- one per stop episode, not
one per frame. The detector re-arms only after the track's speed rises to
or above the threshold again, so a car that stops twice fires twice, and a
car that never moves in between fires once no matter how long it sits.

Calibration policy, inherited from ``trafficlens.analytics.speed``: a
``speed_kmh`` of ``None`` means the camera is uncalibrated, and an
uncalibrated frame is *unknown*, not stopped. A ``None`` speed therefore
neither fires an incident nor accumulates stopped time -- it breaks the
continuous sub-threshold run, so the required duration must be observed
again from scratch once calibrated speeds resume. It does NOT re-arm an
already-fired episode, though: losing calibration is not evidence the
vehicle moved, so a fire / calibration gap / still-stopped sequence stays
one incident.

**Wrong-way crossings.** ``note_crossing`` compares a ``CrossingEvent``'s
direction label against its gate's ``expected_direction`` (one of the
gate's two direction labels; see ``trafficlens.core.gate.Gate``). A
crossing whose label differs fires ``Incident(kind="wrong_way")``, once
per (track, gate) -- ``GateCounter`` already emits at most one crossing
per track per gate, but a re-notified identical event is cheaply absorbed
here anyway. A gate with no ``expected_direction`` never fires; an
``expected_direction`` that matches neither of the gate's labels raises
``ValueError`` (a configuration bug, not a wrong-way vehicle).

This module imports nothing beyond the standard library and
``trafficlens.core``, so it stays testable without a video, model, or
tracker present.
"""

from dataclasses import dataclass, field

from trafficlens.core.constants import (
    INCIDENT_MIN_STOPPED_S,
    INCIDENT_STOPPED_SPEED_KMH,
)
from trafficlens.core.gate import CrossingEvent, Gate


@dataclass(frozen=True)
class Incident:
    """One detected incident, resolved to a single frame and track."""

    kind: str  # "stopped" | "wrong_way"
    track_id: int
    class_name: str
    frame_index: int
    timestamp: float
    detail: str  # human-readable, e.g. "stationary for 12.3 s"


@dataclass
class _StopState:
    """Per-track stopped-vehicle bookkeeping."""

    # Timestamp when the current continuous sub-threshold run began, or
    # None when no run is in progress (moving, or an uncalibrated gap).
    stopped_since: float | None = None
    # Whether the current stop episode has already fired. Cleared only by
    # an at-or-above-threshold speed (genuine movement), never by a None.
    fired: bool = False


class IncidentDetector:
    """Detects stopped vehicles from per-frame speeds and wrong-way
    movement from gate crossings. See the module docstring for the exact
    firing semantics of each kind."""

    def __init__(
        self,
        min_stopped_s: float = INCIDENT_MIN_STOPPED_S,
        stopped_speed_kmh: float = INCIDENT_STOPPED_SPEED_KMH,
    ) -> None:
        if min_stopped_s <= 0.0:
            raise ValueError(f"min_stopped_s must be positive, got {min_stopped_s}")
        if stopped_speed_kmh <= 0.0:
            raise ValueError(
                f"stopped_speed_kmh must be positive, got {stopped_speed_kmh}"
            )
        self.min_stopped_s = min_stopped_s
        self.stopped_speed_kmh = stopped_speed_kmh
        self._stops: dict[int, _StopState] = {}
        # (track_id, gate name) pairs that already fired wrong_way.
        self._wrong_way_fired: set[tuple[int, str]] = set()

    def update(
        self,
        track_id: int,
        class_name: str,
        speed_kmh: float | None,
        timestamp: float,
        frame_index: int,
    ) -> Incident | None:
        """Feed one frame's speed for a track; return a stopped-vehicle
        Incident the first frame its continuous sub-threshold run reaches
        ``min_stopped_s``, else None."""
        state = self._stops.get(track_id)
        if state is None:
            state = _StopState()
            self._stops[track_id] = state

        if speed_kmh is None:
            # Uncalibrated: unknown, not stopped. Break the continuous run
            # (no accumulation across the gap) but do not re-arm a fired
            # episode -- there is no evidence the vehicle moved.
            state.stopped_since = None
            return None

        if speed_kmh >= self.stopped_speed_kmh:
            # Genuine movement: end the episode and re-arm.
            state.stopped_since = None
            state.fired = False
            return None

        if state.stopped_since is None:
            state.stopped_since = timestamp
        duration = timestamp - state.stopped_since
        if state.fired or duration < self.min_stopped_s:
            return None

        state.fired = True
        return Incident(
            kind="stopped",
            track_id=track_id,
            class_name=class_name,
            frame_index=frame_index,
            timestamp=timestamp,
            detail=f"stationary for {duration:.1f} s",
        )

    def note_crossing(self, event: CrossingEvent, gate: Gate) -> Incident | None:
        """Return a wrong_way Incident when the event's direction label
        opposes the gate's expected direction, at most once per
        (track, gate); else None."""
        expected = gate.expected_direction
        if expected is None:
            return None
        if expected not in (gate.label_positive, gate.label_negative):
            raise ValueError(
                f"Gate {gate.name!r} expected_direction {expected!r} matches "
                f"neither of its labels ({gate.label_positive!r}, "
                f"{gate.label_negative!r})"
            )
        if event.direction == expected:
            return None

        key = (event.track_id, event.gate)
        if key in self._wrong_way_fired:
            return None
        self._wrong_way_fired.add(key)

        return Incident(
            kind="wrong_way",
            track_id=event.track_id,
            class_name=event.class_name,
            frame_index=event.frame_index,
            timestamp=event.timestamp,
            detail=(
                f"crossed {event.gate!r} as {event.direction!r} against "
                f"expected flow {expected!r}"
            ),
        )

    def forget(self, track_id: int) -> None:
        """Drop all per-track state, so a recycled tracker ID starts
        fresh. Unknown IDs are a no-op."""
        self._stops.pop(track_id, None)
        self._wrong_way_fired = {
            key for key in self._wrong_way_fired if key[0] != track_id
        }
