"""Incident detection: stopped vehicles and wrong-way crossings.

Counts and speeds describe normal traffic; incidents are the abnormal
events a traffic operator actually needs to act on:

* **stopped** — a tracked object whose calibrated speed stays under a
  threshold for a sustained time: a stalled car, an obstacle on the
  carriageway, or the head of a forming queue. Requires calibration —
  without real speeds there is no honest way to call something stopped,
  so the detector stays silent on uncalibrated scenes.
* **wrong_way** — a gate crossing against the direction the gate expects
  (configured per gate). On a one-way carriageway that is a wrong-way
  driver; on a doorway it is someone leaving through the entrance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trafficlens.counting import CrossingEvent
from trafficlens.geometry import Point


@dataclass(frozen=True)
class Incident:
    """One abnormal event, with everything needed for an operator log."""

    kind: str               # "stopped" | "wrong_way"
    track_id: int
    class_name: str
    frame_index: int
    timestamp: float
    detail: str
    position: Point


@dataclass
class StoppedVehicleDetector:
    """Flags tracks that stay essentially stationary for too long.

    ``update()`` is fed every frame per track. A track fires at most one
    incident per stop: it must move again (with hysteresis, at twice the
    threshold) before it can fire another.
    """

    speed_threshold: float = 3.0     # in the pipeline's configured unit
    min_duration_s: float = 5.0
    _below_since: dict[int, float] = field(default_factory=dict)
    _flagged: set[int] = field(default_factory=set)

    def update(
        self,
        track_id: int,
        class_name: str,
        speed: float | None,
        anchor: Point,
        frame_index: int,
        timestamp: float,
    ) -> Incident | None:
        if speed is None:
            # Unknown speed (uncalibrated, or track too fresh): no claim.
            self._below_since.pop(track_id, None)
            return None
        if speed >= self.speed_threshold:
            self._below_since.pop(track_id, None)
            if speed >= 2 * self.speed_threshold:
                self._flagged.discard(track_id)  # moving again — re-arm
            return None
        since = self._below_since.setdefault(track_id, timestamp)
        duration = timestamp - since
        if duration >= self.min_duration_s and track_id not in self._flagged:
            self._flagged.add(track_id)
            return Incident(
                kind="stopped",
                track_id=track_id,
                class_name=class_name,
                frame_index=frame_index,
                timestamp=timestamp,
                detail=f"stationary for {duration:.1f} s",
                position=anchor,
            )
        return None

    def is_stopped(self, track_id: int) -> bool:
        return track_id in self._flagged

    def forget(self, track_id: int) -> None:
        self._below_since.pop(track_id, None)
        self._flagged.discard(track_id)

    def reset_tracks(self) -> None:
        """Tracker restarted (looping file / reconnect): drop identity state."""
        self._below_since.clear()
        self._flagged.clear()


def wrong_way_incident(event: CrossingEvent, expected_direction: str | None,
                       position: Point) -> Incident | None:
    """An incident when a crossing runs against the gate's expected flow."""
    if expected_direction is None or event.direction == expected_direction:
        return None
    return Incident(
        kind="wrong_way",
        track_id=event.track_id,
        class_name=event.class_name,
        frame_index=event.frame_index,
        timestamp=event.timestamp,
        detail=f"crossed '{event.gate}' {event.direction} — expected {expected_direction}",
        position=position,
    )
