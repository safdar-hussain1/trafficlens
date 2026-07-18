"""Directional gate counting via segment-intersection geometry.

Why not "is the object inside a band around the line?" — that approach
(common in tutorials) fails in both directions at once:

* an object moving faster than the band is tall skips it entirely
  between two frames and is never counted;
* an object that grazes the band without crossing (e.g. changes lanes
  along the line) is counted anyway.

Instead, a crossing fires exactly when the segment between an object's
anchor point on consecutive frames intersects the gate segment. That is
frame-rate independent: a car moving 200 px/frame still produces a
movement segment that geometrically crosses the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trafficlens.geometry import Point, segments_intersect, side_of_line


@dataclass(frozen=True)
class CrossingEvent:
    """One object crossing one gate, with everything needed for a report."""

    track_id: int
    class_name: str
    gate: str
    direction: str          # the gate's label for the crossing direction
    signed_direction: int   # +1 / -1, geometric
    frame_index: int
    timestamp: float        # seconds since stream start
    speed: float | None     # in the configured unit, None if not calibrated yet
    is_violation: bool = False


@dataclass
class Gate:
    """A directed counting line in pixel coordinates."""

    name: str
    start: Point
    end: Point
    label_positive: str = "in"
    label_negative: str = "out"

    def label_for(self, signed: int) -> str:
        return self.label_positive if signed > 0 else self.label_negative


@dataclass
class GateCounter:
    """Counts crossings of one gate, once per track.

    ``counted`` remembers which track IDs already crossed so a track that
    lingers on the gate is not counted twice; tracks are cheap (ints), and
    a stream that runs for days can call :meth:`forget` when the tracker
    retires an ID.
    """

    gate: Gate
    counted: set[int] = field(default_factory=set)
    totals: dict[str, dict[str, int]] = field(default_factory=dict)  # class -> direction -> n
    # Last off-line side seen per track. Needed because an anchor can land
    # exactly ON the gate for a frame; the crossing then resolves against
    # the side the track was last on, not the degenerate "side 0".
    _last_side: dict[int, int] = field(default_factory=dict)

    def update(
        self,
        track_id: int,
        class_name: str,
        prev: Point | None,
        curr: Point,
        frame_index: int,
        timestamp: float,
        speed: float | None = None,
        speed_limit: float | None = None,
    ) -> CrossingEvent | None:
        """Feed one track's anchor movement; returns an event on a crossing."""
        s_curr = side_of_line(self.gate.start, self.gate.end, curr)
        s_prev = None if prev is None else side_of_line(self.gate.start, self.gate.end, prev)
        # The side the track is coming from: usually prev's own side, but if
        # prev sat exactly ON the gate, fall back to the last off-line side.
        came_from = s_prev if s_prev not in (None, 0) else self._last_side.get(track_id)
        if s_curr != 0:
            self._last_side[track_id] = s_curr
        if prev is None or track_id in self.counted:
            return None
        if not segments_intersect(prev, curr, self.gate.start, self.gate.end):
            return None
        if s_curr == 0 or came_from is None or came_from == s_curr:
            # Landed exactly on the gate (resolves next frame), or no side
            # change — a graze along the line is not a crossing.
            return None
        signed = s_curr
        self.counted.add(track_id)
        direction = self.gate.label_for(signed)
        self.totals.setdefault(class_name, {}).setdefault(direction, 0)
        self.totals[class_name][direction] += 1
        return CrossingEvent(
            track_id=track_id,
            class_name=class_name,
            gate=self.gate.name,
            direction=direction,
            signed_direction=signed,
            frame_index=frame_index,
            timestamp=timestamp,
            speed=speed,
            is_violation=bool(speed_limit and speed and speed > speed_limit),
        )

    def forget(self, track_id: int) -> None:
        """Drop a retired track ID so long-running streams stay bounded."""
        self.counted.discard(track_id)
        self._last_side.pop(track_id, None)

    def reset_tracks(self) -> None:
        """Clear per-track memory but keep the tallies.

        Needed when the tracker restarts (a looping file, a source
        reconnect): new tracks may reuse old IDs, and stale ``counted``
        entries would silently suppress their crossings.
        """
        self.counted.clear()
        self._last_side.clear()

    @property
    def total(self) -> int:
        return sum(n for per_dir in self.totals.values() for n in per_dir.values())

    def total_by_direction(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for per_dir in self.totals.values():
            for direction, n in per_dir.items():
                out[direction] = out.get(direction, 0) + n
        return out
