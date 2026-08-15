"""Directional gate counting: turns tracked-object movement into counted
crossings, once per track, per class, per direction.

A ``Gate`` is a directed *finite* line segment, not an infinite line.
``GateCounter`` watches each track's positions frame to frame and, the
first time -- and only the first time -- a track's swept path actually
intersects the bounded gate segment (checked with
``trafficlens.core.geometry.segments_intersect``, not merely a
same-side/opposite-side test against the segment's infinite extension),
emits a ``CrossingEvent`` labelled by which side of the gate's direction
of travel the track ended up on, using the same left/right sign
convention as ``trafficlens.core.geometry``. A crossing that lands
exactly on one of the gate's own endpoints counts -- inclusive bounds --
matching ``segments_intersect``'s treatment of a shared endpoint or
T-junction as an intersection.

This module imports nothing beyond the standard library and
``trafficlens.core.geometry``, so it stays testable without a video,
model, or tracker present.
"""

from dataclasses import dataclass

from trafficlens.core.geometry import (
    Point,
    crossing_direction,
    segment_intersection_param,
    segments_intersect,
    side_of_line,
)


def is_over_limit(speed_kmh: float | None, limit_kmh: float | None) -> bool:
    """True only when a known speed strictly exceeds a set limit.

    Returns False whenever either argument is None -- an unknown speed
    or an unset limit can never be "over" -- and uses a strict ``>``
    comparison, so a speed exactly at the limit is not a violation. This
    is the single place the limit comparison is made; other layers
    (e.g. a later violation-snapshot policy) must import this function
    rather than re-implement the comparison.
    """
    if speed_kmh is None or limit_kmh is None:
        return False
    return speed_kmh > limit_kmh


@dataclass(frozen=True)
class CrossingEvent:
    """One gate crossing: a track's path crossed the gate line exactly
    once, resolved to a single frame and a single direction."""

    track_id: int
    class_name: str
    gate: str
    direction: str
    signed_direction: int
    frame_index: int
    timestamp: float
    crossing_x: float
    crossing_y: float
    speed_kmh: float | None
    is_violation: bool


@dataclass
class Gate:
    """A directed line segment that crossings are counted against.

    ``start -> end`` fixes the direction of travel used by
    ``trafficlens.core.geometry.side_of_line``: ``label_positive`` names
    the +1 (left of travel) side, ``label_negative`` names the -1 (right
    of travel) side. ``expected_direction`` is an optional hint -- one of
    the two labels -- for callers that want to flag crossings against an
    unexpected direction; ``GateCounter`` itself does not use it.
    """

    name: str
    start: Point
    end: Point
    label_positive: str = "in"
    label_negative: str = "out"
    expected_direction: str | None = None

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError(
                f"Gate {self.name!r} has zero length: start and end are "
                f"both {self.start!r}. A zero-length gate can never be crossed."
            )

    @classmethod
    def from_normalized(
        cls,
        name: str,
        start: Point,
        end: Point,
        width: float,
        height: float,
        **kw,
    ) -> "Gate":
        """Build a Gate from normalized [0, 1] coordinates plus the pixel
        frame size they're relative to, converting to pixel coordinates.
        """
        for point_name, point in (("start", start), ("end", end)):
            for axis_name, value in (("x", point[0]), ("y", point[1])):
                if not (0.0 <= value <= 1.0):
                    raise ValueError(
                        f"Gate {name!r} normalized {point_name} {axis_name}="
                        f"{value!r} is out of range [0, 1]"
                    )
        pixel_start = (start[0] * width, start[1] * height)
        pixel_end = (end[0] * width, end[1] * height)
        return cls(name, pixel_start, pixel_end, **kw)


class GateCounter:
    """Counts directional crossings of one Gate, once per track ID.

    Once a track has produced a ``CrossingEvent`` for this gate, it never
    produces another until ``forget(track_id)`` is called: a lingering or
    jittering track counts exactly once, ever -- not once per direction
    change.

    A crossing only counts when the track's swept path genuinely
    intersects the *bounded* gate segment (via
    ``trafficlens.core.geometry.segments_intersect``). Two positions
    landing on opposite sides of the gate's infinite line is necessary
    but not sufficient: e.g. a vehicle on a different carriageway,
    crossing the drawn gate's line far outside its two endpoints, is not
    counted.
    """

    def __init__(self, gate: Gate) -> None:
        self.gate = gate
        self._counted: set[int] = set()
        # Last non-zero side (+1/-1) each track was seen on, and the
        # actual point it was seen at. Needed because an anchor landing
        # exactly on the gate line makes side_of_line (and
        # crossing_direction) return 0, which must be resolved on a
        # later frame against the real previous side -- and the real
        # previous off-line position, for the bounded-segment check --
        # rather than lost.
        self._last_side: dict[int, int] = {}
        self._last_off_line_point: dict[int, Point] = {}
        self.totals: dict[str, dict[str, int]] = {}

    def update(
        self,
        track_id: int,
        class_name: str,
        prev: Point,
        curr: Point,
        frame_index: int,
        timestamp: float,
        speed_kmh: float | None = None,
        speed_limit_kmh: float | None = None,
    ) -> CrossingEvent | None:
        gate_a, gate_b = self.gate.start, self.gate.end
        side_prev_actual = side_of_line(gate_a, gate_b, prev)
        side_curr = side_of_line(gate_a, gate_b, curr)

        if side_prev_actual != 0:
            # The normal case: prev itself is the last off-line position,
            # so the segment to bounds-check is prev -> curr.
            origin = prev
            signed = crossing_direction(gate_a, gate_b, prev, curr)
        else:
            # prev landed exactly on the gate line this frame: resolve
            # against the last off-line side (and position) remembered
            # for this track, not against 0 (which would silently drop
            # the crossing) and not against prev's on-line position
            # (which would give the wrong segment to bounds-check).
            origin = self._last_off_line_point.get(track_id)
            last = self._last_side.get(track_id)
            if last is None or side_curr == 0 or last == side_curr:
                signed = 0
            else:
                signed = side_curr

        if side_curr != 0:
            self._last_side[track_id] = side_curr
            self._last_off_line_point[track_id] = curr
        elif side_prev_actual != 0:
            self._last_side[track_id] = side_prev_actual
            self._last_off_line_point[track_id] = prev

        if signed == 0 or track_id in self._counted:
            return None

        if origin is None or not segments_intersect(origin, curr, gate_a, gate_b):
            # The infinite line was crossed, but the swept path never
            # actually meets the bounded gate segment -- e.g. a parallel
            # carriageway crossing the gate's line far past its ends.
            return None

        self._counted.add(track_id)

        t = segment_intersection_param(origin, curr, gate_a, gate_b)
        if t is None:
            # Parallel/collinear relative to the gate line -- segments_intersect
            # can still be True here (collinear overlap), but there is no
            # single well-defined intersection point; fall back to the
            # object's current position rather than raise.
            crossing_x, crossing_y = curr
        else:
            # segments_intersect already confirmed a genuine bounded
            # intersection, so t should already lie in [0, 1]; clamp
            # defensively against floating-point overshoot at the edges.
            t = max(0.0, min(1.0, t))
            crossing_x = origin[0] + t * (curr[0] - origin[0])
            crossing_y = origin[1] + t * (curr[1] - origin[1])

        direction = self.gate.label_positive if signed == 1 else self.gate.label_negative
        violation = is_over_limit(speed_kmh, speed_limit_kmh)

        class_totals = self.totals.setdefault(class_name, {})
        class_totals[direction] = class_totals.get(direction, 0) + 1

        return CrossingEvent(
            track_id=track_id,
            class_name=class_name,
            gate=self.gate.name,
            direction=direction,
            signed_direction=signed,
            frame_index=frame_index,
            timestamp=timestamp,
            crossing_x=crossing_x,
            crossing_y=crossing_y,
            speed_kmh=speed_kmh,
            is_violation=violation,
        )

    def forget(self, track_id: int) -> None:
        """Clear all memory of track_id, so a recycled tracker ID can be
        counted again."""
        self._counted.discard(track_id)
        self._last_side.pop(track_id, None)
        self._last_off_line_point.pop(track_id, None)

    def total(self) -> int:
        return sum(sum(directions.values()) for directions in self.totals.values())
