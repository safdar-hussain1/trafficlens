"""Exact geometric predicates for gate-crossing detection.

Sign convention
----------------
All points are pixel coordinates in image space, where x grows rightward
and y grows *downward* (the usual image/video convention, not the
math-plot convention). ``side_of_line(a, b, p)`` returns ``+1`` when ``p``
lies to the LEFT of the direction of travel from ``a`` to ``b`` in that
coordinate system, ``-1`` when it lies to the right, and ``0`` when ``p``
is on the line (within ``GEOMETRY_EPS``).

Concretely: for a horizontal gate from ``a=(0, 0)`` to ``b=(10, 0)``
(direction of travel pointing along +x), a point with a *smaller* y sits
"above" the gate on screen -- since y grows down -- and that is the left
side, e.g. ``side_of_line(a, b, (5, -5)) == 1``. A point with a larger y
sits "below" the gate on screen and is the right side, e.g.
``side_of_line(a, b, (5, 5)) == -1``. See
``tests/test_geometry.py::test_side_of_line_left_is_positive_for_horizontal_gate``
for the test that pins this.

This module imports nothing beyond the standard library so it can be
unit-tested with no video, model, or other heavy dependency present, and
so it can be mechanically ported to the TypeScript browser engine.
"""

from trafficlens.core.constants import GEOMETRY_EPS

Point = tuple[float, float]


def side_of_line(a: Point, b: Point, p: Point) -> int:
    """Return which side of line a->b the point p falls on.

    +1 = left of the direction of travel a->b (image coords, y grows down)
    -1 = right of the direction of travel a->b
     0 = p lies on the line a->b, within GEOMETRY_EPS
    """
    cross = (b[1] - a[1]) * (p[0] - a[0]) - (b[0] - a[0]) * (p[1] - a[1])
    if cross > GEOMETRY_EPS:
        return 1
    if cross < -GEOMETRY_EPS:
        return -1
    return 0


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    """True if p -- already known to be collinear with a and b -- lies
    within the closed bounding box of segment a-b. Used only for the
    collinear cases in segments_intersect, where a bounding-box check is
    sufficient because the points are known to lie on one line.
    """
    return (
        min(a[0], b[0]) - GEOMETRY_EPS <= p[0] <= max(a[0], b[0]) + GEOMETRY_EPS
        and min(a[1], b[1]) - GEOMETRY_EPS <= p[1] <= max(a[1], b[1]) + GEOMETRY_EPS
    )


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Return True if closed segments p1-p2 and q1-q2 share at least one
    point.

    Handles the general crossing case plus the awkward degenerate ones:
    collinear overlap, a shared endpoint, and a T-junction (an endpoint of
    one segment touching the interior of the other).
    """
    d1 = side_of_line(q1, q2, p1)
    d2 = side_of_line(q1, q2, p2)
    d3 = side_of_line(p1, p2, q1)
    d4 = side_of_line(p1, p2, q2)

    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    if d1 == 0 and _on_segment(q1, q2, p1):
        return True
    if d2 == 0 and _on_segment(q1, q2, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, q1):
        return True
    if d4 == 0 and _on_segment(p1, p2, q2):
        return True

    return False


def segment_intersection_param(
    p1: Point, p2: Point, q1: Point, q2: Point
) -> float | None:
    """Solve the 2x2 system for where infinite lines p1-p2 and q1-q2 meet,
    and return the parameter t such that ``p1 + t * (p2 - p1)`` is that
    point.

    Returns None when the segments are parallel or either is degenerate
    (zero-length), i.e. when the system's denominator has magnitude below
    GEOMETRY_EPS -- never raises. Used later for sub-frame crossing
    timestamps, so it must be exact, not approximate.

    This solves for the intersection of the two full lines, not just the
    bounded segments; callers that need to know whether the segments
    themselves actually meet should check segments_intersect first.
    """
    rx = p2[0] - p1[0]
    ry = p2[1] - p1[1]
    sx = q2[0] - q1[0]
    sy = q2[1] - q1[1]

    denom = rx * sy - ry * sx
    if abs(denom) < GEOMETRY_EPS:
        return None

    t = ((q1[0] - p1[0]) * sy - (q1[1] - p1[1]) * sx) / denom
    return t


def crossing_direction(gate_a: Point, gate_b: Point, prev: Point, curr: Point) -> int:
    """Return the direction an object crossed the gate line gate_a->gate_b
    while moving from prev to curr.

    +1 = crossed and ended up on the left side of gate_a->gate_b
    -1 = crossed and ended up on the right side
     0 = no crossing (prev and curr are on the same side), OR either
         endpoint lies exactly on the gate line. The on-the-line case is
         deferred to the next frame instead of double-firing a count.
    """
    side_prev = side_of_line(gate_a, gate_b, prev)
    side_curr = side_of_line(gate_a, gate_b, curr)

    if side_prev == 0 or side_curr == 0:
        return 0
    if side_prev == side_curr:
        return 0
    return side_curr


def box_anchor(x1: float, y1: float, x2: float, y2: float) -> Point:
    """Return the bottom-centre point of box (x1, y1)-(x2, y2): where a
    tracked object meets the road, not the box's centre.
    """
    return ((x1 + x2) / 2.0, y2)
