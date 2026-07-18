"""Pure geometry used by counting and speed estimation.

Everything here operates on plain floats/tuples so it can be unit-tested
without any video, model, or OpenCV dependency.
"""

from __future__ import annotations

import math

Point = tuple[float, float]

# Cross products below this magnitude are treated as exactly parallel /
# collinear. Coordinates are pixels, so 1e-9 is far below sub-pixel noise.
_EPS = 1e-9


def side_of_line(a: Point, b: Point, p: Point) -> int:
    """Which side of the infinite line through ``a``->``b`` is ``p`` on?

    Returns +1 / -1 for the two half-planes and 0 when ``p`` is on the
    line. Sign convention: +1 means ``p`` is to the left of the direction
    of travel from ``a`` to ``b`` (in image coordinates, y grows down).
    """
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if cross > _EPS:
        return 1
    if cross < -_EPS:
        return -1
    return 0


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True when closed segments ``p1p2`` and ``q1q2`` intersect.

    Standard orientation test, including the collinear-overlap edge
    cases. Used to decide whether an object's movement between two
    consecutive frames crossed a counting gate.
    """

    def orient(a: Point, b: Point, c: Point) -> int:
        return side_of_line(a, b, c)

    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return (
            min(a[0], b[0]) - _EPS <= c[0] <= max(a[0], b[0]) + _EPS
            and min(a[1], b[1]) - _EPS <= c[1] <= max(a[1], b[1]) + _EPS
        )

    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, p2, q2):
        return True
    if o3 == 0 and on_segment(q1, q2, p1):
        return True
    if o4 == 0 and on_segment(q1, q2, p2):
        return True
    return False


def crossing_direction(gate_a: Point, gate_b: Point, prev: Point, curr: Point) -> int:
    """Signed direction of a crossing of the gate ``gate_a``->``gate_b``.

    Returns +1 when the movement ``prev``->``curr`` crosses from the
    negative half-plane to the positive one, -1 for the opposite, and 0
    when there is no side change (touching or sliding along the gate).
    """
    s_prev = side_of_line(gate_a, gate_b, prev)
    s_curr = side_of_line(gate_a, gate_b, curr)
    if s_prev == s_curr or s_prev == 0 or s_curr == 0:
        # A point exactly on the line resolves on the next frame; this
        # avoids double-firing when an anchor lands on the gate.
        return 0
    return s_curr


def euclidean(a: Point, b: Point) -> float:
    """Distance between two points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])
