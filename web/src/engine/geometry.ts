/** Exact geometric predicates for gate-crossing detection. Mirrors
 * `trafficlens.core.geometry`.
 *
 * Sign convention
 * ----------------
 * All points are pixel coordinates in image space, where x grows rightward and
 * y grows *downward* (the usual image/video convention, not the math-plot
 * convention). `sideOfLine(a, b, p)` returns `+1` when `p` lies to the LEFT of
 * the direction of travel from `a` to `b` in that coordinate system, `-1` when
 * it lies to the right, and `0` when `p` is on the line (within
 * `GEOMETRY_EPS`).
 *
 * Concretely: for a horizontal gate from `a=(0, 0)` to `b=(10, 0)` (direction
 * of travel pointing along +x), a point with a *smaller* y sits "above" the
 * gate on screen -- since y grows down -- and that is the left side, e.g.
 * `sideOfLine(a, b, [5, -5]) === 1`. A point with a larger y sits "below" the
 * gate on screen and is the right side, e.g. `sideOfLine(a, b, [5, 5]) === -1`.
 *
 * The cross product below is written in the same term order as the Python
 * original. Reordering it can change the last bit, and the whole point of this
 * function is a sign comparison against a threshold. */

import { GEOMETRY_EPS } from "../generated/constants";

export type Point = readonly [number, number];

/** Which side of line a->b the point p falls on.
 *
 * +1 = left of the direction of travel a->b (image coords, y grows down)
 * -1 = right of the direction of travel a->b
 *  0 = p lies on the line a->b, within GEOMETRY_EPS */
export function sideOfLine(a: Point, b: Point, p: Point): number {
  const cross = (b[1] - a[1]) * (p[0] - a[0]) - (b[0] - a[0]) * (p[1] - a[1]);
  if (cross > GEOMETRY_EPS) {
    return 1;
  }
  if (cross < -GEOMETRY_EPS) {
    return -1;
  }
  return 0;
}

/** True if p -- already known to be collinear with a and b -- lies within the
 * closed bounding box of segment a-b. Used only for the collinear cases in
 * segmentsIntersect, where a bounding-box check is sufficient because the
 * points are known to lie on one line. */
function onSegment(a: Point, b: Point, p: Point): boolean {
  return (
    Math.min(a[0], b[0]) - GEOMETRY_EPS <= p[0] &&
    p[0] <= Math.max(a[0], b[0]) + GEOMETRY_EPS &&
    Math.min(a[1], b[1]) - GEOMETRY_EPS <= p[1] &&
    p[1] <= Math.max(a[1], b[1]) + GEOMETRY_EPS
  );
}

/** True if the closed segments p1-p2 and q1-q2 share at least one point.
 *
 * Handles the general crossing case plus the awkward degenerate ones:
 * collinear overlap, a shared endpoint, and a T-junction (an endpoint of one
 * segment touching the interior of the other). */
export function segmentsIntersect(
  p1: Point,
  p2: Point,
  q1: Point,
  q2: Point,
): boolean {
  const d1 = sideOfLine(q1, q2, p1);
  const d2 = sideOfLine(q1, q2, p2);
  const d3 = sideOfLine(p1, p2, q1);
  const d4 = sideOfLine(p1, p2, q2);

  if (d1 * d2 < 0 && d3 * d4 < 0) {
    return true;
  }

  if (d1 === 0 && onSegment(q1, q2, p1)) {
    return true;
  }
  if (d2 === 0 && onSegment(q1, q2, p2)) {
    return true;
  }
  if (d3 === 0 && onSegment(p1, p2, q1)) {
    return true;
  }
  if (d4 === 0 && onSegment(p1, p2, q2)) {
    return true;
  }

  return false;
}

/** Solve the 2x2 system for where the infinite lines p1-p2 and q1-q2 meet, and
 * return the parameter t such that `p1 + t * (p2 - p1)` is that point.
 *
 * Returns null when the segments are parallel or either is degenerate
 * (zero-length), i.e. when the system's denominator has magnitude below
 * GEOMETRY_EPS -- never throws.
 *
 * This solves for the intersection of the two full lines, not just the bounded
 * segments; callers that need to know whether the segments themselves actually
 * meet should check segmentsIntersect first. The returned t is UNCLAMPED, and
 * using it without that bounds check is the original counting bug. */
export function segmentIntersectionParam(
  p1: Point,
  p2: Point,
  q1: Point,
  q2: Point,
): number | null {
  const rx = p2[0] - p1[0];
  const ry = p2[1] - p1[1];
  const sx = q2[0] - q1[0];
  const sy = q2[1] - q1[1];

  const denom = rx * sy - ry * sx;
  if (Math.abs(denom) < GEOMETRY_EPS) {
    return null;
  }

  return ((q1[0] - p1[0]) * sy - (q1[1] - p1[1]) * sx) / denom;
}

/** The direction an object crossed the gate line gateA->gateB while moving
 * from prev to curr.
 *
 * +1 = crossed and ended up on the left side of gateA->gateB
 * -1 = crossed and ended up on the right side
 *  0 = no crossing (prev and curr are on the same side), OR either endpoint
 *      lies exactly on the gate line. The on-the-line case is deferred to the
 *      next frame instead of double-firing a count. */
export function crossingDirection(
  gateA: Point,
  gateB: Point,
  prev: Point,
  curr: Point,
): number {
  const sidePrev = sideOfLine(gateA, gateB, prev);
  const sideCurr = sideOfLine(gateA, gateB, curr);

  if (sidePrev === 0 || sideCurr === 0) {
    return 0;
  }
  if (sidePrev === sideCurr) {
    return 0;
  }
  return sideCurr;
}

/** The bottom-centre point of box (x1, y1)-(x2, y2): where a tracked object
 * meets the road, not the box's centre.
 *
 * The box's top edge plays no part in the answer, but it stays in the
 * signature so a whole xyxy box can be spread into this call the way the
 * Python original takes one. */
export function boxAnchor(x1: number, _y1: number, x2: number, y2: number): Point {
  return [(x1 + x2) / 2.0, y2];
}
