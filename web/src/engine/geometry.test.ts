// Ported from tests/test_geometry.py. Every case, every asserted number and
// every pinned sign convention is the Python test's, not this port's: the two
// engines are asserted to make identical crossing decisions later, so these
// tests exist to describe Python's behaviour, not this file's.

import { describe, expect, it } from "vitest";

import {
  boxAnchor,
  crossingDirection,
  segmentIntersectionParam,
  segmentsIntersect,
  sideOfLine,
} from "./geometry";
import type { Point } from "./geometry";

// --- sideOfLine: sign convention --------------------------------------------
//
// Pinned convention: +1 = LEFT of the direction of travel a->b, in image
// coordinates where y grows downward. This concrete case must never silently
// flip -- every later component (crossingDirection, and the counting logic)
// reads this sign.

describe("sideOfLine", () => {
  it("is positive on the left of a horizontal gate", () => {
    // Gate travels along +x (rightward). A point with a smaller y sits
    // "above" the gate on screen (since y grows down), which is the LEFT
    // side of a rightward-facing direction of travel.
    const a: Point = [0.0, 0.0];
    const b: Point = [10.0, 0.0];
    expect(sideOfLine(a, b, [5.0, -5.0])).toBe(1);
  });

  it("is negative on the right of a horizontal gate", () => {
    expect(sideOfLine([0.0, 0.0], [10.0, 0.0], [5.0, 5.0])).toBe(-1);
  });

  it("is zero on the line", () => {
    expect(sideOfLine([0.0, 0.0], [10.0, 0.0], [5.0, 0.0])).toBe(0);
  });

  it("is zero at endpoint a", () => {
    const a: Point = [0.0, 0.0];
    expect(sideOfLine(a, [10.0, 0.0], a)).toBe(0);
  });

  it("flips sign for a reversed gate direction", () => {
    // Reversing the gate direction (b->a instead of a->b) must flip the sign
    // for the same physical point, since "left of travel" depends on which
    // way the gate is defined to face.
    const a: Point = [0.0, 0.0];
    const b: Point = [10.0, 0.0];
    const p: Point = [5.0, -5.0];
    expect(sideOfLine(a, b, p)).toBe(-sideOfLine(b, a, p));
  });
});

// --- segmentsIntersect: the awkward cases ------------------------------------

describe("segmentsIntersect", () => {
  it("finds a simple crossing", () => {
    expect(
      segmentsIntersect([0.0, -5.0], [0.0, 5.0], [-5.0, 0.0], [5.0, 0.0]),
    ).toBe(true);
  });

  it("is false when the segments are far apart", () => {
    expect(
      segmentsIntersect([0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 7.0]),
    ).toBe(false);
  });

  it("finds a collinear overlap", () => {
    expect(
      segmentsIntersect([0.0, 0.0], [10.0, 0.0], [5.0, 0.0], [15.0, 0.0]),
    ).toBe(true);
  });

  it("is false for collinear but disjoint segments", () => {
    expect(
      segmentsIntersect([0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]),
    ).toBe(false);
  });

  it("finds a shared endpoint", () => {
    expect(
      segmentsIntersect([0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 10.0]),
    ).toBe(true);
  });

  it("finds a T-junction", () => {
    // q1 touches the interior of segment p1-p2, not either of its endpoints.
    expect(
      segmentsIntersect([0.0, 0.0], [10.0, 0.0], [5.0, 0.0], [5.0, 5.0]),
    ).toBe(true);
  });

  it("is false for parallel segments", () => {
    expect(
      segmentsIntersect([0.0, 0.0], [10.0, 0.0], [0.0, 5.0], [10.0, 5.0]),
    ).toBe(false);
  });

  it("still catches a fast object", () => {
    // 200 px of travel in one frame across a horizontal gate. This is the
    // frame-rate independence guarantee: a fast-moving object that only ever
    // lands far to one side of the gate must still register as a crossing,
    // because the algorithm checks the full swept segment, not just the
    // object's current position.
    expect(
      segmentsIntersect([100.0, -100.0], [100.0, 100.0], [0.0, 0.0], [200.0, 0.0]),
    ).toBe(true);
  });
});

// --- segmentIntersectionParam ------------------------------------------------

describe("segmentIntersectionParam", () => {
  it("is the midpoint for a symmetric crossing", () => {
    const t = segmentIntersectionParam(
      [5.0, -1.0],
      [5.0, 1.0],
      [0.0, 0.0],
      [10.0, 0.0],
    );
    expect(t).not.toBeNull();
    expect(Math.abs((t as number) - 0.5)).toBeLessThan(1e-12);
  });

  it("lands near the start of the segment for an off-centre crossing", () => {
    const t = segmentIntersectionParam(
      [0.0, -1.0],
      [0.0, 9.0],
      [-5.0, 0.0],
      [5.0, 0.0],
    );
    expect(t).not.toBeNull();
    expect(Math.abs((t as number) - 0.1)).toBeLessThan(1e-12);
  });

  it("is null for parallel segments", () => {
    expect(
      segmentIntersectionParam([0.0, 0.0], [10.0, 0.0], [0.0, 5.0], [10.0, 5.0]),
    ).toBeNull();
  });

  it("is null for collinear segments", () => {
    // Collinear segments have a zero-magnitude cross-product denominator too
    // (direction vectors are parallel), so this must also return null rather
    // than throw or divide by zero.
    expect(
      segmentIntersectionParam([0.0, 0.0], [10.0, 0.0], [5.0, 0.0], [15.0, 0.0]),
    ).toBeNull();
  });

  it("is null for a degenerate zero-length segment", () => {
    // p1 == p2: the segment has no direction, so the 2x2 system is singular.
    expect(
      segmentIntersectionParam([5.0, 5.0], [5.0, 5.0], [0.0, 0.0], [10.0, 0.0]),
    ).toBeNull();
  });
});

// --- crossingDirection --------------------------------------------------------

describe("crossingDirection", () => {
  it("defers when the anchor lands exactly on the gate", () => {
    expect(
      crossingDirection([0.0, 0.0], [10.0, 0.0], [5.0, -1.0], [5.0, 0.0]),
    ).toBe(0);
  });

  it("defers when the start lands exactly on the gate", () => {
    expect(
      crossingDirection([0.0, 0.0], [10.0, 0.0], [5.0, 0.0], [5.0, 1.0]),
    ).toBe(0);
  });

  it("is zero with no crossing (same side)", () => {
    expect(
      crossingDirection([0.0, 0.0], [10.0, 0.0], [5.0, -1.0], [5.0, -2.0]),
    ).toBe(0);
  });

  it("is negative left to right", () => {
    // prev is left of the gate (above), curr is right (below): ends up on the
    // right side, matching sideOfLine's -1 for that side.
    expect(
      crossingDirection([0.0, 0.0], [10.0, 0.0], [5.0, -1.0], [5.0, 1.0]),
    ).toBe(-1);
  });

  it("is positive right to left", () => {
    expect(
      crossingDirection([0.0, 0.0], [10.0, 0.0], [5.0, 1.0], [5.0, -1.0]),
    ).toBe(1);
  });
});

// --- boxAnchor -----------------------------------------------------------------

describe("boxAnchor", () => {
  it("is the bottom centre", () => {
    expect(boxAnchor(10.0, 20.0, 30.0, 60.0)).toEqual([20.0, 60.0]);
  });

  it("is not the box centre", () => {
    // A regression guard against the common mistake of returning the box
    // centre instead of the bottom-centre (where the object meets the road).
    expect(boxAnchor(10.0, 20.0, 30.0, 60.0)).not.toEqual([20.0, 40.0]);
  });
});
