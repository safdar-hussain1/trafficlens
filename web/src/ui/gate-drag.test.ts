/** The gate is the one thing on this page a visitor can move, so the maths that
 * decides what they grabbed and where it lands is pinned here rather than left
 * to be judged by eye in a browser. Everything under test is pure: a segment in,
 * a segment out, no DOM, no pointer events. */

import { describe, expect, test } from "vitest";

import {
  GATE_HANDLE_RADIUS_PX,
  GATE_MIN_LENGTH_PX,
  applyDrag,
  beginDrag,
  distanceToSegment,
  hitTestGate,
  moveGate,
} from "./gate-drag";
import type { Segment } from "./gate-drag";

const FRAME = { width: 1000, height: 500 };

/** A horizontal gate across the middle of the frame, well clear of every edge,
 * so a test that hits a clamp did so on purpose. */
const GATE: Segment = { start: [200, 250], end: [800, 250] };

describe("distanceToSegment", () => {
  test("measures perpendicular distance for a point beside the segment", () => {
    expect(distanceToSegment([500, 290], GATE.start, GATE.end)).toBeCloseTo(40, 12);
  });

  test("measures distance to the nearer ENDPOINT past the segment's ends", () => {
    // 40 px beyond the start and 30 px below it: the perpendicular foot lands
    // off the segment, so the answer is the 50-px hypotenuse to the endpoint,
    // not the 30-px perpendicular an infinite line would give.
    expect(distanceToSegment([160, 280], GATE.start, GATE.end)).toBeCloseTo(50, 12);
  });

  test("is zero on the segment itself", () => {
    expect(distanceToSegment([437, 250], GATE.start, GATE.end)).toBe(0);
  });

  test("falls back to point distance for a degenerate segment", () => {
    // Never produced by a live gate -- Gate itself refuses zero length -- but a
    // divide-by-zero here would return NaN and silently make everything
    // unhittable, which is the failure mode worth excluding.
    expect(distanceToSegment([3, 4], [0, 0], [0, 0])).toBeCloseTo(5, 12);
  });
});

describe("hitTestGate", () => {
  test("picks the NEARER endpoint when the pointer is near both", () => {
    // A short gate whose two endpoints are both inside the handle radius of the
    // pointer: only "nearer wins" can separate them, and the tie-break has to be
    // measured rather than left to argument order.
    const short: Segment = { start: [100, 100], end: [110, 100] };
    expect(hitTestGate(short, [104, 100])).toBe("start");
    expect(hitTestGate(short, [106, 100])).toBe("end");
  });

  test("prefers an endpoint over the body when both are in range", () => {
    // Right on the line AND right on the start handle. The body is at distance
    // zero, so a naive nearest-thing-wins rule would return "body" and the gate
    // could never be re-aimed by its ends.
    expect(hitTestGate(GATE, [200, 250])).toBe("start");
    expect(hitTestGate(GATE, [800, 250])).toBe("end");
  });

  test("returns the body for a pointer near the middle of the segment", () => {
    expect(hitTestGate(GATE, [500, 250 + GATE_HANDLE_RADIUS_PX - 1])).toBe("body");
  });

  test("returns null beyond the radius", () => {
    expect(hitTestGate(GATE, [500, 250 + GATE_HANDLE_RADIUS_PX + 1])).toBeNull();
  });

  test("grabs exactly at the radius -- the boundary is inclusive", () => {
    expect(hitTestGate(GATE, [500, 250 + GATE_HANDLE_RADIUS_PX])).toBe("body");
    expect(hitTestGate(GATE, [200 - GATE_HANDLE_RADIUS_PX, 250])).toBe("start");
  });

  test("honours a caller-supplied radius", () => {
    expect(hitTestGate(GATE, [500, 300])).toBeNull();
    expect(hitTestGate(GATE, [500, 300], 60)).toBe("body");
  });
});

describe("moveGate", () => {
  test("moving the body translates BOTH endpoints by the same delta", () => {
    const moved = moveGate(GATE, "body", 30, -40, FRAME);
    expect(moved.start).toEqual([230, 210]);
    expect(moved.end).toEqual([830, 210]);
  });

  test("moving an endpoint leaves the other endpoint exactly where it was", () => {
    const moved = moveGate(GATE, "start", -50, 60, FRAME);
    expect(moved.start).toEqual([150, 310]);
    expect(moved.end).toEqual(GATE.end);
  });

  test("clamps a dragged endpoint to the frame", () => {
    const moved = moveGate(GATE, "end", 500, 500, FRAME);
    expect(moved.end).toEqual([1000, 500]);
  });

  test("clamps a body drag WITHOUT shortening the gate", () => {
    // The naive implementation clamps each endpoint independently, which
    // squashes a gate dragged into a corner and silently changes what it spans.
    const moved = moveGate(GATE, "body", 900, 0, FRAME);
    expect(moved.end).toEqual([1000, 250]);
    expect(moved.start).toEqual([400, 250]);
    expect(moved.end[0] - moved.start[0]).toBe(600);
  });

  test("clamps a body drag against the near edge too", () => {
    const moved = moveGate(GATE, "body", -900, -900, FRAME);
    expect(moved.start).toEqual([0, 0]);
    expect(moved.end).toEqual([600, 0]);
  });

  test("refuses to collapse a gate to zero length", () => {
    // Gate's constructor throws on zero length, so a drag that produced one
    // would take the page down. The endpoint stops at the minimum instead.
    const moved = moveGate(GATE, "start", 600, 0, FRAME);
    const length = Math.hypot(moved.end[0] - moved.start[0], moved.end[1] - moved.start[1]);
    expect(length).toBeCloseTo(GATE_MIN_LENGTH_PX, 9);
    expect(moved.end).toEqual(GATE.end);
  });

  test("keeps the minimum length along the drag direction, not the old one", () => {
    // Dragging start far past end vertically: the survivor must sit on the side
    // the pointer went to, or the gate would flip back under the user's hand.
    const moved = moveGate({ start: [500, 100], end: [500, 300] }, "start", 0, 200, FRAME);
    expect(moved.start[1]).toBeCloseTo(300 + GATE_MIN_LENGTH_PX, 9);
  });
});

describe("beginDrag / applyDrag", () => {
  test("a grab remembers the segment it started from, so the drag is absolute", () => {
    const grab = beginDrag(GATE, [500, 252]);
    expect(grab?.kind).toBe("body");
    // Two successive pointer positions from ONE grab: the second must be
    // measured from the grab origin, not compounded onto the first.
    const first = applyDrag(grab as NonNullable<typeof grab>, [520, 262], FRAME);
    const second = applyDrag(grab as NonNullable<typeof grab>, [540, 272], FRAME);
    expect(first.start).toEqual([220, 260]);
    expect(second.start).toEqual([240, 270]);
  });

  test("the grabbed endpoint keeps its offset from the pointer -- no jump", () => {
    // The pointer grabs 6 px to the left of the start handle; the endpoint must
    // stay 6 px to the right of the pointer for the whole drag.
    const grab = beginDrag(GATE, [194, 250]);
    expect(grab?.kind).toBe("start");
    const moved = applyDrag(grab as NonNullable<typeof grab>, [394, 250], FRAME);
    expect(moved.start).toEqual([400, 250]);
  });

  test("returns null when nothing was grabbed", () => {
    expect(beginDrag(GATE, [500, 400])).toBeNull();
  });
});
