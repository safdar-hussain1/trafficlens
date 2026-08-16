// Ported from tests/test_gate.py. The bounded-segment section and the
// on-line-deferral section each encode a defect found by review on the Python
// side, so they are ported case for case rather than paraphrased.

import { describe, expect, it } from "vitest";

import { Gate, GateCounter, isOverLimit } from "./gate";
import type { CrossingEvent } from "./gate";

// --- once-per-track counting --------------------------------------------------

describe("once-per-track counting", () => {
  it("counts a lingering track once", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    expect(g.update(1, "car", [5.0, -2.0], [5.0, 2.0], 1, 0.04)).not.toBeNull();
    for (let f = 2; f < 20; f += 1) {
      expect(
        g.update(1, "car", [5.0, 2.0], [5.0, 2.0 + f * 0.01], f, f * 0.04),
      ).toBeNull();
    }
    expect(g.total()).toBe(1);
  });

  it("never fires on touch and retreat", () => {
    // Anchor touches the gate line exactly, then retreats back to the side it
    // came from: never a genuine crossing.
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    expect(g.update(1, "car", [5.0, -2.0], [5.0, 0.0], 1, 0.04)).toBeNull();
    expect(g.update(1, "car", [5.0, 0.0], [5.0, -1.0], 2, 0.08)).toBeNull();
    expect(g.total()).toBe(0);
  });
});

// --- the on-line deferral -------------------------------------------------------

describe("on-line deferral", () => {
  it("resolves an on-line frame against the last off-line side", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    expect(g.update(1, "car", [5.0, -2.0], [5.0, 0.0], 1, 0.04)).toBeNull(); // deferred
    const ev = g.update(1, "car", [5.0, 0.0], [5.0, 2.0], 2, 0.08); // resolves
    expect(ev).not.toBeNull();
    expect((ev as CrossingEvent).signedDirection).toBe(-1);
  });

  it("clears the last-side state on forget", () => {
    // If forget() did not clear the last side, this on-line-prev frame would
    // wrongly resolve against the stale side remembered before forget().
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    expect(g.update(1, "car", [5.0, -2.0], [5.0, 0.0], 1, 0.04)).toBeNull();
    g.forget(1);
    expect(g.update(1, "car", [5.0, 0.0], [5.0, 2.0], 2, 0.08)).toBeNull();
  });

  it("resolves against the last off-line POINT, not against prev", () => {
    // Not in the Python suite, and added because a mutation proved nothing
    // there guards this: replacing the stored last off-line point with `prev`
    // leaves all 541 Python tests green. It is still a real behavioural
    // difference, and this is the case that shows it (verified against the
    // Python engine before it was written here).
    //
    // A track jumps from far outside the short gate's span straight onto the
    // gate's LINE at x = 5, which IS inside the span, and then continues
    // across. Resolving the deferral against `prev` would bounds-check the
    // segment (5, 0) -> (5, 2), which touches the gate, and count. Resolving
    // it against the remembered last off-line position bounds-checks
    // (50, -2) -> (5, 2), which crosses the gate's line at x = 27.5, well past
    // the end of the gate, and correctly does not count.
    const gate = new Gate("g", [4.0, 0.0], [6.0, 0.0]);
    const gc = new GateCounter(gate);
    expect(gc.update(1, "car", [50.0, -2.0], [5.0, 0.0], 1, 0.04)).toBeNull();
    expect(gc.update(1, "car", [5.0, 0.0], [5.0, 2.0], 2, 0.08)).toBeNull();
    expect(gc.total()).toBe(0);

    // The must-survive half, varying only where the last off-line position
    // was: from inside the span, the very same deferral resolves into a count.
    const inSpan = new GateCounter(new Gate("g", [4.0, 0.0], [6.0, 0.0]));
    expect(inSpan.update(1, "car", [5.0, -2.0], [5.0, 0.0], 1, 0.04)).toBeNull();
    const ev = inSpan.update(1, "car", [5.0, 0.0], [5.0, 2.0], 2, 0.08);
    expect(ev).not.toBeNull();
    expect((ev as CrossingEvent).signedDirection).toBe(-1);
    expect(inSpan.total()).toBe(1);
  });
});

// --- direction labels -------------------------------------------------------------

describe("direction labels", () => {
  it("labels opposite directions correctly", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    const evOut = g.update(1, "car", [5.0, -1.0], [5.0, 1.0], 1, 0.04);
    const evIn = g.update(2, "car", [5.0, 1.0], [5.0, -1.0], 1, 0.04);
    expect(evOut).not.toBeNull();
    expect((evOut as CrossingEvent).signedDirection).toBe(-1);
    expect((evOut as CrossingEvent).direction).toBe("out");
    expect(evIn).not.toBeNull();
    expect((evIn as CrossingEvent).signedDirection).toBe(1);
    expect((evIn as CrossingEvent).direction).toBe("in");
  });

  it("uses custom gate labels", () => {
    const gate = new Gate("g", [0.0, 0.0], [10.0, 0.0], {
      labelPositive: "north",
      labelNegative: "south",
    });
    const g = new GateCounter(gate);
    const ev = g.update(1, "car", [5.0, 1.0], [5.0, -1.0], 1, 0.04);
    expect(ev).not.toBeNull();
    expect((ev as CrossingEvent).direction).toBe("north");
  });
});

// --- forget() recycling ------------------------------------------------------------

describe("forget", () => {
  it("allows a recycled track id to count again", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    expect(g.update(1, "car", [5.0, -2.0], [5.0, 2.0], 1, 0.04)).not.toBeNull();
    expect(g.total()).toBe(1);
    // Same id crosses back without forget: must not count again.
    expect(g.update(1, "car", [5.0, 2.0], [5.0, -2.0], 2, 0.08)).toBeNull();
    g.forget(1);
    expect(g.update(1, "car", [5.0, -2.0], [5.0, 2.0], 3, 0.12)).not.toBeNull();
    expect(g.total()).toBe(2);
  });

  it("leaves no per-track record behind", () => {
    // The counter remembers every track it has counted, so a session in which
    // vehicles come and go one after another would grow one permanent record
    // per vehicle if forget() missed any of the three. Python has this as a
    // pipeline-level high-water-mark test; here it is asserted directly,
    // because a mutation showed the deferral tests above are satisfied by the
    // last-off-line-point clearing alone and say nothing about the other two.
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    g.update(1, "car", [5.0, -2.0], [5.0, 2.0], 1, 0.04); // counts, sets all three
    expect(g._counted.size).toBe(1);
    expect(g._lastSide.size).toBe(1);
    expect(g._lastOffLinePoint.size).toBe(1);

    g.forget(1);
    expect(g._counted.size).toBe(0);
    expect(g._lastSide.size).toBe(0);
    expect(g._lastOffLinePoint.size).toBe(0);
    // The tally itself survives -- forget drops per-track state, not counts.
    expect(g.total()).toBe(1);
  });
});

// --- totals ------------------------------------------------------------------------

describe("totals", () => {
  it("accumulates per class and direction", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    g.update(1, "car", [5.0, -1.0], [5.0, 1.0], 1, 0.04); // out
    g.update(2, "car", [5.0, -1.0], [5.0, 1.0], 1, 0.04); // out
    g.update(3, "truck", [5.0, 1.0], [5.0, -1.0], 1, 0.04); // in
    expect(g.totals).toEqual(
      new Map([
        ["car", new Map([["out", 2]])],
        ["truck", new Map([["in", 1]])],
      ]),
    );
    expect(g.total()).toBe(3);
  });
});

// --- CrossingEvent shape ------------------------------------------------------------

describe("CrossingEvent", () => {
  it("is frozen", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    const ev = g.update(1, "car", [5.0, -1.0], [5.0, 1.0], 1, 0.04);
    expect(ev).not.toBeNull();
    expect(() => {
      (ev as { trackId: number }).trackId = 99;
    }).toThrow(TypeError);
  });

  it("records its fields verbatim", () => {
    const g = new GateCounter(new Gate("mygate", [0.0, 0.0], [10.0, 0.0]));
    const ev = g.update(7, "bus", [5.0, -1.0], [5.0, 1.0], 42, 1.68);
    expect(ev).not.toBeNull();
    const e = ev as CrossingEvent;
    expect(e.trackId).toBe(7);
    expect(e.className).toBe("bus");
    expect(e.gate).toBe("mygate");
    expect(e.frameIndex).toBe(42);
    expect(e.timestamp).toBe(1.68);
  });

  it("records the actual intersection for a centred crossing", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    const ev = g.update(1, "car", [5.0, -2.0], [5.0, 2.0], 1, 0.04);
    expect(ev).not.toBeNull();
    expect((ev as CrossingEvent).crossingX).toBeCloseTo(5.0, 12);
    expect((ev as CrossingEvent).crossingY).toBeCloseTo(0.0, 12);
  });

  it("records the actual intersection off centre", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    const ev = g.update(1, "car", [2.0, -1.0], [2.0, 9.0], 1, 0.04);
    expect(ev).not.toBeNull();
    expect((ev as CrossingEvent).crossingX).toBeCloseTo(2.0, 12);
    expect((ev as CrossingEvent).crossingY).toBeCloseTo(0.0, 12);
  });
});

// --- Gate construction and validation ------------------------------------------------

describe("Gate construction", () => {
  it("rejects a zero-length gate", () => {
    expect(() => new Gate("g", [5.0, 5.0], [5.0, 5.0])).toThrow();
  });

  it("converts normalized coordinates to pixels", () => {
    const g = Gate.fromNormalized("g", [0.0, 0.5], [1.0, 0.5], 1920, 1080);
    expect(g.start).toEqual([0.0, 540.0]);
    expect(g.end).toEqual([1920.0, 540.0]);
    expect(g.name).toBe("g");
  });

  it("passes options through fromNormalized", () => {
    const g = Gate.fromNormalized("g", [0.0, 0.0], [1.0, 1.0], 100, 100, {
      labelPositive: "north",
      labelNegative: "south",
      expectedDirection: "north",
    });
    expect(g.labelPositive).toBe("north");
    expect(g.labelNegative).toBe("south");
    expect(g.expectedDirection).toBe("north");
  });

  it("rejects a zero-length gate after conversion", () => {
    // start and end are DIFFERENT normalized points (0.2, 0.5) vs (0.8, 0.5),
    // so a check performed on the raw normalized inputs before conversion
    // would not trip here. width=0 collapses the x-axis, so both convert to
    // the same pixel point (0.0, 50.0) -- this pins that the zero-length
    // rejection happens on the converted pixel coordinates, not on the raw
    // normalized inputs.
    expect(() =>
      Gate.fromNormalized("g", [0.2, 0.5], [0.8, 0.5], 0, 100),
    ).toThrow();
  });

  it("rejects a normalized coordinate above one", () => {
    expect(() =>
      Gate.fromNormalized("g", [0.0, 0.0], [1.5, 1.0], 100, 100),
    ).toThrow();
  });

  it("rejects a negative normalized coordinate", () => {
    expect(() =>
      Gate.fromNormalized("g", [-0.1, 0.0], [1.0, 1.0], 100, 100),
    ).toThrow();
  });
});

// --- bounded-segment intersection (not just the infinite line) ------------------------
//
// sideOfLine / crossingDirection only test which side of the gate's *infinite*
// line a point falls on. A genuine count also requires the swept path to
// intersect the *finite* gate segment -- otherwise a track on a completely
// different carriageway, crossing the drawn gate's line extended far past its
// endpoints, would be counted as traffic through that gate. All gates below
// are short (span x in [4, 6], or a short diagonal) specifically to exercise
// this.

describe("bounded gate segment", () => {
  it("does not count a crossing beyond the gate's end point", () => {
    // Repro from review: gate spans x in [4, 6]; the track crosses the
    // infinite line at x=50, 44 px past the gate's end point.
    const gate = new Gate("g", [4.0, 0.0], [6.0, 0.0]);
    const gc = new GateCounter(gate);
    expect(gc.update(1, "car", [50.0, -2.0], [50.0, 2.0], 1, 0.04)).toBeNull();
    expect(gc.total()).toBe(0);
  });

  it("does not count a crossing before the gate's start point", () => {
    const gate = new Gate("g", [4.0, 0.0], [6.0, 0.0]);
    const gc = new GateCounter(gate);
    expect(gc.update(1, "car", [0.0, -2.0], [0.0, 2.0], 1, 0.04)).toBeNull();
    expect(gc.total()).toBe(0);
  });

  it("counts a crossing within the gate's span", () => {
    // Guard against over-correcting: a genuine crossing inside the short
    // gate's span must still count.
    const gate = new Gate("g", [4.0, 0.0], [6.0, 0.0]);
    const gc = new GateCounter(gate);
    expect(gc.update(1, "car", [5.0, -2.0], [5.0, 2.0], 1, 0.04)).not.toBeNull();
    expect(gc.total()).toBe(1);
  });

  it("counts a crossing exactly at a gate endpoint", () => {
    // Decision, pinned: gate bounds are inclusive of their endpoints,
    // matching segmentsIntersect's treatment of a shared endpoint /
    // T-junction as an intersection. A crossing that lands exactly on the
    // gate's start point still counts.
    const gate = new Gate("g", [4.0, 0.0], [6.0, 0.0]);
    const gc = new GateCounter(gate);
    const ev = gc.update(1, "car", [4.0, -2.0], [4.0, 2.0], 1, 0.04);
    expect(ev).not.toBeNull();
    expect((ev as CrossingEvent).crossingX).toBeCloseTo(4.0, 12);
    expect((ev as CrossingEvent).crossingY).toBeCloseTo(0.0, 12);
  });

  it("does not count a diagonal-gate crossing outside the span", () => {
    // Diagonal gate from (0,0) to (4,4). The path crosses the infinite line
    // x=y at (10, 10), far outside the gate's own span.
    const gate = new Gate("g", [0.0, 0.0], [4.0, 4.0]);
    const gc = new GateCounter(gate);
    expect(gc.update(1, "car", [9.0, 11.0], [11.0, 9.0], 1, 0.04)).toBeNull();
    expect(gc.total()).toBe(0);
  });

  it("still respects the gate bounds on the deferral path", () => {
    // The deferral path (prev lands exactly on the infinite line) must
    // bounds-check against the segment spanning the last *off-line* position
    // to curr -- not skip the bounds check entirely. Without the fix, this
    // would incorrectly count: the side flips (matching the last off-line
    // side), but the swept path never comes near the short gate's actual
    // span.
    const gate = new Gate("g", [4.0, 0.0], [6.0, 0.0]);
    const gc = new GateCounter(gate);
    expect(gc.update(1, "car", [50.0, -2.0], [50.0, 0.0], 1, 0.04)).toBeNull(); // deferred
    expect(gc.update(1, "car", [50.0, 0.0], [50.0, 2.0], 2, 0.08)).toBeNull(); // still out of bounds
    expect(gc.total()).toBe(0);
  });
});

function expectPointOnGate(ev: CrossingEvent, gate: Gate, eps = 1e-6): void {
  const minX = Math.min(gate.start[0], gate.end[0]) - eps;
  const maxX = Math.max(gate.start[0], gate.end[0]) + eps;
  const minY = Math.min(gate.start[1], gate.end[1]) - eps;
  const maxY = Math.max(gate.start[1], gate.end[1]) + eps;
  expect(ev.crossingX).toBeGreaterThanOrEqual(minX);
  expect(ev.crossingX).toBeLessThanOrEqual(maxX);
  expect(ev.crossingY).toBeGreaterThanOrEqual(minY);
  expect(ev.crossingY).toBeLessThanOrEqual(maxY);
}

describe("recorded crossing point lies on the gate", () => {
  it("horizontal", () => {
    const gate = new Gate("g", [4.0, 0.0], [6.0, 0.0]);
    const gc = new GateCounter(gate);
    const ev = gc.update(1, "car", [5.0, -2.0], [5.0, 2.0], 1, 0.04);
    expect(ev).not.toBeNull();
    expectPointOnGate(ev as CrossingEvent, gate);
  });

  it("vertical", () => {
    const gate = new Gate("g", [0.0, 4.0], [0.0, 6.0]);
    const gc = new GateCounter(gate);
    const ev = gc.update(1, "car", [-2.0, 5.0], [2.0, 5.0], 1, 0.04);
    expect(ev).not.toBeNull();
    expectPointOnGate(ev as CrossingEvent, gate);
  });

  it("diagonal", () => {
    const gate = new Gate("g", [0.0, 0.0], [4.0, 4.0]);
    const gc = new GateCounter(gate);
    const ev = gc.update(1, "car", [1.0, 3.0], [3.0, 1.0], 1, 0.04);
    expect(ev).not.toBeNull();
    expectPointOnGate(ev as CrossingEvent, gate);
  });
});

// --- isOverLimit ---------------------------------------------------------------------

describe("isOverLimit", () => {
  it("is true when strictly greater", () => {
    expect(isOverLimit(60.0, 50.0)).toBe(true);
  });

  it("is false when exactly equal", () => {
    expect(isOverLimit(50.0, 50.0)).toBe(false);
  });

  it("is false when under", () => {
    expect(isOverLimit(40.0, 50.0)).toBe(false);
  });

  it("is false when the speed is null", () => {
    expect(isOverLimit(null, 50.0)).toBe(false);
  });

  it("is false when the limit is null", () => {
    expect(isOverLimit(60.0, null)).toBe(false);
  });

  it("is false when both are null", () => {
    expect(isOverLimit(null, null)).toBe(false);
  });
});

// --- GateCounter <-> isOverLimit wiring -----------------------------------------------

describe("violation flagging", () => {
  it("flags a violation when the speed exceeds the limit", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    const ev = g.update(1, "car", [5.0, -1.0], [5.0, 1.0], 1, 0.04, 80.0, 50.0);
    expect(ev).not.toBeNull();
    expect((ev as CrossingEvent).isViolation).toBe(true);
    expect((ev as CrossingEvent).speedKmh).toBe(80.0);
  });

  it("does not flag a violation when the speed equals the limit", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    const ev = g.update(1, "car", [5.0, -1.0], [5.0, 1.0], 1, 0.04, 50.0, 50.0);
    expect(ev).not.toBeNull();
    expect((ev as CrossingEvent).isViolation).toBe(false);
  });

  it("does not flag a violation when speed and limit are missing", () => {
    const g = new GateCounter(new Gate("g", [0.0, 0.0], [10.0, 0.0]));
    const ev = g.update(1, "car", [5.0, -1.0], [5.0, 1.0], 1, 0.04);
    expect(ev).not.toBeNull();
    expect((ev as CrossingEvent).speedKmh).toBeNull();
    expect((ev as CrossingEvent).isViolation).toBe(false);
  });
});
