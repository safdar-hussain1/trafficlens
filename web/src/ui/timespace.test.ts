/** The time-space diagram is the page's whole argument, so its geometry is
 * pinned here rather than judged by eye.
 *
 * The diagram's vertical axis is signed distance from the gate LINE, which is
 * why a count lands exactly on the axis's zero: the crossing is the moment that
 * distance changes sign. The two failure modes of a band rule are the same
 * picture -- a trajectory that steps across zero without ever sampling inside
 * the band, and one that sits inside the band for many frames without ever
 * changing sign. Both are asserted below, because both are claims the page
 * makes to the visitor. */

import { describe, expect, test } from "vitest";

import { sideOfLine } from "../engine/geometry";
import {
  chooseSpanPx,
  crossesBand,
  gateParam,
  projectSample,
  signedDistanceToGate,
  trimWindow,
  withinGateSpan,
} from "./timespace";
import type { TimeSpaceView, Trace } from "./timespace";

/** Left to right at a constant image y -- the shipped motorway framing. */
const GATE = { start: [200, 250] as const, end: [800, 250] as const };

describe("signedDistanceToGate", () => {
  test("agrees in SIGN with the engine's own side-of-line convention", () => {
    // The diagram and the counter must not disagree about which side is which,
    // or the marker would appear on the wrong side of the axis from the count.
    for (const p of [
      [500, 200],
      [500, 300],
      [210, 100],
      [790, 480],
    ] as const) {
      expect(Math.sign(signedDistanceToGate(GATE, p))).toBe(sideOfLine(GATE.start, GATE.end, p));
    }
  });

  test("measures perpendicular pixels, not the raw cross product", () => {
    expect(signedDistanceToGate(GATE, [500, 200])).toBeCloseTo(50, 12);
    expect(signedDistanceToGate(GATE, [500, 330])).toBeCloseTo(-80, 12);
  });

  test("is zero exactly on the line, including beyond the segment's ends", () => {
    expect(signedDistanceToGate(GATE, [500, 250])).toBe(0);
    expect(signedDistanceToGate(GATE, [-400, 250])).toBe(0);
  });

  test("is unchanged by measuring against a longer gate on the same line", () => {
    const longer = { start: [0, 250] as const, end: [1000, 250] as const };
    expect(signedDistanceToGate(longer, [500, 200])).toBeCloseTo(
      signedDistanceToGate(GATE, [500, 200]),
      12,
    );
  });
});

describe("gateParam / withinGateSpan", () => {
  test("runs 0 at the start to 1 at the end", () => {
    expect(gateParam(GATE, [200, 250])).toBeCloseTo(0, 12);
    expect(gateParam(GATE, [800, 250])).toBeCloseTo(1, 12);
    expect(gateParam(GATE, [500, 999])).toBeCloseTo(0.5, 12);
  });

  test("marks a vehicle beyond the gate's ends as outside its span", () => {
    // The parallel carriageway: it crosses the gate's infinite LINE and is never
    // counted, and the diagram has to be able to draw that difference.
    expect(withinGateSpan(GATE, [900, 250])).toBe(false);
    expect(withinGateSpan(GATE, [100, 250])).toBe(false);
    expect(withinGateSpan(GATE, [800, 250])).toBe(true);
  });
});

describe("projectSample", () => {
  const view: TimeSpaceView = {
    now: 20,
    windowS: 10,
    width: 400,
    height: 200,
    spanPx: 100,
    padding: { left: 40, right: 10, top: 10, bottom: 20 },
  };

  test("puts now at the right edge of the plot and the window's start at the left", () => {
    expect(projectSample(view, 20, 0).x).toBeCloseTo(390, 9);
    expect(projectSample(view, 10, 0).x).toBeCloseTo(40, 9);
    expect(projectSample(view, 15, 0).x).toBeCloseTo(215, 9);
  });

  test("puts the gate on the plot's centre line, whatever the padding", () => {
    expect(projectSample(view, 20, 0).y).toBeCloseTo(95, 9);
  });

  test("draws the positive side UP the canvas, matching the video's framing", () => {
    // +1 is up the frame, away from the camera. If the diagram flipped it, a
    // vehicle receding on the video would descend on the chart.
    expect(projectSample(view, 20, 100).y).toBeCloseTo(10, 9);
    expect(projectSample(view, 20, -100).y).toBeCloseTo(180, 9);
  });
});

describe("chooseSpanPx", () => {
  test("never zooms in past the floor, so a still scene is not magnified into noise", () => {
    expect(chooseSpanPx(3, 120)).toBe(120);
  });

  test("grows in round steps rather than continuously", () => {
    // A continuously-fitted axis rescales every frame and the picture crawls;
    // stepping keeps it still until a trajectory genuinely needs more room.
    expect(chooseSpanPx(130, 120)).toBe(150);
    expect(chooseSpanPx(151, 120)).toBe(200);
    expect(chooseSpanPx(200, 120)).toBe(200);
  });
});

describe("trimWindow", () => {
  const trace: Trace = [
    { t: 1, d: 10 },
    { t: 5, d: 20 },
    { t: 12, d: 30 },
    { t: 18, d: 40 },
  ];

  test("keeps the samples inside the window", () => {
    expect(trimWindow(trace, 20, 10).map((s) => s.t)).toEqual([5, 12, 18]);
  });

  test("keeps ONE sample before the window so the line enters from the edge", () => {
    // Dropping it would make every trajectory begin at the left edge at the
    // moment it scrolled in, which reads as a vehicle appearing from nowhere.
    expect(trimWindow(trace, 20, 10)[0]).toEqual({ t: 5, d: 20 });
    expect(trimWindow(trace, 20, 6).map((s) => s.t)).toEqual([12, 18]);
  });

  test("returns nothing for a trace that ended before the window", () => {
    expect(trimWindow(trace, 100, 10)).toEqual([]);
  });
});

describe("crossesBand -- the two failure modes of a band rule", () => {
  const BAND = 20;

  test("a count fires where the sign changes, whatever the band did", () => {
    const stepping: Trace = [
      { t: 0, d: 60 },
      { t: 1, d: -60 },
    ];
    expect(crossesBand(stepping, BAND)).toEqual({ crossed: true, touchedBand: false });
  });

  test("grazing inside the band for a long time is not a crossing", () => {
    const grazing: Trace = [
      { t: 0, d: 12 },
      { t: 1, d: 8 },
      { t: 2, d: 15 },
      { t: 3, d: 19 },
    ];
    expect(crossesBand(grazing, BAND)).toEqual({ crossed: false, touchedBand: true });
  });

  test("an ordinary crossing does both", () => {
    const ordinary: Trace = [
      { t: 0, d: 30 },
      { t: 1, d: 10 },
      { t: 2, d: -10 },
    ];
    expect(crossesBand(ordinary, BAND)).toEqual({ crossed: true, touchedBand: true });
  });

  test("a sample landing exactly ON the line counts as a crossing once it leaves", () => {
    const touching: Trace = [
      { t: 0, d: 30 },
      { t: 1, d: 0 },
      { t: 2, d: -30 },
    ];
    expect(crossesBand(touching, BAND).crossed).toBe(true);
  });

  test("a trace that never approaches does neither", () => {
    const distant: Trace = [
      { t: 0, d: 200 },
      { t: 1, d: 190 },
    ];
    expect(crossesBand(distant, BAND)).toEqual({ crossed: false, touchedBand: false });
  });
});
