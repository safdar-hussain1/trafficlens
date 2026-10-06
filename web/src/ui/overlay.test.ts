/** The fit between a video frame and the canvas it is drawn on, and the parts
 * of the drawing that make a claim.
 *
 * The fit is tested because the pointer rides on it: the gate is dragged in the
 * canvas's coordinates and stored in the frame's, so a wrong scale or a
 * forgotten letterbox offset moves the gate somewhere other than where the
 * finger is -- and, since the gate decides what is counted, quietly changes the
 * measurement as well as the picture. The rest is tested because it labels
 * things: a direction tag on the wrong side of the line names the wrong count. */

import { describe, expect, test } from "vitest";

import { sideOfLine } from "../engine/geometry";
import type { Point } from "../engine/geometry";
import { sourceById } from "./sources";
import {
  GLOW_SECONDS,
  POP_SECONDS,
  boxToFrame,
  chipLabel,
  fitContain,
  frameToBox,
  gateGlow,
  gateNormal,
  popFrame,
} from "./overlay";

const FRAME = { width: 1280, height: 720 };

describe("fitContain", () => {
  test("fills the width and centres vertically when the box is taller", () => {
    const fit = fitContain(FRAME, { width: 640, height: 480 });
    expect(fit.scale).toBeCloseTo(0.5, 12);
    expect(fit.dx).toBeCloseTo(0, 12);
    expect(fit.dy).toBeCloseTo(60, 12);
  });

  test("fills the height and centres horizontally when the box is wider", () => {
    const fit = fitContain(FRAME, { width: 1600, height: 720 });
    expect(fit.scale).toBeCloseTo(1, 12);
    expect(fit.dx).toBeCloseTo(160, 12);
    expect(fit.dy).toBeCloseTo(0, 12);
  });

  test("never crops -- the smaller ratio always wins", () => {
    const fit = fitContain(FRAME, { width: 320, height: 720 });
    expect(fit.scale).toBeCloseTo(0.25, 12);
  });

  test("survives a frame with no decoded pixels yet", () => {
    // A <video> reports 0x0 until metadata arrives, and the page draws before
    // that; a division here would put NaN into every coordinate on the canvas.
    expect(fitContain({ width: 0, height: 0 }, { width: 640, height: 480 })).toEqual({
      scale: 1,
      dx: 0,
      dy: 0,
    });
  });
});

describe("frameToBox / boxToFrame", () => {
  const fit = fitContain(FRAME, { width: 640, height: 480 });

  test("round-trips a point", () => {
    const there = frameToBox([200, 500], fit);
    expect(there).toEqual([100, 310]);
    const back = boxToFrame(there, fit);
    expect(back[0]).toBeCloseTo(200, 9);
    expect(back[1]).toBeCloseTo(500, 9);
  });

  test("maps a pointer in the letterbox bars to a frame position outside the frame", () => {
    // Deliberately unclamped: clamping belongs to the drag, which knows the
    // frame's bounds. Silently pinning here would make a gate dragged into the
    // bar stick to the edge and then jump when it came back.
    expect(boxToFrame([100, 10], fit)[1]).toBeCloseTo(-100, 9);
  });
});

describe("gateNormal", () => {
  const lines: [Point, Point][] = [
    [[0, 0], [100, 0]],
    [[0, 0], [0, 100]],
    [[10, 90], [200, 15]],
    [[300, 40], [20, 260]],
  ];

  test("points to the side the engine calls +1, at any angle", () => {
    // The positive tag is drawn along this normal, and a crossing that ends on
    // the engine's +1 side is counted under the positive label. If the two
    // disagreed, the tag would sit over the other direction's count.
    for (const [a, b] of lines) {
      const normal = gateNormal(a, b);
      expect(Math.hypot(normal[0], normal[1])).toBeCloseTo(1, 12);
      const middle: Point = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      expect(sideOfLine(a, b, [middle[0] + normal[0] * 20, middle[1] + normal[1] * 20])).toBe(1);
      expect(sideOfLine(a, b, [middle[0] - normal[0] * 20, middle[1] - normal[1] * 20])).toBe(-1);
    }
  });

  test("puts the motorway's 'away' tag up the frame, away from the camera", () => {
    const motorway = sourceById("motorway");
    const normal = gateNormal(motorway.gate.start, motorway.gate.end);
    expect(motorway.gate.labelPositive).toBe("away");
    expect(normal[1]).toBeLessThan(0);
  });
});

describe("popFrame", () => {
  test("exists only between its crossing and POP_SECONDS later", () => {
    expect(popFrame(-0.01, false)).toBeNull();
    expect(popFrame(POP_SECONDS + 0.01, false)).toBeNull();
    expect(popFrame(Number.NaN, false)).toBeNull();
    expect(popFrame(0, false)).not.toBeNull();
    expect(popFrame(POP_SECONDS, false)).not.toBeNull();
  });

  test("rises and fades, and never sinks or brightens again", () => {
    let previous = popFrame(0, false);
    expect(previous?.alpha).toBe(1);
    for (let age = 0.05; age <= POP_SECONDS; age += 0.05) {
      const frame = popFrame(age, false);
      expect(frame).not.toBeNull();
      expect(frame!.rise).toBeGreaterThanOrEqual(previous!.rise);
      expect(frame!.alpha).toBeLessThanOrEqual(previous!.alpha);
      expect(frame!.ringAlpha).toBeLessThanOrEqual(previous!.ringAlpha);
      previous = frame;
    }
  });

  test("with reduced motion nothing moves and nothing spreads", () => {
    const first = popFrame(0, true);
    for (const age of [0, 0.3, 0.9, POP_SECONDS]) {
      const frame = popFrame(age, true);
      expect(frame).toEqual(first);
      expect(frame?.ringAlpha).toBe(0);
      expect(frame?.alpha).toBe(1);
    }
  });
});

describe("gateGlow", () => {
  // Stamped at zero so the boundary is exact: 10 + 0.45 - 10 is a hair under
  // 0.45 in floating point, which would test the arithmetic, not the rule.
  test("is full at a crossing and gone GLOW_SECONDS later", () => {
    expect(gateGlow([{ timestamp: 0 }], 0)).toBe(1);
    expect(gateGlow([{ timestamp: 0 }], GLOW_SECONDS / 2)).toBeCloseTo(0.5, 9);
    expect(gateGlow([{ timestamp: 0 }], GLOW_SECONDS)).toBe(0);
    expect(gateGlow([], 10)).toBe(0);
  });

  test("ignores a crossing stamped after now, which a looping clip can leave behind", () => {
    expect(gateGlow([{ timestamp: 12 }], 10)).toBe(0);
  });

  test("with reduced motion it is on or off, never fading", () => {
    expect(gateGlow([{ timestamp: 0 }], GLOW_SECONDS / 2, true)).toBe(1);
    expect(gateGlow([{ timestamp: 0 }], GLOW_SECONDS, true)).toBe(0);
  });
});

describe("chipLabel", () => {
  test("names the type and the track, so the same vehicle keeps the same name", () => {
    expect(chipLabel({ className: "car", trackId: 31 }, false)).toBe("Car 31");
    expect(chipLabel({ className: "truck", trackId: 7 }, false)).toBe("Truck 7");
  });

  test("says wrong way instead when the crossing went against the line", () => {
    expect(chipLabel({ className: "car", trackId: 31 }, true)).toBe("Wrong way");
  });
});
