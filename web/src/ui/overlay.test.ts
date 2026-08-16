/** The fit between a video frame and the canvas it is drawn on.
 *
 * Tested because the pointer rides on it: the gate is dragged in the canvas's
 * coordinates and stored in the frame's, so a wrong scale or a forgotten
 * letterbox offset moves the gate somewhere other than where the finger is --
 * and, since the gate decides what is counted, quietly changes the measurement
 * as well as the picture. */

import { describe, expect, test } from "vitest";

import { boxToFrame, fitContain, frameToBox } from "./overlay";

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
