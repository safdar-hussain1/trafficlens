/** The geometry and the label arithmetic behind the static figures.
 *
 * These are the parts of a chart that go wrong silently. A scale that divides by
 * zero puts every mark at NaN, which renders as an empty plot and reads as
 * missing data; a label run that collides makes two series look like one; a bar
 * rounded at both ends detaches from its baseline and a short one becomes a
 * lozenge. None of that throws, so none of it is caught by the page loading.
 *
 * The drawing functions themselves need a DOM and this suite runs in node, so
 * what is tested here is everything that can be decided without one -- which is
 * deliberately most of it. */

import { describe, expect, test } from "vitest";

import { barPath, linear, pathFrom, shortLevels, spreadLabels, threeSigFigs } from "./figures";

describe("linear", () => {
  test("maps the domain onto the range", () => {
    const scale = linear([0, 1], [100, 200]);
    expect(scale(0)).toBe(100);
    expect(scale(0.5)).toBe(150);
    expect(scale(1)).toBe(200);
  });

  test("maps an inverted range, which is what a y axis is", () => {
    const scale = linear([0, 1], [140, 20]);
    expect(scale(0)).toBe(140);
    expect(scale(1)).toBe(20);
  });

  test("extrapolates rather than clamping", () => {
    // A value outside the domain is a bug in the caller, and a mark drawn outside
    // the plot is how that bug becomes visible. Clamping would hide it against
    // the axis, where it looks like a real reading of exactly zero.
    expect(linear([0, 1], [0, 100])(1.5)).toBe(150);
  });

  test("a zero-width domain collapses to the range start instead of NaN", () => {
    // The failure this exists for: NaN coordinates draw nothing at all, so a
    // one-level sweep would render as an empty panel rather than as one point.
    const scale = linear([5, 5], [10, 90]);
    expect(scale(5)).toBe(10);
    expect(Number.isNaN(scale(7))).toBe(false);
  });
});

describe("pathFrom", () => {
  test("moves to the first point and lines to the rest", () => {
    expect(pathFrom([{ x: 0, y: 1 }, { x: 2, y: 3 }, { x: 4, y: 5 }])).toBe(
      "M0 1 L2 3 L4 5",
    );
  });

  test("rounds to two places, which is well under a rendered pixel", () => {
    expect(pathFrom([{ x: 1.23456, y: 0 }, { x: 2, y: 9.87654 }])).toBe("M1.23 0 L2 9.88");
  });

  test("a single point draws nothing", () => {
    // A lone dot on a line chart reads as a measurement at one level, which is
    // not what a series of length one means -- it means the sweep has one level
    // and the panel has no line to draw.
    expect(pathFrom([{ x: 1, y: 1 }])).toBe("");
    expect(pathFrom([])).toBe("");
  });
});

describe("barPath", () => {
  test("rounds the far end and leaves the baseline square", () => {
    const path = barPath(10, 0, 100, 16, 4);
    // Starts at the baseline, arcs only at the far end, closes back to it.
    expect(path.startsWith("M10 0 H106")).toBe(true);
    expect(path.endsWith("H10 Z")).toBe(true);
    expect((path.match(/A/g) ?? []).length).toBe(2);
  });

  test("a bar shorter than the radius stays a bar", () => {
    // Unclamped, the two arcs meet and overshoot, and a 3-unit bar renders as a
    // circle floating off the axis.
    const path = barPath(0, 0, 3, 16, 8);
    expect(path.includes("NaN")).toBe(false);
    expect(path.startsWith("M0 0")).toBe(true);
  });

  test("a zero-length bar is a degenerate rectangle, not a negative one", () => {
    expect(barPath(0, 0, 0, 10, 4)).toBe("M0 0 H0 V10 H0 Z");
    expect(barPath(0, 0, -5, 10, 4)).toBe("M0 0 H0 V10 H0 Z");
  });
});

describe("spreadLabels", () => {
  test("leaves labels alone when they already clear the gap", () => {
    expect(spreadLabels([10, 30, 50], 9, [0, 100])).toEqual([10, 30, 50]);
  });

  test("separates converging labels by the minimum gap, moving the least it can", () => {
    // Exact, not just "the gaps are big enough". Several arrangements satisfy the
    // gap; only one keeps the lowest label on its own line and pushes the rest
    // away from it, and that is the one that leaves each label next to the line
    // it belongs to. An implementation that merely spaced them out -- dropping the
    // whole run downwards, say -- passes a gap-only assertion while detaching
    // every label from its series.
    expect(spreadLabels([50, 51, 52], 9, [0, 100])).toEqual([50, 59, 68]);
  });

  test("preserves order, so a label never crosses its neighbour", () => {
    // Three trackers at F1 1.0 is the real case: if the arithmetic reordered
    // them, a label would end up beside the wrong line, which is worse than an
    // overlap because it is not visibly wrong.
    const positions = [70, 40, 41, 39];
    const out = spreadLabels(positions, 10, [0, 100]);
    const rank = (values: readonly number[]) =>
      values
        .map((value, index) => ({ value, index }))
        .sort((a, b) => a.value - b.value)
        .map((item) => item.index);
    expect(rank(out)).toEqual(rank(positions));
  });

  test("keeps a pushed run inside its bounds", () => {
    const bounds: readonly [number, number] = [0, 30];
    const out = spreadLabels([28, 29, 30], 10, bounds);
    for (const value of out) {
      expect(value).toBeLessThanOrEqual(bounds[1]);
    }
  });
});

describe("shortLevels", () => {
  test("strips the word every level repeats at the end", () => {
    expect(shortLevels(["0% dropped", "5% dropped", "30% dropped"])).toEqual([
      "0%",
      "5%",
      "30%",
    ]);
    expect(shortLevels(["sigma=0 px", "sigma=1 px", "sigma=8 px"])).toEqual([
      "sigma=0",
      "sigma=1",
      "sigma=8",
    ]);
  });

  test("strips a repeated leading word too", () => {
    expect(shortLevels(["level 30 fps", "level 25 fps"])).toEqual(["30", "25"]);
  });

  test("leaves labels with nothing in common untouched", () => {
    // The negative control: a rule that simply dropped the last word would ruin
    // these, and `p=0.05` has no space in it to drop.
    expect(shortLevels(["p=0.00", "p=0.05", "p=0.30"])).toEqual([
      "p=0.00",
      "p=0.05",
      "p=0.30",
    ]);
    expect(shortLevels(["30 fps", "25 fps"])).toEqual(["30", "25"]);
  });

  test("never empties a label", () => {
    // Identical labels share every word; stripping them all would print a row of
    // blanks under the axis.
    expect(shortLevels(["same thing", "same thing"])).toEqual(["thing", "thing"]);
    expect(shortLevels(["one"])).toEqual(["one"]);
  });
});

describe("threeSigFigs", () => {
  test("prints three significant figures, not three decimals", () => {
    expect(threeSigFigs(0.9142857142857143)).toBe("0.914");
    expect(threeSigFigs(0.05843071786310518)).toBe("0.0584");
    expect(threeSigFigs(21.75)).toBe("21.8");
  });

  test("an exact zero is exact, not 0.00", () => {
    // The engine's F1 at 2 fps really is zero -- it made no predictions at all.
    // Printing 0.00 implies a rounded small number rather than nothing.
    expect(threeSigFigs(0)).toBe("0");
  });

  test("never emits exponent notation, which no table column wants", () => {
    expect(threeSigFigs(0.0000123)).not.toContain("e");
  });

  test("a null is an em dash, not a zero", () => {
    // One benchmark row is genuinely null: at 2 fps the engine reaches the gate
    // with no identities, so the fold from one identity per vehicle is undefined.
    // A zero there would read as a perfect score.
    expect(threeSigFigs(null)).toBe("—");
    expect(threeSigFigs(Number.NaN)).toBe("—");
  });
});
