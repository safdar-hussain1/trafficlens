/** Everything the control room prints goes through here, so the page cannot
 * quietly invent a number it does not have. Two rules are load-bearing rather
 * than cosmetic: an absent measurement formats as an em dash instead of a zero,
 * and the backend readout is a MEDIAN -- a mean lets one 900 ms compile stall
 * describe a session that never ran that slowly again. */

import { describe, expect, test } from "vitest";

import {
  NARROW_NBSP,
  RollingMedian,
  decideCadence,
  formatClock,
  formatCount,
  formatFps,
  formatMs,
  formatSpeed,
  median,
} from "./format";

describe("formatCount", () => {
  test("prints small counts bare", () => {
    expect(formatCount(0)).toBe("0");
    expect(formatCount(7)).toBe("7");
    expect(formatCount(999)).toBe("999");
  });

  test("groups thousands with a narrow no-break space, not a comma", () => {
    // A comma is a decimal separator in the countries this clip was filmed in,
    // so "1,234" is genuinely ambiguous; the space group is not.
    expect(formatCount(1000)).toBe(`1${NARROW_NBSP}000`);
    expect(formatCount(1234567)).toBe(`1${NARROW_NBSP}234${NARROW_NBSP}567`);
  });

  test("refuses anything that is not a counted whole number", () => {
    expect(formatCount(Number.NaN)).toBe("—");
    expect(formatCount(-1)).toBe("—");
    expect(formatCount(2.5)).toBe("—");
    expect(formatCount(Number.POSITIVE_INFINITY)).toBe("—");
  });
});

describe("formatMs / formatFps", () => {
  test("keeps one decimal where it carries information", () => {
    expect(formatMs(12.84)).toBe("12.8");
    expect(formatMs(9.05)).toBe("9.1");
  });

  test("drops the decimal above 100 ms, where it is noise", () => {
    expect(formatMs(123.4)).toBe("123");
  });

  test("prints an em dash for a measurement that has not happened", () => {
    expect(formatMs(null)).toBe("—");
    expect(formatFps(null)).toBe("—");
    expect(formatMs(Number.NaN)).toBe("—");
  });

  test("prints fps to one decimal", () => {
    expect(formatFps(58.53)).toBe("58.5");
    expect(formatFps(8.1)).toBe("8.1");
  });
});

describe("formatClock", () => {
  test("prints minutes and zero-padded seconds", () => {
    expect(formatClock(0)).toBe("0:00");
    expect(formatClock(7.9)).toBe("0:07");
    expect(formatClock(61)).toBe("1:01");
    expect(formatClock(600)).toBe("10:00");
  });

  test("prints an em dash before there is a clock", () => {
    expect(formatClock(null)).toBe("—");
  });
});

describe("formatSpeed", () => {
  test("prints km/h only when the source is calibrated AND the engine has a number", () => {
    expect(formatSpeed(87.4, true)).toBe("87 km/h");
  });

  test("says no speed -- in words -- when the source is not surveyed", () => {
    // The engine returns null rather than a pixel-derived guess; the page has to
    // say the same thing rather than show a blank or a zero.
    expect(formatSpeed(null, false)).toBe("no speed");
    // Even if a number arrived from somewhere, an uncalibrated source may not
    // print it. The refusal is about the source, not about the value.
    expect(formatSpeed(87.4, false)).toBe("no speed");
  });

  test("distinguishes not-yet-measured from not-measurable on a calibrated source", () => {
    expect(formatSpeed(null, true)).toBe("—");
  });
});

describe("median", () => {
  test("takes the middle of an odd-length sample", () => {
    expect(median([5, 1, 3])).toBe(3);
  });

  test("averages the two middles of an even-length sample", () => {
    expect(median([1, 2, 3, 10])).toBe(2.5);
  });

  test("does not disturb the caller's array", () => {
    const values = [5, 1, 3];
    median(values);
    expect(values).toEqual([5, 1, 3]);
  });

  test("has no answer for no samples", () => {
    expect(median([])).toBeNull();
  });

  test("is not the mean -- one outlier must not move it", () => {
    const steady = [10, 10, 10, 10, 10];
    expect(median([...steady, 900])).toBe(10);
  });
});

describe("RollingMedian", () => {
  test("reports the median of the last N samples only", () => {
    const rolling = new RollingMedian(3);
    rolling.push(900);
    rolling.push(10);
    rolling.push(10);
    expect(rolling.value()).toBe(10);
    rolling.push(10);
    // 900 has now rolled out of the window entirely.
    expect(rolling.value()).toBe(10);
    expect(rolling.count).toBe(3);
  });

  test("has no value before the first sample", () => {
    expect(new RollingMedian(8).value()).toBeNull();
  });

  test("forgets everything on reset", () => {
    // Switching source changes the frame size and so the per-frame cost; the
    // badge must not keep reporting the previous source's timing.
    const rolling = new RollingMedian(4);
    rolling.push(12);
    rolling.reset();
    expect(rolling.count).toBe(0);
    expect(rolling.value()).toBeNull();
  });

  test("rejects a non-finite sample rather than poisoning the window", () => {
    const rolling = new RollingMedian(4);
    rolling.push(12);
    rolling.push(Number.NaN);
    expect(rolling.count).toBe(1);
    expect(rolling.value()).toBe(12);
  });
});

describe("decideCadence", () => {
  test("detects every frame when the backend keeps up with the source", () => {
    // 30 fps source: a frame budget of 33.3 ms. 12.8 ms/frame is well inside it.
    const cadence = decideCadence(12.8, 30);
    expect(cadence.stride).toBe(1);
    expect(cadence.label).toBe("every frame");
  });

  test("thins detection when a frame costs more than the source's frame budget", () => {
    // 123 ms/frame against a 33.3 ms budget: one detection per 4 source frames.
    const cadence = decideCadence(123, 30);
    expect(cadence.stride).toBe(4);
    expect(cadence.label).toBe("every 4th frame");
  });

  test("names the 2nd and 3rd strides in English, not as ordinals of digits", () => {
    expect(decideCadence(40, 30).label).toBe("every 2nd frame");
    expect(decideCadence(70, 30).label).toBe("every 3rd frame");
  });

  test("counts a budget met exactly as keeping up", () => {
    expect(decideCadence(1000 / 30, 30).stride).toBe(1);
  });

  test("assumes nothing before the first measurement", () => {
    // No median yet: detect every frame and let the measurement correct it,
    // rather than guessing a stride the machine may not need.
    expect(decideCadence(null, 30).stride).toBe(1);
    expect(decideCadence(null, 30).measured).toBe(false);
  });

  test("caps the stride so a very slow machine still shows motion", () => {
    const cadence = decideCadence(100000, 30);
    expect(cadence.stride).toBe(cadence.maxStride);
  });
});
