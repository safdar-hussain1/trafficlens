/** The pure half of the live readout: how long a bar is, and what the readout
 * says once the line has moved. */

import { describe, expect, test } from "vitest";

import { countingSinceText, share } from "./readout";

describe("share", () => {
  test("is the type's part of the total", () => {
    expect(share(3, 12)).toBe(0.25);
    expect(share(12, 12)).toBe(1);
  });

  test("is an empty bar, not a guess, when nothing has been counted", () => {
    expect(share(0, 0)).toBe(0);
    expect(share(5, 0)).toBe(0);
  });

  test("never draws past the end or backwards", () => {
    expect(share(13, 12)).toBe(1);
    expect(share(-1, 12)).toBe(0);
    expect(share(Number.NaN, 12)).toBe(0);
    expect(share(3, Number.NaN)).toBe(0);
  });
});

describe("countingSinceText", () => {
  test("says nothing while the counts cover the whole session", () => {
    expect(countingSinceText(null)).toBe("");
  });

  test("names the clip time the counts restarted from", () => {
    expect(countingSinceText(7.4)).toBe("Counting since 0:07, when the line moved.");
    expect(countingSinceText(75)).toBe("Counting since 1:15, when the line moved.");
  });
});
