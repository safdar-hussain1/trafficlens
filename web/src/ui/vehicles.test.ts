/** One name and one colour per vehicle type, wherever it is drawn. */

import { readFileSync } from "node:fs";

import { describe, expect, test } from "vitest";

import { SOURCES } from "./sources";
import { COLOURED_TYPES, colourToken, displayName, displayOrder } from "./vehicles";

describe("colourToken", () => {
  test("gives car, truck and bus their own hue and folds every other type into one", () => {
    const own = COLOURED_TYPES.map(colourToken);
    expect(new Set(own).size).toBe(COLOURED_TYPES.length);
    expect(own).not.toContain("--class-other");
    for (const type of ["person", "bicycle", "motorcycle", "anything"]) {
      expect(colourToken(type)).toBe("--class-other");
    }
  });

  test("never gives more types a hue of their own than stay distinguishable", () => {
    // Three is the most the chart palette keeps apart in every pairing, and
    // boxes on a video can sit beside any other box. A fourth hue here has to
    // be re-validated all-pairs first, not just added.
    expect(COLOURED_TYPES.length).toBeLessThanOrEqual(3);
  });

  test("every type a source counts is drawn in a colour the stylesheet defines", () => {
    // The canvas reads these tokens off the document, and the readout's bars
    // and the feed pick their colour by `data-class` in the stylesheet. Two
    // spellings of one mapping, so they are checked against each other: a type
    // whose rule or token went missing would be one colour on the video and
    // another beside it.
    const css = readFileSync(new URL("./styles.css", import.meta.url), "utf8");
    const defined = (token: string): boolean =>
      new RegExp(`${token}:\\s*#[0-9a-f]{6};`).test(css);
    expect(defined("--class-other")).toBe(true);
    for (const type of COLOURED_TYPES) {
      const token = colourToken(type);
      expect(defined(token), token).toBe(true);
      const rule = new RegExp(`\\[data-class="${type}"\\]\\s*\\{\\s*--swatch:\\s*var\\(${token}\\);`);
      expect(css, `the [data-class="${type}"] rule`).toMatch(rule);
    }
    // And every type a source can count resolves to one of those tokens.
    for (const source of SOURCES) {
      for (const [, name] of source.classes) {
        expect(defined(colourToken(name)), name).toBe(true);
      }
    }
  });
});

describe("displayName", () => {
  test("is the detector's class, capitalised", () => {
    expect(displayName("car")).toBe("Car");
    expect(displayName("motorcycle")).toBe("Motorcycle");
    expect(displayName("")).toBe("");
  });
});

describe("displayOrder", () => {
  const name = (type: string): string => type;

  test("lists the coloured types first, in palette order, then the rest as given", () => {
    expect(displayOrder(["car", "motorcycle", "bus", "truck"], name)).toEqual([
      "car", "truck", "bus", "motorcycle",
    ]);
    expect(displayOrder(["person", "bicycle", "car"], name)).toEqual(["car", "person", "bicycle"]);
  });

  test("keeps every type exactly once and leaves its input alone", () => {
    const input = ["person", "bicycle", "car", "motorcycle", "bus", "truck"];
    const before = [...input];
    const ordered = displayOrder(input, name);
    expect([...ordered].sort()).toEqual([...input].sort());
    expect(input).toEqual(before);
  });
});
