/** One name and one colour per vehicle type, wherever it is drawn. */

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

  test("every type a source counts resolves, and to the same colour on every source", () => {
    const seen = new Map<string, string>();
    for (const source of SOURCES) {
      for (const [, name] of source.classes) {
        const token = colourToken(name);
        expect(token).toMatch(/^--class-/);
        expect(seen.get(name) ?? token).toBe(token);
        seen.set(name, token);
      }
    }
    expect(seen.size).toBeGreaterThan(3);
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
