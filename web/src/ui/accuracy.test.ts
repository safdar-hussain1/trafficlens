/** The tally on the demo page: what it counts and where its figures come from.
 *
 * Nothing in it is typed. Each figure is derived from the bake of
 * `reports/counting_accuracy.json`, and these tests pin the derivation: the
 * squares add up to the labelled crossings, the extra counts are the
 * predictions that matched nothing, and the rates are the report's own. */

import { readFileSync } from "node:fs";

import { describe, expect, test } from "vitest";

import { REPORTS } from "../generated/reports";
import { SHIPPED_METHOD, methodByName, modelName, tally, units } from "./accuracy";

describe("tally", () => {
  const value = tally();
  const method = methodByName(SHIPPED_METHOD);

  test("covers every labelled crossing, once", () => {
    expect(value.labelled).toBe(REPORTS.counting.labels.total);
    expect(value.counted + value.missed).toBe(value.labelled);
    expect(value.counted).toBe(method.truePositives);
  });

  test("calls the predictions that matched nothing extra counts", () => {
    expect(value.extra).toBe(method.nPredicted - method.truePositives);
    expect(value.extra).toBeGreaterThanOrEqual(0);
  });

  test("carries the report's own rates, and they agree with the squares", () => {
    expect(value.precision).toBe(method.precision);
    expect(value.recall).toBe(method.recall);
    expect(value.f1).toBe(method.f1);
    // The squares and the rates are two views of one confusion: if the bake's
    // rates stopped agreeing with its counts, the tally would contradict itself.
    expect(value.recall).toBeCloseTo(value.counted / value.labelled, 12);
    expect(value.precision).toBeCloseTo(value.counted / (value.counted + value.extra), 12);
  });

  test("names the detector the benchmark actually ran", () => {
    expect(value.detector).toBe(modelName(REPORTS.counting.detector.model));
    expect(value.detector).toMatch(/^YOLO/);
  });
});

describe("units", () => {
  test("draws one square per labelled crossing, counted ones first, then the extras", () => {
    const value = tally();
    const drawn = units(value);
    expect(drawn.labelled).toHaveLength(value.labelled);
    expect(drawn.labelled.filter((kind) => kind === "hit")).toHaveLength(value.counted);
    expect(drawn.labelled.indexOf("miss")).toBe(value.missed === 0 ? -1 : value.counted);
    expect(drawn.extra).toHaveLength(value.extra);
    expect(new Set(drawn.extra)).toEqual(new Set(value.extra === 0 ? [] : ["false"]));
  });
});

describe("addressing the bake by name", () => {
  test("a method the bake does not have throws instead of returning another one", () => {
    expect(() => methodByName("engine+gates")).toThrow(/no counting method named/);
    expect(() => methodByName("")).toThrow(/no counting method named/);
  });

  test("and the shipped method resolves", () => {
    // The control: a lookup that threw on everything would pass the test above.
    expect(methodByName(SHIPPED_METHOD).method).toBe(SHIPPED_METHOD);
  });
});

describe("modelName", () => {
  test("names a weights file the way prose does", () => {
    expect(modelName("yolo11s.pt")).toBe("YOLO11s");
    expect(modelName("yolo11n-480.onnx")).toBe("YOLO11n-480");
  });
});

describe("the tally and the page are two halves of one figure", () => {
  // `main.ts` mounts the tally inside a guard, so a renamed slot would leave an
  // empty figure on the published page and only a console error behind it.
  // This is what notices instead.
  const html = readFileSync(new URL("../../index.html", import.meta.url), "utf8");
  const source = readFileSync(new URL("./accuracy.ts", import.meta.url), "utf8");

  test("the demo page has exactly one slot for it, on a figure", () => {
    const slots = [...html.matchAll(/<(\w+)[^>]*\bdata-accuracy="([^"]+)"/g)];
    expect(slots.map((match) => match[2])).toEqual(["tally"]);
    expect(slots[0]?.[1]).toBe("figure");
  });

  test("and the module fills that slot by the same name", () => {
    expect(source).toContain(`'[data-accuracy="tally"]'`);
  });
});
