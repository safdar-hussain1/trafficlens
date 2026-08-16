// decodeYolo mirrors `trafficlens.detect.base.decode_yolo`, and the thing that
// makes that hard is not the algorithm -- it is the precision. numpy keeps the
// model's float32 through EVERY step under NEP 50, comparisons included,
// because the Python scalars it mixes in (`conf`, `iou`, `scale`, `pad_x`) are
// weak and adopt the array's dtype. JavaScript would do all of it in float64.
//
// That is the opposite of the rule the rest of `web/src/` follows: the engine
// modules (geometry, gate, kalman, tracker, speed) mirror Python code that is
// float64 throughout, and introducing `Math.fround` there would be the same
// class of bug in the other direction. Both halves follow the same underlying
// rule -- mirror the arithmetic the Python side actually performs, not the
// arithmetic its function names suggest. See the note in `postprocess.ts`.

import { describe, expect, it } from "vitest";

import { readFloat32, readManifest } from "./fixture-loader";
import { decodeYolo, iouOf } from "./postprocess";

const manifest = readManifest();
const { conf, iou, keepClassIds, nClasses } = manifest.decode;

const NAMES = new Map<number, string>([
  [1, "bicycle"],
  [2, "car"],
  [3, "motorcycle"],
  [5, "bus"],
  [7, "truck"],
]);

function runCase(name: "boundary" | "iouexact" | "real") {
  const spec = manifest.decode[name];
  const data = readFloat32(`decode_${name}_raw.bin`);
  const dims = [1, 4 + nClasses, spec.columns] as const;
  expect(data.length).toBe(dims[1] * dims[2]);
  return decodeYolo({ data, dims }, spec.scale, spec.padX, spec.padY, {
    conf,
    iou,
    keepClasses: NAMES,
  });
}

describe("decodeYolo", () => {
  it("keeps the class ids the fixture was generated for", () => {
    expect([...NAMES.keys()].sort((a, b) => a - b)).toEqual([...keepClassIds]);
  });

  for (const name of ["boundary", "iouexact", "real"] as const) {
    it(`reproduces Python's detections exactly on the ${name} fixture`, () => {
      const got = runCase(name);
      const want = manifest.decode[name].expected;
      expect(got.length).toBe(want.length);
      // Exact, not within 1e-5. Every operation `decode_yolo` performs is a
      // single float32 add, subtract, multiply, divide or compare, and each is
      // reproducible in JavaScript by rounding the float64 result once. If
      // this can only pass to a tolerance, the mirror is wrong somewhere and
      // Task 21's boundary fixtures will find it.
      expect(got.map((d) => ({ ...d }))).toEqual(want.map((d) => ({ ...d })));
    });
  }

  it("returns detections ordered by ascending class id, then NMS keep order", () => {
    const got = runCase("boundary");
    const ids = got.map((d) => d.classId);
    expect(ids).toEqual([...ids].sort((a, b) => a - b));
    for (let i = 1; i < got.length; i += 1) {
      const previous = got[i - 1] as (typeof got)[number];
      const current = got[i] as (typeof got)[number];
      if (previous.classId === current.classId) {
        expect(previous.score).toBeGreaterThanOrEqual(current.score);
      }
    }
  });

  // The two halves of the suppression decision, on the same axis (how much two
  // same-class boxes overlap) and differing only in which side of the
  // threshold they land. A `decodeYolo` that suppressed everything would pass
  // the first and fail the second; one that suppressed nothing, the reverse.
  it("suppresses a same-class duplicate above the IoU threshold", () => {
    const got = runCase("boundary").filter((d) => d.classId === 2);
    // Columns 0 and 1 overlap at IoU 0.818; only the 0.90 survives.
    expect(got.some((d) => Math.abs(d.score - 0.9) < 1e-6)).toBe(true);
    expect(got.some((d) => Math.abs(d.score - 0.8) < 1e-6)).toBe(false);
  });

  it("keeps a same-class pair whose overlap is below the threshold", () => {
    const got = runCase("boundary").filter((d) => d.classId === 7);
    // Columns 13 and 14 overlap at IoU 0.4815 -- just below 0.5 -- so both
    // survive, alongside the cross-class column 3.
    expect(got.some((d) => Math.abs(d.score - 0.75) < 1e-6)).toBe(true);
    expect(got.some((d) => Math.abs(d.score - 0.65) < 1e-6)).toBe(true);
  });

  it("never lets one class suppress another", () => {
    const got = runCase("boundary");
    // Columns 2 and 3 are the SAME box at the same score, one car one truck.
    const car = got.find((d) => d.classId === 2 && Math.abs(d.score - 0.7) < 1e-6);
    const truck = got.find((d) => d.classId === 7 && Math.abs(d.score - 0.7) < 1e-6);
    expect(car).toBeDefined();
    expect(truck).toBeDefined();
    expect(car?.x1).toBe(truck?.x1);
    expect(car?.y1).toBe(truck?.y1);
  });

  it("keeps a score exactly equal to conf, and drops one ULP below it", () => {
    const got = runCase("boundary").filter((d) => d.classId === 5);
    expect(got.length).toBe(1);
    // float32(0.35) === 0.3499999940395355, which a float64 comparison against
    // 0.35 rejects. That this survives is the proof `conf` is rounded to
    // float32 before the comparison, exactly as NEP 50 does it.
    expect(got[0]?.score).toBe(Math.fround(conf));
    expect(got[0]?.score).toBeLessThan(conf);
  });

  it("computes the suppression IoU in float32, not float64", () => {
    // Columns 15 and 16: one box nested in the other with exactly half its
    // area, so exact arithmetic gives 0.5 and `nms`'s strict `>` would keep
    // both. In the float32 the decode actually runs in, undoing the pad and
    // scale lands the ratio 1.8e-7 ABOVE 0.5, so the nested box is suppressed.
    // A float64 mirror returns two here and disagrees with Python by a whole
    // detection.
    expect(runCase("boundary").filter((d) => d.classId === 1).length).toBe(1);
    expect(iouOf([2080, 466.66666, 2186.6667, 573.3333], [2080, 493.33334, 2186.6667, 546.6667]))
      .toBe(0.5000001788139343);
  });

  // The strict-vs-inclusive half of the suppression rule, which the boundary
  // fixture cannot test: there the pad and scale do not divide evenly in
  // float32, so its "exactly 0.5" pair actually lands at 0.5000001788 and `>`
  // and `>=` agree. Here scale is 0.5 and the pad an integer, so undoing the
  // letterbox is exact and the nested box's IoU is exactly 0.5.
  it("does not suppress an overlap sitting exactly ON the threshold", () => {
    const got = runCase("iouexact");
    expect(got.filter((d) => d.classId === 2).length).toBe(2);
  });

  // Its control, same axis (where the overlap sits relative to the threshold),
  // one step clear of it: at 0.6 both a `>` and a `>=` rule suppress, so this
  // half passing proves the test above is about the boundary and not about
  // suppression having stopped working altogether.
  it("still suppresses an overlap clear of the threshold", () => {
    const got = runCase("iouexact");
    expect(got.filter((d) => d.classId === 7).length).toBe(1);
  });

  it("drops a class outside the keep set even when it outscores everything", () => {
    const got = runCase("boundary");
    expect(got.every((d) => keepClassIds.includes(d.classId))).toBe(true);
    expect(got.every((d) => d.score <= 0.9)).toBe(true); // the 0.99 person is gone
  });

  it("suppresses something on the real model output", () => {
    // The must-not-be-vacuous check for the realistic fixture: 96 raw columns
    // drawn from one frame's highest-scoring predictions collapse to a handful
    // of boxes, so NMS is doing work rather than the conf filter doing it all.
    const got = runCase("real");
    expect(got.length).toBeGreaterThan(0);
    expect(got.length).toBeLessThan(manifest.decode.real.columns);
  });
});
