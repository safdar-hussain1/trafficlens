// Ported from tests/test_homography.py, restricted to the parts this mirror
// implements: the refusal-to-guess sentinel, the pixel -> metre projection and
// the reprojection-error report. Fitting a homography from correspondences
// stays on the Python side (see homography.ts for why).

import { describe, expect, it } from "vitest";

import { NO_CALIBRATION, RoadPlane } from "./homography";
import type { Point } from "./geometry";

// The exact image -> world (metres) homography of a genuine perspective camera
// looking down a road, as derived in tests/test_homography.py: fx = fy =
// 1000px, principal point (960, 540), camera 8m above the road and 5m behind
// the world origin, tilted 35 degrees down, world X = lateral metres, world
// Y = metres down the road.
const H_TRUE: readonly (readonly number[])[] = [
  [0.007874647112844963, 0.0, -7.559661228331164],
  [0.0, 0.008548295328846017, -0.9884912148527821],
  [0.0, -0.0008063166600676697, 1.0],
];

// A trapezoid marked on the road (two points 5m out, two 30m out, straddling
// the centre line at +-3.5m) and its exact projection through the inverse of
// H_TRUE -- so H_TRUE maps IMG4 back onto WORLD4.
const IMG4: readonly Point[] = [
  [686.1374124964, 476.0372807540398],
  [1233.8625875036003, 476.0372807540398],
  [854.7651148131101, 946.5662269242592],
  [1065.2348851868899, 946.5662269242592],
];

// The float64 values Python's RoadPlane.to_world produces for IMG4 through
// H_TRUE, transcribed from a measured run. Asserted exactly: the projective
// divide is on the speed path, and Task 21 compares speeds across the two
// engines, so a change in evaluation order here must be visible.
const WORLD4_PROJECTED: readonly Point[] = [
  [-3.5, 5.000000000000002],
  [3.5000000000000044, 5.000000000000002],
  [-3.499999999999998, 30.000000000000014],
  [3.499999999999998, 30.000000000000014],
];

// Two further road points, never used to build any plane here.
const IMG_HOLDOUT: readonly Point[] = [
  [878.9810899563789, 712.7097963942266],
  [999.8924299063265, 850.6102205009837],
];
const WORLD_HOLDOUT: readonly Point[] = [
  [-1.5, 12.0],
  [1.0, 20.0],
];

function plane(): RoadPlane {
  return new RoadPlane(H_TRUE);
}

// --- The refusal policy -------------------------------------------------------

describe("the refusal policy", () => {
  it("has a NO_CALIBRATION sentinel that is null", () => {
    expect(NO_CALIBRATION).toBeNull();
  });
});

// --- toWorld ------------------------------------------------------------------

describe("toWorld", () => {
  it("reproduces Python's float64 output for the four fit points", () => {
    const p = plane();
    IMG4.forEach((imgPt, i) => {
      expect(p.toWorld(imgPt)).toEqual(WORLD4_PROJECTED[i]);
    });
  });

  it("maps the surveyed trapezoid back to its metre positions", () => {
    const p = plane();
    const world: readonly Point[] = [
      [-3.5, 5.0],
      [3.5, 5.0],
      [-3.5, 30.0],
      [3.5, 30.0],
    ];
    IMG4.forEach((imgPt, i) => {
      const [wx, wy] = p.toWorld(imgPt);
      const expected = world[i] as Point;
      expect(Math.abs(wx - expected[0])).toBeLessThan(1e-5);
      expect(Math.abs(wy - expected[1])).toBeLessThan(1e-5);
    });
  });
});

// --- reprojectionError ----------------------------------------------------------

describe("reprojectionError", () => {
  it("reports metres, a mean and a max", () => {
    const err = plane().reprojectionError(IMG_HOLDOUT, WORLD_HOLDOUT);
    expect(Object.keys(err).sort()).toEqual(["maxM", "meanM", "perPointM"]);
    expect(err.perPointM.length).toBe(2);
    expect(err.maxM).toBeGreaterThanOrEqual(err.meanM);
    expect(err.meanM).toBeGreaterThanOrEqual(0.0);
    expect(err.maxM).toBe(Math.max(...err.perPointM));
  });

  it("throws on mismatched lengths", () => {
    expect(() =>
      plane().reprojectionError(IMG_HOLDOUT, WORLD_HOLDOUT.slice(0, 1)),
    ).toThrow();
  });

  it("measures a clean plane against held-out points as near zero", () => {
    // The held-out points are the exact projection of their world positions
    // through this very homography, so the residual is pure float64 noise.
    expect(plane().reprojectionError(IMG_HOLDOUT, WORLD_HOLDOUT).meanM).toBeLessThan(
      0.05,
    );
  });

  it("reproduces Python's float64 mean", () => {
    // Eight surveyed points, each displaced from where the plane says it is by
    // a different order of magnitude (1m down to 1e-7m), so the per-point
    // errors span enough range that CPython's compensated `sum` and a plain
    // running total land on different float64 values. Measured from Python:
    // the running total gives 0.13888888749998696, `sum` gives the value
    // pinned here. Without that distinction this test would pass either way.
    const img: readonly Point[] = [
      [927.0654028632011, 916.8047424064658],
      [847.5387606892157, 880.3871691778354],
      [903.4043334018467, 778.3563749084351],
      [806.3159786897583, 788.3692233720836],
      [961.0205814001961, 553.4490863855941],
      [980.7054220923386, 904.4636447542986],
      [913.5163414322337, 940.2349990449602],
      [916.8914760672395, 936.4175928781556],
    ];
    const world: readonly Point[] = [
      [-0.39456701790941673, 25.46358924454293],
      [-2.992408034389285, 22.452441592454434],
      [-1.1907587328036624, 15.20455370930025],
      [-3.3211820048153955, 15.783803056715708],
      [0.014573397958920148, 6.758548076443353],
      [0.6022901489032576, 24.90850776878006],
      [-1.5133664390015866, 29.143059546827455],
      [-1.3858467432888557, 28.643630499488697],
    ];
    const err = plane().reprojectionError(img, world);
    expect(err.meanM).toBe(0.1388888874999869);
    expect(err.maxM).toBe(0.999999999999989);
  });

  it("detects a corrupted correspondence", () => {
    // Move one surveyed world point 3m off where the plane says it is: the
    // measured error must rise well past the noise floor above.
    const corrupted: readonly Point[] = [
      [-1.5, 15.0],
      [1.0, 20.0],
    ];
    const err = plane().reprojectionError(IMG_HOLDOUT, corrupted);
    expect(err.maxM).toBeCloseTo(3.0, 9);
    expect(err.meanM).toBeGreaterThan(1.0);
  });
});
