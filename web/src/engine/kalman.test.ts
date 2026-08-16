// Ported from tests/test_kalman.py: convergence on a noiseless
// constant-velocity target, covariance growth/shrinkage, exact box-format
// round-trips, gating discrimination, run-to-run determinism, and exact
// covariance symmetry.

import { describe, expect, it } from "vitest";

import {
  KALMAN_GATING_CHI2_95_4DOF,
  KALMAN_STD_WEIGHT_VELOCITY,
} from "../generated/constants";
import { KalmanBoxFilter, xyahToXyxy, xyxyToXyah } from "./kalman";
import { Matrix } from "./numeric";

function kf(): KalmanBoxFilter {
  return new KalmanBoxFilter();
}

function trace(m: Matrix): number {
  let total = 0.0;
  for (let i = 0; i < m.rows; i += 1) {
    total += m.get(i, i);
  }
  return total;
}

function isSymmetric(m: Matrix): boolean {
  for (let i = 0; i < m.rows; i += 1) {
    for (let j = 0; j < m.cols; j += 1) {
      if (m.get(i, j) !== m.get(j, i)) {
        return false;
      }
    }
  }
  return true;
}

/** A target moving at exactly constant velocity, in xyah format.
 *
 * Centre starts at (100, 200) and moves (+5, +3) px/frame; the box shape
 * (aspect 0.5, height 40) never changes. All values are exact in float64, so
 * any residual prediction error is the filter's own, not the data's. */
function constantVelocityBoxes(nSteps: number): Float64Array[] {
  const boxes: Float64Array[] = [];
  for (let k = 0; k < nSteps; k += 1) {
    boxes.push(Float64Array.of(100.0 + 5.0 * k, 200.0 + 3.0 * k, 0.5, 40.0));
  }
  return boxes;
}

/** Initiate on boxes[0], then predict+update through the rest.
 *
 * Returns the per-step predicted means; the predicted mean at step k is
 * recorded BEFORE the update with boxes[k], i.e. it is the filter's genuine
 * one-step-ahead forecast. */
function track(
  filter: KalmanBoxFilter,
  boxes: Float64Array[],
): { predictedMeans: Float64Array[]; mean: Float64Array; cov: Matrix } {
  let [mean, cov] = filter.initiate(boxes[0] as Float64Array);
  const predictedMeans: Float64Array[] = [];
  for (let i = 1; i < boxes.length; i += 1) {
    [mean, cov] = filter.predict(mean, cov);
    predictedMeans.push(Float64Array.from(mean));
    [mean, cov] = filter.update(mean, cov, boxes[i] as Float64Array);
  }
  return { predictedMeans, mean, cov };
}

// ---------------------------------------------------------------------------
// Box format conversions
// ---------------------------------------------------------------------------

describe("box format conversions", () => {
  it("converts xyxy to xyah", () => {
    const xyah = xyxyToXyah(Float64Array.of(10.0, 20.0, 50.0, 100.0)); // w=40, h=80
    expect(xyah.length).toBe(4);
    expect(Array.from(xyah)).toEqual([30.0, 60.0, 0.5, 80.0]);
  });

  it("converts xyah to xyxy", () => {
    const box = xyahToXyxy(Float64Array.of(30.0, 60.0, 0.5, 80.0));
    expect(box.length).toBe(4);
    expect(Array.from(box)).toEqual([10.0, 20.0, 50.0, 100.0]);
  });

  it("round-trips xyxy -> xyah -> xyxy to 1e-9", () => {
    const boxes = [
      [10.0, 20.0, 50.0, 100.0],
      [0.0, 0.0, 1.0, 1.0],
      [123.4, 567.8, 234.5, 678.9],
      [-40.0, -80.0, -10.0, -20.0], // negative coords, positive area
      [3.7, 9.1, 1919.3, 1079.6],
    ];
    for (const box of boxes) {
      const back = xyahToXyxy(xyxyToXyah(Float64Array.from(box)));
      for (let i = 0; i < 4; i += 1) {
        expect(Math.abs((back[i] as number) - (box[i] as number))).toBeLessThanOrEqual(1e-9);
      }
    }
  });

  it("round-trips xyah -> xyxy -> xyah to 1e-9", () => {
    const xyahs = [
      [30.0, 60.0, 0.5, 80.0],
      [960.0, 540.0, 1.7777777777777777, 200.0],
      [5.5, 5.5, 3.0, 0.25],
    ];
    for (const xyah of xyahs) {
      const back = xyxyToXyah(xyahToXyxy(Float64Array.from(xyah)));
      for (let i = 0; i < 4; i += 1) {
        expect(Math.abs((back[i] as number) - (xyah[i] as number))).toBeLessThanOrEqual(1e-9);
      }
    }
  });

  it("throws on degenerate boxes", () => {
    // Zero width, zero height, negative width, negative height, fully
    // inverted box: all must fail fast rather than emit NaN/inf aspect ratios
    // that would silently poison the tracker downstream.
    const degenerate = [
      [10.0, 20.0, 10.0, 100.0], // w == 0
      [10.0, 20.0, 50.0, 20.0], // h == 0
      [50.0, 20.0, 10.0, 100.0], // w < 0
      [10.0, 100.0, 50.0, 20.0], // h < 0
      [50.0, 100.0, 10.0, 20.0], // both negative
      [10.0, 20.0, 10.0, 20.0], // zero area point box
    ];
    for (const box of degenerate) {
      expect(() => xyxyToXyah(Float64Array.from(box))).toThrow();
    }
  });
});

// ---------------------------------------------------------------------------
// Filter core behaviour
// ---------------------------------------------------------------------------

describe("filter core", () => {
  it("initiates with the measured box, zero velocity and an inflated velocity variance", () => {
    const [mean, cov] = kf().initiate(Float64Array.of(100.0, 200.0, 0.5, 40.0));
    expect(mean.length).toBe(8);
    expect(cov.rows).toBe(8);
    expect(cov.cols).toBe(8);
    expect(Array.from(mean.subarray(0, 4))).toEqual([100.0, 200.0, 0.5, 40.0]);
    expect(Array.from(mean.subarray(4))).toEqual([0.0, 0.0, 0.0, 0.0]);
    // Velocity variance starts high relative to the per-step velocity process
    // noise (well over an order of magnitude), so the zero initial velocity
    // is pure prior and the first measurements dominate it.
    const h = 40.0;
    const perStepVelVar = (KALMAN_STD_WEIGHT_VELOCITY * h) ** 2;
    expect(cov.get(4, 4)).toBeGreaterThan(10.0 * perStepVelVar);
    expect(cov.get(5, 5)).toBeGreaterThan(10.0 * perStepVelVar);
  });

  it("converges on constant velocity within half a pixel", () => {
    const nSteps = 20;
    const boxes = constantVelocityBoxes(nSteps + 1);
    const { predictedMeans } = track(kf(), boxes);
    // After 20 predict/update cycles the one-step-ahead predicted centre must
    // sit within 0.5 px of the true centre on both axes.
    const finalPred = predictedMeans[predictedMeans.length - 1] as Float64Array;
    const truth = boxes[nSteps] as Float64Array;
    expect(Math.abs((finalPred[0] as number) - (truth[0] as number))).toBeLessThan(0.5);
    expect(Math.abs((finalPred[1] as number) - (truth[1] as number))).toBeLessThan(0.5);
  });

  it("grows the covariance under predict", () => {
    const filter = kf();
    const [mean, cov] = filter.initiate(Float64Array.of(100.0, 200.0, 0.5, 40.0));
    const [, covPred] = filter.predict(mean, cov);
    // Process noise plus velocity coupling: total and positional uncertainty
    // must strictly grow with no measurement in between.
    expect(trace(covPred)).toBeGreaterThan(trace(cov));
    expect(covPred.get(0, 0)).toBeGreaterThan(cov.get(0, 0));
    expect(covPred.get(1, 1)).toBeGreaterThan(cov.get(1, 1));
  });

  it("shrinks the covariance under update", () => {
    const filter = kf();
    let [mean, cov] = filter.initiate(Float64Array.of(100.0, 200.0, 0.5, 40.0));
    [mean, cov] = filter.predict(mean, cov);
    const [, covUpd] = filter.update(
      mean,
      cov,
      Float64Array.of(105.0, 203.0, 0.5, 40.0),
    );
    expect(trace(covUpd)).toBeLessThan(trace(cov));
    expect(covUpd.get(0, 0)).toBeLessThan(cov.get(0, 0));
    expect(covUpd.get(1, 1)).toBeLessThan(cov.get(1, 1));
  });

  it("discriminates a true continuation from a far impostor", () => {
    const filter = kf();
    const boxes = constantVelocityBoxes(11);
    let [mean, cov] = filter.initiate(boxes[0] as Float64Array);
    for (let i = 1; i < boxes.length - 1; i += 1) {
      [mean, cov] = filter.predict(mean, cov);
      [mean, cov] = filter.update(mean, cov, boxes[i] as Float64Array);
    }
    [mean, cov] = filter.predict(mean, cov);

    const trueZ = boxes[boxes.length - 1] as Float64Array;
    const farZ = Float64Array.of(
      (trueZ[0] as number) + 200.0,
      trueZ[1] as number,
      trueZ[2] as number,
      trueZ[3] as number,
    );
    const d = filter.gatingDistance(
      mean,
      cov,
      Matrix.from([Array.from(trueZ), Array.from(farZ)]),
    );

    expect(d.length).toBe(2);
    // The true continuation passes the 95% chi-square gate for 4 DOF; a 200 px
    // impostor fails it by a huge margin.
    expect(d[0] as number).toBeLessThan(KALMAN_GATING_CHI2_95_4DOF);
    expect(d[1] as number).toBeGreaterThan(KALMAN_GATING_CHI2_95_4DOF);
    expect(d[1] as number).toBeGreaterThan(100.0 * (d[0] as number));
  });

  it("agrees between single and batch gating distances", () => {
    const filter = kf();
    let [mean, cov] = filter.initiate(Float64Array.of(100.0, 200.0, 0.5, 40.0));
    [mean, cov] = filter.predict(mean, cov);
    const zs = [
      [101.0, 201.0, 0.5, 40.0],
      [140.0, 260.0, 0.6, 44.0],
      [400.0, 900.0, 0.5, 40.0],
    ];
    const batch = filter.gatingDistance(mean, cov, Matrix.from(zs));
    const singles = zs.map(
      (z) => filter.gatingDistance(mean, cov, Matrix.from([z]))[0] as number,
    );
    expect(Array.from(batch)).toEqual(singles);
  });
});

// ---------------------------------------------------------------------------
// Determinism and numerical hygiene
// ---------------------------------------------------------------------------

describe("determinism and numerical hygiene", () => {
  it("is bit-identical over 20 steps", () => {
    const filter = kf();
    const boxes = constantVelocityBoxes(21);

    const run = (): { means: number[][]; covs: number[][] } => {
      let [mean, cov] = filter.initiate(boxes[0] as Float64Array);
      const means = [Array.from(mean)];
      const covs = [Array.from(cov.data)];
      for (let i = 1; i < boxes.length; i += 1) {
        [mean, cov] = filter.predict(mean, cov);
        [mean, cov] = filter.update(mean, cov, boxes[i] as Float64Array);
        means.push(Array.from(mean));
        covs.push(Array.from(cov.data));
      }
      return { means, covs };
    };

    const a = run();
    const b = run();
    // Exact equality, not approximate: the same input sequence must produce
    // bit-identical output, or the TypeScript parity story falls apart.
    expect(a.means).toEqual(b.means);
    expect(a.covs).toEqual(b.covs);
  });

  it("keeps covariances exactly symmetric", () => {
    const filter = kf();
    const boxes = constantVelocityBoxes(21);
    let [mean, cov] = filter.initiate(boxes[0] as Float64Array);
    expect(isSymmetric(cov)).toBe(true);
    for (let i = 1; i < boxes.length; i += 1) {
      [mean, cov] = filter.predict(mean, cov);
      expect(isSymmetric(cov)).toBe(true);
      [mean, cov] = filter.update(mean, cov, boxes[i] as Float64Array);
      expect(isSymmetric(cov)).toBe(true);
    }
  });
});
