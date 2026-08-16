/** Constant-velocity Kalman filter over bounding boxes in xyah space. Mirrors
 * `trafficlens.track.kalman`.
 *
 * State vector (8,): `[cx, cy, a, h, vx, vy, va, vh]` -- box centre x/y, aspect
 * ratio `a = w/h`, height `h`, and their per-frame velocities. Measurement
 * vector (4,): `[cx, cy, a, h]` -- the same box parameters as observed by the
 * detector.
 *
 * Design rules, all of which the multi-object tracker depends on:
 *
 * - `KalmanBoxFilter` holds NO per-track state. It carries only the constant
 *   transition/measurement matrices; `initiate`, `predict`, `update` and
 *   `gatingDistance` are pure functions of their `(mean, cov)` arguments and
 *   always return new arrays. The tracker owns one `(mean, cov)` pair per
 *   track.
 * - All noise magnitudes scale with the box height `h`: a taller box is a
 *   nearer object, which moves more pixels per frame, so its uncertainty in
 *   pixels is proportionally larger. Every scale factor is a named constant
 *   from the generated constants module -- this file contains no numeric
 *   tunables of its own.
 * - Deterministic: no randomness anywhere; the same input sequence produces
 *   bit-identical output on every run.
 * - The gain comes from a linear solve, never an explicit matrix inverse, and
 *   the gating distance from a Cholesky factor and a triangular solve -- the
 *   same numerical route numpy takes, so the two engines stay comparable.
 *
 * On float64 agreement with numpy: the transition matrix F is the identity
 * plus dt on the position->velocity diagonal, and the measurement matrix H
 * selects the first four state components. Every entry of `F P`, `(F P) F^T`
 * and `H P H^T` is therefore a sum of at most two non-zero terms, which is
 * exact in float64 regardless of the order a BLAS chooses to accumulate in.
 * The four-term sums in the gain product are where the two implementations can
 * differ, and only in the last bit. */

import {
  KALMAN_ASPECT_MEASUREMENT_STD,
  KALMAN_ASPECT_STD,
  KALMAN_ASPECT_VELOCITY_STD,
  KALMAN_INIT_POSITION_STD_FACTOR,
  KALMAN_INIT_VELOCITY_STD_FACTOR,
  KALMAN_STD_WEIGHT_POSITION,
  KALMAN_STD_WEIGHT_VELOCITY,
} from "../generated/constants";
import { Matrix, cholesky, luSolve, matmul } from "./numeric";

// State/measurement dimensions (structural, not tunable).
const NDIM = 4; // measured components: cx, cy, a, h
const DT = 1.0; // one frame per step; frame index is the filter's clock

/** Convert a corner-format box `[x1, y1, x2, y2]` to measurement format
 * `[cx, cy, a, h]` with `a = w/h`.
 *
 * Throws on a zero- or negative-area box (`x2 <= x1` or `y2 <= y1`). Fail fast
 * is deliberate: a detector emitting a degenerate box is an upstream bug, and
 * silently computing `a = w/h` with `h <= 0` would push NaN/inf (or a nonsense
 * negative aspect) into the filter and poison the tracker invisibly. */
export function xyxyToXyah(box: Float64Array): Float64Array {
  const x1 = box[0] as number;
  const y1 = box[1] as number;
  const x2 = box[2] as number;
  const y2 = box[3] as number;
  const w = x2 - x1;
  const h = y2 - y1;
  if (w <= 0.0 || h <= 0.0) {
    throw new Error(
      `degenerate box: width=${w}, height=${h} (from xyxy=[${x1}, ${y1}, ${x2}, ${y2}]); ` +
        "both must be strictly positive",
    );
  }
  return Float64Array.of(x1 + w / 2.0, y1 + h / 2.0, w / h, h);
}

/** Convert `[cx, cy, a, h]` back to `[x1, y1, x2, y2]`. Exact inverse of
 * `xyxyToXyah` up to float64 rounding. */
export function xyahToXyxy(box: Float64Array): Float64Array {
  const cx = box[0] as number;
  const cy = box[1] as number;
  const a = box[2] as number;
  const h = box[3] as number;
  const w = a * h;
  return Float64Array.of(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0);
}

/** Force exact symmetry: `(P + P.T) / 2`.
 *
 * Floating-point matrix products make `P[i][j]` and `P[j][i]` drift apart by a
 * few ULPs per step; left alone, that asymmetry compounds and is the classic
 * silent Kalman failure mode (a covariance that is no longer a valid
 * covariance, and Cholesky factorizations that start to disagree between
 * platforms). Averaging with the transpose restores exact `P == P.T` after
 * every step; the Python original does the same, and the two would otherwise
 * diverge bit by bit. */
function symmetrize(cov: Matrix): Matrix {
  const out = Matrix.zeros(cov.rows, cov.cols);
  for (let i = 0; i < cov.rows; i += 1) {
    for (let j = 0; j < cov.cols; j += 1) {
      out.set(i, j, (cov.get(i, j) + cov.get(j, i)) / 2.0);
    }
  }
  return out;
}

function diagonalFromStd(std: number[]): Matrix {
  const m = Matrix.zeros(std.length, std.length);
  for (let i = 0; i < std.length; i += 1) {
    const s = std[i] as number;
    m.set(i, i, s * s);
  }
  return m;
}

function asColumn(vector: Float64Array): Matrix {
  return new Matrix(vector.length, 1, Float64Array.from(vector));
}

/** Stateless constant-velocity Kalman filter for xyah boxes.
 *
 * Instances hold only the constant transition matrix `F` (identity plus dt on
 * the position->velocity diagonal) and measurement matrix `H` (selects the
 * first four state components). All per-track state flows through the method
 * arguments. */
export class KalmanBoxFilter {
  private readonly motionMat: Matrix;
  private readonly updateMat: Matrix;

  constructor() {
    this.motionMat = Matrix.zeros(2 * NDIM, 2 * NDIM);
    for (let i = 0; i < 2 * NDIM; i += 1) {
      this.motionMat.set(i, i, 1.0);
    }
    for (let i = 0; i < NDIM; i += 1) {
      this.motionMat.set(i, NDIM + i, DT);
    }
    this.updateMat = Matrix.zeros(NDIM, 2 * NDIM);
    for (let i = 0; i < NDIM; i += 1) {
      this.updateMat.set(i, i, 1.0);
    }
  }

  // -- noise profiles --------------------------------------------------------

  /** Per-component std of the position block [cx, cy, a, h] at box height `h`:
   * pixel components scale with h, aspect is absolute. */
  private static stdPosition(h: number): number[] {
    return [
      KALMAN_STD_WEIGHT_POSITION * h,
      KALMAN_STD_WEIGHT_POSITION * h,
      KALMAN_ASPECT_STD,
      KALMAN_STD_WEIGHT_POSITION * h,
    ];
  }

  /** Per-component std of the velocity block [vx, vy, va, vh]. */
  private static stdVelocity(h: number): number[] {
    return [
      KALMAN_STD_WEIGHT_VELOCITY * h,
      KALMAN_STD_WEIGHT_VELOCITY * h,
      KALMAN_ASPECT_VELOCITY_STD,
      KALMAN_STD_WEIGHT_VELOCITY * h,
    ];
  }

  // -- filter steps ----------------------------------------------------------

  /** Start a new track from an unassociated measurement.
   *
   * Returns `[mean, cov]`: mean (8,) with the measured box and zero
   * velocities; cov (8, 8) diagonal, with position variances inflated by
   * `KALMAN_INIT_POSITION_STD_FACTOR` (a single detection is less trustworthy
   * than a tracked state) and velocity variances inflated by
   * `KALMAN_INIT_VELOCITY_STD_FACTOR` (zero velocity is pure ignorance, so the
   * first measurements must dominate it). */
  initiate(boxXyah: Float64Array): [Float64Array, Matrix] {
    const mean = new Float64Array(2 * NDIM);
    mean.set(boxXyah.subarray(0, NDIM), 0);

    const h = boxXyah[3] as number;
    const std = [
      ...KalmanBoxFilter.stdPosition(h).map((s) => KALMAN_INIT_POSITION_STD_FACTOR * s),
      ...KalmanBoxFilter.stdVelocity(h).map((s) => KALMAN_INIT_VELOCITY_STD_FACTOR * s),
    ];
    return [mean, diagonalFromStd(std)];
  }

  /** One constant-velocity step: `x' = F x`, `P' = F P F^T + Q`.
   *
   * The process noise `Q` is diagonal and re-derived each step from the
   * CURRENT height `mean[3]`, so a box growing as it approaches the camera
   * automatically gets a wider motion envelope. */
  predict(mean: Float64Array, cov: Matrix): [Float64Array, Matrix] {
    const h = mean[3] as number;
    const processNoise = diagonalFromStd([
      ...KalmanBoxFilter.stdPosition(h),
      ...KalmanBoxFilter.stdVelocity(h),
    ]);

    const newMean = matmul(this.motionMat, asColumn(mean));
    const propagated = matmul(
      matmul(this.motionMat, cov),
      this.motionMat.transpose(),
    );
    const newCov = Matrix.zeros(propagated.rows, propagated.cols);
    for (let i = 0; i < propagated.rows; i += 1) {
      for (let j = 0; j < propagated.cols; j += 1) {
        newCov.set(i, j, propagated.get(i, j) + processNoise.get(i, j));
      }
    }
    // Symmetrized every step (see symmetrize) so predict/update chains keep P
    // exactly symmetric no matter how they interleave.
    return [Float64Array.from(newMean.data), symmetrize(newCov)];
  }

  /** Project the state distribution into measurement space:
   * `(H x, H P H^T + R)` with height-scaled measurement noise R. */
  private project(mean: Float64Array, cov: Matrix): [Float64Array, Matrix] {
    const h = mean[3] as number;
    const measurementNoise = diagonalFromStd([
      KALMAN_STD_WEIGHT_POSITION * h,
      KALMAN_STD_WEIGHT_POSITION * h,
      KALMAN_ASPECT_MEASUREMENT_STD,
      KALMAN_STD_WEIGHT_POSITION * h,
    ]);

    const projMean = matmul(this.updateMat, asColumn(mean));
    const projected = matmul(matmul(this.updateMat, cov), this.updateMat.transpose());
    const projCov = Matrix.zeros(projected.rows, projected.cols);
    for (let i = 0; i < projected.rows; i += 1) {
      for (let j = 0; j < projected.cols; j += 1) {
        projCov.set(i, j, projected.get(i, j) + measurementNoise.get(i, j));
      }
    }
    return [Float64Array.from(projMean.data), symmetrize(projCov)];
  }

  /** Standard Kalman correction with an xyah measurement.
   *
   * The gain is computed by solving the linear system on the innovation
   * covariance `S` -- NEVER an explicit matrix inverse -- because solving
   * directly is better conditioned and is the numerical route numpy takes. */
  update(
    mean: Float64Array,
    cov: Matrix,
    measurementXyah: Float64Array,
  ): [Float64Array, Matrix] {
    const [projMean, projCov] = this.project(mean, cov);

    // Kalman gain K = P H^T S^{-1}, obtained by solving S K^T = (P H^T)^T.
    const b = matmul(cov, this.updateMat.transpose()); // (8, 4)
    const kalmanGain = luSolve(projCov, b.transpose()).transpose(); // (8, 4)

    const innovation = new Float64Array(NDIM);
    for (let i = 0; i < NDIM; i += 1) {
      innovation[i] = (measurementXyah[i] as number) - (projMean[i] as number);
    }

    const correction = matmul(kalmanGain, asColumn(innovation));
    const newMean = new Float64Array(2 * NDIM);
    for (let i = 0; i < 2 * NDIM; i += 1) {
      newMean[i] = (mean[i] as number) + correction.get(i, 0);
    }

    const reduction = matmul(matmul(kalmanGain, projCov), kalmanGain.transpose());
    const newCov = Matrix.zeros(cov.rows, cov.cols);
    for (let i = 0; i < cov.rows; i += 1) {
      for (let j = 0; j < cov.cols; j += 1) {
        newCov.set(i, j, cov.get(i, j) - reduction.get(i, j));
      }
    }
    // Enforce exact symmetry: the subtraction above loses a few ULPs of
    // P == P.T per step, and that drift compounding silently is the classic
    // Kalman covariance failure (see symmetrize).
    return [newMean, symmetrize(newCov)];
  }

  /** Squared Mahalanobis distance of N xyah measurements from the state's
   * predicted measurement distribution.
   *
   * `measurements` is (N, 4); returns (N,). Distances follow a chi-square
   * distribution with 4 degrees of freedom, so the tracker gates associations
   * at `KALMAN_GATING_CHI2_95_4DOF` (9.4877, the 95% quantile): any pair
   * scoring above it is rejected as having under a 5% chance of being the same
   * object.
   *
   * Computed via Cholesky (`S = L L^T`) and a triangular solve -- never by
   * inverting S -- so each distance is a plain sum of squares `||L^-1 d||^2`. */
  gatingDistance(
    mean: Float64Array,
    cov: Matrix,
    measurements: Matrix,
  ): Float64Array {
    const [projMean, projCov] = this.project(mean, cov);
    const n = measurements.rows;

    const d = Matrix.zeros(n, NDIM); // (N, 4)
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < NDIM; j += 1) {
        d.set(i, j, measurements.get(i, j) - (projMean[j] as number));
      }
    }
    const chol = cholesky(projCov); // lower-triangular L
    // Solve L y = d^T column-wise; squared Mahalanobis is sum(y^2).
    const y = luSolve(chol, d.transpose()); // (4, N)
    const out = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
      let total = 0.0;
      for (let j = 0; j < NDIM; j += 1) {
        total += y.get(j, i) * y.get(j, i);
      }
      out[i] = total;
    }
    return out;
  }
}
