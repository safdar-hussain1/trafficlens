// numeric.ts is the browser's stand-in for the numpy and CPython `math` the
// Python engine leans on, not a mirror of a Python module of ours -- so these
// tests are not a port. They pin each routine against the exact float64 its
// Python counterpart produces for the same inputs, because "agrees with the
// Python engine's arithmetic" is the only property that matters here.

import { describe, expect, it } from "vitest";

import { Matrix, cholesky, hypot, luSolve, matmul } from "./numeric";

// A symmetric positive-definite 4x4, the shape of the filter's innovation
// covariance, and a two-column right-hand side, the shape of its gain solve.
const SPD = Matrix.from([
  [4.0, 1.0, 2.0, 0.5],
  [1.0, 3.0, 0.25, 1.5],
  [2.0, 0.25, 5.0, 1.0],
  [0.5, 1.5, 1.0, 2.5],
]);

describe("matmul", () => {
  it("multiplies rectangular matrices", () => {
    const a = Matrix.from([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ]);
    const b = Matrix.from([
      [7.0, 8.0],
      [9.0, 10.0],
      [11.0, 12.0],
    ]);
    expect(matmul(a, b).toArrays()).toEqual([
      [58.0, 64.0],
      [139.0, 154.0],
    ]);
  });

  it("rejects a shape mismatch", () => {
    expect(() => matmul(Matrix.zeros(2, 3), Matrix.zeros(2, 3))).toThrow();
  });
});

// The two answers to `SPD @ x = RHS` below are both correct and they are not
// the same float64. LAPACK's own recipe -- one multiply, then one subtract, per
// term -- gives REFERENCE_SOLUTION; numpy on this machine is backed by Apple's
// Accelerate, whose kernels fuse the multiply and the add and so keep an extra
// few bits per term, giving ACCELERATE_SOLUTION. The gap is 4.4e-16, three
// ULPs at this magnitude, and it is a property of whichever BLAS numpy was
// built against rather than of either implementation being wrong -- a Linux
// box with OpenBLAS is a third answer again.
//
// So the exact assertion is against the reference recipe, which is what this
// module deliberately implements and what a change in evaluation order here
// would break. The agreement with numpy is asserted separately, as a measured
// bound, because that is all it can honestly be.
const RHS = Matrix.from([
  [1.0, 2.0],
  [3.0, 4.0],
  [5.0, 6.0],
  [7.0, 8.0],
]);
const REFERENCE_SOLUTION = [
  [-0.29534510433386835, -0.16051364365971105],
  [-0.3788121990369182, -0.2182985553772074],
  [0.5650080256821829, 0.6548956661316212],
  [2.8603531300160516, 3.1011235955056184],
];
const ACCELERATE_SOLUTION = [
  [-0.29534510433386835, -0.16051364365971107],
  [-0.37881219903691804, -0.2182985553772071],
  [0.5650080256821829, 0.6548956661316212],
  [2.860353130016051, 3.101123595505618],
];

describe("luSolve", () => {
  it("reproduces LAPACK's own float64 solution exactly", () => {
    expect(luSolve(SPD, RHS).toArrays()).toEqual(REFERENCE_SOLUTION);
  });

  it("agrees with numpy to within a few ULPs", () => {
    const got = luSolve(SPD, RHS).toArrays();
    got.forEach((row, i) => {
      row.forEach((value, j) => {
        const want = (ACCELERATE_SOLUTION[i] as number[])[j] as number;
        expect(Math.abs(value - want)).toBeLessThan(1e-15);
      });
    });
  });

  it("pivots when the leading entry is zero", () => {
    // A zero pivot in column 0: without partial pivoting this divides by zero.
    const a = Matrix.from([
      [0.0, 2.0, 1.0],
      [1.0, 1.0, 1.0],
      [2.0, 1.0, 0.0],
    ]);
    const b = Matrix.from([[1.0], [2.0], [3.0]]);
    expect(luSolve(a, b).toArrays()).toEqual([
      [1.3333333333333333],
      [0.33333333333333337],
      [0.3333333333333333],
    ]);
  });

  it("rejects a singular matrix", () => {
    const singular = Matrix.from([
      [1.0, 2.0],
      [2.0, 4.0],
    ]);
    expect(() => luSolve(singular, Matrix.from([[1.0], [1.0]]))).toThrow();
  });
});

describe("cholesky", () => {
  it("reproduces numpy's lower-triangular factor", () => {
    expect(cholesky(SPD).toArrays()).toEqual([
      [2.0, 0.0, 0.0, 0.0],
      [0.5, 1.6583123951777, 0.0, 0.0],
      [1.0, -0.15075567228888181, 1.9943100880436642, 0.0],
      [0.25, 0.82915619758885, 0.4387482193696061, 1.2479983974348685],
    ]);
  });

  it("rejects a matrix that is not positive definite", () => {
    const indefinite = Matrix.from([
      [1.0, 2.0],
      [2.0, 1.0],
    ]);
    expect(() => cholesky(indefinite)).toThrow();
  });
});

// -- hypot --------------------------------------------------------------------

// [a, b, math.hypot(a, b)] measured from CPython 3.12. The first six are
// constructed to land exactly on SPEED_MAX_STEP_M, which is the comparison the
// speed estimator's outlier rejection actually makes; the rest are ordinary
// values on which the naive `sqrt(a*a + b*b)` route already goes wrong.
const CPYTHON_HYPOT: readonly (readonly [number, number, number])[] = [
  [4.2, 5.6, 7.0],
  [4.949747468305833, 4.949747468305833, 7.0],
  [3.5, 6.06217782649107, 7.0],
  [2.1, 6.675327317569374, 6.997856442989376],
  [0.0, 7.0, 7.0],
  [7.0, 0.0, 7.0],
  [-22.89214196318857, -29.82100473027739, 37.594447552610816],
  [-52.348378114121296, 46.58960113973977, 70.07812515713483],
  [19.093587009262304, -22.626948717790306, 29.606483633788436],
  [23.128034150512313, -75.88264573485145, 79.32894734829294],
  [-78.60275556697864, 41.973273430692025, 89.10751295603434],
  [-37.399970713629926, 60.55678329833839, 71.1750083443781],
  [-43.40569959328, -28.68602901212494, 52.02829055106982],
  [-62.77254372214847, 16.735996609680193, 64.96526631876658],
  [-9.393620336212313, 22.04964539190909, 23.96720603094652],
  [-112.39351717600272, -83.88053400298942, 140.24352636688468],
  [16.361522596290435, 6.452453733272044, 17.587881647570264],
  [45.44095954721474, -57.612129462437935, 73.3760060630743],
  [-41.829855170679565, -21.66683178654871, 47.10826236730268],
  [2.749095668127679, 48.07187372907223, 48.15041610221435],
  [72.72989002511108, 49.90767607252016, 88.20664960208133],
  [-25.383524179244183, 63.275051456684494, 68.17664876337311],
  [39.05822020758882, 30.997933465706012, 49.86397943334283],
  [-56.9756276562355, 27.300135257323248, 63.17847364324444],
];

describe("hypot", () => {
  it("reproduces CPython's float64 exactly", () => {
    for (const [a, b, want] of CPYTHON_HYPOT) {
      expect(hypot(a, b), `hypot(${a}, ${b})`).toBe(want);
    }
  });

  it("is not simply Math.hypot", () => {
    // The discriminating half. If this module ever degrades into an alias for
    // the built-in, the assertion above would still pass on most of the
    // fixture, so the disagreement itself is pinned: on this input CPython
    // returns exactly 7.0 and V8 returns 7.000000000000001, which is the
    // difference between accepting and rejecting a sample sitting exactly on
    // SPEED_MAX_STEP_M.
    const a = 4.949747468305833;
    expect(Math.hypot(a, a)).not.toBe(7.0);
    expect(hypot(a, a)).toBe(7.0);

    const differing = CPYTHON_HYPOT.filter(([a2, b2, want]) => Math.hypot(a2, b2) !== want);
    expect(differing.length).toBeGreaterThan(2);
  });

  it("handles zeros, infinities and NaN as CPython does", () => {
    expect(hypot(0.0, 0.0)).toBe(0.0);
    expect(hypot(-0.0, -0.0)).toBe(0.0);
    expect(hypot(3.0, 0.0)).toBe(3.0);
    expect(hypot(-3.0, 0.0)).toBe(3.0);
    // Infinity wins over NaN: CPython's vector_norm checks the infinity first.
    expect(hypot(Infinity, NaN)).toBe(Infinity);
    expect(hypot(NaN, -Infinity)).toBe(Infinity);
    expect(hypot(NaN, 1.0)).toBeNaN();
  });

  it("stays exact through subnormals and extreme scales", () => {
    // The scaled path: 2**-maxExponent would overflow for subnormal inputs, so
    // those are lifted into the normal range first.
    expect(hypot(3e-323, 4e-323)).toBe(Math.hypot(3e-323, 4e-323));
    expect(hypot(3e300, 4e300)).toBe(5e300);
    expect(hypot(5e-324, 0.0)).toBe(5e-324);
    expect(hypot(1e300, 1e-300)).toBe(1e300);
  });
});
