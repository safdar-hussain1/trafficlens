// numeric.ts is the browser's stand-in for the numpy and CPython `math` the
// Python engine leans on, not a mirror of a Python module of ours -- so these
// tests are not a port. They pin each routine against the exact float64 its
// Python counterpart produces for the same inputs, because "agrees with the
// Python engine's arithmetic" is the only property that matters here.

import { describe, expect, it } from "vitest";

import {
  Matrix,
  cholesky,
  hypot,
  luSolve,
  matmul,
  roundHalfEven,
  sumFloats,
} from "./numeric";

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

// -- sumFloats ----------------------------------------------------------------

// [values, sum(values)] measured from CPython 3.12, chosen so that a plain
// left-to-right running total gives a DIFFERENT float64 on every one -- which
// is what makes them a guard rather than decoration.
const CPYTHON_SUM: readonly (readonly [readonly number[], number])[] = [
  [[0.004121026990720791, -0.1588043462271764, -1.119622652146996e-05, 0.05904334310133245, -16.04792576522008, -413.27586420497323, 1.6654336584406658e-06, -864.1364652107482, -0.05200067904598578], -1293.6079053669155],
  [[-0.3864744613300907, -0.0005437280364773009, 0.0025927070610279063, -1.323668156438947e-06, 9.282140726431408e-06, 61309.646838885266, 2.8904878306444518e-05, -2429158.8689934965, 52028.060252731244, -14081.261447609653, 0.547440013798483], -2329902.2602980947],
  [[3.436365562599506e-05, -3.496186366989619e-05, -1291550.4050773086, -8.434545216917616e-06, -0.07925799635988136, 2.7373307065871515, 2.0990120752445947e-06, -0.0008537694941604889, 752955.9443522951], -538591.8035130065],
  [[43870.059210505875, 0.00018908717829615868, 539242.1635616167, -2.688402163295934e-05, 2.4860596828881583e-05, -68.55120935222325, -3.2775251810425856e-06, -527716.7490814854, 12437.772599861144], 67764.69526493226],
  [[10.889019722990401, 7.147994255032973e-05, 185.2148509910187, 0.17583488789815457, -1.8159582984649097e-06, 0.00015045562139285738, -105734.07334272462, 11508.655444206122], -94029.13797279698],
  [[0.0001225807764542557, 0.0007778978992195526, -50657.68740026214, -20721.888986133803, 1298.3661561079518, 1232465.3381556799, -1.463236707796551, -40.598099253647995], 1162342.067489909],
];

function runningTotal(values: readonly number[]): number {
  let total = 0.0;
  for (const v of values) {
    total += v;
  }
  return total;
}

describe("sumFloats", () => {
  it("reproduces CPython's builtin sum exactly", () => {
    for (const [values, want] of CPYTHON_SUM) {
      expect(sumFloats(values), JSON.stringify(values)).toBe(want);
    }
  });

  it("is not a plain running total", () => {
    // The discriminating half. Every fixture above is one where the naive loop
    // lands on a different float64, so an implementation that quietly dropped
    // the compensation term could not pass the assertion above by luck.
    for (const [values, want] of CPYTHON_SUM) {
      expect(runningTotal(values)).not.toBe(want);
    }
  });

  it("handles the trivial cases", () => {
    // Values measured from CPython, not expressions restating the algorithm --
    // an assertion built out of the implementation's own arithmetic cannot
    // disagree with it. sum([0.1, 0.2, 0.3]) is the interesting one: CPython
    // gives 0.6 where a running total gives 0.6000000000000001.
    expect(sumFloats([])).toBe(0.0);
    expect(sumFloats([1.5])).toBe(1.5);
    expect(sumFloats([0.1, 0.2])).toBe(0.30000000000000004);
    expect(sumFloats([0.1, 0.2, 0.3])).toBe(0.6);
    expect(runningTotal([0.1, 0.2, 0.3])).toBe(0.6000000000000001);
  });

  it("leaves a non-finite total unfolded, as CPython does", () => {
    // CPython does not fold the compensation term into a non-finite total.
    // Folding it turns an overflowed +-inf into NaN -- a different answer from
    // Python's on 11 of these 14 measured cases, and the one direction in which
    // the compensated version was WORSE than the plain running total it
    // replaced. Values below are CPython's actual output.
    expect(sumFloats([1e308, 1e308])).toBe(Infinity);
    expect(sumFloats([1e308, 1e308, -1e308])).toBe(Infinity);
    expect(sumFloats([1e308, 1e308, 1.0])).toBe(Infinity);
    expect(sumFloats([1e308, 1e308, 0.1, 0.2])).toBe(Infinity);
    expect(sumFloats([5e-324, 1e308, 1e308])).toBe(Infinity);
    expect(sumFloats([-1e308, -1e308])).toBe(-Infinity);
    expect(sumFloats([-1e308, -1e308, 1e308])).toBe(-Infinity);
    expect(sumFloats([Infinity, 1.0])).toBe(Infinity);
    expect(sumFloats([1.0, Infinity])).toBe(Infinity);
    expect(sumFloats([Infinity, Infinity])).toBe(Infinity);
    // These three are NaN in CPython too, so they pin that the guard did not
    // over-correct into returning Infinity for a genuine NaN.
    expect(sumFloats([Infinity, -Infinity])).toBeNaN();
    expect(sumFloats([NaN, 1.0])).toBeNaN();
    expect(sumFloats([1.0, NaN])).toBeNaN();
  });
});

// -- CPython's round -----------------------------------------------------------

describe("roundHalfEven", () => {
  // Every expectation below is CPython 3.12's actual `round(x)` output, taken
  // from the interpreter rather than reasoned about.
  it("rounds halfway cases to even, where Math.round rounds up", () => {
    for (const [x, want] of [
      [0.5, 0],
      [1.5, 2],
      [2.5, 2],
      [34.5, 34],
      [35.5, 36],
      // The measured letterbox case: 1280x717 at size 640 gives 717 * 0.5.
      [358.5, 358],
      [359.5, 360],
    ] as const) {
      expect(roundHalfEven(x), `round(${x})`).toBe(want);
    }
    // The reason this helper exists at all: on exact halves the two disagree,
    // and on 358.5 that disagreement is a one-pixel letterbox.
    expect(Math.round(358.5)).toBe(359);
    expect(Math.round(2.5)).toBe(3);
  });

  it("rounds halves toward even for negative values too", () => {
    for (const [x, want] of [
      [-0.5, 0],
      [-1.5, -2],
      [-2.5, -2],
    ] as const) {
      expect(roundHalfEven(x), `round(${x})`).toBe(want);
    }
  });

  // The control: away from an exact half, half-to-even must agree with
  // ordinary rounding. A helper that always rounded down would satisfy several
  // of the cases above and fail every one of these.
  it("agrees with ordinary rounding away from the halfway point", () => {
    for (const [x, want] of [
      [123.456, 123],
      [-123.456, -123],
      [2.675, 3],
      [0.0, 0],
      [-0.0, 0],
      // The largest double below 0.5. `Math.floor(x + 0.5)` gets this wrong:
      // x + 0.5 rounds up to exactly 1.0 in float64, so that shortcut returns
      // 1 where CPython returns 0.
      [0.49999999999999994, 0],
      [-0.49999999999999994, 0],
      [1000000000000000.5, 1000000000000000],
    ] as const) {
      expect(roundHalfEven(x), `round(${x})`).toBe(want);
    }
  });
});
