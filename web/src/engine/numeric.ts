/** The float64 primitives the browser engine needs and JavaScript does not
 * supply, standing in for the numpy and CPython `math` the Python engine
 * imports.
 *
 * This is the one module here with no Python counterpart of its own:
 * `trafficlens.track.kalman` gets `matmul`, `np.linalg.solve` and
 * `np.linalg.cholesky` from numpy, which gets them from LAPACK, and
 * `trafficlens.analytics.speed` gets `hypot` from CPython's math module. The
 * browser has none of that, so the four routines are written out below -- and
 * written to those implementations' own recipes rather than to whatever a
 * textbook (or `Math.hypot`) does, because the two engines are compared
 * numerically later and every avoidable difference in evaluation order is one
 * more source of drift on that comparison. In particular:
 *
 * - `luSolve` mirrors `dgesv`: unblocked right-looking LU with partial
 *   pivoting (`dgetf2`), then the two `dtrsm` substitutions of `dgetrs`.
 *   `dgetf2` forms the reciprocal of the pivot ONCE and multiplies by it
 *   (`dscal`), which is not the same float64 as dividing each entry, so this
 *   does the same.
 * - `cholesky` mirrors `dpotf2`, including that same multiply-by-reciprocal.
 * - `matmul` accumulates each entry over k ascending, one addition at a time.
 * - `hypot` mirrors CPython's `vector_norm`, NOT `Math.hypot`. The two
 *   disagree by an ULP on about a quarter of ordinary inputs, and the speed
 *   estimator compares a hypot against a threshold: measured, `Math.hypot`
 *   returns 7.000000000000001 where CPython returns exactly 7.0 for
 *   `hypot(4.949747468305833, 4.949747468305833)`, which flips the
 *   `> SPEED_MAX_STEP_M` outlier decision on a boundary case.
 *
 * What this does NOT claim: numpy dispatches to a tuned BLAS whose kernels may
 * reassociate or fuse multiply-add, so agreement with numpy is close but not
 * guaranteed bit-for-bit on the matrix products. It is exact wherever the sums
 * involved have at most two non-zero terms, which covers the filter's
 * transition and measurement products (see kalman.ts). */

/** A dense row-major float64 matrix, carrying its shape the way a numpy array
 * does -- an empty matrix still knows how many columns it has, which
 * `associate.assign` relies on to report unmatched columns. */
export class Matrix {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;

  constructor(rows: number, cols: number, data?: Float64Array) {
    if (data !== undefined && data.length !== rows * cols) {
      throw new Error(
        `matrix data has ${data.length} entries, expected ${rows * cols} for ${rows}x${cols}`,
      );
    }
    this.rows = rows;
    this.cols = cols;
    this.data = data ?? new Float64Array(rows * cols);
  }

  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  static filled(rows: number, cols: number, value: number): Matrix {
    const m = new Matrix(rows, cols);
    m.data.fill(value);
    return m;
  }

  /** Build from nested arrays. `cols` is only needed for a matrix with no
   * rows, where the row arrays cannot supply the width. */
  static from(values: readonly (readonly number[])[], cols?: number): Matrix {
    const first = values[0];
    const nCols = cols ?? (first === undefined ? 0 : first.length);
    const m = new Matrix(values.length, nCols);
    for (let i = 0; i < values.length; i += 1) {
      const row = values[i] as readonly number[];
      if (row.length !== nCols) {
        throw new Error(`row ${i} has ${row.length} entries, expected ${nCols}`);
      }
      for (let j = 0; j < nCols; j += 1) {
        m.data[i * nCols + j] = row[j] as number;
      }
    }
    return m;
  }

  get(i: number, j: number): number {
    return this.data[i * this.cols + j] as number;
  }

  set(i: number, j: number, value: number): void {
    this.data[i * this.cols + j] = value;
  }

  clone(): Matrix {
    return new Matrix(this.rows, this.cols, Float64Array.from(this.data));
  }

  transpose(): Matrix {
    const out = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i += 1) {
      for (let j = 0; j < this.cols; j += 1) {
        out.set(j, i, this.get(i, j));
      }
    }
    return out;
  }

  toArrays(): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < this.rows; i += 1) {
      const row: number[] = [];
      for (let j = 0; j < this.cols; j += 1) {
        row.push(this.get(i, j));
      }
      out.push(row);
    }
    return out;
  }
}

/** Matrix product, each entry accumulated over k ascending. */
export function matmul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) {
    throw new Error(
      `shape mismatch: ${a.rows}x${a.cols} cannot multiply ${b.rows}x${b.cols}`,
    );
  }
  const out = new Matrix(a.rows, b.cols);
  for (let i = 0; i < a.rows; i += 1) {
    for (let j = 0; j < b.cols; j += 1) {
      let total = 0.0;
      for (let k = 0; k < a.cols; k += 1) {
        total += a.get(i, k) * b.get(k, j);
      }
      out.set(i, j, total);
    }
  }
  return out;
}

/** Solve `a @ x = b` for x, the way `np.linalg.solve` does: one LU
 * factorization with partial pivoting, then forward and back substitution.
 * Never forms an inverse. Throws on a singular `a`, as numpy raises. */
export function luSolve(a: Matrix, b: Matrix): Matrix {
  const n = a.rows;
  if (a.cols !== n) {
    throw new Error(`luSolve needs a square matrix, got ${a.rows}x${a.cols}`);
  }
  if (b.rows !== n) {
    throw new Error(`right-hand side has ${b.rows} rows, expected ${n}`);
  }

  const lu = a.clone();
  const pivots = new Int32Array(n);

  // dgetf2: right-looking unblocked LU with partial pivoting.
  for (let j = 0; j < n; j += 1) {
    let pivotRow = j;
    let pivotMag = Math.abs(lu.get(j, j));
    for (let i = j + 1; i < n; i += 1) {
      const mag = Math.abs(lu.get(i, j));
      if (mag > pivotMag) {
        pivotMag = mag;
        pivotRow = i;
      }
    }
    const pivot = lu.get(pivotRow, j);
    if (pivot === 0.0) {
      throw new Error("singular matrix");
    }
    pivots[j] = pivotRow;
    if (pivotRow !== j) {
      for (let k = 0; k < n; k += 1) {
        const tmp = lu.get(j, k);
        lu.set(j, k, lu.get(pivotRow, k));
        lu.set(pivotRow, k, tmp);
      }
    }
    // dscal: the reciprocal is formed once and multiplied through, which is
    // not bit-identical to dividing entry by entry.
    const inversePivot = 1.0 / lu.get(j, j);
    for (let i = j + 1; i < n; i += 1) {
      lu.set(i, j, lu.get(i, j) * inversePivot);
    }
    // dger: rank-1 update of the trailing submatrix.
    for (let i = j + 1; i < n; i += 1) {
      const factor = lu.get(i, j);
      for (let k = j + 1; k < n; k += 1) {
        lu.set(i, k, lu.get(i, k) - factor * lu.get(j, k));
      }
    }
  }

  // dgetrs: row interchanges, then the unit-lower and upper substitutions.
  const x = b.clone();
  for (let j = 0; j < n; j += 1) {
    const p = pivots[j] as number;
    if (p !== j) {
      for (let c = 0; c < x.cols; c += 1) {
        const tmp = x.get(j, c);
        x.set(j, c, x.get(p, c));
        x.set(p, c, tmp);
      }
    }
  }
  for (let c = 0; c < x.cols; c += 1) {
    for (let k = 0; k < n; k += 1) {
      const value = x.get(k, c);
      for (let i = k + 1; i < n; i += 1) {
        x.set(i, c, x.get(i, c) - value * lu.get(i, k));
      }
    }
    for (let k = n - 1; k >= 0; k -= 1) {
      const value = x.get(k, c) / lu.get(k, k);
      x.set(k, c, value);
      for (let i = 0; i < k; i += 1) {
        x.set(i, c, x.get(i, c) - value * lu.get(i, k));
      }
    }
  }
  return x;
}

/** Lower-triangular Cholesky factor L of a symmetric positive-definite `a`,
 * such that `a = L @ L.T`, mirroring `dpotf2` (and so `np.linalg.cholesky`).
 * Throws when `a` is not positive definite, as numpy raises. */
export function cholesky(a: Matrix): Matrix {
  const n = a.rows;
  if (a.cols !== n) {
    throw new Error(`cholesky needs a square matrix, got ${a.rows}x${a.cols}`);
  }
  const l = Matrix.zeros(n, n);
  for (let j = 0; j < n; j += 1) {
    let diagonal = a.get(j, j);
    for (let k = 0; k < j; k += 1) {
      diagonal -= l.get(j, k) * l.get(j, k);
    }
    if (!(diagonal > 0.0)) {
      throw new Error("matrix is not positive definite");
    }
    const root = Math.sqrt(diagonal);
    l.set(j, j, root);
    // dgemv then dscal: the trailing column is formed, then scaled by the
    // reciprocal of the diagonal rather than divided by it.
    const inverseRoot = 1.0 / root;
    for (let i = j + 1; i < n; i += 1) {
      let value = a.get(i, j);
      for (let k = 0; k < j; k += 1) {
        value -= l.get(i, k) * l.get(j, k);
      }
      l.set(i, j, value * inverseRoot);
    }
  }
  return l;
}

// -- CPython's hypot ---------------------------------------------------------

// ldexp(1.0, 27) + 1.0: the Veltkamp splitting constant Dekker's exact product
// needs. CPython uses a fused multiply-add where the platform has a reliable
// one and this same split where it does not; both compute the exact product as
// a (hi, lo) pair, so they agree bit for bit.
const VELTKAMP_SPLIT = 134217729.0;

const EXPONENT_VIEW = new DataView(new ArrayBuffer(8));

/** The exponent `frexp` would report: the e with `x = m * 2**e` and
 * `0.5 <= |m| < 1`. Read from the bit pattern, so it is exact for subnormals
 * too (those are first scaled into the normal range by an exact power of
 * two -- 2**1074 overflows a double, hence the two steps). */
function frexpExponent(x: number): number {
  EXPONENT_VIEW.setFloat64(0, x);
  const biased = (EXPONENT_VIEW.getUint32(0) >>> 20) & 0x7ff;
  if (biased !== 0) {
    return biased - 1022;
  }
  EXPONENT_VIEW.setFloat64(0, x * 2 ** 537 * 2 ** 537);
  return (((EXPONENT_VIEW.getUint32(0) >>> 20) & 0x7ff) - 1022) - 1074;
}

/** Dekker's mul12: the exact product of x and y as an unevaluated sum
 * `hi + lo`. */
function exactProduct(x: number, y: number): [number, number] {
  const xt = x * VELTKAMP_SPLIT;
  const xhi = xt - (xt - x);
  const xlo = x - xhi;
  const yt = y * VELTKAMP_SPLIT;
  const yhi = yt - (yt - y);
  const ylo = y - yhi;
  const p = xhi * yhi;
  const q = xhi * ylo + xlo * yhi;
  const hi = p + q;
  const lo = p - hi + q + xlo * ylo;
  return [hi, lo];
}

/** Compensated sum of two floats, `|a| >= |b|`, as an exact `hi + lo`. */
function exactSum(a: number, b: number): [number, number] {
  const hi = a + b;
  return [hi, a - hi + b];
}

/** `Math.sqrt(a * a + b * b)` computed the way CPython's `math.hypot` computes
 * it: lossless scaling by a power of two, Dekker-exact squares, compensated
 * summation, and a final differential correction.
 *
 * The point is not accuracy for its own sake -- it is that the Python engine
 * calls `math.hypot`, its result is compared against `SPEED_MAX_STEP_M`, and a
 * one-ULP disagreement on a boundary case is a different counting decision.
 * `Math.hypot` disagrees with CPython on roughly a quarter of ordinary inputs;
 * this agrees on every one of the 40048 fuzz cases it was measured against,
 * subnormals and extreme scales included. */
export function hypot(a: number, b: number): number {
  const ax = Math.abs(a);
  const bx = Math.abs(b);
  // Infinity wins over NaN, as CPython's vector_norm checks it first.
  if (ax === Infinity || bx === Infinity) {
    return Infinity;
  }
  if (Number.isNaN(ax) || Number.isNaN(bx)) {
    return NaN;
  }
  const max = Math.max(ax, bx);
  if (max === 0.0) {
    return max;
  }

  const maxExponent = frexpExponent(max);
  if (maxExponent < -1023) {
    // ldexp(1.0, -maxExponent) would overflow: lift the subnormals into the
    // normal range first, which is exact, and scale the answer back.
    const DBL_MIN = 2.2250738585072014e-308;
    return DBL_MIN * hypot(ax / DBL_MIN, bx / DBL_MIN);
  }
  const scale = 2 ** -maxExponent;

  let csum = 1.0;
  let frac1 = 0.0;
  let frac2 = 0.0;
  for (const raw of [ax, bx]) {
    const x = raw * scale; // lossless scaling
    const [productHi, productLo] = exactProduct(x, x); // lossless squaring
    const [sumHi, sumLo] = exactSum(csum, productHi); // lossless addition
    csum = sumHi;
    frac1 += productLo; // lossy addition
    frac2 += sumLo; // lossy addition
  }

  let h = Math.sqrt(csum - 1.0 + (frac1 + frac2));
  const [productHi, productLo] = exactProduct(-h, h);
  const [sumHi, sumLo] = exactSum(csum, productHi);
  csum = sumHi;
  frac1 += productLo;
  frac2 += sumLo;
  const residual = csum - 1.0 + (frac1 + frac2);
  h += residual / (2.0 * h); // differential correction
  return h / scale;
}
