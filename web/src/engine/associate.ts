/** Detection-to-track association primitives: vectorised IoU and Hungarian
 * assignment with a cost ceiling and a canonical tie rule. Mirrors
 * `trafficlens.track.associate`.
 *
 * The Python module is the only one in its tracking layer that imports scipy,
 * for `scipy.optimize.linear_sum_assignment`. There is no scipy in a browser,
 * so the solver is written out below -- a port of scipy's own shortest
 * augmenting path (Crouse 2016) -- and the conventions the Python module
 * documents are reproduced exactly:
 *
 * - Costs are plain float64; `Infinity` marks a BARRED pair (cross-class, or
 *   outside the Mahalanobis gate). `assign` substitutes a large finite value
 *   for non-finite entries before solving -- a complete matching is impossible
 *   when infinities block it -- and the post-filter then discards any barred
 *   pair the assignment was forced through, so barred pairs can never surface
 *   as matches.
 * - A candidate pair survives only when `cost[i][j] <= maxCost`; strictly
 *   greater is unmatched. The tracker passes `maxCost = 1 - matchThresh` with
 *   `cost = 1 - IoU`, i.e. a pair matches exactly when `IoU >= matchThresh`.
 * - Tied costs make the optimal match SET itself non-unique, and which optimum
 *   a solver returns is implementation-internal -- a different (even a
 *   differently-versioned) solver may legitimately return another one. `assign`
 *   therefore never trusts the solver's choice: the solver is used only as an
 *   ORACLE for the optimal total, and the returned assignment is reconstructed
 *   canonically (see `canonicalAssignment`) as the
 *   lexicographically-(row, col)-least assignment whose total is within
 *   `TRACK_ASSIGN_TIE_TOL` of that optimum. Any correct optimal solver plugged
 *   into the same reconstruction yields the identical match set, and THAT is
 *   what makes this engine and the Python one allocate the same track IDs --
 *   not any agreement between this solver and scipy's about ties.
 *
 *   Why not the seemingly-simpler additive index perturbation
 *   (`cost + eps * (row * nCols + col)`): over any COMPLETE square assignment
 *   that term sums to a constant (`nCols * sum(rows) + sum(cols)` is
 *   permutation-invariant), so it cannot separate tied square optima
 *   mathematically -- and in float64 its rounding residue actually biases the
 *   solver toward the ANTI-lexicographic pairing on an all-equal 2x2 block.
 *   Exponential index weights would separate optima but underflow for
 *   realistic matrix sizes. Reconstruction against an optimal-total oracle is
 *   the scheme that is provably canonical at every size.
 * - All returned index lists are sorted ascending. */

import { TRACK_ASSIGN_TIE_TOL } from "../generated/constants";
import { Matrix } from "./numeric";

/** Stand-in cost for barred (non-finite) entries when the matrix is handed to
 * the solver. Two bounds pin the value (not a tunable, hence defined here and
 * not in the shared constants):
 * - LOWER: it must dwarf any achievable total of real costs, so a barred pair
 *   is crossed only when no feasible alternative exists at all. Real costs are
 *   1 - IoU <= 1 per pair and an assignment holds at most a few hundred pairs,
 *   so real totals stay under ~1e3; 1e4 clears that.
 * - UPPER: totals that include barred entries must still be comparable to
 *   TRACK_ASSIGN_TIE_TOL (1e-6) precision, or the canonical tie rule would go
 *   blind exactly when barred pairs are involved. A total of a few hundred 1e4
 *   entries stays under ~1e7, whose float64 ulp (~1e-9) leaves three orders of
 *   margin under the tolerance. A 1e9 stand-in would not: ulp(1e11) is ~1e-5,
 *   already coarser than the tolerance. */
export const BARRED_STAND_IN = 1e4;

export interface Assignment {
  matches: [number, number][];
  unmatchedRows: number[];
  unmatchedCols: number[];
}

/** Pairwise IoU of two sets of `[x1, y1, x2, y2]` boxes.
 *
 * `boxesA` is (N, 4), `boxesB` is (M, 4); returns (N, M). Degenerate pairs
 * (union of zero area) score 0.0 rather than dividing by zero. Either input
 * may be empty, giving the correspondingly empty-shaped result. */
export function iouMatrix(boxesA: Matrix, boxesB: Matrix): Matrix {
  const n = boxesA.rows;
  const m = boxesB.rows;
  const out = Matrix.zeros(n, m);
  if (n === 0 || m === 0) {
    return out;
  }

  const areaA = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    areaA[i] =
      Math.max(0.0, boxesA.get(i, 2) - boxesA.get(i, 0)) *
      Math.max(0.0, boxesA.get(i, 3) - boxesA.get(i, 1));
  }
  const areaB = new Float64Array(m);
  for (let j = 0; j < m; j += 1) {
    areaB[j] =
      Math.max(0.0, boxesB.get(j, 2) - boxesB.get(j, 0)) *
      Math.max(0.0, boxesB.get(j, 3) - boxesB.get(j, 1));
  }

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < m; j += 1) {
      const x1 = Math.max(boxesA.get(i, 0), boxesB.get(j, 0));
      const y1 = Math.max(boxesA.get(i, 1), boxesB.get(j, 1));
      const x2 = Math.min(boxesA.get(i, 2), boxesB.get(j, 2));
      const y2 = Math.min(boxesA.get(i, 3), boxesB.get(j, 3));
      const inter = Math.max(0.0, x2 - x1) * Math.max(0.0, y2 - y1);
      const union = (areaA[i] as number) + (areaB[j] as number) - inter;
      out.set(i, j, union > 0.0 ? inter / union : 0.0);
    }
  }
  return out;
}

/** Sum `matrix` over the given pairs by sequential addition in ascending-row
 * order. The order is part of the contract: the Python side accumulates
 * identically so the tolerance comparisons agree. */
function orderedTotal(
  matrix: Matrix,
  rows: readonly number[],
  cols: readonly number[],
): number {
  let total = 0.0;
  for (let i = 0; i < rows.length; i += 1) {
    total += matrix.get(rows[i] as number, cols[i] as number);
  }
  return total;
}

interface SolverResult {
  rows: number[];
  cols: number[];
}

/** One shortest augmenting path from `startRow`, updating `path`,
 * `shortestPathCosts`, `SR` and `SC` in place. Returns the sink column and the
 * path cost, or a sink of -1 when the matrix is infeasible. */
function augmentingPath(
  nc: number,
  cost: Matrix,
  u: Float64Array,
  v: Float64Array,
  path: Int32Array,
  row4col: Int32Array,
  shortestPathCosts: Float64Array,
  startRow: number,
  SR: boolean[],
  SC: boolean[],
  remaining: Int32Array,
): { sink: number; minVal: number } {
  let i = startRow;
  let minVal = 0.0;

  let numRemaining = nc;
  for (let it = 0; it < nc; it += 1) {
    // Filling this up in reverse order is what makes a constant cost matrix
    // solve to the identity assignment.
    remaining[it] = nc - it - 1;
  }

  SR.fill(false);
  SC.fill(false);
  shortestPathCosts.fill(Infinity);

  let sink = -1;
  while (sink === -1) {
    let index = -1;
    let lowest = Infinity;
    SR[i] = true;

    for (let it = 0; it < numRemaining; it += 1) {
      const j = remaining[it] as number;
      const r = minVal + cost.get(i, j) - (u[i] as number) - (v[j] as number);
      if (r < (shortestPathCosts[j] as number)) {
        path[j] = i;
        shortestPathCosts[j] = r;
      }
      // When several nodes share the minimum cost, prefer one that gives a new
      // sink node.
      if (
        (shortestPathCosts[j] as number) < lowest ||
        ((shortestPathCosts[j] as number) === lowest && (row4col[j] as number) === -1)
      ) {
        lowest = shortestPathCosts[j] as number;
        index = it;
      }
    }

    minVal = lowest;
    if (minVal === Infinity) {
      return { sink: -1, minVal }; // infeasible cost matrix
    }

    const j = remaining[index] as number;
    if ((row4col[j] as number) === -1) {
      sink = j;
    } else {
      i = row4col[j] as number;
    }

    SC[j] = true;
    numRemaining -= 1;
    remaining[index] = remaining[numRemaining] as number;
  }

  return { sink, minVal };
}

/** Minimum-cost assignment of rows to columns, returning `min(rows, cols)`
 * pairs with the row indices ascending. A port of scipy's
 * `linear_sum_assignment`; `assign` uses it only as an oracle for the optimal
 * TOTAL, never for which particular optimum it picks. */
function linearSumAssignment(cost: Matrix): SolverResult {
  const nr = cost.rows;
  const nc = cost.cols;
  if (nr === 0 || nc === 0) {
    return { rows: [], cols: [] };
  }
  if (nr > nc) {
    const transposed = linearSumAssignment(cost.transpose());
    const pairs: [number, number][] = transposed.rows.map((c, index) => [
      transposed.cols[index] as number,
      c,
    ]);
    pairs.sort((a, b) => a[0] - b[0]);
    return { rows: pairs.map((p) => p[0]), cols: pairs.map((p) => p[1]) };
  }

  const u = new Float64Array(nr);
  const v = new Float64Array(nc);
  const shortestPathCosts = new Float64Array(nc);
  const path = new Int32Array(nc).fill(-1);
  const col4row = new Int32Array(nr).fill(-1);
  const row4col = new Int32Array(nc).fill(-1);
  const SR: boolean[] = new Array<boolean>(nr).fill(false);
  const SC: boolean[] = new Array<boolean>(nc).fill(false);
  const remaining = new Int32Array(nc);

  for (let curRow = 0; curRow < nr; curRow += 1) {
    const { sink, minVal } = augmentingPath(
      nc,
      cost,
      u,
      v,
      path,
      row4col,
      shortestPathCosts,
      curRow,
      SR,
      SC,
      remaining,
    );
    if (sink < 0) {
      throw new Error("cost matrix is infeasible");
    }

    // Update the dual variables. The parenthesisation is scipy's: `+=` and
    // `-=` group the right-hand side first, which is not the same float64 as
    // accumulating left to right.
    u[curRow] = (u[curRow] as number) + minVal;
    for (let i = 0; i < nr; i += 1) {
      if (SR[i] === true && i !== curRow) {
        u[i] =
          (u[i] as number) +
          (minVal - (shortestPathCosts[col4row[i] as number] as number));
      }
    }
    for (let j = 0; j < nc; j += 1) {
      if (SC[j] === true) {
        v[j] = (v[j] as number) - (minVal - (shortestPathCosts[j] as number));
      }
    }

    // Augment the previous solution along the path just found.
    let j = sink;
    for (;;) {
      const i = path[j] as number;
      row4col[j] = i;
      const next = col4row[i] as number;
      col4row[i] = j;
      j = next;
      if (i === curRow) {
        break;
      }
    }
  }

  const rows: number[] = [];
  const cols: number[] = [];
  for (let i = 0; i < nr; i += 1) {
    rows.push(i);
    cols.push(col4row[i] as number);
  }
  return { rows, cols };
}

function submatrix(
  source: Matrix,
  rows: readonly number[],
  cols: readonly number[],
): Matrix {
  const out = Matrix.zeros(rows.length, cols.length);
  for (let i = 0; i < rows.length; i += 1) {
    for (let j = 0; j < cols.length; j += 1) {
      out.set(i, j, source.get(rows[i] as number, cols[j] as number));
    }
  }
  return out;
}

/** The lexicographically-(row, col)-least full-cardinality assignment whose
 * total is within `TRACK_ASSIGN_TIE_TOL` of `bestTotal`.
 *
 * Rows are fixed in ascending order; for each row, columns are tried in
 * ascending order and the first column whose forced choice still admits a
 * completion within tolerance of the optimum is kept (the residual subproblem
 * is re-solved to check). Matching a row is preferred over leaving it
 * unmatched, which can only happen when rows outnumber columns. Deterministic
 * by construction and independent of which particular optimum the underlying
 * solver would have returned; cost: at most rows x cols residual solves, in
 * practice a handful, on the small matrices a video frame produces. */
function canonicalAssignment(
  solverCost: Matrix,
  bestTotal: number,
): [number, number][] {
  const nRows = solverCost.rows;
  const nCols = solverCost.cols;
  const k = Math.min(nRows, nCols); // finite matrix: every optimum has k pairs
  const chosen: [number, number][] = [];
  let available: number[] = [];
  for (let c = 0; c < nCols; c += 1) {
    available.push(c); // ascending
  }
  let forcedTotal = 0.0;

  for (let r = 0; r < nRows; r += 1) {
    if (chosen.length === k) {
      break;
    }
    let accepted = -1;
    for (const c of available) {
      const needed = k - chosen.length - 1;
      let residualTotal: number;
      if (needed === 0) {
        residualTotal = 0.0;
      } else {
        const restRows: number[] = [];
        for (let rr = r + 1; rr < nRows; rr += 1) {
          restRows.push(rr);
        }
        const restCols = available.filter((x) => x !== c);
        if (restRows.length < needed) {
          continue; // too few rows left to finish at cardinality k
        }
        const sub = submatrix(solverCost, restRows, restCols);
        const solved = linearSumAssignment(sub);
        residualTotal = orderedTotal(sub, solved.rows, solved.cols);
      }
      const candidate = forcedTotal + solverCost.get(r, c) + residualTotal;
      if (candidate <= bestTotal + TRACK_ASSIGN_TIE_TOL) {
        accepted = c;
        break;
      }
    }
    if (accepted < 0) {
      // No column keeps row r inside the optimum: the canonical solution
      // leaves it unmatched (only possible with more rows than columns;
      // otherwise some column always qualifies).
      continue;
    }
    chosen.push([r, accepted]);
    forcedTotal += solverCost.get(r, accepted);
    available = available.filter((x) => x !== accepted);
  }
  return chosen;
}

/** Minimum-cost assignment of rows to columns with a cost ceiling and a
 * canonical tie rule.
 *
 * Solves the Hungarian assignment on `cost` (rows x cols; `Infinity` entries
 * are barred pairs, substituted with a large finite stand-in for the solver)
 * to obtain the optimal total, reconstructs the CANONICAL optimum --
 * lexicographically-(row, col)-least among assignments within
 * `TRACK_ASSIGN_TIE_TOL` of that total, independent of the solver's own tie
 * choices -- and then POST-FILTERS it: any pair with `cost[i][j] > maxCost`
 * (reading the ORIGINAL matrix, so barred pairs always drop out) is broken up
 * and both sides reported unmatched. Matches are sorted ascending by row and
 * both unmatched index lists are ascending. */
export function assign(cost: Matrix, maxCost: number): Assignment {
  const nRows = cost.rows;
  const nCols = cost.cols;
  if (nRows === 0 || nCols === 0) {
    const allRows: number[] = [];
    for (let i = 0; i < nRows; i += 1) {
      allRows.push(i);
    }
    const allCols: number[] = [];
    for (let j = 0; j < nCols; j += 1) {
      allCols.push(j);
    }
    return { matches: [], unmatchedRows: allRows, unmatchedCols: allCols };
  }

  const solverCost = Matrix.zeros(nRows, nCols);
  for (let i = 0; i < nRows; i += 1) {
    for (let j = 0; j < nCols; j += 1) {
      const value = cost.get(i, j);
      solverCost.set(i, j, Number.isFinite(value) ? value : BARRED_STAND_IN);
    }
  }
  const solved = linearSumAssignment(solverCost);
  const bestTotal = orderedTotal(solverCost, solved.rows, solved.cols);

  const matches: [number, number][] = [];
  const matchedRows = new Set<number>();
  const matchedCols = new Set<number>();
  for (const [r, c] of canonicalAssignment(solverCost, bestTotal)) {
    if (cost.get(r, c) > maxCost) {
      continue; // Infinity always lands here: barred pairs drop out
    }
    matches.push([r, c]);
    matchedRows.add(r);
    matchedCols.add(c);
  }

  matches.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const unmatchedRows: number[] = [];
  for (let i = 0; i < nRows; i += 1) {
    if (!matchedRows.has(i)) {
      unmatchedRows.push(i);
    }
  }
  const unmatchedCols: number[] = [];
  for (let j = 0; j < nCols; j += 1) {
    if (!matchedCols.has(j)) {
      unmatchedCols.push(j);
    }
  }
  return { matches, unmatchedRows, unmatchedCols };
}
