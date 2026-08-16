// Ported from the associate section of tests/test_tracker.py, plus a
// brute-force cross-check of the canonical tie rule.
//
// The Python side verified its reconstruction loop against brute force over
// thousands of tied and barred matrices before shipping; that verification was
// never committed, so it is written out here. It is the strongest available
// guard on the hardest part of this mirror: `assign` must return the
// lexicographically-(row, col)-least optimal assignment, whatever the
// underlying solver happens to prefer.

import { describe, expect, it } from "vitest";

import { TRACK_ASSIGN_TIE_TOL } from "../generated/constants";
import { BARRED_STAND_IN, assign, iouMatrix } from "./associate";
import { Matrix } from "./numeric";

// -- iouMatrix ---------------------------------------------------------------

describe("iouMatrix", () => {
  it("has the exact expected values and shape", () => {
    const a = Matrix.from([[0.0, 0.0, 10.0, 10.0]]);
    const b = Matrix.from([
      [0.0, 0.0, 10.0, 10.0], // identical -> 1
      [5.0, 0.0, 15.0, 10.0], // half-shifted -> 50 / 150 = 1/3
      [20.0, 20.0, 30.0, 30.0], // disjoint -> 0
    ]);
    const m = iouMatrix(a, b);
    expect([m.rows, m.cols]).toEqual([1, 3]);
    expect(m.get(0, 0)).toBeCloseTo(1.0, 12);
    expect(m.get(0, 1)).toBeCloseTo(1.0 / 3.0, 12);
    expect(m.get(0, 2)).toBe(0.0);
  });

  it("handles empty inputs", () => {
    const empty = Matrix.zeros(0, 4);
    const some = Matrix.from([[0.0, 0.0, 10.0, 10.0]]);
    expect([iouMatrix(empty, some).rows, iouMatrix(empty, some).cols]).toEqual([0, 1]);
    expect([iouMatrix(some, empty).rows, iouMatrix(some, empty).cols]).toEqual([1, 0]);
    expect([iouMatrix(empty, empty).rows, iouMatrix(empty, empty).cols]).toEqual([0, 0]);
  });
});

// -- assign ------------------------------------------------------------------

describe("assign", () => {
  it("filters pairs above max_cost", () => {
    const cost = Matrix.from([
      [0.1, 0.9],
      [0.9, 0.2],
    ]);
    let result = assign(cost, 0.5);
    expect(result.matches).toEqual([
      [0, 0],
      [1, 1],
    ]);
    expect(result.unmatchedRows).toEqual([]);
    expect(result.unmatchedCols).toEqual([]);

    // Tighten the ceiling: the 0.2 pair now exceeds it and must come apart.
    result = assign(cost, 0.15);
    expect(result.matches).toEqual([[0, 0]]);
    expect(result.unmatchedRows).toEqual([1]);
    expect(result.unmatchedCols).toEqual([1]);
  });

  it("never matches barred pairs", () => {
    // Barred (inf) pairs must never match, even when the assignment solver is
    // forced through them to complete a square assignment.
    const cost = Matrix.from([
      [Infinity, 0.3],
      [0.2, Infinity],
    ]);
    expect(assign(cost, 0.5).matches).toEqual([
      [0, 1],
      [1, 0],
    ]);

    const allBarred = Matrix.from([
      [Infinity, Infinity],
      [Infinity, Infinity],
    ]);
    const barred = assign(allBarred, 0.5);
    expect(barred.matches).toEqual([]);
    expect(barred.unmatchedRows).toEqual([0, 1]);
    expect(barred.unmatchedCols).toEqual([0, 1]);

    const noRows = assign(Matrix.zeros(0, 3), 0.5);
    expect(noRows.matches).toEqual([]);
    expect(noRows.unmatchedRows).toEqual([]);
    expect(noRows.unmatchedCols).toEqual([0, 1, 2]);
  });

  it("breaks exact ties toward the lowest indices", () => {
    // With tied costs the optimal match SET is not unique, and which optimum
    // a solver returns is implementation-internal -- the mirror cannot
    // inherit scipy's choice. assign() therefore canonicalizes: the returned
    // assignment is the lexicographically-(row,col)-least among
    // (near-)optimal assignments, independent of the solver used.
    let result = assign(Matrix.filled(2, 2, 0.5), 0.6);
    expect(result.matches).toEqual([
      [0, 0],
      [1, 1],
    ]);
    expect(result.unmatchedRows).toEqual([]);
    expect(result.unmatchedCols).toEqual([]);

    // Three-way tie: canonical result is the identity pairing, which a 2-swap
    // rule alone could not guarantee (rotations tie pairwise).
    expect(assign(Matrix.filled(3, 3, 0.5), 0.6).matches).toEqual([
      [0, 0],
      [1, 1],
      [2, 2],
    ]);

    // Rectangular ties: WHICH column (or row) goes unmatched is part of the
    // optimum and must be canonical too -- lowest indices match first.
    result = assign(Matrix.filled(1, 2, 0.5), 0.6);
    expect(result.matches).toEqual([[0, 0]]);
    expect(result.unmatchedCols).toEqual([1]);
    result = assign(Matrix.filled(2, 1, 0.5), 0.6);
    expect(result.matches).toEqual([[0, 0]]);
    expect(result.unmatchedRows).toEqual([1]);

    // A genuine cost difference is never overridden by the tie rule.
    expect(
      assign(
        Matrix.from([
          [0.5, 0.4],
          [0.4, 0.5],
        ]),
        0.6,
      ).matches,
    ).toEqual([
      [0, 1],
      [1, 0],
    ]);
  });
});

// -- brute-force cross-check of the canonical rule ---------------------------

/** Every injective partial map from `rows` rows to `cols` columns of exactly
 * `k` pairs, each as an ascending-by-row list of [row, col]. */
function allAssignments(rows: number, cols: number, k: number): number[][][] {
  const out: number[][][] = [];
  const current: number[][] = [];
  const used = new Set<number>();

  const walk = (row: number): void => {
    if (current.length === k) {
      out.push(current.map((pair) => [pair[0] as number, pair[1] as number]));
      return;
    }
    if (row >= rows || rows - row < k - current.length) {
      return;
    }
    for (let c = 0; c < cols; c += 1) {
      if (used.has(c)) {
        continue;
      }
      used.add(c);
      current.push([row, c]);
      walk(row + 1);
      current.pop();
      used.delete(c);
    }
    walk(row + 1); // leave this row unmatched
  };

  walk(0);
  return out;
}

/** Compare two ascending-by-row pair lists lexicographically. */
function comparePairLists(a: number[][], b: number[][]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
    const pa = a[i] as number[];
    const pb = b[i] as number[];
    if ((pa[0] as number) !== (pb[0] as number)) {
      return (pa[0] as number) - (pb[0] as number);
    }
    if ((pa[1] as number) !== (pb[1] as number)) {
      return (pa[1] as number) - (pb[1] as number);
    }
  }
  return a.length - b.length;
}

/** The canonical assignment computed exhaustively, with the same
 * ordered summation and the same barred stand-in `assign` uses. */
function bruteForceAssign(cost: Matrix, maxCost: number): number[][] {
  const k = Math.min(cost.rows, cost.cols);
  if (k === 0) {
    return [];
  }
  const solverCost = Matrix.zeros(cost.rows, cost.cols);
  for (let i = 0; i < cost.rows; i += 1) {
    for (let j = 0; j < cost.cols; j += 1) {
      const v = cost.get(i, j);
      solverCost.set(i, j, Number.isFinite(v) ? v : BARRED_STAND_IN);
    }
  }

  const candidates = allAssignments(cost.rows, cost.cols, k);
  // Summed in ascending-row order, the order assign() accumulates in.
  const totals = candidates.map((pairs) => {
    let total = 0.0;
    for (const pair of pairs) {
      total += solverCost.get(pair[0] as number, pair[1] as number);
    }
    return total;
  });
  const best = Math.min(...totals);

  let winner: number[][] | null = null;
  for (let i = 0; i < candidates.length; i += 1) {
    if ((totals[i] as number) > best + TRACK_ASSIGN_TIE_TOL) {
      continue;
    }
    const cand = candidates[i] as number[][];
    if (winner === null || comparePairLists(cand, winner) < 0) {
      winner = cand;
    }
  }
  return (winner ?? []).filter(
    (pair) => cost.get(pair[0] as number, pair[1] as number) <= maxCost,
  );
}

/** Deterministic 32-bit LCG so the fuzz corpus is identical on every run. */
function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

describe("canonical tie rule against brute force", () => {
  it("agrees on 3000 tied and barred matrices", () => {
    const rand = lcg(20260816);
    const levels = [0.1, 0.2, 0.3, 0.4, 0.5]; // few distinct costs => many ties
    let checked = 0;

    for (let trial = 0; trial < 3000; trial += 1) {
      const rows = 1 + Math.floor(rand() * 4);
      const cols = 1 + Math.floor(rand() * 4);
      const cost = Matrix.zeros(rows, cols);
      for (let i = 0; i < rows; i += 1) {
        for (let j = 0; j < cols; j += 1) {
          if (rand() < 0.2) {
            cost.set(i, j, Infinity); // barred pair
          } else {
            cost.set(i, j, levels[Math.floor(rand() * levels.length)] as number);
          }
        }
      }
      const maxCost = levels[Math.floor(rand() * levels.length)] as number;

      const got = assign(cost, maxCost).matches.map(([r, c]) => [r, c]);
      const want = bruteForceAssign(cost, maxCost);
      expect(got, `trial ${trial}: ${JSON.stringify(cost.toArrays())}`).toEqual(want);
      checked += 1;
    }
    expect(checked).toBe(3000);
  });
});
