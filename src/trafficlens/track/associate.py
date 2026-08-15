"""Detection-to-track association primitives: vectorised IoU and Hungarian
assignment with a cost ceiling and a canonical tie rule.

This is the ONLY module in the tracking layer that imports scipy (for
``scipy.optimize.linear_sum_assignment``); ``trafficlens.track.tracker``
composes these functions and stays numpy/stdlib otherwise, and
``trafficlens.track.kalman`` is numpy-only. The later TypeScript mirror
replaces scipy with a hand-written Hungarian/Jonker-Volgenant solver and
must reproduce the exact conventions documented here:

- Costs are plain float64; ``np.inf`` marks a BARRED pair (cross-class, or
  outside the Mahalanobis gate). ``assign`` substitutes a large finite
  value for non-finite entries before solving -- scipy raises "cost matrix
  is infeasible" when infinities make a complete matching impossible --
  and the post-filter below then discards any barred pair the assignment
  was forced through, so barred pairs can never surface as matches.
- A candidate pair survives only when ``cost[i, j] <= max_cost``; strictly
  greater is unmatched. The tracker passes ``max_cost = 1 - match_thresh``
  with ``cost = 1 - IoU``, i.e. a pair matches exactly when
  ``IoU >= match_thresh`` (up to the IEEE-754 evaluation of both
  expressions, which the mirror reproduces bit-for-bit in float64).
- Tied costs make the optimal match SET itself non-unique, and which
  optimum a solver returns is implementation-internal -- a different (even
  a differently-versioned) solver may legitimately return another one.
  ``assign`` therefore never trusts the solver's choice: the solver is
  used only as an ORACLE for the optimal total, and the returned
  assignment is reconstructed canonically (see ``_canonical_assignment``)
  as the lexicographically-(row, col)-least assignment whose total is
  within ``TRACK_ASSIGN_TIE_TOL`` of that optimum. Any correct optimal
  solver plugged into the same reconstruction yields the identical match
  set, which is what makes the tracker's decisions reproducible in
  TypeScript.

  Why not the seemingly-simpler additive index perturbation
  (``cost + eps * (row * n_cols + col)``): over any COMPLETE square
  assignment that term sums to a constant (``n_cols * sum(rows) +
  sum(cols)`` is permutation-invariant), so it cannot separate tied
  square optima mathematically -- and in float64 its rounding residue
  actually biases the solver toward the ANTI-lexicographic pairing on an
  all-equal 2x2 block. Exponential index weights would separate optima
  but underflow/get absorbed for realistic matrix sizes. Reconstruction
  against an optimal-total oracle is the scheme that is provably
  canonical at every size.
- All returned index lists are sorted ascending.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from trafficlens.core.constants import TRACK_ASSIGN_TIE_TOL

# Stand-in cost for barred (non-finite) entries when the matrix is handed
# to the solver. Two bounds pin the value (not a tunable, hence defined
# here and not in core.constants):
# - LOWER: it must dwarf any achievable total of real costs, so a barred
#   pair is crossed only when no feasible alternative exists at all. Real
#   costs are 1 - IoU <= 1 per pair and an assignment holds at most a few
#   hundred pairs, so real totals stay under ~1e3; 1e4 clears that.
# - UPPER: totals that include barred entries must still be comparable to
#   TRACK_ASSIGN_TIE_TOL (1e-6) precision, or the canonical tie rule
#   would go blind exactly when barred pairs are involved. A total of a
#   few hundred 1e4 entries stays under ~1e7, whose float64 ulp (~1e-9)
#   leaves three orders of margin under the tolerance. A 1e9 stand-in
#   would not: ulp(1e11) is ~1e-5, already coarser than the tolerance.
_BARRED_STAND_IN = 1e4


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of two sets of ``[x1, y1, x2, y2]`` boxes.

    ``boxes_a`` is (N, 4), ``boxes_b`` is (M, 4); returns (N, M) float64.
    Fully vectorised; degenerate pairs (union of zero area) score 0.0
    rather than dividing by zero. Either input may be empty, giving the
    correspondingly empty-shaped result.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m))

    x1 = np.maximum(boxes_a[:, 0, None], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, 1, None], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, 3, None], boxes_b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(
        0.0, boxes_a[:, 3] - boxes_a[:, 1]
    )
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(
        0.0, boxes_b[:, 3] - boxes_b[:, 1]
    )
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)


def _ordered_total(matrix: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    """Sum ``matrix`` over the given pairs by sequential addition in
    ascending-row order. The order is part of the contract: the TS mirror
    must accumulate identically so tolerance comparisons agree."""
    total = 0.0
    for i in range(len(rows)):
        total += float(matrix[rows[i], cols[i]])
    return total


def _canonical_assignment(
    solver_cost: np.ndarray, best_total: float
) -> list[tuple[int, int]]:
    """The lexicographically-(row, col)-least full-cardinality assignment
    whose total is within ``TRACK_ASSIGN_TIE_TOL`` of ``best_total``.

    Rows are fixed in ascending order; for each row, columns are tried in
    ascending order and the first column whose forced choice still admits
    a completion within tolerance of the optimum is kept (the residual
    subproblem is re-solved to check). Matching a row is preferred over
    leaving it unmatched, which can only happen when rows outnumber
    columns. Deterministic by construction and independent of which
    particular optimum the underlying solver would have returned; cost:
    at most rows x cols residual solves, in practice a handful, on the
    small matrices a video frame produces.
    """
    n_rows, n_cols = solver_cost.shape
    k = min(n_rows, n_cols)  # finite matrix: every optimum has k pairs
    chosen: list[tuple[int, int]] = []
    available = list(range(n_cols))  # ascending
    forced_total = 0.0

    for r in range(n_rows):
        if len(chosen) == k:
            break
        accepted = -1
        for c in available:
            needed = k - len(chosen) - 1
            if needed == 0:
                residual_total = 0.0
            else:
                rest_rows = np.arange(r + 1, n_rows)
                rest_cols = np.array([x for x in available if x != c])
                if len(rest_rows) < needed:
                    continue  # too few rows left to finish at cardinality k
                sub = solver_cost[np.ix_(rest_rows, rest_cols)]
                sub_rows, sub_cols = linear_sum_assignment(sub)
                residual_total = _ordered_total(sub, sub_rows, sub_cols)
            candidate = forced_total + float(solver_cost[r, c]) + residual_total
            if candidate <= best_total + TRACK_ASSIGN_TIE_TOL:
                accepted = c
                break
        if accepted < 0:
            # No column keeps row r inside the optimum: the canonical
            # solution leaves it unmatched (only possible with more rows
            # than columns; otherwise some column always qualifies).
            continue
        chosen.append((r, accepted))
        forced_total += float(solver_cost[r, accepted])
        available.remove(accepted)
    return chosen


def assign(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost assignment of rows to columns with a cost ceiling and
    a canonical tie rule.

    Solves the Hungarian assignment on ``cost`` (rows x cols, float64;
    ``np.inf`` entries are barred pairs, substituted with a large finite
    stand-in for the solver -- see module docstring) to obtain the optimal
    total, reconstructs the CANONICAL optimum -- lexicographically-
    (row, col)-least among assignments within ``TRACK_ASSIGN_TIE_TOL`` of
    that total, independent of the solver's own tie choices -- and then
    POST-FILTERS it: any pair with ``cost[i, j] > max_cost`` (reading the
    ORIGINAL matrix, so barred pairs always drop out) is broken up and
    both sides reported unmatched. Returns
    ``(matches, unmatched_rows, unmatched_cols)`` with matches sorted
    ascending by row and both unmatched index lists ascending.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    solver_cost = np.where(np.isfinite(cost), cost, _BARRED_STAND_IN)
    row_ind, col_ind = linear_sum_assignment(solver_cost)
    best_total = _ordered_total(solver_cost, row_ind, col_ind)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in _canonical_assignment(solver_cost, best_total):
        if cost[r, c] > max_cost:  # inf always lands here: barred pairs drop out
            continue
        matches.append((r, c))
        matched_rows.add(r)
        matched_cols.add(c)

    matches.sort()
    unmatched_rows = sorted(set(range(n_rows)) - matched_rows)
    unmatched_cols = sorted(set(range(n_cols)) - matched_cols)
    return matches, unmatched_rows, unmatched_cols
