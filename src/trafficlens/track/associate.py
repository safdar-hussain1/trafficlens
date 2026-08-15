"""Detection-to-track association primitives: vectorised IoU and Hungarian
assignment with a cost ceiling.

This is the ONLY module in the tracking layer that imports scipy (for
``scipy.optimize.linear_sum_assignment``); ``trafficlens.track.tracker``
composes these functions and stays numpy/stdlib otherwise, and
``trafficlens.track.kalman`` is numpy-only. The later TypeScript mirror
replaces this module with a hand-written Jonker-Volgenant/Hungarian solver
and must reproduce the exact conventions documented here:

- Costs are plain float64; ``np.inf`` marks a BARRED pair (cross-class, or
  outside the Mahalanobis gate). ``assign`` substitutes a large finite
  value for non-finite entries before solving -- scipy raises "cost matrix
  is infeasible" when infinities make a complete matching impossible --
  and the post-filter below then discards any barred pair the solver was
  forced through, so barred pairs can never surface as matches.
- A candidate pair survives only when ``cost[i, j] <= max_cost``; strictly
  greater is unmatched. The tracker passes ``max_cost = 1 - match_thresh``
  with ``cost = 1 - IoU``, i.e. a pair matches exactly when
  ``IoU >= match_thresh`` (up to the IEEE-754 evaluation of both
  expressions, which the mirror reproduces bit-for-bit in float64).
- All returned index lists are sorted ascending, so downstream iteration
  order -- and therefore every tracker decision built on it -- is
  deterministic and platform-independent.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# Stand-in cost for barred (non-finite) entries when the matrix is handed
# to the Hungarian solver. Any value dwarfing the largest possible sum of
# real costs works identically (real costs are 1 - IoU <= 1 per pair, and
# a frame holds at most a few hundred pairs); 1e9 leaves nine orders of
# magnitude of headroom, so the solver only crosses a barred pair when no
# feasible alternative exists at all, and the max_cost post-filter then
# drops it. Not a tunable, hence defined here and not in core.constants.
_BARRED_STAND_IN = 1e9


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


def assign(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost assignment of rows to columns with a cost ceiling.

    Solves the Hungarian assignment on ``cost`` (rows x cols, float64;
    ``np.inf`` entries are barred pairs -- see the module docstring for how
    they are made solver-safe), then POST-FILTERS the solution: any
    assigned pair with ``cost[i, j] > max_cost`` is broken up and both
    sides reported unmatched. Returns
    ``(matches, unmatched_rows, unmatched_cols)`` where ``matches`` is a
    list of ``(row, col)`` pairs; matches are sorted ascending by row and
    the unmatched index lists ascending, so callers iterate in one
    reproducible order on every platform.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    solver_cost = np.where(np.isfinite(cost), cost, _BARRED_STAND_IN)
    row_ind, col_ind = linear_sum_assignment(solver_cost)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[r, c] > max_cost:  # inf always lands here: barred pairs drop out
            continue
        matches.append((r, c))
        matched_rows.add(r)
        matched_cols.add(c)

    matches.sort()
    unmatched_rows = sorted(set(range(n_rows)) - matched_rows)
    unmatched_cols = sorted(set(range(n_cols)) - matched_cols)
    return matches, unmatched_rows, unmatched_cols
