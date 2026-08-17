#!/usr/bin/env python3
"""Break every headline claim this project publishes, and require a NAMED test
to notice.

Why this script exists
----------------------
A test suite's size says nothing about what it defends. The only way to find out
whether a published claim is actually held up by a test is to remove the claim's
mechanism and watch. This script does that mechanically: one table row per
headline claim, each row a single-file ``find``/``replace`` that genuinely
removes the mechanism, plus the name of the ONE test that exists to defend it.

A row whose mutation leaves its named test GREEN is a **survivor**: the claim has
no test behind it. A survivor is the output of this tool, not a failure of it.

Why the instrument distrusts itself
-----------------------------------
The recurring defect this repository has met, repeatedly, is *a check that
cannot fail* -- and three times that defect appeared inside verification tooling
whose whole purpose was to prevent it. Five earlier mutation harnesses used a
bare ``str.replace()`` with no anchor assertion and so ran green while mutating
nothing. A battery reporting "0 survivors" while mutating nothing looks exactly
like one reporting it because everything is protected. So:

1. **Every mutation proves it happened.** The anchor must occur EXACTLY ONCE in
   the file, and the resulting bytes must differ from the original. Neither is a
   skip and neither is a pass: both are hard errors that name the claim and stop
   the run (exit 2).
2. **Every named test proves it exists.** Before anything is mutated, each named
   test is run on the clean tree and must both collect at least one test case
   and pass. A node id that collects nothing, or a Vitest ``-t`` filter that
   matches nothing, is a hard error -- otherwise "the test failed" and "the test
   never ran" would be indistinguishable, and a mutation would be scored as
   caught for the wrong reason.
3. **Restoration is from a saved byte copy, never ``git checkout``.** A
   ``git checkout <path>`` revert in this project's history destroyed unrelated
   uncommitted work. After restoring, the bytes must be identical to the saved
   copy; if they are not, the run stops immediately (exit 2) because the
   instrument has changed the repository in a way it did not intend, which is
   more serious than any finding it could produce.
4. **The tree must be clean to start and clean to finish.** This script edits
   tracked files in place. It refuses to run with uncommitted changes present,
   and re-checks ``git status --porcelain`` after the last entry.
5. **The table carries must-SURVIVE controls too.** A table of must-fails proves
   the tests fire. A semantically-equivalent mutation that must NOT redden its
   test proves they discriminate. Each control names the axis it varies, and
   that axis differs from its paired must-fail's.
6. **`tests/test_mutation_battery_smoke.py` proves this script can report a
   survivor at all**, and that it errors on a stale anchor -- against a
   throwaway sandbox repository, so the proof does not depend on this repository
   containing permanently-unprotected code.

Expectations a row may carry
----------------------------
``must_fail``     the named test must go RED. Staying green is a survivor.
``must_survive``  a control: the named test must stay GREEN.
``known_open``    the named test is expected to stay green even though it ought
                  to redden -- a documented open finding, routed elsewhere. The
                  assertion still bites in the other direction: if such an entry
                  is suddenly CAUGHT, the finding has been closed and the row
                  must be promoted to ``must_fail``, which is reported and exits
                  non-zero.

Exit codes
----------
0  every must_fail caught, every control green, every known_open still open.
1  at least one unexpected survivor, reddened control, or closed known_open.
2  a hard error: dirty tree, stale or ambiguous anchor, a named test that
   collects nothing, a mutation that changed no bytes, or a failed restore.

Usage
-----
    PYTHONPATH=src .venv/bin/python scripts/mutation_battery.py
    ... --list                 print the table and exit
    ... --only SUBSTRING       run only rows whose claim contains SUBSTRING
    ... --root DIR             operate on another checkout (the smoke test's
                               sandbox)
    ... --table FILE.json      replace the built-in table (the smoke test)

This script deliberately imports nothing from ``src/`` -- it must not depend on
any of the claims it tests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MUST_FAIL = "must_fail"
MUST_SURVIVE = "must_survive"
KNOWN_OPEN = "known_open"
_EXPECTATIONS = (MUST_FAIL, MUST_SURVIVE, KNOWN_OPEN)

# Words this repository's own guard bans from tracked files. Assembled from
# fragments so this file never contains one, exactly as tests/test_guards.py
# does -- the guard scans this file too.
_BANNED_INFLECTION = "re" + "building"
_BANNED_NEAR_MISS = "re" + "buildable"
_ABSOLUTE_PATH = "/Us" + "ers/asurveyor/trafficlens/private"


class HardError(RuntimeError):
    """The instrument cannot trust its own run. Never a skip, never a pass."""


@dataclass(frozen=True)
class Entry:
    """One claim, one single-file mutation, one named test."""

    claim: str
    path: str
    find: str
    replace: str
    #: ``pytest`` node id, or ``vitest`` "<file>::<name filter>".
    runner: str
    test: str
    expect: str
    #: For a control, the axis it varies. For a known_open, where it is routed.
    #: For a must_fail, what the mutation removes.
    note: str = ""

    def __post_init__(self) -> None:
        if self.expect not in _EXPECTATIONS:
            raise HardError(f"{self.claim}: unknown expectation {self.expect!r}")
        if self.runner not in ("pytest", "vitest"):
            raise HardError(f"{self.claim}: unknown runner {self.runner!r}")
        if self.find == self.replace:
            raise HardError(f"{self.claim}: find and replace are identical")
        if not self.find:
            raise HardError(f"{self.claim}: empty anchor")


@dataclass
class Result:
    entry: Entry
    #: "caught" (test went red) or "survived" (test stayed green).
    outcome: str
    bytes_before: int
    bytes_after: int
    detail: str = ""
    clean_rerun: str = ""

    @property
    def ok(self) -> bool:
        if self.entry.expect == MUST_FAIL:
            return self.outcome == "caught"
        return self.outcome == "survived"

    #: Index of the first byte the mutation changed, and how many bytes differ.
    #: Carried explicitly because a length-preserving mutation would otherwise
    #: print "N -> N bytes" and read exactly like a mutation that did nothing.
    first_changed_byte: int = -1
    differing_bytes: int = 0

    @property
    def byte_change_proof(self) -> str:
        return (
            f"{self.bytes_before} -> {self.bytes_after} bytes, "
            f"{self.differing_bytes} differ from offset {self.first_changed_byte}"
        )


# --- the table ----------------------------------------------------------------
#
# Ordered by the layer each claim lives in. Every `find` below is checked for
# being present EXACTLY ONCE before anything is written, so a refactor that
# moves one of these lines errors loudly instead of quietly protecting nothing.

CLAIMS: list[Entry] = [
    # -- the counting rule -----------------------------------------------------
    Entry(
        claim="crossing rule: a crossing is bounded to the drawn segment",
        path="src/trafficlens/core/gate.py",
        find="        if origin is None or not segments_intersect(origin, curr, gate_a, gate_b):",
        replace="        if origin is None:",
        runner="pytest",
        test="tests/test_gate.py::test_crossing_beyond_gate_end_point_not_counted",
        expect=MUST_FAIL,
        note="removes the bounded-segment test, leaving the infinite line",
    ),
    Entry(
        claim="once-per-track counting",
        path="src/trafficlens/core/gate.py",
        find="        if signed == 0 or track_id in self._counted:",
        replace="        if signed == 0:",
        runner="pytest",
        test=(
            "tests/test_gate.py::"
            "test_a_track_that_crosses_back_and_forth_still_counts_exactly_once"
        ),
        expect=MUST_FAIL,
        note=(
            "stops consulting the already-counted set. "
            "test_lingering_track_counts_once, the obvious candidate, does NOT "
            "defend this: its follow-up frames stay on one side of the gate, so "
            "signed is already 0 and the memory is never consulted"
        ),
    ),
    Entry(
        claim="the on-line deferral (crossing_direction defers, never double-fires)",
        path="src/trafficlens/core/geometry.py",
        find="    if side_prev == 0 or side_curr == 0:\n        return 0",
        replace="    if side_prev == 0 and side_curr == 0:\n        return 0",
        runner="pytest",
        test="tests/test_geometry.py::test_crossing_direction_start_exactly_on_gate_defers",
        expect=MUST_FAIL,
        note="fires on the on-line frame instead of deferring it",
    ),
    Entry(
        claim="the on-line deferral resolves against the last off-line SIDE",
        path="src/trafficlens/core/gate.py",
        find=(
            "            if last is None or side_curr == 0 or last == side_curr:\n"
            "                signed = 0\n"
            "            else:\n"
            "                signed = side_curr"
        ),
        replace="            signed = 0",
        runner="pytest",
        test="tests/test_gate.py::test_on_line_frame_resolves_against_last_off_line_side",
        expect=MUST_FAIL,
        note="drops the deferred crossing instead of resolving it",
    ),
    Entry(
        claim="the deferred resolution uses the stored last OFF-LINE POINT, not prev",
        path="src/trafficlens/core/gate.py",
        find="            origin = self._last_off_line_point.get(track_id)",
        replace="            origin = prev",
        runner="pytest",
        test=(
            "tests/test_gate.py::"
            "test_the_deferred_resolution_bounds_checks_from_the_last_off_line_point"
        ),
        expect=MUST_FAIL,
        note=(
            "bounds-checks the wrong segment. "
            "test_on_line_deferral_still_respects_gate_bounds does NOT defend "
            "this: in its geometry both origins miss the gate, so it cannot "
            "separate them"
        ),
    ),
    # -- speed -----------------------------------------------------------------
    Entry(
        claim="uncalibrated -> no speed, ever",
        path="src/trafficlens/analytics/speed.py",
        find=(
            "        if self._plane is None:\n"
            "            # The refusal is absolute: even pathological internal state\n"
            "            # cannot make an uncalibrated estimator emit a number.\n"
            "            return None\n\n"
        ),
        replace="",
        runner="pytest",
        test="tests/test_speed.py::test_uncalibrated_refusal_survives_pathological_state",
        expect=MUST_FAIL,
        note="removes the refusal from speed_kmh, leaving only observe's short-circuit",
    ),
    Entry(
        claim="the speed fit is component-wise least squares, not cumulative arc length",
        path="src/trafficlens/analytics/speed.py",
        find=(
            "        for t, wx, wy in samples:\n"
            "            dt = t - t_mean\n"
            "            num_x += dt * (wx - x_mean)\n"
            "            num_y += dt * (wy - y_mean)\n"
            "            den += dt * dt\n"
            "        if den == 0.0:\n"
            "            return None  # all in-window samples share one timestamp\n\n"
            "        return math.hypot(num_x / den, num_y / den) * _MPS_TO_KMH"
        ),
        replace=(
            "        path_m = 0.0\n"
            "        for index in range(1, n):\n"
            "            path_m += math.hypot(\n"
            "                samples[index][1] - samples[index - 1][1],\n"
            "                samples[index][2] - samples[index - 1][2],\n"
            "            )\n"
            "        span_s = samples[-1][0] - samples[0][0]\n"
            "        if span_s == 0.0:\n"
            "            return None  # all in-window samples share one timestamp\n\n"
            "        return (path_m / span_s) * _MPS_TO_KMH"
        ),
        runner="pytest",
        test="tests/test_speed.py::test_stopped_vehicle_with_realistic_noise_reads_near_zero",
        expect=MUST_FAIL,
        note="swaps back to the arc-length design, which rectifies anchor noise",
    ),
    Entry(
        claim="control: the arc-length swap still recovers a straight-line 90 km/h",
        path="src/trafficlens/analytics/speed.py",
        find=(
            "        for t, wx, wy in samples:\n"
            "            dt = t - t_mean\n"
            "            num_x += dt * (wx - x_mean)\n"
            "            num_y += dt * (wy - y_mean)\n"
            "            den += dt * dt\n"
            "        if den == 0.0:\n"
            "            return None  # all in-window samples share one timestamp\n\n"
            "        return math.hypot(num_x / den, num_y / den) * _MPS_TO_KMH"
        ),
        replace=(
            "        path_m = 0.0\n"
            "        for index in range(1, n):\n"
            "            path_m += math.hypot(\n"
            "                samples[index][1] - samples[index - 1][1],\n"
            "                samples[index][2] - samples[index - 1][2],\n"
            "            )\n"
            "        span_s = samples[-1][0] - samples[0][0]\n"
            "        if span_s == 0.0:\n"
            "            return None  # all in-window samples share one timestamp\n\n"
            "        return (path_m / span_s) * _MPS_TO_KMH"
        ),
        runner="pytest",
        test="tests/test_speed.py::test_recovers_90_kmh_within_half_kmh",
        expect=MUST_SURVIVE,
        note=(
            "axis: WHICH property the arc-length swap breaks. On noise-free "
            "straight motion arc length equals the chord, so the accuracy test "
            "must stay green -- the stopped-car test above is the only one that "
            "separates the two designs, and a suite that reddened both would "
            "not have located the difference"
        ),
    ),
    Entry(
        claim="SPEED_MAX_STEP_M rejects a physically impossible step (the loose side)",
        path="src/trafficlens/core/constants.py",
        find="SPEED_MAX_STEP_M = 7.0",
        replace="SPEED_MAX_STEP_M = 70.0",
        runner="pytest",
        test=(
            "tests/test_speed.py::"
            "test_the_outlier_threshold_stays_inside_its_physical_justification"
        ),
        expect=MUST_FAIL,
        note="loosens the outlier bound tenfold; this direction was pinned by nothing",
    ),
    Entry(
        claim="SPEED_MAX_STEP_M admits genuine motion (the tight side)",
        path="src/trafficlens/core/constants.py",
        find="SPEED_MAX_STEP_M = 7.0",
        replace="SPEED_MAX_STEP_M = 3.0",
        runner="pytest",
        test=(
            "tests/test_speed.py::"
            "test_the_outlier_threshold_stays_inside_its_physical_justification"
        ),
        expect=MUST_FAIL,
        note="tightens the bound below the physically possible 4.63 m at 15 fps",
    ),
    # -- the tracker -----------------------------------------------------------
    Entry(
        claim="the tracker's second association stage",
        path="src/trafficlens/track/tracker.py",
        find=(
            "        cost = self._stage2_cost(stage2, low)\n"
            "        matches, _, _ = assign(cost, max_cost)\n"
        ),
        replace="        matches: list[tuple[int, int]] = []\n",
        runner="pytest",
        test="tests/test_tracker.py::test_low_confidence_stage_recovers_an_occluded_track",
        expect=MUST_FAIL,
        note="stage 2 never matches anything, so an occluded track cannot be recovered",
    ),
    Entry(
        claim="the Mahalanobis gate bars an IoU-eligible but implausible pair",
        path="src/trafficlens/track/tracker.py",
        find="            cost[i, gating > KALMAN_GATING_CHI2_95_4DOF] = np.inf",
        replace="            cost[i, gating > np.inf] = np.inf",
        runner="pytest",
        test=(
            "tests/test_tracker.py::"
            "test_mahalanobis_gate_bars_a_floor_eligible_displaced_detection"
        ),
        expect=MUST_FAIL,
        note="widens the chi-square gate to infinity, so nothing is ever barred by it",
    ),
    Entry(
        claim="association never crosses classes",
        path="src/trafficlens/track/tracker.py",
        find=(
            "            cost[i, gating > KALMAN_GATING_CHI2_95_4DOF] = np.inf\n"
            "            for j, det in enumerate(dets):\n"
            "                if det.class_name != rec.track.class_name:\n"
            "                    cost[i, j] = np.inf"
        ),
        replace="            cost[i, gating > KALMAN_GATING_CHI2_95_4DOF] = np.inf",
        runner="pytest",
        test="tests/test_tracker.py::test_cross_class_pair_is_never_matched",
        expect=MUST_FAIL,
        note="removes the stage-1 cross-class bar",
    ),
    Entry(
        claim="the reaping boundary is strictly > max_age (the tight side)",
        path="src/trafficlens/track/tracker.py",
        find="            elif tr.state == STATE_CONFIRMED and tr.time_since_update <= self.max_age:",
        replace="            elif tr.state == STATE_CONFIRMED and tr.time_since_update < self.max_age:",
        runner="pytest",
        test="tests/test_tracker.py::test_dropout_of_exactly_max_age_frames_keeps_id",
        expect=MUST_FAIL,
        note="kills a track at exactly max_age, one frame early",
    ),
    Entry(
        claim="the reaping boundary is strictly > max_age (the loose side)",
        path="src/trafficlens/track/tracker.py",
        find="            elif tr.state == STATE_CONFIRMED and tr.time_since_update <= self.max_age:",
        replace="            elif tr.state == STATE_CONFIRMED and tr.time_since_update <= self.max_age + 1:",
        runner="pytest",
        test="tests/test_tracker.py::test_dropout_of_max_age_plus_one_frames_issues_new_id",
        expect=MUST_FAIL,
        note="keeps a track one frame past max_age",
    ),
    Entry(
        claim="a tentative track dies on its first miss",
        path="src/trafficlens/track/tracker.py",
        find="            elif tr.state == STATE_CONFIRMED and tr.time_since_update <= self.max_age:",
        replace="            elif tr.time_since_update <= self.max_age:",
        runner="pytest",
        test="tests/test_tracker.py::test_tentative_track_dies_on_a_single_miss",
        expect=MUST_FAIL,
        note="lets a tentative track coast like a confirmed one",
    ),
    Entry(
        claim="determinism: new ids follow ascending detection order",
        path="src/trafficlens/track/tracker.py",
        find="        for d_idx in unmatched_high_idx:\n            self._start_track(high[d_idx])",
        replace="        for d_idx in reversed(unmatched_high_idx):\n            self._start_track(high[d_idx])",
        runner="pytest",
        test="tests/test_tracker.py::test_new_track_ids_follow_detection_order",
        expect=MUST_FAIL,
        note="allocates ids in descending detection order",
    ),
    Entry(
        claim="determinism: assign() canonicalises ties rather than trusting the solver",
        path="src/trafficlens/track/associate.py",
        find="    for r, c in _canonical_assignment(solver_cost, best_total):",
        replace="    for r, c in zip(row_ind.tolist(), col_ind.tolist()):",
        runner="pytest",
        test=(
            "tests/test_tracker.py::"
            "test_assign_does_not_inherit_which_optimum_the_solver_returns"
        ),
        expect=MUST_FAIL,
        note=(
            "returns whichever optimum scipy happened to pick, which the "
            "TypeScript mirror cannot inherit. "
            "test_assign_breaks_exact_ties_toward_lowest_indices does NOT "
            "defend this: scipy returns the canonical optimum on its matrices "
            "anyway"
        ),
    ),
    Entry(
        claim="control: the unmatched-row complement may be spelled either way",
        path="src/trafficlens/track/associate.py",
        find="    unmatched_rows = sorted(set(range(n_rows)) - matched_rows)",
        replace="    unmatched_rows = sorted(r for r in range(n_rows) if r not in matched_rows)",
        runner="pytest",
        test="tests/test_tracker.py::test_assign_filters_pairs_above_max_cost",
        expect=MUST_SURVIVE,
        note=(
            "axis: the SPELLING of a set complement (set difference vs filter), "
            "against the must-fails above which change WHICH pairs are chosen. "
            "Provably equivalent: both enumerate range(n_rows) minus "
            "matched_rows and sort, and sorted() is total on ints"
        ),
    ),
    Entry(
        claim="control: the high-confidence threshold may be spelled by De Morgan",
        path="src/trafficlens/track/tracker.py",
        find="        high = [d for d in detections if d.score >= self.high_thresh]",
        replace="        high = [d for d in detections if not (d.score < self.high_thresh)]",
        runner="pytest",
        test="tests/test_tracker.py::test_low_confidence_detection_never_starts_a_track",
        expect=MUST_SURVIVE,
        note=(
            "axis: the SPELLING of a threshold comparison, against the "
            "must-fails' removal of a mechanism. Equivalent for every float "
            "score the detector can produce -- scores are in [0, 1] and never "
            "NaN, the only value for which >= and not-< differ"
        ),
    ),
    # -- homography ------------------------------------------------------------
    Entry(
        claim="homography validation rejects a fit whose metre error is too large",
        path="src/trafficlens/core/homography.py",
        find="        if error[\"mean_m\"] > max_mean_error_m:",
        replace="        if False and error[\"mean_m\"] > max_mean_error_m:",
        runner="pytest",
        test=(
            "tests/test_homography.py::"
            "test_validate_raises_when_holdout_mean_error_exceeds_threshold"
        ),
        expect=MUST_FAIL,
        note="disables the reprojection-error threshold",
    ),
    Entry(
        claim="homography validation rejects a degenerate correspondence set",
        path="src/trafficlens/core/homography.py",
        find="        if condition_number > HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER:",
        replace="        if False and condition_number > HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER:",
        runner="pytest",
        test="tests/test_homography.py::test_validate_raises_on_five_point_fit_with_four_collinear_points",
        expect=MUST_FAIL,
        note="disables the rank/uniqueness diagnostic",
    ),
    Entry(
        claim="validate() refuses a 4-point fit with no holdout",
        path="src/trafficlens/core/homography.py",
        find="        if not has_holdout and n == 4:",
        replace="        if False and not has_holdout and n == 4:",
        runner="pytest",
        test=(
            "tests/test_homography.py::"
            "test_validate_raises_on_four_point_fit_without_holdout_even_when_clean"
        ),
        expect=MUST_FAIL,
        note="restores the self-check that is mathematically incapable of failing",
    ),
    Entry(
        claim="control: the correspondence-count guard may be spelled by negation",
        path="src/trafficlens/core/homography.py",
        find=(
            "        if len(image_pts) != len(world_pts):\n"
            "            raise CalibrationError("
        ),
        replace=(
            "        if not (len(image_pts) == len(world_pts)):\n"
            "            raise CalibrationError("
        ),
        runner="pytest",
        test="tests/test_homography.py::test_from_correspondences_rejects_mismatched_point_counts",
        expect=MUST_SURVIVE,
        note=(
            "axis: the SPELLING of an equality guard, against the homography "
            "must-fails which disable a check entirely. Equivalent because "
            "len() returns an int and int equality is total"
        ),
    ),
    Entry(
        claim="control: the box anchor's midpoint may be summed in either order",
        path="src/trafficlens/core/geometry.py",
        find="    return ((x1 + x2) / 2.0, y2)",
        replace="    return ((x2 + x1) / 2.0, y2)",
        runner="pytest",
        test="tests/test_geometry.py::test_box_anchor_is_bottom_centre",
        expect=MUST_SURVIVE,
        note=(
            "axis: OPERAND ORDER of a commutative float add, against the "
            "geometry must-fail which changes a predicate's branch. "
            "Provably equivalent: IEEE-754 addition is commutative, so "
            "x1 + x2 and x2 + x1 are the same float64 for every input"
        ),
    ),
    Entry(
        claim="control: the direction label may test the sign either way",
        path="src/trafficlens/core/gate.py",
        find="        direction = self.gate.label_positive if signed == 1 else self.gate.label_negative",
        replace="        direction = self.gate.label_positive if signed > 0 else self.gate.label_negative",
        runner="pytest",
        test="tests/test_gate.py::test_opposite_directions_get_correct_labels",
        expect=MUST_SURVIVE,
        note=(
            "axis: the SPELLING of a sign test, against the gate must-fails "
            "which remove a rule. Equivalent because the line is reached only "
            "when signed != 0 and side_of_line returns nothing but -1, 0, +1"
        ),
    ),
    # -- the generated TypeScript mirror --------------------------------------
    Entry(
        claim="the constants-sync guard (generated TypeScript must match the Python source)",
        path="src/trafficlens/core/constants.py",
        find="TRACK_MAX_AGE = 30",
        replace="TRACK_MAX_AGE = 31",
        runner="pytest",
        test=(
            "tests/test_constants_sync.py::"
            "test_generated_constants_are_byte_identical_to_a_fresh_export"
        ),
        expect=MUST_FAIL,
        note="moves a Python constant without regenerating the TypeScript artefact",
    ),
    Entry(
        claim="parity: the TypeScript gate honours the bounded segment too",
        path="web/src/engine/gate.ts",
        find="    if (origin === undefined || !segmentsIntersect(origin, curr, gateA, gateB)) {",
        replace="    if (origin === undefined) {",
        runner="vitest",
        test="src/parity.test.ts::emits the same crossings",
        expect=MUST_FAIL,
        note="the mirror stops bounds-checking, so it disagrees with the Python-recorded fixture",
    ),
    Entry(
        claim="parity: the TypeScript tracker's maxAge retire rule",
        path="web/src/engine/tracker.ts",
        find="      } else if (tr.state === STATE_CONFIRMED && tr.timeSinceUpdate <= this.maxAge) {",
        replace="      } else if (tr.state === STATE_CONFIRMED && tr.timeSinceUpdate < this.maxAge) {",
        runner="vitest",
        test="src/engine/tracker.test.ts::keeps the id across exactly max_age frames",
        expect=MUST_FAIL,
        note="tightening <= to < once survived the whole Vitest suite",
    ),
    Entry(
        claim="control: the TypeScript box anchor may be summed in either order",
        path="web/src/engine/geometry.ts",
        find="  return [(x1 + x2) / 2.0, y2];",
        replace="  return [(x2 + x1) / 2.0, y2];",
        runner="vitest",
        test="src/parity.test.ts::allocates the same track ids on the same frames",
        expect=MUST_SURVIVE,
        note=(
            "axis: OPERAND ORDER of a commutative float add in the MIRROR, "
            "against the parity must-fails which change the mirror's decisions. "
            "Provably equivalent: JavaScript numbers are IEEE-754 doubles and "
            "addition is commutative"
        ),
    ),
    # -- the published documents and reports -----------------------------------
    Entry(
        claim="the docs-numbers pin (a README results cell must match its report)",
        path="README.md",
        find="| `engine+gate` | 0.889 | 0.941 | 0.914 | 18 | 16 | 2 | 1 |",
        replace="| `engine+gate` | 0.889 | 0.941 | 0.915 | 18 | 16 | 2 | 1 |",
        runner="pytest",
        test="tests/test_docs_numbers.py::test_the_counting_table_is_pinned_cell_by_cell",
        expect=MUST_FAIL,
        note="moves a published F1 one step in its last digit",
    ),
    Entry(
        claim="the docs-numbers exemption guard is derived, not hand-listed",
        path="tests/test_docs_numbers.py",
        find=(
            "    by_level = {(key[1],): value for key, value in published.items()}\n"
            "    assert by_level == _expected_robustness_rows(), ("
        ),
        replace=(
            "    by_level = {(key[1],): value for key, value in published.items()}\n"
            "    assert sorted(by_level) == sorted(_expected_robustness_rows()), ("
        ),
        runner="pytest",
        test="tests/test_docs_numbers.py::test_no_numeric_pin_can_be_satisfied_by_a_coincidence",
        expect=KNOWN_OPEN,
        note=(
            "weakens the degradation cell test to compare LEVELS only. The "
            "exemption guard at tests/test_docs_numbers.py:1085-1113 claims its "
            "exemptions are derived from the cell-pin builders; they are not -- "
            "it re-lists the field shapes by hand, so the exemption survives the "
            "coverage it names disappearing. Known open finding, routed to the "
            "final review's fix wave (controller notes section 5); do not fix here"
        ),
    ),
    Entry(
        claim="the committed counting report's headline pin",
        path="reports/counting_accuracy.json",
        find="\"engine+gate\": {\n      \"full\": {\n        \"n_predicted\": 18,",
        replace="\"engine+gate\": {\n      \"full\": {\n        \"n_predicted\": 19,",
        runner="pytest",
        test="tests/test_bench_counting.py::test_the_committed_counting_report_is_self_consistent",
        expect=MUST_FAIL,
        note="moves the headline prediction count off its own true_positives + false_positives",
    ),
    Entry(
        claim="the committed counting report's frame-delta histogram",
        path="reports/counting_accuracy.json",
        find=(
            "          \"mean\": 2.1875,\n"
            "          \"min\": 1,\n"
            "          \"max\": 4,\n"
            "          \"histogram\": {\n"
            "            \"1\": 4,\n"
            "            \"2\": 6,\n"
            "            \"3\": 5,\n"
            "            \"4\": 1\n"
            "          }"
        ),
        replace=(
            "          \"mean\": 2.1875,\n"
            "          \"min\": 1,\n"
            "          \"max\": 4,\n"
            "          \"histogram\": {\n"
            "            \"1\": 4,\n"
            "            \"2\": 6,\n"
            "            \"3\": 5,\n"
            "            \"9\": 1\n"
            "          }"
        ),
        runner="pytest",
        test="tests/test_bench_counting.py::test_the_committed_counting_report_is_self_consistent",
        expect=MUST_FAIL,
        note=(
            "fabricates a matched offset outside the asymmetric window while "
            "leaving every published RATE correct -- the graver half of the "
            "defect this histogram exists to rule out"
        ),
    ),
    Entry(
        claim="the committed robustness report's per-level records",
        path="reports/robustness.json",
        find="\"engine+gate\": {\n              \"n_predicted\": 0,",
        replace="\"engine+gate\": {\n              \"n_predicted\": 1,",
        runner="pytest",
        test=(
            "tests/test_bench_robustness.py::"
            "test_every_published_record_is_arithmetically_self_consistent"
        ),
        expect=MUST_FAIL,
        note=(
            "protocols/frame_rate/entries/5/methods/engine+gate/n_predicted "
            "0 -> 1, the move the README's own pin was measured surviving"
        ),
    ),
    Entry(
        claim="the robustness ablation's verdict prose",
        path="reports/robustness.json",
        find="at 30 fps the loose floor scores slightly WORSE",
        replace="at 30 fps the loose floor scores slightly BETTER",
        runner="pytest",
        test=(
            "tests/test_bench_robustness.py::"
            "test_the_published_ablation_attributes_the_engine_collapse_to_its_iou_floor"
        ),
        expect=KNOWN_OPEN,
        note=(
            "the verdict string is pinned by nothing, and already publishes a "
            "claim its own identity rows contradict (an exact tie at all four "
            "protocols). Known open finding, controller notes section 5; do not "
            "fix here"
        ),
    ),
    Entry(
        claim="the tracking report's swept widest-spread level",
        path="reports/tracking.json",
        find=(
            "\"max_spread\": 1.8235294117647058,\n"
            "        \"widest_spread_level\": \"box_jitter@sigma=2 px\","
        ),
        replace=(
            "\"max_spread\": 1.8235294117647058,\n"
            "        \"widest_spread_level\": \"frame_rate@2 fps\","
        ),
        runner="pytest",
        test=(
            "tests/test_bench_tracking.py::"
            "test_the_swept_summary_fields_are_recomputed_from_the_reports_own_series"
        ),
        expect=MUST_FAIL,
        note="hand-falsifies a swept summary field",
    ),
    Entry(
        claim="the tracking report's swept engine-fragmentation extremes",
        path="reports/tracking.json",
        find=(
            "\"engine_fragmentation_ratio_min\": 0.0,\n"
            "        \"engine_fragmentation_ratio_max\": 2.8823529411764706,"
        ),
        replace=(
            "\"engine_fragmentation_ratio_min\": 0.5,\n"
            "        \"engine_fragmentation_ratio_max\": 2.8823529411764706,"
        ),
        runner="pytest",
        test=(
            "tests/test_bench_tracking.py::"
            "test_the_swept_summary_fields_are_recomputed_from_the_reports_own_series"
        ),
        expect=MUST_FAIL,
        note="hand-falsifies the swept minimum the report publishes",
    ),
    Entry(
        claim="the tracking report's invariants sentence",
        path="reports/tracking.json",
        find=(
            "\"invariants_sentence\": \"What does NOT depend on the half-width, "
            "across the whole swept range: the engine is furthest from one "
            "identity per labelled vehicle at 14-19 of the 15-20 levels"
        ),
        replace=(
            "\"invariants_sentence\": \"What does NOT depend on the half-width, "
            "across the whole swept range: the engine is furthest from one "
            "identity per labelled vehicle at 1-2 of the 15-20 levels"
        ),
        runner="pytest",
        test=(
            "tests/test_bench_tracking.py::"
            "test_the_swept_summary_fields_are_recomputed_from_the_reports_own_series"
        ),
        expect=MUST_FAIL,
        note="hand-falsifies the sentence that states what the sweep does NOT depend on",
    ),
    Entry(
        claim="PROTOCOL.md's scoring rule: probable rows are ignore regions",
        path="data/groundtruth/PROTOCOL.md",
        find="a prediction landing there is **neither credited nor charged**",
        replace="a prediction landing there is **charged as a false positive**",
        runner="pytest",
        test=(
            "tests/test_bench_counting.py::"
            "test_the_protocol_states_the_ignore_region_rule_the_scorer_implements"
        ),
        expect=MUST_FAIL,
        note="inverts the scoring rule the certain-only figure is computed under",
    ),
    Entry(
        claim="PROTOCOL.md's scoring rule: the asymmetric match window",
        path="data/groundtruth/PROTOCOL.md",
        find="`[label - 1, label + 4]`",
        replace="`[label - 2, label + 2]`",
        runner="pytest",
        test="tests/test_bench_counting.py::test_the_protocol_states_the_same_match_window_the_scorer_uses",
        expect=MUST_FAIL,
        note="restores the symmetric window the scorer no longer uses",
    ),
    Entry(
        claim="the timing guard's injected-cost floor",
        path="src/trafficlens/bench/harness.py",
        find="        elapsed = perf_counter() - started\n",
        replace="        elapsed = 0.5 * (perf_counter() - started)\n",
        runner="pytest",
        test=(
            "tests/test_bench_counting.py::"
            "test_the_timing_block_holds_one_measurement_per_method_never_a_sum"
        ),
        expect=MUST_FAIL,
        note="halves every measured time; only the FLOOR against the injected cost can see this",
    ),
    Entry(
        claim="the timing guard's injected-cost ceiling (a sum in disguise)",
        path="src/trafficlens/bench/harness.py",
        find="        events[name] = list(produced)\n",
        replace=(
            "        elapsed += sum(e[\"seconds\"] for e in timing.values())\n"
            "        events[name] = list(produced)\n"
        ),
        runner="pytest",
        test=(
            "tests/test_bench_counting.py::"
            "test_the_timing_block_holds_one_measurement_per_method_never_a_sum"
        ),
        expect=MUST_FAIL,
        note="publishes a running total instead of each method's own bracket",
    ),
    # -- the guards ------------------------------------------------------------
    Entry(
        claim="guard: banned words are caught in every inflection (the scan)",
        path="configs/motorway.yaml",
        find="# Motorway sample: a German autobahn filmed from an overpass, three lanes\n",
        replace=(
            "# Motorway sample: a German autobahn filmed from an overpass, three lanes\n"
            f"# note: the gate layout was {_BANNED_INFLECTION} after the survey.\n"
        ),
        runner="pytest",
        test="tests/test_guards.py::test_no_banned_words_in_tracked_files",
        expect=MUST_FAIL,
        note="puts an -ing inflection of a banned stem into a tracked file",
    ),
    Entry(
        claim="control: a longer word that merely starts with a banned stem is spared",
        path="configs/motorway.yaml",
        find="# Motorway sample: a German autobahn filmed from an overpass, three lanes\n",
        replace=(
            "# Motorway sample: a German autobahn filmed from an overpass, three lanes\n"
            f"# note: the gate layout is {_BANNED_NEAR_MISS} if the survey moves.\n"
        ),
        runner="pytest",
        test="tests/test_guards.py::test_no_banned_words_in_tracked_files",
        expect=MUST_SURVIVE,
        note=(
            "axis: whether the token is an INFLECTION of the banned stem or a "
            "longer, unrelated word beginning with it. The paired must-fail "
            "varies the inflection; this varies the word boundary, and a guard "
            "that had simply dropped its \\b anchors would pass that one and "
            "fail this"
        ),
    ),
    Entry(
        claim="guard: the inflection rule itself, not one spelling of it",
        path="tests/test_guards.py",
        find="_INFLECTIONS = r\"(?:s|es|ed|ing|d)?\"",
        replace="_INFLECTIONS = r\"\"",
        runner="pytest",
        test="tests/test_guards.py::test_the_word_guard_catches_inflections_and_not_unrelated_words",
        expect=MUST_FAIL,
        note="narrows the pattern back to bare stems, the hole that ran green before",
    ),
    Entry(
        claim="guard: no absolute user path in any tracked file",
        path="configs/motorway.yaml",
        find="# Motorway sample: a German autobahn filmed from an overpass, three lanes\n",
        replace=(
            "# Motorway sample: a German autobahn filmed from an overpass, three lanes\n"
            f"# survey notes live in {_ABSOLUTE_PATH}\n"
        ),
        runner="pytest",
        test="tests/test_guards.py::test_no_absolute_user_paths_in_tracked_files",
        expect=MUST_FAIL,
        note="puts a machine-specific absolute path into a tracked file",
    ),
    Entry(
        claim="guard: git check-ignore keeps the private paths out",
        path=".gitignore",
        find="data/samples/*",
        replace="# data/samples/*",
        runner="pytest",
        test="tests/test_guards.py::test_private_paths_are_git_ignored",
        expect=MUST_FAIL,
        note="stops ignoring the sample footage, which would then be committable",
    ),
    Entry(
        claim="guard: git check-ignore keeps the PUBLISHED web assets in",
        path=".gitignore",
        find="\n/models/\n",
        replace="\nmodels/\n",
        runner="pytest",
        test="tests/test_guards.py::test_published_web_assets_are_not_git_ignored",
        expect=MUST_FAIL,
        note=(
            "restores the un-anchored rule that also swallows "
            "web/public/models/, so the browser model would never ship"
        ),
    ),
    Entry(
        claim="guard: the process-doc rule still catches documents",
        path="tests/test_guards.py",
        find="_PROCESS_DOC_SUFFIXES = {\".md\", \".markdown\", \".mdx\", \".txt\", \".rst\",",
        replace="_PROCESS_DOC_SUFFIXES = {\".markdown\", \".mdx\", \".txt\", \".rst\",",
        runner="pytest",
        test=(
            "tests/test_guards.py::"
            "test_the_process_doc_rule_still_catches_documents_and_spares_source"
        ),
        expect=MUST_FAIL,
        note="drops markdown from the process-doc rule's scope",
    ),
    Entry(
        claim="guard: the vendored exclusion stays scoped to web/public/",
        path="tests/test_guards.py",
        find="_VENDORED_DIR = \"web/public/\"",
        replace="_VENDORED_DIR = \"\"",
        runner="pytest",
        test=(
            "tests/test_guards.py::"
            "test_vendored_runtime_assets_stay_excluded_from_the_content_guards"
        ),
        expect=MUST_FAIL,
        note="excuses an authored .mjs anywhere in the tree, not only the vendored copies",
    ),
    Entry(
        claim="guard: authored files under web/public/ stay in scope",
        path="tests/test_guards.py",
        find="    \"docs/assets/\",\n]",
        replace="    \"docs/assets/\",\n    \"web/public/\",\n]",
        runner="pytest",
        test="tests/test_guards.py::test_authored_files_under_web_public_are_scanned",
        expect=MUST_FAIL,
        note="re-broadens the skip list over the whole published publicDir",
    ),
    # -- the browser page ------------------------------------------------------
    Entry(
        claim="a string-keyed report lookup on the page must fail loudly, not print a dash",
        path="web/src/ui/results.ts",
        find="  const engine = REPORTS.counting.methods.find((method) => method.method === \"engine+gate\");\n  return engine?.classConsistency ?? Number.NaN;",
        replace="  const engine = REPORTS.counting.methods.find((method) => method.method === \"engine+gates\");\n  return engine?.classConsistency ?? Number.NaN;",
        runner="vitest",
        test="src/ui/results.test.ts::addressing the bake by name must fail loudly, not print a dash",
        expect=KNOWN_OPEN,
        note=(
            "web/src/ui/results.ts:309-312 falls back to ?? Number.NaN, so a "
            "missed key prints an em dash under a published claim instead of "
            "throwing. Known open finding, controller notes section 5; do not "
            "fix here"
        ),
    ),
]


# Two claims are deliberately NOT in the table above, because neither can be
# expressed as a single-file find/replace:
#
# - ``tests/test_guards.py::test_no_attribution_in_git_history`` scans
#   ``git log --all``. Mutating it needs a history rewrite in a scratch clone,
#   not a file edit, and doing that inside a battery that runs on the real
#   checkout would be reckless.
# - ``tests/test_constants_sync.py::test_every_python_constant_is_exported_with_its_exact_value``
#   defends the exporter against emitting a SUBSET. Every single-line change to
#   ``scripts/export_constants.py`` that drops a constant also makes the
#   exporter refuse the source outright (its own contract checks fire first), so
#   the mutation needed is a multi-hunk edit that both loosens the contract and
#   drops the constant.


# --- running one named test ---------------------------------------------------


def _pytest_env(root: Path, pycache: str) -> dict[str, str]:
    env = dict(os.environ)
    # `src` for this repository's layout, `root` so a sandbox checkout with a
    # flat layout also imports.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # THE FALSE-SURVIVOR HAZARD, and why every run gets a cold, throwaway
    # bytecode cache.
    #
    # CPython validates a cached .pyc against the source's SIZE and its mtime
    # TRUNCATED TO WHOLE SECONDS. A mutation that preserves a file's byte length
    # -- `SPEED_MAX_STEP_M = 7.0` to `= 3.0`, say, or any digit swap -- and that
    # lands in the same wall-clock second as the previous write therefore leaves
    # the cached bytecode of the CLEAN file looking valid, and the mutated source
    # is never compiled. The test then passes on the original code and the row is
    # reported as a survivor that is not one.
    #
    # This was not hypothetical: it produced exactly one such phantom survivor
    # on this table's first run, and would silently have produced more as
    # length-preserving rows were added. Redirecting the cache to a fresh empty
    # directory per invocation, with writing disabled, makes every run compile
    # the bytes that are actually on disk.
    env["PYTHONPYCACHEPREFIX"] = pycache
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_pytest(root: Path, node_id: str) -> tuple[str, str]:
    """Return ("pass" | "fail", detail) for one pytest node id.

    Any other outcome -- a node id that collects nothing, an internal error, a
    usage error -- is a HardError. "the test failed" and "the test never ran"
    must never be confused: the second would score a mutation as caught for the
    wrong reason.
    """
    with tempfile.TemporaryDirectory() as pycache:
        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest", node_id,
                "-q", "--no-header", "-p", "no:cacheprovider", "-x",
            ],
            cwd=root, env=_pytest_env(root, pycache),
            capture_output=True, text=True,
        )
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    detail = tail[-1] if tail else ""
    if proc.returncode == 0:
        return "pass", detail
    if proc.returncode == 1:
        return "fail", detail
    raise HardError(
        f"pytest could not run {node_id!r} (exit {proc.returncode}). A named "
        f"test that does not exist would make every mutation look caught:\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )


def _vitest_counts(root: Path, target: str) -> tuple[int, int, int, int]:
    """Run "<file>::<name filter>" and return
    (matched, failed, suite errors, exit code).

    The JSON reporter is used rather than the exit code alone because Vitest
    exits 0 when a ``-t`` filter matches nothing, skipping every test -- so
    "matched" has to be read, not assumed.
    """
    file_path, _, name = target.partition("::")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "vitest.json"
        proc = subprocess.run(
            [
                "npx", "vitest", "run", file_path, "-t", name,
                "--reporter=json", f"--outputFile={out}",
            ],
            cwd=root / "web", capture_output=True, text=True,
        )
        if not out.is_file():
            raise HardError(
                f"vitest produced no JSON report for {target!r} (exit "
                f"{proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        summary = json.loads(out.read_text())
    return (
        int(summary.get("numTotalTests", 0)),
        int(summary.get("numFailedTests", 0)),
        int(summary.get("numFailedTestSuites", 0)),
        proc.returncode,
    )


def run_vitest(root: Path, target: str) -> tuple[str, str]:
    """Return ("pass" | "fail", detail) for "<file>::<name filter>".

    A zero-test run under a mutation means the module stopped loading, which is
    a genuine red; a zero-test run on the CLEAN tree is a hard error, and
    ``assert_named_test_is_real`` is where that is caught.
    """
    total, failed, suite_failures, returncode = _vitest_counts(root, target)
    detail = f"{total} matched, {failed} failed, {suite_failures} suite errors"
    if total == 0 or failed or suite_failures or returncode != 0:
        return "fail", detail
    return "pass", detail


def run_named_test(root: Path, entry: Entry) -> tuple[str, str]:
    if entry.runner == "pytest":
        return run_pytest(root, entry.test)
    return run_vitest(root, entry.test)


def assert_named_test_is_real(root: Path, entry: Entry) -> str:
    """A named test must collect something AND pass on the clean tree."""
    if entry.runner == "vitest":
        total, failed, suite_failures, _rc = _vitest_counts(root, entry.test)
        _file_path, _, name = entry.test.partition("::")
        if total == 0:
            raise HardError(
                f"{entry.claim}: the Vitest filter {name!r} matches NO test. "
                f"Vitest exits 0 on an empty filter, so this row would score "
                f"every mutation as caught for the wrong reason"
            )
        if failed or suite_failures:
            raise HardError(
                f"{entry.claim}: {entry.test} is already failing on the clean "
                f"tree. Every result from this row would be meaningless"
            )
        return f"{total} matched"

    with tempfile.TemporaryDirectory() as pycache:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", entry.test,
             "--collect-only", "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=root, env=_pytest_env(root, pycache),
            capture_output=True, text=True,
        )
    if proc.returncode != 0:
        raise HardError(
            f"{entry.claim}: pytest cannot collect {entry.test!r}:\n"
            f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
        )
    # `--collect-only -q` prints one "<file>: <count>" line per file on modern
    # pytest and one node id per test on older ones. Accept either, and demand a
    # positive total: a node id that collects nothing exits 5, but a
    # parametrised id that matches no parameter set can exit 0 having collected
    # nothing at all.
    collected = 0
    for line in proc.stdout.splitlines():
        counted = re.match(r"^\S+: (\d+)$", line.strip())
        if counted:
            collected += int(counted.group(1))
        elif "::" in line and not line.startswith(" "):
            collected += 1
    if collected == 0:
        raise HardError(
            f"{entry.claim}: pytest collected NO test for {entry.test!r}. A "
            f"named test that never runs would score every mutation as caught "
            f"for the wrong reason:\n{proc.stdout[-1500:]}"
        )
    outcome, detail = run_pytest(root, entry.test)
    if outcome != "pass":
        raise HardError(
            f"{entry.claim}: {entry.test} is already failing on the clean tree "
            f"({detail}). Every result from this row would be meaningless"
        )
    return f"{collected} collected"


# --- the run ------------------------------------------------------------------


def git_status(root: Path) -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


def preflight_anchors(root: Path, entries: list[Entry]) -> None:
    """Prove every anchor exists exactly once BEFORE anything is written.

    A stale anchor is the defect that made five earlier harnesses in this
    project run green while mutating nothing, and an ambiguous one silently
    mutates more than the row describes. Both stop the run.
    """
    for entry in entries:
        path = root / entry.path
        if not path.is_file():
            raise HardError(f"{entry.claim}: {entry.path} does not exist")
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HardError(
                f"{entry.claim}: {entry.path} is not UTF-8 text, so a "
                f"find/replace over it is not defined ({exc})"
            ) from exc
        occurrences = text.count(entry.find)
        if occurrences == 0:
            raise HardError(
                f"{entry.claim}: the anchor is NOT PRESENT in {entry.path}. "
                f"This row is mutating nothing and would report the claim as "
                f"protected. Re-derive the anchor from the file:\n"
                f"  {entry.find[:200]!r}"
            )
        if occurrences > 1:
            raise HardError(
                f"{entry.claim}: the anchor occurs {occurrences} times in "
                f"{entry.path}. An ambiguous anchor mutates more than this row "
                f"describes; widen it until it is unique:\n"
                f"  {entry.find[:200]!r}"
            )


def apply_mutation(path: Path, entry: Entry, before: bytes) -> bytes:
    text = before.decode("utf-8")
    mutated = text.replace(entry.find, entry.replace).encode("utf-8")
    if mutated == before:
        raise HardError(
            f"{entry.claim}: the mutation changed NO bytes of {entry.path}. "
            f"A no-op mutation reports every claim as protected"
        )
    path.write_bytes(mutated)
    written = path.read_bytes()
    if written != mutated:
        raise HardError(f"{entry.claim}: {entry.path} did not take the mutation")
    return mutated


def restore(path: Path, entry: Entry, before: bytes) -> None:
    path.write_bytes(before)
    after = path.read_bytes()
    if after != before:
        raise HardError(
            f"{entry.claim}: RESTORE FAILED for {entry.path}. The battery has "
            f"changed the repository in a way it did not intend. Stop and "
            f"inspect the file before doing anything else"
        )


def _first_difference(before: bytes, after: bytes) -> int:
    limit = min(len(before), len(after))
    for index in range(limit):
        if before[index] != after[index]:
            return index
    return limit


def _difference_count(before: bytes, after: bytes) -> int:
    limit = min(len(before), len(after))
    differing = sum(1 for i in range(limit) if before[i] != after[i])
    return differing + abs(len(before) - len(after))


def run_entry(root: Path, entry: Entry) -> Result:
    path = root / entry.path
    before = path.read_bytes()
    try:
        mutated = apply_mutation(path, entry, before)
        outcome, detail = run_named_test(root, entry)
    finally:
        restore(path, entry, before)
    clean_outcome, clean_detail = run_named_test(root, entry)
    if clean_outcome != "pass":
        raise HardError(
            f"{entry.claim}: {entry.test} does not pass again after the file "
            f"was restored ({clean_detail}). The restore is byte-identical, so "
            f"something else in the tree has moved"
        )
    return Result(
        entry=entry,
        outcome="caught" if outcome == "fail" else "survived",
        bytes_before=len(before),
        bytes_after=len(mutated),
        detail=detail,
        clean_rerun=clean_detail,
        first_changed_byte=_first_difference(before, mutated),
        differing_bytes=_difference_count(before, mutated),
    )


def load_table(path: Path) -> list[Entry]:
    raw = json.loads(path.read_text())
    return [Entry(**row) for row in raw]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--table", default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    entries = load_table(Path(args.table)) if args.table else list(CLAIMS)
    if args.only:
        entries = [e for e in entries if args.only.lower() in e.claim.lower()]
        if not entries:
            print(f"no claim matches {args.only!r}", file=sys.stderr)
            return 2
        print(
            f"PARTIAL RUN: --only {args.only!r} selected {len(entries)} of "
            f"{len(CLAIMS)} rows. This is a debugging aid and is NOT evidence "
            f"about the claims it skipped.",
            flush=True,
        )

    if args.list:
        for entry in entries:
            print(f"{entry.expect:13s} {entry.claim}\n{'':14s}{entry.path} -> {entry.test}")
        print(f"\n{len(entries)} entries")
        return 0

    try:
        dirty = git_status(root)
        if dirty:
            print(
                "REFUSING TO RUN: the tree is not clean.\n"
                "This battery edits tracked files in place, and a crash "
                "mid-run with uncommitted work present risks losing it. "
                "Commit or stash first.\n\n" + dirty,
                file=sys.stderr,
            )
            return 2

        preflight_anchors(root, entries)

        checked: dict[tuple[str, str], str] = {}
        for entry in entries:
            key = (entry.runner, entry.test)
            if key not in checked:
                checked[key] = assert_named_test_is_real(root, entry)

        results: list[Result] = []
        for index, entry in enumerate(entries, start=1):
            print(f"[{index}/{len(entries)}] {entry.claim}", flush=True)
            result = run_entry(root, entry)
            mark = "ok " if result.ok else "!! "
            print(
                f"      {mark}{result.outcome:9s} expected {entry.expect:12s} "
                f"{result.byte_change_proof}",
                flush=True,
            )
            results.append(result)

        left_dirty = git_status(root)
        if left_dirty:
            raise HardError(
                "the battery left the tree dirty after restoring every file:\n"
                + left_dirty
            )
    except HardError as exc:
        print(f"\nHARD ERROR: {exc}", file=sys.stderr)
        return 2

    return report(results)


def report(results: list[Result]) -> int:
    survivors = [r for r in results if r.entry.expect == MUST_FAIL and not r.ok]
    controls = [r for r in results if r.entry.expect == MUST_SURVIVE and not r.ok]
    closed = [r for r in results if r.entry.expect == KNOWN_OPEN and not r.ok]
    open_findings = [r for r in results if r.entry.expect == KNOWN_OPEN and r.ok]

    print("\n" + "=" * 78)
    print(f"{len(results)} claims mutated")
    print(f"  caught by their named test : "
          f"{sum(1 for r in results if r.outcome == 'caught')}")
    print(f"  must-survive controls green: "
          f"{sum(1 for r in results if r.entry.expect == MUST_SURVIVE and r.ok)}"
          f"/{sum(1 for r in results if r.entry.expect == MUST_SURVIVE)}")
    print(f"  known-open findings still open: {len(open_findings)}")

    if open_findings:
        print("\nKNOWN-OPEN FINDINGS (documented, routed elsewhere):")
        for r in open_findings:
            print(f"  - {r.entry.claim}\n      {r.entry.note}")

    if survivors:
        print("\nSURVIVORS -- these claims have NO test behind them:")
        for r in survivors:
            print(
                f"  - {r.entry.claim}\n"
                f"      mutation : {r.entry.path}  ({r.byte_change_proof})\n"
                f"      named test that stayed GREEN: {r.entry.test}\n"
                f"      the mutation removes: {r.entry.note}"
            )
    if controls:
        print("\nCONTROLS THAT REDDENED -- a semantically-equivalent mutation "
              "broke a test, so that test is pinning a spelling, not a claim:")
        for r in controls:
            print(f"  - {r.entry.claim}\n      {r.entry.test}\n      axis: {r.entry.note}")
    if closed:
        print("\nKNOWN-OPEN FINDINGS THAT ARE NOW CAUGHT -- promote these rows "
              "to must_fail and delete the pointer:")
        for r in closed:
            print(f"  - {r.entry.claim}\n      {r.entry.test}")

    failures = len(survivors) + len(controls) + len(closed)
    if failures:
        print(f"\nFAILED: {failures} row(s) did not behave as the table says.")
        return 1
    print("\nPASSED: every claim in the table is defended by its named test, "
          "every control discriminates, and every known-open finding is still "
          "exactly as documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
