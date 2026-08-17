"""Tests for the two-stage multi-object tracker (trafficlens.track.tracker)
and its association primitives (trafficlens.track.associate).

Identity errors here become count errors and speed mis-attributions
downstream, and a TypeScript mirror must later reproduce every decision
exactly, so these tests pin down behaviour precisely: ID stability on
straight-line motion, identity preservation through a crossing that pure
IoU would swap, dropout survival on both sides of ``max_age``, the
low-confidence recovery stage (a test that fails if stage two is deleted),
cross-class barring, tentative-track death, reset semantics, and exact
run-to-run determinism.
"""

import numpy as np
import pytest

from trafficlens.core.constants import (
    TRACK_HIGH_CONF,
    TRACK_LOW_CONF,
    TRACK_MATCH_IOU,
    TRACK_MAX_AGE,
    TRACK_MIN_HITS,
)
from trafficlens.detect.base import Detection
from trafficlens.track.associate import assign, iou_matrix
from trafficlens.track.tracker import Track, Tracker


def _det(
    x: float,
    y: float,
    score: float,
    cls: str = "car",
    w: float = 80.0,
    h: float = 80.0,
) -> Detection:
    """A Detection whose box has top-left (x, y) and size w x h."""
    class_ids = {"car": 2, "truck": 7}
    return Detection(
        x1=float(x),
        y1=float(y),
        x2=float(x + w),
        y2=float(y + h),
        score=float(score),
        class_id=class_ids[cls],
        class_name=cls,
    )


# -- associate.iou_matrix ----------------------------------------------------


def test_iou_matrix_exact_values_and_shape():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],  # identical -> 1
            [5.0, 0.0, 15.0, 10.0],  # half-shifted -> 50 / 150 = 1/3
            [20.0, 20.0, 30.0, 30.0],  # disjoint -> 0
        ]
    )
    m = iou_matrix(a, b)
    assert m.shape == (1, 3)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(1.0 / 3.0)
    assert m[0, 2] == 0.0


def test_iou_matrix_empty_inputs():
    empty = np.zeros((0, 4))
    some = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(empty, some).shape == (0, 1)
    assert iou_matrix(some, empty).shape == (1, 0)
    assert iou_matrix(empty, empty).shape == (0, 0)


# -- associate.assign --------------------------------------------------------


def test_assign_filters_pairs_above_max_cost():
    cost = np.array([[0.1, 0.9], [0.9, 0.2]])
    matches, u_rows, u_cols = assign(cost, max_cost=0.5)
    assert matches == [(0, 0), (1, 1)]
    assert u_rows == [] and u_cols == []

    # Tighten the ceiling: the 0.2 pair now exceeds it and must come apart.
    matches, u_rows, u_cols = assign(cost, max_cost=0.15)
    assert matches == [(0, 0)]
    assert u_rows == [1]
    assert u_cols == [1]


def test_assign_handles_barred_inf_pairs():
    # Barred (inf) pairs must never match, even when the Hungarian solver is
    # forced through them to complete a square assignment.
    cost = np.array([[np.inf, 0.3], [0.2, np.inf]])
    matches, u_rows, u_cols = assign(cost, max_cost=0.5)
    assert matches == [(0, 1), (1, 0)]

    all_barred = np.full((2, 2), np.inf)
    matches, u_rows, u_cols = assign(all_barred, max_cost=0.5)
    assert matches == []
    assert u_rows == [0, 1]
    assert u_cols == [0, 1]

    matches, u_rows, u_cols = assign(np.zeros((0, 3)), max_cost=0.5)
    assert matches == []
    assert u_rows == []
    assert u_cols == [0, 1, 2]


def test_assign_breaks_exact_ties_toward_lowest_indices():
    # With tied costs the optimal match SET is not unique, and which
    # optimum a solver returns is implementation-internal -- the TS mirror
    # cannot inherit scipy's choice. assign() therefore canonicalizes: the
    # returned assignment is the lexicographically-(row,col)-least among
    # (near-)optimal assignments, independent of the solver used.
    matches, u_rows, u_cols = assign(np.full((2, 2), 0.5), max_cost=0.6)
    assert matches == [(0, 0), (1, 1)]
    assert u_rows == [] and u_cols == []

    # Three-way tie: canonical result is the identity pairing, which a
    # 2-swap rule alone could not guarantee (rotations tie pairwise).
    matches, _, _ = assign(np.full((3, 3), 0.5), max_cost=0.6)
    assert matches == [(0, 0), (1, 1), (2, 2)]

    # Rectangular ties: WHICH column (or row) goes unmatched is part of
    # the optimum and must be canonical too -- lowest indices match first.
    matches, u_rows, u_cols = assign(np.full((1, 2), 0.5), max_cost=0.6)
    assert matches == [(0, 0)] and u_cols == [1]
    matches, u_rows, u_cols = assign(np.full((2, 1), 0.5), max_cost=0.6)
    assert matches == [(0, 0)] and u_rows == [1]

    # A genuine cost difference is never overridden by the tie rule.
    matches, _, _ = assign(np.array([[0.5, 0.4], [0.4, 0.5]]), max_cost=0.6)
    assert matches == [(0, 1), (1, 0)]


def test_assign_does_not_inherit_which_optimum_the_solver_returns(monkeypatch):
    """The canonical tie rule must be independent of the SOLVER, not merely of
    the input order.

    ``test_assign_breaks_exact_ties_toward_lowest_indices`` above pins the
    canonical answer, but on these matrices scipy happens to return that same
    optimum on its own -- so deleting the reconstruction entirely and returning
    ``linear_sum_assignment``'s raw output left it green. The mutation battery
    found that, and it matters because the whole point of the reconstruction is
    that the TypeScript mirror runs a DIFFERENT solver: a hand-written
    Jonker-Volgenant, which may legitimately return another optimum of the same
    total.

    So the solver is replaced by one that is equally optimal and deliberately
    prefers the OPPOSITE tie -- the highest columns instead of the lowest -- and
    ``assign`` must still produce the lexicographically-(row, col)-least optimum
    that ``associate``'s docstring specifies. The expected answers below are that
    specification, not this run's output.
    """
    import trafficlens.track.associate as associate_module

    real_solver = associate_module.linear_sum_assignment

    def anti_lexicographic(matrix):
        """An optimal solver that resolves ties toward the HIGHEST columns.

        Solving the column-reversed matrix and mapping the columns back gives an
        assignment of exactly the same total -- reversing columns is a
        permutation, which the assignment optimum is invariant under -- while
        among tied optima it lands on the mirror-image pairing.
        """
        matrix = np.asarray(matrix)
        rows, cols = real_solver(matrix[:, ::-1])
        return rows, matrix.shape[1] - 1 - cols

    # The substitution has to be real, or nothing below proves anything: the
    # stand-in must be optimal AND must return the other optimum.
    tied = np.full((2, 2), 0.5)
    rows, cols = anti_lexicographic(tied)
    assert list(zip(rows.tolist(), cols.tolist())) == [(0, 1), (1, 0)], (
        "the stand-in solver returns the same optimum scipy does, so it cannot "
        "show whether assign() inherits the solver's choice"
    )
    baseline_rows, baseline_cols = real_solver(tied)
    assert sum(tied[r, c] for r, c in zip(rows, cols)) == pytest.approx(
        sum(tied[r, c] for r, c in zip(baseline_rows, baseline_cols))
    ), "the stand-in solver is not optimal, so it is not a legitimate substitute"

    monkeypatch.setattr(
        associate_module, "linear_sum_assignment", anti_lexicographic
    )

    matches, u_rows, u_cols = assign(np.full((2, 2), 0.5), max_cost=0.6)
    assert matches == [(0, 0), (1, 1)], matches
    assert u_rows == [] and u_cols == []
    assert assign(np.full((3, 3), 0.5), max_cost=0.6)[0] == [(0, 0), (1, 1), (2, 2)]

    # Rectangular ties: which column goes unmatched is part of the optimum, and
    # is where a solver's own preference would show up most plainly.
    matches, _, u_cols = assign(np.full((1, 2), 0.5), max_cost=0.6)
    assert matches == [(0, 0)] and u_cols == [1]
    matches, u_rows, _ = assign(np.full((2, 1), 0.5), max_cost=0.6)
    assert matches == [(0, 0)] and u_rows == [1]

    # And a genuine cost difference still wins over the tie rule, so the
    # reconstruction is not simply forcing the identity pairing on everything.
    assert assign(np.array([[0.5, 0.4], [0.4, 0.5]]), max_cost=0.6)[0] == [
        (0, 1), (1, 0)
    ]


# -- Tracker: construction contract ------------------------------------------


def test_tracker_defaults_come_from_constants():
    t = Tracker()
    assert t.high_thresh == TRACK_HIGH_CONF
    assert t.low_thresh == TRACK_LOW_CONF
    assert t.match_thresh == TRACK_MATCH_IOU
    assert t.max_age == TRACK_MAX_AGE
    assert t.min_hits == TRACK_MIN_HITS


# -- Tracker: core identity behaviour ----------------------------------------


def test_single_object_keeps_one_id_for_30_frames():
    t = Tracker()  # defaults: min_hits=3, so output starts at frame 2
    outputs = []
    for f in range(30):
        outputs.append(t.update([_det(10 + 5 * f, 10, 0.9)], f))

    # Tentative for the first min_hits - 1 frames, then exactly one
    # confirmed track per frame with a single stable ID.
    assert outputs[0] == [] and outputs[1] == []
    ids = [tr.track_id for frame in outputs[2:] for tr in frame]
    assert len(ids) == 28
    assert set(ids) == {1}

    last = outputs[-1][0]
    assert isinstance(last, Track)
    assert last.state == "confirmed"
    assert last.age == 30
    assert last.hits == 30
    assert last.time_since_update == 0
    # History records the anchor once per real detection update, including
    # the creation frame -- never for coasting frames.
    assert len(last.history) == 30
    # The anchor is the bottom-centre of the (filter-smoothed) box.
    bx = last.box
    assert last.anchor == ((bx[0] + bx[2]) / 2.0, bx[3])


def test_new_track_ids_follow_detection_order():
    t = Tracker(min_hits=1)
    dets = [
        _det(500, 10, 0.9),
        _det(10, 10, 0.9),
        _det(250, 10, 0.9),
    ]
    tracks = t.update(dets, 0)
    assert [tr.track_id for tr in tracks] == [1, 2, 3]
    # IDs follow the detection-list order, not any spatial order.
    assert tracks[0].box[0] == pytest.approx(500.0, abs=1e-6)
    assert tracks[1].box[0] == pytest.approx(10.0, abs=1e-6)
    assert tracks[2].box[0] == pytest.approx(250.0, abs=1e-6)


def test_crossing_paths_keep_identities():
    # Two identical 100x30 boxes on the same row (cy = 300) drive toward
    # each other at +/-10 px/frame: A cx = 100 + 10t, B cx = 290 - 10t.
    # They pass between t=9 and t=10 with a closest approach of 10 px.
    #
    # Why pure IoU WOULD swap them: at t=10 the detections sit at
    # cx 200 (A) and 190 (B), while the last OBSERVED boxes (t=9) sit at
    # cx 190 (A) and 200 (B). IoU of two width-100 boxes offset s px is
    # (100 - s) / (100 + s), so the IoU matrix against the last boxes is
    #
    #                 det A @200   det B @190
    #   last A @190     0.8182       1.0
    #   last B @200     1.0          0.8182
    #
    # Both greedy (takes the two 1.0 entries first) and Hungarian
    # (swapped total 2.0 > correct total 1.6364) pick the SWAPPED pairing,
    # and every entry clears the 0.8 IoU floor, so nothing in the IoU cost
    # alone prevents the swap. The Kalman filter's velocity knowledge
    # does, twice over (values measured with this exact geometry):
    #   1. predicted boxes at t=10 sit at cx 198.9 (A) / 191.1 (B), so the
    #      correct pairs score IoU ~0.98 and Hungarian prefers them;
    #   2. the squared Mahalanobis distance of the swapped pairs is ~11.5
    #      at t=10 (~18.4 at t=9), above the 9.4877 chi-square gate, so
    #      the swapped pairing is barred outright -- while the correct
    #      pairs score ~0.2 and pass easily.
    w, h, cy = 100.0, 30.0, 300.0

    def frame_dets(t: int) -> list[Detection]:
        cx_a = 100.0 + 10.0 * t
        cx_b = 290.0 - 10.0 * t
        return [
            _det(cx_a - w / 2, cy - h / 2, 0.9, w=w, h=h),
            _det(cx_b - w / 2, cy - h / 2, 0.9, w=w, h=h),
        ]

    tracker = Tracker(min_hits=1)
    seen_ids: set[int] = set()
    for t in range(20):
        tracks = tracker.update(frame_dets(t), t)
        by_id = {tr.track_id: tr for tr in tracks}
        seen_ids.update(by_id)
        assert set(by_id) == {1, 2}, f"frame {t}: expected exactly tracks 1 and 2"
        # The filter-smoothed cx stays within ~2.1 px of ground truth
        # (measured), while a swapped identity would be ~10 px off at the
        # closest approach and ~190 px off by the final frame.
        cx_1 = (by_id[1].box[0] + by_id[1].box[2]) / 2.0
        cx_2 = (by_id[2].box[0] + by_id[2].box[2]) / 2.0
        assert cx_1 == pytest.approx(100.0 + 10.0 * t, abs=4.0), f"frame {t}"
        assert cx_2 == pytest.approx(290.0 - 10.0 * t, abs=4.0), f"frame {t}"

    # No fragmentation either: the whole run used exactly two IDs.
    assert seen_ids == {1, 2}


def test_mahalanobis_gate_bars_a_floor_eligible_displaced_detection(monkeypatch):
    # The chi-square gate needs a discriminator the IoU floor cannot
    # provide, and a symmetric crossing cannot either (a swap preferred by
    # predicted-box IoU implies the swapped pairs are CLOSER to the
    # predictions, so IoU and Mahalanobis agree there). The case that
    # separates them: a mature track whose velocity is established, given a
    # single detection displaced from the prediction by an amount that
    # still clears the IoU floor but exceeds the gate.
    #
    # Geometry (identical to the crossing test's boxes): a 100x30 box at
    # cx = 100 + 10t for t = 0..9, then at t = 10 the only detection sits
    # at cx = 190 instead of the predicted ~198.89 -- displaced ~8.9 px.
    # Measured with the real filter on this exact sequence:
    #   IoU(predicted box, displaced box) ~ (100-8.89)/(100+8.89) = 0.8368
    #     -> clears the 0.8 floor (cost 0.1632 <= max_cost 0.2), so the
    #        IoU cost alone would happily match it;
    #   squared Mahalanobis distance = 11.49 > 9.4877
    #     -> the gate bars the pair.
    # (Warm-up frames all pass the gate; the worst is ~8.39 at t=2, while
    # the velocity prior is still settling.)
    w, h, cy = 100.0, 30.0, 300.0

    def det_at(cx: float) -> Detection:
        return _det(cx - w / 2, cy - h / 2, 0.9, w=w, h=h)

    def run() -> list[Track]:
        tracker = Tracker()  # defaults; min_hits=3 keeps the new track internal
        for t in range(10):
            tracker.update([det_at(100.0 + 10.0 * t)], t)
        return tracker.update([det_at(190.0)], 10)

    # Gate active: the displaced detection must NOT continue track 1. It
    # starts a tentative track instead (internal), and track 1 coasts, so
    # the frame's detector-backed output is empty.
    assert run() == []

    # Same sequence with the gate widened to infinity: now nothing bars
    # the pair, the IoU floor alone accepts it, and track 1 IS continued.
    # This half proves the assertion above really is the gate's doing and
    # not the floor's.
    import trafficlens.track.tracker as tracker_module

    monkeypatch.setattr(tracker_module, "KALMAN_GATING_CHI2_95_4DOF", float("inf"))
    assert [tr.track_id for tr in run()] == [1]


# -- Tracker: dropout on both sides of max_age -------------------------------


def _run_dropout(gap: int) -> tuple[int, Track]:
    """Track one object for 5 frames, hide it for ``gap`` frames, then show
    it again on its constant-velocity path. Returns (id_before, the single
    track present on the reappearance frame).
    """
    tracker = Tracker(min_hits=1, max_age=5)

    def det_at(f: int) -> Detection:
        return _det(50 + 4 * f, 100, 0.9, w=60.0, h=60.0)

    id_before = -1
    for f in range(5):
        id_before = tracker.update([det_at(f)], f)[0].track_id

    for f in range(5, 5 + gap):
        assert tracker.update([], f) == []  # nothing detector-backed

    f = 5 + gap
    tracks = tracker.update([det_at(f)], f)
    assert len(tracks) == 1
    return id_before, tracks[0]


def test_dropout_of_max_age_minus_one_frames_keeps_id():
    id_before, track = _run_dropout(gap=4)  # max_age - 1
    assert track.track_id == id_before
    # History grows only on real-detection updates: 5 before the gap plus
    # the reappearance frame. An append-during-coast bug would make it
    # 6 + gap instead.
    assert len(track.history) == 6


def test_dropout_of_exactly_max_age_frames_keeps_id():
    # The death rule is time_since_update > max_age, strictly: a gap of
    # exactly max_age frames must still survive on prediction alone. This
    # pins the boundary a wrong `>=` rule would get past the +/-1 tests.
    id_before, track = _run_dropout(gap=5)  # == max_age
    assert track.track_id == id_before
    assert len(track.history) == 6


def test_dropout_of_max_age_plus_one_frames_issues_new_id():
    id_before, track = _run_dropout(gap=6)  # max_age + 1
    assert track.track_id != id_before
    # A brand-new track: its history starts over with one anchor.
    assert len(track.history) == 1


# -- Tracker: second association stage ---------------------------------------


def test_low_confidence_stage_recovers_an_occluded_track():
    # The occlusion dip (frames 5..9, score 0.3) lands in the
    # low-confidence band, so only the second association stage can keep
    # the track detector-backed through it. Delete stage two and this test
    # goes red: update() returns only tracks updated by a real detection
    # this frame, so the dip frames would return [] and the [0] below
    # would raise IndexError.
    t = Tracker(high_thresh=0.6, low_thresh=0.2, match_thresh=0.8, max_age=30, min_hits=1)
    ids = []
    for f in range(20):
        score = 0.9 if f < 5 or f > 9 else 0.3  # dips into low-confidence band
        ids.append(t.update([_det(10 + 5 * f, 10, score)], f)[0].track_id)
    assert len(set(ids)) == 1


def test_low_confidence_detection_never_starts_a_track():
    t = Tracker(min_hits=1)
    for f in range(10):
        assert t.update([_det(10, 10, 0.3)], f) == []
    # Nothing was created internally either: the next high-confidence
    # detection takes ID 1, the first ID this tracker ever issues.
    tracks = t.update([_det(10, 10, 0.9)], 10)
    assert [tr.track_id for tr in tracks] == [1]


# -- Tracker: class handling -------------------------------------------------


def test_cross_class_pair_is_never_matched():
    t = Tracker(min_hits=1)
    car_box = (100.0, 100.0, 0.9)
    for f in range(3):
        tracks = t.update([_det(*car_box, cls="car")], f)
    assert [tr.track_id for tr in tracks] == [1]

    # A perfectly-overlapping truck detection: IoU 1.0 with the car
    # track's predicted box, but the cross-class bar must refuse the match
    # and start a fresh track instead.
    tracks = t.update([_det(*car_box, cls="truck")], 3)
    assert [(tr.track_id, tr.class_name) for tr in tracks] == [(2, "truck")]

    # The car comes back: same ID, class untouched by the flicker. The
    # truck track coasts (no truck detection), so it is not in the output.
    tracks = t.update([_det(*car_box, cls="car")], 4)
    assert [(tr.track_id, tr.class_name) for tr in tracks] == [(1, "car")]


# -- Tracker: tentative-track lifecycle --------------------------------------


def test_tentative_track_dies_on_a_single_miss():
    t = Tracker(min_hits=3)
    assert t.update([_det(10, 10, 0.9)], 0) == []  # tentative, internal
    assert t.update([], 1) == []  # one miss: the tentative track dies

    # The object shows up again: a NEW track (ID 2) must be built from
    # scratch and needs min_hits consecutive frames to confirm.
    assert t.update([_det(10, 10, 0.9)], 2) == []
    assert t.update([_det(10, 10, 0.9)], 3) == []
    tracks = t.update([_det(10, 10, 0.9)], 4)
    assert [tr.track_id for tr in tracks] == [2]
    assert tracks[0].state == "confirmed"
    assert tracks[0].hits == 3


# -- Tracker: reset ----------------------------------------------------------


def test_reset_restarts_track_ids_at_1():
    t = Tracker(min_hits=1)
    for f in range(3):
        tracks = t.update([_det(10, 10, 0.9), _det(300, 10, 0.9)], f)
    assert [tr.track_id for tr in tracks] == [1, 2]

    t.reset()
    assert t.update([], 0) == []  # no survivors from before the reset
    tracks = t.update([_det(500, 10, 0.9)], 1)
    assert [tr.track_id for tr in tracks] == [1]


# -- Tracker: determinism ----------------------------------------------------


def _mixed_scenario() -> list[list[Detection]]:
    """A deterministic 30-frame scenario mixing everything at once: a
    crossing pair, an object whose score dips into the low band, a dropout
    that comes back inside max_age, and a truck overlapping a car."""
    frames: list[list[Detection]] = []
    for f in range(30):
        dets: list[Detection] = []
        # crossing pair (cars)
        dets.append(_det(50 + 10 * f, 300, 0.9, w=100.0, h=30.0))
        dets.append(_det(340 - 10 * f, 300, 0.85, w=100.0, h=30.0))
        # score dipper
        dip = 0.3 if 8 <= f <= 12 else 0.9
        dets.append(_det(10 + 4 * f, 600, dip, w=60.0, h=60.0))
        # dropout: visible 0..9 and 15.., hidden in between
        if f < 10 or f >= 15:
            dets.append(_det(700, 10 + 6 * f, 0.9, w=70.0, h=70.0))
        # a truck sharing the road with the first car
        dets.append(_det(50 + 10 * f, 350, 0.7, cls="truck", w=90.0, h=40.0))
        frames.append(dets)
    return frames


def _serialize(tracks: list[Track]) -> list[tuple]:
    return [
        (
            tr.track_id,
            tr.class_name,
            tr.box,
            tr.score,
            tr.age,
            tr.hits,
            tr.time_since_update,
            tr.state,
            tuple(tr.history),
        )
        for tr in tracks
    ]


def test_two_identical_runs_produce_identical_id_sequences():
    runs = []
    for _ in range(2):
        tracker = Tracker(min_hits=2)
        out = []
        for f, dets in enumerate(_mixed_scenario()):
            out.append(_serialize(tracker.update(dets, f)))
        runs.append(out)
    # Exact equality: same IDs in the same order with bit-identical boxes,
    # scores, counters and histories on every frame.
    assert runs[0] == runs[1]
