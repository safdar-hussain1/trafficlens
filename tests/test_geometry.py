"""Tests for trafficlens.core.geometry: exact segment-crossing predicates."""

from trafficlens.core.geometry import (
    box_anchor,
    crossing_direction,
    segment_intersection_param,
    segments_intersect,
    side_of_line,
)


# --- side_of_line: sign convention ------------------------------------------
#
# Pinned convention: +1 = LEFT of the direction of travel a->b, in image
# coordinates where y grows downward. This concrete case must never
# silently flip -- every later component (crossing_direction, and
# eventually the counting logic) reads this sign.

def test_side_of_line_left_is_positive_for_horizontal_gate():
    # Gate travels along +x (rightward). A point with a smaller y sits
    # "above" the gate on screen (since y grows down), which is the LEFT
    # side of a rightward-facing direction of travel.
    a, b = (0.0, 0.0), (10.0, 0.0)
    assert side_of_line(a, b, (5.0, -5.0)) == 1


def test_side_of_line_right_is_negative_for_horizontal_gate():
    a, b = (0.0, 0.0), (10.0, 0.0)
    assert side_of_line(a, b, (5.0, 5.0)) == -1


def test_side_of_line_on_line_is_zero():
    a, b = (0.0, 0.0), (10.0, 0.0)
    assert side_of_line(a, b, (5.0, 0.0)) == 0


def test_side_of_line_endpoint_a_is_on_line():
    a, b = (0.0, 0.0), (10.0, 0.0)
    assert side_of_line(a, b, a) == 0


def test_side_of_line_sign_flips_for_reversed_gate_direction():
    # Reversing the gate direction (b->a instead of a->b) must flip the
    # sign for the same physical point, since "left of travel" depends on
    # which way the gate is defined to face.
    a, b = (0.0, 0.0), (10.0, 0.0)
    p = (5.0, -5.0)
    assert side_of_line(a, b, p) == -side_of_line(b, a, p)


# --- segments_intersect: the awkward cases ----------------------------------

def test_segments_intersect_simple_crossing():
    assert segments_intersect((0.0, -5.0), (0.0, 5.0), (-5.0, 0.0), (5.0, 0.0))


def test_segments_do_not_intersect_when_far_apart():
    assert not segments_intersect((0.0, 0.0), (1.0, 0.0), (5.0, 5.0), (6.0, 7.0))


def test_segments_intersect_collinear_overlap():
    assert segments_intersect((0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (15.0, 0.0))


def test_segments_do_not_intersect_collinear_but_disjoint():
    assert not segments_intersect((0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0))


def test_segments_intersect_shared_endpoint():
    assert segments_intersect((0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (10.0, 10.0))


def test_segments_intersect_t_junction():
    # q1 touches the interior of segment p1-p2, not either of its endpoints.
    assert segments_intersect((0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (5.0, 5.0))


def test_segments_do_not_intersect_when_parallel():
    assert not segments_intersect((0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (10.0, 5.0))


def test_fast_object_still_crosses():
    # 200 px of travel in one frame across a horizontal gate. This is the
    # frame-rate independence guarantee: a fast-moving object that only
    # ever lands far to one side of the gate must still register as a
    # crossing, because the algorithm checks the full swept segment, not
    # just the object's current position.
    assert segments_intersect((100.0, -100.0), (100.0, 100.0), (0.0, 0.0), (200.0, 0.0))


# --- segment_intersection_param ---------------------------------------------

def test_intersection_param_is_midpoint_for_symmetric_crossing():
    t = segment_intersection_param((5.0, -1.0), (5.0, 1.0), (0.0, 0.0), (10.0, 0.0))
    assert abs(t - 0.5) < 1e-12


def test_intersection_param_near_start_of_segment():
    # Off-centre crossing: intersection should land close to the p1 end.
    t = segment_intersection_param((0.0, -1.0), (0.0, 9.0), (-5.0, 0.0), (5.0, 0.0))
    assert abs(t - 0.1) < 1e-12


def test_intersection_param_is_none_for_parallel_segments():
    t = segment_intersection_param((0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (10.0, 5.0))
    assert t is None


def test_intersection_param_is_none_for_collinear_segments():
    # Collinear segments have a zero-magnitude cross-product denominator
    # too (direction vectors are parallel), so this must also return None
    # rather than raise or divide by zero.
    t = segment_intersection_param((0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (15.0, 0.0))
    assert t is None


def test_intersection_param_is_none_for_degenerate_zero_length_segment():
    # p1 == p2: the segment has no direction, so the 2x2 system is singular.
    t = segment_intersection_param((5.0, 5.0), (5.0, 5.0), (0.0, 0.0), (10.0, 0.0))
    assert t is None


# --- crossing_direction ------------------------------------------------------

def test_anchor_exactly_on_gate_defers():
    assert crossing_direction((0.0, 0.0), (10.0, 0.0), (5.0, -1.0), (5.0, 0.0)) == 0


def test_crossing_direction_start_exactly_on_gate_defers():
    assert crossing_direction((0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (5.0, 1.0)) == 0


def test_crossing_direction_no_crossing_same_side_returns_zero():
    assert crossing_direction((0.0, 0.0), (10.0, 0.0), (5.0, -1.0), (5.0, -2.0)) == 0


def test_crossing_direction_left_to_right_is_negative():
    # prev is left of the gate (above), curr is right (below): ends up on
    # the right side, matching side_of_line's -1 for that side.
    assert crossing_direction((0.0, 0.0), (10.0, 0.0), (5.0, -1.0), (5.0, 1.0)) == -1


def test_crossing_direction_right_to_left_is_positive():
    assert crossing_direction((0.0, 0.0), (10.0, 0.0), (5.0, 1.0), (5.0, -1.0)) == 1


# --- box_anchor ---------------------------------------------------------------

def test_box_anchor_is_bottom_centre():
    assert box_anchor(10.0, 20.0, 30.0, 60.0) == (20.0, 60.0)


def test_box_anchor_is_not_box_centre():
    # A regression guard against the common mistake of returning the box
    # centre instead of the bottom-centre (where the object meets the
    # road).
    assert box_anchor(10.0, 20.0, 30.0, 60.0) != (20.0, 40.0)
