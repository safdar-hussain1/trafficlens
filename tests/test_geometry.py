"""Geometry primitives: the correctness of counting rests on these."""

import pytest

from trafficlens.geometry import crossing_direction, euclidean, segments_intersect, side_of_line

LINE_A, LINE_B = (100.0, 300.0), (600.0, 300.0)  # horizontal gate at y=300


class TestSideOfLine:
    def test_above_and_below_horizontal_line(self):
        assert side_of_line(LINE_A, LINE_B, (350, 100)) == -1
        assert side_of_line(LINE_A, LINE_B, (350, 500)) == 1

    def test_point_on_line_is_zero(self):
        assert side_of_line(LINE_A, LINE_B, (250, 300)) == 0

    def test_sides_are_opposite(self):
        p_up, p_down = (350, 299.0), (350, 301.0)
        assert side_of_line(LINE_A, LINE_B, p_up) == -side_of_line(LINE_A, LINE_B, p_down)


class TestSegmentsIntersect:
    def test_clear_crossing(self):
        assert segments_intersect((300, 100), (300, 500), LINE_A, LINE_B)

    def test_parallel_never_intersect(self):
        assert not segments_intersect((100, 200), (600, 200), LINE_A, LINE_B)

    def test_movement_beyond_gate_end_does_not_cross(self):
        # Crosses y=300 but 100px to the right of the gate's end point.
        assert not segments_intersect((700, 100), (700, 500), LINE_A, LINE_B)

    def test_touching_endpoint_counts_as_intersection(self):
        assert segments_intersect((600, 100), (600, 300), LINE_A, LINE_B)

    def test_collinear_overlap(self):
        assert segments_intersect((50, 300), (200, 300), LINE_A, LINE_B)

    def test_collinear_disjoint(self):
        assert not segments_intersect((700, 300), (900, 300), LINE_A, LINE_B)

    def test_huge_jump_still_intersects(self):
        # 400 px in one frame — the case a band check physically cannot catch.
        assert segments_intersect((350, 100), (350, 500), LINE_A, LINE_B)


class TestCrossingDirection:
    def test_downward_crossing_is_positive(self):
        assert crossing_direction(LINE_A, LINE_B, (300, 250), (300, 350)) == 1

    def test_upward_crossing_is_negative(self):
        assert crossing_direction(LINE_A, LINE_B, (300, 350), (300, 250)) == -1

    def test_no_side_change_is_zero(self):
        assert crossing_direction(LINE_A, LINE_B, (300, 250), (400, 250)) == 0

    def test_landing_exactly_on_line_defers(self):
        assert crossing_direction(LINE_A, LINE_B, (300, 250), (300, 300)) == 0


def test_euclidean():
    assert euclidean((0, 0), (3, 4)) == pytest.approx(5.0)
