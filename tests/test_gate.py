"""Tests for trafficlens.core.gate: directional gate counting, once per
track, per class, per direction."""

import dataclasses

import pytest

from trafficlens.core.gate import CrossingEvent, Gate, GateCounter, is_over_limit


# --- once-per-track counting -------------------------------------------------

def test_lingering_track_counts_once():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    assert g.update(1, "car", (5.0, -2.0), (5.0, 2.0), 1, 0.04) is not None
    for f in range(2, 20):
        assert g.update(1, "car", (5.0, 2.0), (5.0, 2.0 + f * 0.01), f, f * 0.04) is None
    assert g.total() == 1


def test_touch_and_retreat_never_fires():
    # Anchor touches the gate line exactly, then retreats back to the
    # side it came from: never a genuine crossing.
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    assert g.update(1, "car", (5.0, -2.0), (5.0, 0.0), 1, 0.04) is None
    assert g.update(1, "car", (5.0, 0.0), (5.0, -1.0), 2, 0.08) is None
    assert g.total() == 0


# --- the on-line deferral ----------------------------------------------------

def test_on_line_frame_resolves_against_last_off_line_side():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    assert g.update(1, "car", (5.0, -2.0), (5.0, 0.0), 1, 0.04) is None   # deferred
    ev = g.update(1, "car", (5.0, 0.0), (5.0, 2.0), 2, 0.08)               # resolves
    assert ev is not None and ev.signed_direction == -1


def test_forget_clears_last_side_state():
    # If forget() did not clear _last_side, this on-line-prev frame would
    # wrongly resolve against the stale side remembered before forget().
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    assert g.update(1, "car", (5.0, -2.0), (5.0, 0.0), 1, 0.04) is None
    g.forget(1)
    assert g.update(1, "car", (5.0, 0.0), (5.0, 2.0), 2, 0.08) is None


# --- direction labels ---------------------------------------------------------

def test_opposite_directions_get_correct_labels():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    ev_out = g.update(1, "car", (5.0, -1.0), (5.0, 1.0), 1, 0.04)
    ev_in = g.update(2, "car", (5.0, 1.0), (5.0, -1.0), 1, 0.04)
    assert ev_out is not None and ev_out.signed_direction == -1
    assert ev_out.direction == "out"
    assert ev_in is not None and ev_in.signed_direction == 1
    assert ev_in.direction == "in"


def test_custom_gate_labels_are_used():
    gate = Gate("g", (0.0, 0.0), (10.0, 0.0), label_positive="north", label_negative="south")
    g = GateCounter(gate)
    ev = g.update(1, "car", (5.0, 1.0), (5.0, -1.0), 1, 0.04)
    assert ev is not None and ev.direction == "north"


# --- forget() recycling -------------------------------------------------------

def test_forget_allows_recycled_track_id_to_count_again():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    ev1 = g.update(1, "car", (5.0, -2.0), (5.0, 2.0), 1, 0.04)
    assert ev1 is not None
    assert g.total() == 1
    # Same id crosses back without forget: must not count again.
    assert g.update(1, "car", (5.0, 2.0), (5.0, -2.0), 2, 0.08) is None
    g.forget(1)
    ev2 = g.update(1, "car", (5.0, -2.0), (5.0, 2.0), 3, 0.12)
    assert ev2 is not None
    assert g.total() == 2


# --- totals --------------------------------------------------------------------

def test_totals_accumulate_per_class_and_direction():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    g.update(1, "car", (5.0, -1.0), (5.0, 1.0), 1, 0.04)    # out
    g.update(2, "car", (5.0, -1.0), (5.0, 1.0), 1, 0.04)    # out
    g.update(3, "truck", (5.0, 1.0), (5.0, -1.0), 1, 0.04)  # in
    assert g.totals == {"car": {"out": 2}, "truck": {"in": 1}}
    assert g.total() == 3


# --- CrossingEvent shape --------------------------------------------------------

def test_crossing_event_is_frozen():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    ev = g.update(1, "car", (5.0, -1.0), (5.0, 1.0), 1, 0.04)
    assert ev is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.track_id = 99


def test_event_fields_recorded_verbatim():
    g = GateCounter(Gate("mygate", (0.0, 0.0), (10.0, 0.0)))
    ev = g.update(7, "bus", (5.0, -1.0), (5.0, 1.0), 42, 1.68)
    assert ev is not None
    assert ev.track_id == 7
    assert ev.class_name == "bus"
    assert ev.gate == "mygate"
    assert ev.frame_index == 42
    assert ev.timestamp == 1.68


def test_crossing_point_is_the_actual_intersection_centred():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    ev = g.update(1, "car", (5.0, -2.0), (5.0, 2.0), 1, 0.04)
    assert ev is not None
    assert ev.crossing_x == pytest.approx(5.0)
    assert ev.crossing_y == pytest.approx(0.0)


def test_crossing_point_off_centre():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    ev = g.update(1, "car", (2.0, -1.0), (2.0, 9.0), 1, 0.04)
    assert ev is not None
    assert ev.crossing_x == pytest.approx(2.0)
    assert ev.crossing_y == pytest.approx(0.0)


# --- Gate construction and validation -------------------------------------------

def test_gate_rejects_zero_length_start_equals_end():
    with pytest.raises(ValueError):
        Gate("g", (5.0, 5.0), (5.0, 5.0))


def test_gate_from_normalized_converts_to_pixel_coordinates():
    g = Gate.from_normalized("g", (0.0, 0.5), (1.0, 0.5), width=1920, height=1080)
    assert g.start == (0.0, 540.0)
    assert g.end == (1920.0, 540.0)
    assert g.name == "g"


def test_gate_from_normalized_passes_through_kwargs():
    g = Gate.from_normalized(
        "g", (0.0, 0.0), (1.0, 1.0), width=100, height=100,
        label_positive="north", label_negative="south", expected_direction="north",
    )
    assert g.label_positive == "north"
    assert g.label_negative == "south"
    assert g.expected_direction == "north"


def test_gate_from_normalized_rejects_zero_length():
    with pytest.raises(ValueError):
        Gate.from_normalized("g", (0.5, 0.5), (0.5, 0.5), width=100, height=100)


def test_gate_from_normalized_rejects_coordinate_above_one():
    with pytest.raises(ValueError):
        Gate.from_normalized("g", (0.0, 0.0), (1.5, 1.0), width=100, height=100)


def test_gate_from_normalized_rejects_negative_coordinate():
    with pytest.raises(ValueError):
        Gate.from_normalized("g", (-0.1, 0.0), (1.0, 1.0), width=100, height=100)


# --- is_over_limit -------------------------------------------------------------

def test_is_over_limit_true_when_strictly_greater():
    assert is_over_limit(60.0, 50.0) is True


def test_is_over_limit_false_when_exactly_equal():
    assert is_over_limit(50.0, 50.0) is False


def test_is_over_limit_false_when_under():
    assert is_over_limit(40.0, 50.0) is False


def test_is_over_limit_false_when_speed_is_none():
    assert is_over_limit(None, 50.0) is False


def test_is_over_limit_false_when_limit_is_none():
    assert is_over_limit(60.0, None) is False


def test_is_over_limit_false_when_both_none():
    assert is_over_limit(None, None) is False


# --- GateCounter <-> is_over_limit wiring ---------------------------------------

def test_update_flags_violation_when_speed_exceeds_limit():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    ev = g.update(1, "car", (5.0, -1.0), (5.0, 1.0), 1, 0.04, speed_kmh=80.0, speed_limit_kmh=50.0)
    assert ev is not None
    assert ev.is_violation is True
    assert ev.speed_kmh == 80.0


def test_update_does_not_flag_violation_when_speed_equals_limit():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    ev = g.update(1, "car", (5.0, -1.0), (5.0, 1.0), 1, 0.04, speed_kmh=50.0, speed_limit_kmh=50.0)
    assert ev is not None
    assert ev.is_violation is False


def test_update_no_violation_when_speed_and_limit_missing():
    g = GateCounter(Gate("g", (0.0, 0.0), (10.0, 0.0)))
    ev = g.update(1, "car", (5.0, -1.0), (5.0, 1.0), 1, 0.04)
    assert ev is not None
    assert ev.speed_kmh is None
    assert ev.is_violation is False
