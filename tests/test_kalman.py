"""Tests for the constant-velocity Kalman filter (trafficlens.track.kalman).

The tracker (multi-object association) builds directly on these functions,
so the tests pin down exact behaviour: convergence on a noiseless
constant-velocity target, covariance growth/shrinkage, exact box-format
round-trips, gating discrimination, bit-identical determinism, and exact
covariance symmetry.
"""

import numpy as np
import pytest

from trafficlens.core.constants import (
    KALMAN_GATING_CHI2_95_4DOF,
    KALMAN_STD_WEIGHT_VELOCITY,
)
from trafficlens.track.kalman import KalmanBoxFilter, xyah_to_xyxy, xyxy_to_xyah


@pytest.fixture()
def kf() -> KalmanBoxFilter:
    return KalmanBoxFilter()


def _constant_velocity_boxes(n_steps: int) -> list[np.ndarray]:
    """A target moving at exactly constant velocity, in xyah format.

    Centre starts at (100, 200) and moves (+5, +3) px/frame; the box shape
    (aspect 0.5, height 40) never changes. All values are exact in float64,
    so any residual prediction error is the filter's own, not the data's.
    """
    boxes = []
    for k in range(n_steps):
        boxes.append(np.array([100.0 + 5.0 * k, 200.0 + 3.0 * k, 0.5, 40.0]))
    return boxes


def _track(kf: KalmanBoxFilter, boxes: list[np.ndarray]):
    """Initiate on boxes[0], then predict+update through the rest.

    Returns (per-step predicted means, final mean, final cov). The
    predicted mean at step k is recorded BEFORE the update with boxes[k],
    i.e. it is the filter's genuine one-step-ahead forecast.
    """
    mean, cov = kf.initiate(boxes[0])
    predicted_means = []
    for z in boxes[1:]:
        mean, cov = kf.predict(mean, cov)
        predicted_means.append(mean.copy())
        mean, cov = kf.update(mean, cov, z)
    return predicted_means, mean, cov


# ---------------------------------------------------------------------------
# Box format conversions
# ---------------------------------------------------------------------------

def test_xyxy_to_xyah_values():
    box = np.array([10.0, 20.0, 50.0, 100.0])  # w=40, h=80
    xyah = xyxy_to_xyah(box)
    assert xyah.shape == (4,)
    np.testing.assert_allclose(xyah, [30.0, 60.0, 0.5, 80.0], atol=1e-12)


def test_xyah_to_xyxy_values():
    xyah = np.array([30.0, 60.0, 0.5, 80.0])
    box = xyah_to_xyxy(xyah)
    assert box.shape == (4,)
    np.testing.assert_allclose(box, [10.0, 20.0, 50.0, 100.0], atol=1e-12)


def test_round_trip_xyxy_xyah_xyxy_to_1e9():
    boxes = [
        np.array([10.0, 20.0, 50.0, 100.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([123.4, 567.8, 234.5, 678.9]),
        np.array([-40.0, -80.0, -10.0, -20.0]),  # negative coords, positive area
        np.array([3.7, 9.1, 1919.3, 1079.6]),
    ]
    for box in boxes:
        back = xyah_to_xyxy(xyxy_to_xyah(box))
        np.testing.assert_allclose(back, box, atol=1e-9, rtol=0.0)


def test_round_trip_xyah_xyxy_xyah_to_1e9():
    xyahs = [
        np.array([30.0, 60.0, 0.5, 80.0]),
        np.array([960.0, 540.0, 1.7777777777777777, 200.0]),
        np.array([5.5, 5.5, 3.0, 0.25]),
    ]
    for xyah in xyahs:
        back = xyxy_to_xyah(xyah_to_xyxy(xyah))
        np.testing.assert_allclose(back, xyah, atol=1e-9, rtol=0.0)


def test_degenerate_boxes_raise_value_error():
    # Zero width, zero height, negative width, negative height, fully
    # inverted box: all must fail fast rather than emit NaN/inf aspect
    # ratios that would silently poison the tracker downstream.
    degenerate = [
        np.array([10.0, 20.0, 10.0, 100.0]),  # w == 0
        np.array([10.0, 20.0, 50.0, 20.0]),   # h == 0
        np.array([50.0, 20.0, 10.0, 100.0]),  # w < 0
        np.array([10.0, 100.0, 50.0, 20.0]),  # h < 0
        np.array([50.0, 100.0, 10.0, 20.0]),  # both negative
        np.array([10.0, 20.0, 10.0, 20.0]),   # zero area point box
    ]
    for box in degenerate:
        with pytest.raises(ValueError):
            xyxy_to_xyah(box)


# ---------------------------------------------------------------------------
# Filter core behaviour
# ---------------------------------------------------------------------------

def test_initiate_shapes_and_zero_velocity(kf):
    mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    np.testing.assert_array_equal(mean[:4], [100.0, 200.0, 0.5, 40.0])
    np.testing.assert_array_equal(mean[4:], [0.0, 0.0, 0.0, 0.0])
    # Velocity variance starts high relative to the per-step velocity
    # process noise (well over an order of magnitude), so the zero initial
    # velocity is pure prior and the first measurements dominate it.
    h = 40.0
    per_step_vel_var = (KALMAN_STD_WEIGHT_VELOCITY * h) ** 2
    assert cov[4, 4] > 10.0 * per_step_vel_var
    assert cov[5, 5] > 10.0 * per_step_vel_var


def test_constant_velocity_converges_within_half_pixel(kf):
    n_steps = 20
    boxes = _constant_velocity_boxes(n_steps + 1)
    predicted_means, _, _ = _track(kf, boxes)
    # After 20 predict/update cycles the one-step-ahead predicted centre
    # must sit within 0.5 px of the true centre on both axes.
    final_pred = predicted_means[-1]
    truth = boxes[n_steps]
    assert abs(final_pred[0] - truth[0]) < 0.5
    assert abs(final_pred[1] - truth[1]) < 0.5


def test_covariance_grows_under_predict(kf):
    mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))
    _, cov_pred = kf.predict(mean, cov)
    # Process noise plus velocity coupling: total and positional
    # uncertainty must strictly grow with no measurement in between.
    assert np.trace(cov_pred) > np.trace(cov)
    assert cov_pred[0, 0] > cov[0, 0]
    assert cov_pred[1, 1] > cov[1, 1]


def test_covariance_shrinks_under_update(kf):
    mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))
    mean, cov = kf.predict(mean, cov)
    _, cov_upd = kf.update(mean, cov, np.array([105.0, 203.0, 0.5, 40.0]))
    assert np.trace(cov_upd) < np.trace(cov)
    assert cov_upd[0, 0] < cov[0, 0]
    assert cov_upd[1, 1] < cov[1, 1]


def test_gating_distance_discriminates_true_from_far(kf):
    boxes = _constant_velocity_boxes(11)
    mean, cov = kf.initiate(boxes[0])
    for z in boxes[1:-1]:
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, z)
    mean, cov = kf.predict(mean, cov)

    true_z = boxes[-1]
    far_z = true_z + np.array([200.0, 0.0, 0.0, 0.0])
    d = kf.gating_distance(mean, cov, np.stack([true_z, far_z]))

    assert d.shape == (2,)
    # The true continuation passes the 95% chi-square gate for 4 DOF;
    # a 200 px impostor fails it by a huge margin.
    assert d[0] < KALMAN_GATING_CHI2_95_4DOF
    assert d[1] > KALMAN_GATING_CHI2_95_4DOF
    assert d[1] > 100.0 * d[0]


def test_gating_distance_single_and_batch_agree(kf):
    mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))
    mean, cov = kf.predict(mean, cov)
    zs = np.array([
        [101.0, 201.0, 0.5, 40.0],
        [140.0, 260.0, 0.6, 44.0],
        [400.0, 900.0, 0.5, 40.0],
    ])
    batch = kf.gating_distance(mean, cov, zs)
    singles = np.array([kf.gating_distance(mean, cov, zs[i : i + 1])[0] for i in range(3)])
    np.testing.assert_array_equal(batch, singles)


# ---------------------------------------------------------------------------
# Determinism and numerical hygiene
# ---------------------------------------------------------------------------

def test_bit_identical_determinism_over_20_steps(kf):
    boxes = _constant_velocity_boxes(21)

    def run():
        mean, cov = kf.initiate(boxes[0])
        means, covs = [mean.copy()], [cov.copy()]
        for z in boxes[1:]:
            mean, cov = kf.predict(mean, cov)
            mean, cov = kf.update(mean, cov, z)
            means.append(mean.copy())
            covs.append(cov.copy())
        return np.stack(means), np.stack(covs)

    means_a, covs_a = run()
    means_b, covs_b = run()
    # Exact equality, not allclose: the same input sequence must produce
    # bit-identical output, or the TypeScript parity story falls apart.
    assert np.array_equal(means_a, means_b)
    assert np.array_equal(covs_a, covs_b)


def test_covariances_exactly_symmetric(kf):
    boxes = _constant_velocity_boxes(21)
    mean, cov = kf.initiate(boxes[0])
    assert np.array_equal(cov, cov.T)
    for z in boxes[1:]:
        mean, cov = kf.predict(mean, cov)
        assert np.array_equal(cov, cov.T)
        mean, cov = kf.update(mean, cov, z)
        assert np.array_equal(cov, cov.T)
