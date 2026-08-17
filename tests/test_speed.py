"""Tests for trafficlens.analytics.speed: world-plane speed estimation and
the refusal-to-guess policy -- an uncalibrated camera reports no speed,
ever."""

import math

import numpy as np
import pytest

from trafficlens.analytics.speed import SpeedEstimator, time_of_flight_kmh
from trafficlens.core.constants import SPEED_MAX_STEP_M
from trafficlens.core.homography import RoadPlane


# --- Synthetic ground-truth plane -------------------------------------------
#
# H_TRUE is the exact image -> world (metres) homography of a genuine
# perspective camera looking down a road; it is the same matrix derived in
# tests/test_homography.py (fx = fy = 1000px, principal point (960, 540),
# camera 8m above the road and 5m behind the world origin, tilted 35 degrees
# down, world X = lateral metres, world Y = metres down the road). Building
# a RoadPlane directly from this matrix gives a plane whose to_world is
# exact by construction, so any error a test measures comes from the speed
# estimator, not from a fitted calibration.
H_TRUE = [
    [0.007874647112844963, 0.0, -7.559661228331164],
    [0.0, 0.008548295328846017, -0.9884912148527821],
    [0.0, -0.0008063166600676697, 1.0],
]

# world -> image: the inverse of H_TRUE, so tests can place a target on the
# road in metres and compute the image anchor a detector would have seen.
_H_WORLD_TO_IMAGE = np.linalg.inv(np.array(H_TRUE, dtype=np.float64))


def _plane() -> RoadPlane:
    return RoadPlane(np.array(H_TRUE, dtype=np.float64), [], [])


def _to_image(world_pt: tuple[float, float]) -> tuple[float, float]:
    vec = _H_WORLD_TO_IMAGE @ np.array([world_pt[0], world_pt[1], 1.0])
    return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))


def _drive(
    est: SpeedEstimator,
    track_id: int,
    fps: float,
    n_frames: int,
    speed_mps: float,
    y_start: float = 5.0,
    t_start: float = 0.0,
) -> None:
    """Feed est the image anchors of a target moving straight down the road
    (world X = 0) at exactly speed_mps, one anchor per frame."""
    for f in range(n_frames):
        t = t_start + f / fps
        world = (0.0, y_start + speed_mps * (t - t_start))
        est.observe(track_id, _to_image(world), t)


# --- The defining refusal ----------------------------------------------------


def test_uncalibrated_estimator_never_reports_speed():
    est = SpeedEstimator(plane=None, fps=25.0, window_s=1.0, min_samples=3)
    for f in range(100):
        est.observe(1, (10.0 + 30.0 * f, 500.0), f / 25.0)
    assert est.speed_kmh(1) is None


def test_uncalibrated_refusal_survives_pathological_state():
    # Even if internal per-track state somehow exists (here: injected by
    # hand), plane is None means speed_kmh returns None -- the refusal is a
    # property of speed_kmh itself, not just of observe declining to buffer.
    est = SpeedEstimator(plane=None, fps=25.0, window_s=1.0, min_samples=3)
    est._tracks[1] = [(f / 25.0, 0.0, float(f)) for f in range(50)]
    assert est.speed_kmh(1) is None


def test_uncalibrated_observe_does_not_buffer():
    est = SpeedEstimator(plane=None, fps=25.0, window_s=1.0, min_samples=3)
    for f in range(100):
        est.observe(1, (10.0 + 30.0 * f, 500.0), f / 25.0)
    assert len(est._tracks) == 0


# --- Estimation accuracy -----------------------------------------------------


def test_recovers_90_kmh_within_half_kmh():
    fps = 25.0
    est = SpeedEstimator(plane=_plane(), fps=fps, window_s=2.0, min_samples=5)
    _drive(est, track_id=7, fps=fps, n_frames=51, speed_mps=25.0)  # 90 km/h
    speed = est.speed_kmh(7)
    assert speed is not None
    assert abs(speed - 90.0) < 0.5


def test_time_of_flight_kmh_exact():
    assert time_of_flight_kmh(0.0, 2.0, 50.0) == 90.0


def test_time_of_flight_kmh_rejects_non_positive_dt():
    with pytest.raises(ValueError):
        time_of_flight_kmh(2.0, 2.0, 50.0)
    with pytest.raises(ValueError):
        time_of_flight_kmh(3.0, 2.0, 50.0)


def test_time_of_flight_kmh_rejects_non_positive_distance():
    with pytest.raises(ValueError):
        time_of_flight_kmh(0.0, 2.0, 0.0)
    with pytest.raises(ValueError):
        time_of_flight_kmh(0.0, 2.0, -50.0)


# --- Outlier rejection -------------------------------------------------------


def _drive_with_outlier(est: SpeedEstimator, fps: float) -> None:
    """A 90 km/h run where frame 25's anchor is a wild detector box: its
    plane-space position is 30m off to the side of where the vehicle really
    is -- far beyond SPEED_MAX_STEP_M, so it must be rejected."""
    for f in range(51):
        t = f / fps
        world = (0.0, 5.0 + 25.0 * t)
        if f == 25:
            world = (30.0, world[1])
        est.observe(3, _to_image(world), t)


def test_single_outlier_is_rejected_not_smoothed():
    fps = 25.0
    clean = SpeedEstimator(plane=_plane(), fps=fps, window_s=2.0, min_samples=5)
    _drive(clean, track_id=3, fps=fps, n_frames=51, speed_mps=25.0)
    clean_speed = clean.speed_kmh(3)

    dirty = SpeedEstimator(plane=_plane(), fps=fps, window_s=2.0, min_samples=5)
    _drive_with_outlier(dirty, fps)
    dirty_speed = dirty.speed_kmh(3)

    assert clean_speed is not None and dirty_speed is not None
    # The wild sample is rejected outright, so the only difference from the
    # clean run is one missing good sample -- far tighter than the 2 km/h
    # the task demands.
    assert abs(dirty_speed - clean_speed) < 0.1
    assert abs(dirty_speed - 90.0) < 0.5


def test_stream_recovers_after_outlier():
    # Rejection is measured against the last ACCEPTED sample, so the good
    # samples after the outlier are accepted and the estimate stays correct.
    fps = 25.0
    est = SpeedEstimator(plane=_plane(), fps=fps, window_s=2.0, min_samples=5)
    _drive_with_outlier(est, fps)
    # Continue the clean stream for another full window past the outlier.
    _drive(est, track_id=3, fps=fps, n_frames=50, speed_mps=25.0,
           y_start=5.0 + 25.0 * (51 / fps), t_start=51 / fps)
    speed = est.speed_kmh(3)
    assert speed is not None
    assert abs(speed - 90.0) < 0.5


def test_two_consecutive_outliers_are_both_rejected():
    # The discriminating case between the two rejection policies: two
    # consecutive wild boxes, placed where they have real leverage on the
    # per-axis least-squares fit -- at the NEWEST edge of the window (max
    # |t - t_mean|, so an accepted one steers the slope hardest) and
    # displaced 30m ALONG the motion axis (the axis whose slope IS the
    # speed; a lateral offset would barely touch it). The shipped policy
    # rejects both: each is >SPEED_MAX_STEP_M from the last ACCEPTED
    # sample. A reject-vs-last-RAW policy rejects the first but accepts the
    # second (1m from the first outlier), and that one edge sample drags
    # the estimate several km/h off -- the corruption this test exists to
    # catch.
    fps = 25.0
    clean = SpeedEstimator(plane=_plane(), fps=fps, window_s=2.0, min_samples=5)
    _drive(clean, track_id=3, fps=fps, n_frames=51, speed_mps=25.0)
    clean_speed = clean.speed_kmh(3)

    dirty = SpeedEstimator(plane=_plane(), fps=fps, window_s=2.0, min_samples=5)
    for f in range(51):
        t = f / fps
        world = (0.0, 5.0 + 25.0 * t)
        if f in (49, 50):  # the last two frames: maximum fit leverage
            world = (0.0, 5.0 + 25.0 * t + 30.0)  # 30m ahead, along-track
        dirty.observe(3, _to_image(world), t)
    dirty_speed = dirty.speed_kmh(3)

    assert clean_speed is not None and dirty_speed is not None
    assert abs(dirty_speed - clean_speed) < 0.1
    assert abs(dirty_speed - 90.0) < 0.5


def test_outlier_threshold_is_in_metres():
    # A step just under the threshold is accepted; one just over is not.
    fps = 25.0
    est = SpeedEstimator(plane=_plane(), fps=fps, window_s=10.0, min_samples=2)
    est.observe(9, _to_image((0.0, 10.0)), 0.0)
    est.observe(9, _to_image((0.0, 10.0 + SPEED_MAX_STEP_M - 0.1)), 1.0)
    assert est.speed_kmh(9) is not None  # both accepted -> 2 samples
    est2 = SpeedEstimator(plane=_plane(), fps=fps, window_s=10.0, min_samples=2)
    est2.observe(9, _to_image((0.0, 10.0)), 0.0)
    est2.observe(9, _to_image((0.0, 10.0 + SPEED_MAX_STEP_M + 0.1)), 1.0)
    assert est2.speed_kmh(9) is None  # second sample rejected -> 1 sample


def test_the_outlier_threshold_stays_inside_its_physical_justification():
    """SPEED_MAX_STEP_M is a PHYSICAL bound, and it is bounded on both sides.

    Written because the constant was pinned in one direction only: every test
    above phrases its step in terms of ``SPEED_MAX_STEP_M`` itself, so they all
    move with the constant and loosening it -- 7.0 to 70.0, which would let a
    detector jumping the width of a motorway straight into the fit -- reddened
    nothing. The mutation battery found that.

    The two step sizes here are derived from the physics written down in
    ``core/constants.py`` beside the constant, NOT from the constant's value:

    - 250 km/h is faster than any road vehicle this product will ever see. That
      is 69.44 m/s, and at 15 fps -- the lowest frame rate the constant's own
      comment calls worth running the pipeline at -- one frame of that genuine
      travel is 4.63 m. A step that size is real motion and must be ACCEPTED,
      which is what stops the threshold being tightened until it starts
      throwing away fast traffic.
    - Twice that, 9.26 m in one frame at 15 fps, is 500 km/h. No road vehicle
      does it, so a step that size can only be a detection error and must be
      REJECTED, which is what stops the threshold being loosened until a wild
      box is admitted as motion.

    An identity homography makes the world step exactly the image step, so the
    metres asserted here are the metres the estimator compares.
    """
    identity_plane = RoadPlane(np.eye(3), [], [])
    metres_per_frame_at_250_kmh_15fps = (250.0 / 3.6) / 15.0  # 4.6296... m

    def accepted_samples(step_m: float) -> int:
        est = SpeedEstimator(
            plane=identity_plane, fps=15.0, window_s=10.0, min_samples=2
        )
        est.observe(1, (0.0, 0.0), 0.0)
        est.observe(1, (0.0, step_m), 1.0)
        return len(est._tracks[1])

    assert accepted_samples(metres_per_frame_at_250_kmh_15fps) == 2, (
        f"a step of {metres_per_frame_at_250_kmh_15fps:.4f}m is one frame of "
        f"genuine 250 km/h travel at 15 fps and must never be rejected; "
        f"SPEED_MAX_STEP_M has been tightened past its own justification"
    )
    assert accepted_samples(2.0 * metres_per_frame_at_250_kmh_15fps) == 1, (
        f"a step of {2.0 * metres_per_frame_at_250_kmh_15fps:.4f}m is 500 km/h "
        f"at 15 fps -- a detection error, not a vehicle -- and must be "
        f"rejected; SPEED_MAX_STEP_M has been loosened past its own "
        f"justification"
    )


def test_step_exactly_at_threshold_is_accepted():
    # Rejection is strictly greater-than: a step of exactly
    # SPEED_MAX_STEP_M metres is accepted. An identity-homography plane
    # makes the world step exact (no projection round-trip rounding).
    identity_plane = RoadPlane(np.eye(3), [], [])
    est = SpeedEstimator(plane=identity_plane, fps=25.0, window_s=10.0,
                         min_samples=2)
    est.observe(1, (0.0, 0.0), 0.0)
    est.observe(1, (0.0, SPEED_MAX_STEP_M), 1.0)
    assert est.speed_kmh(1) is not None  # both samples accepted


# --- min_samples boundary ----------------------------------------------------


def test_min_samples_boundary():
    fps = 25.0
    min_samples = 4
    est = SpeedEstimator(
        plane=_plane(), fps=fps, window_s=10.0, min_samples=min_samples
    )
    _drive(est, track_id=5, fps=fps, n_frames=min_samples - 1, speed_mps=25.0)
    assert est.speed_kmh(5) is None
    # One more observation reaches exactly min_samples: a number appears.
    t = (min_samples - 1) / fps
    est.observe(5, _to_image((0.0, 5.0 + 25.0 * t)), t)
    speed = est.speed_kmh(5)
    assert speed is not None
    assert abs(speed - 90.0) < 0.5


def test_unknown_track_is_none():
    est = SpeedEstimator(plane=_plane(), fps=25.0, window_s=2.0, min_samples=5)
    assert est.speed_kmh(42) is None


# --- Noise floor -------------------------------------------------------------


def test_stopped_vehicle_with_realistic_noise_reads_near_zero():
    # A STOPPED vehicle whose anchor jitters with per-axis Gaussian world
    # noise (sigma = 2cm) must read near-zero speed. This is what rules out
    # fitting cumulative arc length: arc length rectifies noise -- every
    # jitter step adds positive path length -- turning a stationary target
    # into a deterministic phantom speed of several km/h. Per-axis slopes
    # see zero-mean noise and fit ~0 on both axes.
    fps = 30.0
    window_s = 2.0
    rng = np.random.default_rng(42)
    est = SpeedEstimator(plane=_plane(), fps=fps, window_s=window_s,
                         min_samples=5)
    n_frames = int(window_s * fps) + 1  # a full window of samples
    for f in range(n_frames):
        noise_x, noise_y = rng.normal(0.0, 0.02, size=2)
        est.observe(4, _to_image((0.0 + noise_x, 10.0 + noise_y)), f / fps)
    speed = est.speed_kmh(4)
    assert speed is not None
    assert speed < 1.0


# --- Window expiry -----------------------------------------------------------


def test_window_expiry_stopped_then_moving():
    # A vehicle stopped for 5s then moving at 90 km/h for 3s reads ~90, not
    # some average dragged down by ancient stationary history: samples older
    # than window_s have fallen out of the window.
    fps = 25.0
    est = SpeedEstimator(plane=_plane(), fps=fps, window_s=2.0, min_samples=5)
    stop_frames = int(5.0 * fps)
    for f in range(stop_frames):
        est.observe(8, _to_image((0.0, 5.0)), f / fps)
    stopped = est.speed_kmh(8)
    assert stopped is not None
    assert abs(stopped) < 0.5  # reads as stationary while stopped
    _drive(est, track_id=8, fps=fps, n_frames=int(3.0 * fps), speed_mps=25.0,
           y_start=5.0, t_start=stop_frames / fps)
    speed = est.speed_kmh(8)
    assert speed is not None
    assert abs(speed - 90.0) < 0.5


# --- Constructor validation --------------------------------------------------


def test_constructor_rejects_bad_parameters():
    with pytest.raises(ValueError):
        SpeedEstimator(plane=_plane(), fps=0.0)
    with pytest.raises(ValueError):
        SpeedEstimator(plane=_plane(), fps=-25.0)
    with pytest.raises(ValueError):
        SpeedEstimator(plane=_plane(), fps=25.0, window_s=0.0)
    with pytest.raises(ValueError):
        SpeedEstimator(plane=_plane(), fps=25.0, window_s=-1.0)
    with pytest.raises(ValueError):
        SpeedEstimator(plane=_plane(), fps=25.0, min_samples=1)


# --- Lifecycle and determinism -----------------------------------------------


def test_forget_clears_state():
    fps = 25.0
    est = SpeedEstimator(plane=_plane(), fps=fps, window_s=10.0, min_samples=3)
    _drive(est, track_id=2, fps=fps, n_frames=10, speed_mps=25.0)
    assert est.speed_kmh(2) is not None
    est.forget(2)
    assert est.speed_kmh(2) is None
    # A recycled track ID starts from scratch: two fresh samples are still
    # under min_samples even though ten were observed before the forget.
    _drive(est, track_id=2, fps=fps, n_frames=2, speed_mps=25.0)
    assert est.speed_kmh(2) is None


def test_forget_unknown_track_is_a_no_op():
    est = SpeedEstimator(plane=_plane(), fps=25.0, window_s=2.0, min_samples=5)
    est.forget(999)  # must not raise
    assert est.speed_kmh(999) is None


def test_determinism_same_sequence_same_floats():
    fps = 25.0

    def run() -> float:
        est = SpeedEstimator(
            plane=_plane(), fps=fps, window_s=2.0, min_samples=5
        )
        for f in range(51):
            t = f / fps
            # Deterministic zig-zag jitter on top of straight motion.
            wobble = 0.3 * math.sin(2.0 * math.pi * f / 7.0)
            est.observe(6, _to_image((wobble, 5.0 + 25.0 * t)), t)
        speed = est.speed_kmh(6)
        assert speed is not None
        return speed

    assert run() == run()
