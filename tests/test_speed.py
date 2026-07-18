"""Speed estimation against synthetic tracks with known ground truth.

A synthetic camera is built by defining a road-plane rectangle in
metres and its image projection, then *projecting world positions into
image space with the inverse homography*. The estimator only ever sees
image pixels; if it recovers the known world speed, the maths is right.
"""

import numpy as np
import pytest

import cv2

from trafficlens.config import CalibrationConfig
from trafficlens.speed import PlaneCalibration, SpeedEstimator

W, H = 1280.0, 720.0

# A 7 m wide x 40 m long stretch of road as seen by an elevated camera:
# near edge fills the frame bottom, far edge is high and narrow.
IMAGE_QUAD = [(0.25, 0.95), (0.75, 0.95), (0.58, 0.30), (0.42, 0.30)]
WORLD_QUAD = [(0.0, 0.0), (7.0, 0.0), (7.0, 40.0), (0.0, 40.0)]


def make_calibration() -> CalibrationConfig:
    return CalibrationConfig(mode="homography", image_points=IMAGE_QUAD, world_points=WORLD_QUAD)


def world_to_image() -> np.ndarray:
    """Ground-truth projection world->image used to synthesize tracks."""
    img = np.array([(x * W, y * H) for x, y in IMAGE_QUAD], dtype=np.float64)
    world = np.array(WORLD_QUAD, dtype=np.float64)
    m, _ = cv2.findHomography(world, img, method=0)
    return m


def project(m: np.ndarray, p: tuple[float, float]) -> tuple[float, float]:
    v = m @ np.array([p[0], p[1], 1.0])
    return (v[0] / v[2], v[1] / v[2])


class TestPlaneCalibration:
    def test_maps_calibration_corners_exactly(self):
        cal = PlaneCalibration(make_calibration(), int(W), int(H))
        for (ix, iy), world in zip(IMAGE_QUAD, WORLD_QUAD):
            wx, wy = cal.to_world((ix * W, iy * H))
            assert wx == pytest.approx(world[0], abs=1e-6)
            assert wy == pytest.approx(world[1], abs=1e-6)

    def test_degenerate_points_rejected(self):
        collinear = CalibrationConfig(
            mode="homography",
            image_points=[(0.1, 0.5), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5)],
            world_points=WORLD_QUAD,
        )
        with pytest.raises(ValueError, match="degenerate|homography"):
            PlaneCalibration(collinear, int(W), int(H))

    def test_scale_mode(self):
        cal = PlaneCalibration(
            CalibrationConfig(mode="scale", meters_per_pixel=0.05, reference_width=1280),
            1280, 720,
        )
        assert cal.to_world((100.0, 0.0))[0] == pytest.approx(5.0)

    def test_scale_mode_adjusts_for_resolution(self):
        # Same physical scene at 640px wide: each pixel covers twice the metres.
        cal = PlaneCalibration(
            CalibrationConfig(mode="scale", meters_per_pixel=0.05, reference_width=1280),
            640, 360,
        )
        assert cal.to_world((100.0, 0.0))[0] == pytest.approx(10.0)


class TestSpeedEstimator:
    def synthetic_run(self, mps: float, fps: float = 30.0, seconds: float = 3.0,
                      jitter_px: float = 0.0, seed: int = 7) -> float:
        """Drive a track down the road at ``mps``; return the final estimate (km/h)."""
        m = world_to_image()
        est = SpeedEstimator(
            calibration=PlaneCalibration(make_calibration(), int(W), int(H)),
            window_seconds=0.5, smoothing=0.35, unit="kmh",
        )
        rng = np.random.default_rng(seed)
        speed = None
        for i in range(int(seconds * fps)):
            t = i / fps
            world = (3.5, 2.0 + mps * t)  # straight down the lane centre
            px = project(m, world)
            if jitter_px:
                px = (px[0] + rng.normal(0, jitter_px), px[1] + rng.normal(0, jitter_px))
            speed = est.update(1, px, t)
        return speed

    def test_recovers_known_speed_20mps(self):
        # 20 m/s = 72 km/h ground truth
        assert self.synthetic_run(20.0) == pytest.approx(72.0, rel=0.03)

    def test_recovers_known_speed_10mps(self):
        assert self.synthetic_run(10.0) == pytest.approx(36.0, rel=0.03)

    def test_robust_to_pixel_jitter(self):
        # 1.5 px of detector noise must not move the estimate more than ~10%
        assert self.synthetic_run(15.0, jitter_px=1.5) == pytest.approx(54.0, rel=0.10)

    def test_stationary_object_reports_no_speed(self):
        est = SpeedEstimator(
            calibration=PlaneCalibration(make_calibration(), int(W), int(H)),
            window_seconds=0.5, unit="kmh",
        )
        m = world_to_image()
        rng = np.random.default_rng(3)
        px0 = project(m, (3.5, 10.0))
        speed = None
        for i in range(60):
            p = (px0[0] + rng.normal(0, 1.0), px0[1] + rng.normal(0, 1.0))
            speed = est.update(1, p, i / 30.0)
        # jitter alone must not produce a meaningful speed
        assert speed is None or speed < 5.0

    def test_mph_unit(self):
        m = world_to_image()
        est = SpeedEstimator(
            calibration=PlaneCalibration(make_calibration(), int(W), int(H)),
            window_seconds=0.5, smoothing=0.35, unit="mph",
        )
        speed = None
        for i in range(90):
            t = i / 30.0
            speed = est.update(1, project(m, (3.5, 2.0 + 20.0 * t)), t)
        assert speed == pytest.approx(44.7, rel=0.03)  # 20 m/s = 44.7 mph

    def test_forget_clears_state(self):
        est = SpeedEstimator(
            calibration=PlaneCalibration(make_calibration(), int(W), int(H)),
            window_seconds=0.5, unit="kmh",
        )
        m = world_to_image()
        for i in range(30):
            t = i / 30.0
            est.update(1, project(m, (3.5, 2.0 + 20.0 * t)), t)
        est.forget(1)
        assert est.update(1, project(m, (3.5, 30.0)), 2.0) is None
