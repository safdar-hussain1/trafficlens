"""Road-plane homography: maps image pixels to real-world metres so that a
later stage can turn pixel displacement into a speed in km/h.

Policy -- the reason this module exists at all: **an uncalibrated camera
reports no speed, ever -- never a pixel-derived guess.** ``NO_CALIBRATION``
(an alias for ``None``) is the sentinel a caller passes wherever a
``RoadPlane`` is expected but no survey has been done for this camera. A
later task's ``SpeedEstimator`` takes a ``RoadPlane | None`` and must return
``None`` for every speed when it is handed ``NO_CALIBRATION``, rather than
falling back to a raw pixel-per-frame estimate dressed up as a speed. This
module defines that policy and gives ``SpeedEstimator`` the tool
(``RoadPlane``) to honour it; it does not itself estimate any speeds.

A ``RoadPlane`` is built from surveyed correspondences -- pairs of an image
pixel and the real-world metre position (on the road plane) it depicts --
via ``RoadPlane.from_correspondences``. Building a plane only requires being
able to compute *a* homography; it does not vouch for that homography's
quality. ``RoadPlane.validate()`` is the separate, explicit check a caller
must run (and pass) before trusting the plane: it rejects too few surveyed
points, duplicate points, an ill-conditioned (collinear or near-collinear)
configuration, and a reprojection error -- measured in metres, not pixels --
above a configurable threshold.

Uses ``cv2.findHomography(..., method=0)`` for both the exact 4-point solve
and the least-squares solve for more than 4 points, deliberately never
RANSAC: RANSAC would silently discard some of the user's own surveyed
points, and a calibration tool must not quietly drop the ground truth it
was given.
"""

import math

import cv2
import numpy as np

from trafficlens.core.constants import (
    HOMOGRAPHY_MAX_CONDITION_NUMBER,
    HOMOGRAPHY_MAX_MEAN_ERROR_M,
)
from trafficlens.core.geometry import Point

# An uncalibrated camera reports no speed, ever -- never a pixel-derived
# guess. Pass this (or plain None -- they are the same object) wherever a
# RoadPlane is expected but this camera has not been surveyed.
NO_CALIBRATION = None


class CalibrationError(ValueError):
    """A correspondence set cannot produce a trustworthy RoadPlane: too few
    points, a degenerate/duplicate configuration, an ill-conditioned solve,
    or a fit whose measured reprojection error is too large to trust."""


def _normalization_transform(points: np.ndarray) -> np.ndarray:
    """Hartley normalization: the 3x3 similarity transform that recentres
    ``points`` on their centroid and rescales them so their average distance
    from the centroid is sqrt(2).

    This is the standard preconditioning step for a DLT-style linear solve:
    without it, condition numbers (and to a lesser extent solve accuracy)
    are dominated by the arbitrary difference in scale between pixel
    coordinates (hundreds to thousands) and metre coordinates (single or
    double digits) rather than by genuine geometric degeneracy of the point
    configuration -- which is the thing ``_dlt_condition_number`` below
    actually wants to measure.
    """
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = float(np.mean(np.hypot(shifted[:, 0], shifted[:, 1])))
    scale = math.sqrt(2.0) / mean_dist if mean_dist > 0 else 1.0
    return np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def _dlt_condition_number(image_pts: list[Point], world_pts: list[Point]) -> float:
    """Condition number (largest / smallest singular value) of the
    Hartley-normalized direct-linear-transform design matrix for these
    correspondences.

    This is a diagnostic of the correspondence *configuration*, independent
    of whatever numerical method actually produced a RoadPlane's H: a
    near-degenerate configuration (points close to collinear, or nearly
    duplicated) makes this matrix close to rank-deficient even though
    cv2.findHomography will still return a matrix that fits those exact
    points -- see RoadPlane.validate(), which is the caller of this
    function.
    """
    img = np.asarray(image_pts, dtype=np.float64)
    world = np.asarray(world_pts, dtype=np.float64)

    t_img = _normalization_transform(img)
    t_world = _normalization_transform(world)

    ones = np.ones((len(img), 1))
    img_h = (t_img @ np.hstack([img, ones]).T).T
    world_h = (t_world @ np.hstack([world, ones]).T).T

    rows = []
    for (x, y, _), (big_x, big_y, _) in zip(img_h, world_h):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * big_x, y * big_x, big_x])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * big_y, y * big_y, big_y])
    design = np.array(rows, dtype=np.float64)

    singular_values = np.linalg.svd(design, compute_uv=False)
    smallest = float(singular_values[-1])
    if smallest <= 0.0:
        return math.inf
    return float(singular_values[0] / smallest)


class RoadPlane:
    """A calibrated mapping from image pixels to real-world metres on one
    road plane.

    Build via ``RoadPlane.from_correspondences`` -- not by calling
    ``RoadPlane(...)`` directly, which expects an already-computed
    homography matrix.
    """

    def __init__(
        self,
        image_to_world: np.ndarray,
        image_pts: list[Point],
        world_pts: list[Point],
    ) -> None:
        self._h = image_to_world
        self._image_pts = list(image_pts)
        self._world_pts = list(world_pts)

    @classmethod
    def from_correspondences(
        cls, image_pts: list[Point], world_pts: list[Point]
    ) -> "RoadPlane":
        """Fit a RoadPlane from surveyed image-pixel / world-metre pairs.

        Exactly 4 correspondences give the exact DLT solve; more than 4 give
        the least-squares solve over all of them. Both go through
        ``cv2.findHomography(..., method=0)`` -- never RANSAC, which would
        silently drop some of the surveyed points instead of reporting that
        the fit is poor. Raises ``CalibrationError`` (not merely "returns a
        bad plane") when there are too few points, the counts don't match,
        or cv2 cannot compute a homography at all -- the caller must
        additionally call ``.validate()`` to be sure the fit it did compute
        is trustworthy.
        """
        if len(image_pts) != len(world_pts):
            raise CalibrationError(
                f"image_pts and world_pts must have the same length, got "
                f"{len(image_pts)} and {len(world_pts)}"
            )
        if len(image_pts) < 4:
            raise CalibrationError(
                f"need at least 4 correspondences to compute a homography, "
                f"got {len(image_pts)}"
            )

        src = np.asarray(image_pts, dtype=np.float64)
        dst = np.asarray(world_pts, dtype=np.float64)
        h_matrix, _ = cv2.findHomography(src, dst, method=0)
        if h_matrix is None:
            raise CalibrationError(
                "cv2.findHomography could not compute a homography for "
                "these correspondences (the point configuration is "
                "degenerate)"
            )
        return cls(h_matrix, image_pts, world_pts)

    def to_world(self, p: Point) -> Point:
        """Map one image pixel to its real-world (metres) position on this
        road plane."""
        vec = self._h @ np.array([p[0], p[1], 1.0])
        return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))

    def reprojection_error(
        self, image_pts: list[Point], world_pts: list[Point]
    ) -> dict:
        """Measure this plane's error, in metres, against a set of known
        image/world correspondences.

        Returns ``{"mean_m": float, "max_m": float, "per_point_m": [...]}``
        -- deliberately in metres, not pixels, so a person can read the
        number and judge for themselves whether it is good enough for their
        deployment, rather than trying to mentally convert a pixel error
        into a real-world distance.
        """
        if len(image_pts) != len(world_pts):
            raise ValueError(
                f"image_pts and world_pts must have the same length, got "
                f"{len(image_pts)} and {len(world_pts)}"
            )
        per_point_m = []
        for img_pt, world_pt in zip(image_pts, world_pts):
            wx, wy = self.to_world(img_pt)
            per_point_m.append(math.hypot(wx - world_pt[0], wy - world_pt[1]))

        mean_m = sum(per_point_m) / len(per_point_m) if per_point_m else 0.0
        max_m = max(per_point_m) if per_point_m else 0.0
        return {"mean_m": mean_m, "max_m": max_m, "per_point_m": per_point_m}

    def validate(
        self,
        min_points: int = 4,
        max_mean_error_m: float = HOMOGRAPHY_MAX_MEAN_ERROR_M,
        holdout_image_pts: list[Point] | None = None,
        holdout_world_pts: list[Point] | None = None,
    ) -> None:
        """Raise ``CalibrationError`` unless this plane's fit is
        trustworthy. Returns ``None`` (does not raise) when it is.

        Checks, in order:

        1. At least ``min_points`` correspondences were used to build this
           plane. ``from_correspondences`` already enforces the hard
           mathematical minimum of 4; this lets a caller demand a stronger
           survey (e.g. ``min_points=6``) than the bare minimum.
        2. No duplicate image point and no duplicate world point among the
           fit correspondences.
        3. The (Hartley-normalized) DLT design matrix for the fit
           correspondences is not ill-conditioned -- catches collinear and
           near-collinear configurations, which duplicate points also
           produce as a side effect.
        4. The mean reprojection error is at most ``max_mean_error_m``,
           measured against ``holdout_image_pts``/``holdout_world_pts`` when
           given (a genuine out-of-sample check), or against the plane's
           own fit correspondences otherwise. A single mis-surveyed point
           among an otherwise well-conditioned set passes checks 1-3 but is
           exactly what a held-out check is for -- see
           ``tests/test_homography.py::test_condition_number_check_alone_does_not_catch_a_shifted_single_point``.
        """
        n = len(self._image_pts)
        if n < min_points:
            raise CalibrationError(
                f"only {n} correspondence(s) were used to build this plane, "
                f"need at least {min_points}"
            )

        if len(set(self._image_pts)) != n or len(set(self._world_pts)) != n:
            raise CalibrationError(
                "duplicate correspondence points: every image point and "
                "every world point in the fit must be distinct"
            )

        condition_number = _dlt_condition_number(self._image_pts, self._world_pts)
        if condition_number > HOMOGRAPHY_MAX_CONDITION_NUMBER:
            raise CalibrationError(
                f"ill-conditioned correspondence set: condition number "
                f"{condition_number:.3e} exceeds the maximum of "
                f"{HOMOGRAPHY_MAX_CONDITION_NUMBER:.3e} (points are "
                f"collinear or nearly so)"
            )

        if (holdout_image_pts is None) != (holdout_world_pts is None):
            raise ValueError(
                "holdout_image_pts and holdout_world_pts must be given together"
            )
        check_image_pts = (
            holdout_image_pts if holdout_image_pts is not None else self._image_pts
        )
        check_world_pts = (
            holdout_world_pts if holdout_world_pts is not None else self._world_pts
        )
        error = self.reprojection_error(check_image_pts, check_world_pts)
        if error["mean_m"] > max_mean_error_m:
            raise CalibrationError(
                f"mean reprojection error {error['mean_m']:.4f}m exceeds "
                f"the maximum of {max_mean_error_m:.4f}m"
            )
