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
    HOMOGRAPHY_MAX_MEAN_ERROR_M,
    HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER,
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
    """A rank/uniqueness diagnostic of the correspondence *configuration*,
    computed from the singular values of the Hartley-normalized
    direct-linear-transform design matrix, independent of whatever numerical
    method actually produced a RoadPlane's H.

    The design matrix for N correspondences has 2N rows and 9 columns (one
    column per entry of the homogeneous homography vector h, which solves
    ``design @ h = 0``). Its singular values, largest to smallest, are
    sigma_1 >= sigma_2 >= ... The *smallest* one is always the "solve
    residual" direction: for data that is close to exactly consistent with
    one homography (which includes any noise-free synthetic data, and any
    reasonably clean real survey), that smallest singular value is close to
    zero *by construction* -- it reflects how well a single homography
    explains the data, not whether the point configuration could support a
    *unique* homography at all. Dividing by it, unconditionally, therefore
    conflates "this fit has very little residual error" (a good thing) with
    "this configuration is geometrically degenerate" (a bad thing), and
    ends up penalising precise surveys -- the smaller the pixel noise, the
    smaller that residual singular value, the larger (and more falsely
    alarming) the resulting ratio.

    Whether that conflation matters depends on N:

    - Exactly 4 correspondences: the design matrix is 8x9 (rank at most 8),
      so it has exactly 8 singular values, sigma_1..sigma_8, and there is no
      separate "residual" singular value to exclude -- the system is
      exactly determined, not overdetermined, so sigma_8 itself already
      reflects how well-separated the (unique, if the configuration is
      healthy) null-space direction is from the rest. Using
      sigma_1 / sigma_8 here is correct and this is what a 4-point plane
      uses.
    - 5 or more correspondences: the design matrix is 2Nx9 with N>=5, so it
      has 9 singular values (2N > 9), and sigma_9 -- the smallest -- is that
      residual direction described above. Geometric degeneracy (e.g. most
      of the points collinear) instead shows up in sigma_8, the
      second-smallest: it is only close to zero when the design matrix's
      null space is not one-dimensional, i.e. when more than one homography
      (up to scale) is consistent with the *shape* of the points, which is
      what "the points can't uniquely determine a homography" actually
      means. So for 5+ correspondences this function uses
      sigma_1 / sigma_8 and never looks at sigma_9 at all.

    See RoadPlane.validate(), which is the caller of this function, and
    tests/test_homography.py::test_precise_surveys_are_never_rejected_as_degenerate
    for the regression this distinction fixes.
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
    # Exactly 4 points: only 8 singular values exist at all, and the last
    # one is the meaningful one (see docstring). 5+ points: 9 singular
    # values exist; skip the smallest (the residual direction) and use the
    # second-smallest instead.
    index = -1 if len(image_pts) == 4 else -2
    smallest = float(singular_values[index])
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
        3. The fit correspondences are not geometrically degenerate: the
           rank/uniqueness diagnostic in ``_dlt_condition_number`` (see its
           docstring) is at most ``HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER``.
           This catches collinear and near-collinear configurations, which
           duplicate points also produce as a side effect, without
           penalising a precise (low pixel-noise) survey -- see that
           function's docstring for why the diagnostic differs between
           exactly 4 correspondences and 5 or more.
        4. A reprojection-error self-check -- but only when it can actually
           mean something. A homography has 8 degrees of freedom, and each
           correspondence contributes exactly 2 equations, so exactly 4
           correspondences (8 equations) exactly determine the homography:
           the DLT solve reproduces those same 4 points to floating-point
           precision *no matter what the correspondences were*, including a
           badly corrupted survey (see
           ``tests/test_homography.py::test_validate_raises_on_corrupted_four_point_fit_without_holdout``,
           where a 25px error in one of 4 points still self-checks at
           ~1e-6m). So for a 4-point plane, checking reprojection error
           against its own fit points is not merely weak -- it is
           mathematically incapable of ever failing, which makes it
           decoration, not validation. If no ``holdout_image_pts`` /
           ``holdout_world_pts`` are given for a 4-point plane, this method
           raises ``CalibrationError`` outright, rather than silently
           reporting a pass it cannot back up.

           With 5 or more correspondences, the least-squares solve has
           ``2 * (n - 4)`` more equations than unknowns -- genuine residual
           degrees of freedom -- so checking reprojection error against the
           plane's own fit points *is* informative: a single corrupted
           correspondence among 5+ shows up as nonzero self-fit error (see
           ``test_validate_raises_for_a_corrupted_five_point_fit_without_holdout``).
           For 5+ points, self-checking without a holdout is therefore
           allowed.

           Whenever ``holdout_image_pts``/``holdout_world_pts`` are given
           (any point count, including exactly 4), they are used instead of
           the fit points -- a genuine out-of-sample check, and the
           strongest form of this validation.
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
        if condition_number > HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER:
            raise CalibrationError(
                f"ill-conditioned correspondence set: condition number "
                f"{condition_number:.3e} exceeds the maximum of "
                f"{HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER:.3e} (points are "
                f"collinear or nearly so)"
            )

        if (holdout_image_pts is None) != (holdout_world_pts is None):
            raise ValueError(
                "holdout_image_pts and holdout_world_pts must be given together"
            )
        has_holdout = holdout_image_pts is not None

        if not has_holdout and n == 4:
            raise CalibrationError(
                "validate() was called without holdout_image_pts/"
                "holdout_world_pts on a plane built from exactly 4 "
                "correspondences. 4 points exactly determine a homography "
                "(8 equations for 8 degrees of freedom), so this plane "
                "reproduces those same 4 points to floating-point precision "
                "regardless of whether the survey was correct -- checking "
                "reprojection error against them can never fail, so it "
                "cannot validate anything. Either build the plane from 5 or "
                "more correspondences (a least-squares fit has real "
                "residual error to check), or pass "
                "holdout_image_pts/holdout_world_pts: surveyed points that "
                "were not used to build this plane."
            )

        check_image_pts = holdout_image_pts if has_holdout else self._image_pts
        check_world_pts = holdout_world_pts if has_holdout else self._world_pts
        error = self.reprojection_error(check_image_pts, check_world_pts)
        if error["mean_m"] > max_mean_error_m:
            raise CalibrationError(
                f"mean reprojection error {error['mean_m']:.4f}m exceeds "
                f"the maximum of {max_mean_error_m:.4f}m"
            )
