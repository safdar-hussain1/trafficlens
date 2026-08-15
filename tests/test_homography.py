"""Tests for trafficlens.core.homography: road-plane calibration, held-out
reprojection validation, and the refusal-to-guess policy for an
uncalibrated camera."""

import pytest

from trafficlens.core.homography import (
    CalibrationError,
    NO_CALIBRATION,
    RoadPlane,
    _dlt_condition_number,
)
from trafficlens.core.constants import (
    HOMOGRAPHY_MAX_CONDITION_NUMBER,
    HOMOGRAPHY_MAX_MEAN_ERROR_M,
)


# --- Synthetic ground-truth homography fixtures -----------------------------
#
# IMG4 / WORLD4 / IMG_HOLDOUT / WORLD_HOLDOUT below are not arbitrary
# numbers: each is the exact image-plane projection of a chosen world-plane
# (road) point, in metres, through a genuine perspective camera model -- not
# a pure scale or affine map -- so a test built from them actually exercises
# perspective homography recovery, not a degenerate special case.
#
# Derivation (reproducible with numpy): for a plane Z=0 in world metres, the
# world -> image mapping is H_wi = K @ [r1 r2 t], where r1, r2 are the first
# two columns of the camera's rotation matrix R (camera axes expressed in
# world coordinates) and t = -R @ camera_position_world. The parameters used:
#
#   - intrinsics: fx = fy = 1000.0 px, principal point (cx, cy) = (960, 540)
#   - camera position in world metres: (X=0, Y=-5, Z=8), i.e. 5m behind the
#     world origin and 8m above the road
#   - downward tilt from horizontal: 35 degrees, looking forward (+Y, down
#     the road) and down at the road surface
#   - world axes: X = lateral offset across the road (m), Y = distance down
#     the road from the origin (m); the road surface is the plane Z=0
#
# H_true = inverse(H_wi), normalized so H_true[2][2] == 1, is (image ->
# world, up to the usual homogeneous scale):
#
#   [[ 0.007874647112844963,  0.0,                  -7.559661228331164 ],
#    [ 0.0,                   0.008548295328846017,  -0.9884912148527821],
#    [ 0.0,                  -0.0008063166600676697,  1.0               ]]
#
# IMG4 is H_wi applied to WORLD4 (a trapezoid marked on the road: two points
# 5m out, two points 30m out, straddling the centre line at +-3.5m -- a
# realistic surveyed-lane-marking layout). IMG_HOLDOUT/WORLD_HOLDOUT are two
# different road points, also passed through H_wi, that are never given to
# RoadPlane.from_correspondences -- so reprojecting them is a genuine
# out-of-sample measurement, not a check against the fitted points.
IMG4 = [
    (686.1374124964, 476.0372807540398),
    (1233.8625875036003, 476.0372807540398),
    (854.7651148131101, 946.5662269242592),
    (1065.2348851868899, 946.5662269242592),
]
WORLD4 = [(-3.5, 5.0), (3.5, 5.0), (-3.5, 30.0), (3.5, 30.0)]

IMG_HOLDOUT = [
    (878.9810899563789, 712.7097963942266),
    (999.8924299063265, 850.6102205009837),
]
WORLD_HOLDOUT = [(-1.5, 12.0), (1.0, 20.0)]

# Three of these four world points are exactly collinear (all on the line
# X = -3.5); the fourth is off the line. This is the textbook degenerate
# configuration a homography solve cannot resolve -- image points below are
# the same H_wi projection of these world points.
COLLINEAR_WORLD = [(-3.5, 5.0), (-3.5, 15.0), (-3.5, 30.0), (3.5, 20.0)]
IMG_COLLINEAR = [
    (686.1374124964, 476.0372807540398),
    (793.1080480855557, 774.5219291446756),
    (854.7651148131101, 946.5662269242592),
    (1099.6235046721424, 850.6102205009837),
]

# Same as WORLD4/IMG4 except the 4th point is nudged 1mm off the line
# X = -3.5 -- not exactly collinear, but close enough that the DLT solve is
# numerically unstable, even though it still fits its own 4 points exactly.
NEAR_DEGENERATE_WORLD = [(-3.5, 5.0), (3.5, 5.0), (-3.5, 30.0), (-3.499, 30.0)]
IMG_NEAR_DEGENERATE = [
    (686.1374124964, 476.0372807540398),
    (1233.8625875036003, 476.0372807540398),
    (854.7651148131101, 946.5662269242592),
    (854.7951819231635, 946.5662269242592),
]

# WORLD4/IMG4 with the second correspondence replaced by a duplicate of the
# first.
DUPLICATE_WORLD = [(-3.5, 5.0), (-3.5, 5.0), (-3.5, 30.0), (3.5, 30.0)]
DUPLICATE_IMG = [
    (686.1374124964, 476.0372807540398),
    (686.1374124964, 476.0372807540398),
    (854.7651148131101, 946.5662269242592),
    (1065.2348851868899, 946.5662269242592),
]

# All four image points identical: cv2.findHomography itself returns None
# for this input (verified interactively) -- lesser degeneracies (collinear
# points, a single duplicated pair) still produce a (numerically unstable)
# matrix, so this is the only genuine trigger for the "cv2 returned None"
# branch.
ALL_SAME_IMG = [IMG4[0]] * 4


# --- The refusal policy ------------------------------------------------------

def test_no_calibration_sentinel_documents_refusal_policy():
    import trafficlens.core.homography as homography_module

    assert NO_CALIBRATION is None
    assert homography_module.__doc__ is not None
    assert "never a pixel-derived guess" in homography_module.__doc__


# --- CalibrationError type -------------------------------------------------

def test_calibration_error_is_a_value_error():
    assert issubclass(CalibrationError, ValueError)


# --- from_correspondences: hard construction-time failures -----------------

def test_from_correspondences_rejects_fewer_than_four_points():
    with pytest.raises(CalibrationError):
        RoadPlane.from_correspondences(WORLD4[:3], WORLD4[:3])


def test_from_correspondences_rejects_mismatched_point_counts():
    with pytest.raises(CalibrationError):
        RoadPlane.from_correspondences(IMG4, WORLD4[:3])


def test_from_correspondences_raises_when_cv2_returns_none():
    with pytest.raises(CalibrationError):
        RoadPlane.from_correspondences(ALL_SAME_IMG, WORLD4)


# --- to_world: exact for the four fit points --------------------------------

def test_to_world_is_exact_for_the_four_fit_points():
    plane = RoadPlane.from_correspondences(IMG4, WORLD4)
    for img_pt, world_pt in zip(IMG4, WORLD4):
        assert plane.to_world(img_pt) == pytest.approx(world_pt, abs=1e-5)


# --- reprojection_error: shape, units, and held-out corruption detection ---

def test_reprojection_error_returns_metres_shape():
    plane = RoadPlane.from_correspondences(IMG4, WORLD4)
    err = plane.reprojection_error(IMG_HOLDOUT, WORLD_HOLDOUT)
    assert set(err.keys()) == {"mean_m", "max_m", "per_point_m"}
    assert len(err["per_point_m"]) == 2
    assert err["max_m"] >= err["mean_m"] >= 0.0
    assert err["max_m"] == max(err["per_point_m"])


def test_reprojection_error_rejects_mismatched_lengths():
    plane = RoadPlane.from_correspondences(IMG4, WORLD4)
    with pytest.raises(ValueError):
        plane.reprojection_error(IMG_HOLDOUT, WORLD_HOLDOUT[:1])


def test_holdout_reprojection_detects_a_corrupted_correspondence():
    clean = RoadPlane.from_correspondences(IMG4, WORLD4)
    assert clean.reprojection_error(IMG_HOLDOUT, WORLD_HOLDOUT)["mean_m"] < 0.05

    bad_img = [IMG4[0], IMG4[1], IMG4[2], (IMG4[3][0] + 25.0, IMG4[3][1])]
    dirty = RoadPlane.from_correspondences(bad_img, WORLD4)
    assert dirty.reprojection_error(IMG_HOLDOUT, WORLD_HOLDOUT)["mean_m"] > 0.5


# --- validate(): every failure path must actually be reachable -------------

def test_validate_passes_for_a_healthy_configuration():
    plane = RoadPlane.from_correspondences(IMG4, WORLD4)
    assert plane.validate() is None  # must not raise


def test_validate_raises_when_fewer_points_than_min_points_required():
    # 4 points is enough to build a plane at all, but a caller can demand a
    # stronger survey (more points) than the mathematical minimum.
    plane = RoadPlane.from_correspondences(IMG4, WORLD4)
    with pytest.raises(CalibrationError):
        plane.validate(min_points=6)


def test_validate_raises_on_collinear_points():
    plane = RoadPlane.from_correspondences(IMG_COLLINEAR, COLLINEAR_WORLD)
    with pytest.raises(CalibrationError):
        plane.validate()


def test_validate_raises_on_duplicate_points():
    plane = RoadPlane.from_correspondences(DUPLICATE_IMG, DUPLICATE_WORLD)
    with pytest.raises(CalibrationError):
        plane.validate()


def test_validate_raises_on_near_degenerate_configuration():
    plane = RoadPlane.from_correspondences(IMG_NEAR_DEGENERATE, NEAR_DEGENERATE_WORLD)
    with pytest.raises(CalibrationError):
        plane.validate()


def test_validate_raises_when_holdout_mean_error_exceeds_threshold():
    bad_img = [IMG4[0], IMG4[1], IMG4[2], (IMG4[3][0] + 25.0, IMG4[3][1])]
    dirty = RoadPlane.from_correspondences(bad_img, WORLD4)
    with pytest.raises(CalibrationError):
        dirty.validate(holdout_image_pts=IMG_HOLDOUT, holdout_world_pts=WORLD_HOLDOUT)


def test_validate_passes_clean_fit_against_holdout():
    clean = RoadPlane.from_correspondences(IMG4, WORLD4)
    assert clean.validate(holdout_image_pts=IMG_HOLDOUT, holdout_world_pts=WORLD_HOLDOUT) is None


# --- condition-number check: threshold is bracketed by real measurements ---

def test_condition_number_of_healthy_configuration_is_well_below_threshold():
    cond = _dlt_condition_number(IMG4, WORLD4)
    assert cond < HOMOGRAPHY_MAX_CONDITION_NUMBER / 100.0


def test_condition_number_of_near_degenerate_configuration_exceeds_threshold():
    cond = _dlt_condition_number(IMG_NEAR_DEGENERATE, NEAR_DEGENERATE_WORLD)
    assert cond > HOMOGRAPHY_MAX_CONDITION_NUMBER


def test_condition_number_check_alone_does_not_catch_a_shifted_single_point():
    # A single mis-surveyed point (not a geometric degeneracy) barely moves
    # the condition number -- this is precisely why the held-out
    # reprojection-error check exists as an independent validate() path,
    # not as decoration alongside the condition-number check.
    bad_img = [IMG4[0], IMG4[1], IMG4[2], (IMG4[3][0] + 25.0, IMG4[3][1])]
    cond = _dlt_condition_number(bad_img, WORLD4)
    assert cond < HOMOGRAPHY_MAX_CONDITION_NUMBER / 100.0
