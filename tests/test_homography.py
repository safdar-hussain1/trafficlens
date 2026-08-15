"""Tests for trafficlens.core.homography: road-plane calibration, held-out
reprojection validation, and the refusal-to-guess policy for an
uncalibrated camera."""

import numpy as np
import pytest

from trafficlens.core.homography import (
    CalibrationError,
    NO_CALIBRATION,
    RoadPlane,
    _dlt_condition_number,
)
from trafficlens.core.constants import (
    HOMOGRAPHY_MAX_MEAN_ERROR_M,
    HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER,
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

# A 5-point fit, for exercising validate()'s no-holdout self-check on a
# configuration where it is actually allowed to run (see the docstring of
# RoadPlane.validate(): only 5+ points have residual degrees of freedom to
# check against). WORLD5 is WORLD4 plus a 5th road point near the centre of
# the surveyed area; IMG5 is the exact, noiseless H_wi projection -- no
# nudge.
#
# (History: round 1 of this test suite nudged the 5th image point by
# (+2px, -2px) because the OLD degeneracy diagnostic -- sigma_1/sigma_9,
# i.e. dividing by the *smallest* singular value of the DLT design matrix --
# reported any perfectly noiseless N>4 fit as catastrophically ill-
# conditioned (~2.2e16 for this exact fixture), regardless of how well-spread
# the points were: sigma_9 is the "solve residual" direction, near-zero for
# any exactly-consistent data, not a geometry signal. That diagnostic was
# the bug: it punished survey precision, rejecting real sub-pixel-accurate
# surveys most often. The fix uses sigma_1/sigma_8 for N>=5 -- sigma_8 is
# the second-smallest singular value, which stays near-flat regardless of
# noise level (measured ~4.6-4.7 across the noise sweep below) and only
# collapses toward zero for genuine rank deficiency. The nudge is removed:
# this fixture is now the noiseless projection, unmodified, and passing
# cleanly on it is itself proof the diagnostic no longer measures residual.
# See _dlt_condition_number's docstring for the full explanation.)
WORLD5 = WORLD4 + [(0.0, 18.0)]
IMG5 = [
    (686.1374124964, 476.0372807540398),
    (1233.8625875036003, 476.0372807540398),
    (854.7651148131101, 946.5662269242592),
    (1065.2348851868899, 946.5662269242592),
    (960.0000000000002, 823.3672343564774),
]

# IMG5 with the 4th point (index 3) shifted +150px in x. This is a corrupted
# 5-point configuration: not geometrically degenerate (new diagnostic
# ~4.65, indistinguishable from the healthy IMG5/WORLD5 fixture above -- the
# degeneracy check genuinely cannot catch this class of error, by design),
# but its least-squares self-fit reprojection error against its own 5 fit
# points is large (~0.74m mean), because with 5 points the fit no longer has
# to pass through every point exactly. That residual freedom is what makes
# the no-holdout self-check in validate() meaningful for 5+ points, unlike
# the exact 4-point case.
BAD_IMG5 = [
    (686.1374124964, 476.0372807540398),
    (1233.8625875036003, 476.0372807540398),
    (854.7651148131101, 946.5662269242592),
    (1215.2348851868899, 946.5662269242592),
    (960.0000000000002, 823.3672343564774),
]

# A genuinely rank-deficient N=5 configuration: 4 of the 5 world points
# collinear (all on X = -3.5, reusing COLLINEAR_WORLD's first 3 plus one
# more on the same line), 1 point off the line.
#
# This is stronger than the most obvious-sounding degenerate 5-point case --
# "the 5th point collinear with 3 others, leaving 2 points off the line" --
# which was tried first and found NOT to trip the new diagnostic at all
# (measured ~4.60, indistinguishable from the healthy fixture). That is
# mathematically correct, not a gap in the check: a homography's 8 DOF only
# need *some* 4-point subset in general position, and a 5-point set with 3
# collinear points plus 2 independent ones still contains such a subset (any
# 2 of the collinear 3 plus the 2 independent points), so the full
# configuration is not actually rank-deficient. Making every possible
# 4-point subset degenerate requires collinearity in at least 4 of the 5
# points -- leaving only 1 point off the line, so every 4-subset must
# include at least 3 collinear ones. That is what this fixture does, and it
# measures ~1.27e16 on the new diagnostic: genuinely rank-deficient, not a
# residual artifact.
WORLD5_DEGENERATE = COLLINEAR_WORLD + [(-3.5, 22.0)]
IMG5_DEGENERATE = [
    (686.1374124964, 476.0372807540398),
    (793.1080480855557, 774.5219291446756),
    (854.7651148131101, 946.5662269242592),
    (1099.6235046721424, 850.6102205009837),
    (828.9419182764447, 874.5106791927072),
]


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

def test_validate_raises_on_four_point_fit_without_holdout_even_when_clean():
    # A 4-point fit exactly determines the homography (8 equations, 8
    # unknowns), so it reproduces its own 4 points to floating-point
    # precision *by construction* -- checking reprojection error against
    # them can never fail, clean or not. validate() must refuse to
    # pretend that non-check is a pass, rather than silently reporting
    # success for a self-check it cannot actually run.
    plane = RoadPlane.from_correspondences(IMG4, WORLD4)
    with pytest.raises(CalibrationError):
        plane.validate()


def test_validate_raises_on_corrupted_four_point_fit_without_holdout():
    # The reviewer's exact scenario: corrupt one image point by 25px, build
    # the plane, call validate() with no holdout. Before this fix this
    # passed -- the self-fit error against the same corrupted 4 points was
    # ~1e-6m regardless -- even though the true (held-out) error is 0.656m.
    # It must now raise, for the same reason as the clean case above: a
    # 4-point plane cannot self-check at all without a holdout.
    bad_img = [IMG4[0], IMG4[1], IMG4[2], (IMG4[3][0] + 25.0, IMG4[3][1])]
    dirty = RoadPlane.from_correspondences(bad_img, WORLD4)
    with pytest.raises(CalibrationError):
        dirty.validate()


def test_validate_passes_for_a_clean_five_point_fit_without_holdout():
    # With 5+ correspondences the least-squares solve has genuine residual
    # degrees of freedom, so a no-holdout self-check is not mathematically
    # inert -- and a clean 5-point fit passes it. IMG5/WORLD5 is the exact,
    # noiseless H_wi projection with no artificial nudge (see the fixture
    # comment): this fixture passing cleanly is itself the proof that the
    # degeneracy diagnostic no longer measures fit residual.
    plane = RoadPlane.from_correspondences(IMG5, WORLD5)
    assert plane.validate() is None  # must not raise


def test_validate_raises_for_a_corrupted_five_point_fit_without_holdout():
    # Proves the >4-point self-check is genuinely informative, not just
    # "allowed to run": BAD_IMG5 is geometrically well-conditioned (its new
    # degeneracy diagnostic, ~4.65, is indistinguishable from the healthy
    # IMG5/WORLD5 fixture and nowhere near
    # HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER, so the degeneracy check does not
    # catch this) but its own least-squares fit has a large residual against
    # its own 5 fit points (mean ~0.74m) because a single corrupted
    # correspondence among 5 no longer has to be fit exactly. That residual
    # is exactly what the no-holdout self-check for 5+ points exists to
    # catch.
    plane = RoadPlane.from_correspondences(BAD_IMG5, WORLD5)
    with pytest.raises(CalibrationError):
        plane.validate()


def test_validate_raises_on_five_point_fit_with_four_collinear_points():
    # The genuinely rank-deficient N=5 fixture (see WORLD5_DEGENERATE's
    # comment for why a milder "3 of 5 collinear" attempt does not trigger
    # this check, correctly).
    plane = RoadPlane.from_correspondences(IMG5_DEGENERATE, WORLD5_DEGENERATE)
    with pytest.raises(CalibrationError):
        plane.validate()


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
    assert cond < HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER / 100.0


def test_condition_number_of_near_degenerate_configuration_exceeds_threshold():
    cond = _dlt_condition_number(IMG_NEAR_DEGENERATE, NEAR_DEGENERATE_WORLD)
    assert cond > HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER


def test_condition_number_check_alone_does_not_catch_a_shifted_single_point():
    # A single mis-surveyed point (not a geometric degeneracy) barely moves
    # the condition number -- this is precisely why the held-out
    # reprojection-error check exists as an independent validate() path,
    # not as decoration alongside the condition-number check.
    bad_img = [IMG4[0], IMG4[1], IMG4[2], (IMG4[3][0] + 25.0, IMG4[3][1])]
    cond = _dlt_condition_number(bad_img, WORLD4)
    assert cond < HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER / 100.0


# --- regression: the degeneracy check must not punish survey precision -----

def test_precise_surveys_are_never_rejected_as_degenerate():
    # Round-2 regression for the bug the OLD diagnostic had: for an
    # overdetermined (N>=5) fit, dividing by the smallest singular value of
    # the DLT design matrix measures fit *residual*, not geometric
    # degeneracy -- so the more precise (lower pixel noise) a clean,
    # well-spread survey is, the smaller that residual and the larger (more
    # falsely alarming) the resulting ratio. A sub-pixel-precise survey --
    # the best a careful user can produce -- was rejected as "collinear or
    # nearly so" up to 93% of the time by that diagnostic.
    #
    # This sweeps the same shape of experiment (clean, well-spread 5-point
    # surveys, Gaussian pixel noise, 30 trials per noise level) with a fixed
    # seed so it is a reproducible regression, not a one-off measurement.
    # IMG5/WORLD5 (noiseless base projection) is used as the ground truth
    # the noise is added to. Both the raw diagnostic AND the full
    # validate() no-holdout call are checked, so this pins the unit-level
    # fix and the user-visible behaviour together.
    rng = np.random.default_rng(20260815)
    noise_levels_px = [0.1, 0.25, 0.5, 1.0, 2.0]
    trials_per_level = 30
    base = np.array(IMG5, dtype=np.float64)

    false_rejections = []
    for sigma in noise_levels_px:
        for trial in range(trials_per_level):
            noisy = base + rng.normal(0.0, sigma, size=base.shape)
            noisy_pts = [(float(x), float(y)) for x, y in noisy]

            cond = _dlt_condition_number(noisy_pts, WORLD5)
            if cond > HOMOGRAPHY_MAX_RANK_CONDITION_NUMBER:
                false_rejections.append((sigma, trial, "condition_number", cond))
                continue

            plane = RoadPlane.from_correspondences(noisy_pts, WORLD5)
            try:
                plane.validate()
            except CalibrationError as exc:
                false_rejections.append((sigma, trial, "validate", str(exc)))

    assert false_rejections == [], (
        f"{len(false_rejections)} of "
        f"{len(noise_levels_px) * trials_per_level} precise, well-spread "
        f"5-point surveys were falsely rejected as degenerate: "
        f"{false_rejections[:5]}"
    )
