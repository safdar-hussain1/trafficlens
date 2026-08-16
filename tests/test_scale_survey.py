"""The scale investigation on `motorway-a40.webm`, kept reproducible.

`configs/motorway.yaml` no longer ships a calibration block. The evidence
that led to removing it -- the surveyed divider-dash centroids, and the one
structure in the clip that measures cleanly -- lives in
`data/fixtures/motorway_scale_survey.json` instead, and this file drives it.
The clip itself is git-ignored, so the committed points ARE the record and
every assertion below runs off them alone.

`reports/speed_real.json` is the published write-up; this is its regression
test.
"""

import json
import math
from pathlib import Path

import pytest

from trafficlens.bench.scale import (
    SCALE_SURVEY_FIXTURE,
    along_road_transfer,
    divider_step_metres,
    fit_line,
    load_scale_survey,
    road_vanishing_point,
    robust_fit_line,
    surveyed_homography_vanishing_point,
    vanishing_point_spread,
)
from trafficlens.core.homography import CalibrationError, RoadPlane

#: Horizon rows the survey checked every result against. Spans well above
#: and below the vanishing-point row any of the three fitted road-parallel
#: lines implies, so no result below rests on one lucky choice.
HORIZON_ROWS = (340.0, 356.0, 364.0, 372.468, 375.65, 380.0, 400.0)


@pytest.fixture
def survey():
    return load_scale_survey()


def test_the_fixture_is_tracked_and_carries_the_evidence_the_config_lost(survey):
    """A bare deletion of the calibration block would have destroyed the
    only committed record of what was surveyed. This pins that it did
    not."""
    assert SCALE_SURVEY_FIXTURE.is_file()
    assert len(survey["guardrail_beam"]["points_px"]) == 110
    divider_1 = survey["divider_lines"]["divider_1"]
    divider_2 = survey["divider_lines"]["divider_2"]
    assert len(divider_1["points_px"]) == 6
    assert len(divider_2["points_px"]) == 3
    # the deleted block's own numbers, so the investigation is re-runnable
    assert divider_1["assumed_world_y_m"] == [0.0, 18.0, 36.0, 54.0, 72.0, 90.0]
    assert divider_2["assumed_world_y_m"] == [0.0, 18.0, 36.0]


# --- the one positive result: the guardrail ------------------------------------


def test_the_guardrail_beam_is_straight_to_a_quarter_pixel(survey):
    """The clip's one clean measurement, kept as a regression fixture.

    Straightness at this level over a 580 px span also bounds lens
    distortion in the region every other measurement was made in.
    """
    points = survey["guardrail_beam"]["points_px"]
    recorded = survey["guardrail_beam"]["fit"]
    slope, intercept, weighted_rms = robust_fit_line(points)

    assert slope == pytest.approx(recorded["slope"], abs=1e-6)
    assert intercept == pytest.approx(recorded["intercept"], abs=1e-3)
    assert weighted_rms == pytest.approx(recorded["residual_rms_px"], abs=1e-3)
    assert weighted_rms < 0.25

    span = max(x for x, _ in points) - min(x for x, _ in points)
    assert span > 550.0, "a straightness claim needs a long baseline"


def test_the_guardrail_straightness_figure_is_a_robust_one_and_says_so(survey):
    """The 0.243 px is a ROBUST weighted rms, not a plain one, and the
    difference matters: a tenth of the tracked columns lock onto shadow or
    vegetation instead of the beam's groove and sit several pixels off.
    Both numbers are recorded so nobody reads the robust figure as if it
    described every point.
    """
    points = survey["guardrail_beam"]["points_px"]
    _, _, plain_rms, _ = fit_line(points)
    _, _, robust_rms = robust_fit_line(points)

    assert plain_rms > 4.0 * robust_rms, (plain_rms, robust_rms)
    assert survey["guardrail_beam"]["fit"]["plain_residual_rms_px"] == pytest.approx(
        plain_rms, abs=1e-3
    )
    assert survey["guardrail_beam"]["fit"]["inliers_within_0_5px"] == 93


def test_the_guardrail_is_parallel_to_the_road_to_within_one_pixel(survey):
    """The rail's own line, extrapolated to the removed calibration's
    vanishing-point COLUMN, lands on that vanishing point's ROW. Nothing in
    the guardrail measurement knows about the divider survey, so this is an
    independent check on the direction of the road -- and it bounds lens
    distortion across the region every other measurement was made in."""
    vp_x, vp_y = surveyed_homography_vanishing_point(survey)
    assert (vp_x, vp_y) == pytest.approx((858.33, 372.468), abs=0.01)

    slope, intercept, _ = robust_fit_line(survey["guardrail_beam"]["points_px"])
    assert abs((slope * vp_x + intercept) - vp_y) < 1.0


def test_the_guardrail_agrees_with_divider_one_and_disagrees_with_divider_two(
    survey,
):
    """New evidence on which of the two dividers is the odd one out.

    Three road-parallel lines meet pairwise at three points; if all three
    really are road-parallel and coplanar, those are one point. They are
    not, and the ORDERING is the finding: the guardrail -- 110 sub-pixel
    points over 580 px, and wholly independent of the divider survey --
    meets divider 1 within 7 px of the surveyed vanishing point, meets
    divider 2 nearly four times further out, and the two dividers meet each
    other furthest of all.

    That does not identify the cause (a taper, a vertical curve, a
    mis-survey and a different marking type would all show this way), and
    the separations are only 1-3 sigma of a plausible reading error. It
    does say the disagreement is not symmetric between the two lines.
    """
    spread = vanishing_point_spread(survey)["pairs"]
    with_divider_1 = spread["divider_1_x_guardrail"]["distance_from_surveyed_vp_px"]
    with_divider_2 = spread["divider_2_x_guardrail"]["distance_from_surveyed_vp_px"]
    between_dividers = spread["divider_1_x_divider_2"][
        "distance_from_surveyed_vp_px"
    ]

    assert with_divider_1 < 7.0, spread
    assert with_divider_2 > 3.0 * with_divider_1, spread
    assert between_dividers > with_divider_2, spread


# --- D20: the calibration is internally inconsistent ---------------------------


def test_the_two_dividers_disagree_on_the_along_road_period_at_every_horizon(
    survey,
):
    """The finding that removed the calibration block.

    Both dividers' dashes are consecutive and uniform, so under one road
    plane and one along-road scale they must give the same step. They give
    18 m and ~26 m, a factor of ~1.46, and no plausible horizon row
    reconciles them -- which is a defect in the correspondence set itself,
    independent of whether 18 m is the right absolute period.
    """
    ratios = []
    for horizon_row in HORIZON_ROWS:
        step = divider_step_metres(survey, horizon_row=horizon_row)
        ratios.append(step["divider_2_step_m"] / step["assumed_period_m"])
        assert 24.0 < step["divider_2_step_m"] < 28.0, (horizon_row, step)

    assert min(ratios) > 1.4 and max(ratios) < 1.5, ratios


def test_no_horizon_row_that_fits_divider_one_can_also_fit_divider_two(survey):
    """The escape hatch, closed: the horizon row that would make divider 2
    read 18 m destroys divider 1's own ladder, so it is not a horizon row
    at all."""
    divider_1 = survey["divider_lines"]["divider_1"]
    rows = [y for _, y in divider_1["points_px"]]
    metres = divider_1["assumed_world_y_m"]

    best = min(
        HORIZON_ROWS, key=lambda r: along_road_transfer(rows, metres, r)["rms_m"]
    )
    assert along_road_transfer(rows, metres, best)["rms_m"] < 0.25

    reconciling = divider_step_metres(survey, horizon_row=463.0)
    assert reconciling["divider_2_step_m"] == pytest.approx(18.0, abs=1.5)
    assert reconciling["divider_1_rms_m"] > 5.0, reconciling


def test_dropping_divider_two_cannot_rescue_the_calibration(survey):
    """Why the block was removed outright rather than refitted.

    The obvious repair -- drop the disputed divider-2 correspondences and
    refit on divider 1 alone -- is not merely worse, it is impossible: all
    six divider-1 points lie on one straight line in the image and at one
    cross-road offset in the world, and a homography cannot be determined
    from a collinear configuration at all. The disputed points are the only
    cross-road information the survey ever had.
    """
    divider_1 = survey["divider_lines"]["divider_1"]
    image_pts = [tuple(p) for p in divider_1["points_px"]]
    world_pts = [(0.0, y) for y in divider_1["assumed_world_y_m"]]

    # collinear in the image, to a twentieth of a pixel over 354 px
    _, _, _, max_residual = fit_line(image_pts)
    assert max_residual < 0.05, max_residual

    with pytest.raises(CalibrationError):
        RoadPlane.from_correspondences(image_pts, world_pts).validate()


def test_both_dividers_are_uniform_so_neither_is_a_mis_read_ladder(survey):
    """The inconsistency is not one bad dash. In the perspective-free
    coordinate u = 1 / (vp_x - x), equal world spacing is equal spacing in
    u, and both lines are uniform there -- divider 2 more so than divider
    1. Whatever is wrong is wrong about the two lines' relationship, not
    about a single misplaced centroid."""
    vp_x, _ = road_vanishing_point(survey)

    spreads = {}
    for name in ("divider_1", "divider_2"):
        rows = survey["divider_lines"][name]["points_px"]
        u = [1.0 / (vp_x - x) for x, _ in rows]
        steps = [b - a for a, b in zip(u, u[1:])]
        mean = sum(steps) / len(steps)
        variance = sum((s - mean) ** 2 for s in steps) / len(steps)
        spreads[name] = 100.0 * math.sqrt(variance) / mean

    assert spreads["divider_1"] < 3.0, spreads
    assert spreads["divider_2"] < 3.0, spreads
    # divider 2 -- the disputed line -- is the MORE uniform of the two
    assert spreads["divider_2"] < spreads["divider_1"], spreads


# --- the published report says all of this ------------------------------------


def test_the_published_report_carries_the_investigation_not_a_footnote():
    report = json.loads(Path("reports/speed_real.json").read_text())
    assert report["absolute_speed_published"] is False
    assert report["shipped_config_calibrated"] is False
    # every candidate anchor the survey looked for is named with a verdict
    verdicts = {c["candidate"]: c["verdict"] for c in report["anchor_candidates"]}
    assert len(verdicts) >= 5
    assert set(verdicts.values()) <= {
        "PRESENT BUT NOT MEASURABLE",
        "PRESENT AND MEASURABLE BUT UNUSABLE",
        "ABSENT",
        "PRESENT AND MEASURABLE AND CONTRADICTORY",
    }
    # controls are what make "not measurable" a measurement
    controls = report["anchor_candidates"][0]["matched_controls"]
    assert any(c["band"].startswith("positive control") for c in controls)
    assert any(c["band"].startswith("asphalt control") for c in controls)
