"""The along-road scale investigation on the motorway clip, as code.

Tier 2 of the speed validation was meant to calibrate the flagship clip
from lane-divider geometry and publish real speeds. It publishes a negative
result instead: the clip has no independent along-road anchor, so no
absolute km/h derived from it is trustworthy, and
``configs/motorway.yaml`` therefore ships with NO calibration block. This
module holds the arithmetic behind that finding so it stays reproducible
after the block was deleted -- the surveyed points themselves live in
``data/fixtures/motorway_scale_survey.json``, because the clip is
git-ignored and the points are the record.

The measurement instrument
--------------------------
Two facts about a pinhole camera make every along-road measurement here
possible WITHOUT any cross-road calibration, which matters because the
clip's cross-road calibration is exactly what is in dispute.

For any road-parallel 3-D line -- any lateral offset, any height -- the
image of a point at along-road distance ``s`` satisfies

    u = 1 / (vp_x - x)   is AFFINE in s

where ``vp_x`` is the road vanishing point's image COLUMN and ``x`` the
image column of the point. (The image of a 3-D line is
``x(s) = (a s + b) / (c s + d)`` with ``vp_x = a / c``, so
``1 / (vp_x - x) = c (c s + d) / (a d - b c)``, affine in ``s``.) Equal
world spacing is therefore equal spacing in ``u``, whatever the
perspective -- which is what makes "is this structure periodic?" an
answerable question at all.

And for every road-parallel line SIMULTANEOUSLY, with ``vp_y`` the
vanishing point's image ROW and a single shared constant ``K``:

    s = M_line + K / (y - vp_y)

Only the additive ``M_line`` differs per line. So ``K`` fitted on one line
whose spacing is assumed transfers to any other road-parallel line with no
further calibration. Its one vulnerability is camera roll, which the
identity assumes is zero (it uses the vanishing point's row as if it were
the horizon); the clip's measured roll upper bound is 1.97 degrees, worth
about 3 % on the divider-2 result below.

What that instrument found
--------------------------
Divider 1's six surveyed dash centroids fit the assumed 18 m ladder to
0.16 m rms. Applying the SAME ``K`` to divider 2 -- also consecutive, also
uniform, on the same carriageway -- gives about 26 m per dash step, a
factor of ~1.46, at every plausible horizon row. Two dashed dividers on one
carriageway cannot differ by that factor, so the correspondence set the
homography was fitted from is mutually inconsistent, and its flattering
0.12 m self-fit residual hides it.

That defect is independent of whether 18 m is the right absolute period,
and it cannot be repaired by dropping the disputed points: divider 1's
correspondences are collinear (one image line, one cross-road offset) and
determine no homography at all. See
``tests/test_scale_survey.py::test_dropping_divider_two_cannot_rescue_the_calibration``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]

#: The committed record of what was surveyed off the clip. The clip is
#: git-ignored; these points are the evidence.
SCALE_SURVEY_FIXTURE = _ROOT / "data" / "fixtures" / "motorway_scale_survey.json"


def load_scale_survey(path=None) -> dict:
    """Load the committed scale-survey fixture."""
    return json.loads(Path(path or SCALE_SURVEY_FIXTURE).read_text())


def fit_line(points) -> tuple[float, float, float, float]:
    """Plain least-squares ``y = slope * x + intercept`` over image points.

    Returns ``(slope, intercept, residual_rms_px, max_residual_px)``. Use
    this where every point is trustworthy -- the divider dash centroids
    were placed by hand and are. For the guardrail track, where a tenth of
    the columns lock onto shadow rather than the beam, use
    ``robust_fit_line`` and quote both figures.
    """
    array = np.asarray(points, dtype=np.float64)
    x, y = array[:, 0], array[:, 1]
    design = np.vstack([x, np.ones_like(x)]).T
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(np.sqrt(np.mean(residual**2))),
        float(np.max(np.abs(residual))),
    )


def robust_fit_line(points, passes: int = 4) -> tuple[float, float, float]:
    """Iteratively reweighted least-squares line fit with Cauchy weights.

    Returns ``(slope, intercept, weighted_residual_rms_px)``. The weight is
    ``1 / (1 + (r / (2 * scale))**2)`` with ``scale`` the median absolute
    deviation of the residuals about their median, scaled by 1.4826 so it
    estimates a Gaussian sigma. Four passes, which is where the fit stops
    moving on this data.

    The rms it returns is WEIGHTED, so it describes the beam, not the
    outliers -- state it as such wherever it is published.
    """
    array = np.asarray(points, dtype=np.float64)
    x, y = array[:, 0], array[:, 1]
    weight = np.ones(len(array))
    coefficients = np.zeros(2)
    residual = np.zeros(len(array))
    for _ in range(passes):
        design = np.vstack([x, np.ones_like(x)]).T
        coefficients, *_ = np.linalg.lstsq(
            design * weight[:, None], y * weight, rcond=None
        )
        residual = y - (coefficients[0] * x + coefficients[1])
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        weight = 1.0 / (1.0 + (residual / (2.0 * scale)) ** 2)
    weighted_rms = float(
        np.sqrt((residual**2 * weight).sum() / weight.sum())
    )
    return float(coefficients[0]), float(coefficients[1]), weighted_rms


def line_intersection(a, b) -> tuple[float, float]:
    """Where two ``(slope, intercept)`` image lines meet."""
    slope_a, intercept_a = a
    slope_b, intercept_b = b
    if slope_a == slope_b:
        raise ValueError("parallel lines do not intersect")
    x = (intercept_b - intercept_a) / (slope_a - slope_b)
    return (float(x), float(slope_a * x + intercept_a))


def _line(survey: dict, name: str) -> tuple[float, float]:
    """The fitted image line of one of the three road-parallel structures."""
    if name == "guardrail":
        return robust_fit_line(survey["guardrail_beam"]["points_px"])[:2]
    return fit_line(survey["divider_lines"][name]["points_px"])[:2]


def road_vanishing_point(survey: dict) -> tuple[float, float]:
    """The road vanishing point from the two BEST-measured road-parallel
    lines: divider 1 (six hand-placed dash centroids, straight to 0.04 px
    over 354 px) and the guardrail beam (110 sub-pixel points over 580 px).

    Divider 2 is deliberately excluded. It is the line under dispute, and
    it is also the one whose fit rests on only three points spanning 180 px
    -- so including it would let the disputed measurement set the reference
    every other measurement is made against.
    """
    return line_intersection(_line(survey, "divider_1"), _line(survey, "guardrail"))


def surveyed_homography(survey: dict):
    """Reconstruct the exact ``RoadPlane`` that ``configs/motorway.yaml``
    used to ship, from the committed correspondences, so the removed
    calibration's own numbers stay checkable.

    This plane is NOT trustworthy and is not for measuring anything -- it
    is the object under investigation. cv2 is imported through
    ``core.homography`` here rather than at module scope.
    """
    from trafficlens.core.homography import RoadPlane

    divider_1 = survey["divider_lines"]["divider_1"]
    divider_2 = survey["divider_lines"]["divider_2"]
    # The four divider-1 points the config used as FIT points were the
    # 0/18/54/72 m dashes; 36 m and 90 m were its holdout.
    fit_indices = [0, 1, 3, 4]
    image_pts = [tuple(divider_1["points_px"][i]) for i in fit_indices]
    world_pts = [(0.0, divider_1["assumed_world_y_m"][i]) for i in fit_indices]
    image_pts += [tuple(p) for p in divider_2["points_px"]]
    world_pts += [
        (divider_2["assumed_cross_road_offset_m"], y)
        for y in divider_2["assumed_world_y_m"]
    ]
    return RoadPlane.from_correspondences(image_pts, world_pts)


def surveyed_homography_vanishing_point(survey: dict) -> tuple[float, float]:
    """Where the removed calibration put the road vanishing point: the
    image point its homography maps to world infinity along the road."""
    inverse = np.linalg.inv(surveyed_homography(survey)._h)
    at_infinity = inverse @ np.array([0.0, 1.0, 0.0])
    return (
        float(at_infinity[0] / at_infinity[2]),
        float(at_infinity[1] / at_infinity[2]),
    )


def vanishing_point_spread(survey: dict) -> dict:
    """Each pair of the three road-parallel lines meets somewhere. If all
    three really are parallel to the road, all three meetings are the same
    point.

    Returns each pairwise intersection and its distance from the removed
    calibration's own vanishing point. The ordering is the evidence: the
    guardrail -- which knows nothing about the divider survey -- agrees far
    better with divider 1 than with divider 2, and the two dividers agree
    with each other worst of all.
    """
    reference = surveyed_homography_vanishing_point(survey)
    pairs = {
        "divider_1_x_guardrail": ("divider_1", "guardrail"),
        "divider_2_x_guardrail": ("divider_2", "guardrail"),
        "divider_1_x_divider_2": ("divider_1", "divider_2"),
    }
    out = {"surveyed_homography_vp": [float(v) for v in reference], "pairs": {}}
    for label, (a, b) in pairs.items():
        point = line_intersection(_line(survey, a), _line(survey, b))
        out["pairs"][label] = {
            "point": [round(float(v), 3) for v in point],
            "distance_from_surveyed_vp_px": round(
                float(np.hypot(point[0] - reference[0], point[1] - reference[1])), 3
            ),
        }
    return out


def along_road_transfer(image_rows, world_metres, horizon_row: float) -> dict:
    """Fit ``s = M + K / (y - horizon_row)`` to one road-parallel line.

    ``image_rows`` are the image rows of features on that line and
    ``world_metres`` their assumed along-road positions. Returns ``{"M",
    "K", "rms_m", "residuals_m"}``. ``K`` is the shared constant that
    transfers to every other road-parallel line; ``M`` is that line's own
    additive offset and transfers to nothing.
    """
    rows = np.asarray(image_rows, dtype=np.float64)
    metres = np.asarray(world_metres, dtype=np.float64)
    if np.any(np.isclose(rows, horizon_row)):
        raise ValueError(
            f"a feature lies on the horizon row {horizon_row}; the transfer "
            f"is singular there"
        )
    design = np.vstack([np.ones_like(rows), 1.0 / (rows - horizon_row)]).T
    coefficients, *_ = np.linalg.lstsq(design, metres, rcond=None)
    residual = metres - design @ coefficients
    return {
        "M": float(coefficients[0]),
        "K": float(coefficients[1]),
        "rms_m": float(np.sqrt(np.mean(residual**2))),
        "residuals_m": [float(v) for v in residual],
    }


def divider_step_metres(survey: dict, *, horizon_row: float) -> dict:
    """Divider 2's along-road dash step, measured with ``K`` fitted on
    divider 1's assumed ladder -- the D20 result.

    Under one road plane, one along-road scale and two road-parallel lines,
    this must come back equal to the assumed period. It comes back ~1.46x
    it, at every horizon row anyone can defend.
    """
    lines = survey["divider_lines"]
    divider_1, divider_2 = lines["divider_1"], lines["divider_2"]
    fit = along_road_transfer(
        [y for _, y in divider_1["points_px"]],
        divider_1["assumed_world_y_m"],
        horizon_row,
    )
    rows_2 = np.asarray([y for _, y in divider_2["points_px"]], dtype=np.float64)
    positions = fit["K"] / (rows_2 - horizon_row)
    steps = np.diff(positions)
    assumed = float(lines["assumed_period_m"])
    return {
        "horizon_row": float(horizon_row),
        "K": fit["K"],
        "divider_1_rms_m": fit["rms_m"],
        "divider_2_steps_m": [float(v) for v in steps],
        "divider_2_step_m": float(np.mean(steps)),
        "assumed_period_m": assumed,
        "ratio": float(np.mean(steps) / assumed),
    }
