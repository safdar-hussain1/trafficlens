"""Tier-1 speed validation: a synthetic scene whose ground truth is exact.

What makes this tier worth anything is that the truth is an INPUT, not a
measurement. ``simulate_scene`` is handed a list of speeds; it moves
vehicles at exactly those speeds along a known ground plane, projects them
through a known pinhole camera, and emits ``Detection`` objects. The
estimator under test -- ``Tracker`` -> ``RoadPlane`` -> ``SpeedEstimator``
-- then runs on those detections and never touches the truth. The tests
below pin that independence directly (see
``test_the_truth_is_an_input_and_is_untouched_by_everything_the_estimator_sees``):
the truth is invariant under the seed, under the noise level, and under the
box model -- i.e. under every knob that changes what the estimator is shown.

The one thing this tier does NOT measure is the detector: the boxes are
generated, not detected. That is stated in the report's ``limitations``
field and pinned by ``scripts/bench_speed.py``.
"""

import math

import pytest

from trafficlens.analytics.speed import SpeedEstimator, time_of_flight_kmh
from trafficlens.bench.simulate import (
    MEASURED_NOISE_STATISTICS,
    ScenePlane,
    SimulationError,
    noise_from_detection_report,
    score_scene,
    simulate_scene,
    time_of_flight_scores,
)

FPS = 30.0
BANDS = (50.0, 90.0, 130.0)


@pytest.fixture
def plane():
    return ScenePlane()


def rel(err_kmh: float, speed_kmh: float) -> float:
    return 100.0 * abs(err_kmh) / speed_kmh


# --- the instrument itself ----------------------------------------------------


def test_the_scene_survey_builds_a_validated_road_plane_from_two_lines(plane):
    """The synthetic survey has the same SHAPE as a real one -- dash
    centroids on two divider lines a lane apart -- so the plane under test
    is fitted from correspondences exactly the way a deployment's is, not
    handed the camera's own analytic inverse."""
    image_pts, world_pts = plane.survey_points()
    hold_img, hold_world = plane.holdout_points()

    # two distinct cross-road offsets: a single line is collinear and
    # cannot determine a homography at all (see test_scale_survey.py).
    assert len({x for x, _ in world_pts}) == 2

    road = plane.road_plane()
    road.validate(holdout_image_pts=hold_img, holdout_world_pts=hold_world)
    error = road.reprojection_error(hold_img, hold_world)
    assert error["max_m"] < 1e-3, error


def test_the_truth_is_an_input_and_is_untouched_by_everything_the_estimator_sees(
    plane,
):
    """Truth independence, asserted rather than asserted-about.

    Seed, noise level and box model are the three knobs that change what
    the estimator is shown. If any of them could move the truth, the tier
    would be scoring the simulator against itself. None of them may.
    """
    reference = simulate_scene(
        plane, speeds_kmh=BANDS, n_vehicles=3, fps=FPS, seed=0, box_noise_px=0.0
    )
    truth = [reference.truth_speed_kmh(i) for i in range(len(reference.vehicles))]
    assert truth == list(BANDS)

    for seed, noise, model in ((7, 0.0, "footprint"), (0, 4.0, "footprint"),
                               (3, 1.5, "solid")):
        other = simulate_scene(
            plane,
            speeds_kmh=BANDS,
            n_vehicles=3,
            fps=FPS,
            seed=seed,
            box_noise_px=noise,
            box_model=model,
        )
        assert [
            other.truth_speed_kmh(i) for i in range(len(other.vehicles))
        ] == truth
        # and the exact world track, the other truth this tier publishes
        assert other.truth_world(0, 10) == reference.truth_world(0, 10)


def test_a_seed_reproduces_a_scene_and_different_seeds_differ(plane):
    def boxes(seed):
        scene = simulate_scene(
            plane, speeds_kmh=(80.0,), n_vehicles=1, fps=FPS, seed=seed,
            box_noise_px=1.0,
        )
        return [
            (d.x1, d.y1, d.x2, d.y2) for frame in scene.frames for d in frame
        ]

    assert boxes(5) == boxes(5)
    assert boxes(5) != boxes(6)


def test_the_simulator_refuses_a_scene_whose_boxes_overlap(plane):
    """Overlapping boxes would let the tracker swap identities, and a
    speed measured across an identity swap is not a speed. The simulator
    refuses to produce such a scene rather than quietly scoring one."""
    with pytest.raises(SimulationError, match="overlap"):
        simulate_scene(
            plane,
            speeds_kmh=(80.0, 80.0),
            n_vehicles=2,
            fps=FPS,
            seed=0,
            box_noise_px=0.0,
            lane_offsets_m=(1.875,),  # one lane, two vehicles, released together
            lane_release_gap_s=0.0,
        )


# --- noise sigma is a recorded, measured parameter -----------------------------


def test_the_noise_sigma_is_recorded_on_the_scene_as_four_components(plane):
    scalar = simulate_scene(
        plane, speeds_kmh=(80.0,), n_vehicles=1, fps=FPS, seed=0, box_noise_px=1.25
    )
    assert scalar.box_noise_px == (1.25, 1.25, 1.25, 1.25)

    vector = simulate_scene(
        plane,
        speeds_kmh=(80.0,),
        n_vehicles=1,
        fps=FPS,
        seed=0,
        box_noise_px=(0.3, 0.1, 1.0, 0.5),
    )
    assert vector.box_noise_px == (0.3, 0.1, 1.0, 0.5)


def test_noise_sigma_is_read_from_the_measured_detection_noise_report():
    """sigma is calibrated against reports/detection_noise.json, not
    invented. Both published statistics are available and neither is
    silently preferred."""
    report = {
        "residuals": {
            "centre_x": {"std_px": 0.4, "p95_abs_px": 0.2},
            "centre_y": {"std_px": 0.1, "p95_abs_px": 0.09},
            "box_width": {"std_px": 1.0, "p95_abs_px": 1.5},
            "box_height": {"std_px": 0.5, "p95_abs_px": 1.1},
        }
    }
    assert noise_from_detection_report(report, "std_px") == (0.4, 0.1, 1.0, 0.5)
    assert noise_from_detection_report(report, "p95_abs_px") == (0.2, 0.09, 1.5, 1.1)
    assert set(MEASURED_NOISE_STATISTICS) == {"std_px", "p95_abs_px"}
    with pytest.raises(ValueError, match="statistic"):
        noise_from_detection_report(report, "mae_px")


# --- Tier 1: exactness where truth is exact ------------------------------------


def test_zero_noise_recovers_the_exact_input_speeds(plane):
    """The brief's hard requirement, on the chain it is a requirement
    about: homography -> speed, driven by the simulator's own noise-free
    detections. Where the plane is known, the scale is known, and the
    recovery must be exact.

    The 0.1 km/h bound is the brief's; the 0.01 km/h bound below it is
    what keeps this test from passing on slack -- the measured figure is
    four orders of magnitude inside the requirement, so a regression that
    merely stayed under 0.1 would still be caught.
    """
    scene = simulate_scene(
        plane, speeds_kmh=BANDS, n_vehicles=3, fps=FPS, seed=0, box_noise_px=0.0
    )
    score = score_scene(scene, bypass_tracker=True)

    assert set(score.matched_vehicles) == set(range(3))
    for sample in score.settled_samples:
        assert abs(sample.error_kmh) < 0.1, sample
        assert abs(sample.error_kmh) < 0.01, sample


def test_the_full_chain_at_zero_noise_misses_by_the_kalman_lag_and_only_that(
    plane,
):
    """The full tracker -> homography -> speed chain does NOT meet the
    0.1 km/h bar at the top of the speed range, and this pins WHY rather
    than widening the bar.

    Three assertions identify the mechanism as the constant-velocity
    Kalman filter lagging image-space motion that accelerates as a vehicle
    approaches: the settled residual is signed NEGATIVE (a lagging filter
    under-reads), it GROWS with speed (faster vehicle, more image-space
    acceleration to lag), and it VANISHES on the identical detections when
    the tracker is bypassed. A loosened tolerance would satisfy none of
    those.
    """
    scene = simulate_scene(
        plane, speeds_kmh=BANDS, n_vehicles=3, fps=FPS, seed=0, box_noise_px=0.0
    )
    tracked = score_scene(scene)
    bypassed = score_scene(scene, bypass_tracker=True)

    by_speed = {}
    for sample in tracked.settled_samples:
        by_speed.setdefault(sample.truth_kmh, []).append(sample.error_kmh)

    assert sorted(by_speed) == sorted(BANDS)
    means = [sum(v) / len(v) for _, v in sorted(by_speed.items())]

    # 1. signed negative: the filter lags, so the fitted slope under-reads
    assert all(mean < 0.0 for mean in means), means
    # 2. grows with speed, strictly, band by band
    assert all(a > b for a, b in zip(means, means[1:])), means
    # 3. and it is the tracker's: the same detections without it are exact
    assert max(abs(s.error_kmh) for s in bypassed.settled_samples) < 0.01

    # the measured envelope, pinned so a regression that made it worse is
    # a failure rather than a quietly larger published number
    assert max(abs(s.error_kmh) for s in tracked.settled_samples) < 0.4


def test_non_zero_noise_degrades_the_estimate_monotonically(plane):
    """sigma is swept as a multiple of the MEASURED per-component vector,
    so every point on the curve is a stated multiple of a number taken
    from reports/detection_noise.json."""
    measured = (0.3344, 0.0971, 1.0294, 0.5232)
    rmses = []
    for multiple in (0.0, 1.0, 2.0, 4.0):
        errors = []
        for seed in range(6):
            scene = simulate_scene(
                plane,
                speeds_kmh=BANDS,
                n_vehicles=3,
                fps=FPS,
                seed=seed,
                box_noise_px=tuple(multiple * s for s in measured),
            )
            errors += [s.error_kmh for s in score_scene(scene).settled_samples]
        assert errors, f"no settled samples at {multiple}x measured sigma"
        rmses.append(math.sqrt(sum(e * e for e in errors) / len(errors)))

    assert all(a < b for a, b in zip(rmses, rmses[1:])), rmses


# --- Check C: the two estimators are checked against each other ----------------


def test_time_of_flight_and_the_homography_estimator_agree_on_the_simulated_scene(
    plane,
):
    """Check C, as an INSTRUMENT check.

    On the simulated scene both estimators share a known scale, so their
    agreement tests the two estimators against each other. It is not run
    on real footage, where the gate separation is only known through the
    disputed along-road scale (see reports/speed_real.json).
    """
    scene = simulate_scene(
        plane, speeds_kmh=BANDS, n_vehicles=3, fps=FPS, seed=0, box_noise_px=0.0
    )
    results = time_of_flight_scores(scene, gate_far_y_m=100.0, gate_near_y_m=40.0)

    assert len(results) == 3
    for result in results:
        assert abs(result.time_of_flight_kmh - result.truth_kmh) < 1.0, result
        assert abs(result.time_of_flight_kmh - result.homography_kmh) < 1.0, result


def test_time_of_flight_shares_no_per_frame_displacement_with_the_homography(plane):
    """The cross-check is only a cross-check if it is arithmetically
    independent. Its number is reproducible from two crossing instants and
    the surveyed gate separation alone -- nothing per-frame."""
    scene = simulate_scene(
        plane, speeds_kmh=(80.0,), n_vehicles=1, fps=FPS, seed=0, box_noise_px=0.0
    )
    result = time_of_flight_scores(scene, gate_far_y_m=100.0, gate_near_y_m=40.0)[0]

    assert result.time_of_flight_kmh == time_of_flight_kmh(
        result.t_far_s, result.t_near_s, 60.0
    )


# --- the estimator's own refusal still holds on a synthetic scene --------------


def test_an_uncalibrated_estimator_reports_no_speed_on_this_scene(plane):
    """The same scene, scored with plane=None, yields no number at all --
    the policy the shipped motorway config now relies on."""
    scene = simulate_scene(
        plane, speeds_kmh=(80.0,), n_vehicles=1, fps=FPS, seed=0, box_noise_px=0.0
    )
    estimator = SpeedEstimator(None, FPS)
    for index, frame in enumerate(scene.frames):
        for detection in frame:
            anchor = ((detection.x1 + detection.x2) / 2.0, detection.y2)
            estimator.observe(1, anchor, index / FPS)
    assert estimator.speed_kmh(1) is None
