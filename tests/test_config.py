"""Tests for trafficlens.config: fail-fast pydantic models, load_config, and
the shipped configs under configs/ (which must always load)."""

from pathlib import Path

import pytest

from trafficlens.config import (
    AppConfig,
    CalibrationConfig,
    ConfigError,
    DetectorConfig,
    GateConfig,
    SpeedConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _detector(**overrides):
    base = {"classes": ["car", "truck"], "confidence": 0.25, "imgsz": 640}
    base.update(overrides)
    return base


def _app(**overrides):
    base = {"source": "clip.mp4", "detector": _detector()}
    base.update(overrides)
    return base


def _square_calibration(**overrides):
    # 5 correspondences (4 corners + centre) so no holdout is required.
    # The image rectangle maps to the world rectangle by an exact affine
    # map, so the fitted homography reproduces every point exactly.
    base = {
        "image_points": [
            [0.2, 0.8],
            [0.8, 0.8],
            [0.8, 0.3],
            [0.2, 0.3],
            [0.5, 0.55],
        ],
        "world_points": [[0, 0], [10, 0], [10, 20], [0, 20], [5, 10]],
    }
    base.update(overrides)
    return base


# --- unknown keys fail loudly -------------------------------------------------


def test_misspelled_top_level_key_is_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate(_app(sorce="clip.mp4"))


def test_misspelled_nested_key_is_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate(_app(detector=_detector(confidnce=0.5)))


def test_load_config_surfaces_the_file_path_in_the_error(tmp_path):
    bad = tmp_path / "typo.yaml"
    bad.write_text("source: clip.mp4\ndetectr:\n  classes: [car]\n")
    with pytest.raises(ConfigError) as exc:
        load_config(bad)
    assert str(bad) in str(exc.value)


def test_load_config_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "absent.yaml"
    with pytest.raises(ConfigError) as exc:
        load_config(missing)
    assert str(missing) in str(exc.value)


# --- detector -----------------------------------------------------------------


def test_unknown_class_name_fails_at_load():
    with pytest.raises(ValueError) as exc:
        DetectorConfig.model_validate(_detector(classes=["car", "trucks"]))
    assert "trucks" in str(exc.value)


def test_empty_class_list_is_rejected():
    with pytest.raises(ValueError):
        DetectorConfig.model_validate(_detector(classes=[]))


@pytest.mark.parametrize("confidence", [0.0, -0.1, 1.5])
def test_confidence_outside_zero_one_is_rejected(confidence):
    with pytest.raises(ValueError):
        DetectorConfig.model_validate(_detector(confidence=confidence))


@pytest.mark.parametrize("imgsz", [0, -640, 100, 641])
def test_imgsz_must_be_a_positive_multiple_of_32(imgsz):
    with pytest.raises(ValueError):
        DetectorConfig.model_validate(_detector(imgsz=imgsz))


def test_detector_model_defaults_to_yolo11s():
    detector = DetectorConfig.model_validate(_detector())
    assert detector.model == "yolo11s.pt"


# --- gates --------------------------------------------------------------------


def test_zero_length_gate_is_rejected_at_config_load():
    with pytest.raises(ValueError):
        GateConfig.model_validate(
            {"name": "g", "start": [0.5, 0.5], "end": [0.5, 0.5]}
        )


@pytest.mark.parametrize("point", [[-0.1, 0.5], [1.1, 0.5], [0.5, -0.1], [0.5, 1.2]])
def test_gate_coordinates_outside_unit_range_are_rejected(point):
    with pytest.raises(ValueError):
        GateConfig.model_validate({"name": "g", "start": point, "end": [0.9, 0.9]})


def test_expected_direction_must_match_a_label():
    with pytest.raises(ValueError):
        GateConfig.model_validate(
            {
                "name": "g",
                "start": [0.1, 0.5],
                "end": [0.9, 0.5],
                "expected_direction": "sideways",
            }
        )


def test_to_gate_converts_normalized_to_pixels():
    config = GateConfig.model_validate(
        {"name": "g", "start": [0.1, 0.5], "end": [0.9, 0.5]}
    )
    gate = config.to_gate(1280, 720)
    assert gate.name == "g"
    assert gate.start == (pytest.approx(128.0), pytest.approx(360.0))
    assert gate.end == (pytest.approx(1152.0), pytest.approx(360.0))


def test_duplicate_gate_names_are_rejected():
    gate = {"name": "twice", "start": [0.1, 0.5], "end": [0.9, 0.5]}
    with pytest.raises(ValueError) as exc:
        AppConfig.model_validate(_app(gates=[gate, dict(gate)]))
    assert "twice" in str(exc.value)


# --- calibration --------------------------------------------------------------


def test_fewer_than_four_correspondences_are_rejected():
    with pytest.raises(ValueError):
        CalibrationConfig.model_validate(
            {
                "image_points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]],
                "world_points": [[0, 0], [10, 0], [10, 20]],
            }
        )


def test_mismatched_image_and_world_lengths_are_rejected():
    bad = _square_calibration()
    bad["world_points"] = bad["world_points"][:-1]
    with pytest.raises(ValueError):
        CalibrationConfig.model_validate(bad)


def test_mismatched_holdout_lengths_are_rejected():
    bad = _square_calibration(
        holdout_image_points=[[0.5, 0.5]], holdout_world_points=[]
    )
    with pytest.raises(ValueError):
        CalibrationConfig.model_validate(bad)


def test_image_points_outside_unit_range_are_rejected():
    bad = _square_calibration()
    bad["image_points"][0] = [1.5, 0.5]
    with pytest.raises(ValueError):
        CalibrationConfig.model_validate(bad)


def test_exactly_four_points_without_holdout_are_refused_at_load():
    four = {
        "image_points": [[0.2, 0.8], [0.8, 0.8], [0.75, 0.3], [0.25, 0.3]],
        "world_points": [[0, 0], [10, 0], [10, 20], [0, 20]],
    }
    with pytest.raises(ValueError) as exc:
        CalibrationConfig.model_validate(four)
    message = str(exc.value)
    assert "4" in message
    assert "holdout" in message


def test_exactly_four_points_with_holdout_are_accepted():
    four = {
        "image_points": [[0.2, 0.8], [0.8, 0.8], [0.75, 0.3], [0.25, 0.3]],
        "world_points": [[0, 0], [10, 0], [10, 20], [0, 20]],
        "holdout_image_points": [[0.5, 0.55]],
        "holdout_world_points": [[5, 10]],
    }
    CalibrationConfig.model_validate(four)


def test_to_plane_builds_and_validates_a_road_plane():
    config = CalibrationConfig.model_validate(_square_calibration())
    plane = config.to_plane(1280, 720)
    # the synthetic square maps its own centre back to ~ (5, 10) metres
    wx, wy = plane.to_world((0.5 * 1280, 0.55 * 720))
    assert wx == pytest.approx(5.0, abs=1e-6)
    assert wy == pytest.approx(10.0, abs=1e-6)


def test_to_plane_surfaces_calibration_error_with_context():
    from trafficlens.core.homography import CalibrationError

    # collinear image points: builds nothing trustworthy
    bad = CalibrationConfig.model_validate(
        {
            "image_points": [[0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5], [0.9, 0.5]],
            "world_points": [[0, 0], [0, 10], [0, 20], [0, 30], [0, 40]],
        }
    )
    with pytest.raises(CalibrationError) as exc:
        bad.to_plane(1280, 720, context="configs/example.yaml")
    assert "configs/example.yaml" in str(exc.value)


# --- speed --------------------------------------------------------------------


def test_speed_unit_only_accepts_kmh_for_now():
    assert SpeedConfig().unit == "kmh"
    with pytest.raises(ValueError):
        SpeedConfig.model_validate({"unit": "mph"})


def test_speed_limit_must_be_positive_when_set():
    with pytest.raises(ValueError):
        SpeedConfig.model_validate({"limit": -30.0})


# --- integer webcam sources ---------------------------------------------------


def test_integer_source_is_accepted_and_kept_as_string():
    config = AppConfig.model_validate(_app(source=0))
    assert config.source == "0"


# --- shipped configs must always load -----------------------------------------


def test_shipped_configs_exist():
    names = sorted(p.name for p in CONFIGS.glob("*.yaml"))
    assert names == ["motorway.yaml", "street.yaml", "webcam.yaml"]


@pytest.mark.parametrize("name", ["motorway.yaml", "street.yaml", "webcam.yaml"])
def test_every_shipped_config_loads(name):
    config = load_config(CONFIGS / name)
    assert config.source
    assert config.detector.classes


def test_motorway_config_gates_stop_at_the_median():
    config = load_config(CONFIGS / "motorway.yaml")
    assert len(config.gates) == 2
    by_name = {g.name: g for g in config.gates}
    assert set(by_name) == {"inbound", "outbound"}
    assert by_name["inbound"].expected_direction == "in"
    assert by_name["outbound"].expected_direction == "out"
    # the two gates never overlap horizontally: each stays on its own
    # carriageway side of the median
    inbound_max_x = max(by_name["inbound"].start[0], by_name["inbound"].end[0])
    outbound_min_x = min(by_name["outbound"].start[0], by_name["outbound"].end[0])
    assert inbound_max_x < outbound_min_x


def test_motorway_calibration_builds_a_validated_plane():
    config = load_config(CONFIGS / "motorway.yaml")
    assert config.calibration is not None
    plane = config.calibration.to_plane(1280, 720, context="configs/motorway.yaml")
    # the surveyed lane strip: two dash centroids 18 m apart on the first
    # divider line must project ~18 m apart in world metres
    import math

    a = plane.to_world((204.5, 636.0))
    b = plane.to_world((329.7, 585.5))
    assert math.hypot(b[0] - a[0], b[1] - a[1]) == pytest.approx(18.0, abs=0.5)


def test_street_config_has_no_calibration_and_counts_people():
    config = load_config(CONFIGS / "street.yaml")
    assert config.calibration is None
    assert "person" in config.detector.classes
    assert "bicycle" in config.detector.classes
    assert len(config.gates) == 1


def test_webcam_config_uses_device_zero_and_person_class():
    from trafficlens.io.video import classify_spec

    config = load_config(CONFIGS / "webcam.yaml")
    assert classify_spec(config.source) == ("webcam", 0)
    assert config.detector.classes == ["person"]
    assert config.calibration is None
