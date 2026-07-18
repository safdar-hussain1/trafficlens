"""Config validation must fail fast and loudly on bad input."""

import pytest
from pydantic import ValidationError

from trafficlens.config import AppConfig, CalibrationConfig, DetectorConfig, GateConfig, load_config

VALID_YAML = """
source: data/samples/car-detection.mp4
detector:
  model: yolo11n.pt
  classes: [car, truck, bus]
  confidence: 0.4
gates:
  - name: main
    start: [0.1, 0.5]
    end: [0.9, 0.5]
    label_positive: south
    label_negative: north
calibration:
  mode: homography
  image_points: [[0.2, 0.9], [0.8, 0.9], [0.6, 0.3], [0.4, 0.3]]
  world_points: [[0, 0], [7, 0], [7, 40], [0, 40]]
speed:
  unit: kmh
  speed_limit: 80
"""


def test_valid_yaml_loads(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML)
    config = load_config(p)
    assert config.gates[0].name == "main"
    assert config.calibration.mode == "homography"
    assert config.speed.speed_limit == 80


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_yaml_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "typo.yaml"
    p.write_text("soruce: video.mp4\n")  # typo'd key must not pass silently
    with pytest.raises(ValidationError):
        load_config(p)


def test_gate_coordinates_must_be_normalized():
    with pytest.raises(ValidationError, match="normalized"):
        GateConfig(name="g", start=(383, 297), end=(666, 297))  # pixels, not fractions


def test_zero_length_gate_rejected():
    with pytest.raises(ValidationError, match="zero length"):
        GateConfig(name="g", start=(0.5, 0.5), end=(0.5, 0.5))


def test_duplicate_gate_names_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        AppConfig(gates=[
            GateConfig(name="main", start=(0.1, 0.5), end=(0.9, 0.5)),
            GateConfig(name="main", start=(0.1, 0.7), end=(0.9, 0.7)),
        ])


def test_empty_classes_rejected():
    with pytest.raises(ValidationError, match="classes"):
        DetectorConfig(classes=[])


def test_homography_needs_four_points():
    with pytest.raises(ValidationError, match="4 point"):
        CalibrationConfig(
            mode="homography",
            image_points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)],
            world_points=[(0, 0), (7, 0), (7, 40)],
        )


def test_homography_point_counts_must_match():
    with pytest.raises(ValidationError, match="pair up"):
        CalibrationConfig(
            mode="homography",
            image_points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
            world_points=[(0, 0), (7, 0), (7, 40)],
        )


def test_scale_mode_needs_positive_factor():
    with pytest.raises(ValidationError, match="meters_per_pixel"):
        CalibrationConfig(mode="scale")


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        DetectorConfig(confidence=0.99)
