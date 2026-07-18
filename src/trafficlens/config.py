"""Configuration models — validated, fail-fast, YAML-loadable.

All geometry in config files uses **normalized coordinates** (0.0-1.0
fractions of frame width/height), so the same config works at any
resolution and survives a camera being reconfigured from 720p to 4K.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NormPoint = tuple[float, float]


class _StrictModel(BaseModel):
    """Reject unknown keys so a typo in a YAML file fails loudly."""

    model_config = ConfigDict(extra="forbid")


def _check_norm_point(p: NormPoint, where: str) -> NormPoint:
    x, y = p
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(
            f"{where} point ({x}, {y}) is outside [0, 1] — config coordinates "
            "are normalized fractions of frame size, not pixels"
        )
    return p


class GateConfig(_StrictModel):
    """A virtual counting gate: a directed line segment across the frame."""

    name: str = Field(min_length=1, max_length=64)
    start: NormPoint
    end: NormPoint
    label_positive: str = "in"
    label_negative: str = "out"

    @field_validator("start", "end")
    @classmethod
    def _in_unit_square(cls, v: NormPoint) -> NormPoint:
        return _check_norm_point(v, "gate")

    @model_validator(mode="after")
    def _not_degenerate(self) -> "GateConfig":
        if self.start == self.end:
            raise ValueError(f"gate '{self.name}' has zero length (start == end)")
        return self


class CalibrationConfig(_StrictModel):
    """Maps the image road plane to real-world metres.

    Two modes:

    * ``homography`` — four image points (normalized) matched to four
      world points in metres. Exact for a flat road plane; this is what
      real traffic cameras use.
    * ``scale`` — a single metres-per-pixel factor at native resolution.
      A quick approximation for near-orthographic views (drone/overpass
      footage shot straight down the road).
    """

    mode: str = Field(pattern="^(homography|scale)$")
    image_points: list[NormPoint] | None = None
    world_points: list[tuple[float, float]] | None = None
    meters_per_pixel: float | None = None
    reference_width: int = Field(default=1280, gt=0)

    @model_validator(mode="after")
    def _mode_requirements(self) -> "CalibrationConfig":
        if self.mode == "homography":
            if not self.image_points or not self.world_points:
                raise ValueError("homography calibration needs image_points and world_points")
            if len(self.image_points) != len(self.world_points):
                raise ValueError(
                    f"image_points ({len(self.image_points)}) and world_points "
                    f"({len(self.world_points)}) must pair up"
                )
            if len(self.image_points) < 4:
                raise ValueError("homography needs at least 4 point pairs")
            for p in self.image_points:
                _check_norm_point(p, "calibration image")
        else:  # scale
            if self.meters_per_pixel is None or self.meters_per_pixel <= 0:
                raise ValueError("scale calibration needs meters_per_pixel > 0")
        return self


class DetectorConfig(_StrictModel):
    """Model + filtering settings for the detector/tracker."""

    model: str = "yolo11n.pt"
    classes: list[str] = Field(default_factory=lambda: ["car", "truck", "bus", "motorcycle"])
    confidence: float = Field(default=0.35, ge=0.05, le=0.95)
    tracker: str = Field(default="bytetrack", pattern="^(bytetrack|botsort)$")
    device: str = "auto"
    imgsz: int = Field(default=640, ge=160, le=1920)

    @field_validator("classes")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("classes must not be empty — nothing would ever be detected")
        return v


class SpeedConfig(_StrictModel):
    """Speed estimation and violation settings."""

    window_seconds: float = Field(default=0.5, gt=0.1, le=3.0)
    smoothing: float = Field(default=0.35, ge=0.0, le=1.0)
    unit: str = Field(default="kmh", pattern="^(kmh|mph)$")
    speed_limit: float | None = Field(default=None, gt=0)
    min_travel_m: float = Field(default=0.4, ge=0.0)


class AppConfig(_StrictModel):
    """Top-level config: what to detect, where to count, how to calibrate."""

    source: str = "0"
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    gates: list[GateConfig] = Field(default_factory=list)
    calibration: CalibrationConfig | None = None
    speed: SpeedConfig = Field(default_factory=SpeedConfig)

    @model_validator(mode="after")
    def _unique_gate_names(self) -> "AppConfig":
        names = [g.name for g in self.gates]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate gate names: {sorted(dupes)}")
        return self


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config file, failing fast with context."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping, got {type(raw).__name__}")
    return AppConfig.model_validate(raw)
