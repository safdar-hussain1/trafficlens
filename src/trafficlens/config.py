"""Fail-fast configuration for a trafficlens session.

Every model here carries ``model_config = ConfigDict(extra="forbid")``: a
misspelled key anywhere in a YAML file is a hard load-time error, never a
silently ignored setting. The same philosophy applies to values -- unknown
detector class names, out-of-range gate coordinates, zero-length gates,
under-determined calibrations and duplicate gate names all fail at
``load_config`` time, before a single frame is read.

Coordinate convention: every image coordinate in a config file -- gate
endpoints and calibration image points alike -- is NORMALIZED to ``[0, 1]``
(x divided by frame width, y by frame height), so one config works across
resolutions of the same camera view. Conversion to pixels happens through
the explicit ``GateConfig.to_gate(width, height)`` and
``CalibrationConfig.to_plane(width, height)`` methods once the frame size
is known. Calibration world points are metres on the road plane.

This module must import neither cv2 nor torch: it is used by tools (tests,
docs generation, the web build) that run without them. The one cv2-backed
dependency, ``trafficlens.core.homography``, is imported lazily inside
``CalibrationConfig.to_plane``.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from trafficlens.core.classes import class_ids
from trafficlens.core.gate import Gate


class ConfigError(ValueError):
    """A config file could not be loaded: missing, not valid YAML, or its
    contents failed model validation. The message always names the file."""


class DetectorConfig(BaseModel):
    """Which model to run, on which classes, at which threshold and size."""

    # "model" is a perfectly natural key in a detector config; tell pydantic
    # not to reserve the "model_" namespace for itself.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str = "yolo11s.pt"
    classes: list[str]
    confidence: float
    imgsz: int

    @field_validator("classes")
    @classmethod
    def _classes_are_known(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError(
                "detector.classes must name at least one class -- an empty "
                "list would detect nothing, silently"
            )
        class_ids(value)  # raises ValueError naming any unknown class
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"detector.confidence must be in (0, 1], got {value}"
            )
        return value

    @field_validator("imgsz")
    @classmethod
    def _imgsz_is_a_positive_multiple_of_32(cls, value: int) -> int:
        if value <= 0 or value % 32 != 0:
            raise ValueError(
                f"detector.imgsz must be a positive multiple of 32 (the "
                f"model's stride grid), got {value}"
            )
        return value


class GateConfig(BaseModel):
    """One counting gate, in normalized coordinates. ``to_gate`` converts
    to the pixel-space ``trafficlens.core.gate.Gate`` once the frame size
    is known."""

    model_config = ConfigDict(extra="forbid")

    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    label_positive: str = "in"
    label_negative: str = "out"
    expected_direction: str | None = None

    @field_validator("name")
    @classmethod
    def _name_is_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("a gate must have a non-empty name")
        return value

    @field_validator("start", "end")
    @classmethod
    def _coordinates_are_normalized(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        for axis_name, axis in zip(("x", "y"), value):
            if not (0.0 <= axis <= 1.0):
                raise ValueError(
                    f"gate coordinates are normalized: {axis_name}={axis} "
                    f"is outside [0, 1]"
                )
        return value

    @model_validator(mode="after")
    def _geometry_and_labels_are_coherent(self) -> "GateConfig":
        if self.start == self.end:
            raise ValueError(
                f"gate {self.name!r} has identical start and end "
                f"{self.start!r}: a zero-length gate can never be crossed"
            )
        if self.label_positive == self.label_negative:
            raise ValueError(
                f"gate {self.name!r} uses the same label "
                f"{self.label_positive!r} for both directions"
            )
        if self.expected_direction is not None and self.expected_direction not in (
            self.label_positive,
            self.label_negative,
        ):
            raise ValueError(
                f"gate {self.name!r} expected_direction "
                f"{self.expected_direction!r} matches neither label "
                f"({self.label_positive!r}, {self.label_negative!r})"
            )
        return self

    def to_gate(self, width: float, height: float) -> Gate:
        """The pixel-space Gate this config describes on a width x height
        frame."""
        return Gate.from_normalized(
            self.name,
            self.start,
            self.end,
            width,
            height,
            label_positive=self.label_positive,
            label_negative=self.label_negative,
            expected_direction=self.expected_direction,
        )


class CalibrationConfig(BaseModel):
    """Surveyed image/world correspondences for one camera view.

    ``image_points`` are NORMALIZED [0, 1] image coordinates;
    ``world_points`` are metres on the road plane. The optional holdout
    pair holds surveyed points deliberately kept out of the fit, for a
    genuine out-of-sample check in ``RoadPlane.validate``.

    The engine refuses to validate a plane built from exactly 4
    correspondences without a holdout (4 points exactly determine a
    homography, so a self-check on them can never fail); this model
    enforces the same policy at load time, so a config that could never
    pass validation is rejected before any video is opened.
    """

    model_config = ConfigDict(extra="forbid")

    image_points: list[tuple[float, float]]
    world_points: list[tuple[float, float]]
    holdout_image_points: list[tuple[float, float]] = Field(default_factory=list)
    holdout_world_points: list[tuple[float, float]] = Field(default_factory=list)

    @field_validator("image_points", "holdout_image_points")
    @classmethod
    def _image_points_are_normalized(
        cls, value: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        for point in value:
            for axis in point:
                if not (0.0 <= axis <= 1.0):
                    raise ValueError(
                        f"calibration image points are normalized: {point!r} "
                        f"is outside [0, 1] x [0, 1]"
                    )
        return value

    @model_validator(mode="after")
    def _correspondences_are_usable(self) -> "CalibrationConfig":
        if len(self.image_points) != len(self.world_points):
            raise ValueError(
                f"calibration has {len(self.image_points)} image points but "
                f"{len(self.world_points)} world points; they pair one-to-one"
            )
        if len(self.holdout_image_points) != len(self.holdout_world_points):
            raise ValueError(
                f"calibration has {len(self.holdout_image_points)} holdout "
                f"image points but {len(self.holdout_world_points)} holdout "
                f"world points; they pair one-to-one"
            )
        if len(self.image_points) < 4:
            raise ValueError(
                f"a homography needs at least 4 correspondences, got "
                f"{len(self.image_points)}"
            )
        if len(self.image_points) == 4 and not self.holdout_image_points:
            raise ValueError(
                "calibration has exactly 4 correspondences and no holdout "
                "points. 4 points exactly determine a homography, so a "
                "reprojection self-check on them can never fail and the "
                "calibration can never be validated. Survey at least one "
                "more point: either add it as a 5th correspondence or as a "
                "holdout_image_points/holdout_world_points pair."
            )
        return self

    def to_plane(self, width: float, height: float, context: str | None = None):
        """Build AND validate the ``RoadPlane`` for a width x height frame.

        Raises ``trafficlens.core.homography.CalibrationError`` when the
        surveyed points cannot produce a trustworthy plane; ``context``
        (typically the config file path) is prefixed to the error so the
        user knows which file to fix.
        """
        # Deferred import: homography needs cv2, and this module must be
        # importable without it.
        from trafficlens.core.homography import CalibrationError, RoadPlane

        def denormalize(points):
            return [(x * width, y * height) for x, y in points]

        try:
            plane = RoadPlane.from_correspondences(
                denormalize(self.image_points), list(self.world_points)
            )
            plane.validate(
                holdout_image_pts=(
                    denormalize(self.holdout_image_points)
                    if self.holdout_image_points
                    else None
                ),
                holdout_world_pts=(
                    [tuple(p) for p in self.holdout_world_points]
                    if self.holdout_world_points
                    else None
                ),
            )
        except CalibrationError as error:
            prefix = f"{context}: " if context else ""
            raise CalibrationError(f"{prefix}{error}") from error
        return plane


class SpeedConfig(BaseModel):
    """Speed reporting unit and optional enforcement limit. Only km/h for
    now; the unit field exists so mph can be added without a schema
    change."""

    model_config = ConfigDict(extra="forbid")

    unit: Literal["kmh"] = "kmh"
    limit: float | None = None

    @field_validator("limit")
    @classmethod
    def _limit_is_positive(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError(f"speed.limit must be positive when set, got {value}")
        return value


class AppConfig(BaseModel):
    """One complete session configuration: a source plus everything the
    pipeline needs to analyse it."""

    model_config = ConfigDict(extra="forbid")

    source: str
    detector: DetectorConfig
    gates: list[GateConfig] = Field(default_factory=list)
    calibration: CalibrationConfig | None = None
    speed: SpeedConfig = Field(default_factory=SpeedConfig)

    @field_validator("source", mode="before")
    @classmethod
    def _source_accepts_a_bare_webcam_index(cls, value):
        # `source: 0` in YAML parses as an int; keep it as the string "0",
        # which VideoSource.open classifies as webcam index 0.
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value

    @field_validator("source")
    @classmethod
    def _source_is_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("source must not be empty")
        return value

    @model_validator(mode="after")
    def _gate_names_are_unique(self) -> "AppConfig":
        seen: set[str] = set()
        for gate in self.gates:
            if gate.name in seen:
                raise ValueError(
                    f"duplicate gate name {gate.name!r}: gate names key "
                    f"counts and events, so each must be unique"
                )
            seen.add(gate.name)
        return self


def load_config(path) -> AppConfig:
    """Load and fully validate a YAML config file.

    Raises ``ConfigError`` -- always naming the file -- when the file is
    missing, is not valid YAML, is not a mapping, or fails any model
    validation above (including unknown keys anywhere in the tree).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"config file not found: {file_path}")
    try:
        data = yaml.safe_load(file_path.read_text())
    except yaml.YAMLError as error:
        raise ConfigError(f"{file_path}: not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(
            f"{file_path}: a config file must be a YAML mapping, got "
            f"{type(data).__name__}"
        )
    try:
        return AppConfig.model_validate(data)
    except ValidationError as error:
        raise ConfigError(f"{file_path}: {error}") from error
