"""Slit-scan ground truth: an independent way to see every vehicle that
crosses a gate, and a loader that refuses a label set it cannot trust.

A **slit-scan** is the band of pixels lying along the gate segment, one
row per frame, stacked with time running downward. Any object that
physically passes over the gate paints a blob in that image, and the row
of the blob *is* the frame index at which it passed. That makes the
slit-scan an instrument for counting crossings which shares nothing with
the pipeline it is used to score: no detector, no tracker, no
association threshold, no confidence score. A ground truth built from
the system under test would only measure how consistent that system is
with itself, so the ground truth is built from raw pixels instead, and a
human reads it under ``data/groundtruth/PROTOCOL.md``.

**This module never produces a label.** It produces the image a labeller
reads, and it validates the file the labeller writes. Deciding that a
blob is a lorry travelling toward the camera on frame 412 is a human
judgement made against the footage, and it is the only part of the
process that may not be automated.

Sampling rule, stated exactly so review images are reproducible:

- ``samples`` points spaced evenly along the gate segment from ``start``
  to ``end`` inclusive, at parameter ``t = i / (samples - 1)``.
- ``thickness_px`` pixels read perpendicular to the gate at integer
  offsets centred on the line -- an even thickness straddles the line, an
  odd one includes it -- then averaged.
- Each pixel is read by NEAREST NEIGHBOUR: the sample coordinate is
  rounded HALF UP (``floor(v + 0.5)``, so 3.5 -> 4 and -3.5 -> -3) and
  clamped to the frame bounds, so an offset that leaves the frame
  repeats the edge pixel. Clamped, never wrapped and never reflected: a
  wrap would splice the far edge of the frame into a strip sampled at
  the near edge.
- Nearest neighbour rather than bilinear, deliberately: the review image
  should show the pixels the camera recorded, not a blend of them.
  Bilinear interpolation dims and smears a small, fast, high-contrast
  object -- precisely the vehicle that is hardest to label -- while the
  perpendicular average already supplies all the smoothing the strip
  needs. Integer indexing is also exactly reproducible, so the same clip
  yields byte-identical review images on any platform and any OpenCV
  build.
- The average is rounded HALF UP (``floor(v + 0.5)``, not numpy's
  half-to-even ``np.round``) and cast back to the frame's own dtype
  (integer dtypes only; a float frame keeps its floats), so a strip is
  directly writable as an image.

Dependencies are numpy, the standard library and ``trafficlens.core`` /
``trafficlens.io`` only -- no torch, no ultralytics. The measuring
instrument must be installable and runnable without the thing it
measures.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trafficlens.core.classes import VEHICLE_CLASSES
from trafficlens.core.gate import Gate
from trafficlens.io.video import SourceError, VideoSource

#: Perpendicular band, in pixels, averaged into each strip by default.
#: Wide enough that a vehicle is not missed between two consecutive
#: frames' samples, narrow enough that two vehicles a lane apart along
#: the gate stay separate blobs.
DEFAULT_THICKNESS_PX = 9

#: Samples along the gate by default. The shipped motorway gates are
#: 200-520 px long on a 1280-wide frame, so this over-samples slightly
#: rather than dropping pixels.
DEFAULT_SAMPLES = 512

#: The only two confidence values a label may carry. See PROTOCOL.md.
CONFIDENCE_VALUES = ("certain", "probable")

#: The schema version this loader understands.
SCHEMA_VERSION = 1

_TOP_LEVEL_FIELDS = (
    "schema",
    "clip",
    "fps",
    "window",
    "gate",
    "protocol",
    "labeller",
    "labelled_on",
    "crossings",
)
_WINDOW_FIELDS = ("start_frame", "end_frame")
_GATE_FIELDS = ("name", "start", "end")
_CROSSING_FIELDS = ("id", "frame", "class", "direction", "confidence")

#: How far a label set's gate endpoints may sit from the gate it is
#: scored against, in pixels. Half a pixel: the same gate, written down
#: with a rounded decimal, still matches; a gate that was moved does not.
_GATE_TOLERANCE_PX = 0.5

#: How far a label set's declared frame rate may sit from the clip's own.
_FPS_TOLERANCE = 0.01


# --- the instrument ---------------------------------------------------------


def gate_strip(
    frame: np.ndarray,
    gate: Gate,
    thickness_px: int = DEFAULT_THICKNESS_PX,
    samples: int = DEFAULT_SAMPLES,
) -> np.ndarray:
    """The band of pixels lying along ``gate``, resampled to a fixed length.

    Returns ``samples`` values along the gate segment (shape
    ``(samples,)`` for a single-channel frame, ``(samples, channels)``
    otherwise), each the average of ``thickness_px`` pixels read
    perpendicular to the gate. See the module docstring for the exact
    sampling rule.

    The gate is a BOUNDED segment: pixels beyond either endpoint -- on
    the gate line's infinite extension, where another carriageway may
    well be -- are never sampled.
    """
    if samples < 2:
        raise ValueError(
            f"samples must be at least 2 to span the gate's two endpoints, "
            f"got {samples}"
        )
    if thickness_px < 1:
        raise ValueError(
            f"thickness_px must be at least 1 pixel, got {thickness_px}"
        )
    if frame.ndim not in (2, 3):
        raise ValueError(
            f"frame must be a 2-D or 3-D image array, got shape {frame.shape!r}"
        )

    height, width = frame.shape[0], frame.shape[1]
    (start_x, start_y), (end_x, end_y) = gate.start, gate.end
    span_x, span_y = end_x - start_x, end_y - start_y
    length = math.hypot(span_x, span_y)
    if length == 0:
        # Gate.__post_init__ already refuses this; belt and braces, because
        # a zero-length gate here would divide by zero below.
        raise ValueError(f"gate {gate.name!r} has zero length")

    # Unit normal to the gate: the direction the band is thickened along.
    normal_x, normal_y = -span_y / length, span_x / length

    # Positions along the gate, both endpoints inclusive.
    t = np.arange(samples, dtype=np.float64) / (samples - 1)
    along_x = start_x + t * span_x
    along_y = start_y + t * span_y

    # Integer offsets centred on the line: an even thickness straddles it.
    offsets = np.arange(thickness_px, dtype=np.float64) - (thickness_px - 1) / 2.0

    sample_x = along_x[:, None] + offsets[None, :] * normal_x
    sample_y = along_y[:, None] + offsets[None, :] * normal_y

    # Nearest neighbour, rounding half up, clamped to the frame.
    column = np.clip(np.floor(sample_x + 0.5).astype(np.int64), 0, width - 1)
    row = np.clip(np.floor(sample_y + 0.5).astype(np.int64), 0, height - 1)

    band = frame[row, column]                      # (samples, thickness[, ch])
    averaged = band.astype(np.float64).mean(axis=1)
    if np.issubdtype(frame.dtype, np.integer):
        info = np.iinfo(frame.dtype)
        rounded = np.floor(averaged + 0.5)
        return np.clip(rounded, info.min, info.max).astype(frame.dtype)
    return averaged.astype(frame.dtype)


def build_slitscan(
    source,
    gate: Gate,
    frames,
    thickness_px: int = DEFAULT_THICKNESS_PX,
    samples: int = DEFAULT_SAMPLES,
) -> np.ndarray:
    """Stack one ``gate_strip`` per requested frame, time running downward.

    ``source`` is anything iterable that yields
    ``(frame_index, timestamp_s, frame)`` -- in practice a
    ``trafficlens.io.video.VideoSource``, which is SINGLE-PASS: it hands
    out exactly one iterator, ever. This function therefore iterates it
    once, forward only, taking the frames it wants as they go past, and
    stops as soon as the last requested frame has been read.

    ``frames`` is the requested frame indices in strictly increasing
    order (a ``range`` is the usual case). Row ``r`` of the result is the
    strip for ``frames[r]``, so a blob's row maps straight back to a
    frame index.

    Raises ``ValueError`` if the source ends before every requested frame
    has been seen, or skips one: a slit-scan silently missing rows would
    shift every crossing frame a labeller reads off it.
    """
    wanted = [int(index) for index in frames]
    if not wanted:
        raise ValueError("a slit-scan needs at least one frame; frames was empty")
    if wanted[0] < 0:
        raise ValueError(f"frame indices cannot be negative, got {wanted[0]}")
    for previous, current in zip(wanted, wanted[1:]):
        if current <= previous:
            raise ValueError(
                f"frames must be strictly increasing -- the source is read "
                f"forward, once -- but {current} follows {previous}"
            )

    rows: list[np.ndarray] = []
    position = 0
    for frame_index, _timestamp, frame in source:
        target = wanted[position]
        if frame_index < target:
            continue
        if frame_index > target:
            raise ValueError(
                f"the source skipped frame {target}: it went from before it "
                f"straight to frame {frame_index}, so the slit-scan would be "
                f"missing a row and every row below it would be mislabelled"
            )
        rows.append(gate_strip(frame, gate, thickness_px, samples))
        position += 1
        if position >= len(wanted):
            break

    if position < len(wanted):
        raise ValueError(
            f"the source ended after {position} of the {len(wanted)} "
            f"requested frames (it never reached frame {wanted[position]}); "
            f"shorten the window to what the clip actually contains"
        )
    return np.stack(rows, axis=0)


def tile_windows(
    start_frame: int, end_frame: int, tile_frames: int, overlap: int
) -> list[tuple[int, int]]:
    """Split ``[start_frame, end_frame]`` into overlapping tiles.

    Returns inclusive ``(start, end)`` pairs, each at most ``tile_frames``
    long, consecutive tiles sharing ``overlap`` frames. The overlap is
    what stops a crossing being cut in half by a tile boundary: any
    crossing near an edge appears whole in the neighbouring tile. The
    last tile always ends exactly on ``end_frame``.
    """
    if end_frame < start_frame:
        raise ValueError(
            f"end_frame {end_frame} is before start_frame {start_frame}"
        )
    if tile_frames < 1:
        raise ValueError(f"tile_frames must be at least 1, got {tile_frames}")
    if overlap < 0 or overlap >= tile_frames:
        raise ValueError(
            f"overlap must be at least 0 and less than tile_frames "
            f"({tile_frames}), got {overlap} -- an overlap that large would "
            f"never advance through the clip"
        )

    step = tile_frames - overlap
    windows: list[tuple[int, int]] = []
    tile_start = start_frame
    while True:
        tile_end = min(tile_start + tile_frames - 1, end_frame)
        windows.append((tile_start, tile_end))
        if tile_end >= end_frame:
            return windows
        tile_start += step


# --- the label set ----------------------------------------------------------


class GroundTruthError(ValueError):
    """A ground-truth label file could not be loaded or failed validation.

    The message always names the file. A silently accepted bad label set
    poisons every accuracy number computed against it, so every check
    here fails loudly rather than repairing, defaulting or skipping.
    """


@dataclass(frozen=True)
class Crossing:
    """One hand-labelled crossing. ``class_name`` is the JSON's ``class``
    key, renamed only because ``class`` is a Python keyword."""

    id: int
    frame: int
    class_name: str
    direction: str
    confidence: str


@dataclass(frozen=True)
class GroundTruth:
    """A validated label set: what a human saw crossing one gate, in one
    window of one clip, under one protocol."""

    path: Path
    clip: str
    fps: float
    start_frame: int
    end_frame: int
    gate_name: str
    gate_start: tuple[float, float]
    gate_end: tuple[float, float]
    protocol: str
    labeller: str
    labelled_on: str
    crossings: tuple[Crossing, ...]

    @property
    def certain_count(self) -> int:
        """Crossings labelled ``certain``. Report this alongside the total,
        never instead of it: quoting the certain-only figure silently drops
        every hard case and flatters whatever is being scored."""
        return sum(1 for c in self.crossings if c.confidence == "certain")

    def frames_by_direction(self, direction: str) -> tuple[int, ...]:
        return tuple(c.frame for c in self.crossings if c.direction == direction)

    @classmethod
    def load(cls, path, *, gate: Gate, clip_path) -> "GroundTruth":
        """Load and fully validate a label file.

        ``gate`` is the gate the labels were made against -- required,
        because the two direction labels a crossing may carry are the
        gate's own, and because a label set scored against a gate that has
        since moved is worse than no label set. ``clip_path`` is the real
        footage -- also required, because the file's ``clip`` name and
        ``fps`` are checked against it rather than taken on trust.
        """
        label_path = Path(path)
        if not label_path.is_file():
            raise GroundTruthError(f"ground-truth file not found: {label_path}")
        try:
            document = json.loads(label_path.read_text())
        except json.JSONDecodeError as error:
            raise GroundTruthError(
                f"{label_path}: not valid JSON: {error}"
            ) from error

        truth = cls._parse(label_path, document, gate)
        truth._check_gate_name(gate)
        # The clip is what fixes the pixel size the normalized gate
        # coordinates are compared in, so it is verified first.
        width, height = truth._check_clip(Path(clip_path))
        truth._check_gate_geometry(gate, width, height)
        return truth

    # --- parsing and structural validation ---------------------------------

    @classmethod
    def _parse(cls, label_path: Path, document, gate: Gate) -> "GroundTruth":
        def fail(message: str):
            return GroundTruthError(f"{label_path}: {message}")

        if not isinstance(document, dict):
            raise fail(
                f"a ground-truth file must be a JSON object, got "
                f"{type(document).__name__}"
            )
        _require_exact_fields(document, _TOP_LEVEL_FIELDS, "the label file", fail)

        schema = document["schema"]
        if schema != SCHEMA_VERSION:
            raise fail(
                f"unsupported schema {schema!r}; this loader understands "
                f"schema {SCHEMA_VERSION} only"
            )

        clip = document["clip"]
        if not isinstance(clip, str) or not clip:
            raise fail(f"clip must be a non-empty file name, got {clip!r}")
        if "/" in clip or "\\" in clip:
            raise fail(
                f"clip must be a bare file name, not a path, got {clip!r} -- "
                f"the same footage lives at different paths on different "
                f"machines"
            )

        fps = document["fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise fail(f"fps must be a number, got {fps!r}")
        if fps <= 0:
            raise fail(f"fps must be positive, got {fps!r}")

        for field in ("protocol", "labeller", "labelled_on"):
            value = document[field]
            if not isinstance(value, str) or not value:
                raise fail(f"{field} must be a non-empty string, got {value!r}")

        window = document["window"]
        if not isinstance(window, dict):
            raise fail(f"window must be an object, got {type(window).__name__}")
        _require_exact_fields(window, _WINDOW_FIELDS, "window", fail)
        start_frame = _as_index(window["start_frame"], "window.start_frame", fail)
        end_frame = _as_index(window["end_frame"], "window.end_frame", fail)
        if end_frame < start_frame:
            raise fail(
                f"window start_frame {start_frame} is after end_frame "
                f"{end_frame}"
            )

        gate_block = document["gate"]
        if not isinstance(gate_block, dict):
            raise fail(f"gate must be an object, got {type(gate_block).__name__}")
        _require_exact_fields(gate_block, _GATE_FIELDS, "gate", fail)
        gate_name = gate_block["name"]
        if not isinstance(gate_name, str) or not gate_name:
            raise fail(f"gate.name must be a non-empty string, got {gate_name!r}")
        gate_start = _as_normalized_point(gate_block["start"], "gate.start", fail)
        gate_end = _as_normalized_point(gate_block["end"], "gate.end", fail)
        if gate_start == gate_end:
            raise fail(
                f"gate.start and gate.end are both {gate_start!r}: a "
                f"zero-length gate can never be crossed"
            )

        directions = (gate.label_positive, gate.label_negative)
        crossings = cls._parse_crossings(
            document["crossings"], start_frame, end_frame, directions, fail
        )

        return cls(
            path=label_path,
            clip=clip,
            fps=float(fps),
            start_frame=start_frame,
            end_frame=end_frame,
            gate_name=gate_name,
            gate_start=gate_start,
            gate_end=gate_end,
            protocol=document["protocol"],
            labeller=document["labeller"],
            labelled_on=document["labelled_on"],
            crossings=crossings,
        )

    @staticmethod
    def _parse_crossings(
        raw, start_frame: int, end_frame: int, directions, fail
    ) -> tuple[Crossing, ...]:
        if not isinstance(raw, list):
            raise fail(f"crossings must be a list, got {type(raw).__name__}")

        crossings: list[Crossing] = []
        seen_ids: set[int] = set()
        previous_frame: int | None = None
        for position, entry in enumerate(raw):
            where = f"crossings[{position}]"
            if not isinstance(entry, dict):
                raise fail(f"{where} must be an object, got {type(entry).__name__}")
            _require_exact_fields(entry, _CROSSING_FIELDS, where, fail)

            crossing_id = _as_index(entry["id"], f"{where}.id", fail)
            if crossing_id <= 0:
                raise fail(f"{where}.id must be a positive integer, got {crossing_id}")
            if crossing_id in seen_ids:
                raise fail(
                    f"{where} carries the duplicate id {crossing_id}; ids "
                    f"identify one crossing each and must be unique"
                )
            seen_ids.add(crossing_id)

            frame = _as_index(entry["frame"], f"{where}.frame", fail)
            if not (start_frame <= frame <= end_frame):
                raise fail(
                    f"{where} (id {crossing_id}) is at frame {frame}, outside "
                    f"the labelled window {start_frame}-{end_frame}"
                )
            if previous_frame is not None and frame < previous_frame:
                raise fail(
                    f"{where} (id {crossing_id}) is at frame {frame}, before "
                    f"the preceding crossing's frame {previous_frame}; "
                    f"crossings are recorded in frame order (equal frames are "
                    f"allowed -- two vehicles abreast -- but frames never go "
                    f"backwards)"
                )
            previous_frame = frame

            class_name = entry["class"]
            if class_name not in VEHICLE_CLASSES:
                raise fail(
                    f"{where} (id {crossing_id}) has class {class_name!r}, "
                    f"which is not one of the labelling vocabulary: "
                    f"{', '.join(VEHICLE_CLASSES)}"
                )

            direction = entry["direction"]
            if direction not in directions:
                raise fail(
                    f"{where} (id {crossing_id}) has direction {direction!r}, "
                    f"which is neither of the gate's two labels "
                    f"({directions[0]!r}, {directions[1]!r})"
                )

            confidence = entry["confidence"]
            if confidence not in CONFIDENCE_VALUES:
                raise fail(
                    f"{where} (id {crossing_id}) has confidence "
                    f"{confidence!r}; the protocol allows only "
                    f"{' and '.join(CONFIDENCE_VALUES)}"
                )

            crossings.append(
                Crossing(
                    id=crossing_id,
                    frame=frame,
                    class_name=class_name,
                    direction=direction,
                    confidence=confidence,
                )
            )
        return tuple(crossings)

    # --- validation against the world --------------------------------------

    def _check_gate_name(self, gate: Gate) -> None:
        """The label set must describe the very gate it is scored against."""
        if gate.name != self.gate_name:
            raise GroundTruthError(
                f"{self.path}: labelled against gate {self.gate_name!r} but "
                f"loaded against gate {gate.name!r}"
            )

    def _check_clip(self, clip_path: Path) -> tuple[int, int]:
        """The clip name, frame rate and length must match the real file.

        A label set whose frame rate disagrees with the footage has every
        crossing frame in the wrong place, and nothing downstream would
        notice.

        The window's last frame is verified by DECODING to it, not by
        comparing it against the container's advertised frame count. That
        count is an upper bound: ``data/samples/motorway-a40.webm``
        advertises 737 frames and decodes 735 (indices 0-734), so a
        window ending at 736 would pass a header check and then reference
        a frame no decoder ever produces. PROTOCOL.md's rule -- a label
        set must never reference a frame whose existence depends on which
        decoder opened the file -- can only be enforced by asking the
        decoder. This runs once per label set, on a clip that is already
        open, so the cost of playing the window out is affordable.
        """
        if clip_path.name != self.clip:
            raise GroundTruthError(
                f"{self.path}: labelled against clip {self.clip!r} but loaded "
                f"against {clip_path.name!r}"
            )
        try:
            source = VideoSource.open(str(clip_path))
        except SourceError as error:
            raise GroundTruthError(
                f"{self.path}: its clip {clip_path} could not be opened, so "
                f"the label set cannot be verified against the footage: {error}"
            ) from error
        try:
            actual_fps = source.fps
            advertised = source.frame_count
            width, height = source.width, source.height
            if actual_fps is None or abs(actual_fps - self.fps) > _FPS_TOLERANCE:
                raise GroundTruthError(
                    f"{self.path}: labelled at fps {self.fps} but "
                    f"{clip_path.name} reports {actual_fps}; every crossing "
                    f"frame would map to the wrong time"
                )
            last_decoded = _decode_up_to(source, self.end_frame)
        finally:
            source.close()

        if last_decoded < self.end_frame:
            claim = (
                f" (its container advertises {advertised} frames, an upper "
                f"bound this decoder does not reach)"
                if advertised is not None and advertised > last_decoded + 1
                else ""
            )
            raise GroundTruthError(
                f"{self.path}: the labelled window ends at frame "
                f"{self.end_frame} but the clip {clip_path.name} decodes only "
                f"frames 0-{last_decoded}{claim}; a label set must never "
                f"reference a frame whose existence depends on which decoder "
                f"opened the file"
            )
        return width, height

    def _gate_pixel_endpoints(self, width: float, height: float):
        """This label set's gate in pixel coordinates on a width x height
        frame -- the same conversion ``Gate.from_normalized`` makes."""
        return (
            (self.gate_start[0] * width, self.gate_start[1] * height),
            (self.gate_end[0] * width, self.gate_end[1] * height),
        )

    def _check_gate_geometry(self, gate: Gate, width: float, height: float) -> None:
        """Verify this label set's normalized gate lands on ``gate``'s pixel
        endpoints, to within half a pixel."""
        start, end = self._gate_pixel_endpoints(width, height)
        for what, labelled, scored in (("start", start, gate.start), ("end", end, gate.end)):
            if math.dist(labelled, scored) > _GATE_TOLERANCE_PX:
                raise GroundTruthError(
                    f"{self.path}: the labelled gate {what} is at {labelled} "
                    f"px but the gate it is scored against starts at "
                    f"{scored} px; the gate moved after labelling, so these "
                    f"labels describe a different line"
                )


def _decode_up_to(source, end_frame: int) -> int:
    """Play ``source`` forward until ``end_frame`` has been decoded, and
    return the highest frame index the decoder actually produced.

    A return value below ``end_frame`` means the frame does not exist as
    far as this decoder is concerned, whatever the container's header
    claims. Returns -1 for a source that yields nothing at all.
    """
    last_index = -1
    for frame_index, _timestamp, _frame in source:
        last_index = frame_index
        if frame_index >= end_frame:
            break
    return last_index


def _require_exact_fields(mapping: dict, fields, what: str, fail) -> None:
    """Every field present, no field unrecognised. Both directions matter:
    a missing field is an incomplete record, and an unknown one is either
    a typo (whose intended field is therefore missing) or a smuggled-in
    extra the loader would silently ignore."""
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise fail(f"{what} is missing required field(s): {', '.join(missing)}")
    unknown = [key for key in mapping if key not in fields]
    if unknown:
        raise fail(
            f"{what} has unknown field(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(fields)}"
        )


def _as_index(value, what: str, fail) -> int:
    """A frame number or id: a genuine non-negative int, never a float that
    happens to be whole and never a bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise fail(
            f"{what} must be an integer frame number, got {value!r} "
            f"({type(value).__name__})"
        )
    if value < 0:
        raise fail(f"{what} must not be negative, got {value}")
    return value


def _as_normalized_point(value, what: str, fail) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise fail(f"{what} must be a [x, y] pair, got {value!r}")
    point = []
    for axis_name, axis in zip(("x", "y"), value):
        if isinstance(axis, bool) or not isinstance(axis, (int, float)):
            raise fail(f"{what} {axis_name} must be a number, got {axis!r}")
        if not (0.0 <= axis <= 1.0):
            raise fail(
                f"{what} is in normalized coordinates: {axis_name}={axis} is "
                f"outside [0, 1]"
            )
        point.append(float(axis))
    return (point[0], point[1])
