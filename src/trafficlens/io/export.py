"""Export formats every downstream surface reads: the crossing-events CSV,
the summary JSON, and the versioned session JSON that both the browser
dashboard and the benchmark replay consume.

Determinism policy, shared by both JSON writers: keys are sorted
(``sort_keys=True``), indentation is fixed at 2, NaN/Infinity are refused
(``allow_nan=False`` -- they are not JSON and every consumer would break
differently), the file ends with exactly one newline, and floats use the
``json`` module's default serialization -- Python's shortest round-trip
``repr`` -- with no rounding, so writing the same dict twice always
produces byte-identical output and reading it back loses nothing.

This module must import neither cv2 nor torch; it only needs the standard
library and the ``CrossingEvent`` dataclass.
"""

import csv
import dataclasses
import json
from pathlib import Path

from trafficlens.core.gate import CrossingEvent

# The session schema version this module writes and validates. Bump only
# with a consumer-coordinated migration; consumers hard-check it.
SESSION_SCHEMA_VERSION = 1

_EVENT_FIELDS = [field.name for field in dataclasses.fields(CrossingEvent)]


# --- events CSV ---------------------------------------------------------------


def write_events_csv(events: list[CrossingEvent], path) -> None:
    """Write one row per CrossingEvent, header exactly the dataclass
    fields, in dataclass field order.

    Cell encoding (mirrored by ``read_events_csv``): floats and ints via
    ``str`` (shortest round-trip repr), ``speed_kmh=None`` as an empty
    cell, booleans as ``true``/``false``.
    """
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_EVENT_FIELDS)
        for event in events:
            row = []
            for name in _EVENT_FIELDS:
                value = getattr(event, name)
                if value is None:
                    row.append("")
                elif isinstance(value, bool):
                    row.append("true" if value else "false")
                else:
                    row.append(str(value))
            writer.writerow(row)


def read_events_csv(path) -> list[CrossingEvent]:
    """Read a file written by ``write_events_csv`` back into
    CrossingEvent instances, exactly (types included)."""
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != _EVENT_FIELDS:
            raise ValueError(
                f"{path}: header {header!r} does not match the "
                f"CrossingEvent fields {_EVENT_FIELDS!r}"
            )
        events = []
        for row in reader:
            values = dict(zip(header, row))
            events.append(
                CrossingEvent(
                    track_id=int(values["track_id"]),
                    class_name=values["class_name"],
                    gate=values["gate"],
                    direction=values["direction"],
                    signed_direction=int(values["signed_direction"]),
                    frame_index=int(values["frame_index"]),
                    timestamp=float(values["timestamp"]),
                    crossing_x=float(values["crossing_x"]),
                    crossing_y=float(values["crossing_y"]),
                    speed_kmh=(
                        float(values["speed_kmh"]) if values["speed_kmh"] else None
                    ),
                    is_violation=values["is_violation"] == "true",
                )
            )
        return events


# --- deterministic JSON -------------------------------------------------------


def _write_json(payload: dict, path) -> None:
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    Path(path).write_text(text + "\n")


def write_summary_json(summary: dict, path) -> None:
    """Write a summary dict as deterministic JSON (see the module
    docstring for the exact policy)."""
    _write_json(summary, path)


# --- session JSON -------------------------------------------------------------


def write_session_json(session: dict, path) -> None:
    """Validate ``session`` against the schema below, then write it as
    deterministic JSON. Refuses (and writes nothing) when invalid."""
    validate_session_dict(session)
    _write_json(session, path)


def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise ValueError(f"session JSON: {where} is missing required key {key!r}")
    return mapping[key]


def _require_number(mapping: dict, key: str, where: str) -> float:
    value = _require(mapping, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"session JSON: {where}[{key!r}] must be a number, got "
            f"{type(value).__name__}"
        )
    return value


def _require_point(mapping: dict, key: str, where: str, arity: int) -> list:
    """A coordinate list (gate start/end, track box) must have exactly
    ``arity`` elements and every element must be a non-bool number --
    the replay draws these, so a "valid" session must never carry
    strings or booleans where pixels belong."""
    value = _require(mapping, key, where)
    if not (
        isinstance(value, list)
        and len(value) == arity
        and all(
            not isinstance(element, bool) and isinstance(element, (int, float))
            for element in value
        )
    ):
        raise ValueError(
            f"session JSON: {where}.{key} must be a list of exactly "
            f"{arity} numbers, got {value!r}"
        )
    return value


def validate_session_dict(session: dict) -> None:
    """Check that a dict is a valid schema-1 session, raising ``ValueError``
    naming the first offending key.

    The session JSON is the one artifact both the browser dashboard's
    replay and the benchmark harness read, so this is the minimal contract
    they share. Required keys, and what each is for:

    - ``schema`` (int, == 1): version stamp; consumers refuse versions
      they do not know rather than misreading them.
    - ``clip`` (str): which footage this session analysed -- the file's
      basename, or the spec string for webcams/streams. The benchmark
      matches it against ground-truth files; the dashboard displays it.
    - ``fps`` (number > 0): frame rate the timestamps were derived at;
      the replay uses it to schedule frames, the benchmark to convert
      frame tolerances to seconds.
    - ``width`` / ``height`` (positive numbers): pixel frame size; every
      pixel coordinate below is relative to it, and the replay canvas is
      scaled from it.
    - ``gates`` (list): the pixel-space counting gates, each with ``name``,
      ``start`` [x, y], ``end`` [x, y], ``label_positive``,
      ``label_negative`` and ``expected_direction`` (a label or None) --
      enough for the replay to draw them and for the benchmark to know
      which direction labels mean what.
    - ``frames`` (list): per-frame replay states, each with
      ``frame_index`` (int), ``timestamp`` (seconds) and ``tracks`` -- a
      list of ``{track_id, class_name, box [x1, y1, x2, y2], speed_kmh
      (number or None)}``. This is what the replay draws frame by frame.
    - ``events`` (list): the crossing events, one dict per event carrying
      every ``CrossingEvent`` field (``speed_kmh`` may be None). The
      benchmark scores these against ground truth; the replay lists them.

    Extra top-level keys (e.g. ``counts``, ``incidents``, ``meta``) are
    allowed and ignored here: producers may add them without breaking
    older consumers, which is the point of the version stamp.
    """
    if not isinstance(session, dict):
        raise ValueError(
            f"session JSON: expected a dict, got {type(session).__name__}"
        )

    schema = _require(session, "schema", "session")
    # An exact int, and only an int: 1.0 is reachable from real JSON and
    # True == 1 in Python, but neither is the version stamp consumers wrote.
    if (
        isinstance(schema, bool)
        or not isinstance(schema, int)
        or schema != SESSION_SCHEMA_VERSION
    ):
        raise ValueError(
            f"session JSON: schema version {schema!r} is not the supported "
            f"integer version {SESSION_SCHEMA_VERSION}"
        )

    clip = _require(session, "clip", "session")
    if not isinstance(clip, str) or not clip:
        raise ValueError("session JSON: clip must be a non-empty string")

    for key in ("fps", "width", "height"):
        if _require_number(session, key, "session") <= 0:
            raise ValueError(f"session JSON: {key} must be positive")

    gates = _require(session, "gates", "session")
    if not isinstance(gates, list):
        raise ValueError("session JSON: gates must be a list")
    for i, gate in enumerate(gates):
        where = f"gates[{i}]"
        if not isinstance(gate, dict):
            raise ValueError(f"session JSON: {where} must be a dict")
        name = _require(gate, "name", where)
        if not isinstance(name, str) or not name:
            raise ValueError(f"session JSON: {where}.name must be a non-empty string")
        for key in ("start", "end"):
            _require_point(gate, key, where, 2)
        _require(gate, "label_positive", where)
        _require(gate, "label_negative", where)
        _require(gate, "expected_direction", where)  # may be None, must exist

    frames = _require(session, "frames", "session")
    if not isinstance(frames, list):
        raise ValueError("session JSON: frames must be a list")
    for i, frame in enumerate(frames):
        where = f"frames[{i}]"
        if not isinstance(frame, dict):
            raise ValueError(f"session JSON: {where} must be a dict")
        _require_number(frame, "frame_index", where)
        _require_number(frame, "timestamp", where)
        tracks = _require(frame, "tracks", where)
        if not isinstance(tracks, list):
            raise ValueError(f"session JSON: {where}.tracks must be a list")
        for j, track in enumerate(tracks):
            track_where = f"{where}.tracks[{j}]"
            if not isinstance(track, dict):
                raise ValueError(f"session JSON: {track_where} must be a dict")
            _require_number(track, "track_id", track_where)
            _require(track, "class_name", track_where)
            _require_point(track, "box", track_where, 4)
            _require(track, "speed_kmh", track_where)  # number or None, must exist

    events = _require(session, "events", "session")
    if not isinstance(events, list):
        raise ValueError("session JSON: events must be a list")
    for i, event in enumerate(events):
        where = f"events[{i}]"
        if not isinstance(event, dict):
            raise ValueError(f"session JSON: {where} must be a dict")
        for name in _EVENT_FIELDS:
            _require(event, name, where)
