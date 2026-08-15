"""Frame annotation: tracks, gates, counts, speeds and incidents drawn onto
a copy of the footage.

OpenCV only -- no matplotlib, no PIL, no font files -- so an annotated video
can be written on any machine that can decode the source in the first place.
``draw_frame`` never mutates the frame it is given; it draws on a copy and
returns it, because the caller (the pipeline) hands it the same array it may
still be exporting or snapshotting from.

Colour policy: classes get stable, distinct colours from a fixed table (a
class not in the table falls back to one derived from its name, so an
unusual class is still consistent frame to frame rather than flickering),
and anything flagged as an incident is overdrawn in red so it reads as the
exception, not as another class.

Speed policy, inherited from ``trafficlens.analytics.speed``: an
uncalibrated session has no speed and says so. ``speed_label(None)``
returns ``NO_SPEED_LABEL`` -- a visible marker -- rather than an empty
string, because a blank where a number belongs reads as "not measured yet"
when the truth is "never measurable here".
"""

from __future__ import annotations

import cv2
import numpy as np

# What a track's label shows when no speed exists for it. Deliberately not
# blank: see the module docstring.
NO_SPEED_LABEL = "no speed"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_INCIDENT_BGR = (0, 0, 255)
_GATE_BGR = (0, 220, 220)
_TEXT_BGR = (255, 255, 255)

# BGR, per class. Chosen to stay distinguishable against road-grey footage
# and against each other.
_CLASS_BGR = {
    "car": (80, 220, 80),
    "truck": (220, 160, 60),
    "bus": (200, 100, 220),
    "motorcycle": (60, 200, 240),
    "bicycle": (240, 200, 100),
    "person": (120, 120, 250),
}


def class_color(class_name: str) -> tuple[int, int, int]:
    """A stable BGR colour for a class. Known classes come from the table
    above; anything else is hashed to a fixed colour so it at least stays
    the same colour for the whole session."""
    if class_name in _CLASS_BGR:
        return _CLASS_BGR[class_name]
    digest = sum(ord(ch) * (i + 1) for i, ch in enumerate(class_name))
    return (60 + digest * 37 % 180, 60 + digest * 61 % 180, 60 + digest * 97 % 180)


def speed_label(speed_kmh: float | None) -> str:
    """The text a track's speed is drawn with: one decimal place, or the
    explicit ``NO_SPEED_LABEL`` when there is no speed to show."""
    if speed_kmh is None:
        return NO_SPEED_LABEL
    return f"{speed_kmh:.1f} km/h"


def _text(image, message: str, origin, color, scale: float = 0.5) -> None:
    """Draw text with a dark outline underneath, so it stays readable over
    both bright tarmac and dark shadow without a filled label box."""
    x, y = int(origin[0]), int(origin[1])
    cv2.putText(image, message, (x, y), _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, message, (x, y), _FONT, scale, color, 1, cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray,
    tracks,
    gates,
    counters,
    speeds,
    incidents=None,
) -> np.ndarray:
    """Return an annotated copy of ``frame``.

    - ``tracks``: the ``Track``s to draw (the pipeline passes exactly what
      ``Tracker.update`` returned this frame).
    - ``gates``: the pixel-space ``Gate``s, drawn as directed segments with
      an arrow marking ``start -> end``.
    - ``counters``: mapping gate name -> ``GateCounter``, used for the
      per-gate running totals. A gate with no counter simply shows no
      totals.
    - ``speeds``: mapping track id -> km/h or ``None``.
    - ``incidents``: optional ``Incident``s; every track named by one is
      redrawn in red and captioned with the incident kind.
    """
    canvas = frame.copy()
    flagged = {
        incident.track_id: incident.kind for incident in (incidents or ())
    }

    for gate in gates:
        _draw_gate(canvas, gate, counters)

    for track in tracks:
        _draw_track(canvas, track, speeds, flagged)

    return canvas


def _draw_gate(canvas: np.ndarray, gate, counters) -> None:
    start = (int(round(gate.start[0])), int(round(gate.start[1])))
    end = (int(round(gate.end[0])), int(round(gate.end[1])))
    # An arrowed line, so the direction that fixes the +/- labels is
    # visible rather than something the reader has to infer from the config.
    cv2.arrowedLine(canvas, start, end, _GATE_BGR, 2, cv2.LINE_AA, tipLength=0.04)

    counter = _counter_for(counters, gate.name)
    lines = [gate.name]
    if counter is not None:
        for class_name in sorted(counter.totals):
            directions = counter.totals[class_name]
            parts = ", ".join(
                f"{direction} {directions[direction]}"
                for direction in sorted(directions)
            )
            lines.append(f"{class_name}: {parts}")

    anchor_x = min(start[0], end[0])
    anchor_y = min(start[1], end[1]) - 8
    for offset, line in enumerate(reversed(lines)):
        _text(canvas, line, (anchor_x, anchor_y - offset * 16), _GATE_BGR)


def _counter_for(counters, name: str):
    """Look up a gate's counter, accepting either a name -> counter mapping
    or a plain iterable of counters (each of which knows its own gate)."""
    if counters is None:
        return None
    if hasattr(counters, "get"):
        return counters.get(name)
    for counter in counters:
        if counter.gate.name == name:
            return counter
    return None


def _draw_track(canvas: np.ndarray, track, speeds, flagged) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in track.box)
    incident_kind = flagged.get(track.track_id)
    color = _INCIDENT_BGR if incident_kind else class_color(track.class_name)
    thickness = 3 if incident_kind else 2
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    speed = speeds.get(track.track_id) if speeds is not None else None
    _text(
        canvas,
        f"#{track.track_id} {track.class_name}  {speed_label(speed)}",
        (x1, max(12, y1 - 6)),
        color,
    )
    if incident_kind:
        _text(canvas, incident_kind.upper(), (x1, y2 + 16), _INCIDENT_BGR)

    # The anchor is the point every gate and speed decision is made at, so
    # it is drawn: a disagreement between what the box looks like and where
    # the crossing fired is otherwise invisible.
    anchor = track.anchor
    cv2.circle(
        canvas,
        (int(round(anchor[0])), int(round(anchor[1]))),
        3,
        _TEXT_BGR,
        -1,
        cv2.LINE_AA,
    )
