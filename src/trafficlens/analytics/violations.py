"""Speed-limit violation policy and evidence snapshots.

The limit comparison itself is NOT made here: ``check`` delegates to
``trafficlens.core.gate.is_over_limit``, the single place that comparison
lives, inheriting its two policies -- a strict ``>`` (a speed exactly at
the limit is not a violation) and ``None``-safety (an unknown speed, or no
limit at all, is never "over"). Limits are user-set per deployment; the
product never assumes a posted limit, so ``ViolationPolicy(limit_kmh=None)``
is the ordinary unconfigured state and reports no violations, ever.

Snapshot handling is split so path logic stays testable without OpenCV:

- ``snapshot_path`` is pure computation -- a deterministic filename built
  from the event's gate name (sanitised for filesystem safety), track ID,
  and frame index. No I/O of any kind. Distinct crossings get distinct
  paths provided sanitised gate names are unique per deployment: two gate
  display names that differ only in punctuation (e.g. "M40 J3" and
  "M40_J3") sanitise to the same token, so gates must not be named that
  closely.
- ``save_snapshot`` does the I/O: it imports ``cv2`` lazily inside the
  function -- the ONLY place this module touches OpenCV, so the module
  imports cleanly without it -- annotates a COPY of the frame minimally
  (crossing marker plus a speed/limit caption) and writes a JPEG.

This module imports nothing beyond the standard library and
``trafficlens.core`` at module level.
"""

import re
from pathlib import Path

from trafficlens.core.gate import CrossingEvent, is_over_limit

# Characters allowed to survive gate-name sanitisation unchanged; every
# run of anything else (spaces, slashes, quotes, ...) collapses to one
# underscore so a display name like "M40 J3 / north" cannot escape the
# output directory or produce an awkward filename.
_UNSAFE_RUN = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitise(name: str) -> str:
    cleaned = _UNSAFE_RUN.sub("_", name).strip("_")
    # A name that sanitises to nothing (or to bare dots, which would make
    # the stem read as a hidden/relative path fragment) falls back to a
    # fixed token; the track and frame components keep the path unique.
    return cleaned if cleaned.strip(".") else "gate"


class ViolationPolicy:
    """Decides whether a gate crossing violates the user-set speed limit
    and where its evidence snapshot lives."""

    def __init__(self, limit_kmh: float | None) -> None:
        self.limit_kmh = limit_kmh

    def check(self, event: CrossingEvent) -> bool:
        """True only when the event's known speed strictly exceeds a
        configured limit. Delegates to ``is_over_limit`` -- see the module
        docstring for the inherited policies."""
        return is_over_limit(event.speed_kmh, self.limit_kmh)

    def snapshot_path(self, event: CrossingEvent, out_dir: Path) -> Path:
        """Deterministic snapshot filename for this event under
        ``out_dir``. Pure computation: no directories are created and no
        filesystem is touched. (gate, track_id, frame_index) keeps names
        distinct across distinct crossings, assuming sanitised gate names
        are unique per deployment (see the module docstring)."""
        gate = _sanitise(event.gate)
        return out_dir / (
            f"violation_{gate}_track{event.track_id}_frame{event.frame_index}.jpg"
        )

    def save_snapshot(self, frame, event: CrossingEvent, out_dir: Path) -> Path:
        """Write the annotated evidence JPEG for this event and return its
        path (the same path ``snapshot_path`` computes). The caller's
        frame is never mutated; annotation happens on a copy."""
        import cv2  # lazy: the only OpenCV touchpoint in this module

        path = self.snapshot_path(event, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        annotated = frame.copy()
        # Minimal annotation: a marker at the crossing point and one
        # caption line naming the measured speed against the limit.
        center = (int(round(event.crossing_x)), int(round(event.crossing_y)))
        cv2.circle(annotated, center, 8, (0, 0, 255), 2)
        speed = "n/a" if event.speed_kmh is None else f"{event.speed_kmh:.1f}"
        limit = "n/a" if self.limit_kmh is None else f"{self.limit_kmh:.0f}"
        caption = (
            f"track {event.track_id} @ {event.gate}: "
            f"{speed} km/h (limit {limit})"
        )
        cv2.putText(
            annotated,
            caption,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(path), annotated):
            raise OSError(f"cv2.imwrite failed for {path}")
        return path
