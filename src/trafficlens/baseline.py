"""Naive counting baseline, kept for benchmarking and tests.

This is the approach most tutorials teach: count a track when its
centre point falls inside a thin horizontal band around the counting
line. It looks right in demos and is quantifiably wrong two ways:

* **misses** — an object moving faster than the band is tall jumps
  clean over it between frames (a 16 px band vs highway traffic that
  moves 100+ px/frame at 30 fps);
* **phantoms** — an object that enters the band without crossing the
  line (lane change along the line, jittering box) is counted anyway.

The benchmark suite runs this side by side with the segment-intersection
:class:`~trafficlens.counting.GateCounter` on identical track streams so
the failure modes show up as numbers, not opinions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trafficlens.geometry import Point


@dataclass
class NaiveBandCounter:
    """Counts a track once when its point lands inside the band.

    ``band_up``/``band_down`` mirror the classic tutorial band of
    ``line_y - 15 < y < line_y + 1``.
    """

    x_min: float
    x_max: float
    line_y: float
    band_up: float = 15.0
    band_down: float = 1.0
    counted: set[int] = field(default_factory=set)

    def update(self, track_id: int, point: Point) -> bool:
        """Returns True when this update counts the track."""
        x, y = point
        inside = (
            self.x_min < x < self.x_max
            and self.line_y - self.band_up < y < self.line_y + self.band_down
        )
        if inside and track_id not in self.counted:
            self.counted.add(track_id)
            return True
        return False

    @property
    def total(self) -> int:
        return len(self.counted)
