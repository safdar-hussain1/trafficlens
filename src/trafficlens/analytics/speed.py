"""Plane-space speed estimation: world-metre displacement over time, in
km/h.

Policy -- inherited from ``trafficlens.core.homography`` and enforced here:
**an uncalibrated camera reports no speed, ever -- never a pixel-derived
guess.** A ``SpeedEstimator`` built with ``plane=None`` (the
``NO_CALIBRATION`` sentinel) returns ``None`` from ``speed_kmh`` for every
track, unconditionally: the check lives in ``speed_kmh`` itself, so no
amount of observed data -- or even hand-injected internal state -- can make
an uncalibrated estimator emit a number. ``observe`` also short-circuits in
that case, so nothing is buffered for a speed that will never be reported.

How the estimate is made
------------------------
Each observed image anchor is projected to world metres through the
``RoadPlane`` at observe time. Per track, a deque holds the accepted
samples ``(timestamp, wx, wy)`` covering the trailing ``window_s`` seconds.
``speed_kmh`` computes the cumulative ARC LENGTH along the sample path
(the running sum of step distances) and fits it against time by
least squares; the fitted slope is metres per second, converted to km/h by
multiplying by 3.6.

Arc length, not straight-line displacement, deliberately: a vehicle that
changes lanes -- or follows a curved road -- travels its whole path, and
its speedometer (and any enforcement measurement) reads the speed along
that path. Net displacement over the window would under-read every
non-straight trajectory. For straight-line motion the two are identical,
so nothing is lost on the common case.

The least-squares slope is the closed-form two-coefficient solve, written
out explicitly (the TypeScript mirror copies this formula): with
timestamps centred on their mean,

    slope = sum(dt_i * (d_i - d_mean)) / sum(dt_i ** 2)

where ``dt_i = t_i - t_mean`` and ``d_i`` is cumulative arc length.
Centring first keeps the sums small and the arithmetic well-conditioned;
no scipy is needed.

Outlier rejection
-----------------
A sample whose plane-space step from the previous ACCEPTED sample exceeds
``SPEED_MAX_STEP_M`` is rejected at observe time -- never buffered -- so a
single wild detector box cannot spike the speed. The comparison is against
the last ACCEPTED sample, not the last seen one: comparing against the
last seen sample would latch onto the outlier itself and then reject every
good sample after it. The threshold is a physical bound (see its comment
in ``trafficlens.core.constants``): no road vehicle can genuinely travel
that far between consecutive frames.

``time_of_flight_kmh`` is the independent gate-pair estimator used by the
Tier-2 cross-check: a known ground distance between two gates divided by
the crossing-time difference. It shares no state or code path with
``SpeedEstimator``, which is exactly what makes it a cross-check.
"""

import math
from collections import deque

from trafficlens.core.constants import (
    SPEED_MAX_STEP_M,
    SPEED_MIN_SAMPLES,
    SPEED_WINDOW_S,
)
from trafficlens.core.geometry import Point
from trafficlens.core.homography import RoadPlane

# m/s -> km/h: 3600 seconds per hour / 1000 metres per kilometre.
_MPS_TO_KMH = 3.6


class SpeedEstimator:
    """Estimates per-track speeds in km/h from image anchors projected onto
    a calibrated road plane.

    With ``plane=None`` (``NO_CALIBRATION``) every ``speed_kmh`` call
    returns ``None``, always -- this class never falls back to a
    pixel-derived guess.
    """

    def __init__(
        self,
        plane: RoadPlane | None,
        fps: float,
        window_s: float = SPEED_WINDOW_S,
        min_samples: int = SPEED_MIN_SAMPLES,
    ) -> None:
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        if window_s <= 0.0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        if min_samples < 2:
            raise ValueError(
                f"min_samples must be at least 2 (a slope needs two "
                f"points), got {min_samples}"
            )
        self._plane = plane
        self._fps = fps
        self._window_s = window_s
        self._min_samples = min_samples
        # track_id -> deque of accepted (timestamp, wx, wy). The maxlen is a
        # hard memory bound derived from fps: one window at the declared
        # frame rate can hold at most window_s * fps + 1 samples (inclusive
        # endpoints), so even a caller whose timestamps never advance --
        # which would defeat the time-based pruning below -- cannot grow a
        # buffer past one window's worth.
        self._maxlen = int(math.ceil(window_s * fps)) + 1
        self._tracks: dict[int, deque[tuple[float, float, float]]] = {}

    def observe(self, track_id: int, anchor: Point, timestamp: float) -> None:
        """Record one image anchor for a track at a timestamp (seconds).

        Uncalibrated (``plane is None``): returns immediately without
        buffering anything -- there is no speed this data could ever
        contribute to. Otherwise the anchor is projected to world metres;
        a sample stepping further than ``SPEED_MAX_STEP_M`` from the last
        accepted sample is rejected outright, and accepted samples older
        than ``window_s`` before this one are dropped.
        """
        if self._plane is None:
            return

        wx, wy = self._plane.to_world(anchor)
        buf = self._tracks.get(track_id)
        if buf is None:
            buf = deque(maxlen=self._maxlen)
            self._tracks[track_id] = buf

        if buf:
            _, last_x, last_y = buf[-1]
            if math.hypot(wx - last_x, wy - last_y) > SPEED_MAX_STEP_M:
                return  # a bad detection, not vehicle motion: never buffered

        buf.append((timestamp, wx, wy))
        while buf and buf[0][0] < timestamp - self._window_s:
            buf.popleft()

    def speed_kmh(self, track_id: int) -> float | None:
        """Return the track's current speed in km/h, or ``None`` when no
        trustworthy number exists: always ``None`` when the estimator is
        uncalibrated, and otherwise when the track has fewer than
        ``min_samples`` accepted samples inside the trailing window (or its
        in-window timestamps do not span any time at all)."""
        if self._plane is None:
            # The refusal is absolute: even pathological internal state
            # cannot make an uncalibrated estimator emit a number.
            return None

        buf = self._tracks.get(track_id)
        if buf is None or len(buf) < self._min_samples:
            return None

        # observe() prunes on append, so the deque already holds only the
        # trailing window; re-filter against the newest timestamp anyway so
        # the window contract holds no matter how the buffer was reached.
        newest = buf[-1][0]
        samples = [s for s in buf if s[0] >= newest - self._window_s]
        n = len(samples)
        if n < self._min_samples:
            return None

        # Cumulative arc length along the sample path, in metres: the
        # distance the vehicle actually travelled, lane changes and curves
        # included -- not the net start-to-end displacement.
        arc = [0.0] * n
        for i in range(1, n):
            _, x0, y0 = samples[i - 1]
            _, x1, y1 = samples[i]
            arc[i] = arc[i - 1] + math.hypot(x1 - x0, y1 - y0)

        # Closed-form least-squares slope of arc length against time, with
        # timestamps centred on their mean (see module docstring).
        t_mean = sum(s[0] for s in samples) / n
        d_mean = sum(arc) / n
        num = 0.0
        den = 0.0
        for (t, _, _), d in zip(samples, arc):
            dt = t - t_mean
            num += dt * (d - d_mean)
            den += dt * dt
        if den == 0.0:
            return None  # all in-window samples share one timestamp

        return (num / den) * _MPS_TO_KMH

    def forget(self, track_id: int) -> None:
        """Drop all state for a track. A later track with the same
        (recycled) ID starts from scratch. Unknown IDs are a no-op."""
        self._tracks.pop(track_id, None)


def time_of_flight_kmh(t_a: float, t_b: float, distance_m: float) -> float:
    """Speed in km/h of a vehicle covering a known ground distance
    (``distance_m``, metres, surveyed between two gates) between timestamps
    ``t_a`` and ``t_b`` (seconds). Independent of any RoadPlane -- this is
    the Tier-2 cross-check on the homography-based estimate.

    Raises ``ValueError`` (fails fast) when ``t_b - t_a`` or ``distance_m``
    is not positive: a zero or negative interval or distance is a caller
    bug, not a slow vehicle.
    """
    dt = t_b - t_a
    if dt <= 0.0:
        raise ValueError(f"t_b must be after t_a, got dt = {dt}")
    if distance_m <= 0.0:
        raise ValueError(f"distance_m must be positive, got {distance_m}")
    return (distance_m / dt) * _MPS_TO_KMH
