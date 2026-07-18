"""Speed estimation from tracked image positions.

The core problem: pixels are not metres. A car near the camera moves
many pixels per frame; the same car far away moves few. Any speed
computed directly from pixel displacement is wrong everywhere except
one depth. The fix is a **plane homography**: four (or more) image
points with known road-plane coordinates in metres give a 3x3 matrix
that maps every image point on the road to world metres. Displacement
in world space over time is real speed.

Estimates are computed over a sliding time window (not frame-to-frame,
which amplifies detector jitter) and smoothed with an EMA.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from trafficlens.config import CalibrationConfig
from trafficlens.geometry import Point

MPS_TO_KMH = 3.6
MPS_TO_MPH = 2.2369362920544


class PlaneCalibration:
    """Image-plane -> road-plane (metres) transform."""

    def __init__(self, config: CalibrationConfig, frame_width: int, frame_height: int):
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"invalid frame size {frame_width}x{frame_height}")
        self._mode = config.mode
        if config.mode == "homography":
            img = np.array(
                [(x * frame_width, y * frame_height) for x, y in config.image_points],
                dtype=np.float64,
            )
            world = np.array(config.world_points, dtype=np.float64)
            matrix, _ = cv2.findHomography(img, world, method=0)
            if matrix is None:
                raise ValueError("degenerate calibration points — homography could not be computed")
            self._matrix = matrix
            self._scale = None
        else:
            # meters_per_pixel is specified at reference_width; scale it to
            # the actual stream resolution.
            self._matrix = None
            self._scale = config.meters_per_pixel * (config.reference_width / frame_width)

    def to_world(self, p: Point) -> Point:
        """Map an image pixel to road-plane metres."""
        if self._matrix is not None:
            v = self._matrix @ np.array([p[0], p[1], 1.0])
            if abs(v[2]) < 1e-12:
                raise ValueError(f"point {p} maps to infinity — outside the calibrated plane")
            return (v[0] / v[2], v[1] / v[2])
        return (p[0] * self._scale, p[1] * self._scale)


@dataclass
class _TrackSpeedState:
    history: deque  # of (timestamp, world_x, world_y)
    ema: float | None = None


@dataclass
class SpeedEstimator:
    """Per-track speed over a sliding window, EMA-smoothed.

    ``update()`` is fed every frame; it returns the current smoothed
    speed in the configured unit, or ``None`` until the track has
    travelled ``min_travel_m`` within the window (suppresses phantom
    speeds from bounding-box jitter on stationary objects).
    """

    calibration: PlaneCalibration
    window_seconds: float = 0.5
    smoothing: float = 0.35
    unit: str = "kmh"
    min_travel_m: float = 0.4
    _tracks: dict[int, _TrackSpeedState] = field(default_factory=dict)

    def update(self, track_id: int, anchor: Point, timestamp: float) -> float | None:
        state = self._tracks.get(track_id)
        if state is None:
            state = _TrackSpeedState(history=deque())
            self._tracks[track_id] = state

        wx, wy = self.calibration.to_world(anchor)
        state.history.append((timestamp, wx, wy))
        cutoff = timestamp - self.window_seconds
        while state.history and state.history[0][0] < cutoff:
            state.history.popleft()

        if len(state.history) < 2:
            return self._current(state)

        t0, x0, y0 = state.history[0]
        t1, x1, y1 = state.history[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return self._current(state)
        dist = float(np.hypot(x1 - x0, y1 - y0))
        if dist < self.min_travel_m:
            # Not moving meaningfully; decay toward zero rather than
            # reporting jitter as speed.
            state.ema = (1 - self.smoothing) * (state.ema or 0.0)
            return self._current(state)

        mps = dist / dt
        if state.ema is None:
            state.ema = mps
        else:
            state.ema = self.smoothing * mps + (1 - self.smoothing) * state.ema
        return self._current(state)

    def _current(self, state: _TrackSpeedState) -> float | None:
        if state.ema is None:
            return None
        factor = MPS_TO_KMH if self.unit == "kmh" else MPS_TO_MPH
        return state.ema * factor

    def forget(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def reset(self) -> None:
        """Drop all per-track history (tracker restarted, IDs may repeat)."""
        self._tracks.clear()
