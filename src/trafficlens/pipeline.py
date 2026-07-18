"""The frame-processing pipeline: detect -> track -> speed -> count.

One :class:`Pipeline` instance owns all per-stream state (tracker
identity, per-track anchors and speeds, per-gate tallies, the event
log). It is deliberately synchronous and single-stream; the web layer
runs it on a worker thread, the CLI drives it directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from trafficlens.config import AppConfig
from trafficlens.counting import CrossingEvent, Gate, GateCounter
from trafficlens.detection import Detector, Observation
from trafficlens.geometry import Point
from trafficlens.speed import PlaneCalibration, SpeedEstimator

# Frames a track may vanish for before its state is retired. ByteTrack
# keeps IDs alive over short occlusions; this only reaps truly gone tracks.
_STALE_AFTER_FRAMES = 60


@dataclass
class TrackView:
    """What the pipeline knows about one live track right now."""

    track_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    anchor: Point
    speed: float | None
    trail: list[Point]


@dataclass
class FrameResult:
    """Everything computed for one frame."""

    frame_index: int
    timestamp: float
    tracks: list[TrackView]
    events: list[CrossingEvent]
    counts: dict[str, dict[str, dict[str, int]]]  # gate -> class -> direction -> n
    process_ms: float


@dataclass
class _TrackState:
    class_name: str
    prev_anchor: Point | None = None
    trail: list[Point] = field(default_factory=list)
    last_seen: int = 0


class Pipeline:
    """Orchestrates detection, tracking, speed estimation and counting."""

    def __init__(self, config: AppConfig, frame_width: int, frame_height: int, fps: float = 0.0):
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"invalid frame size {frame_width}x{frame_height}")
        self.config = config
        self.frame_size = (frame_width, frame_height)
        self.fps = fps
        self.detector = Detector(config.detector)
        self.counters: list[GateCounter] = [
            GateCounter(
                Gate(
                    name=g.name,
                    start=(g.start[0] * frame_width, g.start[1] * frame_height),
                    end=(g.end[0] * frame_width, g.end[1] * frame_height),
                    label_positive=g.label_positive,
                    label_negative=g.label_negative,
                )
            )
            for g in config.gates
        ]
        self.speed_estimator: SpeedEstimator | None = None
        if config.calibration is not None:
            self.speed_estimator = SpeedEstimator(
                calibration=PlaneCalibration(config.calibration, frame_width, frame_height),
                window_seconds=config.speed.window_seconds,
                smoothing=config.speed.smoothing,
                unit=config.speed.unit,
                min_travel_m=config.speed.min_travel_m,
            )
        self.events: list[CrossingEvent] = []
        self.speed_samples: dict[str, list[float]] = {}
        self._tracks: dict[int, _TrackState] = {}
        self._frame_index = 0
        self._started_at: float | None = None

    def process(self, frame: np.ndarray, timestamp: float | None = None) -> FrameResult:
        """Process one frame. ``timestamp`` in seconds; derived from fps or wall clock if omitted."""
        t0 = time.perf_counter()
        idx = self._frame_index
        self._frame_index += 1
        if timestamp is None:
            if self.fps > 0:
                timestamp = idx / self.fps
            else:
                if self._started_at is None:
                    self._started_at = time.monotonic()
                timestamp = time.monotonic() - self._started_at

        observations = self.detector.track(frame)
        views: list[TrackView] = []
        events: list[CrossingEvent] = []

        for obs in observations:
            state = self._tracks.get(obs.track_id)
            if state is None:
                state = _TrackState(class_name=obs.class_name)
                self._tracks[obs.track_id] = state
            state.last_seen = idx
            anchor = obs.anchor

            speed = None
            if self.speed_estimator is not None:
                speed = self.speed_estimator.update(obs.track_id, anchor, timestamp)

            for counter in self.counters:
                event = counter.update(
                    track_id=obs.track_id,
                    class_name=obs.class_name,
                    prev=state.prev_anchor,
                    curr=anchor,
                    frame_index=idx,
                    timestamp=timestamp,
                    speed=speed,
                    speed_limit=self.config.speed.speed_limit,
                )
                if event is not None:
                    events.append(event)
                    if event.speed is not None:
                        self.speed_samples.setdefault(event.class_name, []).append(event.speed)

            state.prev_anchor = anchor
            state.trail.append(anchor)
            if len(state.trail) > 32:
                del state.trail[0]
            views.append(
                TrackView(
                    track_id=obs.track_id,
                    class_name=obs.class_name,
                    confidence=obs.confidence,
                    box=obs.box,
                    anchor=anchor,
                    speed=speed,
                    trail=list(state.trail),
                )
            )

        self.events.extend(events)
        self._reap_stale(idx)
        return FrameResult(
            frame_index=idx,
            timestamp=timestamp,
            tracks=views,
            events=events,
            counts=self.counts(),
            process_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _reap_stale(self, current_index: int) -> None:
        stale = [
            tid for tid, st in self._tracks.items()
            if current_index - st.last_seen > _STALE_AFTER_FRAMES
        ]
        for tid in stale:
            del self._tracks[tid]
            if self.speed_estimator is not None:
                self.speed_estimator.forget(tid)
            # Deliberately NOT forgetting in counters: a track that
            # crossed and left must stay counted even if the ID is reaped.

    def reset_stream_state(self) -> None:
        """Forget everything tied to track identity, keep tallies and events.

        Call when the input stream restarts (looped file, camera
        reconnect): the tracker will hand out fresh IDs that may collide
        with retired ones, and any per-track state keyed on the old IDs —
        crossing memory, speed history, trails — would corrupt the new
        tracks. Totals and the event log survive; identity does not.
        """
        self._tracks.clear()
        self.detector.reset_tracker()
        for counter in self.counters:
            counter.reset_tracks()
        if self.speed_estimator is not None:
            self.speed_estimator.reset()

    def counts(self) -> dict[str, dict[str, dict[str, int]]]:
        return {c.gate.name: {k: dict(v) for k, v in c.totals.items()} for c in self.counters}

    def summary(self) -> dict:
        """Aggregate stats for exports and the live dashboard."""
        per_gate = {}
        for counter in self.counters:
            per_gate[counter.gate.name] = {
                "total": counter.total,
                "by_direction": counter.total_by_direction(),
                "by_class": {k: dict(v) for k, v in counter.totals.items()},
            }
        speed_stats = {}
        for cls, samples in self.speed_samples.items():
            arr = np.array(samples)
            speed_stats[cls] = {
                "n": len(samples),
                "mean": round(float(arr.mean()), 1),
                "median": round(float(np.median(arr)), 1),
                "p85": round(float(np.percentile(arr, 85)), 1),
                "max": round(float(arr.max()), 1),
            }
        return {
            "frames": self._frame_index,
            "gates": per_gate,
            "events": len(self.events),
            "violations": sum(1 for e in self.events if e.is_violation),
            "speed_unit": self.config.speed.unit,
            "speed_by_class": speed_stats,
        }
