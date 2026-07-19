"""The analysis worker: one thread that owns one video stream.

The web layer never touches OpenCV or the model directly — it starts an
:class:`AnalysisSession`, reads its latest JPEG + stats snapshots, and
applies live config changes through the thread-safe ``update_*`` methods.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2

from trafficlens.annotate import draw_frame
from trafficlens.config import AppConfig, CalibrationConfig, GateConfig
from trafficlens.counting import CrossingEvent, Gate, GateCounter
from trafficlens.incidents import Incident
from trafficlens.pipeline import Pipeline
from trafficlens.speed import PlaneCalibration, SpeedEstimator
from trafficlens.video import VideoSource

_JPEG_QUALITY = 82
_MAX_EVENTS_KEPT = 500
_MAX_VIOLATION_SNAPSHOTS = 24
_MAX_INCIDENTS_KEPT = 200


@dataclass
class ViolationSnapshot:
    seq: int
    event: CrossingEvent
    jpeg: bytes


@dataclass
class IncidentRecord:
    seq: int
    incident: Incident
    jpeg: bytes | None


class AnalysisSession(threading.Thread):
    """Runs the pipeline over a source until stopped (or the file ends)."""

    def __init__(self, config: AppConfig, loop_file: bool = True):
        super().__init__(daemon=True, name="trafficlens-analysis")
        self.config = config
        self.loop_file = loop_file
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.stats: dict = {}
        self.error: str | None = None
        self.finished = False
        self.events: deque[tuple[int, CrossingEvent]] = deque(maxlen=_MAX_EVENTS_KEPT)
        self.violations: deque[ViolationSnapshot] = deque(maxlen=_MAX_VIOLATION_SNAPSHOTS)
        self.incidents: deque[IncidentRecord] = deque(maxlen=_MAX_INCIDENTS_KEPT)
        self._event_seq = 0
        self._incident_seq = 0
        self.pipeline: Pipeline | None = None
        self.source_info: dict = {}
        self._ready = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def wait_ready(self, timeout: float = 60.0) -> None:
        """Block until the model is loaded and the first frame processed."""
        if not self._ready.wait(timeout):
            raise TimeoutError("analysis did not produce a first frame in time")
        if self.error:
            raise RuntimeError(self.error)

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:  # noqa: C901 - one linear worker loop
        try:
            src = VideoSource(self.config.source)
        except Exception as exc:
            self.error = str(exc)
            self._ready.set()
            return
        try:
            info = src.info
            self.source_info = {
                "source": self.config.source,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "frames": info.frame_count,
                "live": info.is_live,
            }
            pipeline = Pipeline(self.config, info.width, info.height, fps=info.fps)
            self.pipeline = pipeline
            frame_interval = 1.0 / info.fps if (not info.is_live and info.fps > 0) else 0.0
            ema_fps = 0.0
            while not self._halt.is_set():
                tick = time.perf_counter()
                got_frame = False
                for _, frame in src.frames():
                    got_frame = True
                    with self._lock:
                        result = pipeline.process(frame)
                        annotated = draw_frame(
                            frame, result, pipeline.counters,
                            speed_unit=pipeline.config.speed.unit,
                            speed_limit=pipeline.config.speed.speed_limit,
                        )
                        for event in result.events:
                            self._event_seq += 1
                            self.events.append((self._event_seq, event))
                            if event.is_violation:
                                self._capture_violation(frame, result, event)
                        for incident in result.incidents:
                            self._incident_seq += 1
                            self.incidents.append(IncidentRecord(
                                seq=self._incident_seq,
                                incident=incident,
                                jpeg=self._crop_track(frame, result, incident.track_id),
                            ))
                        elapsed = time.perf_counter() - tick
                        ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / max(elapsed, 1e-6)) if ema_fps else 1.0 / max(elapsed, 1e-6)
                        ok, buf = cv2.imencode(".jpg", annotated,
                                               [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
                        if ok:
                            self.latest_jpeg = buf.tobytes()
                        self.stats = {
                            "frame": result.frame_index,
                            "timestamp": round(result.timestamp, 2),
                            "process_ms": round(result.process_ms, 1),
                            "fps": round(ema_fps, 1),
                            "live_tracks": len(result.tracks),
                            "summary": pipeline.summary(),
                        }
                    self._ready.set()
                    if self._halt.is_set():
                        break
                    # Pace file playback at native speed so counts/speeds on
                    # screen match real time; live sources set the pace.
                    if frame_interval:
                        sleep_for = frame_interval - (time.perf_counter() - tick)
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                    tick = time.perf_counter()
                if self._halt.is_set() or info.is_live or not self.loop_file or not got_frame:
                    break
                # Loop the file: rewind cleanly. Track state must be cleared,
                # otherwise the jump from last frame to first frame would be
                # seen as physical movement and could fire phantom crossings.
                src.release()
                src = VideoSource(self.config.source)
                with self._lock:
                    pipeline.reset_stream_state()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            src.release()
            self.finished = True
            self._ready.set()

    def _crop_track(self, frame, result, track_id: int) -> bytes | None:
        """JPEG crop of one track's box on this frame, if it is visible."""
        for tv in result.tracks:
            if tv.track_id == track_id:
                x1, y1, x2, y2 = (max(0, int(v)) for v in tv.box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    return None
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                return buf.tobytes() if ok else None
        return None

    def _capture_violation(self, frame, result, event: CrossingEvent) -> None:
        jpeg = self._crop_track(frame, result, event.track_id)
        if jpeg is not None:
            self.violations.append(
                ViolationSnapshot(seq=self._event_seq, event=event, jpeg=jpeg)
            )

    # -- live re-configuration --------------------------------------------

    def update_gates(self, gates: list[GateConfig]) -> None:
        """Replace counting gates, preserving tallies of same-named gates."""
        if self.pipeline is None:
            raise RuntimeError("session not ready")
        with self._lock:
            w, h = self.pipeline.frame_size
            existing = {c.gate.name: c for c in self.pipeline.counters}
            new_counters = []
            for g in gates:
                gate = Gate(
                    name=g.name,
                    start=(g.start[0] * w, g.start[1] * h),
                    end=(g.end[0] * w, g.end[1] * h),
                    label_positive=g.label_positive,
                    label_negative=g.label_negative,
                )
                old = existing.get(g.name)
                if old is not None:
                    old.gate = gate  # keep counted-set and totals, move the line
                    new_counters.append(old)
                else:
                    new_counters.append(GateCounter(gate))
            self.pipeline.counters = new_counters
            self.pipeline.config = self.pipeline.config.model_copy(update={"gates": gates})

    def update_speed(self, speed_limit: float | None, unit: str | None) -> None:
        if self.pipeline is None:
            raise RuntimeError("session not ready")
        with self._lock:
            speed = self.pipeline.config.speed
            updates = {"speed_limit": speed_limit}
            if unit:
                updates["unit"] = unit
            self.pipeline.config = self.pipeline.config.model_copy(
                update={"speed": speed.model_copy(update=updates)}
            )
            if self.pipeline.speed_estimator and unit:
                self.pipeline.speed_estimator.unit = unit

    def update_calibration(self, calibration: CalibrationConfig | None) -> None:
        if self.pipeline is None:
            raise RuntimeError("session not ready")
        with self._lock:
            w, h = self.pipeline.frame_size
            if calibration is None:
                self.pipeline.speed_estimator = None
            else:
                speed = self.pipeline.config.speed
                self.pipeline.speed_estimator = SpeedEstimator(
                    calibration=PlaneCalibration(calibration, w, h),
                    window_seconds=speed.window_seconds,
                    smoothing=speed.smoothing,
                    unit=speed.unit,
                    min_travel_m=speed.min_travel_m,
                )
            self.pipeline.config = self.pipeline.config.model_copy(
                update={"calibration": calibration}
            )

    def events_after(self, seq: int) -> list[dict]:
        out = []
        for s, e in self.events:
            if s > seq:
                out.append({
                    "seq": s, "frame": e.frame_index, "t": round(e.timestamp, 2),
                    "track": e.track_id, "class": e.class_name, "gate": e.gate,
                    "direction": e.direction,
                    "speed": round(e.speed, 1) if e.speed is not None else None,
                    "violation": e.is_violation,
                })
        return out
