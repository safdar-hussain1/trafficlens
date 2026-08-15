"""The end-to-end analysis pipeline: one video source in, one
``SessionResult`` out.

This is the module where every component built so far meets. Per frame:
detect, track, then analytics (speed, gate counting, incidents,
violations). Each stage is separately timed; the per-track lifecycle
below is the correctness core.

Injected detector
-----------------
``run_session`` takes a ``Detector`` -- it never constructs one. This
module therefore imports neither torch, ultralytics nor onnxruntime, and
never will: the concrete adapter is chosen and built by the caller
(``trafficlens.cli.build_detector``), which is also what lets the whole
pipeline be tested end to end against a scripted fake detector with no
model weights present. See
``tests/test_pipeline.py::test_importing_the_pipeline_never_pulls_in_a_detector_backend``.

Per-track lifecycle
-------------------
``Tracker.update`` returns ONLY confirmed tracks that a real detection
updated this frame -- coasting tracks stay internal. Two consequences
shape everything here:

1. **Previous anchors are ours to keep.** ``GateCounter.update`` needs the
   swept segment ``prev -> curr``, and the tracker does not hand one over.
   ``run_session`` therefore stores the last anchor it saw for each track
   id. A track appearing for the first time has NO previous anchor and so
   cannot cross: it is recorded and skipped, never paired with a
   substitute origin. Seeding a missing previous anchor with anything at
   all (the frame origin, the box centre) would fabricate a swept segment
   spanning most of the frame, and any gate it happened to cut would count
   a crossing that never happened.

2. **Deaths are invisible, so tracks are reaped on a clock.** A track that
   stops being returned might be dead, or might merely be coasting through
   an occlusion and about to return. The tracker never announces a death.
   ``TrackReaper`` therefore keeps the frame index each track was last
   returned on and reaps ids unseen for LONGER than the tracker's
   ``max_age``: a confirmed track survives while ``time_since_update <=
   max_age`` and can re-associate at exactly ``max_age``, so a gap of
   ``max_age + 1`` frames is the first moment the track is provably gone.
   Reaping calls ``forget(track_id)`` on every gate counter, the speed
   estimator and the incident detector, and drops our own previous-anchor
   entry. Without it every one of those keeps a permanent per-track
   record and a long run leaks memory in five places at once. The tracker
   allocates ids monotonically and never recycles them, so forgetting is
   pure memory reclamation here -- it can never re-count a vehicle.

Timing
------
``timings`` measures ONE stage per measurement. ``detect``, ``track`` and
``analytics`` each get their own ``perf_counter`` bracket around exactly
their own work; nothing is measured once and attributed to several
stages, and no stage's number contains another's. ``frame`` is the
separately measured total of the per-frame body, recorded so the three
stage means can be checked to genuinely partition it (see
``tests/test_pipeline.py::test_stage_means_sum_to_the_measured_total_frame_time``,
which exists to catch a summed-timing column). Work that is not analysis
-- writing violation snapshots, building replay records, encoding the
annotated video, calling the progress callback -- happens after the
``frame`` bracket closes, so it is charged to no stage and does not
distort the total either.

Statistics are computed from the full per-frame sample list, so ``p95``
is an exact nearest-rank percentile rather than a running approximation.
That costs one float per stage per frame (a few megabytes for a
feature-length run), which is the honest price of an exact tail number.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

from trafficlens.analytics.incidents import Incident, IncidentDetector
from trafficlens.analytics.speed import SpeedEstimator
from trafficlens.analytics.violations import ViolationPolicy
from trafficlens.config import AppConfig
from trafficlens.core.gate import CrossingEvent, Gate, GateCounter
from trafficlens.detect.base import Detector
from trafficlens.io.export import SESSION_SCHEMA_VERSION
from trafficlens.io.video import VideoSource, classify_spec
from trafficlens.track.tracker import Tracker

# The three separately measured stages of one frame, in execution order.
STAGES: tuple[str, ...] = ("detect", "track", "analytics")

# The measured total of the per-frame body. Deliberately NOT one of
# STAGES: it is the yardstick the stage means are checked against, not a
# fourth stage, and summing it with them would double-count the frame.
TOTAL = "frame"

# Frame rate assumed when a source reports none of its own (webcams, some
# streams). It is used ONLY to bound the speed estimator's per-track
# sample buffer -- every timestamp in the session still comes from the
# source itself (wall clock, in that case), so no reported number is
# derived from this guess.
ASSUMED_FPS = 30.0


@dataclass
class SessionResult:
    """Everything one analysed session produced.

    - ``counts``: gate name -> class -> direction label -> count.
    - ``events``: every ``CrossingEvent``, in the order they fired.
    - ``incidents``: every ``Incident`` (stopped, wrong_way), in order.
    - ``speeds``: track id -> that track's last known speed in km/h, or
      ``None`` when the session was uncalibrated. Entries survive the
      track's reaping: this is the session's record, not live state.
    - ``timings``: stage name -> ``{"mean_ms", "p95_ms", "n"}``, plus the
      separately measured ``frame`` total (see the module docstring).
    - ``meta``: ``source``, ``fps``, ``width``, ``height``,
      ``frames_processed``, ``model``, ``classes``, ``calibrated``, and
      the pixel-space ``gates`` the session ran with.
    - ``frames``: per-frame replay records, populated only when
      ``run_session(..., record_frames=True)``. Kept out of the default
      path because a feature-length clip's worth of them is large; the
      CLI turns it on exactly when it is about to write a session JSON.
    """

    counts: dict
    events: list
    incidents: list
    speeds: dict
    timings: dict
    meta: dict
    frames: list = field(default_factory=list)


class TrackReaper:
    """Decides when a track that stopped being returned is provably gone.

    ``Tracker.update`` returns only tracks a detection updated this frame,
    so absence is ambiguous: the track may be coasting. A confirmed track
    dies once ``time_since_update > max_age``, and may re-associate at
    exactly ``max_age``, so a track last seen at frame ``f`` is still
    possibly alive through frame ``f + max_age`` and provably dead from
    ``f + max_age + 1``. That is the exact boundary this implements.
    """

    def __init__(self, max_age: int) -> None:
        self.max_age = max_age
        self._last_seen: dict[int, int] = {}

    def saw(self, track_id: int, frame_index: int) -> None:
        self._last_seen[track_id] = frame_index

    def reap(self, frame_index: int) -> list[int]:
        """Return -- and forget -- every track id provably dead as of
        ``frame_index``, ascending. An id is only ever returned once."""
        dead = sorted(
            track_id
            for track_id, last in self._last_seen.items()
            if frame_index - last > self.max_age
        )
        for track_id in dead:
            del self._last_seen[track_id]
        return dead

    def drain(self) -> list[int]:
        """Return -- and forget -- every remaining track id, ascending.
        Called once at end of session so the last tracks alive are
        released like any other."""
        dead = sorted(self._last_seen)
        self._last_seen.clear()
        return dead

    def __len__(self) -> int:
        return len(self._last_seen)


class _StageTimer:
    """Per-stage duration samples and the summary they reduce to."""

    def __init__(self, names) -> None:
        self._samples: dict[str, list[float]] = {name: [] for name in names}

    def record(self, name: str, seconds: float) -> None:
        self._samples[name].append(seconds)

    def summary(self) -> dict:
        return {name: _summarise(values) for name, values in self._samples.items()}


def _summarise(seconds: list[float]) -> dict:
    """Mean and exact nearest-rank p95 of a list of durations, in
    milliseconds. An empty list summarises to zeros with ``n = 0`` rather
    than to NaN -- the session JSON refuses NaN, and "no samples" is a
    fact worth stating plainly."""
    n = len(seconds)
    if n == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "n": 0}
    mean_ms = (sum(seconds) / n) * 1000.0
    ordered = sorted(seconds)
    # Nearest-rank: the smallest value at or above the 95th percentile
    # position, with n = 1 collapsing to that single sample.
    rank = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    return {"mean_ms": mean_ms, "p95_ms": ordered[rank] * 1000.0, "n": n}


def _clip_name(source: str) -> str:
    """The session JSON's ``clip``: a file's basename, or the raw spec for
    a webcam index or stream URL."""
    kind, value = classify_spec(source)
    return Path(str(value)).name if kind == "file" else str(value)


def _gate_record(gate: Gate) -> dict:
    """One gate as the session JSON describes it: pixel-space endpoints as
    JSON lists, both direction labels, and the expected direction (which
    must be present even when it is None)."""
    return {
        "name": gate.name,
        "start": [float(gate.start[0]), float(gate.start[1])],
        "end": [float(gate.end[0]), float(gate.end[1])],
        "label_positive": gate.label_positive,
        "label_negative": gate.label_negative,
        "expected_direction": gate.expected_direction,
    }


def run_session(
    config: AppConfig,
    detector: Detector,
    *,
    progress=None,
    max_frames: int | None = None,
    record_frames: bool = False,
    snapshot_dir=None,
    save_video=None,
) -> SessionResult:
    """Analyse ``config.source`` with ``detector`` and return the session.

    ``progress`` -- when given -- is called ``progress(frames_processed,
    total_frames)`` once per processed frame, where ``total_frames`` is
    the source's frame count or ``None`` for a webcam or stream.

    ``max_frames`` stops the session after that many frames; ``0``
    processes none. ``record_frames`` populates ``SessionResult.frames``
    for the session JSON. ``snapshot_dir`` -- when given -- receives an
    evidence JPEG per speed-limit violation. ``save_video`` -- when given
    -- receives an annotated copy of the footage.
    """
    counters: dict[str, GateCounter] = {}
    events: list[CrossingEvent] = []
    incidents: list[Incident] = []
    speeds: dict[int, float | None] = {}
    frames_record: list[dict] = []
    timer = _StageTimer((*STAGES, TOTAL))
    snapshot_path = Path(snapshot_dir) if snapshot_dir is not None else None

    with VideoSource.open(config.source) as source:
        width, height = source.width, source.height
        fps = source.fps if source.fps else ASSUMED_FPS
        total_frames = source.frame_count

        gates = [gate_config.to_gate(width, height) for gate_config in config.gates]
        # GateCounter is looked up through the module namespace so tests can
        # substitute an instrumented subclass and watch its internal state.
        counters = {gate.name: GateCounter(gate) for gate in gates}

        plane = (
            config.calibration.to_plane(width, height)
            if config.calibration is not None
            else None
        )
        speed_estimator = SpeedEstimator(plane, fps)
        incident_detector = IncidentDetector()
        violations = ViolationPolicy(config.speed.limit)
        tracker = Tracker()
        reaper = TrackReaper(tracker.max_age)

        previous_anchor: dict[int, tuple[float, float]] = {}
        # The most recent incident per LIVE track, for the annotated video.
        # Keyed by track so highlighting costs O(tracks on screen) per frame
        # rather than O(incidents so far) -- and so it is released by the
        # same reaping as everything else, instead of growing all session.
        live_incidents: dict[int, Incident] = {}
        writer = None
        processed = 0

        try:
            for frame_index, timestamp, frame in source:
                if max_frames is not None and processed >= max_frames:
                    break

                frame_start = perf_counter()

                # --- stage 1: detect -------------------------------------
                started = perf_counter()
                detections = detector.detect(frame)
                timer.record("detect", perf_counter() - started)

                # --- stage 2: track --------------------------------------
                started = perf_counter()
                tracks = tracker.update(detections, frame_index)
                timer.record("track", perf_counter() - started)

                # --- stage 3: analytics ----------------------------------
                started = perf_counter()
                pending_snapshots: list[CrossingEvent] = []
                for track in tracks:
                    track_id = track.track_id
                    anchor = track.anchor
                    reaper.saw(track_id, frame_index)

                    speed_estimator.observe(track_id, anchor, timestamp)
                    speed = speed_estimator.speed_kmh(track_id)
                    speeds[track_id] = speed

                    previous = previous_anchor.get(track_id)
                    if previous is not None:
                        for gate in gates:
                            event = counters[gate.name].update(
                                track_id,
                                track.class_name,
                                previous,
                                anchor,
                                frame_index,
                                timestamp,
                                speed_kmh=speed,
                                speed_limit_kmh=config.speed.limit,
                            )
                            if event is None:
                                continue
                            events.append(event)
                            wrong_way = incident_detector.note_crossing(event, gate)
                            if wrong_way is not None:
                                incidents.append(wrong_way)
                                live_incidents[track_id] = wrong_way
                            if violations.check(event):
                                pending_snapshots.append(event)
                    previous_anchor[track_id] = anchor

                    stopped = incident_detector.update(
                        track_id, track.class_name, speed, timestamp, frame_index
                    )
                    if stopped is not None:
                        incidents.append(stopped)
                        live_incidents[track_id] = stopped

                for dead_id in reaper.reap(frame_index):
                    for counter in counters.values():
                        counter.forget(dead_id)
                    speed_estimator.forget(dead_id)
                    incident_detector.forget(dead_id)
                    previous_anchor.pop(dead_id, None)
                    live_incidents.pop(dead_id, None)
                timer.record("analytics", perf_counter() - started)

                timer.record(TOTAL, perf_counter() - frame_start)

                # --- untimed: I/O and bookkeeping, charged to no stage ----
                if snapshot_path is not None:
                    for event in pending_snapshots:
                        violations.save_snapshot(frame, event, snapshot_path)

                if record_frames:
                    frames_record.append(
                        {
                            "frame_index": frame_index,
                            "timestamp": timestamp,
                            "tracks": [
                                {
                                    "track_id": track.track_id,
                                    "class_name": track.class_name,
                                    "box": [float(v) for v in track.box],
                                    "speed_kmh": speeds.get(track.track_id),
                                }
                                for track in tracks
                            ],
                        }
                    )

                if save_video is not None:
                    if writer is None:
                        writer = _open_writer(save_video, fps, width, height)
                    from trafficlens import annotate

                    writer.write(
                        annotate.draw_frame(
                            frame,
                            tracks,
                            gates,
                            counters,
                            speeds,
                            incidents=list(live_incidents.values()),
                        )
                    )

                processed += 1
                if progress is not None:
                    progress(processed, total_frames)
        finally:
            if writer is not None:
                writer.release()

        # Release the last live tracks the same way any other death is
        # handled, so no per-track state outlives the session.
        for dead_id in reaper.drain():
            for counter in counters.values():
                counter.forget(dead_id)
            speed_estimator.forget(dead_id)
            incident_detector.forget(dead_id)
        previous_anchor.clear()
        live_incidents.clear()

        meta = {
            "source": config.source,
            "clip": _clip_name(config.source),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "frames_processed": processed,
            "model": config.detector.model,
            "classes": list(config.detector.classes),
            "calibrated": plane is not None,
            "gates": [_gate_record(gate) for gate in gates],
        }

    return SessionResult(
        counts={
            name: {
                class_name: dict(directions)
                for class_name, directions in counter.totals.items()
            }
            for name, counter in counters.items()
        },
        events=events,
        incidents=incidents,
        speeds=speeds,
        timings=timer.summary(),
        meta=meta,
        frames=frames_record,
    )


def _open_writer(path, fps: float, width: int, height: int):
    """Open an MJPG AVI writer at ``path``, creating its parent directory.

    MJPG in an AVI container is the one encoder every ``opencv-python``
    wheel ships with on every platform, so an annotated video always
    writes; the container is chosen by the caller's filename either way.
    """
    import cv2

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (int(width), int(height))
    )
    if not writer.isOpened():
        raise OSError(f"could not open a video writer for {out}")
    return writer


def build_session(result: SessionResult) -> dict:
    """Compose the schema-1 session dict from a finished session.

    Requires ``run_session(..., record_frames=True)``: the replay's whole
    point is its per-frame track states, and silently writing a session
    with an empty ``frames`` list would produce a file that validates but
    replays nothing.
    """
    meta = result.meta
    return {
        "schema": SESSION_SCHEMA_VERSION,
        "clip": meta["clip"],
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "gates": meta["gates"],
        "frames": result.frames,
        "events": [asdict(event) for event in result.events],
        "counts": result.counts,
        "incidents": [asdict(incident) for incident in result.incidents],
        "meta": meta,
    }


def build_summary(result: SessionResult) -> dict:
    """The compact human-and-machine readable summary of a session: what
    was counted, what went wrong, and how fast it ran -- without the
    per-frame replay data."""
    return {
        "counts": result.counts,
        "totals": {
            name: sum(
                count
                for directions in classes.values()
                for count in directions.values()
            )
            for name, classes in result.counts.items()
        },
        "events": [asdict(event) for event in result.events],
        "incidents": [asdict(incident) for incident in result.incidents],
        "speeds": {str(track_id): speed for track_id, speed in result.speeds.items()},
        "timings": result.timings,
        "meta": result.meta,
    }
