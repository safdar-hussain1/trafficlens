# Architecture

TrafficLens is a layered pipeline with strict boundaries: geometry knows
nothing about video, counting knows nothing about YOLO, and the web layer
only ever talks to the pipeline through thread-safe snapshots.

```
                 ┌────────────────────────────────────────────────┐
 video file      │                  Pipeline                      │
 webcam index ──▶│  Detector ──▶ observations (id, class, box)    │
 rtsp url        │     │                                          │
 (video.py)      │     ▼                                          │
                 │  SpeedEstimator ── anchor ▶ homography ▶ m/s   │
                 │     │                                          │
                 │     ▼                                          │
                 │  GateCounter(s) ── segment × gate ▶ events     │
                 └────────────────┬───────────────────────────────┘
                                  │ FrameResult (tracks, events, counts)
              ┌───────────────────┼──────────────────┐
              ▼                   ▼                  ▼
         annotate.py         export.py          web/session.py
         (HUD frame)     (csv / json / replay)  (JPEG + stats snapshots)
                                                     │
                                                     ▼
                                              web/server.py (FastAPI)
                                                     │
                                                     ▼
                                              static/ (control room UI)
```

## Module responsibilities

| Module | Owns | Never touches |
|---|---|---|
| `geometry.py` | segment intersection, side-of-line, direction | video, models, config |
| `config.py` | pydantic models, YAML loading, normalized coords | runtime state |
| `detection.py` | YOLO + ByteTrack wrapper, class-name resolution | counting, speed |
| `counting.py` | gates, crossing events, per-class/direction tallies | pixels-to-metres |
| `speed.py` | homography / scale calibration, windowed speed + EMA | detection |
| `baseline.py` | the naive band counter (benchmark reference) | production paths |
| `pipeline.py` | orchestration, per-track state, stale-track reaping | HTTP, drawing |
| `annotate.py` | all OpenCV drawing | analysis decisions |
| `video.py` | source/writer with fail-fast validation | analysis |
| `export.py` | events CSV, summary JSON, dashboard replay JSON | HTTP |
| `web/session.py` | the analysis worker thread, live re-configuration | routing |
| `web/server.py` | FastAPI routes, MJPEG stream | OpenCV, models |

## Design decisions that matter

**Counting is segment intersection, not proximity.** A crossing fires
when the segment between an object's anchor on consecutive frames
intersects the gate segment, with the direction taken from the sign of
the side-of-line test. This is frame-rate independent: an object moving
200 px/frame produces a movement segment that still crosses the gate.
Proximity/band approaches miss exactly those objects and also count
objects that graze the band without crossing (both failure modes are
locked in as tests against `baseline.py` and quantified in
`reports/benchmark.json`).

**The anchor is the bottom-centre of the box.** Counting and speed both
live on the road plane; the bottom-centre is the object's ground-contact
point. Using the box centre puts the reference point mid-air, where
perspective makes it drift relative to the road as the object approaches
the camera.

**Speeds come from a plane homography.** Four image points with known
road-plane coordinates in metres define a 3×3 projective map; track
displacement in world space over a 0.5 s sliding window, EMA-smoothed,
gives speed. A single metres-per-pixel scale is supported for
near-orthographic (overhead) cameras. Pixel displacement alone is never
reported as speed — it is wrong everywhere except one depth.

**Class names are resolved against the model's own vocabulary.**
`Detector` maps requested names through `model.names` at load time and
fails fast on unknown names. Hardcoded class lists drift (COCO index 3
is `motorcycle`; a list that renames it silently filters the wrong
class), so there is exactly one list in the codebase — UI suggestions —
and a test asserts it matches the model vocabulary.

**One session, one thread.** The web layer runs a single
`AnalysisSession` worker that owns the capture loop and publishes JPEG +
stats snapshots under a lock. Live edits (moving a gate, changing the
speed limit, recalibrating) take the same lock and mutate the pipeline
between frames — no restart required. Gate identity is by name, so
moving a gate keeps its tallies while replacing it resets them.

**Track state is reaped, counts are not.** Per-track state (trail,
previous anchor, speed window) is dropped after 60 unseen frames so
day-long streams stay bounded; the counted-ID sets are kept, because a
vehicle that crossed and left must stay counted.

**Looping a file resets tracker identity.** When the demo loops a video,
the jump from last frame to first would otherwise read as physical
movement and can fire phantom crossings; the session clears track state
and resets ByteTrack before rewinding.

## Security posture

The web app binds to `127.0.0.1` by default and has no authentication —
it is an operator console, not a public service. Put it behind a reverse
proxy with auth if it must be reachable over a network, and treat RTSP
credentials in configs as secrets (keep configs with credentials out of
version control).
