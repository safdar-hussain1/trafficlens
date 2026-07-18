"""Result exports: events CSV, summary JSON, and dashboard replay JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from trafficlens.counting import CrossingEvent
from trafficlens.pipeline import FrameResult, Pipeline


def write_events_csv(events: list[CrossingEvent], path: str | Path) -> Path:
    """One row per counted crossing — the audit trail of a session."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame", "timestamp_s", "track_id", "class", "gate", "direction", "speed", "violation"]
        )
        for e in events:
            writer.writerow([
                e.frame_index,
                f"{e.timestamp:.3f}",
                e.track_id,
                e.class_name,
                e.gate,
                e.direction,
                f"{e.speed:.1f}" if e.speed is not None else "",
                int(e.is_violation),
            ])
    return path


def write_summary_json(pipeline: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pipeline.summary(), indent=2))
    return path


class ReplayRecorder:
    """Records per-frame track positions for the interactive dashboard.

    The output JSON is small (positions, not pixels): a browser canvas
    can replay an entire analysed session — boxes, trails, crossings —
    without shipping any video.
    """

    def __init__(self, frame_width: int, frame_height: int, fps: float, stride: int = 1):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.meta = {"width": frame_width, "height": frame_height, "fps": fps, "stride": stride}
        self.stride = stride
        self.frames: list[dict] = []

    def record(self, result: FrameResult) -> None:
        if result.frame_index % self.stride != 0:
            return
        self.frames.append({
            "i": result.frame_index,
            "t": round(result.timestamp, 3),
            "tracks": [
                {
                    "id": tv.track_id,
                    "c": tv.class_name,
                    "b": [round(v, 1) for v in tv.box],
                    "s": round(tv.speed, 1) if tv.speed is not None else None,
                }
                for tv in result.tracks
            ],
            "events": [
                {"id": e.track_id, "c": e.class_name, "g": e.gate, "d": e.direction,
                 "s": round(e.speed, 1) if e.speed is not None else None, "v": int(e.is_violation)}
                for e in result.events
            ],
        })

    def write(self, pipeline: Pipeline, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "gates": [
                {
                    "name": c.gate.name,
                    "start": list(c.gate.start),
                    "end": list(c.gate.end),
                    "labels": [c.gate.label_positive, c.gate.label_negative],
                }
                for c in pipeline.counters
            ],
            "summary": pipeline.summary(),
            "frames": self.frames,
        }
        path.write_text(json.dumps(payload, separators=(",", ":")))
        return path
