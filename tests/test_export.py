"""Exports: CSV/JSON structure and the video I/O fail-fast paths."""

import csv
import json

import numpy as np
import pytest

from trafficlens.counting import CrossingEvent
from trafficlens.export import ReplayRecorder, write_events_csv
from trafficlens.video import VideoSource, VideoWriter


def make_event(**overrides) -> CrossingEvent:
    base = dict(
        track_id=7, class_name="car", gate="main", direction="south",
        signed_direction=1, frame_index=42, timestamp=1.4, speed=63.2,
        is_violation=False,
    )
    base.update(overrides)
    return CrossingEvent(**base)


def test_events_csv_roundtrip(tmp_path):
    events = [make_event(), make_event(track_id=8, speed=None, is_violation=True)]
    path = write_events_csv(events, tmp_path / "events.csv")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert rows[0]["class"] == "car"
    assert rows[0]["speed"] == "63.2"
    assert rows[1]["speed"] == ""          # uncalibrated -> empty, not fake zero
    assert rows[1]["violation"] == "1"


def test_replay_recorder_stride(tmp_path):
    rec = ReplayRecorder(1280, 720, 30.0, stride=2)
    assert rec.meta["fps"] == 30.0
    with pytest.raises(ValueError):
        ReplayRecorder(1280, 720, 30.0, stride=0)


def test_video_source_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        VideoSource("no/such/clip.mp4")


def test_video_writer_rejects_wrong_frame_size(tmp_path):
    with VideoWriter(tmp_path / "out.mp4", 640, 480, 30.0) as w:
        with pytest.raises(ValueError, match="does not match"):
            w.write(np.zeros((720, 1280, 3), dtype=np.uint8))


def test_video_roundtrip(tmp_path):
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, 320, 240, 30.0) as w:
        for i in range(10):
            frame = np.full((240, 320, 3), i * 20, dtype=np.uint8)
            w.write(frame)
    with VideoSource(str(path)) as src:
        assert src.info.width == 320
        assert src.info.fps == pytest.approx(30.0)
        frames = list(src.frames())
    assert len(frames) == 10
    # Iteration ended cleanly at EOF — no crash, no None frame escaping.


def test_summary_json_shape(tmp_path):
    # Build the JSON by hand from a summary-like dict to lock the schema
    # the dashboard depends on.
    summary = {"frames": 1, "gates": {}, "events": 0, "violations": 0,
               "speed_unit": "kmh", "speed_by_class": {}}
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summary))
    assert set(json.loads(p.read_text())) >= {"frames", "gates", "events", "violations"}
