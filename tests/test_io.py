"""Tests for trafficlens.io.video (frame sources) and trafficlens.io.export
(CSV/JSON export formats, including the versioned session schema)."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trafficlens.core.gate import CrossingEvent
from trafficlens.io.export import (
    read_events_csv,
    validate_session_dict,
    write_events_csv,
    write_session_json,
    write_summary_json,
)
from trafficlens.io.video import SourceError, VideoSource, classify_spec

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "motorway-a40.webm"

needs_sample = pytest.mark.skipif(
    not SAMPLE.is_file(),
    reason="sample clip data/samples/motorway-a40.webm not fetched",
)


# --- spec classification (no hardware, no files) ------------------------------


def test_classify_int_is_webcam():
    assert classify_spec(0) == ("webcam", 0)
    assert classify_spec(2) == ("webcam", 2)


def test_classify_digit_string_is_webcam():
    assert classify_spec("0") == ("webcam", 0)
    assert classify_spec("13") == ("webcam", 13)


def test_classify_url_schemes_are_streams():
    for url in (
        "rtsp://192.0.2.7:554/stream1",
        "http://example.invalid/cam.mjpg",
        "https://example.invalid/cam.m3u8",
    ):
        assert classify_spec(url) == ("stream", url)


def test_classify_everything_else_is_a_file_path():
    assert classify_spec("clip.mp4") == ("file", "clip.mp4")
    assert classify_spec("data/samples/motorway-a40.webm") == (
        "file",
        "data/samples/motorway-a40.webm",
    )


# --- file source errors -------------------------------------------------------


def test_missing_file_raises_source_error_naming_the_path(tmp_path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(SourceError) as exc:
        VideoSource.open(str(missing))
    assert str(missing) in str(exc.value)


def test_missing_sample_suggests_fetch_samples():
    with pytest.raises(SourceError) as exc:
        VideoSource.open("data/samples/does-not-exist.webm")
    message = str(exc.value)
    assert "data/samples/does-not-exist.webm" in message
    assert "fetch-samples" in message


def test_existing_but_undecodable_file_names_the_codec_possibility(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video container at all" * 100)
    with pytest.raises(SourceError) as exc:
        VideoSource.open(str(junk))
    message = str(exc.value)
    assert str(junk) in message
    assert "codec" in message.lower()


# --- real file source (skips cleanly when the sample is not fetched) ----------


@needs_sample
def test_sample_properties():
    with VideoSource.open(str(SAMPLE)) as source:
        assert source.fps == pytest.approx(30.0)
        assert source.width == 1280
        assert source.height == 720
        assert source.frame_count == 737


@needs_sample
def test_sample_iteration_yields_indexed_timestamped_frames():
    with VideoSource.open(str(SAMPLE)) as source:
        got = []
        for frame_index, timestamp_s, frame in source:
            got.append((frame_index, timestamp_s, frame.shape))
            if frame_index == 2:
                break
    assert [g[0] for g in got] == [0, 1, 2]
    for i, (_, timestamp_s, shape) in enumerate(got):
        assert timestamp_s == pytest.approx(i / 30.0)
        assert shape == (720, 1280, 3)


@needs_sample
def test_context_manager_releases_the_capture():
    with VideoSource.open(str(SAMPLE)) as source:
        pass
    with pytest.raises(SourceError):
        next(iter(source))


# --- events CSV ---------------------------------------------------------------


def _events():
    return [
        CrossingEvent(
            track_id=3,
            class_name="car",
            gate="inbound",
            direction="in",
            signed_direction=1,
            frame_index=37,
            timestamp=1.2333333333333334,
            crossing_x=412.5,
            crossing_y=576.0,
            speed_kmh=52.31,
            is_violation=True,
        ),
        CrossingEvent(
            track_id=9,
            class_name="truck",
            gate="outbound",
            direction="out",
            signed_direction=-1,
            frame_index=120,
            timestamp=4.0,
            crossing_x=800.0,
            crossing_y=576.25,
            speed_kmh=None,
            is_violation=False,
        ),
    ]


def test_events_csv_round_trips_every_field(tmp_path):
    path = tmp_path / "events.csv"
    events = _events()
    write_events_csv(events, path)
    assert read_events_csv(path) == events


def test_events_csv_header_is_exactly_the_dataclass_fields(tmp_path):
    path = tmp_path / "events.csv"
    write_events_csv(_events(), path)
    header = path.read_text().splitlines()[0]
    expected = ",".join(f.name for f in dataclasses.fields(CrossingEvent))
    assert header == expected


def test_events_csv_empty_list_writes_header_only(tmp_path):
    path = tmp_path / "events.csv"
    write_events_csv([], path)
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert read_events_csv(path) == []


# --- summary / session JSON ---------------------------------------------------


def test_summary_json_is_deterministic_sorted_and_newline_terminated(tmp_path):
    summary = {"zebra": 1, "alpha": {"b": 2.5, "a": [1, 2]}}
    p1, p2 = tmp_path / "one.json", tmp_path / "two.json"
    write_summary_json(summary, p1)
    write_summary_json(summary, p2)
    text = p1.read_text()
    assert text == p2.read_text()
    assert text.endswith("\n")
    assert text.index('"alpha"') < text.index('"zebra"')
    assert json.loads(text) == summary


def _session():
    return {
        "schema": 1,
        "clip": "motorway-a40.webm",
        "fps": 30.0,
        "width": 1280,
        "height": 720,
        "gates": [
            {
                "name": "inbound",
                "start": [76.8, 576.0],
                "end": [588.8, 576.0],
                "label_positive": "in",
                "label_negative": "out",
                "expected_direction": "in",
            }
        ],
        "frames": [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "tracks": [
                    {
                        "track_id": 3,
                        "class_name": "car",
                        "box": [400.0, 500.0, 470.0, 560.0],
                        "speed_kmh": None,
                    }
                ],
            }
        ],
        "events": [dataclasses.asdict(e) for e in _events()],
    }


def test_session_json_round_trips_and_validates(tmp_path):
    path = tmp_path / "session.json"
    session = _session()
    write_session_json(session, path)
    loaded = json.loads(path.read_text())
    assert loaded == session
    validate_session_dict(loaded)


def test_session_json_refuses_an_invalid_session(tmp_path):
    path = tmp_path / "session.json"
    bad = _session()
    del bad["fps"]
    with pytest.raises(ValueError):
        write_session_json(bad, path)
    assert not path.exists()


def test_validate_session_requires_each_top_level_key():
    for key in ("schema", "clip", "fps", "width", "height", "gates", "frames", "events"):
        broken = _session()
        del broken[key]
        with pytest.raises(ValueError) as exc:
            validate_session_dict(broken)
        assert key in str(exc.value)


def test_validate_session_rejects_an_unknown_schema_version():
    broken = _session()
    broken["schema"] = 2
    with pytest.raises(ValueError) as exc:
        validate_session_dict(broken)
    assert "schema" in str(exc.value)


def test_validate_session_rejects_malformed_nested_structures():
    broken = _session()
    broken["gates"] = "not-a-list"
    with pytest.raises(ValueError):
        validate_session_dict(broken)

    broken = _session()
    del broken["gates"][0]["name"]
    with pytest.raises(ValueError):
        validate_session_dict(broken)

    broken = _session()
    del broken["frames"][0]["tracks"][0]["box"]
    with pytest.raises(ValueError):
        validate_session_dict(broken)

    broken = _session()
    del broken["events"][0]["direction"]
    with pytest.raises(ValueError):
        validate_session_dict(broken)


def test_validate_session_tolerates_extra_top_level_keys():
    session = _session()
    session["counts"] = {"inbound": {"car": {"in": 1}}}
    validate_session_dict(session)


# --- import hygiene -----------------------------------------------------------


def test_export_and_config_import_without_cv2_or_torch():
    code = (
        "import sys; "
        "import trafficlens.io.export, trafficlens.config; "
        "assert 'cv2' not in sys.modules, 'cv2 was imported'; "
        "assert 'torch' not in sys.modules, 'torch was imported'"
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
