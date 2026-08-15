"""Tests for the ``trafficlens`` command-line interface and the sample-clip
fetcher.

Nothing here downloads anything or loads a model: the sample fetcher's one
network call is injected, and ``run`` builds its detector through a factory the
tests replace with the same scripted detector ``test_pipeline`` uses. The one
test that genuinely exports an ONNX model skips unless ultralytics and the
yolo11n checkpoint are both present.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from test_pipeline import (
    HEIGHT,
    MID_GATE,
    SCALE_CALIBRATION,
    WIDTH,
    ScriptedDetector,
    two_vehicles,
    write_clip,
)
from trafficlens import cli as cli_module
from trafficlens import samples
from trafficlens.cli import cli

ROOT = Path(__file__).resolve().parents[1]
YOLO11N = ROOT / "yolo11n.pt"

COMMANDS = ["run", "serve", "fetch-samples", "calibrate", "bench", "export-model"]


# --- helpers ------------------------------------------------------------------


def write_config(path: Path, source: Path, **overrides) -> Path:
    data = {
        "source": str(source),
        "detector": {
            "model": "yolo11s.pt",
            "classes": ["car", "truck", "bus"],
            "confidence": 0.25,
            "imgsz": 640,
        },
        "gates": [{**MID_GATE, "start": list(MID_GATE["start"]), "end": list(MID_GATE["end"])}],
    }
    data.update(overrides)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.fixture
def scripted(monkeypatch):
    """Replace the CLI's detector factory with the scripted fake, and record
    the AppConfig it was handed so override tests can inspect it."""
    seen = {}

    def factory(config):
        seen["config"] = config
        return ScriptedDetector(two_vehicles)

    monkeypatch.setattr(cli_module, "build_detector", factory)
    return seen


@pytest.fixture
def session(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", frames=60)
    return write_config(tmp_path / "config.yaml", clip), clip


# --- help ---------------------------------------------------------------------


def test_group_help_lists_every_command():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    for command in COMMANDS:
        assert command in result.output, command


@pytest.mark.parametrize("command", COMMANDS)
def test_every_subcommand_has_working_help(command):
    result = CliRunner().invoke(cli, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_run_help_documents_every_documented_option():
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0, result.output
    for option in (
        "--config",
        "--source",
        "--gate",
        "--classes",
        "--limit",
        "--max-frames",
        "--save-video",
        "--export-dir",
        "--model",
    ):
        assert option in result.output, option


def test_export_model_help_documents_imgsz():
    result = CliRunner().invoke(cli, ["export-model", "--help"])
    assert result.exit_code == 0, result.output
    assert "--imgsz" in result.output


def test_help_works_with_no_detector_backend_importable():
    """`trafficlens --help` must work on a core install: every heavy import
    lives inside the command that needs it, so a machine without torch can
    still discover the CLI."""
    blocker = (
        "import sys, importlib.abc\n"
        "BLOCKED = {'torch', 'ultralytics', 'onnxruntime'}\n"
        "class _Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in BLOCKED:\n"
        "            raise ImportError('blocked for this test: ' + fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "from click.testing import CliRunner\n"
        "from trafficlens.cli import cli\n"
        "runner = CliRunner()\n"
        f"for args in [[]] + [[c] for c in {COMMANDS!r}]:\n"
        "    r = runner.invoke(cli, args + ['--help'])\n"
        "    assert r.exit_code == 0, (args, r.output, r.exception)\n"
        "leaked = sorted(m for m in BLOCKED if m in sys.modules)\n"
        "print('leaked=' + ','.join(leaked))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", blocker],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "leaked="


def test_importing_the_cli_never_pulls_in_a_detector_backend():
    code = (
        "import sys\n"
        "import trafficlens.cli\n"
        "leaked = sorted(m for m in ('torch', 'ultralytics', 'onnxruntime') "
        "if m in sys.modules)\n"
        "print(','.join(leaked))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# --- config errors ------------------------------------------------------------


def test_run_on_a_missing_config_exits_non_zero_with_a_readable_message(tmp_path):
    missing = tmp_path / "nope.yaml"
    result = CliRunner().invoke(cli, ["run", "--config", str(missing)])
    assert result.exit_code != 0
    assert "not found" in result.output
    assert str(missing) in result.output
    assert "Traceback" not in result.output


def test_calibrate_on_a_missing_config_exits_non_zero_with_a_readable_message(tmp_path):
    missing = tmp_path / "nope.yaml"
    result = CliRunner().invoke(cli, ["calibrate", "--config", str(missing)])
    assert result.exit_code != 0
    assert "not found" in result.output
    assert "Traceback" not in result.output


def test_run_on_an_invalid_config_exits_non_zero_without_a_traceback(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("source: clip.mp4\ndetector:\n  classes: [unicorn]\n")
    result = CliRunner().invoke(cli, ["run", "--config", str(bad)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert str(bad) in result.output


def test_run_on_a_missing_video_exits_non_zero_without_a_traceback(tmp_path, scripted):
    config = write_config(tmp_path / "c.yaml", tmp_path / "absent.mp4")
    result = CliRunner().invoke(cli, ["run", "--config", str(config)])
    assert result.exit_code != 0
    assert "not found" in result.output
    assert "Traceback" not in result.output


# --- run ----------------------------------------------------------------------


def test_run_reports_the_counts_it_measured(session, scripted):
    config, _ = session
    result = CliRunner().invoke(cli, ["run", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "mid" in result.output
    assert "car" in result.output and "truck" in result.output


def test_run_writes_the_export_artifacts(session, scripted, tmp_path):
    config, clip = session
    out = tmp_path / "exports"
    result = CliRunner().invoke(
        cli, ["run", "--config", str(config), "--export-dir", str(out)]
    )
    assert result.exit_code == 0, result.output

    events = out / "events.csv"
    summary = out / "summary.json"
    session_json = out / "session.json"
    assert events.is_file() and summary.is_file() and session_json.is_file()

    from trafficlens.io.export import read_events_csv, validate_session_dict
    import json

    assert len(read_events_csv(events)) == 2
    validate_session_dict(json.loads(session_json.read_text()))
    payload = json.loads(summary.read_text())
    assert payload["counts"]["mid"]["car"]["out"] == 1
    assert payload["meta"]["frames_processed"] == 60


def test_run_honours_max_frames(session, scripted):
    config, _ = session
    result = CliRunner().invoke(
        cli, ["run", "--config", str(config), "--max-frames", "5"]
    )
    assert result.exit_code == 0, result.output
    assert "5" in result.output


def test_run_source_override_replaces_the_config_source(tmp_path, scripted):
    original = write_clip(tmp_path / "a.avi", frames=10)
    other = write_clip(tmp_path / "b.avi", frames=12)
    config = write_config(tmp_path / "c.yaml", original)
    result = CliRunner().invoke(
        cli, ["run", "--config", str(config), "--source", str(other), "--max-frames", "3"]
    )
    assert result.exit_code == 0, result.output
    assert scripted["config"].source == str(other)


def test_run_model_classes_and_limit_overrides_reach_the_config(session, scripted):
    config, _ = session
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--config",
            str(config),
            "--model",
            "yolo11n.pt",
            "--classes",
            "car,bus",
            "--limit",
            "50",
            "--max-frames",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    applied = scripted["config"]
    assert applied.detector.model == "yolo11n.pt"
    assert applied.detector.classes == ["car", "bus"]
    assert applied.speed.limit == 50.0


def test_gate_option_replaces_the_configs_gates(session, scripted):
    config, _ = session
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--config",
            str(config),
            "--gate",
            "north,0.0,0.4,1.0,0.4",
            "--gate",
            "south,0.0,0.6,1.0,0.6",
            "--max-frames",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    gates = scripted["config"].gates
    assert [gate.name for gate in gates] == ["north", "south"]
    assert gates[0].start == (0.0, 0.4)
    assert gates[1].end == (1.0, 0.6)


@pytest.mark.parametrize(
    "spec",
    [
        "north,0.0,0.4,1.0",  # too few fields
        "north,0.0,0.4,1.0,0.4,0.9",  # too many fields
        "north,zero,0.4,1.0,0.4",  # not a number
        "north,0.0,0.4,1.0,1.4",  # outside [0, 1]
        ",0.0,0.4,1.0,0.4",  # empty name
    ],
)
def test_a_malformed_gate_spec_exits_non_zero_with_a_readable_message(
    session, scripted, spec
):
    config, _ = session
    result = CliRunner().invoke(
        cli, ["run", "--config", str(config), "--gate", spec]
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "--gate" in result.output or "gate" in result.output


def test_run_writes_an_annotated_video_when_asked(session, scripted, tmp_path):
    config, _ = session
    video = tmp_path / "annotated.avi"
    result = CliRunner().invoke(
        cli,
        ["run", "--config", str(config), "--save-video", str(video), "--max-frames", "20"],
    )
    assert result.exit_code == 0, result.output
    assert video.is_file() and video.stat().st_size > 0


# --- serve / bench stubs ------------------------------------------------------


def test_serve_says_the_web_app_arrives_in_a_later_task():
    result = CliRunner().invoke(cli, ["serve"])
    assert result.exit_code == 0, result.output
    assert "later task" in result.output.lower()


def test_bench_points_at_the_benchmark_scripts():
    result = CliRunner().invoke(cli, ["bench"])
    assert result.exit_code == 0, result.output
    assert "scripts/" in result.output


# --- calibrate ----------------------------------------------------------------


def test_calibrate_reports_the_error_in_metres(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", frames=5)
    config = write_config(
        tmp_path / "c.yaml",
        clip,
        calibration={
            "image_points": [list(p) for p in SCALE_CALIBRATION["image_points"]],
            "world_points": [list(p) for p in SCALE_CALIBRATION["world_points"]],
        },
    )
    result = CliRunner().invoke(cli, ["calibrate", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "m" in result.output
    assert "mean error" in result.output.lower()
    assert "5" in result.output  # five surveyed correspondences


def test_calibrate_reports_held_out_error_when_a_holdout_is_surveyed(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", frames=5)
    config = write_config(
        tmp_path / "c.yaml",
        clip,
        calibration={
            "image_points": [list(p) for p in SCALE_CALIBRATION["image_points"]],
            "world_points": [list(p) for p in SCALE_CALIBRATION["world_points"]],
            # (0.3, 0.7) is pixel (192, 336), which the fitted scaling maps
            # to (19.2, 33.6) m. Surveying it 0.3 m away makes the held-out
            # error a known, distinct number -- so a command that quietly
            # reported the (exact, 0.000 m) fit error under a "held-out"
            # label could not pass this.
            "holdout_image_points": [[0.3, 0.7]],
            "holdout_world_points": [[19.2, 33.9]],
        },
    )
    result = CliRunner().invoke(cli, ["calibrate", "--config", str(config)])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.lower().split())
    assert "fit mean error: 0.000 m" in flat
    assert "held-out mean error: 0.300 m" in flat
    assert "held-out max error: 0.300 m" in flat


def test_calibrate_uses_explicit_frame_size_without_opening_the_video(tmp_path):
    config = write_config(
        tmp_path / "c.yaml",
        tmp_path / "never-fetched.mp4",
        calibration={
            "image_points": [list(p) for p in SCALE_CALIBRATION["image_points"]],
            "world_points": [list(p) for p in SCALE_CALIBRATION["world_points"]],
        },
    )
    result = CliRunner().invoke(
        cli,
        ["calibrate", "--config", str(config), "--width", str(WIDTH), "--height", str(HEIGHT)],
    )
    assert result.exit_code == 0, result.output
    assert "mean error" in result.output.lower()


def test_calibrate_on_an_uncalibrated_config_explains_how_to_add_one(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", frames=5)
    config = write_config(tmp_path / "c.yaml", clip)
    result = CliRunner().invoke(cli, ["calibrate", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "no calibration" in result.output.lower()
    assert "world_points" in result.output


def test_calibrate_prints_survey_guidance(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", frames=5)
    config = write_config(tmp_path / "c.yaml", clip)
    result = CliRunner().invoke(cli, ["calibrate", "--config", str(config)])
    assert result.exit_code == 0, result.output
    lowered = result.output.lower()
    assert "metre" in lowered
    assert "normalized" in lowered or "normalised" in lowered


# --- samples ------------------------------------------------------------------


def test_the_three_sample_clips_are_named_with_licences():
    assert set(samples.SAMPLES) == {
        "motorway-a40.webm",
        "car-detection.mp4",
        "person-bicycle-car-detection.mp4",
    }
    doc = samples.__doc__ or ""
    assert "CC BY 3.0" in doc
    assert "CC BY 4.0" in doc
    for name in samples.SAMPLES:
        assert name in doc, name
    for name, licence in samples.LICENCES.items():
        assert name in samples.SAMPLES
        assert licence


def test_fetch_downloads_into_the_destination(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, path):
        calls.append((url, path))
        path.write_bytes(b"x" * (samples.MIN_BYTES + 1))

    monkeypatch.setattr(samples, "download", fake_download)
    path, downloaded = samples.fetch("car-detection.mp4", tmp_path)
    assert downloaded is True
    assert path == tmp_path / "car-detection.mp4"
    assert path.is_file()
    assert calls[0][0] == samples.SAMPLES["car-detection.mp4"]


def test_fetch_skips_a_file_that_is_already_there(tmp_path, monkeypatch):
    existing = tmp_path / "car-detection.mp4"
    existing.write_bytes(b"x" * (samples.MIN_BYTES + 1))

    def fail(url, path):
        raise AssertionError("must not download an existing file")

    monkeypatch.setattr(samples, "download", fail)
    path, downloaded = samples.fetch("car-detection.mp4", tmp_path)
    assert path == existing
    assert downloaded is False


def test_a_trivially_small_download_is_rejected_and_not_kept(tmp_path, monkeypatch):
    def tiny(url, path):
        path.write_bytes(b"<html>404</html>")

    monkeypatch.setattr(samples, "download", tiny)
    with pytest.raises(samples.SampleError) as excinfo:
        samples.fetch("car-detection.mp4", tmp_path)
    assert "car-detection.mp4" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == [], "a truncated download must be removed"


def test_fetch_rejects_an_unknown_sample_name(tmp_path):
    with pytest.raises(samples.SampleError):
        samples.fetch("not-a-sample.mp4", tmp_path)


def test_fetch_samples_command_reports_each_clip(tmp_path, monkeypatch):
    def fake_download(url, path):
        path.write_bytes(b"x" * (samples.MIN_BYTES + 1))

    monkeypatch.setattr(samples, "download", fake_download)
    result = CliRunner().invoke(cli, ["fetch-samples", "--dest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    for name in samples.SAMPLES:
        assert name in result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(samples.SAMPLES)


def test_fetch_samples_command_reports_a_failure_without_a_traceback(
    tmp_path, monkeypatch
):
    def tiny(url, path):
        path.write_bytes(b"nope")

    monkeypatch.setattr(samples, "download", tiny)
    result = CliRunner().invoke(cli, ["fetch-samples", "--dest", str(tmp_path)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


# --- export-model -------------------------------------------------------------


def test_export_model_requires_an_output_path():
    result = CliRunner().invoke(cli, ["export-model"])
    assert result.exit_code != 0
    assert "--output" in result.output


def test_export_model_writes_only_to_the_given_output_path(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    if not YOLO11N.is_file():
        pytest.skip("yolo11n.pt not present on disk")

    before = {p.name for p in ROOT.iterdir()}
    output = tmp_path / "nested" / "yolo11n.onnx"
    result = CliRunner().invoke(
        cli,
        [
            "export-model",
            "--weights",
            str(YOLO11N),
            "--output",
            str(output),
            "--imgsz",
            "320",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert output.stat().st_size > 1_000_000
    assert {p.name for p in ROOT.iterdir()} == before, (
        "export-model must never leave artifacts in the repository root"
    )
