"""The ``trafficlens`` command-line interface.

Six commands: ``run`` (analyse a source), ``calibrate`` (check a config's
surveyed geometry), ``fetch-samples`` (download the sample clips),
``export-model`` (produce the ONNX the browser engine runs),
``serve`` and ``bench``.

Import policy -- the reason every heavy import in this file sits INSIDE a
command body rather than at the top: ``trafficlens --help`` must work on a
core install, with neither torch, ultralytics nor onnxruntime importable.
A user who has just run ``pip install trafficlens`` and wants to see what
the tool does should get the help text, not an ImportError naming a
package they have not installed yet. So the detector adapters are imported
inside ``build_detector``, the pipeline inside ``run``, and the ONNX
exporter inside ``export-model``. See
``tests/test_cli.py::test_help_works_with_no_detector_backend_importable``,
which asserts this with those three modules blocked outright.

Error policy: every failure a user can cause -- a missing config file, a
malformed ``--gate``, an unfetched clip, a calibration that will not
validate -- is turned into a ``click.ClickException``, so it prints as one
readable ``Error:`` line and exits non-zero. A traceback is reserved for
bugs in this program, which is what a traceback actually means.
"""

from __future__ import annotations

from pathlib import Path

import click

from trafficlens import __version__, samples

# Written into --export-dir by `run`.
EVENTS_CSV = "events.csv"
SUMMARY_JSON = "summary.json"
SESSION_JSON = "session.json"
VIOLATIONS_DIR = "violations"

_CALIBRATION_GUIDANCE = """\
Calibrating a camera for speed
------------------------------
Speed is measured on the road plane, in metres -- never guessed from
pixels. To calibrate, survey at least five points you can see in the frame
AND know the real ground distances between (lane-marking dash ends, stop
lines, kerb joints, anything painted at a documented interval), then add
them to the config's `calibration` block:

  image_points  -- NORMALIZED [0, 1] image coordinates (x / frame width,
                   y / frame height), so one survey works at any
                   resolution of the same camera view.
  world_points  -- the same points in metres on the road plane, in any
                   consistent frame of reference you like.

  holdout_image_points / holdout_world_points -- surveyed points kept OUT
                   of the fit. Four points exactly determine a homography,
                   so a four-point self-check can never fail; a holdout is
                   the only out-of-sample evidence the fit is real.

Spread the points across the whole area you care about, not along one
line: a near-collinear survey is rejected as degenerate.
"""


# --- shared helpers -----------------------------------------------------------


def load_or_fail(config_path):
    """Load a config, turning every load-time failure into a readable
    ``Error:`` line instead of a traceback."""
    from trafficlens.config import ConfigError, load_config

    try:
        return load_config(config_path)
    except ConfigError as error:
        raise click.ClickException(str(error)) from error


def parse_gate(spec: str) -> dict:
    """Parse a ``--gate NAME,X1,Y1,X2,Y2`` value into a gate config dict.

    Coordinates are normalized to [0, 1], the same convention config files
    use, so a gate typed on the command line means the same thing as one
    written in YAML.
    """
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 5:
        raise ValueError(
            f"--gate {spec!r} has {len(parts)} comma-separated fields; the "
            f"form is NAME,X1,Y1,X2,Y2 (5 fields)"
        )
    name, *raw = parts
    if not name:
        raise ValueError(f"--gate {spec!r} has an empty name")
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
    except ValueError as error:
        raise ValueError(
            f"--gate {spec!r}: the four coordinates must be numbers"
        ) from error
    for value in (x1, y1, x2, y2):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"--gate {spec!r}: coordinates are normalized to [0, 1] "
                f"(x / frame width, y / frame height), got {value}"
            )
    return {"name": name, "start": (x1, y1), "end": (x2, y2)}


def apply_overrides(
    config,
    *,
    source=None,
    model=None,
    classes=None,
    limit=None,
    gate_specs=(),
):
    """Return a new, fully re-validated ``AppConfig`` with the command
    line's overrides folded in.

    Re-validation is the point: an override goes through exactly the same
    model validators a YAML file does, so ``--classes car,unicorn`` fails
    at the same place and with the same message a bad config file would.

    ``--gate`` REPLACES the config's gates rather than adding to them --
    repeating the option builds the complete replacement set. Merging
    would leave a user unable to say "just this one gate" without editing
    the file.
    """
    from pydantic import ValidationError

    from trafficlens.config import AppConfig, ConfigError

    data = config.model_dump()
    if source is not None:
        data["source"] = source
    if model is not None:
        data["detector"]["model"] = model
    if classes is not None:
        names = [name.strip() for name in classes.split(",") if name.strip()]
        data["detector"]["classes"] = names
    if limit is not None:
        data["speed"]["limit"] = limit
    if gate_specs:
        try:
            data["gates"] = [parse_gate(spec) for spec in gate_specs]
        except ValueError as error:
            raise click.BadParameter(str(error), param_hint="--gate") from error

    try:
        return AppConfig.model_validate(data)
    except ValidationError as error:
        raise click.ClickException(
            f"the command-line overrides do not validate -- "
            f"{compact_validation_error(error)}"
        ) from error
    except ConfigError as error:  # pragma: no cover - defensive
        raise click.ClickException(str(error)) from error


def compact_validation_error(error) -> str:
    """One readable line per pydantic error: ``where: what``.

    ``str(ValidationError)`` is a multi-line developer dump ending in a
    ``errors.pydantic.dev`` documentation URL -- correct for a stack trace,
    wrong for someone who mistyped a ``--gate``. This keeps the location
    and the reason and drops the rest, including pydantic's ``Value
    error,`` prefix on custom validator messages.
    """
    parts = []
    for entry in error.errors():
        where = ".".join(str(item) for item in entry["loc"]) or "config"
        message = entry["msg"]
        prefix = "Value error, "
        if message.startswith(prefix):
            message = message[len(prefix) :]
        parts.append(f"{where}: {message}")
    return "; ".join(parts)


def build_detector(config):
    """Construct the detector this config asks for.

    A ``.onnx`` model goes to the onnxruntime adapter, anything else to the
    ultralytics one. Both are imported HERE, not at module scope, so the
    CLI itself stays importable without either backend -- and tests replace
    this whole function with a scripted fake, which is what lets the
    pipeline be exercised end to end with no weights on disk.
    """
    detector_config = config.detector
    kwargs = {
        "size": detector_config.imgsz,
        "conf": detector_config.confidence,
        "classes": tuple(detector_config.classes),
    }
    try:
        if detector_config.model.endswith(".onnx"):
            from trafficlens.detect.onnx_yolo import OnnxDetector

            return OnnxDetector(detector_config.model, **kwargs)
        from trafficlens.detect.ultralytics_yolo import UltralyticsDetector

        return UltralyticsDetector(detector_config.model, **kwargs)
    except ImportError as error:
        extra = "onnx" if detector_config.model.endswith(".onnx") else "detect"
        raise click.ClickException(
            f"the detector backend for {detector_config.model!r} is not "
            f"installed: {error}. Install it with "
            f"`pip install 'trafficlens[{extra}]'`."
        ) from error


# --- the command group --------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="trafficlens")
def cli() -> None:
    """Traffic analytics from any camera: gate counting, calibrated speeds,
    and a browser engine that runs the detector on your GPU."""


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path),
    help="YAML session config (see configs/ for worked examples).",
)
@click.option("--source", default=None, help="Override the config's source.")
@click.option(
    "--gate",
    "gate_specs",
    multiple=True,
    metavar="NAME,X1,Y1,X2,Y2",
    help=(
        "Replace the config's gates. Coordinates are normalized to [0, 1]. "
        "Repeat for more than one gate."
    ),
)
@click.option(
    "--classes", default=None, help="Override the detected classes, comma-separated."
)
@click.option(
    "--limit",
    type=float,
    default=None,
    help="Speed limit in km/h; crossings above it are flagged as violations.",
)
@click.option(
    "--max-frames", type=int, default=None, help="Stop after this many frames."
)
@click.option(
    "--save-video",
    type=click.Path(path_type=Path),
    default=None,
    help="Write an annotated copy of the footage here.",
)
@click.option(
    "--export-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        f"Write {EVENTS_CSV}, {SUMMARY_JSON} and {SESSION_JSON} (plus any "
        f"violation snapshots) into this directory."
    ),
)
@click.option("--model", default=None, help="Override the detector weights.")
def run(
    config_path,
    source,
    gate_specs,
    classes,
    limit,
    max_frames,
    save_video,
    export_dir,
    model,
) -> None:
    """Analyse a video source and report what crossed each gate."""
    from trafficlens.core.homography import CalibrationError
    from trafficlens.io.video import SourceError
    from trafficlens.pipeline import SAVE_VIDEO_SUFFIXES, run_session

    config = load_or_fail(config_path)
    config = apply_overrides(
        config,
        source=source,
        model=model,
        classes=classes,
        limit=limit,
        gate_specs=gate_specs,
    )

    # Checked BEFORE the model is loaded: whether a container can hold the
    # annotated video is knowable from the filename alone, so failing on it
    # after a model load and a frame of work would waste the user's time on
    # a typo -- and .webm, which fails, is the extension of this project's
    # own flagship sample clip.
    if save_video is not None:
        suffix = Path(save_video).suffix.lower()
        if suffix not in SAVE_VIDEO_SUFFIXES:
            raise click.ClickException(
                f"--save-video {save_video}: annotated video is written as "
                f"MJPG, which a {suffix or '(no extension)'} container "
                f"cannot hold. Use one of: "
                f"{', '.join(SAVE_VIDEO_SUFFIXES)}."
            )

    detector = build_detector(config)

    snapshot_dir = None
    if export_dir is not None and config.speed.limit is not None:
        snapshot_dir = Path(export_dir) / VIOLATIONS_DIR

    def report(processed: int, total) -> None:
        # Every 100 frames, on stderr, so piping the report itself stays
        # clean. Quiet for the short clips a first run usually starts with.
        if processed % 100 == 0:
            suffix = f" / {total}" if total else ""
            click.echo(f"  ... {processed}{suffix} frames", err=True)

    try:
        result = run_session(
            config,
            detector,
            progress=report,
            max_frames=max_frames,
            record_frames=export_dir is not None,
            snapshot_dir=snapshot_dir,
            save_video=save_video,
        )
    except (SourceError, CalibrationError, OSError) as error:
        # OSError is the backstop for anything the up-front checks cannot
        # know: a writer this platform's OpenCV build refuses, a full disk,
        # a directory that turns unwritable mid-run. A traceback is for a
        # bug in this program, not for the environment saying no.
        raise click.ClickException(str(error)) from error

    _print_report(result)

    if export_dir is not None:
        _write_exports(result, Path(export_dir))
        click.echo(f"\nExported to {export_dir}")
    if save_video is not None:
        click.echo(f"Annotated video written to {save_video}")


def _print_report(result) -> None:
    meta = result.meta
    click.echo(
        f"Source: {meta['source']} "
        f"({meta['width']}x{meta['height']} @ {meta['fps']:.3g} fps)"
    )
    click.echo(f"Frames: {meta['frames_processed']}")
    click.echo(
        "Speeds: "
        + (
            "calibrated, in km/h"
            if meta["calibrated"]
            else "not available (no calibration block in this config)"
        )
    )

    if not result.counts:
        click.echo("\nNo gates configured, so nothing was counted.")
    for gate_name in result.counts:
        classes = result.counts[gate_name]
        click.echo(f"\nGate {gate_name}:")
        if not classes:
            click.echo("  (no crossings)")
        for class_name in sorted(classes):
            directions = classes[class_name]
            parts = "  ".join(
                f"{direction} {directions[direction]}"
                for direction in sorted(directions)
            )
            click.echo(f"  {class_name:<12} {parts}")

    click.echo(f"\nCrossings: {len(result.events)}")
    violations = sum(1 for event in result.events if event.is_violation)
    if violations:
        click.echo(f"Violations: {violations}")
    if result.incidents:
        click.echo(f"Incidents: {len(result.incidents)}")
        for incident in result.incidents:
            click.echo(
                f"  {incident.kind}: track {incident.track_id} "
                f"({incident.class_name}) at {incident.timestamp:.1f}s "
                f"-- {incident.detail}"
            )
    else:
        click.echo("Incidents: 0")

    timings = "  ".join(
        f"{stage} {values['mean_ms']:.2f}" for stage, values in result.timings.items()
    )
    click.echo(f"\nTimings (mean ms/frame): {timings}")
    click.echo(
        "  'frame' totals the analysis stages only -- video decode, export "
        "and annotated-video encoding sit outside it, so it is not "
        "end-to-end throughput."
    )


def _write_exports(result, export_dir: Path) -> None:
    from trafficlens.io.export import (
        write_events_csv,
        write_session_json,
        write_summary_json,
    )
    from trafficlens.pipeline import build_session, build_summary

    export_dir.mkdir(parents=True, exist_ok=True)
    write_events_csv(result.events, export_dir / EVENTS_CSV)
    write_summary_json(build_summary(result), export_dir / SUMMARY_JSON)
    write_session_json(build_session(result), export_dir / SESSION_JSON)


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
def serve(host: str, port: int) -> None:
    """Serve the browser dashboard (not built yet)."""
    click.echo(
        "`trafficlens serve` is a placeholder. The web application -- the "
        "API, the dashboard and the in-browser engine -- arrives in a later "
        "task; this command will start it once it exists."
    )
    click.echo(f"It will bind {host}:{port} when it does.")
    click.echo("Until then, use `trafficlens run --export-dir ...` for results.")


@cli.command("fetch-samples")
@click.option(
    "--dest",
    type=click.Path(path_type=Path),
    default=samples.DEFAULT_DEST,
    show_default=True,
    help="Directory the clips are downloaded into.",
)
def fetch_samples(dest: Path) -> None:
    """Download the Creative Commons sample clips."""
    click.echo(f"Fetching {len(samples.SAMPLES)} sample clips into {dest} ...")
    try:
        for name in samples.SAMPLES:
            path, downloaded = samples.fetch(name, dest)
            size_mb = path.stat().st_size / 1_000_000
            state = "downloaded" if downloaded else "already present"
            click.echo(f"  {name}  ({size_mb:.1f} MB, {state})")
            click.echo(f"      {samples.LICENCES[name]}")
    except samples.SampleError as error:
        raise click.ClickException(str(error)) from error
    click.echo("Done. Point a config's `source` at one of these files.")


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path),
    help="YAML session config whose calibration block is checked.",
)
@click.option(
    "--width",
    type=int,
    default=None,
    help="Frame width in pixels; read from the source when omitted.",
)
@click.option(
    "--height",
    type=int,
    default=None,
    help="Frame height in pixels; read from the source when omitted.",
)
def calibrate(config_path, width, height) -> None:
    """Explain how to survey a camera, and check a config's calibration."""
    from trafficlens.core.homography import CalibrationError

    click.echo(_CALIBRATION_GUIDANCE)
    config = load_or_fail(config_path)

    if config.calibration is None:
        click.echo(
            f"{config_path}: no calibration block, so this session reports no "
            f"speeds at all -- the engine never derives one from bare pixels. "
            f"Add `calibration:` with image_points and world_points as above "
            f"to enable them."
        )
        return

    frame_width, frame_height = _frame_size(config, width, height)
    click.echo(f"Checking {config_path} against a {frame_width}x{frame_height} frame.")

    try:
        plane = config.calibration.to_plane(
            frame_width, frame_height, context=str(config_path)
        )
    except CalibrationError as error:
        raise click.ClickException(str(error)) from error

    def to_pixels(points):
        return [(x * frame_width, y * frame_height) for x, y in points]

    calibration = config.calibration
    fit = plane.reprojection_error(
        to_pixels(calibration.image_points), list(calibration.world_points)
    )
    click.echo(f"  fit correspondences: {len(calibration.image_points)}")
    click.echo(f"  fit mean error:      {fit['mean_m']:.3f} m")
    click.echo(f"  fit max error:       {fit['max_m']:.3f} m")

    if calibration.holdout_image_points:
        holdout = plane.reprojection_error(
            to_pixels(calibration.holdout_image_points),
            list(calibration.holdout_world_points),
        )
        click.echo(
            f"  held-out points:     {len(calibration.holdout_image_points)}"
        )
        click.echo(f"  held-out mean error: {holdout['mean_m']:.3f} m")
        click.echo(f"  held-out max error:  {holdout['max_m']:.3f} m")
        click.echo(
            "\nThe held-out numbers are the ones to trust: those points took "
            "no part in the fit."
        )
    else:
        click.echo(
            "  held-out points:     none surveyed -- the fit error above is a "
            "self-check, not out-of-sample evidence."
        )

    click.echo("\nCalibration validated.")


def _frame_size(config, width, height) -> tuple[int, int]:
    """The pixel frame size the normalized calibration points refer to:
    whatever was passed explicitly, otherwise read from the source."""
    if width is not None and height is not None:
        return width, height
    if width is not None or height is not None:
        raise click.ClickException("pass both --width and --height, or neither")

    from trafficlens.io.video import SourceError, VideoSource

    try:
        with VideoSource.open(config.source) as source:
            return source.width, source.height
    except SourceError as error:
        raise click.ClickException(
            f"{error} Pass --width and --height to check the calibration "
            f"without opening the source."
        ) from error


@cli.command()
def bench() -> None:
    """Point at the benchmark harness (not built yet)."""
    click.echo(
        "`trafficlens bench` is a placeholder. Benchmarking -- accuracy "
        "against labelled ground truth and throughput per backend -- lands "
        "with the scripts under scripts/ in a later task."
    )
    click.echo(
        "Meanwhile, `trafficlens run --export-dir ...` writes the session "
        "JSON those scripts/ tools will score."
    )


@cli.command("export-model")
@click.option(
    "--weights",
    default="yolo11n.pt",
    show_default=True,
    help="Checkpoint to export. yolo11n is the size the browser can run.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Where the .onnx file is written. Parent directories are created.",
)
@click.option(
    "--imgsz",
    default=640,
    show_default=True,
    type=int,
    help="Square input size the graph is fixed at; must be a multiple of 32.",
)
def export_model(weights, output: Path, imgsz: int) -> None:
    """Export a YOLO11 checkpoint to ONNX for the browser engine."""
    import shutil
    import tempfile

    if imgsz <= 0 or imgsz % 32 != 0:
        raise click.ClickException(
            f"--imgsz must be a positive multiple of 32 (the model's stride "
            f"grid), got {imgsz}"
        )
    source_weights = Path(weights)
    if not source_weights.is_file():
        raise click.ClickException(f"weights file not found: {source_weights}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise click.ClickException(
            f"exporting needs ultralytics and torch: {error}. Install them "
            f"with `pip install 'trafficlens[detect]'`."
        ) from error

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ultralytics writes the .onnx next to the checkpoint it was given, so
    # the checkpoint is copied into a scratch directory first. Exporting in
    # place would drop build artifacts wherever the weights happen to live
    # -- usually the repository root.
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / source_weights.name
        shutil.copy2(source_weights, staged)
        click.echo(f"Exporting {source_weights} to ONNX at imgsz={imgsz} ...")
        produced = Path(
            YOLO(str(staged)).export(format="onnx", imgsz=imgsz, dynamic=False)
        )
        shutil.move(str(produced), str(output))

    size_mb = output.stat().st_size / 1_000_000
    click.echo(f"Wrote {output} ({size_mb:.1f} MB).")


if __name__ == "__main__":  # pragma: no cover
    cli()
