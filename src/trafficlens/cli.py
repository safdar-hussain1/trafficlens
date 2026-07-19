"""Command-line interface.

Examples::

    trafficlens run --config configs/highway.yaml --save-video out.mp4
    trafficlens run --source 0 --classes person --gate "door,0.2,0.5,0.8,0.5"
    trafficlens serve --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from trafficlens.config import AppConfig, DetectorConfig, GateConfig, load_config
from trafficlens.export import ReplayRecorder, write_events_csv, write_summary_json
from trafficlens.pipeline import Pipeline
from trafficlens.video import VideoSource, VideoWriter


@click.group()
@click.version_option(package_name="trafficlens")
def cli() -> None:
    """TrafficLens — count and speed-track anything that moves on camera."""


def _parse_gate(spec: str) -> GateConfig:
    """Parse 'name,x1,y1,x2,y2' (normalized coords) from the command line."""
    parts = spec.split(",")
    if len(parts) != 5:
        raise click.BadParameter(f"gate {spec!r} must be name,x1,y1,x2,y2 (normalized 0-1)")
    name, *coords = parts
    try:
        x1, y1, x2, y2 = (float(v) for v in coords)
    except ValueError as exc:
        raise click.BadParameter(f"gate {spec!r}: coordinates must be numbers") from exc
    return GateConfig(name=name, start=(x1, y1), end=(x2, y2))


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config file.")
@click.option("--source", help="Video file, webcam index, or RTSP URL (overrides config).")
@click.option("--model", help="Ultralytics model, e.g. yolo11n.pt (overrides config).")
@click.option("--classes", help="Comma-separated class names (overrides config).")
@click.option("--conf", type=float, help="Detection confidence threshold (overrides config).")
@click.option("--gate", "gates", multiple=True,
              help="Counting gate 'name,x1,y1,x2,y2' in normalized coords; repeatable.")
@click.option("--save-video", type=click.Path(), help="Write the annotated video here.")
@click.option("--export-dir", type=click.Path(), default="exports",
              help="Directory for events.csv / summary.json / replay.json.")
@click.option("--replay/--no-replay", default=False, help="Also record dashboard replay JSON.")
@click.option("--show/--no-show", default=False, help="Open a preview window while processing.")
@click.option("--max-frames", type=int, default=None, help="Stop after N frames (benchmarking).")
def run(config_path, source, model, classes, conf, gates, save_video,
        export_dir, replay, show, max_frames) -> None:
    """Process a video source headlessly (or with --show) and export results."""
    config = load_config(config_path) if config_path else AppConfig()
    if source:
        config = config.model_copy(update={"source": source})
    if gates:
        config = config.model_copy(update={"gates": [_parse_gate(g) for g in gates]})
    det_updates = {}
    if model:
        det_updates["model"] = model
    if classes:
        det_updates["classes"] = [c.strip() for c in classes.split(",") if c.strip()]
    if conf is not None:
        det_updates["confidence"] = conf
    if det_updates:
        config = config.model_copy(
            update={"detector": DetectorConfig(**{**config.detector.model_dump(), **det_updates})}
        )
    if not config.gates:
        click.echo("note: no gates configured — objects will be tracked but nothing counted", err=True)

    from trafficlens.annotate import draw_frame  # deferred: pulls in cv2 GUI bits

    with VideoSource(config.source) as src:
        info = src.info
        click.echo(
            f"source {config.source!r}: {info.width}x{info.height} @ {info.fps:.0f} fps"
            + (f", {info.frame_count} frames" if info.frame_count else " (live)")
        )
        pipeline = Pipeline(config, info.width, info.height, fps=info.fps)
        recorder = ReplayRecorder(info.width, info.height, info.fps) if replay else None
        writer = None
        if save_video:
            writer = VideoWriter(save_video, info.width, info.height, info.fps)
        try:
            for idx, frame in src.frames():
                result = pipeline.process(frame)
                if recorder:
                    recorder.record(result)
                for event in result.events:
                    speed = f" @ {event.speed:.0f} {config.speed.unit}" if event.speed else ""
                    flag = "  ** VIOLATION **" if event.is_violation else ""
                    click.echo(
                        f"[{result.timestamp:7.2f}s] {event.gate}: {event.class_name} "
                        f"#{event.track_id} {event.direction}{speed}{flag}"
                    )
                for inc in result.incidents:
                    click.echo(
                        f"[{result.timestamp:7.2f}s] INCIDENT {inc.kind.upper()}: "
                        f"{inc.class_name} #{inc.track_id} — {inc.detail}"
                    )
                if writer or show:
                    annotated = draw_frame(
                        frame, result, pipeline.counters,
                        speed_unit=config.speed.unit, speed_limit=config.speed.speed_limit,
                    )
                    if writer:
                        writer.write(annotated)
                    if show:
                        import cv2

                        cv2.imshow("TrafficLens", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                if max_frames is not None and result.frame_index + 1 >= max_frames:
                    break
        finally:
            if writer:
                writer.release()

    out = Path(export_dir)
    write_events_csv(pipeline.events, out / "events.csv")
    write_summary_json(pipeline, out / "summary.json")
    if recorder:
        recorder.write(pipeline, out / "replay.json")

    summary = pipeline.summary()
    click.echo("\n=== session summary ===")
    for gate_name, stats in summary["gates"].items():
        dirs = "  ".join(f"{d}: {n}" for d, n in sorted(stats["by_direction"].items()))
        click.echo(f"{gate_name}: {stats['total']} total   {dirs}")
        for cls, per_dir in sorted(stats["by_class"].items()):
            click.echo(f"    {cls}: " + "  ".join(f"{d}: {n}" for d, n in sorted(per_dir.items())))
    if summary["speed_by_class"]:
        unit = "km/h" if summary["speed_unit"] == "kmh" else "mph"
        click.echo(f"speeds at crossing ({unit}):")
        for cls, s in sorted(summary["speed_by_class"].items()):
            click.echo(
                f"    {cls}: n={s['n']} mean={s['mean']} median={s['median']} "
                f"p85={s['p85']} max={s['max']}"
            )
    if summary["violations"]:
        click.echo(f"violations: {summary['violations']}")
    if summary.get("incidents"):
        click.echo("incidents: " + "  ".join(
            f"{kind}: {n}" for kind, n in sorted(summary["incidents"].items())))
    click.echo(f"exports written to {out}/")


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind address. Use 0.0.0.0 only on a trusted network.")
@click.option("--port", default=8000, show_default=True, type=int)
def serve(host: str, port: int) -> None:
    """Start the interactive web app."""
    import uvicorn

    from trafficlens.web.server import app

    click.echo(f"TrafficLens web UI: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


@cli.command("fetch-samples")
@click.option("--dest", type=click.Path(), default="data/samples", show_default=True)
def fetch_samples(dest: str) -> None:
    """Download CC-licensed sample traffic clips for a quick start."""
    from trafficlens.samples import fetch_all

    for path in fetch_all(Path(dest)):
        click.echo(f"ready: {path}")


def main() -> None:
    try:
        cli(standalone_mode=True)
    except Exception as exc:  # pragma: no cover - last-resort formatting
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
