"""Benchmark harness: gate counting vs the naive band baseline, plus model FPS.

Runs the full pipeline on real footage and, on the *same* track stream,
runs the tutorial-style band counter — so the two approaches are compared
on identical detections and any difference is purely the counting rule.

Outputs (under reports/):
  benchmark.json      counting comparison + fps table (dashboard + README)
  replay_<name>.json  per-frame track positions for the dashboard replay
  figures/<name>_frame*.jpg   annotated stills

Usage:
  PYTHONPATH=src python scripts/run_benchmark.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trafficlens.annotate import draw_frame                     # noqa: E402
from trafficlens.baseline import NaiveBandCounter               # noqa: E402
from trafficlens.config import load_config                      # noqa: E402
from trafficlens.export import ReplayRecorder, write_events_csv  # noqa: E402
from trafficlens.pipeline import Pipeline                       # noqa: E402
from trafficlens.video import VideoSource, VideoWriter          # noqa: E402

FIGS = ROOT / "reports" / "figures"


def run_scenario(name: str, config_path: Path, snapshot_frames: list[int],
                 save_video: bool = True, frame_stride: int = 1) -> dict:
    """Full pipeline + naive baseline on one clip; returns the comparison.

    ``frame_stride > 1`` simulates a lower camera frame rate (many CCTV
    feeds run at 8-12 fps): stride 3 turns 30 fps footage into 10 fps.
    """
    config = load_config(config_path)
    src = VideoSource(str(ROOT / config.source))
    info = src.info
    eff_fps = info.fps / frame_stride
    pipeline = Pipeline(config, info.width, info.height, fps=eff_fps)
    recorder = ReplayRecorder(info.width, info.height, eff_fps)

    gate = pipeline.counters[0].gate
    naive = NaiveBandCounter(
        x_min=min(gate.start[0], gate.end[0]),
        x_max=max(gate.start[0], gate.end[0]),
        line_y=(gate.start[1] + gate.end[1]) / 2.0,
    )

    writer = None
    if save_video:
        writer = VideoWriter(FIGS / f"{name}_annotated.mp4", info.width, info.height, eff_fps)

    prev_anchor: dict[int, tuple[float, float]] = {}
    max_step: dict[int, float] = {}
    t0 = time.perf_counter()
    frames = 0
    for idx, frame in src.frames():
        if idx % frame_stride:
            continue
        result = pipeline.process(frame)
        recorder.record(result)
        for tv in result.tracks:
            naive.update(tv.track_id, tv.anchor)
            if tv.track_id in prev_anchor:
                step = abs(tv.anchor[1] - prev_anchor[tv.track_id][1])
                max_step[tv.track_id] = max(max_step.get(tv.track_id, 0.0), step)
            prev_anchor[tv.track_id] = tv.anchor
        annotated = draw_frame(frame, result, pipeline.counters,
                               speed_unit=config.speed.unit,
                               speed_limit=config.speed.speed_limit)
        if writer:
            writer.write(annotated)
        if idx in snapshot_frames:
            cv2.imwrite(str(FIGS / f"{name}_frame{idx:04d}.jpg"), annotated)
        frames += 1
    wall = time.perf_counter() - t0
    if writer:
        writer.release()
    src.release()

    gate_ids = pipeline.counters[0].counted
    naive_ids = naive.counted
    events_csv = ROOT / "reports" / f"events_{name}.csv"
    write_events_csv(pipeline.events, events_csv)
    recorder.write(pipeline, ROOT / "reports" / f"replay_{name}.json")

    summary = pipeline.summary()
    return {
        "clip": config.source,
        "model": config.detector.model,
        "imgsz": config.detector.imgsz,
        "effective_fps": round(eff_fps, 1),
        "frames": frames,
        "duration_s": round(frames / eff_fps, 1) if eff_fps else None,
        "processing_fps": round(frames / wall, 1),
        "gates": summary["gates"],
        "total_all_gates": sum(c.total for c in pipeline.counters),
        # the band comparison is against the FIRST gate only — the naive
        # counter is built on that gate's geometry
        "gate_count": pipeline.counters[0].total,
        "gate_by_class": summary["gates"][pipeline.counters[0].gate.name]["by_class"],
        "gate_by_direction": summary["gates"][pipeline.counters[0].gate.name]["by_direction"],
        "naive_band_count": naive.total,
        "naive_missed_ids": sorted(gate_ids - naive_ids),
        "naive_phantom_ids": sorted(naive_ids - gate_ids),
        "max_anchor_step_px_top5": sorted(max_step.values(), reverse=True)[:5],
        "speed_by_class": summary["speed_by_class"],
        "violations": summary["violations"],
        "incidents": summary["incidents"],
        "incident_log": [
            {"kind": i.kind, "track": i.track_id, "class": i.class_name,
             "t": round(i.timestamp, 2), "detail": i.detail}
            for i in pipeline.incidents
        ],
        "events": [
            {"t": round(e.timestamp, 2), "class": e.class_name, "direction": e.direction,
             "speed": round(e.speed, 1) if e.speed is not None else None,
             "violation": e.is_violation, "track": e.track_id}
            for e in pipeline.events
        ],
    }


def fps_benchmark(clip: Path, models: list[str], n_frames: int = 120) -> list[dict]:
    """Throughput of each model on the same frames (detection+tracking)."""
    from trafficlens.config import AppConfig, DetectorConfig

    rows = []
    for model in models:
        src = VideoSource(str(clip))
        config = AppConfig(
            source=str(clip),
            detector=DetectorConfig(model=model, classes=["car", "truck", "bus", "motorcycle",
                                                          "person", "bicycle"]),
        )
        pipeline = Pipeline(config, src.info.width, src.info.height, fps=src.info.fps)
        n = 0
        t0 = time.perf_counter()
        for idx, frame in src.frames():
            pipeline.process(frame)
            n += 1
            if n >= n_frames:
                break
        wall = time.perf_counter() - t0
        src.release()
        rows.append({"model": model, "frames": n, "fps": round(n / wall, 1),
                     "device": pipeline.detector.device})
        print(f"  {model}: {n / wall:.1f} fps on {pipeline.detector.device}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip the fps sweep")
    args = parser.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    results: dict = {"scenarios": {}}

    print("== scenario: motorway (A40, two carriageways) ==")
    results["scenarios"]["motorway"] = run_scenario(
        "motorway", ROOT / "configs" / "motorway.yaml", snapshot_frames=[90, 300, 480, 650],
    )
    print(json.dumps({k: v for k, v in results["scenarios"]["motorway"].items()
                      if k not in ("events",)}, indent=2))

    print("== scenario: motorway at 10 fps (typical CCTV rate) ==")
    results["scenarios"]["motorway_10fps"] = run_scenario(
        "motorway_10fps", ROOT / "configs" / "motorway.yaml",
        snapshot_frames=[], save_video=False, frame_stride=3,
    )
    print(json.dumps({k: v for k, v in results["scenarios"]["motorway_10fps"].items()
                      if k not in ("events",)}, indent=2))

    print("== scenario: street (person-bicycle-car) ==")
    results["scenarios"]["street"] = run_scenario(
        "street", ROOT / "configs" / "street.yaml", snapshot_frames=[80, 210, 400, 560],
    )
    print(json.dumps({k: v for k, v in results["scenarios"]["street"].items()
                      if k not in ("events",)}, indent=2))

    if not args.quick:
        print("== model throughput ==")
        results["fps"] = fps_benchmark(
            ROOT / "data" / "samples" / "motorway-a40.webm",
            ["yolo26n.pt", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
        )

    out = ROOT / "reports" / "benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
