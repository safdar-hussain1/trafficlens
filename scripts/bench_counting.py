#!/usr/bin/env python3
"""Score the engine, and every standard failure mode, against the
hand-labelled slit-scan ground truth, and write the two reports.

Writes into the tracked ``reports/`` directory:

1. ``reports/counting_accuracy.json`` -- crossing-level precision, recall,
   F1, count error and signed bias for every {tracker} x {counting rule}
   composition, on the full label set and on the ``certain`` rows, plus
   the band rule's miss/phantom trade-off across a ``band_px`` sweep.
2. ``reports/detection_noise.json`` -- the scale of detector box jitter,
   labelled as the proxy it is.

Detections are computed ONCE and cached to a git-ignored path, and every
method is driven from that cache. Two methods differing by a detector
run would be a comparison of detector variance dressed up as a comparison
of counting rules, so the guarantee is made structural: the detector runs
here, in this script, and the harness never sees a model.

Usage:

    PYTHONPATH=src .venv/bin/python scripts/bench_counting.py \
        --config configs/motorway.yaml --gate inbound \
        --gt data/groundtruth/motorway_inbound_gt.json

The first run detects the whole labelled window and writes the cache;
later runs reuse it. The cache is stamped with the detector that produced
it and is refused -- never repaired -- if that detector changes.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trafficlens.bench.harness import (  # noqa: E402
    DEFAULT_MATCH_WINDOW,
    DetectionCacheError,
    build_methods,
    measure_detection_noise,
    median_gate_approach_px_per_frame,
    read_detection_cache,
    run_counting_benchmark,
    sweep_band_px,
    write_detection_cache,
    write_report,
)
from trafficlens.bench.slitscan import GroundTruth  # noqa: E402
from trafficlens.config import load_config  # noqa: E402
from trafficlens.io.video import VideoSource  # noqa: E402

#: Band half-widths, in pixels, the band rule's trade-off is swept over.
#: Spans from far too narrow for near-field motion to wide enough to
#: swallow a fair share of the lower frame, so both failure modes are
#: visible on one curve rather than one being chosen away.
DEFAULT_BAND_VALUES = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score counting accuracy against hand-labelled ground truth."
    )
    parser.add_argument("--config", default="configs/motorway.yaml")
    parser.add_argument("--gate", default="inbound")
    parser.add_argument(
        "--gt", default="data/groundtruth/motorway_inbound_gt.json"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="override the config's detector model (the report records it)",
    )
    parser.add_argument(
        "--cache",
        default="private/bench",
        help="directory for the shared detection cache; must be git-ignored",
    )
    parser.add_argument("--out", default="reports/counting_accuracy.json")
    parser.add_argument("--noise-out", default="reports/detection_noise.json")
    parser.add_argument(
        "--band-values",
        default=",".join(str(value) for value in DEFAULT_BAND_VALUES),
        help="comma-separated band_px values to sweep",
    )
    return parser.parse_args(argv)


def require_git_ignored(directory: Path) -> None:
    """Refuse to write the detection cache anywhere git would track it.

    The cache is large, derived, and reproducible from the clip and the
    weights; committing it would put a detector's output in the repository
    next to the ground truth that must never be derived from one.
    """
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(directory)], cwd=ROOT
        )
    except OSError:
        print(
            f"warning: git is not available, so {directory} could not be "
            f"confirmed git-ignored. Do not commit anything it contains.",
            file=sys.stderr,
        )
        return
    if result.returncode != 0:
        raise SystemExit(
            f"refusing to write the detection cache to {directory}: it is "
            f"NOT git-ignored. Write it under private/ (already ignored) or "
            f"add this path to .gitignore."
        )


def resolve(path_like: str) -> str:
    """A repo-relative path made absolute, so the script runs from
    anywhere."""
    path = Path(path_like)
    return str(path if path.is_absolute() else ROOT / path)


def detect_window(config, source_spec: str, start_frame: int, end_frame: int):
    """Run the real detector over the labelled window, once.

    Imported here, not at module scope, so ``--help`` and every code path
    that hits a warm cache work on a machine with no detector backend
    installed.
    """
    from trafficlens.cli import build_detector

    detector = build_detector(config)
    frames = []
    with VideoSource.open(source_spec) as source:
        for frame_index, timestamp, frame in source:
            if frame_index < start_frame:
                continue
            if frame_index > end_frame:
                break
            frames.append((frame_index, timestamp, detector.detect(frame)))
            if len(frames) % 100 == 0:
                print(f"  detected {len(frames)} frames", flush=True)
    if not frames or frames[-1][0] < end_frame:
        last = frames[-1][0] if frames else None
        raise SystemExit(
            f"the clip ended at frame {last} before the labelled window's "
            f"last frame {end_frame}; the report would be scored against "
            f"labels whose footage was never seen"
        )
    return frames


def load_detections(config, source_spec: str, cache_dir: Path, truth):
    """Read the shared detections from cache, detecting them first if the
    cache is missing or was produced by a different detector."""
    key = {
        "clip": truth.clip,
        "model": config.detector.model,
        "confidence": config.detector.confidence,
        "imgsz": config.detector.imgsz,
        "classes": list(config.detector.classes),
        "start_frame": truth.start_frame,
        "end_frame": truth.end_frame,
    }
    cache_path = cache_dir / f"{Path(truth.clip).stem}_detections.json"
    try:
        detections = read_detection_cache(cache_path, key=key)
        print(f"detections: {len(detections)} frames from cache {cache_path}")
        return detections, key
    except DetectionCacheError as error:
        print(f"detection cache unusable ({error}); running the detector")

    detections = detect_window(
        config, source_spec, truth.start_frame, truth.end_frame
    )
    write_detection_cache(cache_path, detections, key=key)
    print(f"detections: {len(detections)} frames written to {cache_path}")
    return detections, key


def main(argv=None) -> int:
    args = parse_args(argv)

    config = load_config(resolve(args.config))
    if args.model is not None:
        config = config.model_copy(
            update={"detector": config.detector.model_copy(update={"model": args.model})}
        )

    gate_configs = {gate.name: gate for gate in config.gates}
    if args.gate not in gate_configs:
        raise SystemExit(
            f"config {args.config} has no gate named {args.gate!r}; it has: "
            f"{', '.join(sorted(gate_configs)) or '(none)'}"
        )

    source_spec = resolve(config.source)
    with VideoSource.open(source_spec) as probe:
        width, height = probe.width, probe.height
    gate = gate_configs[args.gate].to_gate(width, height)

    truth = GroundTruth.load(
        resolve(args.gt), gate=gate, clip_path=Path(source_spec)
    )
    print(
        f"ground truth {truth.path.name}: {len(truth.crossings)} crossings "
        f"({truth.certain_count} certain) over frames "
        f"{truth.start_frame}-{truth.end_frame} at gate {truth.gate_name}"
    )

    cache_dir = Path(resolve(args.cache))
    require_git_ignored(cache_dir)
    detections, key = load_detections(config, source_spec, cache_dir, truth)

    methods = build_methods(gate)
    print(f"scoring {len(methods)} method(s): {', '.join(sorted(methods))}")
    report = run_counting_benchmark(
        detections,
        truth,
        methods,
        DEFAULT_MATCH_WINDOW,
        gate=gate,
        detector={
            "model": key["model"],
            "confidence": key["confidence"],
            "imgsz": key["imgsz"],
            "classes": key["classes"],
        },
    )

    band_values = [float(token) for token in args.band_values.split(",") if token]
    approach = median_gate_approach_px_per_frame(detections, gate)
    report["band_sweep"] = {
        "tracker": "engine",
        "median_gate_approach_px_per_frame": approach,
        "note": (
            "Miss rate and phantom rate are separate series because in "
            "general they trade: a wider band cures step-over misses and "
            "creates phantoms, so no single band-versus-gate number is a "
            "result rather than a choice of operating point. On THIS clip "
            "they do not trade, and the reason is the approach speed "
            "recorded above -- see the entries note."
        ),
        "entries_note": (
            "A band of half-width b fires when the anchor first comes "
            "within b of the gate line, roughly b / v frames before the "
            "line is reached. At the measured approach speed no band in "
            "this sweep is ever stepped over, so the classic step-over miss "
            "mode never appears; the band rule's dominant error here is the "
            "crossing FRAME, and a mistimed crossing costs a miss AND a "
            "phantom at once, which is why both series rise together. On "
            "faster traffic or a lower frame rate the step-over mode would "
            "reappear and the two would trade as usual."
        ),
        "entries": sweep_band_px(
            detections,
            truth.crossings,
            gate=gate,
            band_values=band_values,
            window=DEFAULT_MATCH_WINDOW,
        ),
    }

    out_path = write_report(resolve(args.out), report)
    noise_path = write_report(
        resolve(args.noise_out), measure_detection_noise(detections)
    )

    print(f"\nwrote {out_path.relative_to(ROOT)} and {noise_path.relative_to(ROOT)}")
    print("\nfull label set, crossing level:")
    print(f"  {'method':22s} {'P':>6s} {'R':>6s} {'F1':>6s} {'pred':>6s} {'bias':>6s}")
    for name in sorted(report["methods"]):
        full = report["methods"][name]["full"]
        print(
            f"  {name:22s} {full['precision']:6.3f} {full['recall']:6.3f} "
            f"{full['f1']:6.3f} {full['n_predicted']:6d} {full['signed_bias']:+6d}"
        )
    print(
        "\nEvery figure above is an UPPER bound: the labelling gate was "
        "chosen in the near field for label reliability, not for the "
        "engine's convenience."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
