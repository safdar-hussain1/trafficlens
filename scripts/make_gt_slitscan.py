#!/usr/bin/env python3
"""Generate the review images a human labels a ground-truth set from.

Writes into a git-ignored directory (``private/gt`` by default):

1. ``<clip>_<gate>_slitscan.png`` -- the whole window as one slit-scan,
   one row per frame, time running downward, with a frame-number axis.
2. ``tiles/<clip>_<gate>_f0000-0119.png`` -- the same slit-scan cut into
   overlapping row bands, each magnified and carrying its own frame-number
   axis, so a blob's row is read directly as a frame index. Tiles overlap
   so that no crossing is cut in half at a boundary.
3. ``frames/<clip>_<gate>_sheet01.png`` -- contact sheets of the full
   frames at a supplied list of candidate frame numbers, with the gate
   drawn, which is how a labeller confirms a blob is a vehicle and assigns
   its class and direction.

This script produces MATERIAL, never labels. It says nothing about how
many vehicles crossed; a human decides that by looking, under
``data/groundtruth/PROTOCOL.md``. Nothing here reads the detector, the
tracker or the counter, and it must stay that way: a ground truth derived
from the system under test measures only that the system agrees with
itself.

Usage:

    PYTHONPATH=src python scripts/make_gt_slitscan.py \
        --config configs/motorway.yaml --gate inbound \
        --start-frame 0 --end-frame 734

    PYTHONPATH=src python scripts/make_gt_slitscan.py \
        --config configs/motorway.yaml --gate inbound \
        --candidates 37,64,102

Omit ``--end-frame`` and the script measures the clip's last decodable
frame itself, which is the safer default: ``motorway-a40.webm``
advertises 737 frames in its container and decodes 735 (0-734), so a
window taken from the header would ask for frames that do not exist.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trafficlens.bench.slitscan import (  # noqa: E402
    DEFAULT_SAMPLES,
    DEFAULT_THICKNESS_PX,
    build_slitscan,
    tile_windows,
)
from trafficlens.config import load_config  # noqa: E402
from trafficlens.io.video import VideoSource  # noqa: E402

AXIS_MARGIN_PX = 74
BANNER_PX = 24
INK = (235, 235, 235)
FAINT = (120, 120, 120)
PAPER = (24, 24, 24)
GATE_BGR = (60, 220, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate slit-scan review images for ground-truth labelling."
    )
    parser.add_argument("--config", required=True, help="session config YAML")
    parser.add_argument("--gate", required=True, help="which gate to scan")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="inclusive last frame; defaults to the clip's last frame minus one",
    )
    parser.add_argument("--thickness", type=int, default=DEFAULT_THICKNESS_PX)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--tile-frames", type=int, default=120)
    parser.add_argument("--tile-overlap", type=int, default=20)
    parser.add_argument(
        "--row-scale",
        type=int,
        default=4,
        help="vertical magnification of a tile, in pixels per frame",
    )
    parser.add_argument(
        "--out",
        default="private/gt",
        help="output directory; must be git-ignored",
    )
    parser.add_argument(
        "--candidates",
        default="",
        help="comma-separated candidate frame numbers, or a path to a file "
        "holding one per line, to render as full-frame contact sheets",
    )
    parser.add_argument("--sheet-cols", type=int, default=2)
    parser.add_argument("--sheet-rows", type=int, default=3)
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=640,
        help="width of each full frame on a contact sheet",
    )
    return parser.parse_args(argv)


# --- output location --------------------------------------------------------


def require_git_ignored(directory: Path) -> None:
    """Refuse to write review images anywhere git would track them.

    Review images are large, derived, and dangerous to commit alongside a
    label set: once a picture is in the repository the labels start being
    edited to match the picture rather than the footage.
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
            f"refusing to write to {directory}: it is NOT git-ignored. "
            f"Review images must never be committed -- write them under "
            f"private/ (already ignored) or add this path to .gitignore."
        )


# --- drawing ----------------------------------------------------------------


def with_frame_axis(
    strip: np.ndarray,
    first_frame: int,
    row_scale: int,
    label_every: int,
    tick_every: int,
    banner: str,
) -> np.ndarray:
    """Magnify a slit-scan vertically and give it a frame-number axis.

    The axis is the whole point of the image: a labeller reads a blob's
    row straight off it as the frame index of a crossing, with no
    counting and no arithmetic.
    """
    rows, width = strip.shape[0], strip.shape[1]
    magnified = cv2.resize(
        strip,
        (width, rows * row_scale),
        interpolation=cv2.INTER_NEAREST,  # keep the recorded pixels, unblended
    )
    if magnified.ndim == 2:
        magnified = cv2.cvtColor(magnified, cv2.COLOR_GRAY2BGR)

    height = magnified.shape[0]
    canvas = np.full(
        (height + BANNER_PX, width + AXIS_MARGIN_PX, 3), PAPER, dtype=np.uint8
    )
    canvas[BANNER_PX:, AXIS_MARGIN_PX:] = magnified
    cv2.putText(canvas, banner, (6, 16), FONT, 0.42, INK, 1, cv2.LINE_AA)

    for row in range(rows):
        frame_number = first_frame + row
        is_label = frame_number % label_every == 0
        if not is_label and frame_number % tick_every != 0:
            continue
        y = BANNER_PX + int((row + 0.5) * row_scale)
        if y >= canvas.shape[0]:
            continue
        colour = INK if is_label else FAINT
        stub = 10 if is_label else 5
        cv2.line(
            canvas,
            (AXIS_MARGIN_PX - 12, y),
            (AXIS_MARGIN_PX + stub, y),
            colour,
            1,
        )
        if is_label:
            cv2.putText(
                canvas, str(frame_number), (4, y + 4), FONT, 0.38, INK, 1,
                cv2.LINE_AA,
            )
    return canvas


def draw_gate(frame: np.ndarray, gate) -> np.ndarray:
    """The gate as the labeller must see it: an arrowed, bounded segment
    with both endpoints marked, because a vehicle passing the line beyond
    an endpoint is not a crossing."""
    canvas = frame.copy()
    start = (int(round(gate.start[0])), int(round(gate.start[1])))
    end = (int(round(gate.end[0])), int(round(gate.end[1])))
    cv2.arrowedLine(canvas, start, end, GATE_BGR, 2, cv2.LINE_AA, tipLength=0.03)
    for point in (start, end):
        cv2.circle(canvas, point, 6, GATE_BGR, 2, cv2.LINE_AA)
    return canvas


def contact_sheet(
    panels: list[tuple[int, float, np.ndarray]],
    cols: int,
    rows: int,
    thumb_width: int,
) -> np.ndarray:
    """Lay full frames out in a grid, each captioned with its frame number
    and timestamp."""
    scale = thumb_width / panels[0][2].shape[1]
    thumb_height = int(round(panels[0][2].shape[0] * scale))
    cell_height = thumb_height + BANNER_PX
    sheet = np.full(
        (rows * cell_height, cols * thumb_width, 3), PAPER, dtype=np.uint8
    )
    for position, (frame_number, timestamp, image) in enumerate(panels):
        row, col = divmod(position, cols)
        thumb = cv2.resize(
            image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA
        )
        top = row * cell_height
        left = col * thumb_width
        cv2.putText(
            sheet,
            f"frame {frame_number}   t = {timestamp:.3f} s",
            (left + 6, top + 16),
            FONT,
            0.5,
            INK,
            1,
            cv2.LINE_AA,
        )
        sheet[top + BANNER_PX:top + cell_height, left:left + thumb_width] = thumb
    return sheet


# --- candidates -------------------------------------------------------------


def parse_candidates(spec: str) -> list[int]:
    if not spec:
        return []
    path = Path(spec)
    if path.is_file():
        text = path.read_text().replace(",", "\n")
    else:
        text = spec.replace(",", "\n")
    numbers = sorted({int(token) for token in text.split() if token.strip()})
    return numbers


def last_decodable_frame(source_spec: str) -> int:
    """Decode the whole clip and report the last frame index that came out.

    A container's own frame count is a claim, not a measurement: this clip
    advertises 737 frames and yields 735. The labelling window has to be
    the frames that actually decode, so it is measured rather than taken
    from the header.
    """
    last = -1
    with VideoSource.open(source_spec) as source:
        for frame_index, _timestamp, _frame in source:
            last = frame_index
    return last


def grab_frames(source_spec: str, wanted: list[int]):
    """Read the wanted frames in one forward pass of a fresh source.

    ``VideoSource`` is single-pass, so this opens its own source rather
    than sharing the one the slit-scan consumed.
    """
    remaining = list(wanted)
    grabbed = []
    with VideoSource.open(source_spec) as source:
        for frame_index, timestamp, frame in source:
            if not remaining:
                break
            if frame_index == remaining[0]:
                grabbed.append((frame_index, timestamp, frame.copy()))
                remaining.pop(0)
    if remaining:
        print(
            f"warning: the clip ended before candidate frame(s) "
            f"{', '.join(str(n) for n in remaining)}",
            file=sys.stderr,
        )
    return grabbed


# --- main -------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    require_git_ignored(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    gate_configs = {gate.name: gate for gate in config.gates}
    if args.gate not in gate_configs:
        raise SystemExit(
            f"config {args.config} has no gate named {args.gate!r}; it has: "
            f"{', '.join(sorted(gate_configs)) or '(none)'}"
        )

    source_spec = config.source
    if not Path(source_spec).is_absolute() and not source_spec.isdigit():
        candidate = ROOT / source_spec
        if candidate.is_file():
            source_spec = str(candidate)

    with VideoSource.open(source_spec) as probe:
        width, height = probe.width, probe.height
        fps = probe.fps
        frame_count = probe.frame_count
    gate = gate_configs[args.gate].to_gate(width, height)

    start_frame = args.start_frame
    end_frame = args.end_frame
    if end_frame is None:
        end_frame = last_decodable_frame(source_spec)
        if end_frame < 0:
            raise SystemExit(f"{config.source} decoded no frames at all")
        print(
            f"no --end-frame given: decoded the clip and found frames "
            f"0-{end_frame} (its container claims {frame_count})"
        )
    if end_frame < start_frame:
        raise SystemExit(f"--end-frame {end_frame} is before --start-frame {start_frame}")

    clip_stem = Path(config.source).stem
    prefix = f"{clip_stem}_{gate.name}"
    print(
        f"clip {config.source}  {width}x{height} @ {fps} fps, "
        f"{frame_count} frames"
    )
    print(f"gate {gate.name}: {gate.start} -> {gate.end} px")
    print(f"window frames {start_frame}-{end_frame} inclusive")

    with VideoSource.open(source_spec) as source:
        scan = build_slitscan(
            source,
            gate,
            range(start_frame, end_frame + 1),
            thickness_px=args.thickness,
            samples=args.samples,
        )
    print(f"slit-scan {scan.shape[0]} rows x {scan.shape[1]} samples")

    written: list[Path] = []

    full_path = out_dir / f"{prefix}_slitscan.png"
    full_banner = (
        f"{clip_stem}  gate {gate.name}  frames {start_frame}-{end_frame}  "
        f"{fps} fps  (1 row = 1 frame)"
    )
    cv2.imwrite(
        str(full_path),
        with_frame_axis(
            scan,
            start_frame,
            row_scale=1,
            label_every=max(1, int(round(fps or 30))),
            tick_every=max(1, int(round((fps or 30) / 6))),
            banner=full_banner,
        ),
    )
    written.append(full_path)

    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    windows = tile_windows(
        start_frame, end_frame, args.tile_frames, args.tile_overlap
    )
    for tile_start, tile_end in windows:
        band = scan[tile_start - start_frame:tile_end - start_frame + 1]
        banner = (
            f"{clip_stem}  gate {gate.name}  frames {tile_start}-{tile_end}  "
            f"(overlap {args.tile_overlap})"
        )
        tile_path = tiles_dir / f"{prefix}_f{tile_start:05d}-{tile_end:05d}.png"
        cv2.imwrite(
            str(tile_path),
            with_frame_axis(
                band,
                tile_start,
                row_scale=args.row_scale,
                label_every=10,
                tick_every=5,
                banner=banner,
            ),
        )
        written.append(tile_path)
    print(f"{len(windows)} tiles of {args.tile_frames} frames, "
          f"{args.tile_overlap}-frame overlap -> {tiles_dir}")

    candidates = parse_candidates(args.candidates)
    if candidates:
        out_of_window = [
            n for n in candidates if not (start_frame <= n <= end_frame)
        ]
        if out_of_window:
            raise SystemExit(
                f"candidate frame(s) {out_of_window} lie outside the window "
                f"{start_frame}-{end_frame}"
            )
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        grabbed = grab_frames(source_spec, candidates)
        panels = [
            (index, timestamp, draw_gate(frame, gate))
            for index, timestamp, frame in grabbed
        ]
        per_sheet = args.sheet_cols * args.sheet_rows
        sheets = 0
        for offset in range(0, len(panels), per_sheet):
            chunk = panels[offset:offset + per_sheet]
            sheets += 1
            sheet_path = frames_dir / f"{prefix}_sheet{sheets:02d}.png"
            cv2.imwrite(
                str(sheet_path),
                contact_sheet(
                    chunk, args.sheet_cols, args.sheet_rows, args.thumb_width
                ),
            )
            written.append(sheet_path)
        print(f"{len(panels)} candidate frames on {sheets} contact sheet(s) "
              f"-> {frames_dir}")

    print(f"\n{len(written)} image(s) written under {out_dir}")
    print(
        "These are review material, not labels. Read them under "
        "data/groundtruth/PROTOCOL.md and write the label file by hand; "
        "never commit anything in this directory."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
