"""Frame annotation: boxes, trails, gates, and a live HUD.

Colours are assigned per class (stable across frames) and drawn with
OpenCV only — no extra drawing dependencies.
"""

from __future__ import annotations

import cv2
import numpy as np

from trafficlens.counting import GateCounter
from trafficlens.pipeline import FrameResult

# Distinct, readable on road footage (BGR).
_CLASS_COLORS = [
    (80, 175, 76),    # green
    (219, 152, 52),   # blue
    (60, 76, 231),    # red
    (156, 89, 182),   # purple
    (18, 156, 243),   # orange
    (133, 160, 22),   # teal
    (196, 121, 234),  # pink
]


def class_color(class_name: str) -> tuple[int, int, int]:
    return _CLASS_COLORS[hash(class_name) % len(_CLASS_COLORS)]


def draw_frame(
    frame: np.ndarray,
    result: FrameResult,
    counters: list[GateCounter],
    speed_unit: str = "kmh",
    speed_limit: float | None = None,
    show_trails: bool = True,
) -> np.ndarray:
    """Draw tracks, gates, and HUD onto a copy of ``frame``."""
    img = frame.copy()
    unit_label = "km/h" if speed_unit == "kmh" else "mph"

    for counter in counters:
        a = tuple(int(v) for v in counter.gate.start)
        b = tuple(int(v) for v in counter.gate.end)
        cv2.line(img, a, b, (0, 215, 255), 2, cv2.LINE_AA)
        for p in (a, b):
            cv2.circle(img, p, 5, (0, 215, 255), -1, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        _label(img, f"{counter.gate.name}: {counter.total}", (mid[0] - 40, mid[1] - 10),
               bg=(0, 215, 255), fg=(20, 20, 20))

    for tv in result.tracks:
        x1, y1, x2, y2 = (int(v) for v in tv.box)
        color = class_color(tv.class_name)
        over_limit = speed_limit is not None and tv.speed is not None and tv.speed > speed_limit
        if over_limit:
            color = (40, 40, 230)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        text = f"#{tv.track_id} {tv.class_name}"
        if tv.speed is not None:
            text += f" {tv.speed:.0f} {unit_label}"
        _label(img, text, (x1, max(18, y1 - 6)), bg=color)

        if show_trails and len(tv.trail) > 1:
            pts = np.array([(int(x), int(y)) for x, y in tv.trail], dtype=np.int32)
            cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
        ax, ay = int(tv.anchor[0]), int(tv.anchor[1])
        cv2.circle(img, (ax, ay), 3, color, -1, cv2.LINE_AA)

    _hud(img, result, counters, unit_label)
    return img


def _label(img, text, org, bg=(60, 60, 60), fg=(255, 255, 255)) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 6), (x + tw + 4, y + 4), bg, -1)
    cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, fg, 1, cv2.LINE_AA)


def _hud(img, result: FrameResult, counters: list[GateCounter], unit_label: str) -> None:
    lines = [f"t={result.timestamp:6.1f}s  frame {result.frame_index}  {result.process_ms:.0f} ms"]
    for counter in counters:
        per_dir = counter.total_by_direction()
        dirs = "  ".join(f"{d}: {n}" for d, n in sorted(per_dir.items()))
        lines.append(f"{counter.gate.name}  total {counter.total}  {dirs}")
    pad, lh = 8, 22
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0] for t in lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width + 2 * pad, 8 + lh * len(lines) + pad), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8 + pad, 8 + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (240, 240, 240), 1, cv2.LINE_AA)
