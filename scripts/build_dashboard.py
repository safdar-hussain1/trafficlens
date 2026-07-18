"""Bake benchmark + replay data into the static dashboard.

docs/index.html is generated from docs/_template.html by replacing the
/*__DATA__*/ placeholder with the real JSON exported by the benchmark
run. Baking (instead of fetch()) means the page works from file://,
screenshots cleanly, and can never drift from the exported results.

Usage:
  python scripts/build_dashboard.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "docs" / "_template.html"
OUTPUT = ROOT / "docs" / "index.html"
PLACEHOLDER = "/*__DATA__*/"


def main() -> None:
    benchmark = json.loads((ROOT / "reports" / "benchmark.json").read_text())
    replay = json.loads((ROOT / "reports" / "replay_motorway.json").read_text())
    benchmark["test_count"] = sum(
        len(re.findall(r"^\s*def test_", p.read_text(), re.M))
        for p in (ROOT / "tests").glob("test_*.py")
    )

    # Thin the replay to keep the page light: cap the number of frames and
    # drop sub-pixel precision (the canvas is ~900 px wide).
    frames = replay["frames"]
    payload = {
        "benchmark": benchmark,
        "replay": {
            "meta": replay["meta"],
            "gates": replay["gates"],
            "summary": replay["summary"],
            "frames": frames,
        },
    }
    blob = json.dumps(payload, separators=(",", ":"))

    template = TEMPLATE.read_text()
    if PLACEHOLDER not in template:
        sys.exit(f"placeholder {PLACEHOLDER} missing from {TEMPLATE}")
    OUTPUT.write_text(template.replace(PLACEHOLDER, f"window.TL_DATA = {blob};"))
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT} ({size_kb:.0f} kB, {len(frames)} replay frames)")


if __name__ == "__main__":
    main()
