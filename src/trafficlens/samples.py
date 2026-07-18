"""Sample footage fetcher.

Gives a brand-new user something to point TrafficLens at within a
minute of cloning, without committing large binaries to this repo.
All clips are redistributable Creative Commons material (see the
attribution section in the README):

* ``motorway-a40.webm`` — busy German motorway from a bridge;
  CC BY 3.0, "Sounds of Changes" via Wikimedia Commons.
* the two Intel IoT DevKit clips — CC BY 4.0.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

_INTEL = "https://github.com/intel-iot-devkit/sample-videos/raw/master"

SAMPLES = {
    "motorway-a40.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/7/75/"
        "Motorway_A40_-_on_bridge_above_the_traffic.webm"
    ),
    "car-detection.mp4": f"{_INTEL}/car-detection.mp4",
    "person-bicycle-car-detection.mp4": f"{_INTEL}/person-bicycle-car-detection.mp4",
}


def fetch(name: str, dest_dir: Path) -> Path:
    """Download one sample (skipping if already present) and return its path."""
    if name not in SAMPLES:
        raise ValueError(f"unknown sample {name!r}; available: {sorted(SAMPLES)}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / name
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(SAMPLES[name], tmp)
    if tmp.stat().st_size < 10_000:
        tmp.unlink()
        raise RuntimeError(f"download of {name} looks truncated — try again")
    tmp.rename(path)
    return path


def fetch_all(dest_dir: Path) -> list[Path]:
    return [fetch(name, dest_dir) for name in SAMPLES]
