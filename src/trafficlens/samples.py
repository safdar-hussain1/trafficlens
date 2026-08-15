"""Sample footage fetcher: three redistributable clips, downloaded on
demand into ``data/samples/``.

Large binaries are never committed to this repository, so a fresh clone has
nothing to point the pipeline at. ``trafficlens fetch-samples`` closes that
gap in one command. Every clip below is Creative Commons licensed and
redistributable; the licence and source of each is recorded HERE, next to
the URL it is fetched from, so the attribution cannot drift away from the
thing it attributes:

- ``motorway-a40.webm`` -- "Motorway A40 - on bridge above the traffic",
  a busy German motorway filmed from an overbridge, by the *Sounds of
  Changes* project via Wikimedia Commons. **CC BY 3.0.**
- ``car-detection.mp4`` -- road traffic sample clip from Intel's IoT DevKit
  ``sample-videos`` collection. **CC BY 4.0.**
- ``person-bicycle-car-detection.mp4`` -- mixed pedestrian/cyclist/vehicle
  sample clip from the same Intel IoT DevKit collection. **CC BY 4.0.**

Download policy: a clip already present on disk is never re-downloaded, a
download lands on a ``.part`` file that is only renamed into place once it
is verified to be non-trivially sized (``MIN_BYTES``), and a truncated or
error-page download is deleted rather than left behind to fail confusingly
later. The smallest of the three clips is ~2.8 MB, so the 100 KB floor has
a ~28x margin over the real thing while still catching every HTML error
page a CDN might hand back with a 200.

This module imports nothing beyond the standard library: fetching samples
must work on a bare core install, before cv2 or a model is involved.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

_INTEL = "https://github.com/intel-iot-devkit/sample-videos/raw/master"

SAMPLES: dict[str, str] = {
    "motorway-a40.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/7/75/"
        "Motorway_A40_-_on_bridge_above_the_traffic.webm"
    ),
    "car-detection.mp4": f"{_INTEL}/car-detection.mp4",
    "person-bicycle-car-detection.mp4": f"{_INTEL}/person-bicycle-car-detection.mp4",
}

LICENCES: dict[str, str] = {
    "motorway-a40.webm": (
        "CC BY 3.0 -- 'Motorway A40 - on bridge above the traffic', "
        "Sounds of Changes project, via Wikimedia Commons"
    ),
    "car-detection.mp4": (
        "CC BY 4.0 -- Intel IoT DevKit sample-videos collection"
    ),
    "person-bicycle-car-detection.mp4": (
        "CC BY 4.0 -- Intel IoT DevKit sample-videos collection"
    ),
}

# Smallest size, in bytes, a completed download may have. See the module
# docstring for how this number was chosen.
MIN_BYTES = 100_000

# Where the shipped configs expect the clips to be, relative to the
# repository root.
DEFAULT_DEST = Path("data") / "samples"


class SampleError(RuntimeError):
    """A sample clip could not be fetched. The message always names the
    clip and says what to do about it."""


def download(url: str, path: Path) -> None:
    """Fetch ``url`` to ``path``. Split out as a module-level function so
    the fetch policy above can be tested without any network access."""
    urllib.request.urlretrieve(url, path)


def fetch(name: str, dest_dir) -> tuple[Path, bool]:
    """Ensure sample ``name`` exists under ``dest_dir``.

    Returns ``(path, downloaded)`` -- ``downloaded`` is False when the file
    was already there and nothing was fetched. Raises ``SampleError`` for
    an unknown name, a failed transfer, or a download that arrives too
    small to be real footage (in which case the partial file is removed,
    never left in place).
    """
    if name not in SAMPLES:
        raise SampleError(
            f"unknown sample {name!r}; available samples are "
            f"{', '.join(sorted(SAMPLES))}"
        )

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / name
    if path.is_file() and path.stat().st_size >= MIN_BYTES:
        return path, False

    partial = path.with_name(path.name + ".part")
    try:
        download(SAMPLES[name], partial)
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise SampleError(
            f"could not download {name} from {SAMPLES[name]}: {error}"
        ) from error

    size = partial.stat().st_size if partial.is_file() else 0
    if size < MIN_BYTES:
        partial.unlink(missing_ok=True)
        raise SampleError(
            f"download of {name} arrived as {size} bytes, below the "
            f"{MIN_BYTES}-byte floor for a real clip -- the server most "
            f"likely returned an error page. Try again, or download "
            f"{SAMPLES[name]} by hand into {dest}."
        )

    partial.replace(path)
    return path, True


def fetch_all(dest_dir=DEFAULT_DEST, progress=None) -> list[tuple[Path, bool]]:
    """Fetch every sample into ``dest_dir``, skipping those already there.

    ``progress`` -- when given -- is called ``progress(name, path,
    downloaded)`` after each clip, so a caller can report as it goes rather
    than in one silent batch.
    """
    results = []
    for name in SAMPLES:
        path, downloaded = fetch(name, dest_dir)
        results.append((path, downloaded))
        if progress is not None:
            progress(name, path, downloaded)
    return results
