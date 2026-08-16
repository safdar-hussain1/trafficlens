# Attribution for the clips served from this directory

Both clips on this site are third-party footage under a Creative Commons
**Attribution** licence. Attribution has to accompany the work as it is
distributed, and so does an indication that the work was changed — and these
files ARE changed: they are excerpts, re-encoded, and in one case scaled down.
The originals live elsewhere under different filenames, so nothing about the
served `.mp4` connects it to its source on its own. That is what this file is
for. It is copied into the published site beside the clips it describes, so a
visitor who can reach the video can reach the credit.

---

## `motorway-a40.mp4`

| | |
|---|---|
| **Title** | Motorway A40 – on bridge above the traffic |
| **Author** | the *Sounds of Changes* project, as credited on the file page |
| **Source** | https://commons.wikimedia.org/wiki/File:Motorway_A40_-_on_bridge_above_the_traffic.webm |
| **File fetched** | https://upload.wikimedia.org/wikipedia/commons/7/75/Motorway_A40_-_on_bridge_above_the_traffic.webm |
| **Licence** | Creative Commons Attribution 3.0 — https://creativecommons.org/licenses/by/3.0/ |

**Changes made.** The original is a 24.6 s VP9/WebM at 1280×720 with an Opus
audio track. The file served here is a 20.0 s excerpt, scaled to 960×540,
re-encoded to H.264 in an MP4 container, with the audio removed.

## `street-aisle.mp4`

| | |
|---|---|
| **Title** | person-bicycle-car-detection (Intel IoT DevKit `sample-videos`) |
| **Author** | Intel Corporation, IoT DevKit sample-videos collection |
| **Source** | https://github.com/intel-iot-devkit/sample-videos |
| **File fetched** | https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4 |
| **Licence** | Creative Commons Attribution 4.0 — https://creativecommons.org/licenses/by/4.0/ |

**Changes made.** The original is 53.9 s at 768×432, 12 fps, with an AAC audio
track. The file served here is a 30.0 s excerpt at the same resolution and
frame rate, re-encoded, with the audio removed. It is renamed `street-aisle` to
describe what it shows rather than what a detector is expected to find in it.

---

## Why the files here are not the originals

`data/samples/` is fetched rather than committed (see `src/trafficlens/samples.py`,
which holds the URLs and the licences above). A published page cannot depend on
a download the visitor has not run, so the site carries its own copies. They are
smaller and shorter than the originals for the same reason any web page's assets
are.

None of this changes what is measured. Gate and calibration coordinates are
normalized, so they are resolution-independent, and the detector letterboxes
every frame to its own input size before inference. The published demo is
nevertheless **not byte-identical footage to what the command-line benchmarks
ran on**, and no accuracy number on this site is derived from these copies.

## Nothing here is claimed as this project's work

The clips are the only third-party media on the site. Everything else served
from this directory's siblings — the detector graph, the runtime, the fonts —
carries its own notice: see `models/MODEL_CARD.md` and `fonts/NOTICE.md`.
