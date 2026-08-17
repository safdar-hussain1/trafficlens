# TrafficLens

Traffic analytics from one camera: it counts what crosses a gate, in which
direction, of which class, and — where the camera is calibrated — how fast.
The same detector and the same tracker run two ways: as a Python package and
CLI on a server, and as a byte-parity TypeScript engine in a browser tab, where
inference happens on the visitor's own GPU and no frame is ever uploaded.

**Live demo: <https://safdar-hussain1.github.io/trafficlens/>** — load one of the
bundled clips or drop in your own file, drag the gate where you want it, and watch
the count. Everything runs locally in the tab.

Every number below is read from a JSON file under [`reports/`](reports/), and
`tests/test_docs_numbers.py` fails if this file and those reports disagree.

---

## Contents

- [What it does](#what-it-does)
- [Install](#install)
- [Commands](#commands)
- [Measured results](#measured-results)
- [The honest negatives](#the-honest-negatives)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Reproducing every number](#reproducing-every-number)
- [Licence and attribution](#licence-and-attribution)

---

## What it does

- **Counts crossings at a drawn gate.** A gate is a line segment, not an infinite
  line: a vehicle crossing the extension of the gate — the next carriageway over —
  is not counted. Each track is counted once, by class, by direction.
- **Names the direction geometrically.** The side of the line a track ends on
  decides the label, so a config states which side is which rather than relying on
  a default that may be inverted for the scene.
- **Flags wrong-way movement and stopped vehicles.** A crossing against a gate's
  declared direction raises an incident; stopped-vehicle detection needs a
  calibrated camera and does not fire without one.
- **Estimates speed only where a camera is calibrated.** Speed comes from a
  road-plane homography, in metres, from a least-squares slope per axis. Where the
  calibration is missing or unusable, the engine returns nothing rather than a
  number it cannot stand behind.
- **Runs the same engine in the browser.** The TypeScript port shares the
  constants file, the preprocessing rules and the decision boundaries, and a
  committed fixture set asserts the two agree.
- **Benchmarks itself against the standard failure modes.** Two naive trackers and
  two naive counting rules are implemented faithfully and scored side by side with
  the engine on hand-labelled ground truth, undegraded and under four separate
  degradations.

---

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/safdar-hussain1/trafficlens.git
cd trafficlens
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[detect,onnx,dev]"
```

The extras are separable on purpose: `detect` pulls in `ultralytics` and `torch`
for the server detector, `onnx` pulls in `onnxruntime` for the ONNX adapter, and
`dev` pulls in the test and figure dependencies. The core package needs none of
them.

Then fetch the Creative Commons sample clips — three files, about 17 MB, into
`data/samples/`:

```bash
.venv/bin/trafficlens fetch-samples
```

Every command below also works from a checkout without installing, which is how
the benchmarks in this repository are driven:

```bash
PYTHONPATH=src .venv/bin/python -m trafficlens.cli --help
```

---

## Commands

**Source:** `src/trafficlens/cli.py`, `--help` on each command.

| command | what it does |
|---|---|
| `trafficlens run` | Analyse a source and report what crossed each gate. Writes `events.csv`, `summary.json` and `session.json` with `--export-dir`. |
| `trafficlens fetch-samples` | Download the sample clips, with their licences, into `data/samples/`. |
| `trafficlens calibrate` | Explain how to survey a camera, and check a config's calibration block against its own held-out points. |
| `trafficlens export-model` | Export a YOLO11 checkpoint to the ONNX graph the browser engine runs. |
| `trafficlens serve` | A placeholder. It prints what it will do and exits; the published site is the browser surface today. |
| `trafficlens bench` | A placeholder. It points at `scripts/`, which is where the benchmarks actually live. |

Count what crosses both carriageways of the motorway clip:

```bash
PYTHONPATH=src .venv/bin/python -m trafficlens.cli run \
  --config configs/motorway.yaml \
  --export-dir out/motorway
```

Draw your own gate, in coordinates normalized to the frame, and write an
annotated video:

```bash
PYTHONPATH=src .venv/bin/python -m trafficlens.cli run \
  --config configs/street.yaml \
  --gate kerbside,0.10,0.62,0.90,0.62 \
  --save-video out/street-annotated.mp4
```

Check a calibration before trusting a speed:

```bash
PYTHONPATH=src .venv/bin/python -m trafficlens.cli calibrate \
  --config configs/webcam.yaml
```

Export the browser graph:

```bash
PYTHONPATH=src .venv/bin/python -m trafficlens.cli export-model \
  --weights yolo11n.pt --imgsz 480 --out web/public/models/yolo11n-480.onnx
```

---

## Measured results

### How to read them

Two things apply to every accuracy figure here, and both make the numbers weaker
than they look.

**They are upper bounds.** The labelled gate sits in the near field and was chosen
for label reliability — the largest, best-separated, least-foreshortened traffic in
the frame. That is the easiest case for association as well as for labelling. The
same engine on the far carriageway, at distance, or in a queue scores lower, and
nothing here says by how much.

**The benchmark's resolution is one event.** There are 17 labelled crossings, so a
single added or removed crossing moves precision by 0.0588 and F1 by about 0.027.
Every counting figure below therefore carries **±0.027 (one event)**, and a
difference between two methods smaller than that is one event rather than a quality
gap. Where this file compares methods it says which side of that line the comparison
falls on: the tracker comparison, for instance, publishes both the 19 levels where
the three trackers differ at all and the 12 where they differ by **more than one
event**, and the second number is the one that survives the instrument.

### Counting accuracy, all nine methods

**Source:** `reports/counting_accuracy.json`.
**Protocol:** `yolo11s.pt` at confidence 0.25, imgsz 640, classes car/truck/bus/motorcycle,
over frames 0 to 734 of `motorway-a40.webm` at 30 fps, gate `inbound` (pixel 76.8, 576.0
to 588.8, 576.0), scored against 17 hand-labelled crossings — 13 car, 4 truck, all
toward the camera, 7 adjudicated certain and 10 probable. A prediction matches a
label when it lands in the asymmetric window [label - 1, label + 4], one-to-one,
greedy nearest-frame-first. The asymmetry is not a tolerance chosen after seeing
output: the labelling record fixes label precision at +0/-4 frames, because a
vehicle's shadow reaches the gate band before its tyres do, and the engine fires on
the box's bottom-centre anchor. Every method is driven from one cached detection
stream, so the comparison isolates the tracker and the counting rule and never
detector variance.

| method | precision | recall | F1 | predicted | matched | false positives | missed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `engine+gate` | 0.889 | 0.941 | 0.914 | 18 | 16 | 2 | 1 |
| `centroid+gate` | 0.941 | 0.941 | 0.941 | 17 | 16 | 1 | 1 |
| `greedy-iou+gate` | 0.941 | 0.941 | 0.941 | 17 | 16 | 1 | 1 |
| `engine+band` | 0.105 | 0.118 | 0.111 | 19 | 2 | 17 | 15 |
| `centroid+band` | 0.056 | 0.059 | 0.057 | 18 | 1 | 17 | 16 |
| `greedy-iou+band` | 0.056 | 0.059 | 0.057 | 18 | 1 | 17 | 16 |
| `engine+per-frame` | 0.030 | 1.000 | 0.058 | 565 | 17 | 548 | 0 |
| `centroid+per-frame` | 0.031 | 1.000 | 0.060 | 554 | 17 | 537 | 0 |
| `greedy-iou+per-frame` | 0.031 | 1.000 | 0.060 | 554 | 17 | 537 | 0 |

`centroid` and `greedy-iou` are the two baseline trackers; `band` and `per-frame`
are the two baseline counting rules. They are standard failure modes of naive
traffic counting, implemented faithfully so the benchmark can price them. The
engine's own tracker is `engine`, and on this clip it scores **below** both
baselines: one added false positive, which is exactly the one-event resolution
above.

Class and direction, for `engine+gate`: 14 of 16 matched crossings carry the
label's own class, a rate of 0.875, with one truck read as a bus and one as a car.
Matching is class-blind, so a class error is charged once rather than twice.
Direction has no discriminating power on this clip — every label and every
prediction is toward the camera — and the report says so itself.

### The certain-only bracket

**Source:** `reports/counting_accuracy.json`.
**Protocol:** the same run, scored over the 7 labels the labeller adjudicated as
certain. The 10 probable labels become ignore regions — the MOTChallenge and COCO
`iscrowd` treatment, transposed from space to the timeline: matching runs jointly
over all 17 labels, then every pair matched to a probable label is removed from
both sides, so a prediction inside an interval nobody could adjudicate is neither
credited nor charged. 10 predictions move to the ignore set here.

| method | precision | recall | F1 | predicted | matched | false positives | missed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `engine+gate` | 0.750 | 0.857 | 0.800 | 8 | 6 | 2 | 1 |
| `centroid+gate` | 0.857 | 0.857 | 0.857 | 7 | 6 | 1 | 1 |
| `greedy-iou+gate` | 0.857 | 0.857 | 0.857 | 7 | 6 | 1 | 1 |
| `engine+band` | 0.056 | 0.143 | 0.080 | 18 | 1 | 17 | 6 |
| `centroid+band` | 0.000 | 0.000 | 0.000 | 17 | 0 | 17 | 7 |
| `greedy-iou+band` | 0.000 | 0.000 | 0.000 | 17 | 0 | 17 | 7 |
| `engine+per-frame` | 0.013 | 1.000 | 0.025 | 555 | 7 | 548 | 0 |
| `centroid+per-frame` | 0.013 | 1.000 | 0.025 | 544 | 7 | 537 | 0 |
| `greedy-iou+per-frame` | 0.013 | 1.000 | 0.025 | 544 | 7 | 537 | 0 |

**Read 0.800 and 0.914 together: they bracket the engine's accuracy.** Three things
travel with them. The gap is dilution arithmetic and not difficulty — both subsets
carry the same errors against a smaller numerator. Certain-only precision under
ignore semantics is itself an upper bound within the certain subset, because a
genuine phantom landing inside a probable label's window is absorbed into the
ignore set, and the ignore mass here is 10 of 17 labels. And deleting the probable
labels outright instead of ignoring them scores `engine+gate` at F1 0.480 — a
figure that means nothing on its own and is published only to show why ignore
regions are necessary, since most of the false alarms it reports are manufactured
by the subsetting.

### The counting rule matters more than the tracker

**Source:** `reports/counting_accuracy.json`.
**Protocol:** the band rule's half-width swept, engine tracker, same stream, same
window as the counting table above. Miss rate and phantom rate are kept as
separate series because in general they trade.

| band half-width, px | predicted | on the right frame | missed | false positives | miss rate | phantom rate |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 18 | 11 | 6 | 7 | 0.3529 | 0.4118 |
| 10 | 18 | 2 | 15 | 16 | 0.8824 | 0.9412 |
| 20 | 19 | 2 | 15 | 17 | 0.8824 | 1.0000 |
| 30 | 19 | 2 | 15 | 17 | 0.8824 | 1.0000 |
| 45 | 20 | 2 | 15 | 18 | 0.8824 | 1.0588 |
| 60 | 20 | 2 | 15 | 18 | 0.8824 | 1.0588 |
| 90 | 21 | 2 | 15 | 19 | 0.8824 | 1.1176 |

The two rates were expected to trade and they do not: they rise together. The
measured gate approach is 1.142 px per frame, at which no band in this sweep is
ever stepped over, so the classic step-over miss cannot occur. What the band rule
gets wrong here is the crossing **frame**, and a mistimed crossing costs a miss and
a phantom at once. This is why the benchmark scores crossings one by one: on totals
alone a rule that predicts 19 crossings against 17 real ones looks almost right
while landing 2 of them on the correct frame.

### What the engine's tracker costs

**Source:** `reports/counting_accuracy.json`.
**Protocol:** one timing measurement per method over the 735 decoded frames,
covering the tracker and the counting rule only. Detections are read from a cache,
so the detector's cost — much the largest per-frame cost in a real session — is
excluded and is identical for every method by construction.

| method | seconds over 735 frames | ms per frame |
|---|---:|---:|
| `engine+gate` | 0.4892 | 0.6656 |
| `centroid+gate` | 0.0225 | 0.0306 |
| `greedy-iou+gate` | 0.0247 | 0.0336 |
| `engine+band` | 0.4797 | 0.6526 |
| `centroid+band` | 0.0202 | 0.0275 |
| `greedy-iou+band` | 0.0229 | 0.0312 |
| `engine+per-frame` | 0.4842 | 0.6587 |
| `centroid+per-frame` | 0.0218 | 0.0297 |
| `greedy-iou+per-frame` | 0.0246 | 0.0335 |

The engine's Kalman filter and Hungarian assignment cost about 21.7 times a
baseline tracker on the same stream.

### Under degradation

**Source:** `reports/robustness.json`.
**Protocol:** the same cached detection stream spoiled four independent ways —
frame rate, whole dropped frames, per-detection dropout, and per-corner box jitter
— seeded 20260815 and byte-identical across runs. The script never constructs a
detector: a sweep whose undegraded row came from a different detector run could not
reduce to `counting_accuracy.json`, so it refuses to run without the cache. The
match window widens on its late side only, by the largest inter-sample gap the
retained pattern actually realises, because ground truth is indexed in original
30 fps frames and a method sampling every Gth frame cannot report a crossing before
its next sample. Every protocol's identity level reproduces the counting table
above exactly, through the general code path with no short-circuit branch.

| degradation | level | engine+gate | centroid+gate | greedy-iou+gate | spread | engine predicted |
|---|---|---:|---:|---:|---:|---:|
| frame rate | 30 fps | 0.914 | 0.941 | 0.941 | 0.0269 | 18 |
| frame rate | 25 fps | 0.909 | 0.914 | 0.941 | 0.0321 | 16 |
| frame rate | 15 fps | 0.800 | 0.914 | 0.941 | 0.1412 | 13 |
| frame rate | 10 fps | 0.692 | 1.000 | 1.000 | 0.3077 | 9 |
| frame rate | 5 fps | 0.211 | 0.970 | 0.903 | 0.7592 | 2 |
| frame rate | 2 fps | 0.000 | 0.455 | 0.300 | 0.4545 | 0 |
| dropped frames | 0% dropped | 0.914 | 0.941 | 0.941 | 0.0269 | 18 |
| dropped frames | 5% dropped | 0.970 | 1.000 | 1.000 | 0.0303 | 16 |
| dropped frames | 10% dropped | 0.909 | 1.000 | 1.000 | 0.0909 | 16 |
| dropped frames | 20% dropped | 0.938 | 0.971 | 1.000 | 0.0625 | 15 |
| dropped frames | 30% dropped | 0.692 | 0.971 | 1.000 | 0.3077 | 9 |
| detection dropout | p=0.00 | 0.914 | 0.941 | 0.941 | 0.0269 | 18 |
| detection dropout | p=0.05 | 0.914 | 0.941 | 0.941 | 0.0269 | 18 |
| detection dropout | p=0.10 | 0.914 | 0.941 | 0.941 | 0.0269 | 18 |
| detection dropout | p=0.20 | 0.882 | 0.882 | 0.882 | 0.0000 | 17 |
| detection dropout | p=0.30 | 0.882 | 0.882 | 0.882 | 0.0000 | 17 |
| box jitter | sigma=0 px | 0.914 | 0.941 | 0.941 | 0.0269 | 18 |
| box jitter | sigma=1 px | 0.857 | 0.882 | 0.882 | 0.0252 | 18 |
| box jitter | sigma=2 px | 0.643 | 0.882 | 0.882 | 0.2395 | 11 |
| box jitter | sigma=4 px | 0.200 | 0.647 | 0.647 | 0.4471 | 3 |
| box jitter | sigma=8 px | 0.105 | 0.235 | 0.278 | 0.1725 | 2 |

The three trackers differ at 19 of the 21 levels, and **the engine's tracker scores
lowest at all 19 of them.** It leads at none. The widest spread is 0.7592, at 5 fps.

That 19 includes differences the instrument cannot resolve, so here is the
decomposition against the ±0.027 one-event step. Of the 19 levels where the three
differ, **12 differ by more than one event, and the engine is lowest at all 12** —
that is the claim that survives the resolution, and it is the one to read. Six more
differ by exactly one event, four of them the undegraded identity levels this family
reduces to, and one — box jitter at sigma = 1 px, spread 0.0252 — differs by less
than one event and should be read as a tie.

Degradation is where a motion model is supposed to earn its cost; here it does the
opposite, and the two results — 21.7 times the CPU, lowest at every level the
instrument can separate — agree with each other.

The band rule's step-over mode does come back under decimation: 2 of the 84 swept
(tracker, rate, band) rows emit fewer events than the gate rule fed by the same
tracker over the same stream, the confound-free signature of a band jumped clean
over, and the highest frame rate at which it happens is 5 fps. It happens with the
`greedy-iou` baseline and never with the engine's tracker — because the engine's
association collapses at a *higher* frame rate than the band's step-over threshold,
so on this clip one failure mode pre-empts the other.

### Where the collapse comes from, and where it does not

**Source:** `reports/robustness.json`.
**Protocol:** `Tracker(match_thresh=...)` varied from the shipped 0.8 to 0.3 —
SORT's published IoU threshold, a value with a citation rather than one chosen to
make a point — with everything else identical: same degraded stream, same counting
rule, same track lifecycle, same match window. A gain counts as explanatory above
0.05 F1.

| degradation | identity level, F1 at 0.8 | identity level, F1 at 0.3 | largest gain from 0.3 | at level |
|---|---:|---:|---:|---|
| frame rate | 0.9143 | 0.9143 | 0.6020 | 5 fps |
| dropped frames | 0.9143 | 0.9143 | 0.2791 | 30% dropped |
| detection dropout | 0.9143 | 0.9143 | 0.0000 | p=0.00 |
| box jitter | 0.9143 | 0.9143 | 0.3652 | sigma=4 px |

Loosening the association floor recovers up to 0.6020 F1 the moment the input
degrades, and that is the mechanism behind the frame-rate, dropped-frame and jitter
collapses. It recovers **nothing** on undegraded footage: the two floors tie exactly
at all four identity levels.

Read plainly, that leaves the shipped 0.8 with almost no case. **Nothing in this
sweep shows it to be measurably better than SORT's 0.3 anywhere except under
detection dropout.** It ties clean, loses on three of the four degradation families,
and wins on one — by 0.0252, at p = 0.20 and above. Twelve of the 21 levels record a
positive gain for the looser floor. The tie undegraded is not an argument for keeping
0.8; the dropout cost is the only argument there is.

That dropout column is also the second finding here. The largest gain the looser
floor produces under detection dropout is 0.0000, so the ablation that explains the
frame-rate, dropped-frame and jitter collapses explains none of this one:
**whatever the engine loses under detection dropout is a second, separate fault and
it is not diagnosed.** The constant ships unchanged and documented rather than
half-fixed — picking a new floor is separate work with its own baseline, and one
clip and two floors is not the evidence for a value.

### The detector box noise the jitter sweep is calibrated against

**Source:** `reports/detection_noise.json`.
**Protocol:** residuals of each track's box against a centred 5-frame median filter
of its own trajectory, over 735 frames, 41 of 48 tracks contributing 4032 samples.
Association is by the greedy-IoU baseline, whose track box is the last *observed*
detection, so these are raw detector boxes rather than Kalman output.

| quantity | std, px | mean absolute, px | p95 absolute, px |
|---|---:|---:|---:|
| box width | 1.0294 | 0.3623 | 1.5660 |
| box height | 0.5232 | 0.2251 | 1.0369 |
| centre x | 0.3344 | 0.0499 | 0.1407 |
| centre y | 0.0971 | 0.0188 | 0.0897 |

**This is a proxy, not ground truth**, and the report says so in its own caveat
field: the filter's smoothing, identity error in the association and genuine
sub-filter motion are all inside these numbers. The distribution is heavy-tailed —
on both centres the standard deviation exceeds the p95, because a few large
excursions dominate the variance while the bulk sits near zero — so read the p95
for the typical case and the standard deviation for the tail, and neither as a
measurement of the detector. The median box is 66.78 px wide, so width jitter is
about 1.5% of a box.

**Source:** `reports/robustness.json`.
**Protocol:** the residuals above converted into the sweep's own knob. Width and
height are differences of two independent corners, so per-corner sigma is
std / sqrt(2); a centre is their mean, so it is std * sqrt(2); a p95 of an absolute
residual converts through the standard normal's 1.959964.

| quantity | per-corner sigma from std, px | from p95, px |
|---|---:|---:|
| box width | 0.7279 | 0.5650 |
| box height | 0.3699 | 0.3741 |
| centre x | 0.4729 | 0.1015 |
| centre y | 0.1374 | 0.0648 |

So the measured level is 0.0648 to 0.7279 px per corner, and the top of the jitter
sweep — sigma = 8 px — is 10.99 to 123.54 times it. Everything to the right of
sigma = 1 px in the degradation table is visibly extrapolation, and the measured
band is drawn on every panel of `reports/figures/robustness_box_jitter.png` so that
this is impossible to miss.

### Identity behaviour at the gate

**Source:** `reports/tracking.json`.
**Protocol:** the same cached stream and the same gate rule; only the tracker
changes. Fragmentation ratio is distinct predicted identities that reach the gate
region, divided by the 17 labelled vehicles — 1.0 is one identity per vehicle, and
a value below 1.0 is an error too, meaning identities that never reached the gate at
all. The gate region is a band of half-width 20 px about the gate segment, the same
constant the band counting rule uses; the sweep over 5 to 60 px is published in the
report because one of its answers depends on that width.

| tracker | identities at the gate | fragmentation ratio | class consistency | confusions |
|---|---:|---:|---:|---|
| engine | 19 | 1.1176 | 0.875 | truck -> bus x1, truck -> car x1 |
| centroid | 18 | 1.0588 | 0.938 | truck -> car x1 |
| greedy-iou | 18 | 1.0588 | 0.938 | truck -> car x1 |

This is a second, independent measurement pointing the same way as the crossing
scores: under degradation the widest fragmentation spread is 1.8235, and the engine
sits furthest from one identity per vehicle at 18 of the 19 levels where the three
differ. The two metrics agree at 18 of the 19 levels where crossing F1 separates the
trackers and disagree at one, at 15 fps, where crossings are lost with almost no
identity signature — mistiming rather than fragmentation. That disagreement is
reported rather than reconciled: the two are measuring different failures.

**No ID-switch count, IDF1, MOTA or MOTP figure appears anywhere in this project.**
All of them need a frame-by-frame identity label set, this clip has none, and a
proxy derived from the tracker's own output would be the tracker grading its own
identities. The four claims this benchmark refuses to make are quoted verbatim in
[the design card](web/public/DESIGN_CARD.md).

### Speed, tier 1: exact synthetic truth

**Source:** `reports/speed_synthetic.json`.
**Protocol:** a simulated camera 8 m above a motorway at 12 degrees of pitch, 1280
by 720, vehicles scored from 30 to 140 m where the projected box spans about 66 px
down to 15 px — the near end matching the real clip's measured median box. Truth is
the caller's own list of speeds and is pure arithmetic in it; the estimate runs
projection, detection, tracker, plane transfer and a least-squares slope, so the two
share the camera model and nothing else. The road plane is fitted from six surveyed
dash centroids plus a two-point holdout, the same shape of survey a deployment does,
and recovers the camera to 6.2e-06 m. A sample counts as settled once a track has
been observed for a full 2 s speed window.

| true speed, km/h | settled samples | mean error, km/h | max absolute error, km/h | RMSE, km/h | max relative error, % |
|---:|---:|---:|---:|---:|---:|
| 30 | 335 | -0.0031 | 0.028 | 0.0052 | 0.093 |
| 50 | 176 | -0.0137 | 0.040 | 0.0177 | 0.079 |
| 70 | 108 | -0.0349 | 0.084 | 0.0409 | 0.120 |
| 90 | 69 | -0.0704 | 0.138 | 0.0770 | 0.153 |
| 110 | 46 | -0.1212 | 0.206 | 0.1287 | 0.187 |
| 130 | 25 | -0.1901 | 0.254 | 0.1934 | 0.195 |

**Two limitations travel with every figure in this table, and both matter more than
the figures.** Calibration error is excluded *by construction* — the true
road-plane-to-image mapping is a homography of the same camera, so the fit recovers
it to machine precision. And the detector is excluded: the boxes are generated from
a known scene, not detected. These are an upper bound on field accuracy, not a
prediction of it, and a real deployment carries survey error on top, usually the
larger term.

The bar was 0.1 km/h and the full chain misses it above about 70 km/h, up to
0.254 km/h at 130. **The bar was not widened.** The miss is characterised instead:
the residual is signed negative at every band, grows monotonically with speed, and
vanishes to 9.8e-05 km/h on the identical detections with the tracker bypassed —
the signature of a constant-velocity Kalman filter lagging image-space motion that
accelerates as a vehicle approaches. The worst relative error is 0.195%.

A second estimator agrees with the first: time of flight between two gates 90 m
apart, which never fits a slope, never differences successive positions and never
uses a window, lands within 0.077 km/h of the plane estimator against a 1.0 km/h
requirement. The report is explicit that this is **not fully independent** and does
not claim to be — the crossing instants come from the same road plane projecting the
track's position, so position is shared and only displacement is not. It cross-checks
the two estimators against each other, which is what it was wanted for; it is not an
outside witness.

**Source:** `reports/speed_synthetic.json`.
**Protocol:** the same scenes at 1x the measured box-noise sigma, 8 seeds, with the
speed estimator fed the tracker's Kalman-smoothed anchor in one column and the raw
detection's own bottom-centre in the other. Zero vehicles are lost to association at
this level, so the two columns differ by the anchor and nothing else.

| true speed, km/h | Kalman anchor RMSE, km/h | raw detection anchor RMSE, km/h | better |
|---:|---:|---:|---|
| 30 | 0.1932 | 0.1935 | Kalman |
| 50 | 0.1891 | 0.1949 | Kalman |
| 70 | 0.1495 | 0.1546 | Kalman |
| 90 | 0.2222 | 0.2048 | raw detection |
| 110 | 0.2033 | 0.1536 | raw detection |
| 130 | 0.2188 | 0.1750 | raw detection |

What the filter's lag buys is not a uniform win. Smoothing helps below 90 km/h and
loses by up to about a quarter above it — in exactly the bands whose zero-noise
residual misses the bar. The tracker was deliberately not retuned: this is a
synthetic instrument with exact truth and no calibration error, and retuning a
shipped filter to it would be fitting the engine to the measuring device.

**Source:** `reports/speed_synthetic.json`.
**Protocol:** the same scenes with per-corner box noise scaled as a multiple of the
measured standard-deviation vector, 8 seeds each. `vehicles lost` counts vehicles the
tracker failed to hold.

| sigma, multiple of measured | vehicles lost | RMSE, km/h | max absolute error, km/h |
|---:|---:|---:|---:|
| 0.00 | 0 of 48 | 0.0557 | 0.254 |
| 0.25 | 0 of 48 | 0.0712 | 0.290 |
| 0.50 | 0 of 48 | 0.1063 | 0.447 |
| 1.00 | 0 of 48 | 0.1911 | 0.912 |
| 2.00 | 5 of 48 | 0.7760 | 10.071 |
| 4.00 | 36 of 48 | 11.7856 | 32.285 |

The jump between 1x and 2x is not the speed chain degrading gracefully — it is
association starting to fail. The sweep stops at 4x because 36 of 48 vehicles are
already lost there, so what survives is a survivorship-biased sample of the easy
tracks rather than a speed measurement.

### Speed, tier 2: the real clip gets no speed at all

The flagship clip's along-road scale has no independent anchor. A dedicated survey
looked for one, found none, and found the shipped correspondences internally
inconsistent as well. The honest bracket on the scale is 18.0 m down to about a
third less, which propagates one for one to speed, and a figure with that much scale
uncertainty is not a speed figure. So `configs/motorway.yaml` ships with **no
calibration block**, the engine returns nothing for speed on that clip by design,
and no absolute km/h from it appears on any surface of this project.

That investigation is a result rather than a caveat, with its matched controls, and
it has its own document: **[CALIBRATION.md](web/public/CALIBRATION.md)**.

### Browser performance

**Source:** the committed harness `scripts/measure_backend.sh`, not a report file —
these timings need a real browser on real hardware, so no Python test can
regenerate them and none pins them.
**Protocol:** `scripts/measure_backend.sh 120 4199` on an Apple M1 Pro under macOS,
headless Chrome with WebGPU enabled. Each iteration is the whole per-frame path the
page actually runs — video decode, letterbox, tensor build, inference — timed by one
outer bracket, after 5 warmup iterations. Medians.

| backend | ms per frame, median | fps | n | renderer |
|---|---:|---:|---:|---|
| WebGPU | 22.05 | 42.06 | 120 | `ANGLE (Apple, ANGLE Metal Renderer: Apple M1 Pro, Unspecified Version)`, hardware |
| WASM, single thread | 126.90 | 7.83 | 60 | n/a (wasm path) |

The renderer string is attached to the WebGPU row because that path genuinely runs
through it, and a software-renderer string would invalidate the figure — the harness
refuses to report a timing at all when it detects one. It is deliberately **not**
attached to the WASM row: WebAssembly inference never touches the GL renderer. A run
forced onto SwiftShader reported 123.85 ms against the hardware run's 126.90 while
the renderer string changed completely, so quoting the string beside the WASM figure
would imply a validation that was never performed.

Single-threaded WebAssembly is the published fallback because GitHub Pages cannot
send the COOP and COEP headers that multi-threaded WebAssembly needs;
`crossOriginIsolated` is confirmed false on the live host. The source clip is 30 fps,
so the WebGPU path runs ahead of real time and the WebAssembly path cannot, which is
why the page decouples detection cadence from render cadence and says which it is
doing.

### Cross-surface parity

**Source:** `reports/parity.json`.
**Protocol:** fixtures written by the Python engine and replayed through the
TypeScript engine in the visitor's own tab at `?selftest=1`, so a green verdict is
about the artefact that is actually serving rather than a build-time copy of it.
Every case is required to straddle a decision boundary — an anchor exactly on the
gate line, an IoU exactly at the association floor, a score exactly at the
confidence threshold, an assignment cost tie, a float32 class-score tie. The
real-clip rows below run the speed path, which needs a road plane, so
`scripts/make_parity_fixtures.py` hands both engines one fixed matrix reconstructed
from the withdrawn survey. **It is an instrument, not a calibration**: it exists so
that a disagreement is two implementations differing rather than two calibrations
differing, the shipped `configs/motorway.yaml` still refuses speed on this clip, and
no km/h figure about this clip is reported anywhere.

| what is pinned | value |
|---|---:|
| committed parity cases | 8 |
| boundary kinds every case must straddle | 6 |
| speed agreement tolerance, km/h | 1e-06 |
| association floor the fixtures straddle | 0.8 |
| the straddling IoU, and its control one ulp below | 0.8 / 0.7999999999999999 |
| Mahalanobis gate the fixtures straddle | 9.4877 |
| frames replayed from the real clip | 150 |
| detections in them | 1069 |
| track identities allocated | 133 |
| crossings emitted | 3 |
| speeds the two engines compared under the parity instrument, and withheld | 473 / 79 |

Crossing decisions agree exactly and speeds agree to 1e-06 km/h. The one-ulp
control is the point: a boundary case that both engines got right by rounding the
same way proves nothing, so each fixture carries a partner one floating-point step
to the other side of the line.

---

## The honest negatives

Nine results that do not flatter this project, in the same detail as the ones that
do. They are also on the site, rendered from the same reports.

1. **The engine's own tracker does not pay for itself on this clip.** On clean
   footage its Kalman filter and Hungarian assignment score 0.914 against 0.941 for
   both simpler baselines — one event, so read it as a tie rather than a gap. Across
   the whole sweep it is lowest of the three at all 19 levels where they differ at
   all, and, more to the point, **lowest at all 12 levels where they differ by more
   than one event**; it leads at none. It costs about 21.7 times a baseline tracker's
   CPU. Sophistication that does not earn its cost is a result.
2. **It stops counting entirely at 2 fps** — 0 predictions against 17 labels, not
   wrong ones, none.
3. **Two pixels of per-corner box jitter cost it 29.7% of its accuracy** (F1 0.643
   against 0.914), at 2.7 to 30.9 times the noise measured on this very clip. Both
   baselines hold 0.882 at that level.
4. **The one degradation it handles gracefully has an undiagnosed cause.**
   Detection dropout degrades smoothly with no cliff, 0.914 to 0.882 across
   p = 0.00 to 0.30 — and the association-floor ablation that explains every other
   collapse explains 0.0000 of this one. That makes it a second, separate fault, and
   it is **undiagnosed**.
5. **Nothing measures the shipped association floor as better than SORT's, except
   under detection dropout.** Loosening 0.8 to 0.3 recovers up to 0.6020 F1 under
   degradation and exactly nothing undegraded, where the two tie. It ties clean, loses
   on three of the four degradation families, and wins on one, by 0.0252. The constant
   is published as a measurement rather than changed: picking a new floor is separate
   work with its own baseline, and this sweep is not the evidence for a value.
6. **Quantising the detector was measured and refused.** int8 dynamic quantisation
   saves 7.69 MB of download and loses 17.6% of detections. In a counting product a
   missed detection is a missed count, so the trade is refused at any download size
   and the page pays the full weight. Details in
   [MODEL_CARD.md](web/public/models/MODEL_CARD.md).
7. **Every accuracy figure here is an upper bound.** One clip, one gate, 17
   labelled crossings, 7 of them adjudicated certain — and that gate was chosen for
   label reliability, which is the easiest case for association as well. No second
   clip was labelled and no claim is made about one.
8. **The flagship clip cannot be given a speed at all.** Five candidate scale
   anchors were searched for and none is usable, so no km/h publishes from it.
9. **Four identity claims this project does not make**, quoted verbatim from the
   benchmark's own record in the [design card](web/public/DESIGN_CARD.md) — no
   ID-switch, IDF1, MOTA or MOTP figure, no identity proxy, no reading of the
   fragmentation ratio as a count of identity errors, and no claim about tracking
   away from the labelled gate.

A tenth, about this repository rather than the engine: the letterbox resize had to
be moved to an integer-exact interpolation before the browser mirror could match it
at all, because the obvious call is intercepted by a hardware abstraction layer that
no TypeScript port can reproduce. The parity that the live demo rests on is a
property of that choice, not of careful porting alone.

---

## Architecture

Full document: **[ARCHITECTURE.md](web/public/ARCHITECTURE.md)**.

```
src/trafficlens/
  core/          geometry predicates, gate counting, road-plane homography, constants
  track/         constant-velocity Kalman filter, association, two-stage tracker
  analytics/     speed, incidents, speed-limit violations
  detect/        shared preprocessing and decoding + ultralytics and ONNX adapters
  io/            frame sources and export formats
  bench/         baselines, degradations, scoring, slit-scan ground truth, simulator
  pipeline.py    one frame in, detections through tracking through analytics out
  cli.py         run / fetch-samples / calibrate / export-model
web/src/
  engine/        the TypeScript mirror: geometry, gate, kalman, associate, tracker, homography, speed
  runtime/       onnxruntime-web session, preprocessing, decoding, model cache
  ui/            the page: controls, overlay, charts, results sections
  generated/     constants.ts and reports.ts, both generated and never hand-edited
scripts/         the benchmarks, the exporters, the page verifier, the timing harness
reports/         every measured number, as JSON
docs/            the published site, byte-reproduced by `npm run build`
```

Three properties hold the two implementations together. The constants live in one
file and the TypeScript copy is generated from it, so a tunable cannot drift.
Preprocessing agrees exactly rather than approximately: the mirror reproduces the
Python letterbox element for element on committed fixtures chosen for the two
hazards that make it hard — a resize weight landing exactly on a half step, and odd
padding — which is what makes "the same detector" a fact rather than a hope. And the
boundary cases are fixtures too, replayed in the browser on the live site.

---

## Tech stack

**Source:** `pyproject.toml` and `web/package.json`.

| package | version range | what it is for |
|---|---|---|
| numpy | >=1.26,<3 | arrays, the whole numeric core |
| scipy | >=1.11,<2 | Hungarian assignment, chi-square gate |
| opencv-python | >=4.9,<5 | video decode, letterbox, homography solve |
| pydantic | >=2.5,<3 | config schema and validation |
| PyYAML | >=6.0,<7 | config files |
| click | >=8.1,<9 | the CLI |
| ultralytics + torch | >=8.4,<8.5 / >=2.2,<3 | the server detector, optional extra `detect` |
| onnxruntime | >=1.19,<2 | the ONNX adapter, optional extra `onnx` |
| pytest | >=8.0,<9 | the Python suite, optional extra `dev` |
| onnxruntime-web | >=1.27.0 <1.28.0 | browser inference on WebGPU or WebAssembly |
| typescript | >=7.0.0 <8.0.0 | the browser engine, no runtime framework |
| vite | >=8.2.0 <9.0.0 | the site build; output is `docs/` |
| vitest | >=3.2.0 <4.0.0 | the browser engine's suite |

The page has no UI framework, no chart library and no font CDN. Fonts are served
from the repository because the page claims that nothing about a frame you load
leaves your device, and a request to a third-party font host on every view would
make that claim false.

---

## Reproducing every number

```bash
# the Python suite
PYTHONPATH=src .venv/bin/python -m pytest tests/

# the browser engine's suite, and the type check
cd web && npm install && npm test && npx tsc --noEmit

# regenerate the site into docs/; a second run is byte-identical
cd web && npm run build

# the benchmarks, in the order their caches allow
PYTHONPATH=src .venv/bin/python scripts/bench_counting.py    # counting_accuracy.json, detection_noise.json
PYTHONPATH=src .venv/bin/python scripts/bench_robustness.py  # robustness.json + figures
PYTHONPATH=src .venv/bin/python scripts/bench_tracking.py    # tracking.json
PYTHONPATH=src .venv/bin/python scripts/bench_speed.py --figures  # speed_synthetic.json, speed_real.json

# the published page's own verdict, in a real browser
scripts/verify_page.sh 'http://127.0.0.1:4199/?selftest=1'

# the browser timings in the table above
scripts/measure_backend.sh 120 4199
```

`bench_counting.py` writes the cached detection stream the other three read; they
refuse to run without it rather than detecting for themselves, so every report in
`reports/` describes the same detections.

---

## Licence and attribution

The code in this repository is **MIT** — see [LICENSE](LICENSE). Three shipped
assets are not, and each carries its own notice beside the file.

**Source:** the notices published with the assets — `web/public/models/MODEL_CARD.md`,
`web/public/clips/NOTICE.md`, `web/public/fonts/NOTICE.md`.

| asset | licence | attribution and changes |
|---|---|---|
| `web/public/models/yolo11n-480.onnx` | **AGPL-3.0** | An ONNX export of Ultralytics YOLO11n. Ultralytics publishes YOLO11 and its pretrained checkpoints under the GNU Affero General Public License v3.0, and exporting a checkpoint to another file format does not change its licence. The AGPL's network clause is the part that matters for a hosted page. Full statement in [MODEL_CARD.md](web/public/models/MODEL_CARD.md). |
| `motorway-a40` clip | CC BY 3.0 | "Motorway A40 – on bridge above the traffic", by the *Sounds of Changes* project, via Wikimedia Commons. **Changed:** excerpted to 20.0 s, scaled to 960x540, re-encoded to H.264, audio removed. |
| `street-aisle` clip | CC BY 4.0 | `person-bicycle-car-detection` from Intel Corporation's IoT DevKit `sample-videos`. **Changed:** excerpted to 30.0 s, re-encoded, audio removed, renamed. |
| Archivo, IBM Plex Sans, IBM Plex Mono | SIL OFL 1.1 | Latin subsets downloaded verbatim from Google Fonts. Copyright 2020 The Archivo Project Authors; copyright 2017 IBM Corp. with Reserved Font Name "Plex". Licence text and source URLs in [fonts/NOTICE.md](web/public/fonts/NOTICE.md). |

Both clips are Creative Commons **Attribution** licences, and both files served here
were **transcoded and excerpted** — a modification the licence requires be
indicated, which is what the "Changed" column and the published notices do. The
served copies are not byte-identical to the footage the command-line benchmarks ran
on, and no accuracy figure on the site is derived from them.

If you want TrafficLens without the AGPL obligation, replace the ONNX file with a
detector whose licence suits you: nothing in either engine is specific to those
weights beyond the input size and the class ordering, both of which are parameters.

---

## The other documents

**Source:** the documents themselves. Nothing in this table is a measurement.

| document | what is in it |
|---|---|
| [DESIGN_CARD.md](web/public/DESIGN_CARD.md) | Intended use, out-of-scope use, the data the numbers come from, and what the system cannot do. |
| [CALIBRATION.md](web/public/CALIBRATION.md) | Why the flagship clip gets no speed: the anchor survey, its matched controls, and the internal contradiction in the shipped correspondences. |
| [ARCHITECTURE.md](web/public/ARCHITECTURE.md) | The real shape of the code, the two implementations, and the guard layers. |
| [MODEL_CARD.md](web/public/models/MODEL_CARD.md) | The browser detector: its graph, its licence, and the quantisation that was measured and refused. |
| [data/groundtruth/PROTOCOL.md](data/groundtruth/PROTOCOL.md) | How the 17 crossings were labelled, decided before any scoring code existed. |
