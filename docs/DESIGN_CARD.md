# TrafficLens — design card

What this system is for, what it is not for, what its numbers rest on, and what it
cannot do. The last section is the longest on purpose.

Every figure here is read from a file under `reports/` in the repository, and
`tests/test_docs_numbers.py` fails if this document and those files disagree.

---

## Intended use

TrafficLens turns footage from **one fixed camera** into counted gate crossings, by
class and by direction, with speeds where — and only where — that camera has been
calibrated against a surveyed road plane.

It is built for the case its benchmarks measure: a camera mounted above or beside a
road, looking along the traffic, with the gate drawn across lanes whose vehicles are
large enough in frame to be detected and tracked reliably. Reasonable uses:

- **Counting and flow.** How many vehicles crossed a line, in which direction, of
  which class, over a period.
- **Occupancy and headway in seconds.** Both are timing quantities and neither needs
  a metric scale.
- **Direction and wrong-way detection.** A crossing against a gate's declared
  direction. The direction convention is stated per gate in the config rather than
  defaulted, because a default that is inverted for the scene turns every ordinary
  crossing into an alert.
- **Relative speed between vehicles**, on an uncalibrated camera. Ratios survive an
  unknown scale; absolute values do not.
- **Absolute speed in km/h — only with a calibration you can defend.** That means a
  surveyed road plane whose along-road scale rests on a measurement, not an
  assumption, validated against points held out of the fit.
- **A demonstration that runs on the visitor's own device.** The published page does
  all inference locally; nothing about a frame you load leaves the browser.

## Out-of-scope use

- **Enforcement, prosecution, or anything with a penalty attached.** The absolute
  speed figures in this project come from a synthetic scene with exact truth and no
  calibration error, and the one real clip that was surveyed properly turned out to
  have no usable scale anchor at all. Nothing here has been validated against an
  instrumented vehicle or a certified reference.
- **Identifying people or vehicles.** There is no re-identification, no plate
  recognition, no face detection, and no cross-camera association. The tracker's
  identities are per-session integers that mean nothing outside the run that made
  them, and no identity is written to any export in a form that could follow a
  vehicle anywhere.
- **Surveillance of individuals.** The engine emits counts, crossings and incidents.
  It was not built to follow a named person or vehicle and does not support it.
- **Safety-critical control.** No part of this has redundancy, failure detection or
  a validated failure rate. The one degradation the engine handles gracefully has an
  undiagnosed cause, which is the opposite of what a safety case needs.
- **Crowd or pedestrian counting.** The default classes are vehicles, the accuracy
  evidence is vehicles at a motorway gate, and pedestrians are neither measured nor
  claimed.
- **Any deployment where a wrong count is expensive**, without labelling your own
  ground truth first on your own footage. See what the numbers rest on, below: they
  rest on one gate of one clip.

---

## What the numbers rest on

**Source:** `reports/counting_accuracy.json`.
**Protocol:** `yolo11s.pt`, imgsz 640, over the whole labelled window of one clip,
one gate, scored against crossings labelled by hand from an independent slit-scan
view of the gate line, under a protocol fixed before any scoring code existed.

| what | how much |
|---|---|
| clips labelled | 1 |
| gates labelled | 1 |
| labelled crossings | 17 |
| of those, adjudicated certain | 7 |
| of those, adjudicated probable | 10 |
| classes represented in the labels | 2 of the 4 the config detects |
| directions represented | 1 |

That is the entire accuracy evidence base for counting. It supports the following
and nothing wider:

- **A bracket, not a point.** The engine's crossing F1 is **0.914** on all 17 labels
  and **0.800** on the 7 certain ones under ignore-region semantics. Both publish
  together because together they bracket the truth.
- **Upper bounds.** The labelled gate is in the near field and was chosen for label
  reliability — the largest, best-separated, least-foreshortened traffic in the
  frame. That is also the easiest case for association. The same engine further away,
  on the far carriageway, or in a queue scores lower, and nothing here says by how
  much.
- **A resolution of one event.** With 17 labels, one added or removed crossing moves
  F1 by about **0.027**, so every counting figure carries **±0.027 (one event)**. A
  comparison finer than that is one event rather than a result, and where this card
  compares methods it says which side of the line the comparison falls on.
- **No second clip.** The other bundled clips are demonstrations. They have no
  ground truth and no figure is derived from them.

Everything about degradation comes from the same 17 labels and the same cached
detections, spoiled four ways: frame rate, dropped frames, per-detection dropout and
per-corner box jitter. The four sweeps share one detection stream by design, so they
compare trackers and counting rules and never detector variance — and equally, they
say nothing about how a detector behaves in rain, at night, or on a camera unlike
this one.

The absolute-speed figures come from a **synthetic** scene whose truth is exact
arithmetic. That is a deliberate instrument, and it excludes the two error sources a
deployment will actually be dominated by: it excludes the detector, and it excludes
calibration error by construction, because the fitted plane recovers the simulated
camera to **6.2e-06** m. Read those figures as a ceiling on the tracking-to-speed
chain, not as field accuracy.

Even as a ceiling it is not a clean pass. The chain was required to land within
0.1 km/h and does so up to about 70 km/h; the worst settled error over the whole
band is **0.254** km/h, at 130. The requirement was not widened to absorb that — the
cause was identified instead, and it is the tracker's own constant-velocity filter
lagging motion that accelerates as a vehicle approaches.

---

## What the system cannot do

The section that decides whether the rest of this document can be trusted.

**1. It cannot give an absolute speed on uncalibrated footage — including its own
flagship clip.** The motorway clip's along-road scale has no independent anchor: a
dedicated survey searched for five kinds of anchor and found none usable, and the
correspondences that used to ship were internally inconsistent besides. The honest
bracket on the scale runs down to **12.2** m from an assumed 18 m, which propagates
one for one to speed as about **-33** %. So `configs/motorway.yaml` ships with no
calibration block, the engine returns nothing for speed on that clip, and no km/h
figure from it appears on any surface. The full investigation is in
[CALIBRATION.md](./CALIBRATION.md).

**2. Its own tracker is measurably beaten by simpler baselines on this footage.** On
clean frames the engine's Kalman filter plus Hungarian assignment scores below both a
centroid tracker and a greedy-IoU tracker — by one event, so read that one as a tie.
Across the whole sweep it is lowest of the three at all **19** levels where they
differ at all, four of which are the undegraded levels the sweep reduces to rather
than degradations; and it is lowest at all **12** levels where they differ by more
than one event, which is the figure that survives the resolution above. It leads at
none, and it costs about **21.7** times a baseline tracker's CPU. Two metrics —
crossing F1 and identity behaviour at the gate — rank them the same way. If you need a
tracker for this kind of footage, the honest reading is that the sophistication is not
paying for itself here.

**3. It stops counting entirely at a low frame rate.** At 2 frames a second it makes
**0.000** F1 — no predictions at all against 17 labels. Not wrong counts: none.

**4. Mild box jitter costs it real accuracy.** Two pixels of per-corner jitter, a
small multiple of the noise measured on this very clip, takes crossing F1 to
**0.643** while both baselines hold 0.882.

**5. It publishes no identity-preservation metric, and offers no proxy.** The
benchmark's record carries 4 refusals and all 4 are quoted verbatim here, because
paraphrasing a refusal is how a refusal softens:

> No ID-switch count, IDF1, MOTA, MOTP or any other identity-preservation figure is
> published, here or on any other surface of this project.

> The fragmentation ratio is not a count of identity errors and must not be quoted
> as one.

> Class consistency is not a measurement of the classifier.

> Nothing here is evidence about tracking on the far carriageway, in the distance,
> or in a queue.

The reason for the first is that all of those metrics need a frame-by-frame identity
label set — which vehicle every box belongs to, across the whole window — and this
clip has none. Producing one is the same human adjudication the 17 crossings
required, at a far larger scale, and it was not done. A proxy derived from the
tracker's own output would be the tracker grading its own identities.

**6. It cannot re-identify anything.** No appearance model, no embedding, no
cross-camera or cross-session matching. A vehicle that leaves the frame and returns
is a new track.

**7. It cannot read a number plate**, and does not try. There is no OCR anywhere in
this project.

**8. It cannot tell you how well the detector detects.** Everything measured here is
downstream of a cached detection stream or of generated boxes. The box-noise figures
the jitter sweep is calibrated against are labelled a proxy in their own report,
because they are residuals against a median filter of the track's own trajectory and
contain the filter's smoothing and any association error along with the detector's
jitter.

**9. It cannot run its own browser demo well without a GPU.** The WebGPU path
carries a full detector ahead of real time on the hardware it was measured on; the
single-threaded WebAssembly fallback runs at about a fifth of that, and it is the
only fallback available, because a static host cannot send the headers that
multi-threaded WebAssembly requires.

**10. It cannot count what its detector does not detect.** The classes are the
config's, and a vehicle the detector misses at 480 or 640 pixels of model input —
small, distant, heavily occluded — is a missed count, not a missed box. That is why
an int8 quantisation of the browser model was refused after measurement: it dropped
detections, and a dropped detection in a counting product is a dropped count.

**11. It cannot be assumed to behave like this on your footage.** Different camera
height, lens, weather, traffic density and frame rate all move these numbers, and
three of the four degradations measured here move them a long way. Label your own
gate before trusting a figure.

---

## In scope and out, at a glance

**Source:** the two sections above; nothing in this table is a measurement.

| in scope | out of scope |
|---|---|
| Counting crossings at a drawn gate | Enforcement or anything with a penalty attached |
| Direction, and wrong-way detection | Identifying a person or a vehicle |
| Occupancy, headway in seconds, flow | Number plates, faces, re-identification |
| Relative speed between vehicles | Absolute speed without a defensible calibration |
| Absolute speed on a surveyed, validated camera | Safety-critical control or automated response |
| A local, private browser demonstration | Pedestrian or crowd counting |
| Benchmarking counting rules and trackers against labelled crossings | Any claim about footage unlike the one labelled clip |

---

## Provenance and licences

The browser detector is an ONNX export of Ultralytics YOLO11n and is **AGPL-3.0** —
see [models/MODEL_CARD.md](./models/MODEL_CARD.md). The rest of the code is MIT. The
demonstration clips are Creative Commons Attribution, transcoded and excerpted, and
credited in [clips/NOTICE.md](./clips/NOTICE.md); the fonts are SIL OFL and credited
in [fonts/NOTICE.md](./fonts/NOTICE.md).
