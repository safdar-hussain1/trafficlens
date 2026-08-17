# TrafficLens — architecture

The real shape of the code: what each layer owns, why the same engine exists twice,
and which guards stop the two copies drifting apart.

---

## The one-frame path

A session is a loop over frames, and every frame takes the same route:

```
frame  ->  detect  ->  track  ->  analytics  ->  gates  ->  events
             |           |            |            |
        letterbox +   Kalman +    speed, if     segment
        decode + NMS  assignment  calibrated    crossing
```

Each arrow is a module boundary that holds. `detect` knows nothing about tracks,
`track` knows nothing about gates, and `analytics` reads tracks without being able to
change them. `pipeline.py` is the only place that knows the order, and it times the
three stages with three disjoint brackets so a published breakdown is measured rather
than apportioned.

## The Python package

**Source:** `src/trafficlens/`, module docstrings. Nothing in this table is a
measurement.

| module | what it owns |
|---|---|
| `core/geometry.py` | Exact predicates: which side of a line a point is on, whether two *segments* cross, the crossing parameter. The signed-side test is the primitive every direction decision reduces to. |
| `core/gate.py` | A gate as a drawn segment, counted once per track, per class, per direction. Uses the segment, never its infinite extension, so a vehicle crossing off the end of the drawn gate is not counted. |
| `core/homography.py` | The road plane: pixels to metres, fitted from correspondences, with a degeneracy diagnostic and a held-out reprojection error. Refuses a fit it cannot honestly validate. |
| `core/constants.py` | Every shared tunable, as `NAME = literal` and nothing else. The single definition both engines read. |
| `track/kalman.py` | A constant-velocity Kalman filter over boxes in centre/aspect/height space. |
| `track/associate.py` | Vectorised IoU, Hungarian assignment with a cost ceiling, and a canonical tie rule so two equal-cost assignments resolve the same way in both languages. |
| `track/tracker.py` | The two-stage tracker: high-confidence detections against every live track, then low-confidence ones against confirmed tracks only, behind a Mahalanobis gate. Holds no numeric tunables of its own. |
| `analytics/speed.py` | Plane-space speed: component-wise least-squares slopes, then their magnitude. Returns nothing where the camera is uncalibrated. |
| `analytics/incidents.py` | Stopped vehicles and wrong-way crossings. |
| `analytics/violations.py` | Speed-limit policy and evidence snapshots. |
| `detect/base.py` | The shared, torch-free path: letterbox, decode, class-wise NMS. Both adapters go through it, so they are comparable rather than separately trusted. |
| `detect/ultralytics_yolo.py` | The server detector adapter. |
| `detect/onnx_yolo.py` | The ONNX adapter — the same graph the browser runs, through the same shared preprocessing. |
| `io/video.py` | Files, webcams and network streams behind one iterator. |
| `io/export.py` | The crossing-events CSV, the summary JSON, and the versioned session JSON that the browser and the benchmarks both replay. |
| `bench/baselines.py` | The standard failure modes: a centroid tracker, a greedy-IoU tracker, a band counting rule, a per-frame counting rule. Implemented faithfully so the benchmark can price them. |
| `bench/degrade.py` | Four independent ways to spoil an input stream. Knows nothing about who consumes it. |
| `bench/scoring.py` | Predicted crossings against labelled ones. Knows nothing about trackers. |
| `bench/slitscan.py` | Slit-scan ground truth: an independent view of the gate line, and a loader that refuses a label set it cannot trust. |
| `bench/simulate.py` | A synthetic scene whose truth is exact, for absolute-speed validation. |
| `bench/scale.py` | The along-road scale investigation, as code rather than as a note. |

### Two seams worth naming

**`scoring` knows nothing about trackers.** It takes labelled crossings and predicted
crossings and returns a score. That is why the same scorer produces the undegraded
table, every degraded level, and the certain-only subset, and why a change to a
tracker cannot quietly change how it is graded.

**`degrade` knows nothing about consumers.** It transforms a detection stream and
hands it back. That is why every protocol's identity level reproduces the undegraded
benchmark exactly, through the general code path with no short-circuit branch —
`p = 0` because `uniform >= 0` always holds, `sigma = 0` because a zero-variance
normal is exactly zero, and so on. The reduction proves the transform rather than
proving a branch.

**Two adapters, one preprocessing path.** `detect/base.py` holds letterbox, decode
and NMS; the ultralytics and ONNX adapters differ only in how they get logits out of a
model. A committed test runs both over a real frame and compares them, which is only
meaningful because neither has its own copy of the pre- or post-processing.

---

## The TypeScript mirror, and why byte parity matters

`web/src/engine/` is the same engine again: `geometry`, `gate`, `kalman`,
`associate`, `tracker`, `homography`, `speed`, `pipeline`. `web/src/runtime/` adds
what a browser needs — an onnxruntime-web session with a WebGPU path and a
single-threaded WebAssembly fallback, the preprocessing that feeds it, the decoder,
and a cache so the model downloads once.

The claim the live page makes is that **the visitor's own GPU runs the same
detector**. "Same" has to mean identical decisions, not similar ones, because a
counting product's output is discrete: a crossing either fires or it does not, and a
boundary case that the two engines round differently is a different answer, not a
smaller one.

Three mechanisms make that true.

**1. The constants are generated, not copied.** `scripts/export_constants.py` parses
`core/constants.py` with Python's `ast` module and writes `web/src/generated/constants.ts`,
comments and source order included. A test regenerates into a temporary file and
asserts byte equality with the committed artefact, so a tunable copied across the
language boundary by hand is a test failure rather than a divergence nobody notices
until the numbers disagree. The parser refuses the whole file rather than skipping
what it does not understand, because a silently dropped constant would regenerate
byte-identically.

**2. Preprocessing agrees element for element.** The letterbox resize is the hardest
part of the mirror and the reason the Python side uses an integer-exact interpolation
rather than the obvious floating-point one: the obvious call is intercepted by a
hardware abstraction layer on this platform, so no faithful port of the documented
algorithm can match what actually executes. The integer-exact variant is
deterministic by design and the mirror reproduces it exactly on committed fixtures
chosen for the two hazards — a resize weight landing precisely on a half step, and
odd padding.

**3. The decision boundaries are fixtures, replayed in the browser.**

**Source:** `reports/parity.json`.
**Protocol:** fixtures written by the Python engine and replayed through the
TypeScript engine in the visitor's own tab at `?selftest=1` — so a green verdict is
about the artefact that is actually serving, not a build-time copy of it. Every case
must straddle a decision boundary, and each carries a control one floating-point step
to the other side of it, because a boundary case both engines happened to round the
same way proves nothing.

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

The straddled boundaries are an anchor exactly on the gate line, an IoU exactly at
the association floor, a score exactly at the confidence threshold, an assignment
cost tie, a float32 class-score tie, and a deferred on-line crossing that has to use
its last off-line origin. Crossing decisions agree exactly; speeds agree to 1e-06 km/h.

---

## The page

`web/src/ui/` has no framework, no chart library and no font CDN. The charts are
inline SVG built from the same numbers as the tables, the fonts are served from the
repository, and the results sections do not contain a single typed figure: they read
`web/src/generated/reports.ts`, which `scripts/build_site_data.py` bakes from
`reports/*.json`.

That artefact is guarded four ways at once, because byte equality between a generator
and its own output is a tautology — an exporter that publishes a subset of its inputs
passes a byte-equality check happily, since regenerating reproduces the same hole. So
there is byte equality with a fresh bake, a hand-written coverage table of pointers
that *must* be published, a containment sweep asserting every published scalar occurs
in its own source report, and a recomputation of every derived value from the source
JSON with the formula written out a second time.

The build itself is `tsc --noEmit`, then Vite into `docs/`, then a manifest step.
`docs/` is the published Pages root and lives in the repository, which means it can go
stale silently: the site could serve a bundle built from sources that no longer exist,
indefinitely, with every test green. `docs/BUILD_MANIFEST.json` records a digest of
every file the build read and every file it wrote, and a test recomputes both sides
from the working tree. Change a source and forget to run the build, hand-edit a file
under `docs/`, or leave a stray file there, and it fails.

`web/public/` is copied verbatim into `docs/`. That is deliberate for the runtime
assets — the wasm binary and the onnxruntime entry point are byte-identical to what
npm published, and a test asserts it, which is only worth something if the bytes it
checks are the bytes the browser executes. It is also why this document, the design
card, the calibration document, the model card and both licence notices live under
`web/public/`: the build empties `docs/` on every run, so a file authored there would
be destroyed by the next build.

---

## The guard layers

**Source:** `tests/`, and the tests named in each row. Nothing in this table is a
measurement.

| layer | guard | what it refuses |
|---|---|---|
| shared constants | `test_constants_sync.py` | A TypeScript constant that no longer matches its Python definition, and a constants file emptied to make the comparison vacuous. |
| cross-language behaviour | `test_parity.py`, `web/src/parity.test.ts` | A boundary case the two engines decide differently, or a fixture set that does not straddle a boundary at all. |
| the published site's numbers | `test_site_data_sync.py` | A figure on the page that no report supports, a report field silently dropped from the bake, and an absolute speed for the uncalibrated clip. |
| the published documents' numbers | `test_docs_numbers.py` | A figure in the README or a card that its report contradicts, a results table with no stated source or protocol, and an emptied document that would otherwise pass by having nothing to check. |
| the published tree | `test_docs_build_manifest.py` | A `docs/` built from sources that have since changed, a hand-edited output, and a stray file that would be served to visitors. |
| the repository's own hygiene | `test_guards.py` | Banned words in any inflection, absolute home-directory paths, process documents committed as source, and published assets accidentally git-ignored. |
| the page in a real browser | `scripts/verify_page.sh`, `?selftest=1` | A live page whose own self-test does not pass, exercised by exit status so it is usable from a mutation check. |

Every one of these carries a floor — a minimum number of constants, cases, figures,
inputs or outputs — because the failure this repository keeps finding is not a wrong
assertion. It is an assertion with nothing left to check.
