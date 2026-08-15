# Ground-truth labelling protocol

Every accuracy number this project publishes is scored against a label
set produced under this protocol. The protocol is written down first, and
fixed before any labelling starts, so that the labels are not quietly
shaped by whatever the engine happens to output. Read the whole of it
before labelling a single crossing.

## The one rule that matters

**Labels are produced from slit-scan review plus full-frame confirmation.
They are never produced, seeded, corrected or "sanity checked" against the
detector, the tracker, or `GateCounter` output.**

A ground truth derived from the system under test measures nothing. If a
label set were built by running the pipeline and fixing up its mistakes,
every count, precision and recall figure downstream would be a measure of
how consistent the pipeline is with itself. So the labelling modality is
deliberately independent of the pipeline:

- **The slit-scan** (`trafficlens.bench.slitscan`) samples only the band
  of pixels lying along the gate segment and stacks one such strip per
  frame, newest at the bottom. Every vehicle that physically passes over
  the gate paints a blob in that image, and the *row* of that blob **is**
  the frame index of the crossing. No detector, no tracker, no
  association threshold, no confidence score from the system under test
  is involved in any part of it: it is raw pixels, resampled and stacked.
- **Full-frame confirmation** then decides whether the blob is a vehicle
  (rather than a shadow, a spray plume, a bird or a compression
  artefact), which class it is, and which way it is going. The labeller
  looks at the frame itself with the gate drawn on it, at the row the
  slit-scan pointed at.

A blob nobody can confirm in the full frame is not a crossing. A vehicle
visible in the full frame whose contact point never reaches the gate
segment is not a crossing either, however obvious the vehicle is.

## Clip, window and gate

| Field | Value |
| --- | --- |
| Clip | `data/samples/motorway-a40.webm` |
| Frame size | 1280 x 720 |
| Frame rate | 30.0 fps (container) |
| Window | frames 0 to 734 inclusive |
| Gate name | `inbound` |
| Gate start | `[0.06, 0.80]` (normalized) |
| Gate end | `[0.46, 0.80]` (normalized) |
| Carriageway | the left-hand carriageway, whose traffic runs **toward** the camera |
| Direction labels | `away` (+1 side, up the frame) and `toward` (-1 side, down the frame) |

The window stops at frame 734 although the container advertises 737
frames. 734 is the last frame that actually decodes: the clip was played
out end to end through `trafficlens.io.video.VideoSource` and the last
index it yielded was 734, so the container's count overstates the
footage by two frames. The window is the frames that exist, measured,
not the frames the header claims. A label set must never reference a
frame whose existence depends on which decoder opened the file.

The generator measures this for itself when `--end-frame` is omitted,
and `build_slitscan` refuses outright to produce a scan for a window the
clip cannot fill, rather than returning a short image whose rows no
longer line up with frame numbers.

### Why this gate

The `inbound` gate was chosen **for label reliability, not for the
engine's convenience**. It sits in the near field, low in the frame,
across the carriageway whose vehicles are largest, best separated and
least foreshortened on screen. Those are the conditions under which a
human can say with confidence how many distinct vehicles crossed and
what each of them was — which is the only thing a ground truth needs to
be good at.

This choice is not neutral for the reported accuracy figures, and that
must be stated wherever those figures are: near-field, well-separated
traffic is also the easiest case for the detector. A number measured
here is an upper bound on what the same engine would score on the far
carriageway, in the distance, or in a queue. It is not a claim about
those conditions. The alternative — labelling where the engine finds it
hard — would produce a label set the human cannot trust either, which
measures nothing at all.

## What counts as a crossing

### The anchor rule

**A vehicle counts as having crossed on the first frame at which its
road-contact point passes over the gate segment.**

The road-contact point is the point where the vehicle meets the road
surface, taken as the mid-point of the contact line between its wheels
and the tarmac — in practice the bottom-centre of the vehicle as it
appears in the frame. Not the centroid of its body, not the roof line,
not the leading edge of the bumper. The contact point is used because it
is the only part of the vehicle that lies on the road plane, so it is
where the vehicle actually is relative to a line drawn on that road.

The gate is a **bounded segment**, not an infinite line. A vehicle on
another carriageway that passes over the gate's extension beyond one of
its two endpoints has not crossed. A vehicle whose contact point passes
exactly through an endpoint has crossed (bounds are inclusive), matching
the engine's own segment test.

The crossing frame is the **first** frame at which the contact point is
on or beyond the gate segment. "On" is not a tie to be broken by
judgement: a contact point sitting exactly on the segment has arrived,
and that frame is the crossing frame, even if the vehicle sits there for
several more frames before it is visibly past. This is the same
convention the engine uses — `trafficlens.core.geometry.segments_intersect`
treats a path touching the segment as an intersection, and
`trafficlens.core.gate` documents that "a crossing that lands exactly on
one of the gate's own endpoints counts" — so ground truth and engine
place a crossing on the same frame rather than one frame apart by
construction.

### Reading a blob: which of its rows is the crossing frame

A vehicle is not a point. It covers the gate for as many frames as it
takes its whole body to pass, so it paints a blob many rows tall — a
near-field lorry can occupy thirty rows or more. Exactly one of those
rows is the crossing frame, and which one depends on the direction of
travel, because the anchor rule is about the road-contact point and not
about the body:

- **Traffic toward the camera** (the ordinary case on this carriageway)
  moves DOWN the frame. The lowest point of the vehicle in the image —
  the base of its front, its contact point — reaches the gate first, so
  the crossing frame is the **first** row of the blob.
- **Traffic away from the camera** moves UP the frame. Its roof and rear
  upper body cross the line first and its contact point last, so the
  crossing frame is the **last** row of the blob.

Take the first or last row of the *vehicle*, not of its shadow, its
spray or its trailer's overhang. Where the blob's edge is soft, take the
first row at which the vehicle is unmistakably present and record
`probable` if that row could plausibly be a few frames either way. In
every case the full-frame check settles it: step to the chosen frame and
the neighbouring ones and see where the contact point actually is
relative to the gate.

### One vehicle, one crossing

Each physical vehicle contributes at most one crossing row, ever, even
if it changes lane over the gate, reverses, or is occluded and reappears.
A vehicle that crosses, then backs up over the gate and crosses again in
the same window is still one crossing, labelled at its first crossing
frame, in its first crossing direction, with `confidence: probable` and
a note in the labelling record.

### Occluded and overlapping vehicles

The slit-scan merges two vehicles into one blob whenever they pass the
gate side by side or nose to tail within a frame or two. The tie-break,
in order:

1. Go to the full frame at the blob's row and count vehicles there, not
   in the slit-scan. Two distinct road-contact points over the gate
   means two crossings, even where the bodies overlap on screen.
2. If the two contact points cross on different frames, label each at
   its own frame. If they genuinely cross on the same frame — abreast in
   adjacent lanes — record two crossings with the **same** frame number
   and different ids. Equal frame numbers are legal in the schema for
   exactly this reason.
3. If the full frame cannot resolve how many vehicles are in the blob
   (a lorry fully hiding a car behind it, for example), step back and
   forward through the neighbouring frames until the vehicles separate,
   and count them there.
4. If they never separate anywhere in the window, label what can be seen
   — the vehicle whose contact point is visible — as `probable`, and
   record the ambiguity in the labelling record. Do not guess a second
   vehicle into existence, and do not delete a visible one because its
   neighbour is unclear.

A vehicle towing a trailer, or an articulated lorry, is **one** vehicle:
one contact point, one crossing, classed by its tractor unit.

### Vehicles only partly in frame

The rule is about the contact point, not the body:

- If the vehicle's road-contact point is visible in the frame and passes
  over the gate segment, it counts — even if the roof, the tail or half
  the body is cut off by the frame edge.
- If the contact point itself is outside the frame at the moment of
  crossing, or is hidden behind the frame edge, the vehicle does **not**
  count. Its crossing frame cannot be established from the evidence.
- A vehicle that enters or leaves the carriageway between the gate
  endpoints — a slip road, a lane change onto the shoulder — counts only
  if its contact point actually passes over the segment.
- A vehicle whose contact point passes beyond either gate **endpoint**
  (that is, off the ends of the segment rather than across it) does not
  count, however clearly it is visible.

## Confidence

Two values only.

- **`certain`** — the labeller can see the crossing in the slit-scan
  *and* confirm the vehicle, its class and its direction in the full
  frame, and would give the same answer on a second pass without
  hesitation. The crossing frame is pinned to within a frame or two.
- **`probable`** — the labeller believes a crossing happened but one of
  the three legs is weak: the blob is merged with a neighbour, the
  contact point is briefly hidden, the class is arguable (a large van
  against a small lorry), or the crossing frame could plausibly be
  several frames either side. Use `probable` rather than dropping the
  crossing, and rather than promoting a guess to `certain`.

If neither applies — if the labeller cannot say a vehicle crossed at all
— there is no row. A missing row is an honest statement; an invented
`probable` is not.

Accuracy figures computed against this label set must report the
`certain`-only figure alongside the all-rows figure. A number quoted
against `certain` rows alone, without saying so, silently drops the hard
cases and flatters the engine.

## Scoring tolerance

Fixed here, in the document that states the labelling rules, for the same
reason the labelling rules themselves are fixed before any labelling: a
tolerance chosen after seeing the engine's output is a tolerance chosen
to make the engine look good.

A prediction matches a label when it names the same gate, carries the
same direction, and its frame lies in the closed, **asymmetric** interval
`[label - 1, label + 4]`. Matching is one-to-one: a label is consumed by
at most one prediction and a prediction by at most one label, resolved
nearest-frame first, so a burst of predictions cannot be scored against a
single label. An unmatched prediction is a false positive; an unmatched
label is a miss.

An earlier version of this section fixed a **symmetric ±2 frames**. That
number was wrong, and it is corrected here rather than quietly replaced,
because a scoring tolerance that changes without an argument attached is
exactly the thing this section exists to prevent.

**Why the interval is asymmetric.** It encodes a known bias in the
**labels**, not slack for the engine.

- The crossing frame of a label is machine-proposed from the **first row
  of the vehicle's blob** in a drift-stabilised slit-scan, then
  human-confirmed. A vehicle's **shadow reaches the gate band before its
  tyres do**, so the first blob row is systematically **early**.
- `LABELLING_RECORD.md` measures that lead frame-by-frame against the
  footage on two isolated vehicles and states the resulting label
  precision as **+0/-4 frames**: a label is never late, and may be up to
  four frames early.
- The engine fires on the bottom-centre anchor — the road-contact point,
  the tyres — which is the thing the label is early relative to. So a
  correct engine prediction sits **0 to 4 frames after** its label.

A symmetric ±2 window therefore scores a correct 3-frames-late prediction
as a **miss and a false alarm at once**, understating accuracy twice
over. The `-1` side is the ordinary sub-frame slack of a frame-quantised
crossing; the `+4` side is the measured label lead, and nothing more.

**This is still not permission to widen the window.** The warning the
earlier version gave is right and stands: a prediction consistently
late or early by a fixed amount is a finding about the engine, and a
tolerance wide enough to hide it would also be wide enough to match a
prediction to the wrong vehicle. That warning is aimed at **symmetric
widening chosen after seeing engine output**, which this is not — the
+0/-4 figure was measured and written down in `LABELLING_RECORD.md`
before any scoring code existed, and the interval is asymmetric precisely
because it tracks that measurement rather than buying slack in both
directions.

**The window stays effectively disjoint between neighbouring vehicles.**
The closest pair of labels in this set is **5 frames** apart (411 and
416). Their windows are `[410, 415]` and `[415, 420]`: they touch at
exactly one frame, and one-to-one nearest-frame-first matching makes that
touch harmless, since the frame can be consumed by only one of them. A
symmetric ±4 window — the naive way to admit a 4-frame lead — would give
`[407, 415]` and `[412, 420]`, genuinely overlapping across four frames.
The fix is the asymmetry, not a wider window.

Any scorer must publish the **signed** frame offsets of its matched
pairs. The asymmetry is a claim that predictions run late; the offsets
are the evidence for or against it, and a scorer that hid them could
widen its window indefinitely without anyone noticing.

Any report must record this interval as a **pair** of numbers with the
reason attached. A single scalar "tolerance" field would misdescribe the
scorer to every later reader.

`probable` rows carry no more frame precision than `certain` rows — the
flag records doubt about the crossing, its class or its exact frame —
so they are matched under the same window and reported separately rather
than under a looser one.

## Class vocabulary

Exactly the five names in `trafficlens.core.classes.VEHICLE_CLASSES`:

`bicycle`, `car`, `motorcycle`, `bus`, `truck`

Guidance for the boundary cases on a motorway clip:

- A van, a pickup, a box truck, a tipper, an articulated lorry and its
  trailer are all `truck`.
- A coach is a `bus`.
- A car towing a caravan or a trailer is a `car`.
- Anything the labeller cannot place in one of the five names is
  `probable` on its nearest class, with a note; it is never given a
  sixth class name, because a name outside this vocabulary is rejected
  by the loader.

## Direction vocabulary

Exactly the two labels the gate itself carries: `away` and `toward`. The
`inbound` gate runs left to right at a constant image y, so the side up
the frame — away from the camera — is `away`, and the side down the
frame — toward the camera — is `toward`. Ordinary traffic on this
carriageway is `toward`. A direction string outside the gate's own two
labels is rejected by the loader.

Direction is read from the full frame, from where the vehicle came from
and went to across the gate — not from the slit-scan, which shows only
that something passed.

## Procedure

1. Generate the review images:

   ```
   PYTHONPATH=src python scripts/make_gt_slitscan.py \
       --config configs/motorway.yaml --gate inbound \
       --start-frame 0 --end-frame 734 --out private/gt
   ```

   This writes the full slit-scan and a set of overlapping row-banded
   tiles into `private/gt/`, which is git-ignored. Review images are
   never committed: they are large, they are derived, and anything
   committed alongside a label set invites the label set being edited to
   match a picture rather than the footage.

2. Read the tiles top to bottom. Each tile carries a frame-number axis
   down its left edge, so a blob's row is read directly as a frame
   index. Tiles overlap by 20 frames so that no crossing is cut in half
   by a tile boundary; a crossing appearing in the overlap of two tiles
   is one crossing, not two.

3. Write down every candidate row: a frame index for each blob.

4. Confirm the candidates against the footage:

   ```
   PYTHONPATH=src python scripts/make_gt_slitscan.py \
       --config configs/motorway.yaml --gate inbound \
       --start-frame 0 --end-frame 734 --out private/gt \
       --candidates 37,64,102,...
   ```

   This writes contact sheets of those full frames with the gate drawn.
   For each candidate: is it a vehicle, which class, which direction, and
   is the crossing frame right? Step to neighbouring frames where the
   answer is not clean.

5. Write the label file (see the schema below) and load it through
   `trafficlens.bench.slitscan.GroundTruth.load`, which enforces every
   rule in this document that can be checked mechanically. It rejects a
   crossing outside the window, a duplicate id, an unknown class, a
   direction that is not one of the gate's two labels, a confidence
   outside `{certain, probable}`, frames that go backwards, a gate whose
   endpoints have moved by more than 0.5 px, a window whose last frame
   the clip's decoder never produces, and a clip name or frame rate that
   does not match the real file. A label set that does not load is not a
   label set.

6. Record, alongside the label file: who labelled it, on what date, how
   long it took, and every case that was ambiguous and why. The
   ambiguous cases are the part of the record that a later reader needs
   most.

## Label file schema

`schema: 1`. All fields are required. Unknown fields are rejected.

```json
{"schema": 1, "clip": "motorway-a40.webm", "fps": 30.0,
 "window": {"start_frame": 0, "end_frame": 734},
 "gate": {"name": "inbound", "start": [0.06, 0.80], "end": [0.46, 0.80]},
 "protocol": "data/groundtruth/PROTOCOL.md",
 "labeller": "<who>", "labelled_on": "<date>",
 "crossings": [{"id": 1, "frame": 37, "class": "car", "direction": "toward", "confidence": "certain"}]}
```

The block above is the schema, with a placeholder crossing to show the
shape of a row. It is not a label set and no number in it is a claim
about the footage.

Field notes:

- `clip` is the file *name*, not a path; it is checked against the real
  file the label set is loaded against.
- `fps` must match the clip's container frame rate to within 0.01.
- `window` is inclusive at both ends, and its `end_frame` must be a
  frame the clip's decoder actually produces — the loader plays the clip
  forward to check, rather than trusting the container's advertised
  frame count, which is only an upper bound.
- `gate.start` / `gate.end` are normalized `[0, 1]` coordinates and must
  match the gate the label set is scored against to within **0.5 px**
  once converted at the clip's frame size: the same gate written down
  with rounded decimals still matches, a gate that was moved does not.
- `crossings` are ordered by frame, non-decreasing. Ids are unique
  positive integers.
- `protocol` points at this file, so a label set always carries the rules
  it was made under.

## How the slit-scan is sampled

Stated here so a reviewer can reproduce the exact review images, and so
a change in the sampling is visible as a change to this document.

- The strip for one frame is `samples` points spaced evenly along the
  gate segment from `start` to `end` inclusive, at parameter
  `t = i / (samples - 1)`.
- At each of those points, `thickness_px` pixels are read along the
  direction perpendicular to the gate, at integer offsets centred on the
  gate line, and averaged. An even `thickness_px` therefore straddles
  the line; an odd one includes the line itself.
- Each pixel is read by **nearest neighbour**, rounding the sample
  coordinate **half up** — `floor(v + 0.5)`, so 3.5 becomes 4 — and
  clamping to the frame bounds, so an offset that leaves the frame
  repeats the edge pixel. Clamped, never wrapped and never reflected.
- Nearest neighbour, not bilinear, is used deliberately. The review
  image should show the pixels the camera recorded, not a blend of them:
  bilinear interpolation dims and smears a small, fast, high-contrast
  object — exactly the vehicle that is hardest to label — and the
  perpendicular average already supplies all the smoothing the strip
  needs. Nearest neighbour is also exactly reproducible with integer
  indexing, so the same clip yields byte-identical review images on any
  platform and any OpenCV build.
- The averaged value is rounded **half up** — `floor(v + 0.5)`, not
  numpy's half-to-even `np.round`, so an average of exactly 4.5 becomes
  5 — and cast back to the frame's own dtype, so the strip is directly
  writable as an image.
- Rows are stacked in increasing frame order, so the vertical axis of
  the slit-scan is time running downward.
