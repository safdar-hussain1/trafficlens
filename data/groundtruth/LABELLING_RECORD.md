# Labelling record — `motorway_inbound_gt.json`

This record exists so a reader can judge how much weight the label set carries,
and reproduce or contest any single decision. It documents what was
human-adjudicated, what was machine-derived, and what was rejected and why.
`PROTOCOL.md` states the rules; this states how they were applied.

## What a human decided, and what a machine proposed

| Field | Source |
|---|---|
| **Existence** of each crossing | **Human.** Every entry was confirmed by looking at the full frame at that time with the gate drawn, in the gate-band crop. Two proposals were rejected on this basis (below). |
| **Class** | **Human**, read from the same full-frame crop. |
| **Direction** | **Human**, but trivially: this carriageway runs toward the camera throughout the clip, and no reversed crossing occurs. |
| **Frame** | **Machine-derived**, from the first row of the vehicle's blob in a drift-stabilised slit-scan. See the precision statement below. |
| **Confidence** | **Human.** `probable` marks entries whose blob overlaps a neighbour's, or which sit at the clip boundary. |

The proposal step used raw stacked gate-strip pixels only — no detector, no
tracker, no `GateCounter`, no threshold from the system under test. That
satisfies `PROTOCOL.md`'s independence rule. It is nonetheless a machine
proposal, and this document says so rather than presenting the frames as
hand-read.

## Frame precision — the honest limit

The crossing frame is the first slit-scan row at which the vehicle's road
contact enters the gate band. That convention was verified frame-by-frame
against the footage on two isolated vehicles:

- dark saloon, blob 321–365: wheels reach the gate at f322–325, roof clears at f355–360
- Audi, blob 473–530: wheels at f475–476, roof clears at f520–524

**A vehicle's shadow can lead its true tyre contact by 1–4 frames**, so each
frame value should be read as accurate to about **+0/−4 frames**, not exact.
Any scorer must use a matching tolerance at least this wide; `PROTOCOL.md`
fixes the tolerance to be used.

This convention is the opposite of what an earlier draft of the tooling brief
asserted. That draft claimed the roof reaches the gate first for approaching
traffic, which is geometrically backwards — a vehicle moving down the image
contacts the gate band with its wheels first. Proposals generated under the
wrong convention were **discarded**, not corrected; they were late by 8–78
frames (0.3–2.6 s).

## Rejections

| Proposal | Frame | Decision | Reason |
|---|---|---|---|
| #14 | 531 | **Rejected** | Duplicate of the crossing at f527 — the same white DAF articulated lorry. Its trailer produces a second blob; one vehicle, one crossing. |
| #17 | 659 | **Rejected** | No vehicle's road contact inside the proposed span. The blob is the rear edge of the trailer belonging to the lorry already recorded at f647. |

Both rejections remove *duplicate structure from articulated vehicles*, which is
the characteristic failure of blob proposal on this footage and is the reason a
human adjudication step is not optional.

## Counts

17 crossings: **13 cars, 4 trucks**, all `toward`. **7 are `certain`; 10 are
`probable`** (verified by loading the file, not counted by hand).

## How to use this set, and how not to

- Compute accuracy against the **certain-only subset** and against the **full
  set**. The two bracket the truth. Publishing only the flattering one is a
  misuse of the `confidence` field.
- The gate was chosen for label reliability — near-field, best-separated
  traffic — so any figure measured here is an **upper bound** on what the same
  engine scores on the far carriageway, in the distance, or in a queue. That
  caveat travels with the number.
- Occlusion rises sharply after roughly frame 400: of the nine crossings before
  it, six are `certain`; of the eight after, most are `probable`. A per-half
  breakdown is more informative than a single figure.

## Outstanding

The second clip's label set (`street_gt.json`) does not exist. Nothing has been
done toward it — no window, no gate, no protocol section. Any claim about the
street clip is unsupported until it is labelled under this same protocol.
