# Calibration, and why the flagship clip gets no speed

`configs/motorway.yaml` ships with **no calibration block**. The engine therefore
returns nothing for every speed on `motorway-a40.webm`, and no absolute km/h figure
from that clip is published anywhere in this project — not in the README, not on the
site, not in this document. `reports/parity.json` does carry one, from a matrix
reconstructed there to hand two engines the same plane: an instrument, not a
calibration.

That is not an omission. It is the result of a survey that went looking for a scale
anchor in this footage and did not find one, and this document publishes the survey
rather than the conclusion alone. Matched controls are what make "not measurable" a
measurement instead of a shrug, so they are here too.

Every figure below is read from `reports/speed_real.json`, and
`tests/test_docs_numbers.py` fails if this document and that file disagree.

---

## What a speed needs, and what this clip has

Speed comes from a road-plane homography: pixels map to metres on the road surface,
the tracker's anchor is transferred through that map, and a least-squares slope per
world axis gives velocity. The map is fitted from surveyed correspondences — image
points whose real-world positions are known.

For a motorway filmed from an overpass, the only repeating structure with a known
real-world spacing is usually the **lane-divider marking**. That is what the original
survey used, assuming the German `Leitlinie` geometry of a 6 m painted stroke in an
18 m period. The along-road scale, and therefore every km/h the product would print,
rested entirely on that 18 m.

Three independent problems ended that.

### 1. The stated justification is falsified by the clip's own paint

Measured in a perspective-free coordinate, so foreshortening is removed before the
ratio is taken, the painted stroke is about **0.50** of the period. A 6 m stroke in an
18 m period requires 0.333. It is visually unmistakable in the near field: the paint
is as long as the gap.

Centroid-to-centroid spacing does not depend on how a period divides into stroke and
gap, so this does not disprove the 18 m *period* by itself — but it removes the
period's entire stated basis. The three innocent explanations for a duty cycle that
high — smear from the camera's own drift, resolution smearing at distance, and two
different marking types in one frame — were each tested against the footage, and the
report records that each fails.

### 2. Nothing else in the clip can anchor the along-road scale

**Source:** `reports/speed_real.json`.
**Protocol:** each candidate was measured with a vanishing-point transfer identity
that needs no cross-road calibration, validated against synthetic cameras to machine
precision, and — for the guardrail posts — against matched controls processed
identically. Camera roll, the method's one vulnerability, is bounded at 1.97 degrees.
5 candidate anchors were measured, which is every repeating structure the frame
contains; none is usable.

| candidate | verdict |
|---|---|
| Guardrail (Schutzplanke) posts | PRESENT BUT NOT MEASURABLE |
| Delineator posts (Leitpfosten) | PRESENT AND MEASURABLE BUT UNUSABLE |
| Median barrier segment joints | ABSENT |
| Other repeating along-road structure | ABSENT |
| A second marking periodicity (divider 2) | PRESENT AND MEASURABLE AND CONTRADICTORY |

The median is a raised earth and gravel bank carrying a steel W-beam, not a precast
concrete barrier, so it has no segment joints to measure. The other repeating
structure is not there: one sign gantry, one direction sign, three round signs, all
singletons.

The delineator posts are the interesting near-miss. Four of them on the right verge
transfer to 68 to 82 m per interval, mean about 74 m, across every plausible horizon
row. That matches no German delineator spacing — the standard is 50 m on straights and
25 m in curves — and the three intervals rise monotonically by 15%, which is not
scatter. The section visibly curves and the two far posts sit in the bend, so the
standard being compared against is not even fixed; roll moves this particular transfer
by up to 17%; and two pixels of error on the far posts is 10% of the outermost
interval on its own. It is evidence **against** 18 m, not evidence for any
replacement.

### 3. The guardrail posts are present but statistically indistinguishable from asphalt

**Source:** `reports/speed_real.json`.
**Protocol:** band-limited local period in the perspective-free coordinate
`u = 1 / (vp_x - x)`, over four windows across the image, for the guardrail post band
and for four matched controls processed identically — two featureless asphalt patches,
the shoulder, and the region behind the rail — plus a positive control on the
divider-1 dashes, whose period is known.

| band | mean period, u | spread |
|---|---:|---:|
| guardrail post band | 3.72e-05 | 9.1 |
| asphalt control A | 3.01e-05 | 31.8 |
| asphalt control B | 3.3e-05 | 22.8 |
| shoulder control | 3.05e-05 | 14.6 |
| behind-rail control | 3.32e-05 | 14.2 |
| positive control: divider-1 dashes | 0.0003639 | 1.3 |

The spread column is the percentage variation across the four windows. The posts sit
among the controls, their peaks repeatedly land on the edge of the search band — the
signature of broadband noise rather than a line — and a full-span comb correlation
scores the asphalt control *higher* than the posts.

**The positive control is what makes this a measurement.** The same method recovers
the known dash period to 1.3%, so it works where a periodicity exists. The posts
simply are not resolvable in this footage.

---

## The shipped correspondences contradicted themselves

This part is independent of whether 18 m is the right absolute period, and it is what
removed the calibration block rather than merely doubting it.

Both dividers' surveyed dashes are consecutive and uniform. Under one road plane and
one along-road scale they must give the same step. They do not.

**Source:** `reports/speed_real.json`.
**Protocol:** divider 2's dash step re-measured under the scale calibrated on divider
1's assumed 18 m ladder, at every plausible horizon row, with divider 1's own fit
residual reported beside it so that a row cannot be chosen to flatter the answer.

| horizon row, px | divider 1 fit RMS, m | divider 2 step, m | ratio |
|---:|---:|---:|---:|
| 340 | 1.2352 | 26.7832 | 1.4880 |
| 356 | 0.7549 | 26.5448 | 1.4747 |
| 364 | 0.4876 | 26.3968 | 1.4665 |
| 372.468 | 0.2094 | 26.2137 | 1.4563 |
| 375.65 | 0.1556 | 26.1366 | 1.4520 |
| 380 | 0.2336 | 26.0229 | 1.4457 |
| 400 | 1.1905 | 25.3417 | 1.4079 |

Divider 2 reads about 26 m per step where divider 1 reads 18.0, a ratio of **1.4079**
to **1.4880** across the whole range. Two dashed dividers on the same carriageway
cannot differ by that much.

The escape hatch is closed and was checked rather than assumed: the horizon row that
would make divider 2 read 18 m is 463, which drives divider 1's own fit residual from
a fifth of a metre to 8.28 m and sits far below the visible horizon, so it is not a
horizon row. Roll can move the divider-2 figure by at most about 3%. There is no
reconciliation.

**And the fit's own residual hid all of this.** The removed calibration scored a mean
self-fit residual of 0.0746 m — over correspondences that disagree with each other by
a factor of about 1.45.

**Source:** `reports/speed_real.json`.
**Protocol:** the residuals of the calibration that used to ship, kept for the
record. Self-fit is against the correspondences the plane was fitted to; holdout is
against points held out of the fit.

| residual | mean, m | worst, m |
|---|---:|---:|
| self-fit, against its own correspondences | 0.0746 | 0.1310 |
| holdout, against points kept out of the fit | 0.3770 | 0.6069 |

The out-of-sample figure is several times the in-sample one however the comparison is
drawn: five times mean against mean, and 4.6 times worst against worst. The report's
own note on these two rows says "about three times worse", which compares the holdout
MEAN with the self-fit WORST case — 0.3770 against 0.1310 — a like-for-unlike pairing
that reads milder than either like-for-like one. Every column is in the table above so
the reader can pick; none of the three readings rescues the fit. That gap was already a
warning in the config's own comment before any of the above was measured — the same
failure family this repository has met more than once: a fit scoring well against its
own points while the data underneath is mutually inconsistent.

## The repair was attempted and is impossible

The obvious fix is to drop the disputed divider-2 correspondences and refit on divider
1 alone with an honest holdout. That cannot work, and the impossibility is pinned by a
committed test rather than argued for: divider 1's six surveyed points lie on one
straight line in the image — straight to 0.041 px over a 354 px span — and at one
cross-road offset in the world. A collinear configuration determines no homography at
all, and `cv2.findHomography` returns nothing for it. **The disputed divider-2 points
were the only cross-road information the survey ever had.**

---

## Two convergences, recorded as suggestive and explicitly not as anchors

Both halves of each have to be said together, or the pair becomes an anchor by
implication.

**The 12.2 m convergence.** If the true period were the roughly 12.2 m the delineator
reading points to, the measured 0.499 stroke-to-period duty puts the painted stroke at
**6.08 m** — exactly the standard 6 m. Under the assumed 18 m the same measurement
puts the stroke at **8.97 m**, which corresponds to no German marking. Two independent
lines therefore both say 18 m is too large, by about a third.

**And it is not an anchor**, because it rests entirely on the delineator measurement
that this same survey rejects as unsound — non-uniform by 15%, on a curving section,
matching no standard spacing. A number that depends on a rejected measurement cannot
become a calibration.

**The roadworks hypothesis.** A marking geometry matching no standard `Leitlinie`,
round speed-restriction signs whose digits cannot be read even upscaled eight times,
and separately measured slow dense traffic are all consistent with a roadworks
layout. That would explain the whole cluster at once.

**And it is a hypothesis, not an anchor.** German roadworks markings have their own
geometry, and adopting it would be another assumption of exactly the kind that
produced this problem in the first place.

---

## The honest bracket, and why it is not a speed figure

**Source:** `reports/speed_real.json`.
**Protocol:** the upper end is the period the removed calibration assumed; the lower
end is what the delineator reading implies. Scale error transfers to speed one for
one.

| quantity | value |
|---|---:|
| assumed along-road period, m | 18.0 |
| lower end of the honest bracket, m | 12.2 |
| implied band on every speed, % | -33 |

A figure published as "V km/h, somewhere between two thirds of V and V" is not a
speed figure. **Publishing nothing is the defensible product**, so:

- Every speed the engine reports for this clip is `None`.
- Stopped-vehicle detection never fires on it, because that feature requires
  calibrated speed by design.
- The speed-limit violation feature can only be demonstrated on this clip with a
  limit the user sets explicitly. The posted limit in the footage is not legible —
  upscaled eight times, the digits cannot be read — so no published figure references
  it and no speed distribution is validated against it.

---

## What the clip still licenses

The scale failure is specific. It removes absolute metric quantities and nothing
else, so the following remain fully measurable on this footage and are what the
counting benchmarks actually use:

Taken from the survey's own list, in `reports/speed_real.json`, of what an unanchored
along-road scale does and does not invalidate:

- counts and flow rate
- occupancy
- headway in seconds
- lane changes
- direction and wrong-way detection
- speed ratios between vehicles

---

## The one clean measurement the survey produced

Not everything failed. The guardrail beam itself is the best independent check on the
direction of the road available in this clip, it costs nothing, and it is committed as
a regression fixture at `data/fixtures/motorway_scale_survey.json`.

**Source:** `reports/speed_real.json`.
**Protocol:** the beam's row tracked per image column across 110 columns, fitted
robustly, and compared against the vanishing point the shipped homography implied.
Both the robust weighted residual and the plain one are published, because the robust
figure describes the beam and not every tracked column.

| quantity | value |
|---|---:|
| tracked columns | 110 |
| robust weighted residual RMS, px | 0.243 |
| plain residual RMS, px | 1.792 |
| worst plain residual, px | 6.356 |
| columns within 0.5 px of the fit | 93 |
| agreement with the surveyed vanishing point, px | 0.951 |

The rail is straight to a quarter of a pixel and parallel to the road to about one
pixel at the vanishing point. About a tenth of the tracked columns lock onto cast
shadow or dead vegetation under the beam and sit several pixels off, which is why the
plain residual is published beside the robust one rather than instead of it.
Straightness at that level also bounds lens distortion in the region every other
measurement in this investigation was made in.

---

## What would change this, and what would not

**Would:** a capture of the same site with a visible ground-truth baseline, or
GNSS-tracked probe-vehicle passes. Two gates a known distance apart do not count,
because on this clip that distance is itself known only through the scale in dispute.

**Would not:** more analysis of this clip. Two approaches are specifically dead and
should not be brought back without new evidence. Vehicle-footprint scale checks
measure the *cross-road* axis, and a homography admits an independent scale per world
axis, so they cannot corroborate the along-road period at all. And guardrail post
spacing is dead unless a higher-resolution or lower-viewpoint capture appears — the
matched controls above are the bar any revival has to clear.

---

## Calibrating your own camera

The engine's calibration path is not the problem; this clip is. If you can survey
your own scene, `trafficlens calibrate` explains the procedure and checks the result:

```bash
PYTHONPATH=src .venv/bin/python -m trafficlens.cli calibrate --config configs/webcam.yaml
```

Two things the checker enforces, both of which exist because a calibration that
validates itself is worthless. It **refuses** — rather than warning — when the
correspondence count equals the solve's degrees of freedom and no holdout is
supplied, because a four-point homography reproduces its own four points to about
1e-06 m however wrong the survey was. And it reports a held-out reprojection error,
because that is the only number that can fall.

Survey your along-road scale against something you have measured, not against a
standard you have inferred from the footage. That is the whole lesson of this
document.
