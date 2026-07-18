# Speed calibration guide

Pixels are not metres. A car near the camera crosses hundreds of pixels
per second; the same car far away crosses a handful. Any "speed" computed
from raw pixel displacement is only correct at one distance from the
camera — everywhere else it is wrong, sometimes by 3–5×. TrafficLens
therefore refuses to invent speeds: until you calibrate, speed fields are
empty (never zero, never a guess).

## Option 1 — homography (recommended)

Mark **four points on the road surface** whose real-world rectangle you
know. TrafficLens computes the 3×3 projective transform that maps the
image road plane to metres and measures all movement in that space.

In the web UI: press **Calibrate speed**, click the four corners in
order — *near-left → near-right → far-right → far-left* — then enter the
rectangle's real width and length in metres.

In YAML (coordinates are fractions of the frame):

```yaml
calibration:
  mode: homography
  image_points: [[0.15, 0.95], [0.85, 0.95], [0.65, 0.35], [0.35, 0.35]]
  world_points: [[0, 0], [7, 0], [7, 18], [0, 18]]   # metres
```

### Where to find known distances on a road

| Reference | Typical size |
|---|---|
| One traffic lane | 3.0–3.7 m wide (3.5 m on most highways) |
| Lane-marking dash | 3 m painted (with 9 m gaps on highways; local rules vary) |
| Dash start → next dash start | ~12 m on highways |
| Zebra-crossing stripe | 0.5 m stripe + 0.5 m gap |
| A parked car | ~4.5 m long |

Pick points **on the road surface** (paint, kerb lines, manhole covers),
not on poles or fences — the homography is only valid for the plane the
points lie on. Spread the four points as wide and as deep as possible;
a tiny rectangle amplifies click error.

## Option 2 — metres per pixel (overhead cameras only)

If the camera looks nearly straight down, perspective distortion is
small and one scale factor is enough:

```yaml
calibration:
  mode: scale
  meters_per_pixel: 0.033
  reference_width: 768   # the frame width the factor was measured at
```

Measure it from anything of known size in frame: if a 3 m lane dash
spans 90 px, the factor is 3/90 = 0.033. `reference_width` makes the
factor survive a resolution change.

## How the estimate is computed

1. Each tracked object's **bottom-centre** (ground contact point) is
   mapped to road-plane metres every frame.
2. Speed = world displacement over a **0.5 s sliding window** (not
   frame-to-frame, which amplifies box jitter), smoothed with an EMA.
3. Objects that move less than `min_travel_m` (default 0.4 m) inside the
   window report **no speed** — bounding-box jitter on a parked car must
   not read as movement.
4. The speed attached to a crossing event is the estimate at the moment
   of crossing; per-class medians / 85th percentiles accumulate in the
   session summary. (The p85 is the metric traffic engineers actually
   use to set speed limits.)

## Sanity-checking your calibration

- Free-flowing urban traffic should median around the posted limit;
  highway p85 typically sits 5–15 km/h above it.
- If far-away vehicles read faster than near ones (or vice versa), the
  calibration points are off — re-click with more spread.
- Walking people are 4–6 km/h; a cyclist 12–25 km/h. Point the camera at
  a footpath and you have a free ground-truth check.

## Accuracy limits worth knowing

- The synthetic-camera tests in `tests/test_speed.py` recover known
  speeds within ~3% and stay within ~10% under 1.5 px of detector noise.
  Real-world error is dominated by calibration quality, not the maths.
- Everything is measured on the road plane. Tall trucks' boxes bounce
  more, so their instantaneous speeds are noisier than cars'.
- Extreme camera angles (near-horizontal) stretch the far field so much
  that a few pixels equal many metres; keep gates and calibration
  rectangles in the nearer two-thirds of the image.
