/** The geometry of the time-space diagram.
 *
 * Traffic engineering's own native chart: time along one axis, position along
 * the road on the other, one line per vehicle. Here "position along the road"
 * is the SIGNED perpendicular distance from the gate line, in image pixels,
 * which is what makes the picture and the counting rule the same object -- the
 * gate is the axis's zero, and a count is exactly a trajectory changing sign.
 * It is also the geometry of the slit-scan that produced this project's ground
 * truth, so the page's signature visual and its accuracy protocol are one idea.
 *
 * The sign convention is `geometry.sideOfLine`'s, not a second one invented
 * here, and `timespace.test.ts` asserts the two agree: if they drifted, a count
 * would appear on the opposite side of the axis from the vehicle that caused
 * it. */

import type { Point } from "../engine/geometry";

export interface Segment {
  readonly start: Point;
  readonly end: Point;
}

/** One point of a trajectory: a timestamp in seconds and a signed distance from
 * the gate line in pixels. */
export interface Sample {
  readonly t: number;
  readonly d: number;
}

export type Trace = readonly Sample[];

export interface TimeSpaceView {
  /** The right edge of the plot, in clip seconds. */
  readonly now: number;
  /** How many seconds the plot shows. */
  readonly windowS: number;
  readonly width: number;
  readonly height: number;
  /** Half the vertical range, in image pixels: the plot shows -spanPx..+spanPx. */
  readonly spanPx: number;
  readonly padding: {
    readonly left: number;
    readonly right: number;
    readonly top: number;
    readonly bottom: number;
  };
}

/** Signed perpendicular distance, in pixels, from the gate's infinite line.
 *
 * Positive is `sideOfLine`'s +1 -- left of the gate's own direction of travel,
 * which for the shipped left-to-right gates is up the frame, away from the
 * camera. */
export function signedDistanceToGate(gate: Segment, p: Point): number {
  const abx = gate.end[0] - gate.start[0];
  const aby = gate.end[1] - gate.start[1];
  const length = Math.hypot(abx, aby);
  if (length === 0) {
    return 0;
  }
  const cross = aby * (p[0] - gate.start[0]) - abx * (p[1] - gate.start[1]);
  const distance = cross / length;
  // Normalise -0 to 0: a point ON the line has no side, and letting a negative
  // zero out would give `Math.sign` a -0 that disagrees with `sideOfLine`'s 0.
  return distance === 0 ? 0 : distance;
}

/** Where p projects onto the gate, as a parameter running 0 at `start` to 1 at
 * `end`. Values outside [0, 1] are past the gate's ends and are returned
 * unclamped, because how far past is the useful part. */
export function gateParam(gate: Segment, p: Point): number {
  const abx = gate.end[0] - gate.start[0];
  const aby = gate.end[1] - gate.start[1];
  const lengthSquared = abx * abx + aby * aby;
  if (lengthSquared === 0) {
    return 0;
  }
  return ((p[0] - gate.start[0]) * abx + (p[1] - gate.start[1]) * aby) / lengthSquared;
}

/** True while p is between the gate's two ends.
 *
 * A gate is a bounded segment, not a line, so a vehicle on another carriageway
 * can cross the line and never be counted. The diagram draws that difference
 * rather than asserting it, which is why this is a separate question from the
 * distance above. */
export function withinGateSpan(gate: Segment, p: Point): boolean {
  const t = gateParam(gate, p);
  return t >= 0 && t <= 1;
}

/** Place one sample on the plot. Positive distance draws UP the canvas, so a
 * vehicle receding on the video climbs on the chart. */
export function projectSample(
  view: TimeSpaceView,
  t: number,
  d: number,
): { x: number; y: number } {
  const plotWidth = view.width - view.padding.left - view.padding.right;
  const plotHeight = view.height - view.padding.top - view.padding.bottom;
  const age = view.now - t;
  const x = view.padding.left + plotWidth * (1 - age / view.windowS);
  const centre = view.padding.top + plotHeight / 2;
  const y = centre - (d / view.spanPx) * (plotHeight / 2);
  return { x, y };
}

/** Vertical half-range for the plot: at least `floor`, otherwise the next step
 * up that contains everything drawn.
 *
 * Stepped rather than fitted: an axis refitted to the data every frame makes
 * every trajectory crawl as the scale breathes, which reads as motion the
 * traffic is not doing. */
export const SPAN_STEP_PX = 50;

export function chooseSpanPx(maxAbsDistance: number, floor: number): number {
  const wanted = Math.ceil(Math.max(0, maxAbsDistance) / SPAN_STEP_PX) * SPAN_STEP_PX;
  return Math.max(floor, wanted);
}

/** The samples the plot should draw for a trace: everything inside the window,
 * plus the last sample before it so the line enters from the left edge instead
 * of appearing there. */
export function trimWindow(trace: Trace, now: number, windowS: number): Sample[] {
  const cutoff = now - windowS;
  const kept: Sample[] = [];
  let carry: Sample | undefined;
  for (const sample of trace) {
    if (sample.t < cutoff) {
      carry = sample;
      continue;
    }
    if (kept.length === 0 && carry !== undefined) {
      kept.push(carry);
    }
    kept.push(sample);
  }
  return kept;
}

/** The two things a band rule can say about a trajectory, kept separate.
 *
 * `crossed` is the engine's rule: the signed distance changed sign, so the path
 * met the line. `touchedBand` is the baseline's rule: some sample landed within
 * `bandPx` of the line. They come apart in both directions, and the diagram
 * shows both cases happening to real vehicles -- a fast one that steps clean
 * over a thin band (crossed, never touched) and a slow one that sits inside it
 * without ever changing sign (touched, never crossed). That is why the
 * band-counting baseline both misses and double-fires, drawn rather than
 * argued. */
export function crossesBand(
  trace: Trace,
  bandPx: number,
): { crossed: boolean; touchedBand: boolean } {
  let touchedBand = false;
  let crossed = false;
  let lastSign = 0;
  for (const sample of trace) {
    if (Math.abs(sample.d) <= bandPx) {
      touchedBand = true;
    }
    const sign = Math.sign(sample.d);
    if (sign !== 0) {
      if (lastSign !== 0 && sign !== lastSign) {
        crossed = true;
      }
      lastSign = sign;
    }
  }
  return { crossed, touchedBand };
}
