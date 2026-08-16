/** Directional gate counting: turns tracked-object movement into counted
 * crossings, once per track, per class, per direction. Mirrors
 * `trafficlens.core.gate`.
 *
 * A `Gate` is a directed *finite* line segment, not an infinite line.
 * `GateCounter` watches each track's positions frame to frame and, the first
 * time -- and only the first time -- a track's swept path actually intersects
 * the bounded gate segment (checked with `segmentsIntersect`, not merely a
 * same-side/opposite-side test against the segment's infinite extension),
 * emits a `CrossingEvent` labelled by which side of the gate's direction of
 * travel the track ended up on, using the same left/right sign convention as
 * `geometry.ts`. A crossing that lands exactly on one of the gate's own
 * endpoints counts -- inclusive bounds -- matching `segmentsIntersect`'s
 * treatment of a shared endpoint or T-junction as an intersection. */

import {
  crossingDirection,
  segmentIntersectionParam,
  segmentsIntersect,
  sideOfLine,
} from "./geometry";
import type { Point } from "./geometry";

/** True only when a known speed strictly exceeds a set limit.
 *
 * Returns false whenever either argument is null -- an unknown speed or an
 * unset limit can never be "over" -- and uses a strict `>` comparison, so a
 * speed exactly at the limit is not a violation. This is the single place the
 * limit comparison is made; other layers must call this function rather than
 * re-implement the comparison. */
export function isOverLimit(
  speedKmh: number | null,
  limitKmh: number | null,
): boolean {
  if (speedKmh === null || limitKmh === null) {
    return false;
  }
  return speedKmh > limitKmh;
}

/** One gate crossing: a track's path crossed the gate line exactly once,
 * resolved to a single frame and a single direction. Frozen on creation, as
 * the Python original is a frozen dataclass. */
export interface CrossingEvent {
  readonly trackId: number;
  readonly className: string;
  readonly gate: string;
  readonly direction: string;
  readonly signedDirection: number;
  readonly frameIndex: number;
  readonly timestamp: number;
  readonly crossingX: number;
  readonly crossingY: number;
  readonly speedKmh: number | null;
  readonly isViolation: boolean;
}

export interface GateOptions {
  /** Name of the +1 (left of travel) side. Defaults to "in". */
  labelPositive?: string;
  /** Name of the -1 (right of travel) side. Defaults to "out". */
  labelNegative?: string;
  /** Optional hint -- one of the two labels -- for callers that want to flag
   * crossings against an unexpected direction; GateCounter does not use it. */
  expectedDirection?: string | null;
}

/** A directed line segment that crossings are counted against.
 *
 * `start -> end` fixes the direction of travel used by `sideOfLine`. */
export class Gate {
  readonly name: string;
  readonly start: Point;
  readonly end: Point;
  readonly labelPositive: string;
  readonly labelNegative: string;
  readonly expectedDirection: string | null;

  constructor(name: string, start: Point, end: Point, options: GateOptions = {}) {
    if (start[0] === end[0] && start[1] === end[1]) {
      throw new Error(
        `Gate ${JSON.stringify(name)} has zero length: start and end are both ` +
          `[${start[0]}, ${start[1]}]. A zero-length gate can never be crossed.`,
      );
    }
    this.name = name;
    this.start = start;
    this.end = end;
    this.labelPositive = options.labelPositive ?? "in";
    this.labelNegative = options.labelNegative ?? "out";
    this.expectedDirection = options.expectedDirection ?? null;
  }

  /** Build a Gate from normalized [0, 1] coordinates plus the pixel frame size
   * they are relative to, converting to pixel coordinates. The zero-length
   * check then runs on the CONVERTED coordinates, so a frame dimension of zero
   * is caught even when the normalized points differ. */
  static fromNormalized(
    name: string,
    start: Point,
    end: Point,
    width: number,
    height: number,
    options: GateOptions = {},
  ): Gate {
    const named: [string, Point][] = [
      ["start", start],
      ["end", end],
    ];
    for (const [pointName, point] of named) {
      const axes: [string, number][] = [
        ["x", point[0]],
        ["y", point[1]],
      ];
      for (const [axisName, value] of axes) {
        if (!(value >= 0.0 && value <= 1.0)) {
          throw new Error(
            `Gate ${JSON.stringify(name)} normalized ${pointName} ${axisName}=` +
              `${value} is out of range [0, 1]`,
          );
        }
      }
    }
    const pixelStart: Point = [start[0] * width, start[1] * height];
    const pixelEnd: Point = [end[0] * width, end[1] * height];
    return new Gate(name, pixelStart, pixelEnd, options);
  }
}

/** Counts directional crossings of one Gate, once per track ID.
 *
 * Once a track has produced a `CrossingEvent` for this gate, it never produces
 * another until `forget(trackId)` is called: a lingering or jittering track
 * counts exactly once, ever -- not once per direction change.
 *
 * A crossing only counts when the track's swept path genuinely intersects the
 * *bounded* gate segment. Two positions landing on opposite sides of the
 * gate's infinite line is necessary but not sufficient: e.g. a vehicle on a
 * different carriageway, crossing the drawn gate's line far outside its two
 * endpoints, is not counted. */
export class GateCounter {
  readonly gate: Gate;
  /** class name -> direction label -> count. */
  readonly totals = new Map<string, Map<string, number>>();

  // The three per-track records, named after the Python attributes they mirror
  // and reachable for the same reason: a counter that never forgot a track
  // would grow one permanent record per vehicle ever seen, and the only way to
  // assert that boundedness is to look. `forget` must empty all three.
  readonly _counted = new Set<number>();
  // Last non-zero side (+1/-1) each track was seen on, and the actual point it
  // was seen at. Needed because an anchor landing exactly on the gate line
  // makes sideOfLine (and crossingDirection) return 0, which must be resolved
  // on a later frame against the real previous side -- and the real previous
  // off-line position, for the bounded-segment check -- rather than lost.
  readonly _lastSide = new Map<number, number>();
  readonly _lastOffLinePoint = new Map<number, Point>();

  constructor(gate: Gate) {
    this.gate = gate;
  }

  update(
    trackId: number,
    className: string,
    prev: Point,
    curr: Point,
    frameIndex: number,
    timestamp: number,
    speedKmh: number | null = null,
    speedLimitKmh: number | null = null,
  ): CrossingEvent | null {
    const gateA = this.gate.start;
    const gateB = this.gate.end;
    const sidePrevActual = sideOfLine(gateA, gateB, prev);
    const sideCurr = sideOfLine(gateA, gateB, curr);

    let origin: Point | undefined;
    let signed: number;
    if (sidePrevActual !== 0) {
      // The normal case: prev itself is the last off-line position, so the
      // segment to bounds-check is prev -> curr.
      origin = prev;
      signed = crossingDirection(gateA, gateB, prev, curr);
    } else {
      // prev landed exactly on the gate line this frame: resolve against the
      // last off-line side (and position) remembered for this track, not
      // against 0 (which would silently drop the crossing) and not against
      // prev's on-line position (which would give the wrong segment to
      // bounds-check).
      origin = this._lastOffLinePoint.get(trackId);
      const last = this._lastSide.get(trackId);
      if (last === undefined || sideCurr === 0 || last === sideCurr) {
        signed = 0;
      } else {
        signed = sideCurr;
      }
    }

    if (sideCurr !== 0) {
      this._lastSide.set(trackId, sideCurr);
      this._lastOffLinePoint.set(trackId, curr);
    } else if (sidePrevActual !== 0) {
      this._lastSide.set(trackId, sidePrevActual);
      this._lastOffLinePoint.set(trackId, prev);
    }

    if (signed === 0 || this._counted.has(trackId)) {
      return null;
    }

    if (origin === undefined || !segmentsIntersect(origin, curr, gateA, gateB)) {
      // The infinite line was crossed, but the swept path never actually meets
      // the bounded gate segment -- e.g. a parallel carriageway crossing the
      // gate's line far past its ends.
      return null;
    }

    this._counted.add(trackId);

    let crossingX: number;
    let crossingY: number;
    const t = segmentIntersectionParam(origin, curr, gateA, gateB);
    if (t === null) {
      // Parallel/collinear relative to the gate line -- segmentsIntersect can
      // still be true here (collinear overlap), but there is no single
      // well-defined intersection point; fall back to the object's current
      // position rather than throw.
      crossingX = curr[0];
      crossingY = curr[1];
    } else {
      // segmentsIntersect already confirmed a genuine bounded intersection, so
      // t should already lie in [0, 1]; clamp defensively against
      // floating-point overshoot at the edges.
      const clamped = Math.max(0.0, Math.min(1.0, t));
      crossingX = origin[0] + clamped * (curr[0] - origin[0]);
      crossingY = origin[1] + clamped * (curr[1] - origin[1]);
    }

    const direction =
      signed === 1 ? this.gate.labelPositive : this.gate.labelNegative;
    const violation = isOverLimit(speedKmh, speedLimitKmh);

    let classTotals = this.totals.get(className);
    if (classTotals === undefined) {
      classTotals = new Map<string, number>();
      this.totals.set(className, classTotals);
    }
    classTotals.set(direction, (classTotals.get(direction) ?? 0) + 1);

    return Object.freeze({
      trackId,
      className,
      gate: this.gate.name,
      direction,
      signedDirection: signed,
      frameIndex,
      timestamp,
      crossingX,
      crossingY,
      speedKmh,
      isViolation: violation,
    });
  }

  /** Clear all memory of trackId, so a recycled tracker ID can be counted
   * again. */
  forget(trackId: number): void {
    this._counted.delete(trackId);
    this._lastSide.delete(trackId);
    this._lastOffLinePoint.delete(trackId);
  }

  total(): number {
    let sum = 0;
    for (const directions of this.totals.values()) {
      for (const count of directions.values()) {
        sum += count;
      }
    }
    return sum;
  }
}
