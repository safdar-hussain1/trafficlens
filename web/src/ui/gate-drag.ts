/** Grabbing and moving the gate.
 *
 * The gate is the only geometry on the page a visitor can change, and changing
 * it changes what gets counted -- so what happens under the pointer is decided
 * here, in pure arithmetic, rather than inside an event handler where it could
 * only be judged by eye.
 *
 * Two rules exist because the obvious implementation gets them wrong. An
 * endpoint always wins over the body: a pointer on an endpoint is also, by
 * definition, on the line, so "nearest thing wins" would make a gate impossible
 * to re-aim. And a body drag is clamped as one object rather than per endpoint,
 * because clamping the endpoints independently silently shortens a gate pushed
 * into a corner -- and a shorter gate spans fewer lanes, which is a change to
 * the measurement, not to the picture. */

import type { Point } from "../engine/geometry";

export interface Segment {
  readonly start: Point;
  readonly end: Point;
}

export interface FrameSize {
  readonly width: number;
  readonly height: number;
}

export type GrabKind = "start" | "end" | "body";

/** What was grabbed, and the state it was grabbed in. The ORIGIN segment is
 * kept so every pointer position during one drag is resolved against the
 * segment as it was when the drag began: resolving against the live segment
 * compounds rounding and clamping, and a gate dragged into an edge and back
 * would not return to where it started. */
export interface Grab {
  readonly kind: GrabKind;
  readonly origin: Segment;
  readonly pointer: Point;
}

/** Grab radius in FRAME pixels (the video's own coordinates, not CSS pixels),
 * so the handle is the same size relative to the gate at every display size.
 * The caller converts. */
export const GATE_HANDLE_RADIUS_PX = 14;

/** The shortest gate a drag may produce. `Gate` itself throws on zero length --
 * a zero-length gate can never be crossed -- so the drag has to stop short of
 * it rather than hand the engine something it refuses. */
export const GATE_MIN_LENGTH_PX = 24;

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** Distance from p to the CLOSED segment a-b: perpendicular where the foot
 * falls inside, and to the nearer endpoint where it does not. */
export function distanceToSegment(p: Point, a: Point, b: Point): number {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const lengthSquared = abx * abx + aby * aby;
  if (lengthSquared === 0) {
    return Math.hypot(p[0] - a[0], p[1] - a[1]);
  }
  const t = clamp(((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / lengthSquared, 0, 1);
  return Math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby));
}

/** What the pointer is over: an endpoint, the segment body, or nothing.
 *
 * Both bounds are inclusive -- a pointer exactly at the radius grabs -- because
 * the alternative is a one-pixel dead ring that only shows up as "it sometimes
 * doesn't grab". */
export function hitTestGate(
  segment: Segment,
  point: Point,
  radius: number = GATE_HANDLE_RADIUS_PX,
): GrabKind | null {
  const toStart = Math.hypot(point[0] - segment.start[0], point[1] - segment.start[1]);
  const toEnd = Math.hypot(point[0] - segment.end[0], point[1] - segment.end[1]);
  if (toStart <= radius || toEnd <= radius) {
    return toStart <= toEnd ? "start" : "end";
  }
  if (distanceToSegment(point, segment.start, segment.end) <= radius) {
    return "body";
  }
  return null;
}

/** Push a moved endpoint back out to the minimum length if the drag collapsed
 * the gate onto its other end. The direction is taken from the requested
 * position where there is one, so the endpoint stays on the side the pointer
 * went to; only an exact collision falls back to the drag's own direction, and
 * a drag with no direction at all falls back to the segment's. */
function keepMinimumLength(
  moved: Point,
  fixed: Point,
  dx: number,
  dy: number,
  fallback: Point,
): Point {
  const length = Math.hypot(moved[0] - fixed[0], moved[1] - fixed[1]);
  if (length >= GATE_MIN_LENGTH_PX) {
    return moved;
  }
  const candidates: [number, number][] = [
    [moved[0] - fixed[0], moved[1] - fixed[1]],
    [dx, dy],
    [fallback[0] - fixed[0], fallback[1] - fixed[1]],
    [1, 0],
  ];
  const direction = candidates.find(([x, y]) => Math.hypot(x, y) > 0) as [number, number];
  const norm = Math.hypot(direction[0], direction[1]);
  return [
    fixed[0] + (direction[0] / norm) * GATE_MIN_LENGTH_PX,
    fixed[1] + (direction[1] / norm) * GATE_MIN_LENGTH_PX,
  ];
}

/** Move `kind` by (dx, dy) and return the resulting segment, clamped into the
 * frame and never shorter than `GATE_MIN_LENGTH_PX`.
 *
 * This is the single movement mechanism: the pointer path and the keyboard path
 * both go through it, so a gate nudged with the arrow keys obeys exactly the
 * same clamps as one dragged with a finger. */
export function moveGate(
  segment: Segment,
  kind: GrabKind,
  dx: number,
  dy: number,
  frame: FrameSize,
): Segment {
  if (kind === "body") {
    // Clamp the TRANSLATION, not the endpoints: the gate keeps its length and
    // its angle, and simply stops when its leading end reaches the edge.
    const minX = Math.min(segment.start[0], segment.end[0]);
    const maxX = Math.max(segment.start[0], segment.end[0]);
    const minY = Math.min(segment.start[1], segment.end[1]);
    const maxY = Math.max(segment.start[1], segment.end[1]);
    const tx = clamp(dx, -minX, frame.width - maxX);
    const ty = clamp(dy, -minY, frame.height - maxY);
    return {
      start: [segment.start[0] + tx, segment.start[1] + ty],
      end: [segment.end[0] + tx, segment.end[1] + ty],
    };
  }

  const movedFrom = kind === "start" ? segment.start : segment.end;
  const fixed = kind === "start" ? segment.end : segment.start;
  const requested: Point = [
    clamp(movedFrom[0] + dx, 0, frame.width),
    clamp(movedFrom[1] + dy, 0, frame.height),
  ];
  const lengthened = keepMinimumLength(requested, fixed, dx, dy, movedFrom);
  const moved: Point = [
    clamp(lengthened[0], 0, frame.width),
    clamp(lengthened[1], 0, frame.height),
  ];
  return kind === "start" ? { start: moved, end: fixed } : { start: fixed, end: moved };
}

/** Begin a drag at `point`, or return null if it grabbed nothing. */
export function beginDrag(
  segment: Segment,
  point: Point,
  radius: number = GATE_HANDLE_RADIUS_PX,
): Grab | null {
  const kind = hitTestGate(segment, point, radius);
  return kind === null ? null : { kind, origin: segment, pointer: point };
}

/** Where the gate is now, for a pointer at `point` during `grab`.
 *
 * Measured from the grab's own origin, so the grabbed point stays under the
 * pointer for the whole drag and nothing jumps on the first move. */
export function applyDrag(grab: Grab, point: Point, frame: FrameSize): Segment {
  return moveGate(
    grab.origin,
    grab.kind,
    point[0] - grab.pointer[0],
    point[1] - grab.pointer[1],
    frame,
  );
}
