/** Drawing the camera view: the frame, the boxes, the trails and the gate.
 *
 * The video is deliberately the quieter half of the stage. Every YOLO demo is a
 * video with boxes on it, and boxes only show that detection happened; the
 * claim this product makes is about counting, and that claim is legible in the
 * diagram next door. So this canvas draws the minimum that lets a visitor
 * believe the diagram: what was detected, where it has been, and where the gate
 * is that they can move.
 *
 * The fit maths is separated out and tested because the pointer depends on it:
 * a gate is dragged in CSS pixels and lives in frame pixels, and if the two
 * disagree the handle drifts away from the finger. */

import type { CrossingEvent } from "../engine/gate";
import type { Point } from "../engine/geometry";
import type { TrackView } from "../engine/pipeline";
import type { Segment } from "./gate-drag";

export interface Size {
  readonly width: number;
  readonly height: number;
}

/** How a frame of `frame` size sits inside a `box`-sized canvas, letterboxed:
 * one scale for both axes, centred, never cropping. */
export interface Fit {
  readonly scale: number;
  readonly dx: number;
  readonly dy: number;
}

export function fitContain(frame: Size, box: Size): Fit {
  if (frame.width <= 0 || frame.height <= 0) {
    return { scale: 1, dx: 0, dy: 0 };
  }
  const scale = Math.min(box.width / frame.width, box.height / frame.height);
  return {
    scale,
    dx: (box.width - frame.width * scale) / 2,
    dy: (box.height - frame.height * scale) / 2,
  };
}

export function frameToBox(p: Point, fit: Fit): Point {
  return [p[0] * fit.scale + fit.dx, p[1] * fit.scale + fit.dy];
}

/** The inverse: a pointer position in the box's coordinates, in frame pixels.
 * Unclamped -- a drag that leaves the frame is a real event and the clamp
 * belongs to `moveGate`, which knows the frame's bounds. */
export function boxToFrame(p: Point, fit: Fit): Point {
  return [(p[0] - fit.dx) / fit.scale, (p[1] - fit.dy) / fit.scale];
}

export interface Palette {
  readonly structure: string;
  readonly structureQuiet: string;
  readonly alert: string;
  readonly text: string;
  readonly textDim: string;
  readonly trace: string;
  readonly panel: string;
  readonly sunk: string;
  readonly line: string;
}

const TOKENS: Record<keyof Palette, string> = {
  structure: "--structure",
  structureQuiet: "--structure-quiet",
  alert: "--alert",
  text: "--text",
  textDim: "--text-dim",
  trace: "--trace",
  panel: "--panel",
  sunk: "--sunk",
  line: "--line",
};

/** Read the live theme's colours off the document, so the canvases follow the
 * same tokens as the markup instead of carrying a second, drifting copy. */
export function readPalette(element: Element): Palette {
  const style = getComputedStyle(element);
  const out = {} as Record<keyof Palette, string>;
  for (const [key, token] of Object.entries(TOKENS) as [keyof Palette, string][]) {
    out[key] = style.getPropertyValue(token).trim() || "#888";
  }
  return out as Palette;
}

export type Trail = readonly { readonly t: number; readonly p: Point }[];

export interface OverlayScene {
  readonly frame: Size;
  readonly gate: Segment;
  readonly gateLabels: { readonly positive: string; readonly negative: string };
  readonly tracks: readonly TrackView[];
  readonly trails: ReadonlyMap<number, Trail>;
  readonly events: readonly CrossingEvent[];
  readonly now: number;
  /** Frame pixels per CSS pixel is derived from the fit; this is the device
   * pixel ratio the canvas backing store was sized at. */
  readonly dpr: number;
  readonly source: CanvasImageSource | null;
  readonly wrongWay: ReadonlySet<number>;
  /** When true the crossing marker is drawn at a fixed size instead of
   * expanding: the information is the marker, not its motion. */
  readonly reducedMotion: boolean;
}

/** How long a trail stays on the video, in seconds of clip time. */
export const TRAIL_SECONDS = 2.5;

/** How long a crossing marker stays on the video after it fires. */
export const MARKER_SECONDS = 1.5;

function line(ctx: CanvasRenderingContext2D, a: Point, b: Point, width: number, colour: string): void {
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.lineWidth = width;
  ctx.strokeStyle = colour;
  ctx.stroke();
}

/** Draw one frame of the camera view and return the fit the pointer must use. */
export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  box: Size,
  scene: OverlayScene,
  palette: Palette,
): Fit {
  ctx.save();
  ctx.setTransform(scene.dpr, 0, 0, scene.dpr, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  ctx.fillStyle = palette.sunk;
  ctx.fillRect(0, 0, box.width, box.height);

  const fit = fitContain(scene.frame, box);
  if (scene.source !== null) {
    ctx.drawImage(
      scene.source,
      fit.dx,
      fit.dy,
      scene.frame.width * fit.scale,
      scene.frame.height * fit.scale,
    );
  }

  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // Trails first, so a box is never hidden behind its own history.
  ctx.globalAlpha = 0.85;
  for (const trail of scene.trails.values()) {
    const visible = trail.filter((sample) => scene.now - sample.t <= TRAIL_SECONDS);
    if (visible.length < 2) {
      continue;
    }
    ctx.beginPath();
    visible.forEach((sample, index) => {
      const [x, y] = frameToBox(sample.p, fit);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.lineWidth = 2;
    ctx.strokeStyle = palette.trace;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // Boxes. Amber is reserved for an actual alert -- here, a track crossing
  // against the gate's expected direction -- and everything else is structure.
  ctx.font = "600 12px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.textBaseline = "top";
  for (const track of scene.tracks) {
    const alerting = scene.wrongWay.has(track.trackId);
    const colour = alerting ? palette.alert : palette.structure;
    const [x1, y1] = frameToBox([track.box[0], track.box[1]], fit);
    const [x2, y2] = frameToBox([track.box[2], track.box[3]], fit);
    ctx.lineWidth = 2;
    ctx.strokeStyle = colour;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${track.className} ${track.trackId}`;
    const width = ctx.measureText(label).width + 8;
    ctx.fillStyle = colour;
    ctx.fillRect(x1, Math.max(0, y1 - 16), width, 16);
    ctx.fillStyle = palette.panel;
    ctx.fillText(label, x1 + 4, Math.max(0, y1 - 16) + 2);
  }

  // The gate, over a dark casing so it reads against any footage.
  const a = frameToBox(scene.gate.start, fit);
  const b = frameToBox(scene.gate.end, fit);
  line(ctx, a, b, 7, "rgb(0 0 0 / 45%)");
  line(ctx, a, b, 3, palette.structure);

  // Direction labels, one on each side of the gate's midpoint, so the two
  // count columns in the panel below can be told apart on the picture.
  const midpoint: Point = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const length = Math.hypot(dx, dy) || 1;
  const normal: Point = [dy / length, -dx / length];
  ctx.font = "600 11px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.textAlign = "center";
  for (const [label, sign] of [
    [scene.gateLabels.positive, 1],
    [scene.gateLabels.negative, -1],
  ] as const) {
    const at: Point = [midpoint[0] + normal[0] * 18 * sign, midpoint[1] + normal[1] * 18 * sign];
    ctx.fillStyle = "rgb(0 0 0 / 45%)";
    const width = ctx.measureText(label).width + 8;
    ctx.fillRect(at[0] - width / 2, at[1] - 7, width, 14);
    ctx.fillStyle = palette.panel;
    ctx.fillText(label, at[0], at[1] - 5);
  }
  ctx.textAlign = "left";

  // Endpoint discs: the visible target for the two handle buttons sitting over
  // this canvas.
  for (const point of [a, b]) {
    ctx.beginPath();
    ctx.arc(point[0], point[1], 7, 0, Math.PI * 2);
    ctx.fillStyle = palette.panel;
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = palette.structure;
    ctx.stroke();
  }

  // Fresh crossings, at the point on the gate where they happened.
  for (const event of scene.events) {
    const age = scene.now - event.timestamp;
    if (age < 0 || age > MARKER_SECONDS) {
      continue;
    }
    const at = frameToBox([event.crossingX, event.crossingY], fit);
    ctx.globalAlpha = scene.reducedMotion ? 0.9 : 1 - age / MARKER_SECONDS;
    ctx.beginPath();
    ctx.arc(at[0], at[1], scene.reducedMotion ? 9 : 6 + age * 8, 0, Math.PI * 2);
    ctx.lineWidth = 2;
    ctx.strokeStyle = palette.structure;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  ctx.restore();
  return fit;
}
