/** The time–space diagram: the page's signature element, drawn live.
 *
 * Time runs left to right; the vertical axis is signed distance from the gate
 * line, in image pixels. Each tracked vehicle is one line. The gate is the
 * axis's zero, so a trajectory touching zero IS the crossing -- the count and
 * the picture are the same event rather than two representations that have to
 * be trusted to agree.
 *
 * Three things are drawn that an ordinary chart would leave out, because each
 * one is a claim the page makes:
 *
 *   - the ±20 px band, so both failure modes of band counting are visible in
 *     the same picture: a fast vehicle stepping clean across it between two
 *     frames, and a slow one sitting inside it without ever changing sign;
 *   - a dashed trajectory while a vehicle is past the gate's ENDS, which is the
 *     difference between a line and a bounded segment -- the far carriageway
 *     crosses this axis constantly and is never counted;
 *   - a marker for every crossing the engine actually emitted, pointing the way
 *     it went, so direction is shape rather than colour.
 *
 * Colour is not carrying identity anywhere here. Vehicle class is read from the
 * table beside the chart, not from a hue, which is why a page with two accents
 * and a neutral ink can draw forty vehicles at once without inventing a
 * fortieth colour. */

import { BASELINE_BAND_PX } from "../generated/constants";
import type { CrossingEvent } from "../engine/gate";
import type { Palette, Size } from "./overlay";
import {
  chooseSpanPx,
  projectSample,
  trimWindow,
} from "./timespace";
import type { TimeSpaceView, Trace } from "./timespace";

/** One vehicle's trajectory in diagram space, plus whether each sample was
 * within the gate's span. */
export interface DiagramTrace {
  readonly trackId: number;
  readonly samples: Trace;
  /** Same length as `samples`; false where the vehicle was past the gate's
   * ends and so could never be counted. */
  readonly inSpan: readonly boolean[];
}

export interface DiagramScene {
  readonly now: number;
  readonly windowS: number;
  readonly traces: readonly DiagramTrace[];
  readonly events: readonly CrossingEvent[];
  readonly floorSpanPx: number;
  readonly dpr: number;
  readonly running: boolean;
}

const PADDING = { left: 46, right: 10, top: 12, bottom: 24 };

export const DIAGRAM_BAND_PX = BASELINE_BAND_PX;

function axisFont(size: number): string {
  return `400 ${size}px 'IBM Plex Mono', ui-monospace, monospace`;
}

export function drawDiagram(
  ctx: CanvasRenderingContext2D,
  box: Size,
  scene: DiagramScene,
  palette: Palette,
): void {
  const maxAbs = scene.traces.reduce((worst, trace) => {
    for (const sample of trace.samples) {
      worst = Math.max(worst, Math.abs(sample.d));
    }
    return worst;
  }, 0);

  const view: TimeSpaceView = {
    now: scene.now,
    windowS: scene.windowS,
    width: box.width,
    height: box.height,
    spanPx: chooseSpanPx(maxAbs, scene.floorSpanPx),
    padding: PADDING,
  };

  ctx.save();
  ctx.setTransform(scene.dpr, 0, 0, scene.dpr, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  ctx.fillStyle = palette.panel;
  ctx.fillRect(0, 0, box.width, box.height);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  const left = PADDING.left;
  const right = box.width - PADDING.right;
  const topOf = (d: number): number => projectSample(view, scene.now, d).y;

  // The band, first: everything else has to sit on top of it.
  const bandTop = topOf(DIAGRAM_BAND_PX);
  const bandBottom = topOf(-DIAGRAM_BAND_PX);
  ctx.fillStyle = palette.structureQuiet;
  ctx.fillRect(left, bandTop, right - left, bandBottom - bandTop);

  // Distance grid. Round pixel values, labelled in mono, deliberately quiet.
  ctx.font = axisFont(10);
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  const step = view.spanPx / 2;
  for (let d = -view.spanPx; d <= view.spanPx + 1e-9; d += step) {
    const y = topOf(d);
    if (d !== 0) {
      ctx.strokeStyle = palette.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(right, y);
      ctx.stroke();
    }
    ctx.fillStyle = palette.textDim;
    ctx.fillText(`${Math.round(d)}`, left - 6, y);
  }

  // Time axis: one tick per second of window, labelled in whole seconds before
  // now, so a visitor can read how long ago a crossing was.
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const bottom = box.height - PADDING.bottom;
  const tickEvery = scene.windowS <= 8 ? 2 : 4;
  for (let ago = 0; ago <= scene.windowS + 1e-9; ago += tickEvery) {
    const { x } = projectSample(view, scene.now - ago, 0);
    ctx.strokeStyle = palette.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, bottom);
    ctx.lineTo(x, bottom + 4);
    ctx.stroke();
    ctx.fillStyle = palette.textDim;
    ctx.fillText(ago === 0 ? "now" : `−${ago}s`, x, bottom + 6);
  }

  // The gate: the axis's zero.
  const zero = topOf(0);
  ctx.strokeStyle = palette.structure;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(left, zero);
  ctx.lineTo(right, zero);
  ctx.stroke();

  // Trajectories. Solid inside the gate's span, dashed past its ends: a vehicle
  // on the far carriageway crosses this zero without ever being counted, and
  // the picture has to be able to say why.
  ctx.lineWidth = 2;
  ctx.strokeStyle = palette.trace;
  for (const trace of scene.traces) {
    const samples = trimWindow(trace.samples, scene.now, scene.windowS);
    if (samples.length < 2) {
      continue;
    }
    const offset = trace.samples.length - samples.length;
    for (let i = 1; i < samples.length; i += 1) {
      const previous = samples[i - 1] as { t: number; d: number };
      const current = samples[i] as { t: number; d: number };
      const inSpan =
        (trace.inSpan[offset + i] ?? true) && (trace.inSpan[offset + i - 1] ?? true);
      const from = projectSample(view, previous.t, previous.d);
      const to = projectSample(view, current.t, current.d);
      ctx.setLineDash(inSpan ? [] : [3, 4]);
      ctx.globalAlpha = inSpan ? 0.95 : 0.5;
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  // Crossings: a triangle on the axis, pointing the way the vehicle went. The
  // direction is the shape, not the colour, so it survives being printed, being
  // read by a colour-blind visitor, and being looked at in either theme.
  for (const event of scene.events) {
    if (event.timestamp < scene.now - scene.windowS || event.timestamp > scene.now) {
      continue;
    }
    const { x } = projectSample(view, event.timestamp, 0);
    const up = event.signedDirection === 1;
    const size = 6;
    ctx.beginPath();
    if (up) {
      ctx.moveTo(x, zero - size);
      ctx.lineTo(x - size * 0.8, zero + size * 0.6);
      ctx.lineTo(x + size * 0.8, zero + size * 0.6);
    } else {
      ctx.moveTo(x, zero + size);
      ctx.lineTo(x - size * 0.8, zero - size * 0.6);
      ctx.lineTo(x + size * 0.8, zero - size * 0.6);
    }
    ctx.closePath();
    ctx.fillStyle = palette.structure;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = palette.panel;
    ctx.stroke();
  }

  // Axis title, once, in the corner it cannot collide with.
  ctx.save();
  ctx.translate(12, (PADDING.top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = axisFont(10);
  ctx.fillStyle = palette.textDim;
  ctx.fillText("px from gate", 0, 0);
  ctx.restore();

  if (!scene.running && scene.traces.length === 0) {
    ctx.font = axisFont(12);
    ctx.fillStyle = palette.textDim;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    // Above the band rather than through it: the placeholder must not sit on
    // top of the gate line, which is the one thing already drawn.
    ctx.fillText(
      "Trajectories draw here once the detector runs.",
      (left + right) / 2,
      PADDING.top + (bottom - PADDING.top) * 0.25,
    );
  }

  ctx.restore();
}
