/** Turning measurements into the strings the control room prints.
 *
 * Centralised for one reason: the page's claim is that every number on it was
 * measured on the visitor's own machine, and the fastest way to break that
 * claim is for one readout to print a plausible zero where it has nothing. So
 * "no measurement yet" formats as an em dash everywhere, and the one case that
 * is not an absence -- an uncalibrated source, where the engine refuses to
 * derive a speed from pixels -- says `no speed` in words instead. */

/** Narrow no-break space (U+202F), the thousands group separator used here. A
 * comma is a decimal separator across most of Europe, including where the
 * flagship clip was filmed. */
export const NARROW_NBSP = " ";

/** What the page prints where it has nothing to print. */
export const NO_VALUE = "—";

export interface Cadence {
  /** Detect one source frame in every `stride`. 1 = every frame. */
  readonly stride: number;
  /** The phrase the page shows, e.g. `every 2nd frame`. */
  readonly label: string;
  /** False until a real per-frame time has been measured. */
  readonly measured: boolean;
  /** The cap `stride` is held to; exposed so a caller can say when it is hit. */
  readonly maxStride: number;
}

/** Beyond this, thinning further stops helping: the tracker's association is
 * call-clocked, so a stride this long already means each detection sees a scene
 * that moved a long way, and the trails stop being continuous motion. */
export const MAX_DETECT_STRIDE = 8;

function isCount(value: number): boolean {
  return Number.isFinite(value) && Number.isInteger(value) && value >= 0;
}

/** A whole count, grouped in threes. Anything that is not a count -- negative,
 * fractional, NaN, infinite -- prints as the em dash rather than being coerced
 * into a number the page never counted. */
export function formatCount(value: number): string {
  if (!isCount(value)) {
    return NO_VALUE;
  }
  const digits = String(value);
  let out = "";
  for (let i = 0; i < digits.length; i += 1) {
    const remaining = digits.length - i;
    if (i > 0 && remaining % 3 === 0) {
      out += NARROW_NBSP;
    }
    out += digits[i];
  }
  return out;
}

/** Milliseconds per frame, without a unit. One decimal is meaningful at the
 * tens of milliseconds a GPU takes; at hundreds it is noise, so it is dropped
 * rather than printed as false precision. */
export function formatMs(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) {
    return NO_VALUE;
  }
  return ms >= 100 ? String(Math.round(ms)) : ms.toFixed(1);
}

export function formatFps(fps: number | null): string {
  if (fps === null || !Number.isFinite(fps)) {
    return NO_VALUE;
  }
  return fps.toFixed(1);
}

/** Elapsed clip time as m:ss. */
export function formatClock(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) {
    return NO_VALUE;
  }
  const whole = Math.floor(seconds);
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${String(whole % 60).padStart(2, "0")}`;
}

/** A speed, or the reason there is not one.
 *
 * `calibrated` is about the SOURCE, not about the value: an uncalibrated camera
 * prints `no speed` even if a number were somehow handed in, because the engine
 * refuses to turn pixels into metres without a survey and the interface has to
 * tell the same truth. On a calibrated source a null is an ordinary
 * not-yet-measured, which is the em dash. */
export function formatSpeed(kmh: number | null, calibrated: boolean): string {
  if (!calibrated) {
    return "no speed";
  }
  if (kmh === null || !Number.isFinite(kmh)) {
    return NO_VALUE;
  }
  return `${Math.round(kmh)} km/h`;
}

/** The middle value, averaging the two middles of an even-length sample. Does
 * not disturb the caller's array. */
export function median(values: readonly number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1
    ? (sorted[middle] as number)
    : ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2;
}

/** The median of the last `capacity` samples.
 *
 * A median rather than a mean, and this is the whole point of the class: the
 * first inference of a session includes shader compilation and can cost a
 * hundred times the steady-state frame. A mean would carry that stall into
 * every figure the page prints for the rest of the session. */
export class RollingMedian {
  readonly capacity: number;
  private readonly samples: number[] = [];

  constructor(capacity: number) {
    if (!Number.isInteger(capacity) || capacity < 1) {
      throw new Error(`capacity must be a positive integer, got ${capacity}`);
    }
    this.capacity = capacity;
  }

  get count(): number {
    return this.samples.length;
  }

  /** Record a sample. A non-finite one is dropped: it is not a measurement,
   * and one NaN would make every median from here on NaN. */
  push(value: number): void {
    if (!Number.isFinite(value)) {
      return;
    }
    this.samples.push(value);
    if (this.samples.length > this.capacity) {
      this.samples.shift();
    }
  }

  value(): number | null {
    return median(this.samples);
  }

  /** Forget every sample. Called when the thing being timed changes -- a new
   * source has a different frame size and therefore a different per-frame cost,
   * and carrying the old machine's numbers into the new one would report a
   * measurement of something that is no longer running. */
  reset(): void {
    this.samples.length = 0;
  }
}

const ORDINALS = new Map<number, string>([
  [2, "2nd"],
  [3, "3rd"],
]);

/** How often to run the detector, given what a frame actually costs here.
 *
 * Detection is decoupled from render: the page draws every frame and detects on
 * whatever cadence the measured backend can sustain. WebGPU on this clip runs
 * ahead of real time and detects every frame; single-threaded wasm cannot, and
 * saying so is better than dropping to a slideshow and letting the visitor
 * conclude the engine is broken. Before any measurement exists the answer is
 * "every frame" -- a guess would be a number nobody measured. */
export function decideCadence(msPerFrame: number | null, sourceFps: number): Cadence {
  const budgetMs = 1000 / sourceFps;
  const measured = msPerFrame !== null && Number.isFinite(msPerFrame) && msPerFrame > 0;
  const raw = measured ? Math.ceil((msPerFrame as number) / budgetMs) : 1;
  const stride = Math.min(MAX_DETECT_STRIDE, Math.max(1, raw));
  const label =
    stride === 1 ? "every frame" : `every ${ORDINALS.get(stride) ?? `${stride}th`} frame`;
  return { stride, label, measured, maxStride: MAX_DETECT_STRIDE };
}
