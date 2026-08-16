/** The browser's `letterbox`, mirroring `trafficlens.detect.base.letterbox`
 * to the bit.
 *
 * This module carries the single hardest parity requirement in the project,
 * and the reason is worth stating once, here, because it is not obvious and it
 * cost a task to find.
 *
 * `letterbox` resizes with `cv2.INTER_LINEAR_EXACT`, never plain
 * `cv2.INTER_LINEAR`. That is not a style preference. OpenCV's `hal::resize`
 * begins with `CALL_HAL(resize, cv_hal_resize, ...)`, handing a vendor HAL
 * first refusal before any OpenCV code runs, and the build this project is
 * developed against reports `Custom HAL: YES (carotene, KleidiCV)`. Measured
 * on that build, a faithful port of OpenCV's own documented fixed-point
 * INTER_LINEAR matches `cv2` byte for byte on 36 of 40 random shapes and on
 * 0/31518 pixels of each single-axis sweep -- then diverges wholesale in a
 * sharp band where both axes downscale by less than 3x. This product's own
 * shape, 1280x720 into 480, sits in that band: 34% of pixels differ on a noise
 * frame. No port can match that, because what it would have to match is a
 * vendor's NEON kernel rather than a specification, and a different wheel is a
 * different answer again.
 *
 * `INTER_LINEAR_EXACT` is OpenCV's own answer to that problem: integer Q8.8
 * arithmetic throughout, with the two coefficients per output index computed
 * in `softdouble` -- a software float64, deterministic by construction. The
 * HAL does not claim it. `resizeBitExact` below is that algorithm, and it
 * reproduces `cv2.INTER_LINEAR_EXACT` on 0 differing pixels out of 989706
 * across 240 random shapes, including 0 of 388800 at the shipped shape.
 *
 * Everything else here is `letterbox`'s nine documented steps in order. The
 * two that bite are step 3, which rounds with CPython's half-to-even rather
 * than `Math.round` (see `roundHalfEven`), and step 5, which floors the pad so
 * an odd remainder lands at the right and bottom. */

import { roundHalfEven } from "../engine/numeric";
import { DETECT_DEFAULT_INPUT_SIZE, LETTERBOX_PAD_VALUE } from "../generated/constants";

/** Q8.8: `ufixedpoint16`'s 1.0. */
const ONE_Q8 = 256;
const MAX_U16 = 0xffff;

export interface LetterboxGeometry {
  /** The single factor both axes are scaled by. */
  readonly scale: number;
  readonly resizedWidth: number;
  readonly resizedHeight: number;
  /** Left and top pad only; an odd remainder goes right and bottom. */
  readonly padX: number;
  readonly padY: number;
}

export interface LetterboxResult extends LetterboxGeometry {
  /** `(1, 3, size, size)` float32, CHW, RGB, in [0, 1]. */
  readonly tensor: Float32Array;
  readonly size: number;
}

/** Steps 1-3 and 5: everything about a letterbox that does not touch pixels.
 *
 * Split out because it is the half Task 21 needs to reason about, and because
 * the shipped 1280x720 case can be asserted here without materialising a
 * 2.7 MB tensor. */
export function letterboxGeometry(
  width: number,
  height: number,
  size: number = DETECT_DEFAULT_INPUT_SIZE,
): LetterboxGeometry {
  const scale = Math.min(size / width, size / height);
  const resizedWidth = roundHalfEven(width * scale);
  const resizedHeight = roundHalfEven(height * scale);
  return {
    scale,
    resizedWidth,
    resizedHeight,
    // Python's `//` on non-negative ints; the odd pixel lands right/bottom
    // because only the left/top pad is ever computed.
    padX: Math.floor((size - resizedWidth) / 2),
    padY: Math.floor((size - resizedHeight) / 2),
  };
}

interface AxisTable {
  /** Source index of the first of the two taps, per destination index. */
  readonly offsets: Int32Array;
  /** Q8.8 weights of the two taps, interleaved. */
  readonly weights: Uint16Array;
  /** Destination indices below this clamp to the first source sample. */
  readonly min: number;
  /** Destination indices at or above this clamp to the last source sample. */
  readonly max: number;
}

/** `interpolationLinear<uint8_t>::getCoeffs`.
 *
 * `scale` is formed as the reciprocal of the inverse scale, exactly as OpenCV
 * does -- `1 / (dst / src)` is not always the same float64 as `src / dst`, and
 * a coefficient one ULP out changes a pixel. */
function axisTable(sourceSize: number, destinationSize: number): AxisTable {
  const scale = 1 / (destinationSize / sourceSize);
  const offsets = new Int32Array(destinationSize);
  const weights = new Uint16Array(destinationSize * 2);
  let min = 0;
  let max = destinationSize;

  for (let d = 0; d < destinationSize; d += 1) {
    const position = scale * (d + 0.5) - 0.5;
    const whole = Math.floor(position);
    if (whole >= 0 && sourceSize > 1) {
      if (whole < sourceSize - 1) {
        offsets[d] = whole;
        // `ufixedpoint16(softdouble)`: cvRound, which is half-to-EVEN, then
        // clamped into uint16. Never `Math.round`, which is half-up.
        const upper = Math.min(roundHalfEven((position - whole) * ONE_Q8), MAX_U16);
        weights[d * 2] = ONE_Q8 > upper ? ONE_Q8 - upper : 0;
        weights[d * 2 + 1] = upper;
      } else {
        offsets[d] = sourceSize - 1;
        max = Math.min(max, d);
      }
    } else {
      min = Math.max(min, d + 1);
    }
  }
  return { offsets, weights, min, max };
}

/** One source row into a Q8.8 line of `destinationWidth * channels` samples --
 * `hlineResizeCn<uint8_t, ufixedpoint16, 2, true, cn>`. */
function horizontalLine(
  source: Uint8Array,
  rowStart: number,
  channels: number,
  table: AxisTable,
  destinationWidth: number,
  out: Uint16Array,
): void {
  const { offsets, weights, min, max } = table;
  for (let d = 0; d < min; d += 1) {
    for (let c = 0; c < channels; c += 1) {
      out[d * channels + c] = (source[rowStart + c] as number) << 8;
    }
  }
  for (let d = min; d < max; d += 1) {
    const base = rowStart + (offsets[d] as number) * channels;
    const w0 = weights[d * 2] as number;
    const w1 = weights[d * 2 + 1] as number;
    for (let c = 0; c < channels; c += 1) {
      const a = Math.min(w0 * (source[base + c] as number), MAX_U16);
      const b = Math.min(w1 * (source[base + channels + c] as number), MAX_U16);
      out[d * channels + c] = Math.min(a + b, MAX_U16);
    }
  }
  if (max < destinationWidth) {
    const base = rowStart + (offsets[destinationWidth - 1] as number) * channels;
    for (let d = max; d < destinationWidth; d += 1) {
      for (let c = 0; c < channels; c += 1) {
        out[d * channels + c] = (source[base + c] as number) << 8;
      }
    }
  }
}

/** `cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR_EXACT)` for 8-bit
 * interleaved images. Reproduces it exactly; see the module comment for the
 * measurement and for why the approximate variant is unusable here. */
export function resizeBitExact(
  source: Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  channels: number,
  destinationWidth: number,
  destinationHeight: number,
): Uint8Array {
  const expected = sourceWidth * sourceHeight * channels;
  if (source.length !== expected) {
    throw new Error(
      `source has ${source.length} bytes, expected ${expected} for ` +
        `${sourceWidth}x${sourceHeight}x${channels}`,
    );
  }
  const x = axisTable(sourceWidth, destinationWidth);
  const y = axisTable(sourceHeight, destinationHeight);
  const lineLength = destinationWidth * channels;
  const out = new Uint8Array(destinationHeight * lineLength);

  // Two Q8.8 line buffers, reused. Rows are produced in ascending order and
  // each vertical tap pair is (iy, iy + 1), so at most two horizontal lines are
  // ever live -- the same reuse `resize_bitExactInvoker` does with its
  // `linebuf`, and the reason this does not allocate per row.
  const lines = [new Uint16Array(lineLength), new Uint16Array(lineLength)];
  const cached = [-1, -1];

  const lineFor = (row: number): Uint16Array => {
    const slot = row % 2;
    const buffer = lines[slot] as Uint16Array;
    if (cached[slot] !== row) {
      horizontalLine(source, row * sourceWidth * channels, channels, x, destinationWidth, buffer);
      cached[slot] = row;
    }
    return buffer;
  };

  const writeFlat = (line: Uint16Array, destinationRow: number): void => {
    const base = destinationRow * lineLength;
    for (let i = 0; i < lineLength; i += 1) {
      // `vlineSet`: a Q8.8 sample straight to uint8, rounded.
      out[base + i] = Math.min(((line[i] as number) + 128) >> 8, 255);
    }
  };

  for (let d = 0; d < destinationHeight; d += 1) {
    if (d < y.min) {
      writeFlat(lineFor(0), d);
    } else if (d >= y.max) {
      writeFlat(lineFor(sourceHeight - 1), d);
    } else {
      const top = y.offsets[d] as number;
      const b0 = y.weights[d * 2] as number;
      const b1 = y.weights[d * 2 + 1] as number;
      const first = lineFor(top);
      const second = lineFor(Math.min(top + 1, sourceHeight - 1));
      const base = d * lineLength;
      for (let i = 0; i < lineLength; i += 1) {
        // Q16.16 accumulation, saturating, then rounded to uint8. Divided
        // rather than shifted: the accumulator can exceed 2**31, where `>>`
        // would wrap into a negative. The division is exact -- the numerator
        // is an integer below 2**33 and the divisor a power of two -- and
        // `| 0` truncates it, which for a non-negative value is the floor the
        // shift would have produced.
        const accumulated = Math.min(
          (first[i] as number) * b0 + (second[i] as number) * b1,
          0xffffffff,
        );
        out[base + i] = Math.min((accumulated + 32768) / 65536, 255) | 0;
      }
    }
  }
  return out;
}

/** `letterbox` over pixels the browser already holds: RGB, row-major, uint8,
 * no alpha -- what a canvas hands back once the alpha channel is dropped.
 *
 * Python receives BGR from OpenCV and converts to RGB at step 7; a real
 * `<video>` frame is already RGB, so the conversion has nowhere to happen and
 * this takes RGB directly. The two agree because the resize is per-channel and
 * so does not care about channel order. */
export function letterboxRgb(
  rgb: Uint8Array,
  width: number,
  height: number,
  size: number = DETECT_DEFAULT_INPUT_SIZE,
): LetterboxResult {
  const expected = width * height * 3;
  if (rgb.length !== expected) {
    throw new Error(`frame has ${rgb.length} bytes, expected ${expected} for ${width}x${height}`);
  }
  const geometry = letterboxGeometry(width, height, size);
  const { resizedWidth, resizedHeight, padX, padY } = geometry;
  const resized = resizeBitExact(rgb, width, height, 3, resizedWidth, resizedHeight);

  // Steps 6, 8 and 9 fused: the canvas is filled with the pad grey and the
  // resized image written into it, straight into CHW float32 rather than via
  // an intermediate HWC uint8 canvas. Same arithmetic, one pass.
  const plane = size * size;
  const tensor = new Float32Array(3 * plane);
  tensor.fill(LETTERBOX_PAD_VALUE / 255.0);

  for (let row = 0; row < resizedHeight; row += 1) {
    const sourceRow = row * resizedWidth * 3;
    const destinationRow = (padY + row) * size + padX;
    for (let column = 0; column < resizedWidth; column += 1) {
      const source = sourceRow + column * 3;
      const destination = destinationRow + column;
      tensor[destination] = (resized[source] as number) / 255.0;
      tensor[plane + destination] = (resized[source + 1] as number) / 255.0;
      tensor[2 * plane + destination] = (resized[source + 2] as number) / 255.0;
    }
  }
  return { ...geometry, tensor, size };
}

/** The pixel source a frame can be letterboxed from. `HTMLVideoElement` is the
 * live case; the others are what tests and offscreen pipelines use. */
export type FrameSource =
  | HTMLVideoElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageBitmap;

function sourceSize(source: FrameSource): { width: number; height: number } {
  if (typeof HTMLVideoElement !== "undefined" && source instanceof HTMLVideoElement) {
    // videoWidth/Height are the decoded frame; width/height are CSS pixels and
    // would silently letterbox the wrong thing on a styled element.
    return { width: source.videoWidth, height: source.videoHeight };
  }
  const sized = source as { width: number; height: number };
  return { width: sized.width, height: sized.height };
}

let scratch: OffscreenCanvas | HTMLCanvasElement | undefined;
let scratchContext: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | undefined;

/** Read a frame's pixels at their natural size and letterbox them.
 *
 * The scratch canvas is sized to the SOURCE, never to `size`: letting
 * `drawImage` do the scaling would hand the resampling to the browser, whose
 * filter is implementation-defined and is not the one Python uses. Every
 * resample happens in `resizeBitExact`. */
export function letterbox(
  source: FrameSource,
  size: number = DETECT_DEFAULT_INPUT_SIZE,
): LetterboxResult {
  const { width, height } = sourceSize(source);
  if (width === 0 || height === 0) {
    throw new Error("frame source has no decoded pixels yet");
  }
  if (scratch === undefined || scratch.width !== width || scratch.height !== height) {
    scratch =
      typeof OffscreenCanvas === "undefined"
        ? document.createElement("canvas")
        : new OffscreenCanvas(width, height);
    scratch.width = width;
    scratch.height = height;
    scratchContext = undefined;
  }
  if (scratchContext === undefined) {
    // willReadFrequently keeps the surface on the CPU; without it every
    // getImageData round-trips the GPU and costs more than the inference.
    scratchContext = (scratch as HTMLCanvasElement).getContext("2d", {
      willReadFrequently: true,
    }) as CanvasRenderingContext2D;
    if (scratchContext === null || scratchContext === undefined) {
      throw new Error("could not obtain a 2d context to read frame pixels");
    }
  }
  const context = scratchContext;
  context.drawImage(source as CanvasImageSource, 0, 0);
  const { data } = context.getImageData(0, 0, width, height);

  // RGBA to RGB. The alpha channel is opaque for video and carries nothing.
  const rgb = new Uint8Array(width * height * 3);
  for (let pixel = 0, target = 0; target < rgb.length; pixel += 4, target += 3) {
    rgb[target] = data[pixel] as number;
    rgb[target + 1] = data[pixel + 1] as number;
    rgb[target + 2] = data[pixel + 2] as number;
  }
  return letterboxRgb(rgb, width, height, size);
}
