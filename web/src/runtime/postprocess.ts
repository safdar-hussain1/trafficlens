/** The browser's `decodeYolo` and `nms`, mirroring
 * `trafficlens.detect.base.decode_yolo` and `.nms`.
 *
 * PRECISION, AND WHY THIS MODULE IS THE OPPOSITE OF `web/src/engine/`.
 *
 * The engine modules beside this one -- geometry, gate, kalman, tracker, speed
 * -- mirror Python that is float64 from end to end, and reaching for
 * `Math.fround` in any of them would be a bug. This module is the other case.
 * `decode_yolo` receives the model's float32 output and numpy keeps float32
 * through every subsequent operation, because under NEP 50 the Python scalars
 * it mixes in (`conf`, `iou`, `scale`, `pad_x`, the literal `2.0`) are WEAK
 * and adopt the array's dtype rather than promoting it. So the box arithmetic,
 * the IoU, and both threshold COMPARISONS all happen in float32.
 *
 * The rule underneath both cases is the same one, and it is the rule to apply
 * to anything added here later: mirror the arithmetic the Python side actually
 * performs, not the arithmetic its function names suggest.
 *
 * Two measured consequences, both of which change a decision rather than an
 * ULP:
 *
 * - `float32(0.35)` is 0.3499999940395355. A prediction scoring exactly that
 *   passes numpy's `>= conf` and fails a float64 `>= 0.35`. Rounding `conf`
 *   once, up front, is what keeps that detection.
 * - Undoing the pad and scale in float32 moves an IoU that is exactly 0.5 in
 *   exact arithmetic to 0.5000001788139343, which is on the other side of
 *   `nms`'s strict `>`. A float64 mirror returns one more detection than
 *   Python on the committed boundary fixture.
 *
 * Every arithmetic step below is therefore wrapped in `Math.fround`. That is
 * exact, not approximate: float64 has more than twice float32's precision, so
 * a single add, subtract, multiply or divide of two float32 values, computed
 * in float64 and rounded once, is bit-identical to the float32 operation. */

import type { Detection } from "../engine/tracker";

const f32 = Math.fround;

export interface RawOutput {
  /** The model output, densely packed. */
  readonly data: Float32Array;
  /** `[1, 4 + nClasses, N]`, as onnxruntime reports it. */
  readonly dims: readonly number[];
}

export interface DecodeOptions {
  /** Minimum class score, INCLUSIVE, mirroring `score >= conf`. */
  readonly conf: number;
  /** Suppression threshold, STRICT, mirroring `iou_vals > iou`. */
  readonly iou: number;
  /** The classes to keep, and their names. `decode_yolo` names a survivor
   * `COCO_CLASSES[class_id]`, but it can only ever emit a class that passed
   * `keep_class_ids`, so carrying names for exactly those is the same
   * function over the reachable domain -- and avoids a hand-copied
   * 80-entry table drifting from the Python one. The argmax below still runs
   * over EVERY class, which is what makes this equivalent: a column whose best
   * class is one we do not keep is dropped, never re-assigned to its best
   * kept class. */
  readonly keepClasses: ReadonlyMap<number, string>;
}

/** Intersection over union of two `x1, y1, x2, y2` boxes, in float32, exactly
 * as `nms` computes it: degenerate sides clamp to zero, and a zero union
 * yields zero rather than a division. */
export function iouOf(
  a: readonly [number, number, number, number] | ArrayLike<number>,
  b: readonly [number, number, number, number] | ArrayLike<number>,
): number {
  const ax1 = f32(a[0] as number);
  const ay1 = f32(a[1] as number);
  const ax2 = f32(a[2] as number);
  const ay2 = f32(a[3] as number);
  const bx1 = f32(b[0] as number);
  const by1 = f32(b[1] as number);
  const bx2 = f32(b[2] as number);
  const by2 = f32(b[3] as number);

  const areaA = f32(f32(Math.max(0, f32(ax2 - ax1))) * f32(Math.max(0, f32(ay2 - ay1))));
  const areaB = f32(f32(Math.max(0, f32(bx2 - bx1))) * f32(Math.max(0, f32(by2 - by1))));

  const overlapWidth = f32(Math.max(0, f32(Math.min(ax2, bx2) - Math.max(ax1, bx1))));
  const overlapHeight = f32(Math.max(0, f32(Math.min(ay2, by2) - Math.max(ay1, by1))));
  const intersection = f32(overlapWidth * overlapHeight);

  const union = f32(f32(areaA + areaB) - intersection);
  return union > 0 ? f32(intersection / union) : 0;
}

/** Greedy NMS over boxes of ONE class, returning kept indices in the order
 * they were kept.
 *
 * Mirrors `nms`'s determinism rule exactly: boxes are visited by descending
 * score with ties broken by ascending original index -- numpy's
 * `argsort(-scores, kind="stable")`. A box that has itself been suppressed
 * never suppresses another.
 *
 * The `i - j` tie-break below is REDUNDANT, and knowingly so. The array being
 * sorted starts as `[0, 1, 2, ...]`, and `Array.prototype.sort` has been
 * required to be stable since ES2019, so equal scores already keep their
 * ascending-index order without it. Removing it is an equivalent mutant --
 * measured: all 190 tests still pass. It stays because it states the rule at
 * the point of use rather than resting on a language guarantee the reader has
 * to remember, and because it is the line that corresponds to numpy's
 * `kind="stable"`. It is documentation that happens to execute, not a
 * mechanism any test protects. */
export function nms(
  boxes: readonly (readonly [number, number, number, number])[],
  scores: readonly number[],
  iou: number,
): number[] {
  const threshold = f32(iou);
  const order = scores.map((_, index) => index);
  order.sort((i, j) => {
    const a = f32(-(scores[i] as number));
    const b = f32(-(scores[j] as number));
    return a === b ? i - j : a - b;
  });

  const suppressed = new Uint8Array(scores.length);
  const keep: number[] = [];
  for (const i of order) {
    if (suppressed[i] === 1) {
      continue;
    }
    keep.push(i);
    for (const j of order) {
      if (j === i || suppressed[j] === 1) {
        continue;
      }
      if (iouOf(boxes[i] as readonly number[], boxes[j] as readonly number[]) > threshold) {
        suppressed[j] = 1;
      }
    }
  }
  return keep;
}

/** Decode a raw `(1, 4 + nClasses, N)` YOLO11 output into detections in
 * ORIGINAL image coordinates, mirroring `decode_yolo` step for step: argmax
 * over the class rows, inclusive `conf` threshold, class filter, cxcywh to
 * xyxy with the letterbox undone, then class-wise NMS.
 *
 * Output order is ascending class id, then each class's own NMS keep order. */
export function decodeYolo(
  output: RawOutput,
  scale: number,
  padX: number,
  padY: number,
  options: DecodeOptions,
): Detection[] {
  const [batch, rows, columns] = output.dims as [number, number, number];
  if (batch !== 1) {
    throw new Error(`expected a batch of 1, got dims [${output.dims.join(", ")}]`);
  }
  if (output.data.length !== rows * columns) {
    throw new Error(
      `output has ${output.data.length} values, expected ${rows * columns} for ` +
        `[${output.dims.join(", ")}]`,
    );
  }
  const classCount = rows - 4;
  const { data } = output;

  const conf = f32(options.conf);
  const scale32 = f32(scale);
  const padX32 = f32(padX);
  const padY32 = f32(padY);
  const half = f32(2.0);
  // Hoisted out of the column loop: this runs 4725 times per frame at the
  // shipped input size, and allocating two closures per column is 9450
  // allocations a frame for nothing.
  const undoX = (v: number): number => f32(f32(v - padX32) / scale32);
  const undoY = (v: number): number => f32(f32(v - padY32) / scale32);

  const perClass = new Map<
    number,
    { boxes: [number, number, number, number][]; scores: number[] }
  >();

  for (let column = 0; column < columns; column += 1) {
    // numpy's argmax returns the FIRST maximum on a tie; `>` keeps the first.
    let bestClass = 0;
    let bestScore = data[4 * columns + column] as number;
    for (let k = 1; k < classCount; k += 1) {
      const score = data[(4 + k) * columns + column] as number;
      if (score > bestScore) {
        bestScore = score;
        bestClass = k;
      }
    }
    if (!(bestScore >= conf)) {
      continue;
    }
    const className = options.keepClasses.get(bestClass);
    if (className === undefined) {
      continue;
    }

    const cx = data[column] as number;
    const cy = data[columns + column] as number;
    const width = data[2 * columns + column] as number;
    const height = data[3 * columns + column] as number;
    const halfWidth = f32(width / half);
    const halfHeight = f32(height / half);

    const box: [number, number, number, number] = [
      undoX(f32(cx - halfWidth)),
      undoY(f32(cy - halfHeight)),
      undoX(f32(cx + halfWidth)),
      undoY(f32(cy + halfHeight)),
    ];

    let bucket = perClass.get(bestClass);
    if (bucket === undefined) {
      bucket = { boxes: [], scores: [] };
      perClass.set(bestClass, bucket);
    }
    bucket.boxes.push(box);
    bucket.scores.push(bestScore);
  }

  const detections: Detection[] = [];
  for (const classId of [...perClass.keys()].sort((a, b) => a - b)) {
    const { boxes, scores } = perClass.get(classId) as {
      boxes: [number, number, number, number][];
      scores: number[];
    };
    for (const index of nms(boxes, scores, options.iou)) {
      const box = boxes[index] as [number, number, number, number];
      detections.push({
        x1: box[0],
        y1: box[1],
        x2: box[2],
        y2: box[3],
        score: scores[index] as number,
        classId,
        className: options.keepClasses.get(classId) as string,
      });
    }
  }
  return detections;
}
