/** Two-stage multi-object tracker over the constant-velocity Kalman filter.
 * Mirrors `trafficlens.track.tracker`.
 *
 * ByteTrack-style association adapted to per-class traffic tracking. One
 * `Tracker.update()` call is one video frame; the tracker owns all state (the
 * `KalmanBoxFilter` itself is stateless). Every rule below is exact and
 * deterministic, and the Python engine implements the same ones.
 *
 * Association algorithm (per `update()` call)
 * ---------------------------------------------
 * 1. Every live track is predicted one step; its `age` and `timeSinceUpdate`
 *    advance and its `box` becomes the predicted box.
 * 2. Stage 1: HIGH-confidence detections (`score >= highThresh`) against ALL
 *    live tracks, confirmed and tentative. Cost is `1 - IoU` between the
 *    predicted and detected boxes; a pair is barred (cost Infinity) when its
 *    squared Mahalanobis gating distance exceeds the chi-square 95% gate
 *    `KALMAN_GATING_CHI2_95_4DOF`, or when the classes differ.
 * 3. Stage 2: tracks left unmatched by stage 1 that are CONFIRMED, against
 *    LOW-confidence detections (`lowThresh <= score < highThresh`). IoU-only
 *    cost, no Mahalanobis bar (an occluded box is exactly the box the filter
 *    is most unsure about); the cross-class bar still applies.
 * 4. Unmatched HIGH-confidence detections start tentative tracks; unmatched
 *    low-confidence detections are discarded and can never start a track.
 * 5. Lifecycle: a tentative track that misses even ONE frame dies immediately
 *    (only an unbroken run of hits can confirm it, which keeps one-frame NMS
 *    ghosts out of the output); a tentative track with `hits >= minHits`
 *    becomes confirmed; a confirmed track dies when `timeSinceUpdate > maxAge`,
 *    so a detection gap of up to exactly `maxAge` frames survives on
 *    prediction alone and may re-associate.
 *
 * Cost / threshold semantics
 * --------------------------
 * `matchThresh` is an IoU FLOOR: a pair may match only when
 * `IoU >= matchThresh`. Both stages implement it as `assign(cost, maxCost)`
 * with `cost = 1 - IoU` and `maxCost = 1 - matchThresh`, and `assign` drops
 * every assigned pair with `cost > maxCost`. The same float64 expressions are
 * evaluated on both sides so borderline IoUs resolve identically.
 *
 * Class policy
 * ------------
 * Association never crosses classes -- implemented by barring cross-class
 * pairs in the cost matrix, NOT by per-class sub-trackers, so ID allocation
 * stays a single global deterministic sequence. A track's `className` is FIXED
 * at creation (its first detection's class) and never renamed: detector class
 * flicker must not rename an existing track. A flickered detection that cannot
 * match anything simply starts (or fails to start) its own track under the
 * usual rules.
 *
 * Output policy
 * -------------
 * `update()` returns only CONFIRMED tracks that were updated with a real
 * detection in THIS frame (`timeSinceUpdate === 0`), in ascending trackId
 * order. Tentative tracks are internal, and coasting confirmed tracks stay
 * internal while they coast: a frame's output is exactly the detector-backed
 * state of that frame. Deaths are invisible to the caller, so any per-track
 * bookkeeping outside the tracker must reap by last-seen against `maxAge`,
 * strictly `>`. The returned `Track` objects are the tracker's own live
 * records (not copies); they keep mutating on later `update()` calls, which
 * lets downstream consumers hold a reference and watch `history` grow.
 *
 * `Track.box` always reflects the Kalman state -- the corrected state on
 * matched frames, the predicted state while coasting -- so anchors move
 * smoothly through occlusions instead of jumping. `Track.history` gets the
 * anchor appended ONLY on frames with a real detection update (creation
 * included), never while coasting: downstream speed estimation must average
 * over measured motion, not over the filter's own extrapolation, or a long
 * coast would fabricate distance travelled.
 *
 * ID allocation
 * -------------
 * IDs start at 1 and increase monotonically for the tracker's lifetime; new
 * IDs are assigned in ascending detection-index order among the unmatched
 * high-confidence detections of the frame. `reset()` clears all tracks AND
 * restarts the counter at 1.
 *
 * `frameIndex` is the caller's frame clock, carried for interface parity with
 * downstream event stamping; association itself is clocked purely by
 * successive `update()` calls (one call = one frame). */

import {
  KALMAN_GATING_CHI2_95_4DOF,
  TRACK_HIGH_CONF,
  TRACK_LOW_CONF,
  TRACK_MATCH_IOU,
  TRACK_MAX_AGE,
  TRACK_MIN_HITS,
} from "../generated/constants";
import { assign, iouMatrix } from "./associate";
import { boxAnchor } from "./geometry";
import type { Point } from "./geometry";
import { KalmanBoxFilter, xyahToXyxy, xyxyToXyah } from "./kalman";
import { Matrix } from "./numeric";

/** One decoded, post-NMS detection in ORIGINAL image pixel coordinates.
 * Mirrors `trafficlens.detect.base.Detection`; Task 20's ONNX runtime produces
 * these. */
export interface Detection {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
  readonly score: number;
  readonly classId: number;
  readonly className: string;
}

export const STATE_TENTATIVE = "tentative";
export const STATE_CONFIRMED = "confirmed";

/** One tracked object's public state (see the module comment for the exact
 * box/history semantics). */
export class Track {
  trackId: number;
  className: string; // fixed at creation; never renamed by class flicker
  box: [number, number, number, number]; // xyxy, Kalman-state-derived
  score: number; // score of the most recent matched detection
  age: number; // frames since creation, inclusive
  hits: number; // total matched-detection updates, creation included
  timeSinceUpdate: number; // frames since the last matched detection
  state: string; // STATE_TENTATIVE or STATE_CONFIRMED
  readonly history: Point[] = []; // anchor per updated frame

  constructor(
    trackId: number,
    className: string,
    box: [number, number, number, number],
    score: number,
    age: number,
    hits: number,
    timeSinceUpdate: number,
    state: string,
  ) {
    this.trackId = trackId;
    this.className = className;
    this.box = box;
    this.score = score;
    this.age = age;
    this.hits = hits;
    this.timeSinceUpdate = timeSinceUpdate;
    this.state = state;
  }

  /** Bottom-centre of the current box: where the object meets the road. */
  get anchor(): Point {
    return boxAnchor(this.box[0], this.box[1], this.box[2], this.box[3]);
  }
}

/** Internal record pairing a public Track with its Kalman state. */
interface Live {
  track: Track;
  mean: Float64Array;
  cov: Matrix;
}

export interface TrackerOptions {
  highThresh?: number;
  lowThresh?: number;
  matchThresh?: number;
  maxAge?: number;
  minHits?: number;
}

/** Two-stage per-class multi-object tracker. All tunables default to the
 * generated constants (the source of truth the Python engine also reads). */
export class Tracker {
  readonly highThresh: number;
  readonly lowThresh: number;
  readonly matchThresh: number;
  readonly maxAge: number;
  readonly minHits: number;

  private readonly kf = new KalmanBoxFilter();
  private live: Live[] = []; // creation order == ascending trackId
  private nextId = 1;

  constructor(options: TrackerOptions = {}) {
    this.highThresh = options.highThresh ?? TRACK_HIGH_CONF;
    this.lowThresh = options.lowThresh ?? TRACK_LOW_CONF;
    this.matchThresh = options.matchThresh ?? TRACK_MATCH_IOU;
    this.maxAge = options.maxAge ?? TRACK_MAX_AGE;
    this.minHits = options.minHits ?? TRACK_MIN_HITS;
  }

  /** Drop every track and restart the ID counter at 1, restoring the exact
   * state of a freshly constructed Tracker. */
  reset(): void {
    this.live = [];
    this.nextId = 1;
  }

  // -- per-frame step --------------------------------------------------------

  /** Advance one frame: predict, associate in two stages, spawn and retire
   * tracks. Returns the confirmed tracks updated by a detection this frame,
   * ascending by trackId. */
  update(detections: readonly Detection[], _frameIndex: number): Track[] {
    // _frameIndex is the caller's clock only; association is call-clocked.
    const high = detections.filter((d) => d.score >= this.highThresh);
    const low = detections.filter(
      (d) => this.lowThresh <= d.score && d.score < this.highThresh,
    );

    // 1. Predict every live track one step; ageing happens here so an
    // unmatched track needs no second pass.
    for (const rec of this.live) {
      const [mean, cov] = this.kf.predict(rec.mean, rec.cov);
      rec.mean = mean;
      rec.cov = cov;
      rec.track.age += 1;
      rec.track.timeSinceUpdate += 1;
      rec.track.box = Tracker.stateBox(rec.mean);
    }

    const maxCost = 1.0 - this.matchThresh;

    // 2. Stage 1: high-confidence detections vs ALL tracks.
    const stage1 = assign(this.stage1Cost(this.live, high), maxCost);
    for (const [tIdx, dIdx] of stage1.matches) {
      this.applyDetection(this.live[tIdx] as Live, high[dIdx] as Detection);
    }

    // 3. Stage 2: still-unmatched CONFIRMED tracks vs low-confidence
    // detections, IoU-only (no Mahalanobis bar), cross-class bar kept.
    const stage2Recs = stage1.unmatchedRows
      .map((i) => this.live[i] as Live)
      .filter((rec) => rec.track.state === STATE_CONFIRMED);
    const stage2 = assign(this.stage2Cost(stage2Recs, low), maxCost);
    for (const [tIdx, dIdx] of stage2.matches) {
      this.applyDetection(stage2Recs[tIdx] as Live, low[dIdx] as Detection);
    }

    // 4. Unmatched high-confidence detections start tentative tracks, in
    // ascending detection-index order (ID determinism).
    for (const dIdx of stage1.unmatchedCols) {
      this.startTrack(high[dIdx] as Detection);
    }

    // 5. Retire: tentative tracks die on their first miss; confirmed tracks
    // die once their gap exceeds maxAge.
    const survivors: Live[] = [];
    for (const rec of this.live) {
      const tr = rec.track;
      if (tr.timeSinceUpdate === 0) {
        survivors.push(rec);
      } else if (tr.state === STATE_CONFIRMED && tr.timeSinceUpdate <= this.maxAge) {
        survivors.push(rec);
      }
    }
    this.live = survivors;

    // live is creation-ordered, so this is ascending by trackId.
    return this.live
      .filter(
        (rec) =>
          rec.track.state === STATE_CONFIRMED && rec.track.timeSinceUpdate === 0,
      )
      .map((rec) => rec.track);
  }

  // -- cost matrices ---------------------------------------------------------

  /** (recs, dets) cost matrix: 1 - IoU, with cross-class pairs and pairs
   * outside the chi-square gate barred at Infinity. */
  private stage1Cost(recs: readonly Live[], dets: readonly Detection[]): Matrix {
    if (recs.length === 0 || dets.length === 0) {
      return Matrix.zeros(recs.length, dets.length);
    }
    const trackBoxes = Matrix.from(recs.map((rec) => [...rec.track.box]));
    const detBoxes = Matrix.from(dets.map((d) => [d.x1, d.y1, d.x2, d.y2]));
    const cost = Tracker.oneMinusIou(trackBoxes, detBoxes);

    const measurements = Matrix.zeros(dets.length, 4);
    for (let j = 0; j < dets.length; j += 1) {
      const xyah = xyxyToXyah(
        Float64Array.of(
          detBoxes.get(j, 0),
          detBoxes.get(j, 1),
          detBoxes.get(j, 2),
          detBoxes.get(j, 3),
        ),
      );
      for (let k = 0; k < 4; k += 1) {
        measurements.set(j, k, xyah[k] as number);
      }
    }

    recs.forEach((rec, i) => {
      const gating = this.kf.gatingDistance(rec.mean, rec.cov, measurements);
      for (let j = 0; j < dets.length; j += 1) {
        if ((gating[j] as number) > KALMAN_GATING_CHI2_95_4DOF) {
          cost.set(i, j, Infinity);
        }
      }
      dets.forEach((det, j) => {
        if (det.className !== rec.track.className) {
          cost.set(i, j, Infinity);
        }
      });
    });
    return cost;
  }

  /** (recs, dets) cost matrix: 1 - IoU with the cross-class bar only -- no
   * Mahalanobis gate (see the module comment). */
  private stage2Cost(recs: readonly Live[], dets: readonly Detection[]): Matrix {
    if (recs.length === 0 || dets.length === 0) {
      return Matrix.zeros(recs.length, dets.length);
    }
    const trackBoxes = Matrix.from(recs.map((rec) => [...rec.track.box]));
    const detBoxes = Matrix.from(dets.map((d) => [d.x1, d.y1, d.x2, d.y2]));
    const cost = Tracker.oneMinusIou(trackBoxes, detBoxes);
    recs.forEach((rec, i) => {
      dets.forEach((det, j) => {
        if (det.className !== rec.track.className) {
          cost.set(i, j, Infinity);
        }
      });
    });
    return cost;
  }

  private static oneMinusIou(trackBoxes: Matrix, detBoxes: Matrix): Matrix {
    const iou = iouMatrix(trackBoxes, detBoxes);
    const cost = Matrix.zeros(iou.rows, iou.cols);
    for (let i = 0; i < iou.rows; i += 1) {
      for (let j = 0; j < iou.cols; j += 1) {
        cost.set(i, j, 1.0 - iou.get(i, j));
      }
    }
    return cost;
  }

  // -- track mutations -------------------------------------------------------

  private static stateBox(mean: Float64Array): [number, number, number, number] {
    const b = xyahToXyxy(mean.subarray(0, 4));
    return [b[0] as number, b[1] as number, b[2] as number, b[3] as number];
  }

  /** Fold a matched detection into a track: Kalman correction, box, score,
   * counters, confirmation, history. */
  private applyDetection(rec: Live, det: Detection): void {
    const measurement = xyxyToXyah(Float64Array.of(det.x1, det.y1, det.x2, det.y2));
    const [mean, cov] = this.kf.update(rec.mean, rec.cov, measurement);
    rec.mean = mean;
    rec.cov = cov;
    const tr = rec.track;
    tr.box = Tracker.stateBox(rec.mean);
    tr.score = det.score;
    tr.hits += 1;
    tr.timeSinceUpdate = 0;
    if (tr.state === STATE_TENTATIVE && tr.hits >= this.minHits) {
      tr.state = STATE_CONFIRMED;
    }
    tr.history.push(tr.anchor);
  }

  /** Spawn a tentative track from an unmatched high-confidence detection; its
   * class is fixed here, forever. */
  private startTrack(det: Detection): void {
    const measurement = xyxyToXyah(Float64Array.of(det.x1, det.y1, det.x2, det.y2));
    const [mean, cov] = this.kf.initiate(measurement);
    const track = new Track(
      this.nextId,
      det.className,
      Tracker.stateBox(mean),
      det.score,
      1,
      1,
      0,
      1 >= this.minHits ? STATE_CONFIRMED : STATE_TENTATIVE,
    );
    this.nextId += 1;
    track.history.push(track.anchor);
    this.live.push({ track, mean, cov });
  }
}
