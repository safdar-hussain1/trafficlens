/** Plane-space speed estimation: world-metre displacement over time, in km/h.
 * Mirrors `trafficlens.analytics.speed`.
 *
 * Policy -- inherited from `homography.ts` and enforced here: **an uncalibrated
 * camera reports no speed, ever -- never a pixel-derived guess.** A
 * `SpeedEstimator` built with `plane = null` (the `NO_CALIBRATION` sentinel)
 * returns `null` from `speedKmh` for every track, unconditionally: the check
 * lives in `speedKmh` itself, so no amount of observed data -- or even
 * hand-injected internal state -- can make an uncalibrated estimator emit a
 * number. `observe` also short-circuits in that case, so nothing is buffered
 * for a speed that will never be reported.
 *
 * How the estimate is made
 * ------------------------
 * Each observed image anchor is projected to world metres through the
 * `RoadPlane` at observe time. Per track, a buffer holds the accepted samples
 * `[timestamp, wx, wy]` covering the trailing `windowS` seconds. `speedKmh`
 * fits `wx(t)` and `wy(t)` SEPARATELY by least squares over the window; the
 * two fitted slopes form the mean velocity vector in metres per second, and
 * the speed is its magnitude, `hypot(slopeX, slopeY)`, converted to km/h by
 * multiplying by 3.6.
 *
 * Per-axis slopes, not cumulative arc length, deliberately: arc length
 * rectifies noise. Every jitter step adds a POSITIVE path increment (for
 * zero-mean Gaussian jitter the expected step is sigma * sqrt(pi)), so a
 * stopped vehicle's anchor noise integrates into a deterministic phantom speed
 * of several km/h -- a monotone bias no fit over the summed path can remove.
 * Per-axis coordinate noise is zero-mean, so each slope goes to zero for a
 * stopped vehicle, and for straight motion the vector magnitude is identical
 * to the arc-length fit.
 *
 * Known limitation, stated plainly: within-window curvature reads slightly
 * low. The velocity vector is a chord across the window, so a path that bends
 * inside the window (a lane change, a curve) has its speed under-read by the
 * chord-versus-arc difference. Over a 2s window at road speeds and road
 * curvatures the effect is negligible; it is the accepted price of a zero
 * noise floor at 0 km/h.
 *
 * Each least-squares slope is the closed-form two-coefficient solve, written
 * out explicitly, with timestamps centred on their mean:
 *
 *     slopeX = sum(dt_i * (wx_i - wxMean)) / sum(dt_i ** 2)
 *     slopeY = sum(dt_i * (wy_i - wyMean)) / sum(dt_i ** 2)
 *
 * where `dt_i = t_i - tMean`. Centring first keeps the sums small and the
 * arithmetic well-conditioned. The accumulation order matches the Python
 * original's, so the two fits agree to the last bits they can.
 *
 * Outlier rejection
 * -----------------
 * A sample whose plane-space step from the previous ACCEPTED sample exceeds
 * `SPEED_MAX_STEP_M` is rejected at observe time -- never buffered -- so a
 * single wild detector box cannot spike the speed. The comparison is against
 * the last ACCEPTED sample, not the last raw (seen) one. Against a single
 * outlier the raw-sample variant would also recover -- it loses the outlier
 * plus the one good frame measured against it, two frames in total -- but it
 * fails on CONSECUTIVE outliers at the same wrong location: the second wild
 * box lands within threshold of the first and gets accepted, corrupting the
 * estimate. Measuring against the last accepted sample rejects every sample in
 * such a burst.
 *
 * `timeOfFlightKmh` is the independent gate-pair estimator used by the Tier-2
 * cross-check: a known ground distance between two gates divided by the
 * crossing-time difference. It shares no state or code path with
 * `SpeedEstimator`, which is exactly what makes it a cross-check. */

import {
  SPEED_MAX_STEP_M,
  SPEED_MIN_SAMPLES,
  SPEED_WINDOW_S,
} from "../generated/constants";
import type { Point } from "./geometry";
import type { RoadPlane } from "./homography";
import { hypot } from "./numeric";

// m/s -> km/h: 3600 seconds per hour / 1000 metres per kilometre.
const MPS_TO_KMH = 3.6;

/** One accepted plane-space sample: [timestamp, worldX, worldY]. */
export type Sample = [number, number, number];

/** Estimates per-track speeds in km/h from image anchors projected onto a
 * calibrated road plane.
 *
 * With `plane = null` (`NO_CALIBRATION`) every `speedKmh` call returns `null`,
 * always -- this class never falls back to a pixel-derived guess. */
export class SpeedEstimator {
  /** trackId -> buffer of accepted samples. Named to mirror the Python
   * attribute of the same name, and left reachable for the same reason: the
   * refusal test injects pathological state here to prove the refusal lives in
   * `speedKmh` and not merely in `observe` declining to buffer. */
  readonly _tracks = new Map<number, Sample[]>();

  private readonly plane: RoadPlane | null;
  private readonly windowS: number;
  private readonly minSamples: number;
  private readonly maxlen: number;

  constructor(
    plane: RoadPlane | null,
    fps: number,
    windowS: number = SPEED_WINDOW_S,
    minSamples: number = SPEED_MIN_SAMPLES,
  ) {
    if (fps <= 0.0) {
      throw new Error(`fps must be positive, got ${fps}`);
    }
    if (windowS <= 0.0) {
      throw new Error(`windowS must be positive, got ${windowS}`);
    }
    if (minSamples < 2) {
      throw new Error(
        `minSamples must be at least 2 (a slope needs two points), got ${minSamples}`,
      );
    }
    this.plane = plane;
    this.windowS = windowS;
    this.minSamples = minSamples;
    // A hard memory bound derived from fps: one window at the declared frame
    // rate can hold at most windowS * fps + 1 samples (inclusive endpoints), so
    // even a caller whose timestamps never advance -- which would defeat the
    // time-based pruning below -- cannot grow a buffer past one window's worth.
    this.maxlen = Math.ceil(windowS * fps) + 1;
  }

  /** Record one image anchor for a track at a timestamp (seconds).
   *
   * Uncalibrated (`plane === null`): returns immediately without buffering
   * anything -- there is no speed this data could ever contribute to.
   * Otherwise the anchor is projected to world metres; a sample stepping
   * further than `SPEED_MAX_STEP_M` from the last accepted sample is rejected
   * outright, and accepted samples older than `windowS` before this one are
   * dropped. */
  observe(trackId: number, anchor: Point, timestamp: number): void {
    if (this.plane === null) {
      return;
    }

    const [wx, wy] = this.plane.toWorld(anchor);
    let buf = this._tracks.get(trackId);
    if (buf === undefined) {
      buf = [];
      this._tracks.set(trackId, buf);
    }

    if (buf.length > 0) {
      const last = buf[buf.length - 1] as Sample;
      if (hypot(wx - last[1], wy - last[2]) > SPEED_MAX_STEP_M) {
        return; // a bad detection, not vehicle motion: never buffered
      }
    }

    buf.push([timestamp, wx, wy]);
    if (buf.length > this.maxlen) {
      buf.shift();
    }
    while (buf.length > 0 && (buf[0] as Sample)[0] < timestamp - this.windowS) {
      buf.shift();
    }
  }

  /** The track's current speed in km/h, or `null` when no trustworthy number
   * exists: always `null` when the estimator is uncalibrated, and otherwise
   * when the track has fewer than `minSamples` accepted samples inside the
   * trailing window (or its in-window timestamps do not span any time at all).
   *
   * The window trails the track's NEWEST sample, not the caller's clock, so a
   * track that stops being observed keeps reporting its last in-window speed
   * until `forget()` is called -- callers must forget dead tracks. */
  speedKmh(trackId: number): number | null {
    if (this.plane === null) {
      // The refusal is absolute: even pathological internal state cannot make
      // an uncalibrated estimator emit a number.
      return null;
    }

    const buf = this._tracks.get(trackId);
    if (buf === undefined || buf.length < this.minSamples) {
      return null;
    }

    // observe() prunes on append, so the buffer already holds only the
    // trailing window; re-filter against the newest timestamp anyway so the
    // window contract holds no matter how the buffer was reached.
    const newest = (buf[buf.length - 1] as Sample)[0];
    const samples = buf.filter((s) => s[0] >= newest - this.windowS);
    const n = samples.length;
    if (n < this.minSamples) {
      return null;
    }

    // Closed-form least-squares slopes of wx(t) and wy(t), fitted separately
    // with timestamps centred on their mean (see the module comment: per-axis
    // noise is zero-mean, so a stopped vehicle's jitter fits ~0 on both axes
    // instead of rectifying into a phantom positive speed the way cumulative
    // arc length would).
    let tSum = 0.0;
    let xSum = 0.0;
    let ySum = 0.0;
    for (const s of samples) {
      tSum += s[0];
    }
    for (const s of samples) {
      xSum += s[1];
    }
    for (const s of samples) {
      ySum += s[2];
    }
    const tMean = tSum / n;
    const xMean = xSum / n;
    const yMean = ySum / n;

    let numX = 0.0;
    let numY = 0.0;
    let den = 0.0;
    for (const [t, wx, wy] of samples) {
      const dt = t - tMean;
      numX += dt * (wx - xMean);
      numY += dt * (wy - yMean);
      den += dt * dt;
    }
    if (den === 0.0) {
      return null; // all in-window samples share one timestamp
    }

    return hypot(numX / den, numY / den) * MPS_TO_KMH;
  }

  /** Drop all state for a track. A later track with the same (recycled) ID
   * starts from scratch. Unknown IDs are a no-op. */
  forget(trackId: number): void {
    this._tracks.delete(trackId);
  }
}

/** Speed in km/h of a vehicle covering a known ground distance (`distanceM`,
 * metres, surveyed between two gates) between timestamps `tA` and `tB`
 * (seconds). Independent of any RoadPlane -- this is the Tier-2 cross-check on
 * the homography-based estimate.
 *
 * Throws (fails fast) when `tB - tA` or `distanceM` is not positive: a zero or
 * negative interval or distance is a caller bug, not a slow vehicle. */
export function timeOfFlightKmh(tA: number, tB: number, distanceM: number): number {
  const dt = tB - tA;
  if (dt <= 0.0) {
    throw new Error(`tB must be after tA, got dt = ${dt}`);
  }
  if (distanceM <= 0.0) {
    throw new Error(`distanceM must be positive, got ${distanceM}`);
  }
  return (distanceM / dt) * MPS_TO_KMH;
}
