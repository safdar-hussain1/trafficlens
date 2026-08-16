// Ported from tests/test_speed.py: world-plane speed estimation and the
// refusal-to-guess policy -- an uncalibrated camera reports no speed, ever.
//
// Where the Python test drew random jitter from numpy's default_rng(42), the
// drawn values are transcribed below so both suites fit the same data. Several
// tests additionally pin the exact float64 km/h Python produces, measured from
// a run of the Python estimator: those are the rehearsal for Task 21's
// cross-engine speed agreement.
//
// Those pins are `toBe`, deliberately. They were `toBeCloseTo(..., 9)` first,
// and that was a guard in name only: a summation-order change moves these
// values by ~1e-14 and the tolerance was ~5e-10, so review reversed the
// accumulation loop and all 21 tests stayed green -- while the shipped code
// was at that moment carrying exactly such a defect (a plain running total
// where CPython 3.12's builtin sum() compensates). Exact equality is the whole
// point of a parity pin; anything looser cannot see the class of bug it exists
// to catch.

import { describe, expect, it } from "vitest";

import { SPEED_MAX_STEP_M } from "../generated/constants";
import { RoadPlane } from "./homography";
import { SpeedEstimator, timeOfFlightKmh } from "./speed";
import type { Point } from "./geometry";

// H_TRUE is the exact image -> world (metres) homography of a genuine
// perspective camera looking down a road; it is the same matrix derived in
// tests/test_homography.py (fx = fy = 1000px, principal point (960, 540),
// camera 8m above the road and 5m behind the world origin, tilted 35 degrees
// down, world X = lateral metres, world Y = metres down the road). Building a
// RoadPlane directly from this matrix gives a plane whose toWorld is exact by
// construction, so any error a test measures comes from the speed estimator,
// not from a fitted calibration.
const H_TRUE: readonly (readonly number[])[] = [
  [0.007874647112844963, 0.0, -7.559661228331164],
  [0.0, 0.008548295328846017, -0.9884912148527821],
  [0.0, -0.0008063166600676697, 1.0],
];

// world -> image: the inverse of H_TRUE (transcribed from numpy's inverse of
// the matrix above), so tests can place a target on the road in metres and
// compute the image anchor a detector would have seen.
const H_WORLD_TO_IMAGE: readonly (readonly number[])[] = [
  [126.9898175333877, 99.86300989090614, 1058.7137079659171],
  [0.0, 129.01131005730477, 127.52654660829415],
  [0.0, 0.10402396863636057, 1.1028267791311637],
];

function makePlane(): RoadPlane {
  return new RoadPlane(H_TRUE);
}

function toImage(worldPt: Point): Point {
  const [x, y] = worldPt;
  const r0 = H_WORLD_TO_IMAGE[0] as readonly number[];
  const r1 = H_WORLD_TO_IMAGE[1] as readonly number[];
  const r2 = H_WORLD_TO_IMAGE[2] as readonly number[];
  const u = (r0[0] as number) * x + (r0[1] as number) * y + (r0[2] as number);
  const v = (r1[0] as number) * x + (r1[1] as number) * y + (r1[2] as number);
  const w = (r2[0] as number) * x + (r2[1] as number) * y + (r2[2] as number);
  return [u / w, v / w];
}

/** Feed est the image anchors of a target moving straight down the road
 * (world X = 0) at exactly speedMps, one anchor per frame. */
function drive(
  est: SpeedEstimator,
  trackId: number,
  fps: number,
  nFrames: number,
  speedMps: number,
  yStart = 5.0,
  tStart = 0.0,
): void {
  for (let f = 0; f < nFrames; f += 1) {
    const t = tStart + f / fps;
    est.observe(trackId, toImage([0.0, yStart + speedMps * (t - tStart)]), t);
  }
}

// --- The defining refusal ------------------------------------------------------

describe("the refusal to guess", () => {
  it("never reports a speed when uncalibrated", () => {
    const est = new SpeedEstimator(null, 25.0, 1.0, 3);
    for (let f = 0; f < 100; f += 1) {
      est.observe(1, [10.0 + 30.0 * f, 500.0], f / 25.0);
    }
    expect(est.speedKmh(1)).toBeNull();
  });

  it("refuses even with pathological internal state", () => {
    // Even if internal per-track state somehow exists (here: injected by
    // hand), a null plane means speedKmh returns null -- the refusal is a
    // property of speedKmh itself, not just of observe declining to buffer.
    const est = new SpeedEstimator(null, 25.0, 1.0, 3);
    const injected: [number, number, number][] = [];
    for (let f = 0; f < 50; f += 1) {
      injected.push([f / 25.0, 0.0, f]);
    }
    est._tracks.set(1, injected);
    expect(est.speedKmh(1)).toBeNull();
  });

  it("buffers nothing when uncalibrated", () => {
    const est = new SpeedEstimator(null, 25.0, 1.0, 3);
    for (let f = 0; f < 100; f += 1) {
      est.observe(1, [10.0 + 30.0 * f, 500.0], f / 25.0);
    }
    expect(est._tracks.size).toBe(0);
  });
});

// --- Estimation accuracy ---------------------------------------------------------

describe("estimation accuracy", () => {
  it("recovers 90 km/h within half a km/h", () => {
    const fps = 25.0;
    const est = new SpeedEstimator(makePlane(), fps, 2.0, 5);
    drive(est, 7, fps, 51, 25.0); // 90 km/h
    const speed = est.speedKmh(7);
    expect(speed).not.toBeNull();
    expect(Math.abs((speed as number) - 90.0)).toBeLessThan(0.5);
    // Parity pin: the float64 km/h the Python estimator produces on this exact
    // sequence.
    expect(speed as number).toBe(89.99999999999986);
  });

  it("computes time of flight exactly", () => {
    expect(timeOfFlightKmh(0.0, 2.0, 50.0)).toBe(90.0);
  });

  it("rejects a non-positive time of flight interval", () => {
    expect(() => timeOfFlightKmh(2.0, 2.0, 50.0)).toThrow();
    expect(() => timeOfFlightKmh(3.0, 2.0, 50.0)).toThrow();
  });

  it("rejects a non-positive time of flight distance", () => {
    expect(() => timeOfFlightKmh(0.0, 2.0, 0.0)).toThrow();
    expect(() => timeOfFlightKmh(0.0, 2.0, -50.0)).toThrow();
  });
});

// --- Outlier rejection -----------------------------------------------------------

/** A 90 km/h run where frame 25's anchor is a wild detector box: its
 * plane-space position is 30m off to the side of where the vehicle really is
 * -- far beyond SPEED_MAX_STEP_M, so it must be rejected. */
function driveWithOutlier(est: SpeedEstimator, fps: number): void {
  for (let f = 0; f < 51; f += 1) {
    const t = f / fps;
    const y = 5.0 + 25.0 * t;
    const world: Point = f === 25 ? [30.0, y] : [0.0, y];
    est.observe(3, toImage(world), t);
  }
}

describe("outlier rejection", () => {
  it("rejects a single outlier rather than smoothing it", () => {
    const fps = 25.0;
    const clean = new SpeedEstimator(makePlane(), fps, 2.0, 5);
    drive(clean, 3, fps, 51, 25.0);
    const cleanSpeed = clean.speedKmh(3);

    const dirty = new SpeedEstimator(makePlane(), fps, 2.0, 5);
    driveWithOutlier(dirty, fps);
    const dirtySpeed = dirty.speedKmh(3);

    expect(cleanSpeed).not.toBeNull();
    expect(dirtySpeed).not.toBeNull();
    // The wild sample is rejected outright, so the only difference from the
    // clean run is one missing good sample -- far tighter than the 2 km/h the
    // task demands.
    expect(Math.abs((dirtySpeed as number) - (cleanSpeed as number))).toBeLessThan(0.1);
    expect(Math.abs((dirtySpeed as number) - 90.0)).toBeLessThan(0.5);
  });

  it("matches Python's float64 on the single-outlier run", () => {
    // Deliberately a separate test from the one above, which asserts that the
    // estimate RECOVERS. The two claims come apart: a reject-against-last-RAW
    // policy also recovers here (it loses one extra good frame and still lands
    // within 0.1 km/h), so the recovery test is the honest must-survive control
    // for that mutation -- but it lands on a different float64, which only an
    // exact pin can see. Folding this assertion into the test above would cost
    // that mutation its control.
    const fps = 25.0;
    const dirty = new SpeedEstimator(makePlane(), fps, 2.0, 5);
    driveWithOutlier(dirty, fps);
    expect(dirty.speedKmh(3)).toBe(89.99999999999986);
  });

  it("recovers after an outlier", () => {
    // Rejection is measured against the last ACCEPTED sample, so the good
    // samples after the outlier are accepted and the estimate stays correct.
    const fps = 25.0;
    const est = new SpeedEstimator(makePlane(), fps, 2.0, 5);
    driveWithOutlier(est, fps);
    // Continue the clean stream for another full window past the outlier.
    drive(est, 3, fps, 50, 25.0, 5.0 + 25.0 * (51 / fps), 51 / fps);
    const speed = est.speedKmh(3);
    expect(speed).not.toBeNull();
    expect(Math.abs((speed as number) - 90.0)).toBeLessThan(0.5);
    expect(speed as number).toBe(89.99999999999974);
  });

  it("rejects both of two consecutive outliers", () => {
    // The discriminating case between the two rejection policies: two
    // consecutive wild boxes, placed where they have real leverage on the
    // per-axis least-squares fit -- at the NEWEST edge of the window (max
    // |t - t_mean|, so an accepted one steers the slope hardest) and displaced
    // 30m ALONG the motion axis (the axis whose slope IS the speed; a lateral
    // offset would barely touch it). The shipped policy rejects both: each is
    // > SPEED_MAX_STEP_M from the last ACCEPTED sample. A reject-vs-last-RAW
    // policy rejects the first but accepts the second (1m from the first
    // outlier), and that one edge sample drags the estimate several km/h off
    // -- the corruption this test exists to catch.
    const fps = 25.0;
    const clean = new SpeedEstimator(makePlane(), fps, 2.0, 5);
    drive(clean, 3, fps, 51, 25.0);
    const cleanSpeed = clean.speedKmh(3);

    const dirty = new SpeedEstimator(makePlane(), fps, 2.0, 5);
    for (let f = 0; f < 51; f += 1) {
      const t = f / fps;
      const base = 5.0 + 25.0 * t;
      const world: Point = f === 49 || f === 50 ? [0.0, base + 30.0] : [0.0, base];
      dirty.observe(3, toImage(world), t);
    }
    const dirtySpeed = dirty.speedKmh(3);

    expect(cleanSpeed).not.toBeNull();
    expect(dirtySpeed).not.toBeNull();
    expect(Math.abs((dirtySpeed as number) - (cleanSpeed as number))).toBeLessThan(0.1);
    expect(Math.abs((dirtySpeed as number) - 90.0)).toBeLessThan(0.5);
    expect(dirtySpeed as number).toBe(89.9999999999999);
  });

  it("measures the outlier threshold in metres", () => {
    // A step just under the threshold is accepted; one just over is not.
    const fps = 25.0;
    const est = new SpeedEstimator(makePlane(), fps, 10.0, 2);
    est.observe(9, toImage([0.0, 10.0]), 0.0);
    est.observe(9, toImage([0.0, 10.0 + SPEED_MAX_STEP_M - 0.1]), 1.0);
    expect(est.speedKmh(9)).not.toBeNull(); // both accepted -> 2 samples

    const est2 = new SpeedEstimator(makePlane(), fps, 10.0, 2);
    est2.observe(9, toImage([0.0, 10.0]), 0.0);
    est2.observe(9, toImage([0.0, 10.0 + SPEED_MAX_STEP_M + 0.1]), 1.0);
    expect(est2.speedKmh(9)).toBeNull(); // second sample rejected -> 1 sample
  });

  it("accepts a step exactly at the threshold", () => {
    // Rejection is strictly greater-than: a step of exactly SPEED_MAX_STEP_M
    // metres is accepted. An identity-homography plane makes the world step
    // exact (no projection round-trip rounding).
    const identityPlane = new RoadPlane([
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ]);
    const est = new SpeedEstimator(identityPlane, 25.0, 10.0, 2);
    est.observe(1, [0.0, 0.0], 0.0);
    est.observe(1, [0.0, SPEED_MAX_STEP_M], 1.0);
    expect(est.speedKmh(1)).not.toBeNull(); // both samples accepted
  });

  it("resolves a DIAGONAL step on the threshold the way Python does", () => {
    // Not in the Python suite, and added because the mirror got this wrong
    // first time. The axis-aligned case above cannot separate the two
    // languages -- a hypot with one zero argument is exact everywhere. A
    // diagonal one can: for d = 4.949747468305833, CPython's math.hypot(d, d)
    // is exactly 7.0 (so the step is NOT > SPEED_MAX_STEP_M and the sample is
    // accepted, measured: 25.2 km/h), while JavaScript's Math.hypot(d, d)
    // returns 7.000000000000001 and would reject it. One ULP further out,
    // both agree the sample is an outlier.
    //
    // This is the shape of case Task 21 constructs, so both halves are pinned.
    const identityPlane = new RoadPlane([
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ]);
    const onThreshold = 4.949747468305833;
    const accepted = new SpeedEstimator(identityPlane, 25.0, 10.0, 2);
    accepted.observe(1, [0.0, 0.0], 0.0);
    accepted.observe(1, [onThreshold, onThreshold], 1.0);
    expect(accepted.speedKmh(1)).toBe(25.2);

    const justOver = 4.949747468305834;
    const rejected = new SpeedEstimator(identityPlane, 25.0, 10.0, 2);
    rejected.observe(1, [0.0, 0.0], 0.0);
    rejected.observe(1, [justOver, justOver], 1.0);
    expect(rejected.speedKmh(1)).toBeNull();
  });
});

// --- min_samples boundary ---------------------------------------------------------

describe("min samples boundary", () => {
  it("reports nothing below min samples and a number at exactly min samples", () => {
    const fps = 25.0;
    const minSamples = 4;
    const est = new SpeedEstimator(makePlane(), fps, 10.0, minSamples);
    drive(est, 5, fps, minSamples - 1, 25.0);
    expect(est.speedKmh(5)).toBeNull();
    // One more observation reaches exactly minSamples: a number appears.
    const t = (minSamples - 1) / fps;
    est.observe(5, toImage([0.0, 5.0 + 25.0 * t]), t);
    const speed = est.speedKmh(5);
    expect(speed).not.toBeNull();
    expect(Math.abs((speed as number) - 90.0)).toBeLessThan(0.5);
    expect(speed as number).toBe(89.99999999999996);
  });

  it("returns null for an unknown track", () => {
    const est = new SpeedEstimator(makePlane(), 25.0, 2.0, 5);
    expect(est.speedKmh(42)).toBeNull();
  });
});

// --- Noise floor ------------------------------------------------------------------

// The 61 (x, y) pairs numpy's default_rng(42) draws from normal(0.0, 0.02),
// transcribed so this test fits the same jitter the Python test fits.
const JITTER: readonly (readonly [number, number])[] = [
  [0.006094341595088627, -0.020799682124809912],
  [0.015009023916129145, 0.018811294327824277],
  [-0.03902070377307673, -0.026043590137246362],
  [0.0025568080633457074, -0.006324851846871644],
  [-0.0003360231500857759, -0.017060878551471603],
  [0.017587959497256573, 0.015555838708578967],
  [0.0013206139512243209, 0.022544824139360656],
  [0.009350186845040912, -0.017185849257664764],
  [0.007375015681649977, -0.019177652016579977],
  [0.017569006026145452, -0.000998518219725058],
  [-0.0036972472709052113, -0.013618590888078827],
  [0.024450826773480604, -0.0030905896413760433],
  [-0.008566556443262145, -0.007042671009764592],
  [0.010646183711066974, 0.007308881287281567],
  [0.008254652231919768, 0.008616420060157655],
  [0.04283295201740923, -0.008128300327692312],
  [-0.010244854581430747, -0.016275454564957555],
  [0.012319588451509914, 0.02257944585441783],
  [-0.0022789491530975014, -0.01680312953925056],
  [-0.01648962431382479, 0.013011855756494023],
  [0.014865083424068845, 0.0108630853661039],
  [-0.013310194145773887, 0.0046432264613343955],
  [0.0023337161828145647, 0.004373771934580259],
  [0.017428575558963797, 0.004471910975493646],
  [0.013578271261437899, 0.0013515813897778293],
  [0.005782387973799683, 0.012625764516770808],
  [-0.02914311639711333, -0.006393424327146027],
  [-0.00940745308585591, -0.012777556964866838],
  [-0.005502845024533675, 0.029898826224687917],
  [-0.017316622313864865, 0.019365567091829617],
  [-0.0336573954323161, -0.006697700599715497],
  [0.003255061302100112, 0.011724446627185563],
  [0.0142245315958571, 0.015866944703998506],
  [-0.006974501444968751, -0.009247035853291343],
  [0.01715951762514308, -0.003826086497632298],
  [-0.025513726466758438, -0.022665744280069615],
  [-0.018389045720032228, 0.009943214881075281],
  [0.002848514721411305, 0.013809707081355364],
  [-0.008545052926730686, 0.0031707938215342845],
  [0.012511807879346734, -0.006186930794404768],
  [0.00913550475114823, -0.013238518821333025],
  [-0.007261076931301436, -0.007634757879966582],
  [-0.023916792911780792, 0.009739449615711637],
  [-0.009388046804054478, 0.0002498823745537486],
  [0.009614933178118179, 0.008930623520598882],
  [0.013307702179455726, -0.0019697096901884724],
  [-0.008465966240883076, -0.001594364218127981],
  [-0.0337466886791606, -0.028942249448461747],
  [-0.026453992247088047, -0.019944936552029637],
  [0.007995484534468732, -0.018109581107201216],
  [-0.007563251080787794, 0.02598456595572131],
  [-0.007125279421228519, 0.014750311369341731],
  [-0.01867235360019754, -0.0041087511573526005],
  [-0.019000441098211626, -0.00678066151801125],
  [0.01680616274914791, -0.034546408463846975],
  [0.008688472870917147, 0.0047547120466455576],
  [-0.011882999113935888, -0.028921157087769093],
  [0.0014425901542773902, -0.010589854181276049],
  [0.004653524227094079, 0.0004370429104688576],
  [0.03203557782641831, -0.004787112549460485],
  [-0.0204699498524373, 0.003585512699126323],
];

describe("noise floor", () => {
  it("reads a jittering stopped vehicle as near zero", () => {
    // A STOPPED vehicle whose anchor jitters with per-axis Gaussian world
    // noise (sigma = 2cm) must read near-zero speed. This is what rules out
    // fitting cumulative arc length: arc length rectifies noise -- every
    // jitter step adds positive path length -- turning a stationary target
    // into a deterministic phantom speed of several km/h. Per-axis slopes see
    // zero-mean noise and fit ~0 on both axes.
    const fps = 30.0;
    const est = new SpeedEstimator(makePlane(), fps, 2.0, 5);
    const nFrames = 2.0 * fps + 1; // a full window of samples
    for (let f = 0; f < nFrames; f += 1) {
      const [noiseX, noiseY] = JITTER[f] as readonly [number, number];
      est.observe(4, toImage([0.0 + noiseX, 10.0 + noiseY]), f / fps);
    }
    const speed = est.speedKmh(4);
    expect(speed).not.toBeNull();
    expect(speed as number).toBeLessThan(1.0);
    expect(speed as number).toBe(0.02062706934059718);
  });
});

// --- Window expiry -----------------------------------------------------------------

describe("window expiry", () => {
  it("reads the current speed after a stop, not an average", () => {
    // A vehicle stopped for 5s then moving at 90 km/h for 3s reads ~90, not
    // some average dragged down by ancient stationary history: samples older
    // than windowS have fallen out of the window.
    const fps = 25.0;
    const est = new SpeedEstimator(makePlane(), fps, 2.0, 5);
    const stopFrames = 5.0 * fps;
    for (let f = 0; f < stopFrames; f += 1) {
      est.observe(8, toImage([0.0, 5.0]), f / fps);
    }
    const stopped = est.speedKmh(8);
    expect(stopped).not.toBeNull();
    expect(Math.abs(stopped as number)).toBeLessThan(0.5); // reads as stationary
    expect(stopped as number).toBe(0.0);

    drive(est, 8, fps, 3.0 * fps, 25.0, 5.0, stopFrames / fps);
    const speed = est.speedKmh(8);
    expect(speed).not.toBeNull();
    expect(Math.abs((speed as number) - 90.0)).toBeLessThan(0.5);
    expect(speed as number).toBe(89.99999999999989);
  });
});

// --- Constructor validation ----------------------------------------------------------

describe("constructor validation", () => {
  it("rejects bad parameters", () => {
    expect(() => new SpeedEstimator(makePlane(), 0.0)).toThrow();
    expect(() => new SpeedEstimator(makePlane(), -25.0)).toThrow();
    expect(() => new SpeedEstimator(makePlane(), 25.0, 0.0)).toThrow();
    expect(() => new SpeedEstimator(makePlane(), 25.0, -1.0)).toThrow();
    expect(() => new SpeedEstimator(makePlane(), 25.0, 2.0, 1)).toThrow();
  });
});

// --- Lifecycle and determinism -------------------------------------------------------

describe("lifecycle and determinism", () => {
  it("clears state on forget", () => {
    const fps = 25.0;
    const est = new SpeedEstimator(makePlane(), fps, 10.0, 3);
    drive(est, 2, fps, 10, 25.0);
    expect(est.speedKmh(2)).not.toBeNull();
    est.forget(2);
    expect(est.speedKmh(2)).toBeNull();
    // A recycled track ID starts from scratch: two fresh samples are still
    // under minSamples even though ten were observed before the forget.
    drive(est, 2, fps, 2, 25.0);
    expect(est.speedKmh(2)).toBeNull();
  });

  it("is a no-op to forget an unknown track", () => {
    const est = new SpeedEstimator(makePlane(), 25.0, 2.0, 5);
    expect(() => est.forget(999)).not.toThrow();
    expect(est.speedKmh(999)).toBeNull();
  });

  it("produces the same float for the same sequence", () => {
    const fps = 25.0;
    const run = (): number => {
      const est = new SpeedEstimator(makePlane(), fps, 2.0, 5);
      for (let f = 0; f < 51; f += 1) {
        const t = f / fps;
        // Deterministic zig-zag jitter on top of straight motion.
        const wobble = 0.3 * Math.sin((2.0 * Math.PI * f) / 7.0);
        est.observe(6, toImage([wobble, 5.0 + 25.0 * t]), t);
      }
      const speed = est.speedKmh(6);
      expect(speed).not.toBeNull();
      return speed as number;
    };
    expect(run()).toBe(run());
    expect(run()).toBe(90.00003255540452);
  });
});
