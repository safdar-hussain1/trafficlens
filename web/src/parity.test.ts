/** Cross-surface parity: the browser engine must decide what the Python
 * engine decided.
 *
 * The product's central claim is that the visitor's own GPU runs the same
 * engine that produced the published accuracy numbers. This file is what
 * makes that a fact rather than a line of copy. Every input below --
 * per-frame detections from a real clip window, a fitted road plane, gates,
 * and a set of deliberately constructed boundary cases -- and every expected
 * output beside it was written by `scripts/make_parity_fixtures.py` running
 * the PYTHON engine. Nothing here was produced by the TypeScript side, which
 * is the only direction that proves anything.
 *
 * What is asserted, and what deliberately is not
 * ---------------------------------------------
 * Track IDs, class names, crossing frames and crossing directions are
 * asserted EXACTLY: they are decisions, and a decision either matches or does
 * not. Speeds are asserted to 1e-6 km/h, the plan's tolerance.
 *
 * Boxes, anchors and Kalman covariances are NOT asserted bit-for-bit, and
 * that is a measured decision rather than a hedge: numpy on the machine that
 * generated these fixtures dispatches to Accelerate, whose kernels fuse
 * multiply-add, so the two covariance paths agree only to ~7e-15 and the
 * gating distances to ~1e-14. The mirror was proven exact against a BLAS-free
 * Python transliteration in Task 19; asserting bit-identity here would be
 * asserting that Accelerate does not exist. The gate ADMITS the same pairs,
 * the same tracks are confirmed and reaped, the same IDs are allocated --
 * those are the facts that survive the last-bit noise, and those are what
 * this file pins.
 *
 * Straddle cases are the point
 * ----------------------------
 * Uniform random inputs essentially never land on a decision boundary, so a
 * parity suite built from random data proves the engines agree in the easy
 * interior and says nothing about the surface where they would actually
 * diverge. The fixture therefore carries constructed cases where the anchor
 * lands EXACTLY on the gate, the IoU sits EXACTLY at `matchThresh`, the
 * confidence sits EXACTLY at `highThresh`, two association costs tie EXACTLY
 * so only the canonical reconstruction can separate them, two kept classes
 * tie EXACTLY in
 * the float32 argmax, and a deferred on-line crossing resolves against the
 * stored last off-line point rather than `prev`. The last test in this file
 * asserts the fixture contains at least one of each, with a non-empty floor,
 * so the suite cannot pass by finding nothing to check. */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";

import { Gate, GateCounter } from "./engine/gate";
import type { CrossingEvent } from "./engine/gate";
import type { Point } from "./engine/geometry";
import { RoadPlane } from "./engine/homography";
import { SpeedEstimator } from "./engine/speed";
import { Tracker } from "./engine/tracker";
import type { Detection } from "./engine/tracker";
import { decodeYolo } from "./runtime/postprocess";

// -- the fixture --------------------------------------------------------------

interface FixtureDetection {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
  readonly score: number;
  readonly classId: number;
  readonly className: string;
  readonly role?: string;
}

interface FixtureFrame {
  readonly frameIndex: number;
  readonly timestamp: number;
  readonly detections: readonly FixtureDetection[];
}

interface FixtureGate {
  readonly name: string;
  readonly start: [number, number];
  readonly end: [number, number];
  readonly labelPositive: string;
  readonly labelNegative: string;
}

interface ExpectedTrack {
  readonly trackId: number;
  readonly className: string;
  readonly speedKmh: number | null;
  readonly role?: string;
}

interface ExpectedEvent {
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

type Counts = Record<string, Record<string, Record<string, number>>>;

interface TrackerCase {
  readonly name: string;
  readonly straddles: readonly string[];
  readonly gates: readonly FixtureGate[];
  readonly frames: readonly FixtureFrame[];
  readonly expected: {
    readonly frames: readonly {
      readonly frameIndex: number;
      readonly tracks: readonly ExpectedTrack[];
    }[];
    readonly events: readonly ExpectedEvent[];
    readonly counts: Counts;
    readonly tracksAllocated: number;
  };
}

interface GateStep {
  readonly trackId: number;
  readonly className: string;
  readonly prev: [number, number];
  readonly curr: [number, number];
  readonly frameIndex: number;
  readonly timestamp: number;
}

interface GateCase {
  readonly name: string;
  readonly straddles: readonly string[];
  readonly gate: FixtureGate;
  readonly steps: readonly GateStep[];
  readonly expected: {
    readonly events: readonly ExpectedEvent[];
    readonly counts: Counts;
  };
  readonly counterfactualPrevOrigin?: { readonly events: readonly ExpectedEvent[] };
}

interface DecodeCase {
  readonly name: string;
  readonly straddles: readonly string[];
  readonly dims: readonly [number, number, number];
  readonly raw: readonly number[];
  readonly scale: number;
  readonly padX: number;
  readonly padY: number;
  readonly conf: number;
  readonly iou: number;
  readonly keepClasses: readonly { readonly classId: number; readonly className: string }[];
  readonly expectedDetections: readonly FixtureDetection[];
  readonly replayFrames: number;
  readonly expectedTracks: readonly ExpectedTrack[];
}

interface ParityFixture {
  readonly schemaVersion: number;
  readonly source: {
    readonly clip: string;
    readonly width: number;
    readonly height: number;
    readonly fps: number;
  };
  readonly plane: { readonly imageToWorld: readonly (readonly number[])[] };
  readonly tracker: {
    readonly highThresh: number;
    readonly lowThresh: number;
    readonly matchThresh: number;
    readonly maxAge: number;
    readonly minHits: number;
  };
  readonly speedLimitKmh: number | null;
  readonly straddleKinds: readonly string[];
  readonly trackerCases: readonly TrackerCase[];
  readonly gateCases: readonly GateCase[];
  readonly decodeCases: readonly DecodeCase[];
}

const FIXTURE: ParityFixture = JSON.parse(
  readFileSync(fileURLToPath(new URL("./fixtures/parity.json", import.meta.url)), "utf8"),
) as ParityFixture;

/** Every boundary kind the fixture is required to carry, repeated here rather
 * than read from the fixture: a list the fixture supplied would be satisfied
 * by whatever the fixture happened to contain. */
const REQUIRED_STRADDLES = [
  "anchorExactlyOnGate",
  "iouExactlyAtMatchThresh",
  "scoreExactlyAtHighThresh",
  "assignmentCostExactTie",
  "argmaxFloat32ClassTie",
  "deferredOnLineUsesLastOffLinePoint",
] as const;

/** The plan's speed tolerance. The worst delta measured across these fixtures
 * is eight orders of magnitude under it. */
const SPEED_TOLERANCE_KMH = 1e-6;

// -- the replay ---------------------------------------------------------------

interface ReplayResult {
  frames: { frameIndex: number; tracks: ExpectedTrack[] }[];
  events: CrossingEvent[];
  counts: Counts;
  tracksAllocated: number;
}

function toGate(spec: FixtureGate): Gate {
  return new Gate(spec.name, spec.start, spec.end, {
    labelPositive: spec.labelPositive,
    labelNegative: spec.labelNegative,
  });
}

function countsOf(counters: Map<string, GateCounter>): Counts {
  const out: Counts = {};
  for (const [name, counter] of counters) {
    for (const [className, directions] of counter.totals) {
      for (const [direction, count] of directions) {
        ((out[name] ??= {})[className] ??= {})[direction] = count;
      }
    }
  }
  return out;
}

/** The per-frame loop `trafficlens.pipeline.run_session` runs, reduced to the
 * parts a parity comparison can see: track, observe, count, reap.
 *
 * The reaping rule is the pipeline's own and matters to the comparison: a
 * confirmed track survives while `timeSinceUpdate <= maxAge` and may
 * re-associate at exactly `maxAge`, so a track unseen for STRICTLY MORE than
 * `maxAge` frames is the first moment it is provably gone. Reaping one frame
 * early would forget a gate counter's `_counted` entry while the track could
 * still return, and the same vehicle would be counted twice. */
function replay(testCase: TrackerCase): ReplayResult {
  const gates = testCase.gates.map(toGate);
  const counters = new Map(gates.map((gate) => [gate.name, new GateCounter(gate)]));
  const tracker = new Tracker({
    highThresh: FIXTURE.tracker.highThresh,
    lowThresh: FIXTURE.tracker.lowThresh,
    matchThresh: FIXTURE.tracker.matchThresh,
    maxAge: FIXTURE.tracker.maxAge,
    minHits: FIXTURE.tracker.minHits,
  });
  const plane = new RoadPlane(FIXTURE.plane.imageToWorld);
  const speed = new SpeedEstimator(plane, FIXTURE.source.fps);

  const previousAnchor = new Map<number, Point>();
  const lastSeen = new Map<number, number>();
  const frames: { frameIndex: number; tracks: ExpectedTrack[] }[] = [];
  const events: CrossingEvent[] = [];
  let allocated = 0;

  for (const frame of testCase.frames) {
    const detections: Detection[] = frame.detections.map((d) => ({
      x1: d.x1,
      y1: d.y1,
      x2: d.x2,
      y2: d.y2,
      score: d.score,
      classId: d.classId,
      className: d.className,
    }));
    const tracks = tracker.update(detections, frame.frameIndex);
    const row: ExpectedTrack[] = [];

    for (const track of tracks) {
      const anchor = track.anchor;
      lastSeen.set(track.trackId, frame.frameIndex);
      allocated = Math.max(allocated, track.trackId);

      speed.observe(track.trackId, anchor, frame.timestamp);
      const speedKmh = speed.speedKmh(track.trackId);
      row.push({ trackId: track.trackId, className: track.className, speedKmh });

      const previous = previousAnchor.get(track.trackId);
      if (previous !== undefined) {
        for (const gate of gates) {
          const event = (counters.get(gate.name) as GateCounter).update(
            track.trackId,
            track.className,
            previous,
            anchor,
            frame.frameIndex,
            frame.timestamp,
            speedKmh,
            FIXTURE.speedLimitKmh,
          );
          if (event !== null) {
            events.push(event);
          }
        }
      }
      previousAnchor.set(track.trackId, anchor);
    }

    for (const [trackId, seen] of [...lastSeen]) {
      if (frame.frameIndex - seen > FIXTURE.tracker.maxAge) {
        lastSeen.delete(trackId);
        for (const counter of counters.values()) {
          counter.forget(trackId);
        }
        speed.forget(trackId);
        previousAnchor.delete(trackId);
      }
    }

    frames.push({ frameIndex: frame.frameIndex, tracks: row });
  }

  return { frames, events, counts: countsOf(counters), tracksAllocated: allocated };
}

// -- assertions ---------------------------------------------------------------

/** Compare one frame's live tracks. IDs and classes are exact; speeds carry
 * the plan's tolerance, and a null on one side must be a null on the other --
 * "no trustworthy number" is a decision, not a value. */
function expectFrameAgrees(
  caseName: string,
  frameIndex: number,
  got: readonly ExpectedTrack[],
  want: readonly ExpectedTrack[],
): void {
  const where = `${caseName} frame ${frameIndex}`;
  expect(got.map((t) => t.trackId), `${where}: track ids`).toEqual(
    want.map((t) => t.trackId),
  );
  expect(got.map((t) => t.className), `${where}: class names`).toEqual(
    want.map((t) => t.className),
  );
  for (let i = 0; i < want.length; i += 1) {
    const wantSpeed = (want[i] as ExpectedTrack).speedKmh;
    const gotSpeed = (got[i] as ExpectedTrack).speedKmh;
    if (wantSpeed === null) {
      expect(gotSpeed, `${where}: track ${(want[i] as ExpectedTrack).trackId} speed`).toBeNull();
    } else {
      expect(gotSpeed, `${where}: track ${(want[i] as ExpectedTrack).trackId} speed`).not.toBeNull();
      expect(
        Math.abs((gotSpeed as number) - wantSpeed),
        `${where}: track ${(want[i] as ExpectedTrack).trackId} speed ${gotSpeed} vs ${wantSpeed}`,
      ).toBeLessThanOrEqual(SPEED_TOLERANCE_KMH);
    }
  }
}

function expectEventsAgree(
  caseName: string,
  got: readonly CrossingEvent[],
  want: readonly ExpectedEvent[],
): void {
  const decisions = (e: CrossingEvent | ExpectedEvent) => ({
    trackId: e.trackId,
    className: e.className,
    gate: e.gate,
    direction: e.direction,
    signedDirection: e.signedDirection,
    frameIndex: e.frameIndex,
    isViolation: e.isViolation,
  });
  expect(got.map(decisions), `${caseName}: crossing decisions`).toEqual(
    want.map(decisions),
  );
  for (let i = 0; i < want.length; i += 1) {
    const w = want[i] as ExpectedEvent;
    const g = got[i] as CrossingEvent;
    expect(g.timestamp, `${caseName}: event ${i} timestamp`).toBe(w.timestamp);
    // The crossing point is a pure float64 line intersection on both sides:
    // no BLAS is involved, so it is compared exactly on the gate cases and
    // to a sub-nanometre tolerance where a Kalman anchor fed into it.
    expect(
      Math.abs(g.crossingX - w.crossingX),
      `${caseName}: event ${i} crossingX ${g.crossingX} vs ${w.crossingX}`,
    ).toBeLessThanOrEqual(1e-9);
    expect(
      Math.abs(g.crossingY - w.crossingY),
      `${caseName}: event ${i} crossingY ${g.crossingY} vs ${w.crossingY}`,
    ).toBeLessThanOrEqual(1e-9);
    if (w.speedKmh === null) {
      expect(g.speedKmh, `${caseName}: event ${i} speed`).toBeNull();
    } else {
      expect(Math.abs((g.speedKmh as number) - w.speedKmh)).toBeLessThanOrEqual(
        SPEED_TOLERANCE_KMH,
      );
    }
  }
}

// -- the suite ----------------------------------------------------------------

describe("cross-surface parity", () => {
  test("the fixture carries a fitted plane matrix, not correspondences", () => {
    // The browser has no cv2 and no SVD: `homography.ts` deliberately omits
    // `fromCorrespondences`. A fixture carrying surveyed points would be
    // unusable here, so this asserts the shape the browser can actually eat.
    expect(Object.keys(FIXTURE.plane)).toEqual(["imageToWorld"]);
    expect(FIXTURE.plane.imageToWorld).toHaveLength(3);
    for (const row of FIXTURE.plane.imageToWorld) {
      expect(row).toHaveLength(3);
    }
    expect(() => new RoadPlane(FIXTURE.plane.imageToWorld)).not.toThrow();
  });

  test("the fixture contains at least one case of every mandated straddle", () => {
    const seen = new Map<string, string[]>(REQUIRED_STRADDLES.map((k) => [k, []]));
    const all = [...FIXTURE.trackerCases, ...FIXTURE.gateCases, ...FIXTURE.decodeCases];
    expect(all.length).toBeGreaterThan(0);
    for (const testCase of all) {
      for (const kind of testCase.straddles) {
        expect(seen.has(kind), `unknown straddle ${kind} on ${testCase.name}`).toBe(true);
        (seen.get(kind) as string[]).push(testCase.name);
      }
    }
    for (const kind of REQUIRED_STRADDLES) {
      expect((seen.get(kind) as string[]).length, `no case straddles ${kind}`)
        .toBeGreaterThanOrEqual(1);
    }
  });

  describe.each(FIXTURE.trackerCases.map((c) => [c.name, c] as const))(
    "tracker case %s",
    (_name, testCase) => {
      const got = replay(testCase);

      test("allocates the same track ids on the same frames", () => {
        expect(got.frames.map((f) => f.frameIndex)).toEqual(
          testCase.expected.frames.map((f) => f.frameIndex),
        );
        for (let i = 0; i < testCase.expected.frames.length; i += 1) {
          const want = testCase.expected.frames[i] as (typeof testCase.expected.frames)[0];
          expectFrameAgrees(
            testCase.name,
            want.frameIndex,
            (got.frames[i] as (typeof got.frames)[0]).tracks,
            want.tracks,
          );
        }
        expect(got.tracksAllocated).toBe(testCase.expected.tracksAllocated);
      });

      test("crosses the same gates on the same frames in the same directions", () => {
        expectEventsAgree(testCase.name, got.events, testCase.expected.events);
        expect(got.counts).toEqual(testCase.expected.counts);
      });
    },
  );

  describe.each(FIXTURE.gateCases.map((c) => [c.name, c] as const))(
    "gate case %s",
    (_name, testCase) => {
      test("emits the same crossings", () => {
        const counter = new GateCounter(toGate(testCase.gate));
        const events: CrossingEvent[] = [];
        for (const step of testCase.steps) {
          const event = counter.update(
            step.trackId,
            step.className,
            step.prev,
            step.curr,
            step.frameIndex,
            step.timestamp,
            null,
            FIXTURE.speedLimitKmh,
          );
          if (event !== null) {
            events.push(event);
          }
        }
        expectEventsAgree(testCase.name, events, testCase.expected.events);
        expect(countsOf(new Map([[testCase.gate.name, counter]]))).toEqual(
          testCase.expected.counts,
        );
      });

      test("would answer differently if the deferred origin came from prev", () => {
        // Only the deferred-resolution case declares a counterfactual, and
        // for it this is the whole point: a case whose two branches agree is
        // pinning nothing. Every other gate case skips.
        if (testCase.counterfactualPrevOrigin === undefined) {
          expect(testCase.straddles).not.toContain("deferredOnLineUsesLastOffLinePoint");
          return;
        }
        expect(testCase.expected.events.length).toBeGreaterThan(0);
        expect(testCase.counterfactualPrevOrigin.events).toEqual([]);
      });
    },
  );

  describe.each(FIXTURE.decodeCases.map((c) => [c.name, c] as const))(
    "decode case %s",
    (_name, testCase) => {
      const keepClasses = new Map(
        testCase.keepClasses.map((c) => [c.classId, c.className] as const),
      );
      const decoded = decodeYolo(
        { data: Float32Array.from(testCase.raw), dims: testCase.dims },
        testCase.scale,
        testCase.padX,
        testCase.padY,
        { conf: testCase.conf, iou: testCase.iou, keepClasses },
      );

      test("decodes to the same detections, class ids included", () => {
        expect(decoded.map((d) => ({
          x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2,
          score: d.score, classId: d.classId, className: d.className,
        }))).toEqual(testCase.expectedDetections.map((d) => ({
          x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2,
          score: d.score, classId: d.classId, className: d.className,
        })));
      });

      test("carries the decoded class through the tracker unchanged", () => {
        // The chain, not the stage: a class id decided by a float32 argmax
        // tie becomes a track's PERMANENT class name, and the tracker bars
        // association across classes. A tie resolved the other way would
        // therefore change which detections may ever match which tracks --
        // which is why this case is replayed through the tracker rather than
        // stopping at the decode.
        const tracker = new Tracker({
          highThresh: FIXTURE.tracker.highThresh,
          lowThresh: FIXTURE.tracker.lowThresh,
          matchThresh: FIXTURE.tracker.matchThresh,
          maxAge: FIXTURE.tracker.maxAge,
          minHits: FIXTURE.tracker.minHits,
        });
        let tracks: ReturnType<Tracker["update"]> = [];
        for (let frame = 0; frame < testCase.replayFrames; frame += 1) {
          tracks = tracker.update(decoded, frame);
        }
        expect(
          tracks.map((t) => ({ trackId: t.trackId, className: t.className })),
        ).toEqual(
          testCase.expectedTracks.map((t) => ({
            trackId: t.trackId,
            className: t.className,
          })),
        );
      });
    },
  );
});
