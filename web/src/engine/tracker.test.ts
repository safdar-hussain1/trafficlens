// Ported from the Tracker section of tests/test_tracker.py.
//
// Identity errors here become count errors and speed mis-attributions
// downstream, and this mirror must reproduce every Python decision exactly, so
// these tests pin down behaviour precisely: ID stability on straight-line
// motion, identity preservation through a crossing that pure IoU would swap,
// dropout survival on both sides of max_age, the low-confidence recovery
// stage, cross-class barring, tentative-track death, reset semantics, and
// exact run-to-run determinism.

import { describe, expect, it, vi } from "vitest";

import {
  TRACK_HIGH_CONF,
  TRACK_LOW_CONF,
  TRACK_MATCH_IOU,
  TRACK_MAX_AGE,
  TRACK_MIN_HITS,
} from "../generated/constants";
import { Track, Tracker } from "./tracker";
import type { Detection } from "./tracker";

const CLASS_IDS: Record<string, number> = { car: 2, truck: 7 };

/** A Detection whose box has top-left (x, y) and size w x h. */
function det(
  x: number,
  y: number,
  score: number,
  cls = "car",
  w = 80.0,
  h = 80.0,
): Detection {
  return {
    x1: x,
    y1: y,
    x2: x + w,
    y2: y + h,
    score,
    classId: CLASS_IDS[cls] as number,
    className: cls,
  };
}

// -- construction contract ---------------------------------------------------

describe("Tracker construction", () => {
  it("takes its defaults from the generated constants", () => {
    const t = new Tracker();
    expect(t.highThresh).toBe(TRACK_HIGH_CONF);
    expect(t.lowThresh).toBe(TRACK_LOW_CONF);
    expect(t.matchThresh).toBe(TRACK_MATCH_IOU);
    expect(t.maxAge).toBe(TRACK_MAX_AGE);
    expect(t.minHits).toBe(TRACK_MIN_HITS);
  });
});

// -- core identity behaviour -------------------------------------------------

describe("core identity behaviour", () => {
  it("keeps one id for 30 frames on a single object", () => {
    const t = new Tracker(); // defaults: minHits=3, so output starts at frame 2
    const outputs: Track[][] = [];
    for (let f = 0; f < 30; f += 1) {
      outputs.push(t.update([det(10 + 5 * f, 10, 0.9)], f));
    }

    // Tentative for the first minHits - 1 frames, then exactly one confirmed
    // track per frame with a single stable ID.
    expect(outputs[0]).toEqual([]);
    expect(outputs[1]).toEqual([]);
    const ids = outputs.slice(2).flatMap((frame) => frame.map((tr) => tr.trackId));
    expect(ids.length).toBe(28);
    expect(new Set(ids)).toEqual(new Set([1]));

    const last = (outputs[outputs.length - 1] as Track[])[0] as Track;
    expect(last).toBeInstanceOf(Track);
    expect(last.state).toBe("confirmed");
    expect(last.age).toBe(30);
    expect(last.hits).toBe(30);
    expect(last.timeSinceUpdate).toBe(0);
    // History records the anchor once per real detection update, including the
    // creation frame -- never for coasting frames.
    expect(last.history.length).toBe(30);
    // The anchor is the bottom-centre of the (filter-smoothed) box.
    const bx = last.box;
    expect(last.anchor).toEqual([(bx[0] + bx[2]) / 2.0, bx[3]]);
  });

  it("allocates new ids in detection order", () => {
    const t = new Tracker({ minHits: 1 });
    const dets = [det(500, 10, 0.9), det(10, 10, 0.9), det(250, 10, 0.9)];
    const tracks = t.update(dets, 0);
    expect(tracks.map((tr) => tr.trackId)).toEqual([1, 2, 3]);
    // IDs follow the detection-list order, not any spatial order.
    expect((tracks[0] as Track).box[0]).toBeCloseTo(500.0, 6);
    expect((tracks[1] as Track).box[0]).toBeCloseTo(10.0, 6);
    expect((tracks[2] as Track).box[0]).toBeCloseTo(250.0, 6);
  });

  it("keeps identities through crossing paths", () => {
    // Two identical 100x30 boxes on the same row (cy = 300) drive toward each
    // other at +/-10 px/frame: A cx = 100 + 10t, B cx = 290 - 10t. They pass
    // between t=9 and t=10 with a closest approach of 10 px.
    //
    // Why pure IoU WOULD swap them: at t=10 the detections sit at cx 200 (A)
    // and 190 (B), while the last OBSERVED boxes (t=9) sit at cx 190 (A) and
    // 200 (B). IoU of two width-100 boxes offset s px is (100 - s)/(100 + s),
    // so the IoU matrix against the last boxes is
    //
    //                 det A @200   det B @190
    //   last A @190     0.8182       1.0
    //   last B @200     1.0          0.8182
    //
    // Both greedy (takes the two 1.0 entries first) and Hungarian (swapped
    // total 2.0 > correct total 1.6364) pick the SWAPPED pairing, and every
    // entry clears the 0.8 IoU floor, so nothing in the IoU cost alone
    // prevents the swap. The Kalman filter's velocity knowledge does, twice
    // over (values measured with this exact geometry):
    //   1. predicted boxes at t=10 sit at cx 198.9 (A) / 191.1 (B), so the
    //      correct pairs score IoU ~0.98 and Hungarian prefers them;
    //   2. the squared Mahalanobis distance of the swapped pairs is ~11.5 at
    //      t=10 (~18.4 at t=9), above the 9.4877 chi-square gate, so the
    //      swapped pairing is barred outright -- while the correct pairs
    //      score ~0.2 and pass easily.
    const w = 100.0;
    const h = 30.0;
    const cy = 300.0;

    const frameDets = (t: number): Detection[] => {
      const cxA = 100.0 + 10.0 * t;
      const cxB = 290.0 - 10.0 * t;
      return [
        det(cxA - w / 2, cy - h / 2, 0.9, "car", w, h),
        det(cxB - w / 2, cy - h / 2, 0.9, "car", w, h),
      ];
    };

    const tracker = new Tracker({ minHits: 1 });
    const seenIds = new Set<number>();
    for (let t = 0; t < 20; t += 1) {
      const tracks = tracker.update(frameDets(t), t);
      const byId = new Map(tracks.map((tr) => [tr.trackId, tr]));
      for (const id of byId.keys()) {
        seenIds.add(id);
      }
      expect([...byId.keys()].sort(), `frame ${t}: expected exactly tracks 1 and 2`).toEqual([1, 2]);
      // The filter-smoothed cx stays within ~2.1 px of ground truth
      // (measured), while a swapped identity would be ~10 px off at the
      // closest approach and ~190 px off by the final frame.
      const t1 = byId.get(1) as Track;
      const t2 = byId.get(2) as Track;
      const cx1 = (t1.box[0] + t1.box[2]) / 2.0;
      const cx2 = (t2.box[0] + t2.box[2]) / 2.0;
      expect(Math.abs(cx1 - (100.0 + 10.0 * t)), `frame ${t}`).toBeLessThan(4.0);
      expect(Math.abs(cx2 - (290.0 - 10.0 * t)), `frame ${t}`).toBeLessThan(4.0);
    }

    // No fragmentation either: the whole run used exactly two IDs.
    expect([...seenIds].sort()).toEqual([1, 2]);
  });

  it("bars a floor-eligible displaced detection with the Mahalanobis gate", async () => {
    // The chi-square gate needs a discriminator the IoU floor cannot provide,
    // and a symmetric crossing cannot either (a swap preferred by
    // predicted-box IoU implies the swapped pairs are CLOSER to the
    // predictions, so IoU and Mahalanobis agree there). The case that
    // separates them: a mature track whose velocity is established, given a
    // single detection displaced from the prediction by an amount that still
    // clears the IoU floor but exceeds the gate.
    //
    // Geometry (identical to the crossing test's boxes): a 100x30 box at
    // cx = 100 + 10t for t = 0..9, then at t = 10 the only detection sits at
    // cx = 190 instead of the predicted ~198.89 -- displaced ~8.9 px.
    // Measured with the real filter on this exact sequence:
    //   IoU(predicted box, displaced box) ~ (100-8.89)/(100+8.89) = 0.8368
    //     -> clears the 0.8 floor (cost 0.1632 <= maxCost 0.2), so the IoU
    //        cost alone would happily match it;
    //   squared Mahalanobis distance = 11.49 > 9.4877
    //     -> the gate bars the pair.
    // (Warm-up frames all pass the gate; the worst is ~8.39 at t=2, while the
    // velocity prior is still settling.)
    const w = 100.0;
    const h = 30.0;
    const cy = 300.0;
    const detAt = (cx: number): Detection =>
      det(cx - w / 2, cy - h / 2, 0.9, "car", w, h);

    const run = (Ctor: typeof Tracker): Track[] => {
      const tracker = new Ctor(); // defaults; minHits=3 keeps the new track internal
      for (let t = 0; t < 10; t += 1) {
        tracker.update([detAt(100.0 + 10.0 * t)], t);
      }
      return tracker.update([detAt(190.0)], 10);
    };

    // Gate active: the displaced detection must NOT continue track 1. It
    // starts a tentative track instead (internal), and track 1 coasts, so the
    // frame's detector-backed output is empty.
    expect(run(Tracker)).toEqual([]);

    // Same sequence with the gate widened to infinity: now nothing bars the
    // pair, the IoU floor alone accepts it, and track 1 IS continued. This
    // half proves the assertion above really is the gate's doing and not the
    // floor's.
    try {
      vi.resetModules();
      vi.doMock("../generated/constants", async () => {
        const actual =
          await vi.importActual<typeof import("../generated/constants")>(
            "../generated/constants",
          );
        return { ...actual, KALMAN_GATING_CHI2_95_4DOF: Number.POSITIVE_INFINITY };
      });
      const wideGate = await import("./tracker");
      expect(run(wideGate.Tracker).map((tr) => tr.trackId)).toEqual([1]);
    } finally {
      vi.doUnmock("../generated/constants");
      vi.resetModules();
    }
  });
});

// -- dropout on both sides of max_age ----------------------------------------

/** Track one object for 5 frames, hide it for `gap` frames, then show it again
 * on its constant-velocity path. Returns [idBefore, the single track present
 * on the reappearance frame]. */
function runDropout(gap: number): [number, Track] {
  const tracker = new Tracker({ minHits: 1, maxAge: 5 });
  const detAt = (f: number): Detection => det(50 + 4 * f, 100, 0.9, "car", 60.0, 60.0);

  let idBefore = -1;
  for (let f = 0; f < 5; f += 1) {
    idBefore = (tracker.update([detAt(f)], f)[0] as Track).trackId;
  }

  for (let f = 5; f < 5 + gap; f += 1) {
    expect(tracker.update([], f)).toEqual([]); // nothing detector-backed
  }

  const f = 5 + gap;
  const tracks = tracker.update([detAt(f)], f);
  expect(tracks.length).toBe(1);
  return [idBefore, tracks[0] as Track];
}

describe("dropout", () => {
  it("keeps the id across max_age - 1 frames", () => {
    const [idBefore, track] = runDropout(4);
    expect(track.trackId).toBe(idBefore);
    // History grows only on real-detection updates: 5 before the gap plus the
    // reappearance frame. An append-during-coast bug would make it 6 + gap.
    expect(track.history.length).toBe(6);
  });

  it("keeps the id across exactly max_age frames", () => {
    // The death rule is timeSinceUpdate > maxAge, strictly: a gap of exactly
    // maxAge frames must still survive on prediction alone. This pins the
    // boundary a wrong `>=` rule would get past the +/-1 tests.
    const [idBefore, track] = runDropout(5);
    expect(track.trackId).toBe(idBefore);
    expect(track.history.length).toBe(6);
  });

  it("issues a new id across max_age + 1 frames", () => {
    const [idBefore, track] = runDropout(6);
    expect(track.trackId).not.toBe(idBefore);
    // A brand-new track: its history starts over with one anchor.
    expect(track.history.length).toBe(1);
  });
});

// -- second association stage ------------------------------------------------

describe("low-confidence stage", () => {
  it("recovers an occluded track", () => {
    // The occlusion dip (frames 5..9, score 0.3) lands in the low-confidence
    // band, so only the second association stage can keep the track
    // detector-backed through it. Delete stage two and this test goes red:
    // update() returns only tracks updated by a real detection this frame, so
    // the dip frames would return [] and the [0] below would be undefined.
    const t = new Tracker({
      highThresh: 0.6,
      lowThresh: 0.2,
      matchThresh: 0.8,
      maxAge: 30,
      minHits: 1,
    });
    const ids: number[] = [];
    for (let f = 0; f < 20; f += 1) {
      const score = f < 5 || f > 9 ? 0.9 : 0.3; // dips into low-confidence band
      const tracks = t.update([det(10 + 5 * f, 10, score)], f);
      expect(tracks.length, `frame ${f}`).toBe(1);
      ids.push((tracks[0] as Track).trackId);
    }
    expect(new Set(ids).size).toBe(1);
  });

  it("never starts a track from a low-confidence detection", () => {
    const t = new Tracker({ minHits: 1 });
    for (let f = 0; f < 10; f += 1) {
      expect(t.update([det(10, 10, 0.3)], f)).toEqual([]);
    }
    // Nothing was created internally either: the next high-confidence
    // detection takes ID 1, the first ID this tracker ever issues.
    expect(t.update([det(10, 10, 0.9)], 10).map((tr) => tr.trackId)).toEqual([1]);
  });
});

// -- class handling ----------------------------------------------------------

describe("class handling", () => {
  it("never matches a cross-class pair", () => {
    const t = new Tracker({ minHits: 1 });
    const carBox: [number, number, number] = [100.0, 100.0, 0.9];
    let tracks: Track[] = [];
    for (let f = 0; f < 3; f += 1) {
      tracks = t.update([det(carBox[0], carBox[1], carBox[2], "car")], f);
    }
    expect(tracks.map((tr) => tr.trackId)).toEqual([1]);

    // A perfectly-overlapping truck detection: IoU 1.0 with the car track's
    // predicted box, but the cross-class bar must refuse the match and start a
    // fresh track instead.
    tracks = t.update([det(carBox[0], carBox[1], carBox[2], "truck")], 3);
    expect(tracks.map((tr) => [tr.trackId, tr.className])).toEqual([[2, "truck"]]);

    // The car comes back: same ID, class untouched by the flicker. The truck
    // track coasts (no truck detection), so it is not in the output.
    tracks = t.update([det(carBox[0], carBox[1], carBox[2], "car")], 4);
    expect(tracks.map((tr) => [tr.trackId, tr.className])).toEqual([[1, "car"]]);
  });
});

// -- tentative-track lifecycle -----------------------------------------------

describe("tentative tracks", () => {
  it("dies on a single miss", () => {
    const t = new Tracker({ minHits: 3 });
    expect(t.update([det(10, 10, 0.9)], 0)).toEqual([]); // tentative, internal
    expect(t.update([], 1)).toEqual([]); // one miss: the tentative track dies

    // The object shows up again: a NEW track (ID 2) must be built from scratch
    // and needs minHits consecutive frames to confirm.
    expect(t.update([det(10, 10, 0.9)], 2)).toEqual([]);
    expect(t.update([det(10, 10, 0.9)], 3)).toEqual([]);
    const tracks = t.update([det(10, 10, 0.9)], 4);
    expect(tracks.map((tr) => tr.trackId)).toEqual([2]);
    expect((tracks[0] as Track).state).toBe("confirmed");
    expect((tracks[0] as Track).hits).toBe(3);
  });
});

// -- reset -------------------------------------------------------------------

describe("reset", () => {
  it("restarts track ids at 1", () => {
    const t = new Tracker({ minHits: 1 });
    let tracks: Track[] = [];
    for (let f = 0; f < 3; f += 1) {
      tracks = t.update([det(10, 10, 0.9), det(300, 10, 0.9)], f);
    }
    expect(tracks.map((tr) => tr.trackId)).toEqual([1, 2]);

    t.reset();
    expect(t.update([], 0)).toEqual([]); // no survivors from before the reset
    expect(t.update([det(500, 10, 0.9)], 1).map((tr) => tr.trackId)).toEqual([1]);
  });
});

// -- determinism -------------------------------------------------------------

/** A deterministic 30-frame scenario mixing everything at once: a crossing
 * pair, an object whose score dips into the low band, a dropout that comes
 * back inside max_age, and a truck overlapping a car. */
function mixedScenario(): Detection[][] {
  const frames: Detection[][] = [];
  for (let f = 0; f < 30; f += 1) {
    const dets: Detection[] = [];
    // crossing pair (cars)
    dets.push(det(50 + 10 * f, 300, 0.9, "car", 100.0, 30.0));
    dets.push(det(340 - 10 * f, 300, 0.85, "car", 100.0, 30.0));
    // score dipper
    const dip = f >= 8 && f <= 12 ? 0.3 : 0.9;
    dets.push(det(10 + 4 * f, 600, dip, "car", 60.0, 60.0));
    // dropout: visible 0..9 and 15.., hidden in between
    if (f < 10 || f >= 15) {
      dets.push(det(700, 10 + 6 * f, 0.9, "car", 70.0, 70.0));
    }
    // a truck sharing the road with the first car
    dets.push(det(50 + 10 * f, 350, 0.7, "truck", 90.0, 40.0));
    frames.push(dets);
  }
  return frames;
}

function serialize(tracks: Track[]): unknown[] {
  return tracks.map((tr) => [
    tr.trackId,
    tr.className,
    tr.box,
    tr.score,
    tr.age,
    tr.hits,
    tr.timeSinceUpdate,
    tr.state,
    tr.history.map((p) => [p[0], p[1]]),
  ]);
}

describe("determinism", () => {
  it("produces identical id sequences on two identical runs", () => {
    const runs: unknown[][] = [];
    for (let r = 0; r < 2; r += 1) {
      const tracker = new Tracker({ minHits: 2 });
      const out: unknown[] = [];
      mixedScenario().forEach((dets, f) => {
        out.push(serialize(tracker.update(dets, f)));
      });
      runs.push(out);
    }
    // Exact equality: same IDs in the same order with bit-identical boxes,
    // scores, counters and histories on every frame.
    expect(runs[0]).toEqual(runs[1]);
  });
});
