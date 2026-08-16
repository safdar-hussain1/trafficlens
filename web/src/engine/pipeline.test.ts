/** The pipeline's frame clock has to move forwards.
 *
 * `step()` releases a departed track's state on `frameIndex - lastSeen >
 * maxAge`. That test is a subtraction, not a comparison against "now", so a
 * caller that hands the same pipeline a frame counter which starts again at 0
 * makes the difference NEGATIVE and reaping simply stops: the gate counters
 * keep every `_counted` entry, the previous anchors of vehicles that left the
 * frame stay behind, and the speed estimator keeps their history.
 *
 * That is not hypothetical. `app.ts` initialised `frameIndex = 0` inside its
 * detect loop while `stop()` and `run()` deliberately kept the pipeline, so
 * stopping and restarting handed a zero to a pipeline whose last-seen entries
 * were hundreds of frames ahead. These tests pin the mechanism the fix relies
 * on, in the engine where it lives. */

import { describe, expect, test } from "vitest";

import { Gate } from "./gate";
import { SessionPipeline } from "./pipeline";
import type { Detection } from "./tracker";

const MAX_AGE = 4;

function pipeline(): SessionPipeline {
  return new SessionPipeline({
    gates: [new Gate("g", [0, 50], [100, 50])],
    plane: null,
    fps: 30,
    speedLimitKmh: null,
    tracker: { maxAge: MAX_AGE, minHits: 2 },
  });
}

/** One box, walked down the frame so its bottom-centre anchor crosses y = 50.
 * The step is 2 px because `TRACK_MATCH_IOU` is 0.8: a 40 px box moving 2 px a
 * frame overlaps itself by 0.90, and one moving faster would be a new track
 * every frame instead of the single track this test is about. */
const WALK = 16;

function box(step: number): Detection[] {
  const y2 = 44 + step * 2;
  return [
    { x1: 40, y1: y2 - 40, x2: 80, y2, score: 0.9, classId: 2, className: "car" },
  ];
}

/** Run the object through, then idle for `idle` frames with the given clock. */
function walkThenIdle(clock: (i: number) => number, idle: number): SessionPipeline {
  const p = pipeline();
  let frame = 0;
  for (let i = 0; i < WALK; i += 1, frame += 1) {
    p.step(box(i), frame, frame / 30);
  }
  for (let i = 0; i < idle; i += 1, frame += 1) {
    p.step([], clock(i), frame / 30);
  }
  return p;
}

describe("SessionPipeline reaping", () => {
  test("releases a departed track once the gap exceeds maxAge", () => {
    const monotonic = walkThenIdle((i) => WALK + i, MAX_AGE + 2);
    expect(monotonic.retainedTracks).toBe(0);
  });

  test("holds the track while the gap is still within maxAge", () => {
    // The control on the other side of the boundary: reaping must not be eager
    // either, or a track that re-associates at exactly maxAge would have had
    // its `_counted` entry thrown away and would be counted twice.
    const early = walkThenIdle((i) => WALK + i, MAX_AGE);
    expect(early.retainedTracks).toBe(1);
  });

  test("a frame clock that restarts at 0 stops reaping altogether", () => {
    // The defect, reproduced: same frames, same detections, same idle period --
    // only the clock differs, and nothing is ever released.
    const rewound = walkThenIdle(() => 0, MAX_AGE + 2);
    expect(rewound.retainedTracks).toBe(1);
  });

  test("a track that was counted stays counted while its state is retained", () => {
    // Why the retained state matters rather than merely leaking: the gate's
    // `_counted` set is what stops a lingering track being counted twice, and
    // `forget()` on reap is the only thing that ever empties it.
    const p = walkThenIdle((i) => WALK + i, 0);
    expect(p.total()).toBe(1);
    expect(p.retainedTracks).toBe(1);
  });
});
