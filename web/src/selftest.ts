/** `?selftest=1`: does the shipped page still decide what the Python engine
 * decided?
 *
 * The committed parity fixtures were written by `scripts/make_parity_fixtures.py`
 * running the PYTHON engine -- every input and every expected output. This
 * module replays them through the SAME objects the control room runs: the real
 * `SessionPipeline`, the real `decodeYolo`, the real generated constants, in the
 * visitor's own browser, from the deployed bundle. A green title here is
 * therefore a statement about the artefact that is actually serving, not about
 * a test environment that resembles it.
 *
 * The verdict goes into `document.title` because that is the one piece of page
 * state a headless Chrome can be polled for over the DevTools endpoint without
 * evaluating anything in the page -- see `scripts/verify_page.sh`.
 *
 * A check that cannot fail proves nothing, so this one is mutation-checked:
 * corrupting a single constant in `generated/constants.ts` must turn the title
 * red. The report below lists every check by name for that reason -- when it
 * does go red, the name says which decision moved. */

import { Gate, GateCounter } from "./engine/gate";
import type { CrossingEvent } from "./engine/gate";
import { RoadPlane } from "./engine/homography";
import { SessionPipeline } from "./engine/pipeline";
import type { Counts } from "./engine/pipeline";
import { Tracker } from "./engine/tracker";
import type { Detection } from "./engine/tracker";
import { decodeYolo } from "./runtime/postprocess";

/** The plan's speed tolerance, and the same one `parity.test.ts` uses. */
const SPEED_TOLERANCE_KMH = 1e-6;

/** Crossing points are a pure float64 line intersection on both sides. */
const POSITION_TOLERANCE_PX = 1e-9;

/** Floors on the WORK PERFORMED, not on what the fixture says about itself.
 *
 * Every comparison in this file is `got` against `want`, and an empty `got`
 * compares equal to an empty `want`. A fixture whose `frames` and `steps` arrays
 * were all emptied therefore reports a full green board: `SessionPipeline` gets
 * constructed and `step()` is never called, `GateCounter` gets constructed and
 * `update()` is never called, and dozens of checks pass by comparing nothing to
 * nothing. That was reproduced -- `PASS 38/38`, exit 0, with the engine never
 * asked a single question. The straddle list above does not catch it either,
 * because `straddles` is a label the fixture writes about itself.
 *
 * So these numbers are counted while the engine runs and asserted at the end.
 * They are set at the committed fixture's own totals rather than at 1: a floor
 * of "more than zero" is defeated by leaving one frame in place, and a fixture
 * that legitimately shrinks should have to be re-justified here. They are `>=`
 * so the fixture may grow.
 *
 * `MIN_CHECKS` is the last line of defence: `scripts/verify_page.sh` matches on
 * the `PASS` prefix alone, so `PASS 0/0` would exit 0. It cannot arise now. */
const MIN_TRACKER_CASES = 4;
const MIN_GATE_CASES = 3;
const MIN_DECODE_CASES = 1;
/** `pipeline.step()` calls: 162 across the four committed tracker cases. */
const MIN_PIPELINE_STEPS = 162;
/** `counter.update()` calls in the gate cases: 12 committed. */
const MIN_GATE_UPDATES = 12;
/** Detections fed to the tracker across every tracker case: 1098 committed. */
const MIN_DETECTIONS_REPLAYED = 1098;
/** Crossing events the ENGINE emitted: 3 from the tracker cases, 4 from the
 * gate cases. Zero here means nothing was ever counted. */
const MIN_EVENTS_EMITTED = 7;
/** Detections `decodeYolo` returned across the decode cases. */
const MIN_DETECTIONS_DECODED = 2;
/** Checks scored before this floor is itself scored: 67 on the committed
 * fixture, 68 in the title once this one is counted. */
const MIN_CHECKS = 67;

/** Every boundary kind the fixture must carry, written out here rather than
 * read from the fixture: a list the fixture supplied would be satisfied by
 * whatever the fixture happened to contain. */
const REQUIRED_STRADDLES = [
  "anchorExactlyOnGate",
  "iouExactlyAtMatchThresh",
  "scoreExactlyAtHighThresh",
  "assignmentCostExactTie",
  "argmaxFloat32ClassTie",
  "deferredOnLineUsesLastOffLinePoint",
] as const;

export interface CheckResult {
  readonly name: string;
  readonly passed: boolean;
  readonly detail: string;
}

export interface SelftestReport {
  readonly passed: number;
  readonly failed: number;
  readonly checks: readonly CheckResult[];
  readonly title: string;
}

interface Recorder {
  check(name: string, ok: boolean, detail?: string): void;
}

function recorder(): { results: CheckResult[]; api: Recorder } {
  const results: CheckResult[] = [];
  return {
    results,
    api: {
      check(name, ok, detail = "") {
        results.push({ name, passed: ok, detail });
      },
    },
  };
}

function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function near(a: number | null, b: number | null, tolerance: number): boolean {
  if (a === null || b === null) {
    return a === b;
  }
  return Math.abs(a - b) <= tolerance;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
type Fixture = any;

function toGate(spec: Fixture): Gate {
  return new Gate(spec.name, spec.start, spec.end, {
    labelPositive: spec.labelPositive,
    labelNegative: spec.labelNegative,
  });
}

function decisionsOf(event: CrossingEvent | Fixture): unknown {
  return {
    trackId: event.trackId,
    className: event.className,
    gate: event.gate,
    direction: event.direction,
    signedDirection: event.signedDirection,
    frameIndex: event.frameIndex,
    isViolation: event.isViolation,
  };
}

function checkEvents(
  record: Recorder,
  label: string,
  got: readonly CrossingEvent[],
  want: readonly Fixture[],
): void {
  record.check(
    `${label}: crossing decisions`,
    same(got.map(decisionsOf), want.map(decisionsOf)),
    `${got.length} events, expected ${want.length}`,
  );
  for (let i = 0; i < Math.min(got.length, want.length); i += 1) {
    const g = got[i] as CrossingEvent;
    const w = want[i] as Fixture;
    record.check(
      `${label}: event ${i} crossing point`,
      near(g.crossingX, w.crossingX, POSITION_TOLERANCE_PX) &&
        near(g.crossingY, w.crossingY, POSITION_TOLERANCE_PX) &&
        g.timestamp === w.timestamp,
      `(${g.crossingX}, ${g.crossingY}) vs (${w.crossingX}, ${w.crossingY})`,
    );
    record.check(
      `${label}: event ${i} speed`,
      near(g.speedKmh, w.speedKmh, SPEED_TOLERANCE_KMH),
      `${String(g.speedKmh)} vs ${String(w.speedKmh)}`,
    );
  }
}

function countsOfCounter(name: string, counter: GateCounter): Counts {
  const out: Counts = {};
  for (const [className, directions] of counter.totals) {
    for (const [direction, count] of directions) {
      ((out[name] ??= {})[className] ??= {})[direction] = count;
    }
  }
  return out;
}

/** Replay every fixture case through the shipped engine and score it. */
export function runSelftest(fixture: Fixture): SelftestReport {
  const { results, api } = recorder();

  // The fixture has to be the one that was committed, carrying every mandated
  // boundary case: a suite with nothing to check passes trivially.
  const seen = new Set<string>();
  for (const testCase of [
    ...fixture.trackerCases,
    ...fixture.gateCases,
    ...fixture.decodeCases,
  ] as Fixture[]) {
    for (const kind of testCase.straddles as string[]) {
      seen.add(kind);
    }
  }
  for (const kind of REQUIRED_STRADDLES) {
    api.check(`fixture carries ${kind}`, seen.has(kind));
  }

  // Counted as the engine runs, asserted at the end. See the floors above.
  let pipelineSteps = 0;
  let detectionsReplayed = 0;
  let gateUpdates = 0;
  let eventsEmitted = 0;
  let detectionsDecoded = 0;

  for (const testCase of fixture.trackerCases as Fixture[]) {
    const pipeline = new SessionPipeline({
      gates: (testCase.gates as Fixture[]).map(toGate),
      plane: new RoadPlane(fixture.plane.imageToWorld),
      fps: fixture.source.fps,
      speedLimitKmh: fixture.speedLimitKmh,
      tracker: {
        highThresh: fixture.tracker.highThresh,
        lowThresh: fixture.tracker.lowThresh,
        matchThresh: fixture.tracker.matchThresh,
        maxAge: fixture.tracker.maxAge,
        minHits: fixture.tracker.minHits,
      },
    });
    const events: CrossingEvent[] = [];
    const rows: { frameIndex: number; tracks: Fixture[] }[] = [];

    api.check(
      `${testCase.name}: fixture supplies frames to replay`,
      (testCase.frames as Fixture[]).length > 0,
      `${(testCase.frames as Fixture[]).length} frames`,
    );
    for (const frame of testCase.frames as Fixture[]) {
      const detections = (frame.detections as Fixture[]).map(
        (d): Detection => ({
          x1: d.x1,
          y1: d.y1,
          x2: d.x2,
          y2: d.y2,
          score: d.score,
          classId: d.classId,
          className: d.className,
        }),
      );
      const step = pipeline.step(detections, frame.frameIndex, frame.timestamp);
      pipelineSteps += 1;
      detectionsReplayed += detections.length;
      eventsEmitted += step.events.length;
      events.push(...step.events);
      rows.push({
        frameIndex: frame.frameIndex,
        tracks: step.tracks.map((t) => ({
          trackId: t.trackId,
          className: t.className,
          speedKmh: t.speedKmh,
        })),
      });
    }

    const wantFrames = testCase.expected.frames as Fixture[];
    api.check(
      `${testCase.name}: same frames`,
      same(
        rows.map((r) => r.frameIndex),
        wantFrames.map((f: Fixture) => f.frameIndex),
      ),
    );
    let idsAgree = true;
    let speedsAgree = true;
    for (let i = 0; i < Math.min(rows.length, wantFrames.length); i += 1) {
      const got = (rows[i] as Fixture).tracks as Fixture[];
      const want = (wantFrames[i] as Fixture).tracks as Fixture[];
      if (
        !same(
          got.map((t) => [t.trackId, t.className]),
          want.map((t) => [t.trackId, t.className]),
        )
      ) {
        idsAgree = false;
      }
      for (let j = 0; j < Math.min(got.length, want.length); j += 1) {
        if (
          !near(
            (got[j] as Fixture).speedKmh,
            (want[j] as Fixture).speedKmh,
            SPEED_TOLERANCE_KMH,
          )
        ) {
          speedsAgree = false;
        }
      }
    }
    api.check(`${testCase.name}: track ids and classes`, idsAgree);
    api.check(`${testCase.name}: speeds within ${SPEED_TOLERANCE_KMH} km/h`, speedsAgree);
    api.check(
      `${testCase.name}: tracks allocated`,
      pipeline.tracksAllocated === testCase.expected.tracksAllocated,
      `${pipeline.tracksAllocated} vs ${testCase.expected.tracksAllocated}`,
    );
    checkEvents(api, testCase.name, events, testCase.expected.events as Fixture[]);
    api.check(
      `${testCase.name}: counts`,
      same(pipeline.counts(), testCase.expected.counts),
      JSON.stringify(pipeline.counts()),
    );
  }

  for (const testCase of fixture.gateCases as Fixture[]) {
    const counter = new GateCounter(toGate(testCase.gate));
    const events: CrossingEvent[] = [];
    api.check(
      `${testCase.name}: fixture supplies steps to replay`,
      (testCase.steps as Fixture[]).length > 0,
      `${(testCase.steps as Fixture[]).length} steps`,
    );
    for (const step of testCase.steps as Fixture[]) {
      gateUpdates += 1;
      const event = counter.update(
        step.trackId,
        step.className,
        step.prev,
        step.curr,
        step.frameIndex,
        step.timestamp,
        null,
        fixture.speedLimitKmh,
      );
      if (event !== null) {
        events.push(event);
        eventsEmitted += 1;
      }
    }
    checkEvents(api, testCase.name, events, testCase.expected.events as Fixture[]);
    api.check(
      `${testCase.name}: counts`,
      same(countsOfCounter(testCase.gate.name, counter), testCase.expected.counts),
    );
  }

  for (const testCase of fixture.decodeCases as Fixture[]) {
    const keepClasses = new Map(
      (testCase.keepClasses as Fixture[]).map((c) => [c.classId, c.className] as const),
    );
    const decoded = decodeYolo(
      { data: Float32Array.from(testCase.raw as number[]), dims: testCase.dims },
      testCase.scale,
      testCase.padX,
      testCase.padY,
      { conf: testCase.conf, iou: testCase.iou, keepClasses },
    );
    detectionsDecoded += decoded.length;
    const shape = (d: Fixture): unknown => ({
      x1: d.x1,
      y1: d.y1,
      x2: d.x2,
      y2: d.y2,
      score: d.score,
      classId: d.classId,
      className: d.className,
    });
    api.check(
      `${testCase.name}: decoded detections`,
      same(decoded.map(shape), (testCase.expectedDetections as Fixture[]).map(shape)),
    );

    const tracker = new Tracker({
      highThresh: fixture.tracker.highThresh,
      lowThresh: fixture.tracker.lowThresh,
      matchThresh: fixture.tracker.matchThresh,
      maxAge: fixture.tracker.maxAge,
      minHits: fixture.tracker.minHits,
    });
    let tracks: ReturnType<Tracker["update"]> = [];
    for (let frame = 0; frame < testCase.replayFrames; frame += 1) {
      tracks = tracker.update(decoded, frame);
    }
    api.check(
      `${testCase.name}: decoded class survives the tracker`,
      same(
        tracks.map((t) => [t.trackId, t.className]),
        (testCase.expectedTracks as Fixture[]).map((t) => [t.trackId, t.className]),
      ),
    );
  }

  // The floors. Everything above this point compares `got` with `want` and is
  // satisfied by an empty fixture; these are the only checks that fail when the
  // engine was never actually asked anything.
  const floors: readonly (readonly [string, number, number])[] = [
    ["tracker cases replayed", (fixture.trackerCases as Fixture[]).length, MIN_TRACKER_CASES],
    ["gate cases replayed", (fixture.gateCases as Fixture[]).length, MIN_GATE_CASES],
    ["decode cases replayed", (fixture.decodeCases as Fixture[]).length, MIN_DECODE_CASES],
    ["pipeline steps performed", pipelineSteps, MIN_PIPELINE_STEPS],
    ["detections fed to the tracker", detectionsReplayed, MIN_DETECTIONS_REPLAYED],
    ["gate counter updates performed", gateUpdates, MIN_GATE_UPDATES],
    ["crossing events emitted by the engine", eventsEmitted, MIN_EVENTS_EMITTED],
    ["detections decoded", detectionsDecoded, MIN_DETECTIONS_DECODED],
  ];
  for (const [name, got, floor] of floors) {
    api.check(`floor: ${name}`, got >= floor, `${got} (floor ${floor})`);
  }
  // Counted before this check is pushed, so the floor is about the checks that
  // scored work rather than about itself.
  const scored = results.length;
  api.check(
    `floor: checks scored`,
    scored >= MIN_CHECKS,
    `${scored} (floor ${MIN_CHECKS})`,
  );

  const failed = results.filter((r) => !r.passed).length;
  const passed = results.length - failed;
  const first = results.find((r) => !r.passed);
  const title =
    failed === 0
      ? `PASS ${passed}/${results.length} — browser engine agrees with the Python engine`
      : `FAIL ${failed}/${results.length} — first: ${first?.name ?? "unknown"}`;
  return { passed, failed, checks: results, title };
}

/** Render the verdict into the page and, crucially, the tab title. */
export function renderSelftest(report: SelftestReport): void {
  document.title = report.title;
  const root = document.createElement("div");
  root.className = "report";
  root.dataset["verdict"] = report.failed === 0 ? "PASS" : "FAIL";

  const heading = document.createElement("h1");
  heading.textContent = report.title;
  root.append(heading);

  const list = document.createElement("pre");
  list.textContent = report.checks
    .map((check) => `${check.passed ? "ok  " : "FAIL"} ${check.name}${check.detail === "" ? "" : `  (${check.detail})`}`)
    .join("\n");
  root.append(list);
  document.body.replaceChildren(root);
}

export async function runSelftestPage(): Promise<SelftestReport> {
  document.title = "RUNNING selftest";
  try {
    const module = (await import("./fixtures/parity.json")) as { default: Fixture };
    const report = runSelftest(module.default);
    renderSelftest(report);
    return report;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const report: SelftestReport = {
      passed: 0,
      failed: 1,
      checks: [{ name: "selftest ran at all", passed: false, detail: message }],
      title: `FAIL selftest could not run — ${message}`,
    };
    renderSelftest(report);
    return report;
  }
}
