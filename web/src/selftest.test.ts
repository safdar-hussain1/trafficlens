/** The selftest's floors: can `?selftest=1` still notice an empty fixture?
 *
 * `runSelftest` compares what the shipped engine produced with what the Python
 * engine produced, and every one of those comparisons is satisfied by an empty
 * fixture -- `[]` equals `[]`. That is not hypothetical: emptying every
 * `frames` and every `steps` array in the committed fixture produced a full
 * green board with `SessionPipeline.step()` never called once. The check ran
 * against the SHIPPED artefact, which is the worst possible place for a check
 * that cannot notice it has nothing to do.
 *
 * These tests are the hollow-fixture attack, kept. The control is deliberately
 * on a different axis: a fixture that is smaller but still real must survive,
 * so the floors are shown to be about work performed rather than about size. */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";

import { runSelftest } from "./selftest";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Fixture = any;

const FIXTURE = JSON.parse(
  readFileSync(fileURLToPath(new URL("./fixtures/parity.json", import.meta.url)), "utf8"),
) as Fixture;

function clone(): Fixture {
  return JSON.parse(JSON.stringify(FIXTURE)) as Fixture;
}

function floorNames(report: ReturnType<typeof runSelftest>): string[] {
  return report.checks.filter((c) => !c.passed).map((c) => c.name);
}

describe("the committed fixture", () => {
  const report = runSelftest(clone());

  test("passes, and the verdict says so", () => {
    expect(floorNames(report)).toEqual([]);
    expect(report.failed).toBe(0);
    expect(report.title.startsWith("PASS ")).toBe(true);
  });

  test("scores every check it claims to", () => {
    expect(report.checks.length).toBe(report.passed + report.failed);
    expect(report.title).toContain(`${report.passed}/${report.checks.length}`);
  });
});

describe("the hollow-fixture attack", () => {
  test("emptying every frames[] and steps[] fails, and does not pass vacuously", () => {
    const hollow = clone();
    for (const testCase of hollow.trackerCases as Fixture[]) {
      testCase.frames = [];
      testCase.expected.frames = [];
      testCase.expected.events = [];
      testCase.expected.counts = {};
      testCase.expected.tracksAllocated = 0;
    }
    for (const testCase of hollow.gateCases as Fixture[]) {
      testCase.steps = [];
      testCase.expected.events = [];
      testCase.expected.counts = {};
    }

    const report = runSelftest(hollow);

    expect(report.failed).toBeGreaterThan(0);
    expect(report.title.startsWith("FAIL ")).toBe(true);
    // The specific floors, named: every one of these is about work the engine
    // performed, and none of them is a label the fixture writes about itself.
    expect(floorNames(report)).toEqual(
      expect.arrayContaining([
        "floor: pipeline steps performed",
        "floor: detections fed to the tracker",
        "floor: gate counter updates performed",
        "floor: crossing events emitted by the engine",
      ]),
    );
  });

  test("deleting the decode cases fails", () => {
    const hollow = clone();
    hollow.decodeCases = [];
    const report = runSelftest(hollow);
    expect(report.title.startsWith("FAIL ")).toBe(true);
    expect(floorNames(report)).toEqual(
      expect.arrayContaining([
        "floor: decode cases replayed",
        "floor: detections decoded",
        "floor: checks scored",
      ]),
    );
  });

  test("dropping a single tracker case fails", () => {
    const hollow = clone();
    hollow.trackerCases = (hollow.trackerCases as Fixture[]).slice(0, -1);
    const report = runSelftest(hollow);
    expect(report.title.startsWith("FAIL ")).toBe(true);
    expect(floorNames(report)).toEqual(
      expect.arrayContaining(["floor: tracker cases replayed"]),
    );
  });

  test("truncating one case's frames fails even though every comparison agrees", () => {
    // The subtlest form: the case still exists, still carries its straddle
    // labels, and its expectations are trimmed to match, so nothing disagrees.
    const hollow = clone();
    const big = (hollow.trackerCases as Fixture[]).find(
      (c) => (c.frames as Fixture[]).length > 100,
    ) as Fixture;
    big.frames = (big.frames as Fixture[]).slice(0, 2);
    big.expected.frames = (big.expected.frames as Fixture[]).slice(0, 2);
    big.expected.events = [];
    big.expected.counts = {};
    big.expected.tracksAllocated = 0;
    const report = runSelftest(hollow);
    expect(report.title.startsWith("FAIL ")).toBe(true);
    expect(floorNames(report)).toEqual(
      expect.arrayContaining([
        "floor: pipeline steps performed",
        "floor: crossing events emitted by the engine",
      ]),
    );
  });
});

describe("the control, on a different axis", () => {
  test("a fixture edited in a way that does not remove work still passes", () => {
    // Axis: fixture METADATA, not fixture volume. The straddle labels are
    // re-ordered and the schema version is bumped -- nothing the engine reads
    // and no frame removed. If this went red the floors would be a smoke alarm
    // rather than a check on work performed.
    const edited = clone();
    edited.schemaVersion = 99;
    for (const testCase of edited.trackerCases as Fixture[]) {
      testCase.straddles = [...(testCase.straddles as string[])].reverse();
    }
    const report = runSelftest(edited);
    expect(floorNames(report)).toEqual([]);
    expect(report.title.startsWith("PASS ")).toBe(true);
  });
});
