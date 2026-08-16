/** What the incidents panel is allowed to claim.
 *
 * The panel used to say two things at once. `renderPanels` returned early with
 * "none possible" for any uncalibrated source -- all three of them -- and then
 * `app.ts` overwrote the same element with the wrong-way list, which the
 * motorway gate can genuinely produce because it names an expected direction.
 * A panel that declares incidents impossible and then lists one is worse than
 * either statement alone.
 *
 * These are the real shipped sources, not invented ones: the divergence only
 * exists because `SOURCES` disagrees with itself about what is reachable. */

import { describe, expect, test } from "vitest";

import { incidentState } from "./controls";
import { sourceById } from "./sources";

const MOTORWAY = sourceById("motorway");
const STREET = sourceById("street");

describe("incidentState", () => {
  test("an uncalibrated gate that names a direction does NOT say impossible", () => {
    expect(MOTORWAY.calibrated).toBe(false);
    expect(MOTORWAY.gate.expectedDirection).toBe("toward");
    const state = incidentState({
      calibrated: MOTORWAY.calibrated,
      expectedDirection: MOTORWAY.gate.expectedDirection,
      wrongWay: [],
    });
    expect(state.kind).toBe("none");
    expect(state.kind === "none" ? state.reason : "").toContain("wrong-way");
  });

  test("and lists the crossings when there are some", () => {
    const state = incidentState({
      calibrated: MOTORWAY.calibrated,
      expectedDirection: MOTORWAY.gate.expectedDirection,
      wrongWay: ["car 7 crossed inbound the wrong way"],
    });
    expect(state).toEqual({
      kind: "alerts",
      lines: ["car 7 crossed inbound the wrong way"],
    });
  });

  test("an uncalibrated gate with no expected direction is genuinely impossible", () => {
    // The control: the claim "none possible" is not being removed, only moved
    // to the case where it is true. The street gate names no direction, so
    // neither incident kind can fire on it.
    expect(STREET.gate.expectedDirection).toBe(null);
    const state = incidentState({
      calibrated: STREET.calibrated,
      expectedDirection: STREET.gate.expectedDirection,
      wrongWay: [],
    });
    expect(state.kind).toBe("impossible");
  });

  test("a wrong-way crossing outranks the refusal on every source", () => {
    for (const expectedDirection of ["toward", null]) {
      for (const calibrated of [true, false]) {
        expect(
          incidentState({ calibrated, expectedDirection, wrongWay: ["one"] }).kind,
        ).toBe("alerts");
      }
    }
  });
});
