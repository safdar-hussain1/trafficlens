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

import type { BackendProbe } from "../runtime/backend";
import { badgeContent, incidentState } from "./controls";
import { sourceById } from "./sources";

/** A probe naming a real GPU. The renderer string is the one C12 recorded. */
const HARDWARE: BackendProbe = {
  ep: "wasm",
  renderer: "ANGLE (Apple, ANGLE Metal Renderer: Apple M1 Pro)",
  adapter: "no adapter",
  isHardwareRenderer: true,
};

const SOFTWARE: BackendProbe = {
  ...HARDWARE,
  renderer: "SwiftShader",
  isHardwareRenderer: false,
};

function state(probe: BackendProbe, ep: string | null) {
  return { probe, ep, msPerFrame: 40, fps: 25, cadence: null };
}

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

describe("badgeContent", () => {
  test("the WASM path does not print the GL renderer beside its ms/frame", () => {
    // C12: the WebGL renderer string describes the WebGL/WebGPU path, which
    // WASM inference never touches. Measured -- forcing a software renderer
    // moved the WASM figure by about 2 % while the string changed completely --
    // so printing it beside the figure asserts a hardware attribution nobody
    // performed. The numbers file was corrected for exactly this; the badge a
    // visitor actually reads was not.
    const content = badgeContent(state(HARDWARE, "wasm"));
    expect(content.ep).toBe("WASM");
    expect(content.renderer).toBe("n/a (wasm path)");
    expect(content.renderer).not.toContain("Apple");
    expect(content.figures).toContain("40.0 ms/frame");
  });

  test("and the WebGPU path still names the device it is running on", () => {
    // The control, varying the EXECUTION PROVIDER rather than the probe: a
    // badge that had simply stopped printing renderers would satisfy the
    // assertion above while losing a true attribution.
    const content = badgeContent(state(HARDWARE, "webgpu"));
    expect(content.ep).toBe("WebGPU");
    expect(content.renderer).toBe(HARDWARE.renderer);
  });

  test("the software-renderer caveat is attached to the path it describes", () => {
    // Same axis as the renderer string: on WASM the caveat would be describing
    // a device that ran none of the work being timed.
    expect(badgeContent(state(SOFTWARE, "wasm")).softwareWarning).toBe(false);
    expect(badgeContent(state(SOFTWARE, "webgpu")).softwareWarning).toBe(true);
    // The control, varying the PROBE rather than the provider: hardware on the
    // same path must not warn, or the flag would be pinned to the path alone.
    expect(badgeContent(state(HARDWARE, "webgpu")).softwareWarning).toBe(false);
  });

  test("the session's provider outranks the probe's preference", () => {
    // `ep` is what the session was actually created with; the probe only says
    // what to ask for. A WebGPU probe that fell back to WASM must read WASM.
    const fellBack: BackendProbe = { ...HARDWARE, ep: "webgpu" };
    expect(badgeContent(state(fellBack, "wasm")).ep).toBe("WASM");
    expect(badgeContent(state(fellBack, null)).ep).toBe("WebGPU");
  });

  test("before the probe returns there is nothing to attribute", () => {
    const content = badgeContent({
      probe: null, ep: null, msPerFrame: null, fps: null, cadence: null,
    });
    expect(content.probed).toBe(false);
    expect(content.figures).toEqual([]);
    expect(content.softwareWarning).toBe(false);
  });
});
