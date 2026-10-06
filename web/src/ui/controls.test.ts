/** What the status pill is allowed to claim about this machine.
 *
 * The pill prints what was measured in this tab and names what ran it. The
 * hazards are attribution, not arithmetic: a GPU named beside a figure the GPU
 * never produced, a software-renderer caveat attached to the wrong path, or a
 * probe's preference reported as if it were the provider the session actually
 * got. Each test below varies one of those axes and holds the others. */

import { describe, expect, test } from "vitest";

import type { BackendProbe } from "../runtime/backend";
import { badgeContent } from "./controls";
import { decideCadence } from "./format";

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

describe("badgeContent", () => {
  test("the WASM path does not print the GL renderer beside its ms/frame", () => {
    // C12: the WebGL renderer string describes the WebGL/WebGPU path, which
    // WASM inference never touches. Measured -- forcing a software renderer
    // moved the WASM figure by about 2 % while the string changed completely --
    // so printing it beside the figure asserts a hardware attribution nobody
    // performed.
    const content = badgeContent(state(HARDWARE, "wasm"));
    expect(content.ep).toBe("WASM");
    expect(content.renderer).toBe("n/a (wasm path)");
    expect(content.renderer).not.toContain("Apple");
    expect(content.msPerFrame).toBe("40.0 ms/frame");
    expect(content.fps).toBe("25.0 fps");
  });

  test("and the WebGPU path still names the device it is running on", () => {
    // The control, varying the EXECUTION PROVIDER rather than the probe: a
    // pill that had simply stopped printing renderers would satisfy the
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
    expect(content.renderer).toBe("");
    expect(content.msPerFrame).toBeNull();
    expect(content.fps).toBeNull();
    expect(content.cadence).toBeNull();
    expect(content.softwareWarning).toBe(false);
  });

  test("a figure that has not been measured is absent, not zero", () => {
    // The probe has returned but nothing has run yet: the provider is known
    // and the figures are not, and a default here would be a number nobody
    // measured.
    const content = badgeContent({
      probe: HARDWARE, ep: null, msPerFrame: null, fps: null, cadence: null,
    });
    expect(content.probed).toBe(true);
    expect(content.msPerFrame).toBeNull();
    expect(content.fps).toBeNull();
  });

  test("the cadence is said out loud only when it is not every frame", () => {
    // Every frame is the expected case and saying it is noise; anything less is
    // what a visitor would otherwise mistake for a stuttering demo.
    const everyFrame = decideCadence(20, 30);
    const thinned = decideCadence(70, 30);
    expect(everyFrame.stride).toBe(1);
    expect(thinned.stride).toBeGreaterThan(1);
    expect(badgeContent({ ...state(HARDWARE, "wasm"), cadence: everyFrame }).cadence).toBeNull();
    expect(badgeContent({ ...state(HARDWARE, "wasm"), cadence: thinned }).cadence).toBe(
      `detecting ${thinned.label}`,
    );
  });
});
