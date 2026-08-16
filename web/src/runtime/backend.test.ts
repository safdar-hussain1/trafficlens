// The backend probe decides whether the visitor's own GPU runs the detector or
// a wasm fallback does, and the product publishes which one it chose along with
// the renderer string. A probe that silently reported "webgpu" on a machine
// without it, or that reported a software renderer as if it were hardware,
// would make the published timing meaningless -- so the tests below are about
// the probe telling the truth, not about it succeeding.

import { describe, expect, it } from "vitest";

import { SOFTWARE_RENDERER_MARKERS, probeBackend } from "./backend";

const APPLE_M1 = "ANGLE (Apple, ANGLE Metal Renderer: Apple M1 Pro, Unspecified Version)";

function environment(overrides: {
  adapter?: { info?: Record<string, unknown> } | null;
  requestAdapter?: () => Promise<unknown>;
  renderer?: string;
} = {}) {
  const gpu =
    overrides.requestAdapter !== undefined
      ? { requestAdapter: overrides.requestAdapter }
      : overrides.adapter !== undefined
        ? { requestAdapter: async () => overrides.adapter ?? null }
        : undefined;
  return {
    ...(gpu === undefined ? {} : { gpu }),
    readRenderer: () => overrides.renderer ?? APPLE_M1,
  } as Parameters<typeof probeBackend>[0];
}

describe("probeBackend", () => {
  it("selects webgpu and reports the adapter when an adapter is granted", async () => {
    const probe = await probeBackend(
      environment({ adapter: { info: { vendor: "apple", architecture: "common-3" } } }),
    );
    expect(probe.ep).toBe("webgpu");
    expect(probe.renderer).toBe(APPLE_M1);
    expect(probe.adapter).toContain("apple");
  });

  // The must-fall-back half. Same probe, one axis varied: whether the browser
  // exposes navigator.gpu at all.
  it("falls back to wasm when the browser has no WebGPU", async () => {
    const probe = await probeBackend(environment({}));
    expect(probe.ep).toBe("wasm");
    expect(probe.renderer).toBe(APPLE_M1);
  });

  it("falls back to wasm when WebGPU exists but grants no adapter", async () => {
    const probe = await probeBackend(environment({ adapter: null }));
    expect(probe.ep).toBe("wasm");
    // Asserting the reason, not just the provider. Deleting the null check
    // still yields "wasm" -- reading `.info` off null throws and the catch
    // below it returns wasm anyway -- so a test that only checked `ep` would
    // pass with the check removed (measured: that mutation survived). What
    // actually breaks is the diagnostic: "no adapter granted" degrades into a
    // TypeError about reading properties of null, which is the difference
    // between a page that can explain itself and one that cannot.
    expect(probe.adapter).toBe("WebGPU present but no adapter granted");
  });

  it("falls back to wasm when requesting an adapter throws", async () => {
    const probe = await probeBackend(
      environment({
        requestAdapter: async () => {
          throw new Error("gpu process crashed");
        },
      }),
    );
    expect(probe.ep).toBe("wasm");
    expect(probe.adapter).toContain("gpu process crashed");
  });

  it("still answers when the renderer cannot be read at all", async () => {
    const probe = await probeBackend({
      gpu: { requestAdapter: async () => ({ info: {} }) },
      readRenderer: () => {
        throw new Error("no webgl context");
      },
    });
    expect(probe.ep).toBe("webgpu");
    expect(probe.renderer).toBe("unknown");
  });

  // Notes §3: a software-renderer string invalidates a published timing. The
  // probe cannot refuse to run, but it must mark the result so a number taken
  // from it is never published as hardware.
  it("marks a software renderer so its timings are not published as hardware", async () => {
    for (const marker of SOFTWARE_RENDERER_MARKERS) {
      const probe = await probeBackend(
        environment({ adapter: { info: {} }, renderer: `ANGLE (${marker} Direct3D11)` }),
      );
      expect(probe.isHardwareRenderer).toBe(false);
    }
  });

  it("reports a real GPU renderer string as hardware", async () => {
    const probe = await probeBackend(environment({ adapter: { info: {} } }));
    expect(probe.isHardwareRenderer).toBe(true);
  });
});
