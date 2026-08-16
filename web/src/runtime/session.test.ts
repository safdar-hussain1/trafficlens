// createSession is thin on purpose -- it exists so the execution-provider
// choice, the wasm asset paths and the byte-accurate progress are decided in
// one place rather than at the call site. What is worth testing is exactly
// that: that the chosen backend reaches onnxruntime, and that a WebGPU session
// that cannot be created falls back rather than leaving the page dead.

import { describe, expect, it } from "vitest";

import {
  ISOLATED_WASM_THREADS,
  ORT_ENTRY,
  browserDeps,
  createSession,
  vendoredDirectory,
  vendoredUrl,
} from "./session";

const BYTES = new Uint8Array([0, 1, 2, 3]).buffer;

function deps(overrides: Partial<Parameters<typeof createSession>[3]> = {}) {
  const created: Array<{ providers: unknown; options: Record<string, unknown> }> = [];
  return {
    created,
    value: {
      load: async () => BYTES,
      create: async (_bytes: ArrayBuffer, options: Record<string, unknown>) => {
        created.push({ providers: options["executionProviders"], options });
        return { inputNames: ["images"], outputNames: ["output0"] };
      },
      ...overrides,
    } as Parameters<typeof createSession>[3],
  };
}

describe("createSession", () => {
  it("asks onnxruntime for the WebGPU provider when the probe chose webgpu", async () => {
    const d = deps();
    await createSession("/models/m.onnx", "webgpu", undefined, d.value);
    expect(d.created[0]?.providers).toEqual(["webgpu"]);
  });

  // Same call, one axis varied: the execution provider the probe selected.
  it("asks for the wasm provider when the probe chose wasm", async () => {
    const d = deps();
    await createSession("/models/m.onnx", "wasm", undefined, d.value);
    expect(d.created[0]?.providers).toEqual(["wasm"]);
  });

  it("forwards download progress to the caller", async () => {
    const seen: number[] = [];
    const d = deps({
      load: async (_url: string, onProgress?: (p: { loaded: number }) => void) => {
        onProgress?.({ loaded: 512 });
        return BYTES;
      },
    } as never);
    await createSession("/models/m.onnx", "wasm", (p) => seen.push(p.loaded), d.value);
    expect(seen).toEqual([512]);
  });

  it("falls back to wasm when a WebGPU session will not initialise", async () => {
    let attempt = 0;
    const d = deps({
      create: async (_bytes: ArrayBuffer, options: Record<string, unknown>) => {
        attempt += 1;
        if (attempt === 1) throw new Error("no available backend found");
        return { inputNames: ["images"], outputNames: ["output0"], options };
      },
    } as never);
    const session = await createSession("/models/m.onnx", "webgpu", undefined, d.value);
    expect(attempt).toBe(2);
    expect(session.ep).toBe("wasm");
  });

  // The must-not-swallow half: a wasm session that fails has nothing left to
  // fall back to, so the error has to surface rather than be retried forever.
  it("surfaces a wasm session failure instead of looping", async () => {
    const d = deps({
      create: async () => {
        throw new Error("corrupt model");
      },
    } as never);
    await expect(
      createSession("/models/m.onnx", "wasm", undefined, d.value),
    ).rejects.toThrow(/corrupt model/);
  });
});

// --- the browser wiring -------------------------------------------------------
//
// `createSession` above is exercised through injected deps, which leaves the
// REAL deps -- the ones that decide where onnxruntime fetches its wasm from and
// how many threads it asks for -- with no coverage at all. Both were mutated
// and both survived the whole suite:
//
//   wasmPaths -> "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/"  190 passed
//   numThreads -> always 4                                             190 passed
//
// The first is the one that matters. It undoes this task's headline: the
// vendored-identity test would stay green while the browser executed a
// different wasm binary entirely, so `public/` would be provably pristine and
// provably unused.

describe("browserDeps", () => {
  const BASE = "https://example.github.io/trafficlens/";

  function fakeOrt() {
    return {
      env: { wasm: { wasmPaths: "", numThreads: 0 } },
      InferenceSession: {
        create: async () => ({ inputNames: ["images"], outputNames: ["output0"] }),
      },
    };
  }

  async function build(overrides: Parameters<typeof browserDeps>[0] = {}) {
    const ort = fakeOrt();
    const seen: string[] = [];
    const deps = browserDeps({
      baseUri: BASE,
      importOrt: async (url: string) => {
        seen.push(url);
        return ort as never;
      },
      ...overrides,
    });
    await deps.create(new ArrayBuffer(4), { executionProviders: ["wasm"] });
    return { ort, seen };
  }

  it("imports the vendored entry point from the page's own directory", async () => {
    const { seen } = await build();
    expect(seen).toEqual([`${BASE}${ORT_ENTRY}`]);
  });

  it("points wasmPaths at the directory holding the vendored entry", async () => {
    const { ort } = await build();
    // Derived from the entry rather than compared to a literal, so this states
    // the invariant -- the loader and binary sit BESIDE the entry point -- and
    // not merely today's string.
    expect(ort.env.wasm.wasmPaths).toBe(
      new URL("./", `${BASE}${ORT_ENTRY}`).href,
    );
    expect(ort.env.wasm.wasmPaths).toBe(vendoredDirectory(BASE));
  });

  it("keeps the runtime assets on the page's own origin", async () => {
    const { ort } = await build();
    // The half that kills a CDN: a jsdelivr URL resolves to a directory and
    // would satisfy any test that only checked the path ended in a slash.
    expect(new URL(ort.env.wasm.wasmPaths).origin).toBe(new URL(BASE).origin);
    expect(ort.env.wasm.wasmPaths.startsWith(BASE)).toBe(true);
  });

  it("resolves the runtime relative to a project subpath, not the domain root", async () => {
    // GitHub Pages serves this site from /trafficlens/, so an absolute
    // "/ort.webgpu.mjs" would 404. The control on the same axis is the
    // domain-root case, which must keep working.
    expect(vendoredUrl(ORT_ENTRY, BASE)).toBe(`${BASE}${ORT_ENTRY}`);
    expect(vendoredUrl(ORT_ENTRY, "https://example.com/")).toBe(
      `https://example.com/${ORT_ENTRY}`,
    );
  });

  it("asks for a single wasm thread when the page is not cross-origin isolated", async () => {
    const { ort } = await build({ isolated: false });
    // Binding, per the plan: GitHub Pages cannot send COOP/COEP, so
    // SharedArrayBuffer is unavailable and the published fallback figure was
    // measured single-threaded.
    expect(ort.env.wasm.numThreads).toBe(1);
  });

  // The control, varying exactly one axis: whether the page is isolated. A
  // hardcoded 1 would pass the test above on its own and be just as wrong.
  it("asks for more threads only when the page IS cross-origin isolated", async () => {
    const { ort } = await build({ isolated: true });
    expect(ort.env.wasm.numThreads).toBe(ISOLATED_WASM_THREADS);
    expect(ISOLATED_WASM_THREADS).toBeGreaterThan(1);
  });
});
