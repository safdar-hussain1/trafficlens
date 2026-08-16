// createSession is thin on purpose -- it exists so the execution-provider
// choice, the wasm asset paths and the byte-accurate progress are decided in
// one place rather than at the call site. What is worth testing is exactly
// that: that the chosen backend reaches onnxruntime, and that a WebGPU session
// that cannot be created falls back rather than leaving the page dead.

import { describe, expect, it } from "vitest";

import { createSession } from "./session";

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
