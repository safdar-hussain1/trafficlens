/** Creating the onnxruntime session the detector runs in.
 *
 * Thin on purpose. Its job is to make three decisions in ONE place instead of
 * at every call site: which execution provider to ask for, where the runtime's
 * own assets live, and what happens when a WebGPU session will not start.
 *
 * The onnxruntime bundle is imported at RUNTIME from `public/`, not bundled.
 * That is deliberate: `web/public/ort.webgpu.mjs` and the two
 * `ort-wasm-simd-threaded.jsep.*` files are byte-identical copies of what npm
 * published, and `vendored.test.ts` asserts it. If the bundler inlined and
 * transformed its own copy instead, that assertion would be about a file the
 * browser never executes. Importing the vendored URL keeps the bytes the test
 * checks and the bytes that run the model the same bytes. */

import { loadCachedBytes, type LoadProgress } from "./cache";
import type { ExecutionProvider } from "./backend";

/** The subset of `ort.InferenceSession` this project uses. Declared here so
 * neither the app nor the tests need onnxruntime's types at build time. */
export interface InferenceSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
}

export interface RuntimeSession {
  readonly session: InferenceSession;
  /** The provider the session was ACTUALLY created with, which is not always
   * the one that was asked for -- see the fallback below. */
  readonly ep: ExecutionProvider;
  readonly inputName: string;
  readonly outputName: string;
}

export interface SessionDeps {
  load(url: string, onProgress?: (progress: LoadProgress) => void): Promise<ArrayBuffer>;
  create(model: ArrayBuffer, options: Record<string, unknown>): Promise<InferenceSession>;
}

/** Resolve a vendored asset against the page, so the site works from a
 * project subpath on Pages as well as from a domain root. */
function vendoredUrl(name: string): string {
  return new URL(name, document.baseURI).href;
}

function browserDeps(): SessionDeps {
  return {
    load: (url, onProgress) => loadCachedBytes(url, { ...(onProgress ? { onProgress } : {}) }),
    create: async (model, options) => {
      const entry = vendoredUrl("ort.webgpu.mjs");
      // @vite-ignore: this URL is resolved by the browser at runtime and must
      // not be rewritten into a bundled chunk; see the module comment.
      const ort = (await import(/* @vite-ignore */ entry)) as {
        env: { wasm: { wasmPaths: string; numThreads: number } };
        InferenceSession: {
          create(model: ArrayBuffer, options: Record<string, unknown>): Promise<InferenceSession>;
        };
      };
      // The loader glue and the .wasm binary sit beside the entry point.
      ort.env.wasm.wasmPaths = new URL("./", entry).href;
      // GitHub Pages cannot send COOP/COEP, so `crossOriginIsolated` is false
      // and SharedArrayBuffer -- and therefore multi-threaded wasm -- is
      // unavailable. Asking for threads anyway makes onnxruntime probe, fail
      // and fall back noisily; asking for one is the honest configuration and
      // is the one the published fallback figure was measured under.
      ort.env.wasm.numThreads = globalThis.crossOriginIsolated === true ? 4 : 1;
      return ort.InferenceSession.create(model, options);
    },
  };
}

function optionsFor(ep: ExecutionProvider): Record<string, unknown> {
  return {
    executionProviders: [ep],
    graphOptimizationLevel: "all",
  };
}

/** Download (or reuse) the model and create a session on `ep`.
 *
 * A WebGPU session that will not initialise falls back to wasm rather than
 * failing the page: `probeBackend` can only tell us an adapter was granted,
 * and a session can still fail afterwards on a driver onnxruntime cannot use.
 * A wasm failure is NOT retried -- there is nothing further to fall back to,
 * and looping would turn a clear error into a hang. */
export async function createSession(
  url: string,
  ep: ExecutionProvider,
  onProgress?: ((progress: LoadProgress) => void) | undefined,
  deps: SessionDeps = browserDeps(),
): Promise<RuntimeSession> {
  const model = await deps.load(url, onProgress);

  const describe = (session: InferenceSession, provider: ExecutionProvider): RuntimeSession => ({
    session,
    ep: provider,
    inputName: session.inputNames[0] as string,
    outputName: session.outputNames[0] as string,
  });

  if (ep === "wasm") {
    return describe(await deps.create(model, optionsFor("wasm")), "wasm");
  }

  try {
    return describe(await deps.create(model, optionsFor("webgpu")), "webgpu");
  } catch {
    return describe(await deps.create(model, optionsFor("wasm")), "wasm");
  }
}
