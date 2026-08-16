/** Creating the onnxruntime session the detector runs in.
 *
 * Thin on purpose. Its job is to make three decisions in ONE place instead of
 * at every call site: which execution provider to ask for, where the runtime's
 * own assets live, and what happens when a WebGPU session will not start.
 *
 * The onnxruntime bundle is imported at RUNTIME from `public/`, not bundled.
 * That is deliberate: `web/public/ort.webgpu.mjs` and the two
 * `ort-wasm-simd-threaded.asyncify.*` files are byte-identical copies of what
 * npm published, and `vendored.test.ts` asserts it. If the bundler inlined and
 * transformed its own copy instead, that assertion would be about a file the
 * browser never executes. Importing the vendored URL keeps the bytes the test
 * checks and the bytes that run the model the same bytes.
 *
 * That last sentence is only true while `env.wasm.wasmPaths` also points at
 * `public/`. It is a one-line change to send it at a CDN instead, and
 * `vendored.test.ts` would stay green while the browser fetched a completely
 * different wasm binary -- the files in `public/` would be provably pristine
 * and provably unused. So `browserDeps` is exported and
 * `session.test.ts` asserts the resolved paths land beside the vendored entry
 * and on the page's own origin. */

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

/** The slice of the onnxruntime module this file configures and calls. */
export interface OrtModule {
  readonly env: { readonly wasm: { wasmPaths: string; numThreads: number } };
  readonly InferenceSession: {
    create(model: ArrayBuffer, options: Record<string, unknown>): Promise<InferenceSession>;
  };
}

/** The name of the vendored entry point, and so of the directory every other
 * vendored runtime asset is resolved against. */
export const ORT_ENTRY = "ort.webgpu.mjs";

/** Threads to request when the page IS cross-origin isolated. Never reached on
 * GitHub Pages; see `browserDeps`. */
export const ISOLATED_WASM_THREADS = 4;

export interface BrowserDepsOptions {
  /** Defaults to the live `document.baseURI`. */
  readonly baseUri?: string | undefined;
  /** Defaults to a real dynamic `import()`. */
  readonly importOrt?: ((url: string) => Promise<OrtModule>) | undefined;
  /** Defaults to the live `crossOriginIsolated`. */
  readonly isolated?: boolean | undefined;
  /** A token for the model's CONTENT, passed through to the cache so that
   * replacing the graph at the same path is a miss rather than an invisible
   * no-op for returning visitors. */
  readonly contentVersion?: string | undefined;
}

/** Resolve a vendored asset against the page, so the site works from a project
 * subpath on Pages as well as from a domain root. Relative, never rooted: an
 * absolute "/ort.webgpu.mjs" would 404 under a project subpath. */
export function vendoredUrl(name: string, baseUri?: string): string {
  return new URL(name, baseUri ?? document.baseURI).href;
}

/** The directory the runtime loads its wasm loader and binary from.
 *
 * Derived from the entry point rather than written out, so it cannot drift away
 * from the file the bundle is actually imported from. */
export function vendoredDirectory(baseUri?: string): string {
  return new URL("./", vendoredUrl(ORT_ENTRY, baseUri)).href;
}

export function browserDeps(options: BrowserDepsOptions = {}): SessionDeps {
  const importOrt =
    options.importOrt ??
    // @vite-ignore: resolved by the browser at run time and must not be
    // rewritten into a bundled chunk; see the module comment.
    ((url: string) => import(/* @vite-ignore */ url) as Promise<OrtModule>);
  return {
    load: (url, onProgress) =>
      loadCachedBytes(url, {
        ...(onProgress ? { onProgress } : {}),
        ...(options.contentVersion === undefined
          ? {}
          : { version: options.contentVersion }),
      }),
    create: async (model, sessionOptions) => {
      const entry = vendoredUrl(ORT_ENTRY, options.baseUri);
      const ort = await importOrt(entry);
      // The loader glue and the .wasm binary sit beside the entry point -- and
      // must keep doing so, or the vendored-identity test is checking files the
      // browser never fetches.
      ort.env.wasm.wasmPaths = vendoredDirectory(options.baseUri);
      // GitHub Pages cannot send COOP/COEP, so `crossOriginIsolated` is false
      // and SharedArrayBuffer -- and therefore multi-threaded wasm -- is
      // unavailable. Asking for threads anyway makes onnxruntime probe, fail
      // and fall back noisily; asking for one is the honest configuration and
      // is the one the published fallback figure was measured under.
      const isolated = options.isolated ?? globalThis.crossOriginIsolated === true;
      ort.env.wasm.numThreads = isolated ? ISOLATED_WASM_THREADS : 1;
      return ort.InferenceSession.create(model, sessionOptions);
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
