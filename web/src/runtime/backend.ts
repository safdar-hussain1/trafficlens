/** Which engine runs the detector, and on what hardware.
 *
 * The product's central claim is that the visitor's own GPU runs the model, so
 * the page has to be able to say -- truthfully, per visitor -- whether it did.
 * This probe answers that before any 10.7 MB download starts, and it is
 * deliberately incapable of optimism: WebGPU is claimed only once an adapter
 * has actually been granted, because `navigator.gpu` exists in browsers that
 * then refuse an adapter (no hardware, a blocklisted driver, a crashed GPU
 * process), and a page that had announced WebGPU by then would be wrong. */

/** Substrings that mark a renderer as a CPU rasteriser pretending to be a GPU.
 *
 * Notes §3 of this task's plan is binding: every published hardware timing
 * carries its `glRenderer` string, and a software-renderer string invalidates
 * the number. The probe cannot refuse to run on such a machine -- the demo
 * should still work -- so instead it labels the result, and anything that
 * publishes a timing is expected to check `isHardwareRenderer` first. */
export const SOFTWARE_RENDERER_MARKERS = [
  "swiftshader",
  "llvmpipe",
  "software rasterizer",
  "microsoft basic render",
  "software adapter",
] as const;

export type ExecutionProvider = "webgpu" | "wasm";

export interface BackendProbe {
  /** The execution provider onnxruntime should be asked for. */
  readonly ep: ExecutionProvider;
  /** The unmasked WebGL renderer string, or `"unknown"` if it cannot be read.
   * WebGL rather than WebGPU because WebGPU's own `adapter.info` is empty on
   * most configurations -- including headless Chrome, where it was measured --
   * while `WEBGL_debug_renderer_info` names the actual device. */
  readonly renderer: string;
  /** Whatever the WebGPU adapter would say about itself, or the reason there
   * is no adapter. Diagnostic, never parsed. */
  readonly adapter: string;
  /** False when `renderer` names a CPU rasteriser; see the markers above. */
  readonly isHardwareRenderer: boolean;
}

interface AdapterLike {
  readonly info?: Record<string, unknown> | undefined;
}

export interface ProbeEnvironment {
  readonly gpu?: { requestAdapter(): Promise<AdapterLike | null> } | undefined;
  /** Reads the unmasked renderer string, or throws if it cannot. */
  readonly readRenderer: () => string;
}

function readRendererFromWebGl(): string {
  const canvas = document.createElement("canvas");
  const gl = (canvas.getContext("webgl2") ??
    canvas.getContext("webgl")) as WebGLRenderingContext | null;
  if (gl === null) {
    throw new Error("no WebGL context available");
  }
  const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
  const parameter =
    debugInfo === null
      ? gl.RENDERER
      : (debugInfo as { UNMASKED_RENDERER_WEBGL: number }).UNMASKED_RENDERER_WEBGL;
  return String(gl.getParameter(parameter));
}

function browserEnvironment(): ProbeEnvironment {
  const gpu = (navigator as Navigator & { gpu?: ProbeEnvironment["gpu"] }).gpu;
  return {
    ...(gpu === undefined ? {} : { gpu }),
    readRenderer: readRendererFromWebGl,
  };
}

export function isHardwareRenderer(renderer: string): boolean {
  const lowered = renderer.toLowerCase();
  return !SOFTWARE_RENDERER_MARKERS.some((marker) => lowered.includes(marker));
}

/** Decide the execution provider and record what the machine is.
 *
 * Never throws: a probe that failed would leave the page with nothing to say,
 * and "wasm on an unknown renderer" is both a usable answer and an honest one. */
export async function probeBackend(
  environment: ProbeEnvironment = browserEnvironment(),
): Promise<BackendProbe> {
  let renderer: string;
  try {
    renderer = environment.readRenderer();
  } catch {
    renderer = "unknown";
  }

  if (environment.gpu === undefined) {
    return {
      ep: "wasm",
      renderer,
      adapter: "no WebGPU in this browser",
      isHardwareRenderer: isHardwareRenderer(renderer),
    };
  }

  try {
    const adapter = await environment.gpu.requestAdapter();
    if (adapter === null || adapter === undefined) {
      return {
        ep: "wasm",
        renderer,
        adapter: "WebGPU present but no adapter granted",
        isHardwareRenderer: isHardwareRenderer(renderer),
      };
    }
    return {
      ep: "webgpu",
      renderer,
      // `info` is `{}` on many configurations; kept verbatim rather than
      // dressed up, so a reader can tell "empty" from "not asked".
      adapter: JSON.stringify(adapter.info ?? {}),
      isHardwareRenderer: isHardwareRenderer(renderer),
    };
  } catch (error) {
    return {
      ep: "wasm",
      renderer,
      adapter: `adapter request failed: ${String(
        error instanceof Error ? error.message : error,
      )}`,
      isHardwareRenderer: isHardwareRenderer(renderer),
    };
  }
}
