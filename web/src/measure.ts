/** `?measure=1`: the committed measurement harness.
 *
 * Any performance figure this project publishes has to be regenerable from the
 * repository by someone who was not there. A number produced by a scratch page
 * that was then deleted is a memory, not a measurement, so the page that
 * measures is committed and drives the SAME per-frame path the control room
 * runs -- letterbox, inference, decode -- on the same clip, through the same
 * session.
 *
 * The verdict goes into `document.title` for the same reason the selftest's
 * does: `scripts/measure_backend.sh` reads it over the DevTools endpoint
 * without evaluating anything in the page.
 *
 * Every figure it prints carries the renderer string, and a SOFTWARE renderer
 * invalidates the number rather than annotating it. Chrome will happily run
 * this whole harness on SwiftShader -- measured, it produced
 * `ep=wasm ms=123.85 fps=8.05 hardware=false` -- and a line that says MEASURE
 * is a line someone will quote. So the software case reports `MEASURE-FAIL`,
 * which `scripts/verify_page.sh` exits non-zero on, and there is no number on
 * that line to lift out of context.
 *
 * The renderer string is also NOT printed as though it validated the wasm
 * figure. `glRenderer` describes the WebGL/WebGPU path; wasm inference never
 * touches it. Under SwiftShader the renderer changed completely and the wasm
 * median moved about 2%, which is exactly what "this string does not describe
 * this number" looks like. The wasm line therefore says `renderer=n/a (wasm
 * path)` and the detail block says why. */

import {
  DETECT_DEFAULT_CONF,
  DETECT_DEFAULT_NMS_IOU,
} from "./generated/constants";
import { MODEL_CONTENT_VERSION, MODEL_INPUT_SIZE, MODEL_URL } from "./model-asset";
import { probeBackend } from "./runtime/backend";
import type { ExecutionProvider } from "./runtime/backend";
import { decodeYolo } from "./runtime/postprocess";
import { letterbox } from "./runtime/preprocess";
import { ORT_ENTRY, browserDeps, createSession, vendoredUrl } from "./runtime/session";
import { median } from "./ui/format";
import { keepClassesOf, sourceById } from "./ui/sources";

/** Discarded before timing starts: the first inferences of a WebGPU session
 * include shader compilation and pipeline creation, which are a real cost --
 * once -- and would otherwise describe every frame after them. */
const DEFAULT_WARMUP = 12;
const DEFAULT_FRAMES = 120;

function report(line: string, body: string): void {
  document.title = line;
  const root = document.createElement("div");
  root.className = "report";
  const heading = document.createElement("h1");
  heading.textContent = line;
  const pre = document.createElement("pre");
  pre.textContent = body;
  root.append(heading, pre);
  document.body.replaceChildren(root);
}

async function readyVideo(url: string): Promise<HTMLVideoElement> {
  const video = document.createElement("video");
  video.src = new URL(url, document.baseURI).href;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = "auto";
  await new Promise<void>((resolve, reject) => {
    const fail = (): void => {
      reject(new Error(`could not load ${url}`));
    };
    video.addEventListener("loadeddata", () => {
      resolve();
    }, { once: true });
    video.addEventListener("error", fail, { once: true });
  });
  await video.play();
  return video;
}

export async function runMeasurePage(params: URLSearchParams): Promise<void> {
  const frames = Number(params.get("frames") ?? DEFAULT_FRAMES);
  const warmup = Number(params.get("warmup") ?? DEFAULT_WARMUP);
  const source = sourceById(params.get("source") ?? "motorway");
  document.title = "MEASURING";

  try {
    const probe = await probeBackend();
    const requested = (params.get("ep") ?? probe.ep) as ExecutionProvider;
    const session = await createSession(
      new URL(MODEL_URL, document.baseURI).href,
      requested,
      undefined,
      browserDeps({ contentVersion: MODEL_CONTENT_VERSION }),
    );
    const ort = (await import(/* @vite-ignore */ vendoredUrl(ORT_ENTRY))) as {
      Tensor: new (type: string, data: Float32Array, dims: readonly number[]) => unknown;
    };
    const video = await readyVideo(source.url as string);
    const keepClasses = keepClassesOf(source);

    const samples: number[] = [];
    let detections = 0;
    let wallStart = 0;
    for (let i = 0; i < warmup + frames; i += 1) {
      const started = performance.now();
      const input = letterbox(video, MODEL_INPUT_SIZE);
      const output = (await session.session.run({
        [session.inputName]: new ort.Tensor("float32", input.tensor, [
          1,
          3,
          input.size,
          input.size,
        ]),
      })) as Record<string, { data: Float32Array; dims: readonly number[] }>;
      const raw = output[session.outputName] as { data: Float32Array; dims: readonly number[] };
      detections += decodeYolo(
        { data: raw.data, dims: raw.dims },
        input.scale,
        input.padX,
        input.padY,
        { conf: DETECT_DEFAULT_CONF, iou: DETECT_DEFAULT_NMS_IOU, keepClasses },
      ).length;
      const elapsed = performance.now() - started;
      if (i === warmup) {
        wallStart = started;
      }
      if (i >= warmup) {
        samples.push(elapsed);
      }
    }
    const wall = (performance.now() - wallStart) / 1000;
    const ms = median(samples) ?? Number.NaN;
    const fps = samples.length / wall;

    // The wasm figure is not validated by the GL renderer string, so it is not
    // printed as though it were. It still carries the machine's renderer in the
    // detail block, where it can be read as context rather than as evidence.
    const rendererField =
      session.ep === "wasm" ? "n/a (wasm path)" : probe.renderer;
    const detail = [
      `requested ep      ${requested}`,
      `created ep        ${session.ep}`,
      `renderer          ${probe.renderer}`,
      `hardware renderer ${probe.isHardwareRenderer}`,
    ];

    if (!probe.isHardwareRenderer) {
      // Not a number with a caveat: not a measurement. Printed without the
      // figure so there is nothing here to quote.
      report(
        `MEASURE-FAIL software renderer, so this machine cannot produce a ` +
          `hardware timing — ep=${session.ep} renderer=${probe.renderer}`,
        [
          ...detail,
          "",
          "A software rasteriser was detected. The harness ran and produced",
          `a median of ${ms.toFixed(3)} ms over ${samples.length} frames, which is`,
          "deliberately not reported as a MEASURE line: it describes an",
          "emulated device, not hardware anyone runs this page on.",
        ].join("\n"),
      );
      return;
    }

    report(
      `MEASURE ep=${session.ep} ms=${ms.toFixed(2)} fps=${fps.toFixed(2)} ` +
        `n=${samples.length} hardware=${probe.isHardwareRenderer} renderer=${rendererField}`,
      [
        ...detail,
        ...(session.ep === "wasm"
          ? [
              "renderer note     the GL renderer above does NOT validate this",
              "                  number: wasm inference never touches that path.",
            ]
          : []),
        `adapter           ${probe.adapter}`,
        `source            ${source.id} (${source.url ?? "camera"})`,
        `input size        ${MODEL_INPUT_SIZE}`,
        `warmup frames     ${warmup} (discarded)`,
        `timed frames      ${samples.length}`,
        `median ms/frame   ${ms.toFixed(3)}   (letterbox + inference + decode)`,
        `min / max ms      ${Math.min(...samples).toFixed(3)} / ${Math.max(...samples).toFixed(3)}`,
        `throughput fps    ${fps.toFixed(3)}   (timed frames / wall clock)`,
        `detections seen   ${detections}`,
      ].join("\n"),
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    report(`MEASURE-FAIL ${message}`, message);
  }
}
