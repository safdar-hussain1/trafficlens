/** The control room.
 *
 * Everything the previous tasks measured exists so this page can be true, so
 * the rules it is built to are about truth rather than polish:
 *
 *   - Detection is decoupled from render. The page draws every frame and runs
 *     the detector on the cadence the MEASURED backend can sustain, and says
 *     which it is doing rather than quietly dropping to a slideshow.
 *   - Nothing is reported that was not measured in this tab, in this session.
 *     The backend badge carries the renderer string beside every timing.
 *   - The gate is the visitor's. Moving it recomputes the counts from that
 *     moment, and the interface says so instead of showing a total that mixes
 *     two different geometries.
 *   - Where the engine refuses to answer -- a speed on an unsurveyed camera --
 *     the interface refuses in the same words.
 *
 * The frame loop and the detection loop are separate on purpose. The render
 * loop is a `requestAnimationFrame` chain that only ever draws; the detect loop
 * is an async chain that awaits inference and is therefore paced by the
 * hardware. Neither waits on the other. */

import { Gate } from "../engine/gate";
import type { CrossingEvent } from "../engine/gate";
import type { Point } from "../engine/geometry";
import { SessionPipeline } from "../engine/pipeline";
import type { TrackView } from "../engine/pipeline";
import {
  DETECT_DEFAULT_CONF,
  DETECT_DEFAULT_NMS_IOU,
} from "../generated/constants";
import { MODEL_CONTENT_VERSION, MODEL_INPUT_SIZE, MODEL_URL } from "../model-asset";
import { probeBackend } from "../runtime/backend";
import type { BackendProbe } from "../runtime/backend";
import { letterbox } from "../runtime/preprocess";
import { decodeYolo } from "../runtime/postprocess";
import { ORT_ENTRY, browserDeps, createSession, vendoredUrl } from "../runtime/session";
import type { RuntimeSession } from "../runtime/session";
import { drawDiagram } from "./charts";
import type { DiagramTrace } from "./charts";
import {
  applyTheme,
  collectElements,
  currentTheme,
  markSelectedSource,
  renderAlerts,
  renderBadge,
  renderPanels,
  renderSwitcher,
  renderThemeToggle,
} from "./controls";
import type { Elements } from "./controls";
import { RollingMedian, decideCadence, formatClock } from "./format";
import type { Cadence } from "./format";
import { GATE_HANDLE_RADIUS_PX, applyDrag, beginDrag, moveGate } from "./gate-drag";
import type { Grab, GrabKind, Segment } from "./gate-drag";
import { boxToFrame, drawOverlay, frameToBox, readPalette } from "./overlay";
import type { Fit, Palette, Trail } from "./overlay";
import { SOURCES, keepClassesOf, sourceById } from "./sources";
import type { SourceSpec } from "./sources";
import { signedDistanceToGate, withinGateSpan } from "./timespace";

/** Seconds of history the diagram shows. Long enough for a vehicle to cross
 * the frame at motorway speed, short enough that the lines stay separable. */
const DIAGRAM_WINDOW_S = 12;

/** Trajectory history kept per track, in seconds. Slightly longer than the
 * diagram window so a line entering from the left edge is already drawn. */
const HISTORY_S = DIAGRAM_WINDOW_S + 2;

/** Samples the rolling backend median is taken over: about four seconds of
 * WebGPU inference, so the figure settles quickly and still forgets a stall. */
const TIMING_WINDOW = 120;

/** The diagram never zooms in past this, in image pixels either side. */
const DIAGRAM_FLOOR_SPAN_PX = 150;

interface Sample {
  readonly t: number;
  readonly p: Point;
}

export class ControlRoom {
  private readonly elements: Elements;
  private readonly video: HTMLVideoElement;
  private readonly videoCtx: CanvasRenderingContext2D;
  private readonly chartCtx: CanvasRenderingContext2D;
  private readonly reducedMotion: boolean;

  private source: SourceSpec = SOURCES[0] as SourceSpec;
  private frameSize = { width: 0, height: 0 };
  private gate: Segment = { start: [0, 0], end: [1, 1] };
  private pipeline: SessionPipeline | null = null;
  private probe: BackendProbe | null = null;
  private session: RuntimeSession | null = null;
  private tensorFactory:
    | ((data: Float32Array, dims: readonly number[]) => unknown)
    | null = null;

  private running = false;
  private detectGeneration = 0;
  private fit: Fit = { scale: 1, dx: 0, dy: 0 };
  private grab: Grab | null = null;

  private readonly trails = new Map<number, Sample[]>();
  private tracks: readonly TrackView[] = [];
  private events: CrossingEvent[] = [];
  private wrongWay: string[] = [];
  private readonly wrongWayIds = new Set<number>();

  private readonly frameMs = new RollingMedian(TIMING_WINDOW);
  private detectionTimes: number[] = [];
  private cadence: Cadence = decideCadence(null, 30);
  private status = "";
  private lastTimestamp = 0;
  private palette: Palette | null = null;
  private paletteKey = "";

  constructor() {
    this.elements = collectElements();
    this.reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.video = document.createElement("video");
    this.video.playsInline = true;
    this.video.muted = true;
    this.video.loop = true;
    this.video.preload = "auto";
    this.video.crossOrigin = "anonymous";

    const videoCtx = this.elements.videoCanvas.getContext("2d");
    const chartCtx = this.elements.chartCanvas.getContext("2d");
    if (videoCtx === null || chartCtx === null) {
      throw new Error("this browser has no 2d canvas context");
    }
    this.videoCtx = videoCtx;
    this.chartCtx = chartCtx;
  }

  /** The live video element, exposed for the headless webcam check: the stub
   * replaces `getUserMedia`, and the verifier needs to see the same element the
   * detector reads. */
  get videoElement(): HTMLVideoElement {
    return this.video;
  }

  async start(): Promise<void> {
    renderThemeToggle(this.elements.themeToggle);
    this.elements.themeToggle.addEventListener("click", () => {
      applyTheme(currentTheme() === "dark" ? "light" : "dark");
      renderThemeToggle(this.elements.themeToggle);
    });

    renderSwitcher(this.elements.switcher, this.source.id, (id) => {
      void this.selectSource(sourceById(id));
    });
    this.elements.startButton.addEventListener("click", () => {
      void this.toggleRunning();
    });
    this.elements.resetButton.addEventListener("click", () => {
      this.resetCounts();
    });
    this.installGateControls();

    renderBadge(this.elements.badge, {
      probe: null,
      ep: null,
      msPerFrame: null,
      fps: null,
      cadence: null,
    });

    // Probed before anything is downloaded: the page can say what this machine
    // will run before it asks the visitor to pay 10.7 MB to find out.
    this.probe = await probeBackend();
    this.renderBadge();

    await this.selectSource(this.source);
    this.renderFrame();
  }

  // -- sources ----------------------------------------------------------------

  private async selectSource(source: SourceSpec): Promise<void> {
    const wasRunning = this.running;
    this.stop();
    this.source = source;
    markSelectedSource(this.elements.switcher, source.id);
    this.elements.videoCaption.textContent = source.caption;
    this.clearSession();

    try {
      await this.attachSource(source);
    } catch (error) {
      this.setStatus(
        source.kind === "camera"
          ? `The camera could not be opened: ${describe(error)}. The clips below still run.`
          : `That clip could not be loaded: ${describe(error)}.`,
      );
      return;
    }

    this.frameSize = { width: this.video.videoWidth, height: this.video.videoHeight };
    this.gate = gateSegment(source, this.frameSize);
    this.pipeline = this.buildPipeline();
    this.setStatus("");
    this.renderPanels();
    if (wasRunning) {
      await this.run();
    }
    this.renderFrame();
  }

  private async attachSource(source: SourceSpec): Promise<void> {
    if (source.kind === "camera") {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      });
      this.video.srcObject = stream;
      this.video.removeAttribute("src");
    } else {
      this.releaseCamera();
      this.video.srcObject = null;
      this.video.src = new URL(source.url as string, document.baseURI).href;
      this.video.load();
    }
    await this.awaitMetadata();
    if (source.kind === "camera") {
      // A live stream has nothing to seek to and no first frame until it plays.
      await this.video.play();
    } else {
      this.video.currentTime = 0;
      await once(this.video, "seeked", 4000).catch(() => undefined);
    }
  }

  private awaitMetadata(): Promise<void> {
    if (this.video.readyState >= 1 && this.video.videoWidth > 0) {
      return Promise.resolve();
    }
    return once(this.video, "loadedmetadata", 15000).then(() => undefined);
  }

  private buildPipeline(): SessionPipeline {
    return new SessionPipeline({
      gates: [this.gateObject()],
      // No source reachable from this page has an independent along-road
      // survey, so there is no plane and every speed is null -- by refusal,
      // not by absence of data.
      plane: null,
      fps: this.source.fps,
      speedLimitKmh: null,
    });
  }

  private gateObject(): Gate {
    return new Gate(this.source.gate.name, this.gate.start, this.gate.end, {
      labelPositive: this.source.gate.labelPositive,
      labelNegative: this.source.gate.labelNegative,
      expectedDirection: this.source.gate.expectedDirection,
    });
  }

  // -- running ----------------------------------------------------------------

  private async toggleRunning(): Promise<void> {
    if (this.running) {
      this.stop();
      this.renderFrame();
      return;
    }
    await this.run();
  }

  private async run(): Promise<void> {
    if (this.pipeline === null) {
      return;
    }
    this.elements.startButton.disabled = true;
    try {
      await this.ensureSession();
    } catch (error) {
      this.setStatus(`The detector could not start: ${describe(error)}`);
      this.elements.startButton.disabled = false;
      return;
    }
    this.elements.startButton.disabled = false;
    this.elements.startButton.textContent = "Stop";
    this.elements.emptyState.hidden = true;
    this.elements.videoNote.textContent = "running";
    this.running = true;
    this.detectGeneration += 1;
    try {
      await this.video.play();
    } catch {
      // Autoplay refusal on a muted, user-initiated play is not expected; if it
      // happens the detect loop simply waits for frames.
    }
    void this.detectLoop(this.detectGeneration);
  }

  private stop(): void {
    this.running = false;
    this.grab = null;
    this.detectGeneration += 1;
    this.video.pause();
    this.elements.startButton.textContent = "Start";
    this.elements.videoNote.textContent = "stopped";
  }

  private clearSession(): void {
    this.trails.clear();
    this.tracks = [];
    this.events = [];
    this.wrongWay = [];
    this.wrongWayIds.clear();
    this.frameMs.reset();
    this.detectionTimes = [];
    this.lastTimestamp = 0;
  }

  private resetCounts(): void {
    this.pipeline = this.buildPipeline();
    this.clearSession();
    this.renderPanels();
    this.renderFrame();
  }

  private async ensureSession(): Promise<void> {
    if (this.session !== null) {
      return;
    }
    const probe = this.probe;
    this.elements.progress.hidden = false;
    this.elements.emptyText.textContent = "Downloading the detector — this happens once.";

    const session = await createSession(
      new URL(MODEL_URL, document.baseURI).href,
      probe?.ep ?? "wasm",
      (progress) => {
        if (progress.total > 0) {
          this.elements.progress.max = progress.total;
          this.elements.progress.value = progress.loaded;
        } else {
          this.elements.progress.removeAttribute("value");
        }
        this.elements.emptyText.textContent = progress.fromCache
          ? "Detector loaded from this browser's cache."
          : `Downloading the detector — ${(progress.loaded / 1e6).toFixed(1)} MB.`;
      },
      browserDeps({ contentVersion: MODEL_CONTENT_VERSION }),
    );
    this.session = session;

    // The runtime module is imported from the same vendored URL the session
    // used, so this is the same module instance the session is running on --
    // the browser caches it -- and its Tensor is the one that session accepts.
    const ort = (await import(/* @vite-ignore */ vendoredUrl(ORT_ENTRY))) as {
      Tensor: new (type: string, data: Float32Array, dims: readonly number[]) => unknown;
    };
    this.tensorFactory = (data, dims) => new ort.Tensor("float32", data, dims);
    this.elements.progress.hidden = true;
    this.renderBadge();
  }

  // -- the detect loop --------------------------------------------------------

  private async detectLoop(generation: number): Promise<void> {
    const session = this.session;
    const makeTensor = this.tensorFactory;
    if (session === null || makeTensor === null) {
      return;
    }
    const keepClasses = keepClassesOf(this.source);
    let frameIndex = 0;
    let lastDetectedAt = -Infinity;

    while (this.running && generation === this.detectGeneration) {
      const now = this.video.currentTime;
      const spacing = this.cadence.stride / this.source.fps;
      if (this.video.readyState < 2 || this.video.videoWidth === 0) {
        await nextFrame();
        continue;
      }
      if (now < lastDetectedAt + spacing && now >= lastDetectedAt) {
        await nextFrame();
        continue;
      }
      lastDetectedAt = now;

      try {
        // The whole per-frame path is timed, not just the inference:
        // letterboxing and decoding are costs the visitor pays too, and a
        // number that left them out would not be the frame time the page is
        // actually achieving.
        const started = performance.now();
        const input = letterbox(this.video, MODEL_INPUT_SIZE);
        const output = (await session.session.run({
          [session.inputName]: makeTensor(input.tensor, [1, 3, input.size, input.size]),
        })) as Record<string, { data: Float32Array; dims: readonly number[] }>;
        const raw = output[session.outputName] as {
          data: Float32Array;
          dims: readonly number[];
        };
        const detections = decodeYolo(
          { data: raw.data, dims: raw.dims },
          input.scale,
          input.padX,
          input.padY,
          { conf: DETECT_DEFAULT_CONF, iou: DETECT_DEFAULT_NMS_IOU, keepClasses },
        );
        this.recordTiming(performance.now() - started);
        this.consume(detections, frameIndex, now);
        frameIndex += 1;
      } catch (error) {
        this.setStatus(`Inference stopped: ${describe(error)}`);
        this.stop();
        return;
      }
    }
  }

  private recordTiming(elapsedMs: number): void {
    this.frameMs.push(elapsedMs);
    const stamp = performance.now();
    this.detectionTimes.push(stamp);
    while (
      this.detectionTimes.length > 2 &&
      stamp - (this.detectionTimes[0] as number) > 2000
    ) {
      this.detectionTimes.shift();
    }
    this.cadence = decideCadence(this.frameMs.value(), this.source.fps);
    this.renderBadge();
  }

  /** Measured detections per second, over the last two seconds of wall clock.
   * Not derived from the median: a derived rate would hide the cost of
   * everything around inference, which the visitor is also paying. */
  private measuredFps(): number | null {
    if (this.detectionTimes.length < 2) {
      return null;
    }
    const first = this.detectionTimes[0] as number;
    const last = this.detectionTimes[this.detectionTimes.length - 1] as number;
    const seconds = (last - first) / 1000;
    return seconds > 0 ? (this.detectionTimes.length - 1) / seconds : null;
  }

  private consume(
    detections: Parameters<SessionPipeline["step"]>[0],
    frameIndex: number,
    timestamp: number,
  ): void {
    const pipeline = this.pipeline;
    if (pipeline === null) {
      return;
    }
    // The clips loop, so the clip clock jumps backwards. Counts carry across --
    // those vehicles really did cross -- but the drawn history cannot: the
    // diagram's axis IS clip time, and trajectories from before the loop would
    // sit in the future of the axis and never age out.
    if (timestamp < this.lastTimestamp - 0.5) {
      this.trails.clear();
      this.events = [];
      this.wrongWayIds.clear();
    }
    this.lastTimestamp = timestamp;

    const step = pipeline.step(detections, frameIndex, timestamp);
    this.tracks = step.tracks;

    for (const track of step.tracks) {
      const trail = this.trails.get(track.trackId) ?? [];
      trail.push({ t: timestamp, p: track.anchor });
      while (trail.length > 1 && timestamp - (trail[0] as Sample).t > HISTORY_S) {
        trail.shift();
      }
      this.trails.set(track.trackId, trail);
    }
    for (const [trackId, trail] of this.trails) {
      const last = trail[trail.length - 1];
      if (last === undefined || timestamp - last.t > HISTORY_S) {
        this.trails.delete(trackId);
      }
    }

    const expected = this.source.gate.expectedDirection;
    for (const event of step.events) {
      this.events.push(event);
      if (expected !== null && event.direction !== expected) {
        this.wrongWayIds.add(event.trackId);
        this.wrongWay.unshift(
          `${formatClock(event.timestamp)}  ${event.className} ${event.trackId} went ${event.direction}`,
        );
        this.wrongWay = this.wrongWay.slice(0, 6);
      }
    }
    this.events = this.events.filter((event) => timestamp - event.timestamp <= HISTORY_S);
    this.renderPanels();
  }

  // -- the render loop --------------------------------------------------------

  /** Draw one frame. Runs whether or not the detector is going: the page has to
   * be legible with the engine stopped. */
  renderFrame(): void {
    const palette = this.themePalette();
    const dpr = Math.min(3, globalThis.devicePixelRatio || 1);
    const now = this.video.currentTime;

    const videoBox = sizeCanvas(this.elements.videoCanvas, dpr);
    this.fit = drawOverlay(
      this.videoCtx,
      videoBox,
      {
        frame: this.frameSize,
        gate: this.gate,
        gateLabels: {
          positive: this.source.gate.labelPositive,
          negative: this.source.gate.labelNegative,
        },
        tracks: this.tracks,
        trails: this.trailsForOverlay(),
        events: this.events,
        now,
        dpr,
        source: this.video.readyState >= 2 ? this.video : null,
        reducedMotion: this.reducedMotion,
        wrongWay: this.wrongWayIds,
      },
      palette,
    );
    const chartBox = sizeCanvas(this.elements.chartCanvas, dpr);
    drawDiagram(
      this.chartCtx,
      chartBox,
      {
        now,
        windowS: DIAGRAM_WINDOW_S,
        traces: this.diagramTraces(),
        events: this.events,
        floorSpanPx: DIAGRAM_FLOOR_SPAN_PX,
        dpr,
        running: this.running,
      },
      palette,
    );
    // Last, after both canvases have been measured: these writes invalidate
    // layout, and doing them earlier would force a synchronous recalculation on
    // the next `getBoundingClientRect` every single frame.
    this.positionHandles();
  }

  /** The theme's colours, recomputed only when the theme actually changes.
   * `getComputedStyle` is a layout read, and doing nine of them per frame beside
   * the handle writes below is the classic way to make a canvas page stutter. */
  private themePalette(): Palette {
    const key = `${document.documentElement.getAttribute("data-theme") ?? "system"}:${
      matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
    }`;
    if (this.palette === null || key !== this.paletteKey) {
      this.palette = readPalette(document.documentElement);
      this.paletteKey = key;
    }
    return this.palette;
  }

  private trailsForOverlay(): Map<number, Trail> {
    const out = new Map<number, Trail>();
    for (const [trackId, samples] of this.trails) {
      out.set(trackId, samples);
    }
    return out;
  }

  /** Trajectories in diagram space, recomputed from stored image positions
   * every frame.
   *
   * Recomputed rather than accumulated because the vertical axis is distance
   * from THIS gate: when the visitor moves the gate, every trajectory already
   * drawn has to move with it, or the two views would be showing different
   * geometry and the page's claim that they are one truth would be false. */
  private diagramTraces(): DiagramTrace[] {
    const traces: DiagramTrace[] = [];
    for (const [trackId, samples] of this.trails) {
      traces.push({
        trackId,
        samples: samples.map((sample) => ({
          t: sample.t,
          d: signedDistanceToGate(this.gate, sample.p),
        })),
        inSpan: samples.map((sample) => withinGateSpan(this.gate, sample.p)),
      });
    }
    return traces;
  }

  // -- the gate ---------------------------------------------------------------

  private installGateControls(): void {
    const stage = this.elements.stage;
    stage.addEventListener("pointerdown", (event) => {
      const point = this.pointerToFrame(event);
      const grab = beginDrag(this.gate, point, this.grabRadius());
      if (grab === null) {
        return;
      }
      this.grab = grab;
      stage.setPointerCapture(event.pointerId);
      event.preventDefault();
    });
    stage.addEventListener("pointermove", (event) => {
      if (this.grab === null) {
        return;
      }
      this.setGate(applyDrag(this.grab, this.pointerToFrame(event), this.frameSize));
      event.preventDefault();
    });
    for (const type of ["pointerup", "pointercancel"] as const) {
      stage.addEventListener(type, (event) => {
        if (this.grab === null) {
          return;
        }
        this.grab = null;
        stage.releasePointerCapture(event.pointerId);
      });
    }

    const kinds: [HTMLButtonElement, GrabKind][] = [
      [this.elements.handles.start, "start"],
      [this.elements.handles.body, "body"],
      [this.elements.handles.end, "end"],
    ];
    for (const [button, kind] of kinds) {
      button.addEventListener("keydown", (event) => {
        const step = event.shiftKey ? 16 : 4;
        const deltas: Record<string, [number, number]> = {
          ArrowLeft: [-step, 0],
          ArrowRight: [step, 0],
          ArrowUp: [0, -step],
          ArrowDown: [0, step],
        };
        const delta = deltas[event.key];
        if (delta === undefined) {
          return;
        }
        event.preventDefault();
        this.setGate(moveGate(this.gate, kind, delta[0], delta[1], this.frameSize));
      });
    }
  }

  /** The grab radius in FRAME pixels: a constant on-screen radius, converted,
   * so the handle feels the same size on a phone and on a 4K display. */
  private grabRadius(): number {
    return GATE_HANDLE_RADIUS_PX / (this.fit.scale || 1);
  }

  private pointerToFrame(event: PointerEvent): Point {
    const rect = this.elements.videoCanvas.getBoundingClientRect();
    return boxToFrame([event.clientX - rect.left, event.clientY - rect.top], this.fit);
  }

  private setGate(next: Segment): void {
    this.gate = next;
    if (this.pipeline !== null) {
      this.pipeline.replaceGates([this.gateObject()], this.video.currentTime);
      this.wrongWay = [];
      this.wrongWayIds.clear();
      this.events = [];
    }
    this.renderPanels();
    this.renderFrame();
  }

  private positionHandles(): void {
    const points: [HTMLButtonElement, Point][] = [
      [this.elements.handles.start, this.gate.start],
      [this.elements.handles.end, this.gate.end],
      [
        this.elements.handles.body,
        [
          (this.gate.start[0] + this.gate.end[0]) / 2,
          (this.gate.start[1] + this.gate.end[1]) / 2,
        ],
      ],
    ];
    for (const [button, point] of points) {
      const [x, y] = frameToBox(point, this.fit);
      button.style.left = `${x}px`;
      button.style.top = `${y}px`;
    }
  }

  // -- rendering the markup ---------------------------------------------------

  private renderBadge(): void {
    renderBadge(this.elements.badge, {
      probe: this.probe,
      ep: this.session?.ep ?? null,
      msPerFrame: this.frameMs.value(),
      fps: this.measuredFps(),
      cadence: this.frameMs.count > 0 ? this.cadence : null,
    });
  }

  private renderPanels(): void {
    const pipeline = this.pipeline;
    const counts = pipeline?.counts() ?? {};
    const gateCounts = counts[this.source.gate.name] ?? {};

    const perClass = this.source.classes.map(
      ([, name]) =>
        [name, Object.values(gateCounts[name] ?? {}).reduce((a, b) => a + b, 0)] as const,
    );
    const directions = [this.source.gate.labelPositive, this.source.gate.labelNegative];
    const perDirection = directions.map((direction) => {
      let total = 0;
      for (const byDirection of Object.values(gateCounts)) {
        total += byDirection[direction] ?? 0;
      }
      return [direction, total] as const;
    });

    renderPanels(this.elements, {
      source: this.source,
      total: pipeline?.total() ?? 0,
      perClass,
      perDirection,
      countingSince: pipeline?.countingSinceTimestamp ?? null,
      wrongWay: this.wrongWay,
    });
    if (this.wrongWay.length > 0) {
      renderAlerts(this.elements, this.wrongWay);
    }
  }

  private setStatus(message: string): void {
    this.status = message;
    this.elements.statusLine.textContent = message;
  }

  /** The last status message, for the headless checks. */
  get statusMessage(): string {
    return this.status;
  }

  private releaseCamera(): void {
    const stream = this.video.srcObject as MediaStream | null;
    if (stream !== null && typeof stream.getTracks === "function") {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    }
  }
}

// -- helpers ------------------------------------------------------------------

function gateSegment(source: SourceSpec, frame: { width: number; height: number }): Segment {
  return {
    start: [source.gate.start[0] * frame.width, source.gate.start[1] * frame.height],
    end: [source.gate.end[0] * frame.width, source.gate.end[1] * frame.height],
  };
}

function sizeCanvas(canvas: HTMLCanvasElement, dpr: number): { width: number; height: number } {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  return { width, height };
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      resolve();
    });
  });
}

function once(target: EventTarget, type: string, timeoutMs: number): Promise<Event> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      target.removeEventListener(type, handler);
      reject(new Error(`timed out waiting for ${type}`));
    }, timeoutMs);
    const handler = (event: Event): void => {
      clearTimeout(timer);
      target.removeEventListener(type, handler);
      resolve(event);
    };
    target.addEventListener(type, handler);
  });
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Boot the control room and keep drawing. */
export async function mountControlRoom(): Promise<ControlRoom> {
  const room = new ControlRoom();
  await room.start();
  const draw = (): void => {
    room.renderFrame();
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
  return room;
}
