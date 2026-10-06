/** The parts of the demo that are markup rather than canvas: the status pill,
 * the source tabs, the theme and the element lookup.
 *
 * Every writer here takes what was actually measured and prints it, or prints
 * nothing. There is no default number anywhere in this file. */

import type { BackendProbe } from "../runtime/backend";
import { formatFps, formatMs } from "./format";
import type { Cadence } from "./format";
import type { ReadoutElements } from "./readout";
import { SOURCES } from "./sources";

export const THEME_STORAGE_KEY = "trafficlens-theme";

export type Theme = "light" | "dark";

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) {
    throw new Error(`missing element #${id}`);
  }
  return found as T;
}

export interface Elements extends ReadoutElements {
  readonly badge: HTMLElement;
  readonly runState: HTMLElement;
  readonly perf: HTMLElement;
  readonly themeToggle: HTMLButtonElement;
  readonly videoCanvas: HTMLCanvasElement;
  readonly stage: HTMLElement;
  readonly handles: {
    readonly start: HTMLButtonElement;
    readonly body: HTMLButtonElement;
    readonly end: HTMLButtonElement;
  };
  readonly gateHint: HTMLElement;
  readonly emptyState: HTMLElement;
  readonly launchButton: HTMLButtonElement;
  readonly emptyText: HTMLElement;
  readonly progress: HTMLProgressElement;
  readonly videoCaption: HTMLElement;
  readonly switcher: HTMLElement;
  readonly startButton: HTMLButtonElement;
  readonly resetButton: HTMLButtonElement;
  readonly statusLine: HTMLElement;
}

export function collectElements(): Elements {
  return {
    badge: element("backend-badge"),
    runState: element("run-state"),
    perf: element("perf"),
    themeToggle: element<HTMLButtonElement>("theme-toggle"),
    videoCanvas: element<HTMLCanvasElement>("video-canvas"),
    stage: element("stage-video"),
    handles: {
      start: element<HTMLButtonElement>("handle-start"),
      body: element<HTMLButtonElement>("handle-body"),
      end: element<HTMLButtonElement>("handle-end"),
    },
    gateHint: element("gate-hint"),
    emptyState: element("empty-state"),
    launchButton: element<HTMLButtonElement>("launch-button"),
    emptyText: element("empty-text"),
    progress: element<HTMLProgressElement>("load-progress"),
    videoCaption: element("video-caption"),
    vms: element("vms"),
    totalCount: element("total-count"),
    countingSince: element("counting-since"),
    directionCounts: element("direction-counts"),
    classCounts: element("class-counts"),
    feed: element("feed"),
    switcher: element("source-switcher"),
    startButton: element<HTMLButtonElement>("start-button"),
    resetButton: element<HTMLButtonElement>("reset-button"),
    statusLine: element("status-line"),
  };
}

// -- theme --------------------------------------------------------------------

export function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(THEME_STORAGE_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

export function systemTheme(): Theme {
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Apply a theme, writing it to storage FIRST.
 *
 * The order is the whole point: if the document were stamped first and the
 * write then failed -- or the page reloaded between the two -- the visitor's
 * choice would be lost on the next load, which is the one moment it matters. */
export function applyTheme(theme: Theme): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Storage unavailable: the choice holds for this page view only, which is
    // still better than refusing to change it.
  }
  document.documentElement.setAttribute("data-theme", theme);
}

export function currentTheme(): Theme {
  const attribute = document.documentElement.getAttribute("data-theme");
  return attribute === "light" || attribute === "dark" ? attribute : systemTheme();
}

/** The toggle names where it goes, and its icon shows it: a moon while the page
 * is light, a sun while it is dark. */
export function renderThemeToggle(button: HTMLButtonElement): void {
  const theme = currentTheme();
  const next = theme === "dark" ? "light" : "dark";
  button.dataset["theme"] = theme;
  button.setAttribute("aria-label", `Switch to the ${next} theme`);
  button.title = `Switch to the ${next} theme`;
}

/** Wire a theme toggle. Shared by both pages. */
export function installThemeToggle(button: HTMLButtonElement): void {
  renderThemeToggle(button);
  button.addEventListener("click", () => {
    applyTheme(currentTheme() === "dark" ? "light" : "dark");
    renderThemeToggle(button);
  });
}

// -- status -------------------------------------------------------------------

export interface BadgeState {
  readonly probe: BackendProbe | null;
  /** The provider the session was actually created with, once there is one. */
  readonly ep: string | null;
  readonly msPerFrame: number | null;
  readonly fps: number | null;
  readonly cadence: Cadence | null;
}

/** What the GL renderer string describes on the WASM path: nothing.
 *
 * `BackendProbe.renderer` is read from `WEBGL_debug_renderer_info` and names
 * the device the WebGL/WebGPU path would use. WASM inference never touches it.
 * Printing it beside `WASM` and a ms/frame figure implies a hardware
 * attribution that was never performed -- measured, and corrected in the
 * numbers file as C12: a run forced onto a software renderer moved the WASM
 * figure by about 2 % while the renderer string changed completely. So on the
 * WASM path the status says so, rather than naming a GPU that is not running
 * anything. */
const RENDERER_NOT_APPLICABLE = "n/a (wasm path)";

/** What the status pill says, decided before any of it is turned into
 * elements.
 *
 * Separated from `renderStatus` because the decision is what can be wrong: the
 * renderer attribution above is a claim about a measurement, and a claim needs
 * a test. The suite runs without a DOM, so the DOM half stays a thin mapping
 * over this. */
export interface BadgeContent {
  readonly ep: "WebGPU" | "WASM";
  /** Empty while the probe is still running. */
  readonly renderer: string;
  /** The per-frame time, or null before one has been measured. */
  readonly msPerFrame: string | null;
  /** Measured detections per second, or null before there are any. */
  readonly fps: string | null;
  /** Said only when the detector is NOT running on every frame: that is the
   * case a visitor would otherwise mistake for a broken demo. */
  readonly cadence: string | null;
  /** Whether the software-renderer caveat applies to the number beside it. */
  readonly softwareWarning: boolean;
  readonly probed: boolean;
}

export function badgeContent(state: BadgeState): BadgeContent {
  const { probe } = state;
  if (probe === null) {
    return {
      ep: "WASM",
      renderer: "",
      msPerFrame: null,
      fps: null,
      cadence: null,
      softwareWarning: false,
      probed: false,
    };
  }
  const onGpu = (state.ep ?? probe.ep) === "webgpu";
  return {
    ep: onGpu ? "WebGPU" : "WASM",
    renderer: onGpu ? probe.renderer : RENDERER_NOT_APPLICABLE,
    msPerFrame: state.msPerFrame === null ? null : `${formatMs(state.msPerFrame)} ms/frame`,
    fps: state.fps === null ? null : `${formatFps(state.fps)} fps`,
    cadence:
      state.cadence === null || state.cadence.stride === 1
        ? null
        : `detecting ${state.cadence.label}`,
    // Only on the path the renderer can affect. Beside a WASM figure the
    // caveat would be describing a device that ran none of it.
    softwareWarning: onGpu && !probe.isHardwareRenderer,
    probed: true,
  };
}

export type RunState = "checking" | "ready" | "loading" | "live" | "paused";

const RUN_TEXT: Record<RunState, string> = {
  checking: "Checking this device…",
  ready: "Ready",
  loading: "Loading the detector",
  live: "Live",
  paused: "Paused",
};

/** The status pill: a traffic-light dot, the state in words, then what this
 * machine is running and what it measured. The run state is the live region;
 * the figures are not, so a screen reader hears "Live" once rather than a frame
 * rate thirty times a second. */
export function renderStatus(
  elements: Pick<Elements, "badge" | "runState" | "perf">,
  run: RunState,
  state: BadgeState,
): void {
  const content = badgeContent(state);
  elements.badge.dataset["state"] = run;
  elements.badge.dataset["software"] = String(content.softwareWarning);
  if (elements.runState.textContent !== RUN_TEXT[run]) {
    elements.runState.textContent = RUN_TEXT[run];
  }
  if (!content.probed) {
    elements.perf.replaceChildren();
    return;
  }

  const engine = document.createElement("b");
  engine.textContent = content.ep === "WebGPU" ? "WebGPU" : "WebAssembly";
  const parts: HTMLElement[] = [engine];
  const showFigures = run === "live" || run === "paused";
  if (showFigures && content.fps !== null) {
    parts.push(span(content.fps));
  }
  if (showFigures && content.cadence !== null) {
    parts.push(span(content.cadence));
  }
  if (content.softwareWarning) {
    const warn = span("software renderer");
    warn.className = "status__warn";
    warn.title = "Not a hardware timing";
    parts.push(warn);
  }
  elements.perf.replaceChildren(...parts);
  // The detail a visitor only wants when they ask for it.
  elements.perf.title = [
    content.renderer === "" ? null : `Renderer: ${content.renderer}`,
    showFigures ? content.msPerFrame : null,
  ]
    .filter((part): part is string => part !== null)
    .join("\n");
}

function span(text: string): HTMLElement {
  const node = document.createElement("span");
  node.textContent = text;
  return node;
}

// -- source tabs ----------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";

/** One small line icon per source, drawn as paths rather than an icon font. */
const SOURCE_ICONS: Readonly<Record<string, readonly string[]>> = {
  motorway: ["M6.5 3 3 17", "M13.5 3 17 17", "M10 3.5v2.4", "M10 8.8v2.4", "M10 14.1v2.4"],
  street: ["M3 6.5h14", "M3 13.5h14", "M5 8.5v3", "M8.3 8.5v3", "M11.7 8.5v3", "M15 8.5v3"],
  webcam: ["M3.2 6.5h9.6a1.2 1.2 0 0 1 1.2 1.2v4.6a1.2 1.2 0 0 1-1.2 1.2H3.2A1.2 1.2 0 0 1 2 12.3V7.7a1.2 1.2 0 0 1 1.2-1.2Z", "M14 9.2 18 7v6l-4-2.2"],
};

function sourceIcon(id: string): SVGSVGElement | null {
  const paths = SOURCE_ICONS[id];
  if (paths === undefined) {
    return null;
  }
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 20 20");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  for (const d of paths) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "1.8");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    svg.append(path);
  }
  return svg;
}

export function renderSwitcher(
  container: HTMLElement,
  selected: string,
  onSelect: (id: string) => void,
): void {
  container.replaceChildren(
    ...SOURCES.map((source) => {
      const button = document.createElement("button");
      button.type = "button";
      const icon = sourceIcon(source.id);
      const label = document.createElement("span");
      label.textContent = source.label;
      button.append(...(icon === null ? [] : [icon]), label);
      button.setAttribute("aria-pressed", String(source.id === selected));
      button.addEventListener("click", () => {
        onSelect(source.id);
      });
      return button;
    }),
  );
}

export function markSelectedSource(container: HTMLElement, selected: string): void {
  const buttons = [...container.querySelectorAll("button")];
  buttons.forEach((button, index) => {
    button.setAttribute("aria-pressed", String(SOURCES[index]?.id === selected));
  });
}
