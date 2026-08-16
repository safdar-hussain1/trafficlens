/** The parts of the control room that are markup rather than canvas: the
 * backend badge, the panels, the source switcher and the theme.
 *
 * Every writer here takes what was actually measured and prints it, or prints
 * nothing. There is no default number anywhere in this file. */

import type { BackendProbe } from "../runtime/backend";
import { formatCount, formatMs, formatFps, formatClock, NO_VALUE } from "./format";
import type { Cadence } from "./format";
import type { SourceSpec } from "./sources";
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

export interface Elements {
  readonly badge: HTMLElement;
  readonly themeToggle: HTMLButtonElement;
  readonly videoCanvas: HTMLCanvasElement;
  readonly chartCanvas: HTMLCanvasElement;
  readonly stage: HTMLElement;
  readonly handles: {
    readonly start: HTMLButtonElement;
    readonly body: HTMLButtonElement;
    readonly end: HTMLButtonElement;
  };
  readonly emptyState: HTMLElement;
  readonly emptyText: HTMLElement;
  readonly progress: HTMLProgressElement;
  readonly videoNote: HTMLElement;
  readonly videoCaption: HTMLElement;
  readonly chartNote: HTMLElement;
  readonly totalCount: HTMLElement;
  readonly totalUnit: HTMLElement;
  readonly countingSince: HTMLElement;
  readonly classCounts: HTMLTableElement;
  readonly directionCounts: HTMLTableElement;
  readonly speedReadout: HTMLElement;
  readonly speedReason: HTMLElement;
  readonly incidents: HTMLElement;
  readonly switcher: HTMLElement;
  readonly startButton: HTMLButtonElement;
  readonly resetButton: HTMLButtonElement;
  readonly statusLine: HTMLElement;
}

export function collectElements(): Elements {
  return {
    badge: element("backend-badge"),
    themeToggle: element<HTMLButtonElement>("theme-toggle"),
    videoCanvas: element<HTMLCanvasElement>("video-canvas"),
    chartCanvas: element<HTMLCanvasElement>("chart-canvas"),
    stage: element("stage-video"),
    handles: {
      start: element<HTMLButtonElement>("handle-start"),
      body: element<HTMLButtonElement>("handle-body"),
      end: element<HTMLButtonElement>("handle-end"),
    },
    emptyState: element("empty-state"),
    emptyText: element("empty-text"),
    progress: element<HTMLProgressElement>("load-progress"),
    videoNote: element("video-note"),
    videoCaption: element("video-caption"),
    chartNote: element("chart-note"),
    totalCount: element("total-count"),
    totalUnit: element("total-unit"),
    countingSince: element("counting-since"),
    classCounts: element<HTMLTableElement>("class-counts"),
    directionCounts: element<HTMLTableElement>("direction-counts"),
    speedReadout: element("speed-readout"),
    speedReason: element("speed-reason"),
    incidents: element("incidents"),
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

export function renderThemeToggle(button: HTMLButtonElement): void {
  const theme = currentTheme();
  const next = theme === "dark" ? "light" : "dark";
  button.textContent = theme === "dark" ? "Light theme" : "Dark theme";
  button.setAttribute("aria-label", `Switch to the ${next} theme`);
}

// -- badge --------------------------------------------------------------------

export interface BadgeState {
  readonly probe: BackendProbe | null;
  /** The provider the session was actually created with, once there is one. */
  readonly ep: string | null;
  readonly msPerFrame: number | null;
  readonly fps: number | null;
  readonly cadence: Cadence | null;
}

/** The badge names what is running and what it measured, on this machine, in
 * this session. A software renderer invalidates any hardware claim, so it is
 * called out rather than printed as if it were a GPU. */
export function renderBadge(badge: HTMLElement, state: BadgeState): void {
  const { probe } = state;
  if (probe === null) {
    badge.replaceChildren(text("span", "badge__renderer", "Checking what this machine can run…"));
    return;
  }
  const ep = (state.ep ?? probe.ep) === "webgpu" ? "WebGPU" : "WASM";
  const parts: HTMLElement[] = [
    text("b", "", ep),
    text("span", "badge__renderer", probe.renderer),
  ];
  if (state.msPerFrame !== null) {
    parts.push(text("span", "", `${formatMs(state.msPerFrame)} ms/frame`));
  }
  if (state.fps !== null) {
    parts.push(text("span", "", `${formatFps(state.fps)} fps`));
  }
  if (state.cadence !== null) {
    parts.push(text("span", "", `detecting ${state.cadence.label}`));
  }
  if (!probe.isHardwareRenderer) {
    parts.push(text("span", "badge__warn", "software renderer — not a hardware timing"));
  }
  badge.dataset["software"] = String(!probe.isHardwareRenderer);
  badge.replaceChildren(...parts);
}

function text(tag: string, className: string, content: string): HTMLElement {
  const node = document.createElement(tag);
  if (className !== "") {
    node.className = className;
  }
  node.textContent = content;
  return node;
}

// -- panels -------------------------------------------------------------------

function renderFigures(
  table: HTMLTableElement,
  rows: readonly (readonly [string, number])[],
): void {
  const body = table.tBodies[0] ?? table.createTBody();
  body.replaceChildren(
    ...rows.map(([label, value]) => {
      const row = document.createElement("tr");
      row.dataset["zero"] = String(value === 0);
      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = label;
      const td = document.createElement("td");
      td.textContent = formatCount(value);
      row.append(th, td);
      return row;
    }),
  );
}

export interface PanelState {
  readonly source: SourceSpec;
  readonly total: number;
  readonly perClass: readonly (readonly [string, number])[];
  readonly perDirection: readonly (readonly [string, number])[];
  readonly countingSince: number | null;
  readonly wrongWay: readonly string[];
}

export function renderPanels(elements: Elements, state: PanelState): void {
  elements.totalCount.textContent = formatCount(state.total);
  elements.totalUnit.textContent = `crossings of ${state.source.gate.name}`;
  elements.countingSince.textContent =
    state.countingSince === null
      ? ""
      : `Counting from ${formatClock(state.countingSince)}, when the gate last moved.`;

  renderFigures(elements.classCounts, state.perClass);
  renderFigures(elements.directionCounts, state.perDirection);

  // The refusal, in words, exactly as the engine makes it. An uncalibrated
  // source has no speed at all -- not a zero, not a blank.
  elements.speedReadout.textContent = state.source.calibrated ? NO_VALUE : "no speed";
  elements.speedReason.textContent = state.source.speedNote;

  if (!state.source.calibrated) {
    elements.incidents.replaceChildren(
      text("span", "refusal", "none possible"),
      text(
        "p",
        "reason",
        "Stopped-vehicle detection compares a calibrated speed against a threshold, so it cannot fire on an unsurveyed camera.",
      ),
    );
    return;
  }
  if (state.wrongWay.length === 0) {
    elements.incidents.replaceChildren(text("span", "refusal", "none"));
    return;
  }
  const list = document.createElement("ul");
  list.className = "alert-list";
  list.append(...state.wrongWay.map((line) => text("li", "", line)));
  elements.incidents.replaceChildren(list);
}

/** Wrong-way crossings are alerts and get the amber treatment, on any source:
 * they need no calibration, only a gate that names the direction it expects. */
export function renderAlerts(elements: Elements, lines: readonly string[]): void {
  if (lines.length === 0) {
    return;
  }
  const list = document.createElement("ul");
  list.className = "alert-list";
  list.append(...lines.map((line) => text("li", "", line)));
  elements.incidents.replaceChildren(
    text("span", "alert", `${lines.length} wrong-way crossing${lines.length === 1 ? "" : "s"}`),
    list,
  );
}

// -- source switcher ----------------------------------------------------------

export function renderSwitcher(
  container: HTMLElement,
  selected: string,
  onSelect: (id: string) => void,
): void {
  container.replaceChildren(
    ...SOURCES.map((source) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = source.label;
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
