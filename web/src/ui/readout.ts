/** The live readout beside the video: the total on the sign, the two
 * directions, the vehicle types and the crossings as they happen.
 *
 * Every figure here is a count the engine made in this tab, in this session,
 * against the line as it is now; nothing has a default. It is written on every
 * detection, so it changes only what changed: the sign flips only the LEDs that
 * differ, a bar moves by transform, and a list is redrawn only when what it
 * lists has changed. */

import { formatClock, formatCount } from "./format";
import { displayName } from "./vehicles";
import { VmsSign } from "./vms";

/** One counted crossing, as the feed shows it. */
export interface FeedEntry {
  /** Increases with every crossing in a session, so the newest are known. */
  readonly id: number;
  /** Clip time of the crossing, in seconds. */
  readonly timestamp: number;
  readonly className: string;
  readonly direction: string;
  readonly wrongWay: boolean;
}

/** How many crossings the feed keeps on screen. */
export const FEED_LENGTH = 6;

export interface ReadoutState {
  readonly total: number;
  /** Every type the source counts, in the source's own order, with its count. */
  readonly perClass: readonly (readonly [string, number])[];
  /** The two directions: the line's positive side first. */
  readonly perDirection: readonly (readonly [string, number])[];
  /** Clip time the counts restarted from, or null if they cover the session. */
  readonly countingSince: number | null;
  /** Newest first. */
  readonly feed: readonly FeedEntry[];
  /** On-screen angle, in degrees, of the line's positive side. The arrows in
   * the readout turn with it, so they always match the arrows on the video. */
  readonly positiveAngleDeg: number;
}

export interface ReadoutElements {
  readonly vms: HTMLElement;
  readonly totalCount: HTMLElement;
  readonly countingSince: HTMLElement;
  readonly directionCounts: HTMLElement;
  readonly classCounts: HTMLElement;
  readonly feed: HTMLElement;
}

/** The share of the total a type's bar shows. Zero when nothing has been
 * counted: an empty bar, not a guessed one. */
export function share(count: number, total: number): number {
  if (!(total > 0) || !(count > 0)) {
    return 0;
  }
  return Math.min(1, count / total);
}

/** The sentence under the sign once the line has moved. */
export function countingSinceText(countingSince: number | null): string {
  return countingSince === null
    ? ""
    : `Counting since ${formatClock(countingSince)}, when the line moved.`;
}

const SVG_NS = "http://www.w3.org/2000/svg";

/** A small arrow pointing right, turned by the caller. */
export function arrowIcon(): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", "M2.5 8h9M8 3.5 12.5 8 8 12.5");
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "currentColor");
  path.setAttribute("stroke-width", "2");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.append(path);
  return svg;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text = "",
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className !== "") {
    node.className = className;
  }
  if (text !== "") {
    node.textContent = text;
  }
  return node;
}

export class Readout {
  private readonly elements: ReadoutElements;
  private readonly sign: VmsSign;
  private directionKey = "";
  private classKey = "";
  /** Null until the feed has been drawn once, so an empty first feed still
   * draws its empty message. */
  private feedKey: string | null = null;
  private newestShown = 0;
  private readonly directionCells = new Map<string, { value: HTMLElement; arrow: SVGSVGElement; row: HTMLElement }>();
  private readonly classRows = new Map<string, { row: HTMLElement; bar: HTMLElement; value: HTMLElement }>();

  constructor(elements: ReadoutElements) {
    this.elements = elements;
    this.sign = new VmsSign(elements.vms);
  }

  render(state: ReadoutState): void {
    this.sign.show(state.total);
    setText(this.elements.totalCount, formatCount(state.total));
    setText(this.elements.countingSince, countingSinceText(state.countingSince));
    this.renderDirections(state);
    this.renderClasses(state);
    this.renderFeed(state);
  }

  /** Forget what has been shown, so a new source starts with nothing marked
   * as new and every list is drawn afresh for its own types. */
  reset(): void {
    this.directionKey = "";
    this.classKey = "";
    this.feedKey = null;
    this.newestShown = 0;
  }

  private renderDirections(state: ReadoutState): void {
    const key = state.perDirection.map(([name]) => name).join("|");
    if (key !== this.directionKey) {
      this.directionCells.clear();
      const items = state.perDirection.map(([name], index) => {
        const row = el("li", "direction");
        const label = el("span", "direction__name");
        const arrow = arrowIcon();
        arrow.dataset["side"] = index === 0 ? "positive" : "negative";
        label.append(arrow, el("span", "", name));
        const value = el("span", "direction__value num");
        row.append(label, value);
        this.directionCells.set(name, { value, arrow, row });
        return row;
      });
      this.elements.directionCounts.replaceChildren(...items);
      this.directionKey = key;
    }
    state.perDirection.forEach(([name, count], index) => {
      const cell = this.directionCells.get(name);
      if (cell === undefined) {
        return;
      }
      setText(cell.value, formatCount(count));
      cell.row.dataset["zero"] = String(count === 0);
      const angle = state.positiveAngleDeg + (index === 0 ? 0 : 180);
      cell.arrow.style.transform = `rotate(${angle.toFixed(1)}deg)`;
    });
  }

  private renderClasses(state: ReadoutState): void {
    const key = state.perClass.map(([name]) => name).join("|");
    if (key !== this.classKey) {
      this.classRows.clear();
      const items = state.perClass.map(([name]) => {
        const row = el("li", "type");
        row.dataset["class"] = name;
        const bar = el("span", "type__bar");
        bar.append(el("i", ""));
        const value = el("span", "type__value num");
        row.append(el("span", "type__swatch"), el("span", "type__name", displayName(name)), bar, value);
        this.classRows.set(name, { row, bar, value });
        return row;
      });
      this.elements.classCounts.replaceChildren(...items);
      this.classKey = key;
    }
    for (const [name, count] of state.perClass) {
      const row = this.classRows.get(name);
      if (row === undefined) {
        continue;
      }
      setText(row.value, formatCount(count));
      row.row.dataset["zero"] = String(count === 0);
      row.bar.style.setProperty("--share", share(count, state.total).toFixed(4));
    }
  }

  private renderFeed(state: ReadoutState): void {
    const { feed } = state;
    // The arrows follow the line, so the key carries its angle as well as the
    // crossings: a turned line has to turn the arrows already listed.
    const key = `${feed.map((entry) => entry.id).join(",")}@${state.positiveAngleDeg.toFixed(0)}`;
    if (key === this.feedKey) {
      return;
    }
    this.feedKey = key;
    if (feed.length === 0) {
      this.elements.feed.replaceChildren(
        el("li", "feed__empty", "Crossings appear here as vehicles pass the line."),
      );
      this.newestShown = 0;
      return;
    }
    const newest = this.newestShown;
    const positive = state.perDirection[0]?.[0];
    this.elements.feed.replaceChildren(
      ...feed.slice(0, FEED_LENGTH).map((entry) => {
        const item = el("li", entry.id > newest ? "feed__item is-new" : "feed__item");
        item.dataset["class"] = entry.className;
        item.dataset["wrong"] = String(entry.wrongWay);
        // The same arrow the direction counts and the video's tags show, so a
        // crossing reads the same way in all three places.
        const arrow = arrowIcon();
        const angle = state.positiveAngleDeg + (entry.direction === positive ? 0 : 180);
        arrow.style.transform = `rotate(${angle.toFixed(1)}deg)`;
        const direction = el("span", "feed__dir");
        direction.append(arrow, el("span", "", entry.wrongWay ? "wrong way" : entry.direction));
        item.append(
          el("span", "feed__time", formatClock(entry.timestamp)),
          el("span", "feed__swatch"),
          el("span", "feed__what", displayName(entry.className)),
          direction,
        );
        return item;
      }),
    );
    this.newestShown = Math.max(newest, ...feed.map((entry) => entry.id));
  }
}

function setText(node: HTMLElement, text: string): void {
  if (node.textContent !== text) {
    node.textContent = text;
  }
}
