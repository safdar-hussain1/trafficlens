/** The demo page's one measured claim, drawn so it reads at a glance.
 *
 * A precision, a recall and an F1 are three numbers a visitor has to decode.
 * What they summarise is simpler and fits on one line: of the crossings a
 * person labelled by hand, how many did the engine count, how many did it
 * miss, and how many times did it count something that was not there. So each
 * labelled crossing is drawn as one square, and the extra counts sit after a
 * gap, and the three rates follow underneath for anyone who wants them.
 *
 * Every figure is read from `../generated/reports.ts` -- the bake of
 * `reports/counting_accuracy.json` -- and none is typed here. The method is
 * addressed by its published name and a missing one throws, for the same
 * reason `results.ts` does: a lookup that fell back would print some other
 * method's figures under this claim. */

import { REPORTS } from "../generated/reports";
import { rate } from "./kit";

/** The engine's own tracker with the gate rule: what the page ships. */
export const SHIPPED_METHOD = "engine+gate";

export function methodByName(name: string) {
  const found = REPORTS.counting.methods.find((item) => item.method === name);
  if (found === undefined) {
    throw new Error(
      `no counting method named "${name}"; the bake carries ` +
        REPORTS.counting.methods.map((item) => item.method).join(", "),
    );
  }
  return found;
}

export interface Tally {
  /** Crossings a person labelled. */
  readonly labelled: number;
  /** Labelled crossings the engine counted. */
  readonly counted: number;
  /** Labelled crossings it did not count. */
  readonly missed: number;
  /** Counts it made that matched no labelled crossing. */
  readonly extra: number;
  readonly precision: number;
  readonly recall: number;
  readonly f1: number;
  /** The detector the benchmark ran, as a visitor would name it. */
  readonly detector: string;
}

/** "yolo11s.pt" as the model is called in prose: "YOLO11s". */
export function modelName(file: string): string {
  return file.replace(/\.(pt|onnx)$/i, "").replace(/^yolo/i, "YOLO");
}

export function tally(): Tally {
  const method = methodByName(SHIPPED_METHOD);
  const labelled = method.nGroundTruth;
  const counted = method.truePositives;
  return {
    labelled,
    counted,
    missed: labelled - counted,
    extra: method.nPredicted - counted,
    precision: method.precision,
    recall: method.recall,
    f1: method.f1,
    detector: modelName(REPORTS.counting.detector.model),
  };
}

export type UnitKind = "hit" | "miss" | "false";

/** The squares, in reading order: every labelled crossing, counted ones
 * first, then the extra counts. The gap between the two groups is drawn by the
 * caller, because it is not a crossing. */
export function units(value: Tally): { readonly labelled: UnitKind[]; readonly extra: UnitKind[] } {
  return {
    labelled: [
      ...new Array<UnitKind>(value.counted).fill("hit"),
      ...new Array<UnitKind>(value.missed).fill("miss"),
    ],
    extra: new Array<UnitKind>(value.extra).fill("false"),
  };
}

const SVG_NS = "http://www.w3.org/2000/svg";

function glyph(kind: UnitKind): SVGSVGElement | null {
  if (kind === "miss") {
    return null;
  }
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", kind === "hit" ? "M3.5 8.5 6.5 11.5 12.5 4.5" : "M4.5 4.5l7 7M11.5 4.5l-7 7");
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "currentColor");
  path.setAttribute("stroke-width", "2.2");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.append(path);
  return svg;
}

function unit(kind: UnitKind, tag: "li" | "span" = "li"): HTMLElement {
  const node = document.createElement(tag);
  node.className = "unit";
  node.dataset["kind"] = kind;
  const mark = glyph(kind);
  if (mark !== null) {
    node.append(mark);
  }
  return node;
}

function text<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  content: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className !== "") {
    node.className = className;
  }
  node.textContent = content;
  return node;
}

export function mountAccuracy(): void {
  const slot = document.querySelector<HTMLElement>('[data-accuracy="tally"]');
  if (slot === null) {
    throw new Error('missing accuracy slot [data-accuracy="tally"]');
  }
  const value = tally();
  const squares = units(value);

  const hero = document.createElement("div");
  hero.className = "tally__hero";
  hero.append(
    text("span", "tally__value", `${value.counted} of ${value.labelled}`),
    text("span", "tally__unit", "real crossings counted"),
  );

  // The squares restate the legend below them, so they are hidden from a
  // screen reader rather than read out one by one.
  const strip = document.createElement("ul");
  strip.className = "tally__units";
  strip.setAttribute("aria-hidden", "true");
  strip.append(...squares.labelled.map((kind) => unit(kind)));
  if (squares.extra.length > 0) {
    const gap = document.createElement("li");
    gap.className = "unit-gap";
    strip.append(gap, ...squares.extra.map((kind) => unit(kind)));
  }

  const legend = document.createElement("ul");
  legend.className = "tally__legend";
  const entries: [UnitKind, string][] = [
    ["hit", `Counted (${value.counted})`],
    ["miss", `Missed (${value.missed})`],
    ["false", `Counted with no real crossing (${value.extra})`],
  ];
  legend.append(
    ...entries.map(([kind, label]) => {
      const item = document.createElement("li");
      item.append(unit(kind, "span"), text("span", "", label));
      return item;
    }),
  );

  const scores = document.createElement("dl");
  scores.className = "tally__scores";
  const rates: [string, number][] = [
    ["Precision", value.precision],
    ["Recall", value.recall],
    ["F1", value.f1],
  ];
  scores.append(
    ...rates.map(([term, figure]) => {
      const pair = document.createElement("div");
      pair.append(text("dt", "", term), text("dd", "", rate(figure)));
      return pair;
    }),
  );

  const caption = text(
    "figcaption",
    "",
    `Scored offline with the Python engine and the ${value.detector} detector, on one clip and ` +
      "one line, which is the easy case, so read this as a best case. The page runs the same " +
      "counting engine with the lighter YOLO11n.",
  );

  slot.replaceChildren(hero, strip, legend, scores, caption);
}
