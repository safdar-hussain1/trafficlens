/** How the results sections print a measurement, and what they print it into.
 *
 * Two things live here and nothing else: the formatters, which are this page's
 * precision policy in code, and the small set of structures every measured
 * section is built from -- the protocol strip, the data table, the figure run,
 * the plate. They know nothing about any particular benchmark, which is the seam:
 * a section decides WHAT to show, this module decides how a figure is allowed to
 * look, and the two cannot drift into disagreeing about precision.
 *
 * The one exception is the resolution pair, which reads the baked label figure.
 * It is here because it is a formatting decision -- how many digits a rate on
 * this page is entitled to -- and it must be the same sentence everywhere it
 * appears. */

import { REPORTS } from "../generated/reports";
import { threeSigFigs } from "./figures";
import type { Series } from "./figures";

// -- formatting ---------------------------------------------------------------
//
// Exported where a test can reach them: these are the functions that decide how
// much precision the page claims, which is a decision the measurement makes and
// not the renderer.

/** A rate, to three significant figures. Anything more is precision the
 * seventeen labelled crossings cannot support. */
export const rate = threeSigFigs;

/** Would this fixed-decimal rendering turn a measured non-zero into a zero?
 *
 * The defect this exists to make impossible, class-wide: a rounding policy that
 * can print a measurement as `0.0` is a policy that can publish a perfect score
 * the instrument never recorded. It bit the product's ONLY absolute speed claim
 * -- the tier-one settled mean relative error is 0.0390 %, and one fixed decimal
 * rendered it `0.0 %` -- and the comment on `threeSigFigs` had already named the
 * hazard: printing a zero there would read as a perfect score.
 *
 * An exact zero is a different thing and must keep printing as zero: at 2 fps the
 * engine really made no predictions at all. So the test is "non-zero in, zero
 * out", not "small". */
function roundsToZero(value: number, text: string): boolean {
  return value !== 0 && Number(text) === 0;
}

/** A percentage, one decimal -- unless one decimal would erase the measurement,
 * in which case three significant figures, the same precision policy `rate` uses.
 *
 * `0.0390 %` and `0.0 %` are not the same claim, and the second one is not true. */
export function percent(fraction: number): string {
  if (!Number.isFinite(fraction)) {
    return "—";
  }
  const value = fraction * 100;
  const text = value.toFixed(1);
  return `${roundsToZero(value, text) ? threeSigFigs(value) : text} %`;
}

/** Megabytes, decimal, as a download is quoted. Two decimals: the difference
 * between 7.69 and 7.7 MB is not the point, but a bare 8 MB overstates it.
 * A saving too small for two decimals is printed rather than rounded away. */
export function megabytes(bytes: number): string {
  if (!Number.isFinite(bytes)) {
    return "—";
  }
  const value = bytes / 1e6;
  const text = value.toFixed(2);
  return `${roundsToZero(value, text) ? threeSigFigs(value) : text} MB`;
}

/** A count, grouped in threes with a narrow no-break space -- the same separator
 * the control room uses, because a comma is a decimal separator across most of
 * Europe including where the flagship clip was filmed. */
export function count(value: number): string {
  // An em dash where there is no count, the same as every other formatter here
  // -- this was the one that lacked the guard. The architecture table reads the
  // model card's cells, which are `number | null`, and without it a cell the
  // card leaves empty printed the literal string `NaN` in a published table.
  if (!Number.isFinite(value)) {
    return "—";
  }
  return value.toLocaleString("en-GB").replace(/,/g, " ");
}

/** A figure with a fixed number of decimals, or an em dash where there is none.
 * Falls back to the shortest honest form rather than rounding a measurement away
 * to zero -- see `roundsToZero`. */
export function fixed(value: number | null, digits: number): string {
  if (value === null || !Number.isFinite(value)) {
    return "—";
  }
  const text = value.toFixed(digits);
  return roundsToZero(value, text) ? scientific(value) : text;
}

/** A signed figure with a real minus sign, and an explicit plus.
 *
 * A hyphen-minus is a hyphen; at the size these are set it reads as a dash
 * joining two words. The explicit plus matters more: the scale bracket is
 * `−33 %/+0 %`, and a bare `0` at the upper end loses the fact that the
 * assumption is the ceiling rather than the middle of a symmetric band.
 *
 * `+0` is therefore a real reading and stays. A non-zero that merely ROUNDS to
 * zero is not, and gets the precision it needs instead. */
export function signed(value: number, digits = 0): string {
  if (!Number.isFinite(value)) {
    return "—";
  }
  const text = Math.abs(value).toFixed(digits);
  const body = roundsToZero(value, text) ? threeSigFigs(Math.abs(value)) : text;
  return value < 0 ? `−${body}` : `+${body}`;
}

/** A very small or very large number in the shortest honest form.
 *
 * The tier-1 holdout error is 6.19e-06 m and the noise sweep spans four decades,
 * so a fixed number of decimals prints either zeros or noise. Exponent notation
 * is what a reader of a residual expects anyway. */
export function scientific(value: number, digits = 3): string {
  if (!Number.isFinite(value)) {
    return "—";
  }
  if (value !== 0 && (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e5)) {
    return value.toExponential(digits - 1);
  }
  return threeSigFigs(value);
}

/** The counting resolution as a bare figure, for a stat tile.
 *
 * A tile has room for a number and not for a sentence, so the resolution now
 * appears on this page in two shapes. They are one function apart on purpose:
 * a tile that formatted `oneEventF1` itself would be a second spelling of this
 * page's precision policy, free to drift from the caption sitting under it. */
export function countingResolution(): string {
  return `± ${rate(REPORTS.counting.resolution.oneEventF1)}`;
}

/** The resolution note that must travel with every counting figure on this page.
 *
 * One added or removed event moves F1 by this much, so two methods closer
 * together than this differ by one event rather than in quality. Computed in the
 * bake from the label count and the engine's own operating point. */
export function resolutionNote(): string {
  return `${countingResolution()} F1 (one event)`;
}

/** The same warning, in the fragmentation metric's own terms.
 *
 * The identity benchmark does not score F1, so the sentence above would be the
 * wrong units -- but the hazard is identical and it is why this note exists at
 * all. Fragmentation is predicted identities divided by labelled vehicles, and
 * the denominator is the label count, so ONE identity more or fewer moves the
 * ratio by exactly `1 / labels`. Every difference smaller than that is one
 * identity, not a tracker being better at keeping them; and a row-by-row
 * comparison of three trackers is precisely where a reader over-reads one. */
export function fragmentationResolution(): string {
  const denominator = REPORTS.tracking.metricDefinitions.fragmentationRatio.denominator;
  return `± ${rate(1 / denominator)}`;
}

export function fragmentationResolutionNote(): string {
  return `${fragmentationResolution()} fragmentation (one identity)`;
}

// -- DOM helpers --------------------------------------------------------------

export type Child = Node | string;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: readonly Child[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
}

export function p(text: string, className = ""): HTMLParagraphElement {
  return h("p", className === "" ? {} : { class: className }, [text]);
}

export function list(items: readonly string[], className = "plain-list"): HTMLUListElement {
  return h("ul", { class: className }, items.map((item) => h("li", {}, [item])));
}

/** A disclosure for the material a reader wants only when they are checking.
 *
 * The caveats, the full level-by-level tables and the metric definitions are all
 * load-bearing and all long. Collapsed, they stay on the page and reachable --
 * `<details>` content is in the document, so it is searchable, printable and
 * indexable -- without pushing the results themselves below three screens. */
export function disclosure(summary: string, children: readonly Child[]): HTMLDetailsElement {
  return h("details", { class: "reveal" }, [h("summary", {}, [summary]), ...children]);
}

/** The protocol strip: the conditions, in mono, above the numbers they govern. */
export function protocol(items: readonly string[]): HTMLElement {
  const strip = h("p", { class: "protocol" }, []);
  items.forEach((item, index) => {
    if (index > 0) {
      strip.append(h("span", { class: "protocol__sep", "aria-hidden": "true" }, ["·"]));
    }
    strip.append(h("span", {}, [item]));
  });
  return strip;
}

export interface Column {
  readonly head: string;
  /** Numeric columns are right-aligned and set in mono with tabular figures. */
  readonly numeric?: boolean;
}

export type Cell = string | { readonly text: string; readonly emphasis: boolean };

/** A data table. Scrolls inside its own box rather than pushing the page sideways. */
export function table(
  caption: string,
  columns: readonly Column[],
  rows: readonly (readonly Cell[])[],
  groups: readonly { readonly label: string; readonly rows: readonly (readonly Cell[])[] }[] = [],
): HTMLElement {
  const head = h(
    "thead",
    {},
    [
      h(
        "tr",
        {},
        columns.map((column) =>
          h("th", { scope: "col", class: column.numeric === true ? "num" : "" }, [column.head]),
        ),
      ),
    ],
  );

  const body = (source: readonly (readonly Cell[])[]): HTMLTableSectionElement =>
    h(
      "tbody",
      {},
      source.map((row) =>
        h(
          "tr",
          {},
          row.map((cell, index) => {
            const value = typeof cell === "string" ? cell : cell.text;
            const emphasis = typeof cell === "string" ? false : cell.emphasis;
            const column = columns[index];
            const numeric = column?.numeric === true;
            const classes = [numeric ? "num" : "", emphasis ? "row-emphasis" : ""]
              .filter((part) => part !== "")
              .join(" ");
            return index === 0
              ? h("th", { scope: "row", class: classes }, [value])
              : h("td", { class: classes }, [value]);
          }),
        ),
      ),
    );

  const sections: Child[] = [head];
  if (groups.length > 0) {
    for (const group of groups) {
      const section = body(group.rows);
      section.prepend(
        h("tr", { class: "group" }, [
          h("th", { scope: "colgroup", colspan: String(columns.length) }, [group.label]),
        ]),
      );
      sections.push(section);
    }
  } else {
    sections.push(body(rows));
  }

  return h("div", { class: "table-wrap" }, [
    h("table", { class: "data" }, [h("caption", {}, [caption]), ...sections]),
  ]);
}

/** A run of figures, each with its own label. The page's smallest unit of
 * evidence: a measured number that cannot appear without saying what it is. */
export function figures(items: readonly (readonly [string, string])[]): HTMLElement {
  return h(
    "dl",
    { class: "figure-run" },
    items.flatMap(([term, value]) => [h("dt", {}, [term]), h("dd", {}, [value])]),
  );
}

/** One stat tile: a label, a figure set large in mono, and an optional note.
 *
 * The same vocabulary the control room above uses for a live count -- an
 * uppercase label over a big mono figure -- so a measurement taken last month
 * and a measurement taken in this tab read as one instrument rather than as a
 * dashboard and a report stapled together. `lead` marks the one figure a row is
 * about; it is drawn as a rule in the structural accent, which is the only
 * colour this page spends on anything that is not a chart mark. */
export interface Tile {
  readonly label: string;
  readonly value: string;
  readonly note?: string;
  readonly lead?: boolean;
}

/** A row of stat tiles.
 *
 * A definition list, because that is what a label and its figure are, and
 * because it keeps the pairing for a reader who never sees the grid. `"minor"`
 * is for a row that qualifies the row above it rather than leading a section:
 * same component, one step down in size, so the reading order is visible before
 * a word is read. */
export function tiles(items: readonly Tile[], variant: "" | "minor" = ""): HTMLElement {
  return h(
    "dl",
    { class: variant === "minor" ? "tiles tiles--minor" : "tiles" },
    items.map((item) =>
      h("div", { class: item.lead === true ? "tile tile--lead" : "tile" }, [
        h("dt", { class: "tile__label" }, [item.label]),
        h("dd", { class: "tile__value" }, [
          item.value,
          ...(item.note === undefined ? [] : [h("span", { class: "tile__note" }, [item.note])]),
        ]),
      ]),
    ),
  );
}

/** The face of an honest-negative card: one figure, large, and what it is.
 *
 * The card's claim sentence is authored in the markup; this is the measurement
 * under it. Everything else the finding rests on goes in the disclosure beside
 * it rather than being cut -- the words are what this page shed, not the
 * figures. */
export function headlineFigure(term: string, value: string): HTMLElement {
  return h("p", { class: "finding__figure" }, [
    h("span", { class: "finding__value" }, [value]),
    h("span", { class: "finding__label" }, [term]),
  ]);
}

/** A figure and its caption. `<figure>` because that is what this is, and the
 * caption is where the chart says what a reader may and may not read off it.
 *
 * `before` is for a legend, which belongs ABOVE the art: identity has to be
 * established before the marks are read, not explained after them. */
export function plate(
  chart: Element,
  caption: string,
  before: readonly Child[] = [],
  after: readonly Child[] = [],
): HTMLElement {
  return h("figure", { class: "plate" }, [
    ...before,
    h("div", { class: "plate__art" }, [chart]),
    h("figcaption", {}, [caption]),
    ...after,
  ]);
}

export function legend(series: readonly Series[]): HTMLElement {
  return h(
    "p",
    { class: "fig-legend" },
    series.map((item) =>
      h("span", { class: item.emphasis ? "key key--emphasis" : "key key--quiet" }, [
        h("i", { class: item.dashed ? "key__dash" : "" }, []),
        item.label,
      ]),
    ),
  );
}

