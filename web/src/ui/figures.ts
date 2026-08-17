/** The static figures below the control room, drawn as SVG.
 *
 * Three of them, and each one exists because a claim on this page is easier to
 * check as a picture than as a sentence:
 *
 *   - `crossingRuleDiagram` is the counting rule itself, in the same geometry the
 *     live diagram uses -- a gate at zero, time to the right, one line per
 *     vehicle. The explanation and the running chart are therefore the same
 *     picture rather than two things a reader has to trust agree. It carries all
 *     four cases at once: a counted crossing, a band stepped clean over, a band
 *     sat inside, and a vehicle past the segment's ends.
 *   - `robustnessSmallMultiples` is four degradation protocols side by side, one
 *     panel each, three trackers per panel.
 *   - `matchedControlsChart` is the scale survey's negative result WITH its
 *     controls, which is what makes "not measurable" a measurement.
 *
 * Why SVG and not canvas, when the live diagram is canvas: these figures never
 * change, and an SVG's marks can be coloured by CSS class. That means the theme
 * toggle recolours them with no redraw, they survive with the engine stopped, and
 * they scale to a phone without a device-pixel-ratio dance. The live diagram
 * redraws sixty times a second and canvas is right for that; nothing here does.
 *
 * No chart library. The page's headline claim is that nothing leaves the device
 * after load, so a chart fetched from another origin would make it false -- and
 * bundling one to draw eleven polylines would be the wrong trade anyway.
 *
 * Colour does one job here and it is emphasis, not identity: the engine's own
 * tracker is the only chromatic mark and the two baseline trackers share the
 * de-emphasis grey, told apart by stroke pattern and by a label at the line's
 * end. That is deliberate. This palette has exactly one structural accent -- sign
 * blue -- and its second colour, works amber, is reserved for alerts; a
 * degradation curve is not an alert. A third hue invented for a third series
 * would break both rules at once. */

import { BASELINE_BAND_PX } from "../generated/constants";

const SVG_NS = "http://www.w3.org/2000/svg";

export interface Point {
  readonly x: number;
  readonly y: number;
}

/** A linear map from data to viewBox units. Kept as a plain function rather than
 * a class because every caller wants exactly this and nothing else. */
export function linear(
  domain: readonly [number, number],
  range: readonly [number, number],
): (value: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  // A zero-width domain would divide by zero and put every mark at NaN, which
  // renders as nothing at all -- an empty chart that looks like missing data
  // rather than like a bug. Collapse to the range's start instead.
  const span = d1 - d0;
  if (span === 0) {
    return () => r0;
  }
  return (value) => r0 + ((value - d0) / span) * (r1 - r0);
}

/** A polyline's `d`. Empty for fewer than two points: a one-point path draws a
 * dot the reader would read as a measurement at a single level. */
export function pathFrom(points: readonly Point[]): string {
  if (points.length < 2) {
    return "";
  }
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"}${round(point.x)} ${round(point.y)}`)
    .join(" ");
}

/** A horizontal bar whose FAR end is rounded and whose baseline end is square.
 *
 * `rx` on a rect rounds all four corners, which detaches the bar from its
 * baseline and makes a short bar read as a lozenge floating in the plot. The
 * radius is also clamped to half the bar's length, so a bar shorter than the
 * radius stays a bar instead of turning into a semicircle. */
export function barPath(x0: number, y: number, length: number, height: number, radius: number): string {
  const r = Math.max(0, Math.min(radius, length / 2, height / 2));
  const end = x0 + Math.max(0, length);
  if (r === 0) {
    return `M${round(x0)} ${round(y)} H${round(end)} V${round(y + height)} H${round(x0)} Z`;
  }
  return (
    `M${round(x0)} ${round(y)} H${round(end - r)}` +
    ` A${round(r)} ${round(r)} 0 0 1 ${round(end)} ${round(y + r)}` +
    ` V${round(y + height - r)}` +
    ` A${round(r)} ${round(r)} 0 0 1 ${round(end - r)} ${round(y + height)}` +
    ` H${round(x0)} Z`
  );
}

/** Nudge label positions apart without letting them leave their box.
 *
 * Direct end-labels only work while the series separate at the right edge, and
 * these do converge -- three trackers at F1 1.0 sit on top of each other. Moving
 * them the minimum distance that clears a gap keeps each label next to its own
 * line; stacking them in a column would detach them from the lines and read as
 * noise. Order is preserved, so a label never crosses its neighbour. */
export function spreadLabels(
  positions: readonly number[],
  minGap: number,
  bounds: readonly [number, number],
): number[] {
  const order = positions
    .map((value, index) => ({ value, index }))
    .sort((a, b) => a.value - b.value);
  const out = positions.slice();
  let previous = bounds[0] - minGap;
  for (const item of order) {
    const placed = Math.max(item.value, previous + minGap);
    out[item.index] = placed;
    previous = placed;
  }
  // A downward pass, so a run pushed past the bottom edge comes back up rather
  // than piling against it.
  let ceiling = bounds[1] + minGap;
  for (const item of [...order].reverse()) {
    const placed = Math.min(out[item.index] as number, ceiling - minGap);
    out[item.index] = placed;
    ceiling = placed;
  }
  return out;
}

/** Level labels with the part they all share taken off the front and the back.
 *
 * The sweeps label their levels in full -- `0% dropped`, `5% dropped`, … and
 * `sigma=0 px`, `sigma=1 px`, … which is right in a table and unreadable as a
 * row of axis ticks, where the repeated word collides with its neighbours. The
 * shared part is not lost: each panel prints the knob's own name underneath.
 *
 * Only whole space-separated words are stripped, so `p=0.05` keeps its `p=` and
 * a set of labels with nothing in common is returned untouched. */
export function shortLevels(labels: readonly string[]): string[] {
  if (labels.length < 2) {
    return labels.slice();
  }
  const words = labels.map((item) => item.split(" "));
  let leading = 0;
  while (
    words.every((parts) => parts.length > leading + 1) &&
    words.every((parts) => parts[leading] === words[0]?.[leading])
  ) {
    leading += 1;
  }
  let trailing = 0;
  while (
    words.every((parts) => parts.length > leading + trailing + 1) &&
    words.every(
      (parts) =>
        parts[parts.length - 1 - trailing] ===
        (words[0] as string[])[(words[0] as string[]).length - 1 - trailing],
    )
  ) {
    trailing += 1;
  }
  return words.map((parts) =>
    parts.slice(leading, trailing === 0 ? undefined : parts.length - trailing).join(" "),
  );
}

/** Three significant figures, without exponent notation.
 *
 * Every rate on this page is measured against a labelled set of seventeen
 * crossings, so one event moves F1 by about 0.027. Printing 0.9142857142857143
 * claims a precision seventeen crossings cannot support.
 *
 * `null` prints as an em dash, and one benchmark row genuinely is null: at 2 fps
 * the engine reaches the gate with no identities at all, so the multiplicative
 * fold from one identity per vehicle is undefined rather than zero. Printing a
 * zero there would read as a perfect score. */
export function threeSigFigs(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "—";
  }
  if (value === 0) {
    return "0";
  }
  const text = value.toPrecision(3);
  return text.includes("e") ? String(Number(text)) : text;
}

function round(value: number): number {
  // Two decimals in viewBox units is well under a rendered pixel at every size
  // these figures are drawn at, and it keeps the emitted markup readable.
  return Math.round(value * 100) / 100;
}

// -- element helpers ----------------------------------------------------------

type Attrs = Record<string, string | number>;

function el<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  children: readonly (SVGElement | string)[] = [],
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs)) {
    node.setAttribute(name, String(value));
  }
  for (const child of children) {
    node.append(child);
  }
  return node;
}

/** A figure's root. `role="img"` plus a title and description, because a chart
 * that is only a picture is unreadable to anyone using a screen reader -- and
 * every figure here also has a table beside it carrying the same values. */
function figure(
  viewBox: string,
  title: string,
  description: string,
  children: readonly SVGElement[],
): SVGSVGElement {
  const titleId = `t-${slug(title)}`;
  const descriptionId = `d-${slug(title)}`;
  return el(
    "svg",
    {
      viewBox,
      role: "img",
      "aria-labelledby": `${titleId} ${descriptionId}`,
      preserveAspectRatio: "xMidYMid meet",
    },
    [
      el("title", { id: titleId }, [title]),
      el("desc", { id: descriptionId }, [description]),
      ...children,
    ],
  );
}

function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function label(x: number, y: number, text: string, className = "fig__label"): SVGTextElement {
  return el("text", { x: round(x), y: round(y), class: className }, [text]);
}

// -- figure one: how a crossing is decided ------------------------------------

/** The four cases, in one picture, at the geometry the engine actually uses.
 *
 * Frame samples are drawn as dots because the rule is about the STEP between two
 * of them: the engine tests whether the segment joining consecutive anchors meets
 * the gate segment. Draw only the smooth line and the reader sees a continuous
 * path crossing a line, which is not what is computed and hides both band
 * failures. */
export function crossingRuleDiagram(): SVGSVGElement {
  const width = 760;
  const height = 300;
  const plot = { left: 74, right: 736, top: 40, bottom: 250 };
  const zero = (plot.top + plot.bottom) / 2;
  const band = 15; // viewBox units standing for the +/- band, not pixels

  const marks: SVGElement[] = [];

  // Where the gate segment stops. Declared before the band, because the band is
  // a band around the SEGMENT and has to stop with it -- drawn to the full width
  // it would put the fourth case inside a band that does not reach it.
  const gateEnd = 560;

  // The band first: everything sits on top of it.
  marks.push(
    el("rect", {
      x: plot.left,
      y: zero - band,
      width: gateEnd - plot.left,
      height: band * 2,
      class: "fig__band",
    }),
  );
  marks.push(
    label(plot.left + 6, zero - band - 6, `± ${BASELINE_BAND_PX} px band`, "fig__note"),
  );

  // The gate. Solid across its span, hairline past its ends: the gate is a
  // SEGMENT, and the difference is a whole class of vehicle that is never
  // counted.
  marks.push(
    el("line", {
      x1: plot.left,
      y1: zero,
      x2: gateEnd,
      y2: zero,
      class: "fig__gate",
    }),
  );
  marks.push(
    el("line", {
      x1: gateEnd,
      y1: zero,
      x2: plot.right,
      y2: zero,
      class: "fig__gate fig__gate--beyond",
    }),
  );
  marks.push(el("circle", { cx: gateEnd, cy: zero, r: 4, class: "fig__gate-end" }));
  // Below the line, where nothing else is drawn between the third case and the
  // gate's end. Above it collides with the case that sits inside the band.
  marks.push(label(gateEnd - 6, zero + 26, "gate ends", "fig__note--right"));

  // The axis is named once, in words rather than tick values: this figure is
  // about geometry, and a number on it would invite reading it as a measurement.
  marks.push(label(plot.left, plot.top - 14, "one side of the gate", "fig__note"));
  marks.push(label(plot.left, plot.bottom + 20, "the other side", "fig__note"));
  marks.push(label(plot.right, plot.bottom + 20, "time →", "fig__note--right"));
  marks.push(
    el("line", {
      x1: plot.left,
      y1: plot.top,
      x2: plot.left,
      y2: plot.bottom,
      class: "fig__axis",
    }),
  );

  // Short tags in the art; the sentence each one abbreviates is in the caption
  // list under the figure. Long annotations inside an SVG cannot be measured
  // before they are drawn, so they collide with each other and run off the edge
  // at exactly the widths nobody tested -- and they are unreadable at phone
  // scale anyway, because a viewBox shrinks its text with everything else.
  const cases: {
    readonly samples: readonly Point[];
    readonly counted: boolean;
    readonly crossing?: Point;
    readonly tag: string;
    readonly tagAt: Point;
    readonly tagAnchor: "start" | "end";
  }[] = [
    {
      // Ordinary crossing: samples either side, one count, on the frame the step
      // met the line.
      samples: [
        { x: 100, y: zero - 84 },
        { x: 134, y: zero - 58 },
        { x: 168, y: zero - 28 },
        { x: 202, y: zero + 4 },
        { x: 236, y: zero + 36 },
        { x: 270, y: zero + 66 },
      ],
      counted: true,
      crossing: { x: 198, y: zero },
      tag: "counted",
      tagAt: { x: 100, y: zero - 96 },
      tagAnchor: "start",
    },
    {
      // Fast vehicle: one step clears the whole band, so no sample is ever inside
      // it. The band rule reports nothing; the gate rule counts it, because the
      // STEP crossed even though no sample was close.
      samples: [
        { x: 316, y: zero - 76 },
        { x: 352, y: zero - 26 },
        { x: 388, y: zero + 26 },
        { x: 424, y: zero + 76 },
      ],
      counted: true,
      crossing: { x: 370, y: zero },
      tag: "counted — the band misses it",
      tagAt: { x: 316, y: zero - 96 },
      tagAnchor: "start",
    },
    {
      // Slow vehicle inside the band, never changing side. The band rule fires on
      // every frame it lingers; the gate rule fires not at all, correctly.
      samples: [
        { x: 460, y: zero - 11 },
        { x: 480, y: zero - 7 },
        { x: 500, y: zero - 9 },
        { x: 520, y: zero - 5 },
        { x: 540, y: zero - 10 },
      ],
      counted: false,
      tag: "not counted — the band counts it five times",
      tagAt: { x: 456, y: zero - 42 },
      tagAnchor: "start",
    },
    {
      // Past the segment's end: crosses the gate's infinite LINE and is never
      // counted. This is the far carriageway, every second, on the real clip.
      samples: [
        { x: 596, y: zero + 78 },
        { x: 632, y: zero + 46 },
        { x: 668, y: zero + 12 },
        { x: 704, y: zero - 22 },
      ],
      counted: false,
      tag: "not counted — past the end",
      tagAt: { x: 736, y: zero + 100 },
      tagAnchor: "end",
    },
  ];

  for (const item of cases) {
    marks.push(
      el("path", {
        d: pathFrom(item.samples),
        class: item.counted ? "fig__trace fig__trace--counted" : "fig__trace",
      }),
    );
    for (const sample of item.samples) {
      marks.push(
        el("circle", { cx: sample.x, cy: sample.y, r: 3.4, class: "fig__sample" }),
      );
    }
    if (item.crossing !== undefined) {
      marks.push(
        el("path", {
          d: markerPath(item.crossing),
          class: "fig__crossing",
        }),
      );
    }
    marks.push(
      label(
        item.tagAt.x,
        item.tagAt.y,
        item.tag,
        item.tagAnchor === "end" ? "fig__note--right" : "fig__note",
      ),
    );
  }

  return figure(
    `0 0 ${width} ${height}`,
    "How a crossing is decided",
    "Four vehicle paths against one gate segment. A count fires where the step " +
      "between two frame samples meets the segment. A path that steps clean over " +
      "a band is still counted; a path that sits inside the band without changing " +
      "side is not; a path that crosses the gate's line past its end is not.",
    marks,
  );
}

/** A triangle on the gate line: direction is shape, so it survives greyscale,
 * colour blindness and a printout. The same mark the live diagram draws. */
function markerPath(at: Point): string {
  const size = 6;
  return (
    `M${round(at.x)} ${round(at.y + size)}` +
    ` L${round(at.x - size * 0.85)} ${round(at.y - size * 0.65)}` +
    ` L${round(at.x + size * 0.85)} ${round(at.y - size * 0.65)} Z`
  );
}

// -- figure two: the robustness family, as small multiples --------------------

export interface Series {
  readonly key: string;
  readonly label: string;
  readonly values: readonly number[];
  /** The engine's own tracker gets the accent; the baselines share the grey. */
  readonly emphasis: boolean;
  readonly dashed: boolean;
}

export interface Panel {
  readonly title: string;
  readonly knob: string;
  readonly levels: readonly string[];
  readonly series: readonly Series[];
}

const PANEL = { width: 300, height: 186 };
const PANEL_PLOT = { left: 34, right: 254, top: 16, bottom: 142 };

/** One panel of the family: crossing F1 against degradation level.
 *
 * The x axis is ORDINAL -- levels at equal spacing, in the order the sweep ran
 * them. The levels are not equally spaced in any unit (30, 25, 15, 10, 5, 2 fps;
 * 0, 1, 2, 4, 8 px), so drawing them on a linear axis would put a shape on the
 * curve that belongs to the sampling choice rather than to the engine.
 *
 * The hairline at the undegraded level is the panel's reference: every panel
 * starts from the same clean-footage score, so "did it hold up" is read as
 * distance from that line -- the same question, and the same geometry, as a
 * trajectory's distance from the gate. */
export function robustnessPanel(panel: Panel): SVGSVGElement {
  const x = linear([0, Math.max(1, panel.levels.length - 1)], [PANEL_PLOT.left, PANEL_PLOT.right]);
  const y = linear([0, 1], [PANEL_PLOT.bottom, PANEL_PLOT.top]);
  const marks: SVGElement[] = [];

  for (const value of [0, 0.25, 0.5, 0.75, 1]) {
    const at = y(value);
    marks.push(
      el("line", {
        x1: PANEL_PLOT.left,
        y1: at,
        x2: PANEL_PLOT.right,
        y2: at,
        class: "fig__grid",
      }),
    );
    if (value === 0 || value === 0.5 || value === 1) {
      marks.push(label(PANEL_PLOT.left - 6, at + 3, value.toFixed(1), "fig__tick"));
    }
  }

  const reference = panel.series.find((series) => series.emphasis)?.values[0];
  if (reference !== undefined) {
    marks.push(
      el("line", {
        x1: PANEL_PLOT.left,
        y1: y(reference),
        x2: PANEL_PLOT.right,
        y2: y(reference),
        class: "fig__reference",
      }),
    );
  }

  shortLevels(panel.levels).forEach((levelLabel, index) => {
    marks.push(
      label(x(index), PANEL_PLOT.bottom + 14, levelLabel, "fig__tick fig__tick--x"),
    );
  });

  const endLabelY = spreadLabels(
    panel.series.map((series) => y(series.values[series.values.length - 1] ?? 0)),
    9,
    [PANEL_PLOT.top, PANEL_PLOT.bottom],
  );

  panel.series.forEach((series, seriesIndex) => {
    const points = series.values.map((value, index) => ({ x: x(index), y: y(value) }));
    const classes = [
      "fig__series",
      series.emphasis ? "fig__series--emphasis" : "fig__series--quiet",
      series.dashed ? "fig__series--dashed" : "",
    ]
      .filter((part) => part !== "")
      .join(" ");
    marks.push(el("path", { d: pathFrom(points), class: classes }));
    points.forEach((point, index) => {
      const dot = el("circle", {
        cx: point.x,
        cy: point.y,
        r: 3.2,
        class: `fig__dot ${series.emphasis ? "fig__dot--emphasis" : "fig__dot--quiet"}`,
      });
      // A native tooltip, so a value is reachable by pointer as well as from the
      // table below. The table is the accessible twin; this is a convenience.
      dot.append(
        el("title", {}, [
          `${series.label} · ${panel.levels[index] ?? ""} · F1 ${threeSigFigs(
            series.values[index] ?? 0,
          )}`,
        ]),
      );
      marks.push(dot);
    });
    marks.push(
      label(
        PANEL_PLOT.right + 4,
        (endLabelY[seriesIndex] ?? 0) + 3,
        threeSigFigs(series.values[series.values.length - 1] ?? 0),
        `fig__end ${series.emphasis ? "fig__end--emphasis" : "fig__end--quiet"}`,
      ),
    );
  });

  marks.push(label(PANEL_PLOT.left, 8, panel.title, "fig__panel-title"));
  marks.push(label(PANEL_PLOT.left, PANEL.height - 6, panel.knob, "fig__note"));

  return figure(
    `0 0 ${PANEL.width} ${PANEL.height}`,
    `${panel.title}: crossing F1 by level`,
    `Crossing F1 for ${panel.series
      .map((series) => series.label)
      .join(", ")} across ${panel.levels.length} levels of ${panel.knob}. ` +
      `The hairline marks the undegraded score. Every value is in the table below.`,
    marks,
  );
}

// -- figure three: the scale survey's matched controls ------------------------

export interface ControlBand {
  readonly band: string;
  readonly spreadPercent: number;
  /** The candidate under test, as opposed to a control. */
  readonly candidate: boolean;
  /** The known-periodic line the method is checked against. */
  readonly positive: boolean;
}

/** Period spread per band, candidate against matched controls.
 *
 * What this picture shows, and it is important that the caption claims no more:
 * the candidate's period spread is the tightest of the bands that are not
 * known-periodic, and it is still an order of magnitude looser than the positive
 * control -- a line whose period IS known. The positive control is drawn as a
 * reference rule as well as a bar, so that gap is a distance rather than two
 * numbers to subtract.
 *
 * What this picture does NOT show is the indistinguishability itself. That rests
 * on a full-span comb correlation, which scores the asphalt control HIGHER than
 * the posts, and on the candidate's peaks repeatedly landing on the search-band
 * edge -- the signature of broadband noise rather than a line. Neither is a
 * spread, so neither is plotted, and the description says so. This description
 * previously read "the candidate sits among the controls", which is the opposite
 * of what its own bars draw: 9.10 per cent against 31.8, 22.8, 14.6 and 14.2 is
 * visibly the tightest of the five. A caption asserting what its picture denies
 * costs the reader more than no caption at all. */
export function matchedControlsChart(bands: readonly ControlBand[]): SVGSVGElement {
  // Sized generously in viewBox units on purpose. A viewBox scales its text with
  // its geometry, so a small coordinate system rendered wide prints 9-unit labels
  // at 27 rendered pixels -- and the longest row label here is thirty-four
  // characters, which then also runs off the left edge.
  const rowHeight = 34;
  const plot = { left: 300, right: 830, top: 30 };
  const width = 900;
  const height = plot.top + bands.length * rowHeight + 40;
  const worst = Math.max(...bands.map((band) => band.spreadPercent), 1);
  const x = linear([0, ceilTo(worst, 10)], [plot.left, plot.right]);
  const marks: SVGElement[] = [];

  for (let value = 0; value <= ceilTo(worst, 10); value += 10) {
    marks.push(
      el("line", {
        x1: x(value),
        y1: plot.top - 6,
        x2: x(value),
        y2: plot.top + bands.length * rowHeight,
        class: "fig__grid",
      }),
    );
    marks.push(
      label(x(value), plot.top + bands.length * rowHeight + 14, `${value}`, "fig__tick"),
    );
  }
  marks.push(
    label(
      plot.left,
      plot.top - 12,
      "spread of the measured period, per cent",
      "fig__note",
    ),
  );

  const positive = bands.find((band) => band.positive);
  if (positive !== undefined) {
    marks.push(
      el("line", {
        x1: x(positive.spreadPercent),
        y1: plot.top - 6,
        x2: x(positive.spreadPercent),
        y2: plot.top + bands.length * rowHeight,
        class: "fig__reference",
      }),
    );
  }

  bands.forEach((band, index) => {
    const top = plot.top + index * rowHeight + 8;
    const barHeight = 16;
    const bar = el("path", {
      d: barPath(plot.left, top, x(band.spreadPercent) - plot.left, barHeight, 4),
      class: `fig__bar ${band.candidate ? "fig__bar--emphasis" : "fig__bar--quiet"}`,
    });
    bar.append(
      el("title", {}, [`${band.band}: ${threeSigFigs(band.spreadPercent)} per cent`]),
    );
    marks.push(bar);
    marks.push(
      label(
        plot.left - 8,
        top + barHeight - 3,
        band.band,
        `fig__row ${band.candidate ? "fig__row--emphasis" : ""}`,
      ),
    );
    marks.push(
      label(
        x(band.spreadPercent) + 6,
        top + barHeight - 3,
        threeSigFigs(band.spreadPercent),
        "fig__end",
      ),
    );
  });

  return figure(
    `0 0 ${width} ${height}`,
    "Period spread: the guardrail candidate against matched controls",
    "Spread of the measured local period for the guardrail post band and for " +
      "the controls processed identically, plus a positive control whose period " +
      "is known. The candidate has the tightest spread of the bands that are not " +
      "known-periodic, and is still an order of magnitude looser than the " +
      "positive control, which the hairline marks. Spread is not what shows the " +
      "candidate to be indistinguishable from asphalt; the comb correlation and " +
      "the search-band-edge peaks are, and they are in the note beside this " +
      "figure rather than in it.",
    marks,
  );
}

function ceilTo(value: number, step: number): number {
  return Math.ceil(value / step) * step;
}
