/** The measured results, below the control room.
 *
 * The division of labour with `index.html` is deliberate and it is the rule this
 * module is built around:
 *
 *   **The markup carries the argument. This module carries the figures.**
 *
 * Every heading and every claim sentence is authored in the HTML, so the page
 * reads with JavaScript off and a crawler sees prose rather than an empty div.
 * Every NUMBER is rendered here, from `../generated/reports.ts`, which
 * `scripts/build_site_data.py` bakes out of `reports/*.json`. Nothing numeric is
 * typed into either surface. That is not tidiness: a figure typed beside a baked
 * table reads as protected when it is not, and this project has already caught
 * four figures restated where they had stopped being true.
 *
 * What changed, and why the shape of this module changed with it: the results
 * half read as an article. It is a dashboard now. Each section leads with a row
 * of stat tiles -- the same uppercase-label-over-big-mono-figure the control room
 * uses for a live count -- then the chart or the compact table, then a
 * single-line caption. The protocol strips, the report's own verdict prose, the
 * level-by-level tables and the caveats have not been deleted; they are behind a
 * disclosure per section. **The words moved. The numbers did not.** A visitor who
 * never opens one still gets every headline figure, and one who is checking gets
 * every condition it was measured under.
 *
 * Where a number could not be sourced from a report, it is absent rather than
 * approximated. The whole-loop backend timings are the case that came up: they
 * are measured in the visitor's own tab and reported by the badge at the top of
 * the page, and there is no report file holding them, so no results section
 * quotes one.
 *
 * Split three ways, and the seams are about who knows what: `kit.ts` decides how
 * a figure may look and knows nothing about any benchmark; `figures.ts` draws the
 * three static charts; `results-speed.ts` carries the two speed sections. What is
 * left here is the remaining sections and the mount. */

import { REPORTS } from "../generated/reports";
import { crossingRuleDiagram, robustnessPanel } from "./figures";
import type { Panel, Series } from "./figures";
import {
  count,
  countingResolution,
  disclosure,
  figures,
  fixed,
  fragmentationResolution,
  fragmentationResolutionNote,
  h,
  headlineFigure,
  legend,
  list,
  megabytes,
  p,
  percent,
  plate,
  protocol,
  rate,
  resolutionNote,
  scientific,
  signed,
  table,
  tiles,
} from "./kit";
import type { Cell, Child } from "./kit";
import { speedTierOneSection, speedTierTwoSection, usableAnchorCount } from "./results-speed";

// Re-exported because the scale survey's verdict table lives with the survey,
// in `results-speed.ts`, while the negatives card that reports "none found" is
// built here. One definition, reachable from both, and from the test that
// asserts an unclassifiable verdict throws rather than being counted as another
// failure.
export { usableAnchorCount };

// -- addressing the bake by name, which must not fail quietly -----------------
//
// Several claims on this page name one level of one sweep: the 2 fps collapse,
// the two ends of the jitter sweep, the dropout protocol. The bake addresses
// those by their label string, and a string lookup that misses returns
// `undefined` -- which the formatters then print as an em dash, UNDER an authored
// sentence that asserts the number. `tests/test_site_data_sync.py` cannot catch
// it either, because its coverage table reaches sweep entries by index.
//
// So every lookup by name is resolved once, here, at module load, and throws on
// absence. `mountResults` already throws on a missing slot for exactly this
// reason; a missing measurement is the same failure one layer down. The throw is
// what makes a rename loud: it fails the vitest suite (which imports this module)
// and it fails the page, rather than publishing a dash.

export function protocolNamed(name: string) {
  const found = REPORTS.robustness.protocols.find((item) => item.name === name);
  if (found === undefined) {
    throw new Error(
      `no robustness protocol named "${name}"; the bake carries ` +
        REPORTS.robustness.protocols.map((item) => item.name).join(", "),
    );
  }
  return found;
}

/** One counting method, addressed by its published name.
 *
 * The same rule as `protocolNamed`, one report across: `methods.find(...)` with
 * a `?? Number.NaN` fallback prints an em dash where a figure should be, and
 * every call site sits directly under a sentence or a tile that asserts the
 * figure. A rename must be loud. */
export function methodNamed(name: string) {
  const found = REPORTS.counting.methods.find((item) => item.method === name);
  if (found === undefined) {
    throw new Error(
      `no counting method named "${name}"; the bake carries ` +
        REPORTS.counting.methods.map((item) => item.method).join(", "),
    );
  }
  return found;
}

export function levelOf(protocolName: string, levelLabel: string) {
  const found = protocolNamed(protocolName);
  const entry = found.entries.find((item) => item.levelLabel === levelLabel);
  if (entry === undefined) {
    throw new Error(
      `robustness protocol "${protocolName}" has no level "${levelLabel}"; its ` +
        `levels are ${found.entries.map((item) => item.levelLabel).join(", ")}`,
    );
  }
  return entry;
}

/** Whether the association-floor ablation says the floor explains a protocol.
 *
 * Read as a verdict rather than as `doesNotExplain.includes(name)`: a renamed
 * protocol would make that `includes` false, and false there prints "the floor
 * explains it: yes" -- the exact opposite of the finding, silently. The two lists
 * must classify the protocol exactly once. */
export function floorExplains(name: string): boolean {
  const floor = REPORTS.robustness.associationFloor;
  const yes = (floor.explains as readonly string[]).includes(name);
  const no = (floor.doesNotExplain as readonly string[]).includes(name);
  if (yes === no) {
    throw new Error(
      `the association-floor ablation classifies "${name}" as neither explained ` +
        `nor unexplained (explains: ${floor.explains.join(", ")}; does not: ` +
        `${floor.doesNotExplain.join(", ")})`,
    );
  }
  return yes;
}

function floorGainFor(name: string): number {
  const row = REPORTS.robustness.associationFloor.largestGainByProtocol.find(
    (item) => item.name === name,
  );
  if (row === undefined) {
    throw new Error(`the association-floor ablation has no row for "${name}"`);
  }
  return row.gain;
}

const FRAME_RATE_FLOOR = levelOf("frame_rate", "2 fps");
const JITTER_CLEAN = levelOf("box_jitter", "sigma=0 px");
const JITTER_SIGMA_2 = levelOf("box_jitter", "sigma=2 px");
const DROPOUT = protocolNamed("detection_dropout");
const DROPOUT_FLOOR_GAIN = floorGainFor("detection_dropout");
const DROPOUT_EXPLAINED = floorExplains("detection_dropout");

/** How many distinct strings a list holds.
 *
 * Used where the page states a count of ONE -- one clip labelled, one gate
 * labelled. Those were typed as `"1"`, which is a figure inlined in TS prose
 * beside baked figures, and it read as protected when it was not: a second
 * labelled clip would have left the page still saying one. Counted from the
 * baked clip and gate names instead, so the row moves when the evidence does. */
export function distinct(values: readonly string[]): number {
  return new Set(values).size;
}

const USABLE_ANCHORS = usableAnchorCount(REPORTS.speedReal.anchorCandidates);

// -- the method and tracker vocabulary ----------------------------------------

const TRACKER_LABEL: Record<string, string> = {
  engine: "engine (Kalman + Hungarian)",
  centroid: "centroid baseline",
  "greedy-iou": "greedy-IoU baseline",
};

const RULE_LABEL: Record<string, string> = {
  gate: "the gate rule — the step's path must meet the segment",
  band: "the band rule — count when within a fixed distance of the line",
  "per-frame": "the per-frame rule — count on every frame the vehicle is near",
};

function trackerOf(method: string): string {
  return method.split("+")[0] ?? method;
}

function ruleOf(method: string): string {
  return method.split("+").slice(1).join("+");
}

/** One tracker's figure beside the two baselines', for a tile's note.
 *
 * The comparison is the point of every tile in the identity row, and a tile that
 * printed the engine's number alone would be the same defect the prose had: a
 * figure with nothing to read it against. */
function against(values: readonly number[]): string {
  return `baselines ${values.map((value) => rate(value)).join(" / ")}`;
}

// -- section: counting accuracy ----------------------------------------------

function countingSection(): readonly Child[] {
  const data = REPORTS.counting;
  const engine = methodNamed("engine+gate");
  const timing = new Map(data.timing.map((row) => [row.method, row.msPerFrame]));

  const rows = (rule: string) =>
    data.methods
      .filter((method) => ruleOf(method.method) === rule)
      .map((method): readonly Cell[] => {
        const tracker = trackerOf(method.method);
        const emphasis = tracker === "engine";
        return [
          { text: TRACKER_LABEL[tracker] ?? tracker, emphasis },
          { text: count(method.nPredicted), emphasis },
          { text: count(method.truePositives), emphasis },
          { text: rate(method.precision), emphasis },
          { text: rate(method.recall), emphasis },
          { text: rate(method.f1), emphasis },
          { text: fixed(timing.get(method.method) ?? null, 3), emphasis },
        ];
      });

  return [
    tiles([
      { label: "F1", value: rate(engine.f1), note: "engine, gate rule", lead: true },
      { label: "precision", value: rate(engine.precision) },
      { label: "recall", value: rate(engine.recall) },
      {
        label: "resolution",
        value: countingResolution(),
        note: "one event, on F1",
      },
    ]),
    tiles(
      [
        {
          label: "F1, certain labels only",
          value: rate(engine.certainOnlyF1),
          note: "the same crossings, adjudicated strictly",
        },
        {
          label: "labels adjudicated certain",
          value: `${count(data.labels.certain)} of ${count(data.labels.total)}`,
          note: `${count(data.labels.probable)} probable`,
        },
      ],
      "minor",
    ),
    table(
      `${count(data.labels.total)} hand-labelled crossings, one clip and one gate, scored one ` +
        `by one — upper bounds. Two methods closer together than ${resolutionNote()} differ by ` +
        `one event, not in quality. Protocol in the README.`,
      [
        { head: "tracker" },
        { head: "predicted", numeric: true },
        { head: "matched", numeric: true },
        { head: "precision", numeric: true },
        { head: "recall", numeric: true },
        { head: "F1", numeric: true },
        { head: "ms/frame", numeric: true },
      ],
      [],
      ["gate", "band", "per-frame"].map((rule) => ({
        label: RULE_LABEL[rule] ?? rule,
        rows: rows(rule),
      })),
    ),
    disclosure("Protocol, matching, the band sweep, and what these figures do not say", [
      protocol([
        data.clip,
        `${count(data.frames)} frames`,
        `${fixed(data.fps, 0)} fps`,
        `gate ${data.gate.name}`,
        `${data.labels.total} labelled crossings (${data.labels.certain} certain, ${data.labels.probable} probable)`,
        `${data.detector.model} at conf ${data.detector.confidence}, ${data.detector.imgsz} px`,
        `match window +${data.matchWindow.framesAfter}/−${data.matchWindow.framesBefore} frames`,
        resolutionNote(),
      ]),
      figures([
        [
          "the engine's tracker, against a baseline tracker's CPU",
          `${rate(data.engineCpuMultiple)}×`,
        ],
        ["one event, on precision", `± ${rate(data.resolution.oneEventPrecision)}`],
        ["class agreement on matched crossings, engine + gate", rate(engineClassConsistency())],
        [
          "greedy matching equalled maximum cardinality here",
          data.matching.greedyEqualsMaxCardinality ? "yes" : "no",
        ],
      ]),
      p(
        "Timing covers the tracker and the counting rule only. Detections are read from a " +
          "cache, so the detector's cost is excluded and is identical for every method by " +
          "construction.",
        "aside",
      ),
      p(data.matching.rule),
      p(data.matching.limitation),
      p(data.matchWindow.reason),
      table(
        `Band half-width swept with the ${REPORTS.counting.bandSweep.tracker} tracker. ` +
          `Median approach speed at the gate: ` +
          `${rate(REPORTS.counting.bandSweep.approachPxPerFrame)} px per frame.`,
        [
          { head: "half-width px", numeric: true },
          { head: "predicted", numeric: true },
          { head: "matched", numeric: true },
          { head: "miss rate", numeric: true },
          { head: "phantom rate", numeric: true },
          { head: "F1", numeric: true },
        ],
        REPORTS.counting.bandSweep.entries.map((entry) => [
          fixed(entry.bandPx, 0),
          count(entry.nPredicted),
          count(entry.truePositives),
          rate(entry.missRate),
          rate(entry.phantomRate),
          rate(entry.f1),
        ]),
      ),
      p(REPORTS.counting.bandSweep.note),
      p(REPORTS.counting.bandSweep.entriesNote),
      list(data.caveats),
    ]),
  ];
}

function engineClassConsistency(): number {
  return methodNamed("engine+gate").classConsistency;
}

// -- section: robustness ------------------------------------------------------

const PROTOCOL_TITLE: Record<string, string> = {
  frame_rate: "Frame rate",
  dropped_frames: "Dropped frames",
  detection_dropout: "Detection dropout",
  box_jitter: "Box jitter",
};

function robustnessSeries(entries: readonly { readonly engine: { readonly f1: number }; readonly centroid: { readonly f1: number }; readonly greedyIou: { readonly f1: number } }[]): readonly Series[] {
  return [
    {
      key: "engine",
      label: "engine",
      values: entries.map((entry) => entry.engine.f1),
      emphasis: true,
      dashed: false,
    },
    {
      key: "centroid",
      label: "centroid",
      values: entries.map((entry) => entry.centroid.f1),
      emphasis: false,
      dashed: false,
    },
    {
      key: "greedy-iou",
      label: "greedy-IoU",
      values: entries.map((entry) => entry.greedyIou.f1),
      emphasis: false,
      dashed: true,
    },
  ];
}

/** Levels where the three trackers differ AND the engine is the lowest of them.
 *
 * The intersection, not the raw list. `levelsWhereEngineLowest` counts a
 * three-way tie as "lowest" -- the benchmark says so explicitly, and it is the
 * right convention there, because it means "not beaten". Quoted on its own beside
 * a verdict that says "19 of 21", though, the raw 21 reads as a stronger claim
 * than the measurement makes. */
function engineLowestWhereTheyDiffer(): number {
  const separation = REPORTS.robustness.trackerSeparation;
  const differ = new Set<string>(separation.levelsWhereTrackersDiffer);
  return separation.levelsWhereEngineLowest.filter((level) => differ.has(level)).length;
}

/** How many of the swept levels actually degrade the input.
 *
 * `engineLeadsAnyDegradedLevel` is scoped to the degraded levels, so a tile
 * reporting it against `levelsMeasured` would state the finding over four rows
 * the flag never considered. The identity levels are the undegraded ones -- one
 * per protocol, the knob at its no-op setting -- so the difference is the
 * denominator the flag was computed under. It throws rather than printing a
 * nonsense count if the two ever stop being nested. */
export function degradedLevelCount(): number {
  const separation = REPORTS.robustness.trackerSeparation;
  const degraded = separation.levelsMeasured - separation.identityLevels.length;
  if (degraded <= 0) {
    throw new Error(
      `the sweep reports ${separation.levelsMeasured} levels and ` +
        `${separation.identityLevels.length} of them undegraded, which leaves no ` +
        `degraded level for the engine to lead at`,
    );
  }
  return degraded;
}

function robustnessSection(): readonly Child[] {
  const data = REPORTS.robustness;
  const panels: Panel[] = data.protocols.map((item) => ({
    title: PROTOCOL_TITLE[item.name] ?? item.name,
    knob: item.knob,
    levels: item.entries.map((entry) => entry.levelLabel),
    series: robustnessSeries(item.entries),
  }));

  const grid = h(
    "div",
    { class: "small-multiples" },
    panels.map((panel) => robustnessPanel(panel)),
  );

  const separation = data.trackerSeparation;

  return [
    plate(
      grid,
      `Crossing F1 against degradation level, one panel per protocol; the hairline is the ` +
        `undegraded score and each panel names its own knob. Levels sit at equal spacing ` +
        `because they are ordered steps of a sweep, not points on a scale. Two trackers ` +
        `closer together than ${resolutionNote()} differ by one event, not in quality.`,
      [legend(panels[0]?.series ?? [])],
    ),
    tiles([
      {
        label: "engine lowest",
        value: `${count(engineLowestWhereTheyDiffer())} of ${count(separation.levelsWhereTrackersDiffer.length)}`,
        note: "levels where the three trackers differ at all",
        lead: true,
      },
      {
        label: "levels where it leads",
        value: separation.engineLeadsAnyDegradedLevel ? "some" : "none",
        // The flag is about the DEGRADED levels, so the denominator has to be
        // too: `levelsMeasured` counts the undegraded rows in as well, and a
        // tile that read "none of 21" would be claiming over four levels this
        // measurement never looked at.
        note: `of ${count(degradedLevelCount())} degraded levels`,
      },
      {
        label: "widest F1 spread",
        value: rate(separation.maxF1Spread),
        note: "across the three trackers",
      },
      {
        label: "undegraded spread",
        value: rate(separation.maxIdentityF1Spread),
        note: `over ${count(separation.identityLevels.length)} identity levels`,
      },
    ]),
    associationFloorBlock(),
    disclosure("Protocol, the jitter calibration, every level, and what these figures do not say", [
      protocol([
        data.clip,
        `${data.protocols.length} protocols`,
        `${separation.levelsMeasured} levels`,
        `seed ${data.seed}`,
        "gate rule held fixed",
        `${data.labels.total} labelled crossings`,
        resolutionNote(),
      ]),
      p(separation.verdict, "verdict"),
      figures([
        ["levels measured", count(separation.levelsMeasured)],
        ["levels where the three trackers differ", count(separation.levelsWhereTrackersDiffer.length)],
        [
          "the engine is lowest on every undegraded level",
          separation.engineLowestOnEveryIdentityLevel ? "yes" : "no",
        ],
        ["measured box-width residual, std", `${rate(data.jitter.medianBoxWidthPx)} px median box width`],
        [
          "per-corner sigma equivalent to the measurement",
          `${rate(data.jitter.cornerSigmaEquivalentPx.minPx)} to ${rate(data.jitter.cornerSigmaEquivalentPx.maxPx)} px`,
        ],
        [
          `the sweep's top level (σ = ${fixed(data.jitter.stressAtMaxSigma.sigmaPx, 0)} px), as a multiple of that`,
          `${rate(data.jitter.stressAtMaxSigma.lowest)}× to ${rate(data.jitter.stressAtMaxSigma.highest)}×`,
        ],
        [
          "σ = 2 px, as a multiple of that",
          `${rate(data.jitter.stressAtSigma2.lowest)}× to ${rate(data.jitter.stressAtSigma2.highest)}×`,
        ],
      ]),
      p(data.jitter.note, "aside"),
      p(data.jitter.cornerSigmaEquivalentPx.method, "aside"),
      p(data.reduction.claim, "aside"),
      ...data.protocols.map((item) =>
        table(
          `${PROTOCOL_TITLE[item.name] ?? item.name} — knob ${item.knob}, ` +
            `${item.seeded ? `seeded ${data.seed}` : "deterministic, unseeded"}. ${item.note}`,
          [
            { head: "level" },
            { head: "tracker" },
            { head: "predicted", numeric: true },
            { head: "matched", numeric: true },
            { head: "precision", numeric: true },
            { head: "recall", numeric: true },
            { head: "F1", numeric: true },
            { head: "frames kept", numeric: true },
            { head: "window +", numeric: true },
          ],
          item.entries.flatMap((entry) =>
            (
              [
                ["engine", entry.engine],
                ["centroid", entry.centroid],
                ["greedy-IoU", entry.greedyIou],
              ] as const
            ).map((pair): readonly Cell[] => {
              const [name, scores] = pair;
              const emphasis = name === "engine";
              return [
                { text: entry.levelLabel, emphasis },
                { text: name, emphasis },
                { text: count(scores.nPredicted), emphasis },
                { text: count(scores.truePositives), emphasis },
                { text: rate(scores.precision), emphasis },
                { text: rate(scores.recall), emphasis },
                { text: rate(scores.f1), emphasis },
                { text: `${count(entry.framesKept)} / ${count(entry.framesTotal)}`, emphasis },
                { text: count(entry.windowWidenedBy), emphasis },
              ];
            }),
          ),
        ),
      ),
      list(data.caveats),
    ]),
  ];
}

/** The two-line takeaway that replaced the ablation's own paragraph.
 *
 * Every figure in it is counted from the bake rather than restated: the number
 * of protocols the floor explains, the number it does not, and the shipped floor
 * itself. The report's own verdict prose is still on the page, one disclosure
 * down -- this is the reading, not a replacement for the record. */
export function floorTakeaway(): string {
  const floor = REPORTS.robustness.associationFloor;
  const total = floor.explains.length + floor.doesNotExplain.length;
  return (
    `The floor explains ${count(floor.explains.length)} of ${count(total)} collapses. It does ` +
    `not explain ${floor.doesNotExplain.join(", ")}, which is a second and undiagnosed fault. ` +
    `On undegraded footage the two floors tie exactly; loosening it only costs F1 under ` +
    `detection dropout, and that single result is the whole argument for keeping ` +
    `${floor.shippedFloor}.`
  );
}

function associationFloorBlock(): HTMLElement {
  const floor = REPORTS.robustness.associationFloor;
  return h("div", { class: "block" }, [
    h("h3", {}, ["Is it the association floor that does it?"]),
    table(
      `Loosening the engine's IoU association floor from ${floor.shippedFloor} to ` +
        `${floor.floors[1]}, changing nothing else. The comparison value is ` +
        `${floor.comparisonFloorSource}`,
      [
        { head: "protocol" },
        { head: "worst level" },
        { head: "F1 recovered", numeric: true },
        { head: "over the threshold", numeric: true },
      ],
      floor.largestGainByProtocol.map((row): readonly Cell[] => {
        // Widened deliberately: `explains` is a literal tuple from the baked
        // `as const`, so `includes` would otherwise only accept a member of
        // itself -- and the question being asked is whether this protocol is one.
        const explained = (floor.explains as readonly string[]).includes(row.name);
        return [
          { text: PROTOCOL_TITLE[row.name] ?? row.name, emphasis: !explained },
          { text: row.levelLabel, emphasis: !explained },
          { text: rate(row.gain), emphasis: !explained },
          { text: row.gain >= floor.gainThreshold ? "yes" : "no", emphasis: !explained },
        ];
      }),
    ),
    p(floorTakeaway(), "verdict"),
    disclosure("The ablation level by level, including the undegraded rows", [
      figures([
        ["shipped floor", String(floor.shippedFloor)],
        ["gain that counts as an explanation", rate(floor.gainThreshold)],
        ["protocols the floor explains", floor.explains.join(", ")],
        ["protocols it does not explain", floor.doesNotExplain.join(", ")],
      ]),
      p(floor.heldFixed, "aside"),
      p(floor.verdict, "verdict"),
      ...floor.byProtocol.map((item) =>
        table(
          `${PROTOCOL_TITLE[item.name] ?? item.name} — crossing F1 at the shipped floor ` +
            `${floor.shippedFloor} against the looser ${floor.floors[1]}, with only ` +
            `Tracker(match_thresh=…) changed.`,
          [
            { head: "level" },
            { head: `F1 at ${floor.shippedFloor}`, numeric: true },
            { head: `F1 at ${floor.floors[1]}`, numeric: true },
            { head: "recovered", numeric: true },
            { head: "predicted, shipped → looser", numeric: true },
          ],
          item.entries.map((entry): readonly Cell[] => {
            const gain = entry.f1Loosened - entry.f1Shipped;
            const identity = entry === item.entries[0];
            return [
              { text: entry.levelLabel, emphasis: identity },
              { text: rate(entry.f1Shipped), emphasis: identity },
              { text: rate(entry.f1Loosened), emphasis: identity },
              { text: gain === 0 ? "nothing" : rate(gain), emphasis: identity },
              {
                text: `${count(entry.predictedShipped)} → ${count(entry.predictedLoosened)}`,
                emphasis: identity,
              },
            ];
          }),
        ),
      ),
    ]),
  ]);
}

// -- section: identity at the gate -------------------------------------------

/** One tracker's undegraded identity row, addressed by name.
 *
 * The same rule as `methodNamed`, one report across, and it matters more here
 * than it did in the prose it replaced: the identity tiles print the engine's
 * fragmentation and class agreement AS the section's headline figures, with the
 * baselines beside them. `find(...)?.fragmentationRatio ?? NaN` would print an em
 * dash in a tile the size of a hero number. A rename must be loud. */
export function cleanTracker(name: string) {
  const found = REPORTS.tracking.clean.trackers.find((row) => row.tracker === name);
  if (found === undefined) {
    throw new Error(
      `the identity benchmark's undegraded row has no tracker named "${name}"; it ` +
        `carries ${REPORTS.tracking.clean.trackers.map((row) => row.tracker).join(", ")}`,
    );
  }
  return found;
}

function trackingSection(): readonly Child[] {
  const data = REPORTS.tracking;
  const clean = data.clean.trackers;
  const engine = cleanTracker("engine");
  const baselines = clean.filter((row) => row.tracker !== "engine");

  return [
    tiles([
      {
        label: "fragmentation",
        value: rate(engine.fragmentationRatio),
        note: `engine — ${against(baselines.map((row) => row.fragmentationRatio))}`,
        lead: true,
      },
      {
        label: "class agreement",
        value: rate(engine.classConsistency),
        note: `engine — ${against(baselines.map((row) => row.classConsistency))}`,
      },
      {
        label: "resolution",
        value: fragmentationResolution(),
        note: "one identity, on fragmentation",
      },
    ]),
    table(
      `The ${data.clean.levelLabel} level, with the ${data.countingRule} rule held ` +
        `fixed. A fragmentation ratio of 1.0 is one predicted identity per labelled vehicle; ` +
        `identity deviation is the multiplicative fold from 1.0, so losing identities is not ` +
        `flattered against splitting them. ${fragmentationResolutionNote()}.`,
      [
        { head: "tracker" },
        { head: "identities at the gate", numeric: true },
        { head: "identities that crossed", numeric: true },
        { head: "fragmentation", numeric: true },
        { head: "identity deviation", numeric: true },
        { head: "crossing-id ratio", numeric: true },
        { head: "class agreement", numeric: true },
      ],
      clean.map((row): readonly Cell[] => {
        const emphasis = row.tracker === "engine";
        return [
          { text: TRACKER_LABEL[row.tracker] ?? row.tracker, emphasis },
          { text: count(row.gateRegionTrackIds), emphasis },
          { text: count(row.crossingTrackIds), emphasis },
          { text: rate(row.fragmentationRatio), emphasis },
          { text: rate(row.identityDeviation), emphasis },
          { text: rate(row.crossingIdRatio), emphasis },
          { text: rate(row.classConsistency), emphasis },
        ];
      }),
    ),
    claimsNotMadeBlock(),
    disclosure("Protocol, the metric definitions, every level, and what these figures do not say", [
      protocol([
        data.clip,
        `${count(data.frames)} frames`,
        `${data.labels.total} labelled vehicles`,
        `gate region ± ${fixed(data.gateRegion.halfWidthPx, 0)} px`,
        `${data.countingRule} rule held fixed`,
        `${data.separation.levelsMeasured} levels`,
        fragmentationResolutionNote(),
      ]),
      p(data.cleanDegeneracy.verdict, "verdict"),
      p(data.separation.verdict, "verdict"),
      p(data.agreement.verdict, "verdict"),
      figures([
        ["fragmentation spread on clean footage", rate(data.cleanDegeneracy.spread)],
        ["widest fragmentation spread under degradation", rate(data.separation.maxSpread)],
        ["levels where the trackers differ", count(data.separation.levelsWhereTrackersDiffer.length)],
        [
          "levels where the engine is furthest from one identity per vehicle",
          count(data.separation.levelsWhereEngineFurthest.length),
        ],
        [
          "levels where this metric and crossing F1 agree",
          `${count(data.agreement.levelsWhereTheyAgree.length)} of ${count(data.agreement.levelsWhereF1Separates.length)}`,
        ],
        ["levels where they disagree", data.agreement.levelsWhereTheyDisagree.join(", ")],
        ["fragmentation denominator", count(data.metricDefinitions.fragmentationRatio.denominator)],
        ["gate region half-width", `${fixed(data.gateRegion.halfWidthPx, 0)} px`],
        ["half-widths swept", data.gateRegionSweep.halfWidthsPx.map((value) => fixed(value, 0)).join(", ")],
      ]),
      p(data.metricDefinitions.fragmentationRatio.definition),
      p(data.metricDefinitions.fragmentationRatio.notAClaim),
      p(data.metricDefinitions.identityDeviation.definition),
      p(data.metricDefinitions.classConsistency.definition),
      p(data.gateRegion.definition),
      p(data.gateRegion.sourceOfTheHalfWidth),
      p(data.gateRegionSweep.why),
      p(data.gateRegionSweep.invariants),
      p(data.countingRuleNote),
      p(data.separation.furthestTieRule),
      p(data.agreement.criterion),
      ...data.protocols.map((item) =>
        table(
          `${PROTOCOL_TITLE[item.name] ?? item.name} — knob ${item.knob}.`,
          [
            { head: "level" },
            { head: "tracker" },
            { head: "fragmentation", numeric: true },
            { head: "identity deviation", numeric: true },
          ],
          item.entries.flatMap((entry) =>
            entry.trackers.map((row): readonly Cell[] => {
              const emphasis = row.tracker === "engine";
              return [
                { text: entry.levelLabel, emphasis },
                { text: row.tracker, emphasis },
                { text: rate(row.fragmentationRatio), emphasis },
                { text: rate(row.identityDeviation), emphasis },
              ];
            }),
          ),
        ),
      ),
      list(data.caveats),
    ]),
  ];
}

/** The refusals, headline visible and reason on expand.
 *
 * One disclosure per claim rather than one holding all four: the claim itself is
 * the thing that must be read without asking, and burying four headlines behind
 * a single summary would hide the refusals rather than compress them. Word for
 * word from the benchmark's own record -- paraphrasing a refusal is how a refusal
 * softens. */
function claimsNotMadeBlock(): HTMLElement {
  return h("div", { class: "block" }, [
    h("h3", {}, ["What is not claimed"]),
    h(
      "div",
      { class: "claims" },
      REPORTS.tracking.claimsNotMade.map((item) => disclosure(item.claim, [p(item.reason)])),
    ),
  ]);
}

// -- section: the honest negatives -------------------------------------------
//
// The claim sentence is authored in the markup, one line per card. What lands
// here is the measurement behind it: the FIRST pair of each list is the card's
// headline figure, drawn large; the rest sit in the card's own disclosure. The
// order is therefore load-bearing, which is why `mountResults` reads it by
// position and `results.test.ts` reads the rows it pins by their term.

export function negativeFigures(): Record<string, readonly (readonly [string, string])[]> {
  const counting = REPORTS.counting;
  const robustness = REPORTS.robustness;
  const speedReal = REPORTS.speedReal;
  const model = REPORTS.model;

  const dropoutFirstEntry = DROPOUT.entries[0];
  const dropoutLastEntry = DROPOUT.entries[DROPOUT.entries.length - 1];
  if (dropoutFirstEntry === undefined || dropoutLastEntry === undefined) {
    throw new Error("the detection-dropout sweep has no levels");
  }
  const bandRows = counting.methods.filter((method) => method.method.endsWith("+band"));
  const perFrame = counting.methods.filter((method) => method.method.endsWith("+per-frame"));
  const engineGate = methodNamed("engine+gate");

  return {
    "negative-tracker": [
      ["engine F1, undegraded", rate(engineGate.f1)],
      [
        "the two baselines, undegraded",
        counting.methods
          .filter((method) => method.method.endsWith("+gate") && trackerOf(method.method) !== "engine")
          .map((method) => rate(method.f1))
          .join(" and "),
      ],
      [
        "levels where the three differ, and the engine is lowest",
        `${count(engineLowestWhereTheyDiffer())} of ${count(robustness.trackerSeparation.levelsWhereTrackersDiffer.length)}`,
      ],
      [
        "levels where it leads",
        robustness.trackerSeparation.engineLeadsAnyDegradedLevel ? "some" : "none",
      ],
      ["its CPU, against a baseline tracker's", `${rate(counting.engineCpuMultiple)}×`],
      ["resolution", resolutionNote()],
    ],
    "negative-collapse": [
      [`engine F1 at ${FRAME_RATE_FLOOR.levelLabel}`, rate(FRAME_RATE_FLOOR.engine.f1)],
      [
        "predictions it made there",
        `${count(FRAME_RATE_FLOOR.engine.nPredicted)} against ${count(REPORTS.robustness.labels.total)} labels`,
      ],
      [`engine F1 at ${JITTER_CLEAN.levelLabel}`, rate(JITTER_CLEAN.engine.f1)],
      [`engine F1 at ${JITTER_SIGMA_2.levelLabel}`, rate(JITTER_SIGMA_2.engine.f1)],
      [
        "σ = 2 px, as a multiple of the measured detector noise",
        `${rate(robustness.jitter.stressAtSigma2.lowest)}× to ${rate(robustness.jitter.stressAtSigma2.highest)}×`,
      ],
    ],
    "negative-dropout": [
      ["F1 the looser association floor recovers here", rate(DROPOUT_FLOOR_GAIN)],
      ["so the floor explains it", DROPOUT_EXPLAINED ? "yes" : "no"],
      [
        `engine F1 at ${dropoutFirstEntry.levelLabel}`,
        rate(dropoutFirstEntry.engine.f1),
      ],
      [
        `engine F1 at ${dropoutLastEntry.levelLabel}`,
        rate(dropoutLastEntry.engine.f1),
      ],
      [
        "every step in between",
        DROPOUT.entries.map((entry) => rate(entry.engine.f1)).join(" → "),
      ],
    ],
    "negative-floor": [
      ["most F1 the looser floor recovers", rate(robustness.associationFloor.largestGain.gain)],
      ["where it recovers it", robustness.associationFloor.largestGain.levelLabel],
      ["shipped association floor", String(robustness.associationFloor.shippedFloor)],
      ["floor compared against", String(robustness.associationFloor.floors[1])],
      [
        "what it recovers on undegraded footage",
        identityRowGains()
          .map((gain) => (gain === 0 ? "nothing" : rate(gain)))
          .join(", "),
      ],
      ["protocols the floor explains", robustness.associationFloor.explains.join(", ")],
      ["and does not", robustness.associationFloor.doesNotExplain.join(", ")],
    ],
    "negative-int8": [
      ["detections lost", percent(model.detectionsLostFraction)],
      ["download saved", megabytes(model.bytesSaved)],
      [
        "boxes that survive are placed well",
        `mean IoU ${rate(model.meanIouOfMatched.int8 ?? Number.NaN)}`,
      ],
      [
        "protocol",
        `${model.sampledFrames} frames sampled every ${model.sampleStride}th, conf ${model.confidence}, NMS IoU ${model.nmsIou}`,
      ],
    ],
    "negative-band": [
      ["band rule F1, engine tracker", rate(methodNamed("engine+band").f1)],
      [
        // The label count comes from the bake even inside a term string. It was
        // typed here as `17` two lines from the baked value it restates, which is
        // exactly the drift the bake exists to prevent -- and it read as
        // protected when nothing was watching it.
        `band rule predictions against ${count(counting.labels.total)} labels`,
        bandRows.map((method) => count(method.nPredicted)).join(", "),
      ],
      [
        "of those, landing on the right frame",
        bandRows.map((method) => count(method.truePositives)).join(", "),
      ],
      [
        "per-frame rule over-count",
        perFrame.map((method) => count(method.countError)).join(", "),
      ],
      ["and that over-count is a", "LOWER bound — a vehicle stopped on the gate emits nothing"],
    ],
    "negative-one-clip": [
      // Counted from the bake, not typed. Every labelled benchmark on this page
      // names its own clip and the counting benchmark names its own gate, so a
      // second clip or a second gate moves these rows by itself instead of
      // leaving the page asserting one.
      ["labelled crossings, on one clip and one gate", count(counting.labels.total)],
      ["clips labelled", count(distinct([counting.clip, robustness.clip, REPORTS.tracking.clip]))],
      ["gates labelled", count(distinct([counting.gate.name]))],
      [
        "of which adjudicated as certain",
        `${count(counting.labels.certain)}; ${count(counting.labels.probable)} probable`,
      ],
      ["so every accuracy figure here is an", "UPPER bound"],
    ],
    "negative-scale": [
      [
        "the honest bracket, propagated to every speed",
        `${signed(speedReal.bracket.bandPercent[0] ?? 0)} % to ${signed(speedReal.bracket.bandPercent[1] ?? 0)} %`,
      ],
      ["anchor candidates searched for", count(speedReal.anchorCandidates.length)],
      ["usable ones found", count(USABLE_ANCHORS)],
      [
        "honest bracket on the along-road scale",
        `${fixed(speedReal.bracket.lowerM, 1)} m to ${fixed(speedReal.bracket.upperM, 1)} m`,
      ],
      ["km/h published from this clip", speedReal.absoluteSpeedPublished ? "yes" : "none"],
    ],
  };
}

/** What the looser association floor recovers at each protocol's IDENTITY level.
 *
 * The undegraded row is the control on the ablation: if loosening the floor
 * helped there too, the finding would be "0.3 is a better default" rather than
 * "0.8 is narrow". As measured it does neither -- the two floors tie exactly at
 * every identity level -- so the tie is not what argues for keeping 0.8. The
 * only measurement that does is the detection-dropout row, where loosening
 * costs F1. */
function identityRowGains(): readonly number[] {
  return REPORTS.robustness.associationFloor.byProtocol.map((item) => {
    const identity = item.entries[0];
    return identity === undefined ? Number.NaN : identity.f1Loosened - identity.f1Shipped;
  });
}

// -- section: architecture ----------------------------------------------------

function architectureSection(): readonly Child[] {
  const parity = REPORTS.parity;
  const noise = REPORTS.detectionNoise;
  const model = REPORTS.model;

  return [
    table(
      "The cross-surface fixtures. Written by the Python engine, replayed through the " +
        "browser engine in the visitor's own tab at ?selftest=1, so a green verdict is about " +
        "the artefact that is actually serving.",
      [{ head: "what is pinned" }, { head: "value", numeric: true }],
      [
        ["committed cases", count(parity.caseCount)],
        ["boundary kinds each case must include", count(parity.straddleKinds.length)],
        ["speed agreement tolerance, km/h", scientific(parity.speedToleranceKmh)],
        ["association floor the fixtures straddle", rate(parity.iouStraddle.matchThresh)],
        ["the straddling IoU, and its control one ulp below", `${parity.iouStraddle.iou} / ${parity.iouStraddle.controlIou}`],
        ["Mahalanobis gate the fixtures straddle", rate(parity.iouStraddle.gatingChi2)],
        ["frames replayed from the real clip", count(parity.realClip.frames)],
        ["detections in them", count(parity.realClip.detections)],
        ["track identities allocated", count(parity.realClip.tracksAllocated)],
        ["crossings emitted", count(parity.realClip.events)],
      ],
    ),
    table(
      `The detector the page downloads, and the one it refused. Both graphs decoded through ` +
        `identical letterboxing and class-wise NMS at confidence ${model.confidence} and IoU ` +
        `${model.nmsIou}, over ${model.sampledFrames} frames sampled every ` +
        `${model.sampleStride}th from the motorway clip.`,
      [
        { head: "" },
        { head: "float32, shipped", numeric: true },
        { head: "int8 dynamic, refused", numeric: true },
      ],
      [
        [
          "file size",
          count(model.fileSizeBytes.float32 ?? Number.NaN),
          count(model.fileSizeBytes.int8 ?? Number.NaN),
        ],
        [
          "detections over the sample",
          count(model.detections.float32 ?? Number.NaN),
          count(model.detections.int8 ?? Number.NaN),
        ],
        ["recall against float32", "—", rate(model.recallAgainstFloat32.int8 ?? Number.NaN)],
        ["mean IoU of matched boxes", "—", rate(model.meanIouOfMatched.int8 ?? Number.NaN)],
      ],
    ),
    disclosure("Protocol, and the detector noise the robustness sweep is calibrated against", [
      protocol([
        "one core, two implementations",
        `${parity.caseCount} committed parity cases`,
        `${parity.straddleKinds.length} boundary kinds`,
        `speeds agree to ${scientific(parity.speedToleranceKmh)} km/h`,
        `crossing decisions agree exactly`,
      ]),
      p(
        `int8 saves ${megabytes(model.bytesSaved)} of download and loses ` +
          `${percent(model.detectionsLostFraction)} of detections. In a counting product a ` +
          `missed detection is a missed count, so the trade is refused at any download size.`,
        "verdict",
      ),
      p(noise.caveat, "aside"),
      p(noise.association, "aside"),
      table(
        `Residuals against a ${noise.medianFilterFrames}-frame median over ` +
          `${count(noise.frames)} frames: ${count(noise.tracksContributing)} of ` +
          `${count(noise.tracksSeen)} tracks contributed. Median box on this clip is ` +
          `${rate(noise.medianBoxWidthPx)} × ${rate(noise.medianBoxHeightPx)} px.`,
        [
          { head: "quantity" },
          { head: "samples", numeric: true },
          { head: "std px", numeric: true },
          { head: "mean abs px", numeric: true },
          { head: "p95 abs px", numeric: true },
        ],
        (
          [
            ["box width", noise.residuals.boxWidth],
            ["box height", noise.residuals.boxHeight],
            ["centre x", noise.residuals.centreX],
            ["centre y", noise.residuals.centreY],
          ] as const
        ).map(([name, row]) => [
          name,
          count(row.n),
          rate(row.stdPx),
          rate(row.maePx),
          rate(row.p95AbsPx),
        ]),
      ),
    ]),
  ];
}

// -- section: the reference ---------------------------------------------------
//
// Command names are checked against the real CLI by
// `tests/test_reference_matches_the_cli.py`, in both directions: a command the
// page invents fails, and a command the CLI grows without being listed fails too.

interface Command {
  readonly name: string;
  readonly what: string;
  readonly implemented: boolean;
}

export const COMMANDS: readonly Command[] = [
  {
    name: "run",
    what: "Analyse a video source and report what crossed each gate.",
    implemented: true,
  },
  {
    name: "calibrate",
    what: "Explain how to survey a camera, and check a config's road plane against its own holdout.",
    implemented: true,
  },
  {
    name: "export-model",
    what: "Export a YOLO11 checkpoint to ONNX at a fixed input size, for the browser engine.",
    implemented: true,
  },
  {
    name: "fetch-samples",
    what: "Download the Creative Commons sample clips this project measures on.",
    implemented: true,
  },
  { name: "serve", what: "Serve the browser dashboard.", implemented: false },
  { name: "bench", what: "Point at the benchmark harness.", implemented: false },
];

interface ApiEntry {
  readonly python: string;
  readonly typescript: string;
  readonly what: string;
}

export const API: readonly ApiEntry[] = [
  {
    python: "trafficlens.core.gate.Gate",
    typescript: "engine/gate.ts — Gate",
    what: "A directed line segment. Tests whether a step's path meets it, and on which side it ended.",
  },
  {
    python: "trafficlens.core.gate.GateCounter",
    typescript: "engine/gate.ts — GateCounter",
    what: "Counts each identity once, by class and by direction, and never again however long it lingers.",
  },
  {
    python: "trafficlens.core.homography.RoadPlane",
    typescript: "engine/homography.ts — RoadPlane",
    what: "Image to road-plane metres, fitted from surveyed correspondences with a holdout.",
  },
  {
    python: "trafficlens.track.tracker.Tracker",
    typescript: "engine/tracker.ts — Tracker",
    what: "Constant-velocity Kalman prediction, Mahalanobis gating, then an IoU association floor.",
  },
  {
    // Not the same shape on both sides, and the difference is real rather than an
    // oversight: Python runs a whole clip in one call and returns a result, while
    // the browser is handed one frame at a time by a render loop it does not own.
    // The parity fixtures compare them per step, which is what makes the two
    // comparable despite that.
    python: "trafficlens.pipeline.run_session",
    typescript: "engine/pipeline.ts — SessionPipeline",
    what: "Detections in, tracks and crossing events out. The path both surfaces are compared through, step by step.",
  },
  {
    python: "trafficlens.core.constants",
    typescript: "generated/constants.ts",
    what: "Every tunable both engines read. The TypeScript side is generated from the Python and never written by hand.",
  },
];

function referenceSection(): readonly Child[] {
  return [
    table(
      "The command line. Two commands are placeholders and say so here as well as in --help.",
      [{ head: "command" }, { head: "what it does" }, { head: "state" }],
      COMMANDS.map((command) => [
        `trafficlens ${command.name}`,
        command.what,
        command.implemented ? "works" : "not built",
      ]),
    ),
    table(
      "The objects worth reaching for, and their mirrors in the browser engine. Both sides " +
        "are compared against each other by the committed parity fixtures rather than trusted " +
        "separately.",
      [{ head: "Python" }, { head: "TypeScript" }, { head: "what it is" }],
      API.map((entry) => [entry.python, entry.typescript, entry.what]),
    ),
  ];
}

// -- mounting -----------------------------------------------------------------

const SECTIONS: Record<string, () => readonly Child[]> = {
  "rule-diagram": () => [
    plate(
      crossingRuleDiagram(),
      "One gate segment and four vehicle paths: a count fires where the step between two frame " +
        "samples meets the segment, which is why a path that clears the band in one step is " +
        "still counted, one that sits inside the band is not, and one crossing the gate's line " +
        "past its end never is.",
    ),
  ],
  counting: countingSection,
  robustness: robustnessSection,
  tracking: trackingSection,
  "speed-tier-1": speedTierOneSection,
  "speed-tier-2": speedTierTwoSection,
  architecture: architectureSection,
  reference: referenceSection,
};

/** Every `data-results` slot this module fills.
 *
 * Exported because the slots and this module are two halves of one document and
 * today they agree only because they were written together. `mountResults` throws
 * on a missing slot, but nothing compared the two SETS -- and the in-page selftest
 * returns before the results mount, so the headless check could not catch a
 * mismatch either. `results.test.ts` reads the authored markup and compares. */
export function slotNames(): readonly string[] {
  return [
    ...Object.keys(SECTIONS),
    ...Object.keys(negativeFigures()),
    "negative-tracking",
  ];
}

function slot(name: string): HTMLElement {
  const found = document.querySelector<HTMLElement>(`[data-results="${name}"]`);
  if (found === null) {
    throw new Error(`missing results slot [data-results="${name}"]`);
  }
  return found;
}

/** One honest-negative card's body: the headline figure, then the rest.
 *
 * The first pair is the card's face. The remainder is not cut -- it is the
 * evidence the claim rests on -- so it sits in a disclosure on the same card,
 * counted in the summary so a reader knows how much is there before opening it. */
function findingBody(items: readonly (readonly [string, string])[]): readonly Child[] {
  const head = items[0];
  if (head === undefined) {
    throw new Error("an honest-negative card was given no figures at all");
  }
  const rest = items.slice(1);
  return [
    headlineFigure(head[0], head[1]),
    ...(rest.length === 0
      ? []
      : [disclosure(`every figure (${count(items.length)})`, [figures(rest)])]),
  ];
}

/** Fill every results slot in the page.
 *
 * Throws on a missing slot rather than skipping it: the slots and this module are
 * two halves of one document, and a section that silently rendered nothing would
 * leave an authored heading standing over no evidence at all -- which on this
 * page is worse than an error. */
export function mountResults(): void {
  for (const [name, build] of Object.entries(SECTIONS)) {
    slot(name).replaceChildren(...build());
  }
  for (const [name, items] of Object.entries(negativeFigures())) {
    slot(name).replaceChildren(...findingBody(items));
  }
  const claims = REPORTS.tracking.claimsNotMade;
  slot("negative-tracking").replaceChildren(
    headlineFigure("identity claims this project does not make", count(claims.length)),
    disclosure(
      `every claim (${count(claims.length)})`,
      // Word for word from the benchmark's own record, as a list rather than a
      // figure run: they are sentences, and a figure run's value column would
      // squeeze the term column to nothing and break it one letter per line.
      [list(claims.map((item) => item.claim))],
    ),
  );
}
