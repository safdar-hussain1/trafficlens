/** The measured results, below the control room.
 *
 * The division of labour with `index.html` is deliberate and it is the rule this
 * module is built around:
 *
 *   **The markup carries the argument. This module carries the figures.**
 *
 * Every heading, every lead paragraph and every claim sentence is authored in the
 * HTML, so the page reads with JavaScript off and a crawler sees prose rather
 * than an empty div. Every NUMBER is rendered here, from
 * `../generated/reports.ts`, which `scripts/build_site_data.py` bakes out of
 * `reports/*.json`. Nothing numeric is typed into either surface. That is not
 * tidiness: a figure typed beside a baked table reads as protected when it is
 * not, and this project has already caught four figures restated where they had
 * stopped being true.
 *
 * The other rule the sections are built to: **every figure carries its protocol
 * within one glance.** Each measured section opens with a protocol strip -- clip,
 * frames, rate, gate, label count, detector, match window, resolution -- in mono,
 * directly under the heading and above the numbers it governs. It costs a line
 * per section and it is the reason a reader can trust a rate on this page: the
 * conditions are not in a footnote, they are in the same eyeful.
 *
 * Where a number could not be sourced from a report, it is absent rather than
 * approximated. The whole-loop backend timings are the case that came up: they
 * are measured in the visitor's own tab and reported by the badge at the top of
 * the page, and there is no report file holding them, so no results section
 * quotes one.
 *
 * Split three ways, and the seams are about who knows what: `kit.ts` decides how
 * a figure may look and knows nothing about any benchmark; `figures.ts` draws the
 * three static charts; `results-speed.ts` carries the two speed sections, the
 * longest pair and the pair most likely to be read line by line. What is left
 * here is the remaining sections and the mount. */

import { REPORTS } from "../generated/reports";
import { crossingRuleDiagram, robustnessPanel } from "./figures";
import type { Panel, Series } from "./figures";
import {
  count,
  disclosure,
  figures,
  fixed,
  fragmentationResolutionNote,
  h,
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
} from "./kit";
import type { Cell, Child } from "./kit";
import { speedTierOneSection, speedTierTwoSection } from "./results-speed";

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
 * both call sites sit directly under a sentence that asserts the figure. A
 * rename must be loud. */
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

/** Verdicts the scale survey uses, and which of them would license a scale.
 *
 * The negatives section states that none of the five anchor candidates was
 * usable. That was typed as `"0"`. It is computed here from the verdict the
 * survey itself recorded for each candidate, against a table that names the
 * verdict which WOULD count -- so the row is not a constant, and a verdict this
 * page has never seen refuses to be classified rather than being quietly counted
 * as another failure. */
const ANCHOR_VERDICT_LICENSES_A_SCALE: Record<string, boolean> = {
  ABSENT: false,
  "PRESENT BUT NOT MEASURABLE": false,
  "PRESENT AND MEASURABLE BUT UNUSABLE": false,
  "PRESENT AND MEASURABLE AND CONTRADICTORY": false,
  "PRESENT AND MEASURABLE AND CONSISTENT": true,
};

export function usableAnchorCount(candidates: readonly { readonly verdict: string }[]): number {
  return candidates.filter((candidate) => {
    const licensed = ANCHOR_VERDICT_LICENSES_A_SCALE[candidate.verdict];
    if (licensed === undefined) {
      throw new Error(
        `the scale survey recorded a verdict this page cannot classify: ` +
          `"${candidate.verdict}"`,
      );
    }
    return licensed;
  }).length;
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

// -- section: counting accuracy ----------------------------------------------

function countingSection(): readonly Child[] {
  const data = REPORTS.counting;
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
    table(
      `Crossings scored one by one against ${data.labels.total} labels. ` +
        `Two methods closer together than ${resolutionNote()} differ by one event, not in quality.`,
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
    figures([
      [
        "the engine's tracker, against a baseline tracker's CPU",
        `${rate(data.engineCpuMultiple)}×`,
      ],
      ["one event, on precision", `± ${rate(data.resolution.oneEventPrecision)}`],
      ["one event, on F1", `± ${rate(data.resolution.oneEventF1)}`],
      ["class agreement on matched crossings, engine + gate", rate(engineClassConsistency())],
    ]),
    p(
      "Timing covers the tracker and the counting rule only. Detections are read from a cache, " +
        "so the detector's cost — by far the largest per-frame cost in a real session — is " +
        "excluded, and is identical for every method by construction.",
      "aside",
    ),
    disclosure("How the matching works, and where it is not optimal", [
      p(data.matching.rule),
      p(data.matching.limitation),
      figures([
        [
          "greedy matching equalled maximum cardinality here",
          data.matching.greedyEqualsMaxCardinality ? "yes" : "no",
        ],
        ["match window", `+${data.matchWindow.framesAfter}/−${data.matchWindow.framesBefore} frames`],
      ]),
      p(data.matchWindow.reason),
    ]),
    disclosure(`What these figures do not say (${data.caveats.length})`, [list(data.caveats)]),
    disclosure("The band rule across every width tried", [
      p(REPORTS.counting.bandSweep.note),
      p(REPORTS.counting.bandSweep.entriesNote),
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
    protocol([
      data.clip,
      `${data.protocols.length} protocols`,
      `${separation.levelsMeasured} levels`,
      `seed ${data.seed}`,
      "gate rule held fixed",
      `${data.labels.total} labelled crossings`,
      resolutionNote(),
    ]),
    plate(
      grid,
      "Crossing F1 against degradation level, one panel per protocol. Levels sit at equal " +
        "spacing because they are ordered steps of the sweep, not points on a linear scale; " +
        "each panel names its own knob underneath. The hairline is the undegraded score. Every " +
        "value is in the level-by-level table further down.",
      [legend(panels[0]?.series ?? [])],
    ),
    p(separation.verdict, "verdict"),
    figures([
      ["levels measured", count(separation.levelsMeasured)],
      ["levels where the three trackers differ", count(separation.levelsWhereTrackersDiffer.length)],
      [
        "of those, levels where the engine scores lowest",
        `${count(engineLowestWhereTheyDiffer())} of ${count(separation.levelsWhereTrackersDiffer.length)}`,
      ],
      ["the engine leads at any degraded level", separation.engineLeadsAnyDegradedLevel ? "yes" : "no"],
      ["widest F1 spread across trackers", rate(separation.maxF1Spread)],
      [
        "widest spread on undegraded footage",
        `${rate(separation.maxIdentityF1Spread)} (${separation.identityLevels.length} identity levels)`,
      ],
      [
        "the engine is lowest on every undegraded level",
        separation.engineLowestOnEveryIdentityLevel ? "yes" : "no",
      ],
    ]),
    p(data.reduction.claim, "aside"),
    associationFloorBlock(),
    jitterBlock(),
    disclosure("Every level, every tracker", [
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
    ]),
    disclosure(`What these figures do not say (${data.caveats.length})`, [list(data.caveats)]),
  ];
}

function associationFloorBlock(): HTMLElement {
  const floor = REPORTS.robustness.associationFloor;
  return h("div", { class: "block" }, [
    h("h3", {}, ["Is the association floor what does it?"]),
    p(floor.heldFixed, "aside"),
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
    p(floor.verdict, "verdict"),
    figures([
      ["shipped floor", String(floor.shippedFloor)],
      ["gain that counts as an explanation", rate(floor.gainThreshold)],
      ["protocols the floor explains", floor.explains.join(", ")],
      ["protocols it does not explain", floor.doesNotExplain.join(", ")],
    ]),
    disclosure("The ablation level by level, including the undegraded rows", [
      p(
        "The undegraded rows are the control on this ablation: where the input has not been " +
          "degraded at all, loosening the floor recovers nothing — the two tie exactly. What " +
          "stops this being a straight argument for the looser floor is the detection-dropout " +
          "rows above, where loosening it costs F1.",
        "aside",
      ),
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

function jitterBlock(): HTMLElement {
  const jitter = REPORTS.robustness.jitter;
  return h("div", { class: "block" }, [
    h("h3", {}, ["How much jitter is 2 px of jitter?"]),
    p(jitter.note, "aside"),
    figures([
      ["measured box-width residual, std", `${rate(jitter.medianBoxWidthPx)} px median box width`],
      [
        "per-corner sigma equivalent to the measurement",
        `${rate(jitter.cornerSigmaEquivalentPx.minPx)} to ${rate(jitter.cornerSigmaEquivalentPx.maxPx)} px`,
      ],
      [
        `the sweep's top level (σ = ${fixed(jitter.stressAtMaxSigma.sigmaPx, 0)} px), as a multiple of that`,
        `${rate(jitter.stressAtMaxSigma.lowest)}× to ${rate(jitter.stressAtMaxSigma.highest)}×`,
      ],
      [
        "σ = 2 px, as a multiple of that",
        `${rate(jitter.stressAtSigma2.lowest)}× to ${rate(jitter.stressAtSigma2.highest)}×`,
      ],
    ]),
    p(jitter.cornerSigmaEquivalentPx.method, "aside"),
  ]);
}

// -- section: identity at the gate -------------------------------------------

function trackingSection(): readonly Child[] {
  const data = REPORTS.tracking;
  return [
    protocol([
      data.clip,
      `${count(data.frames)} frames`,
      `${data.labels.total} labelled vehicles`,
      `gate region ± ${fixed(data.gateRegion.halfWidthPx, 0)} px`,
      `${data.countingRule} rule held fixed`,
      `${data.separation.levelsMeasured} levels`,
      // The metric here is fragmentation rather than F1, so the counting note
      // would be the wrong units -- but the hazard is the same one, and this is
      // the section that invites it most: seventeen labels means a one-identity
      // difference is the smallest step the instrument can take, and the tables
      // below put three trackers side by side row after row.
      fragmentationResolutionNote(),
    ]),
    table(
      `Undegraded footage, ${data.clean.levelLabel}. A fragmentation ratio of 1.0 is one ` +
        `predicted identity per labelled vehicle; identity deviation is the multiplicative ` +
        `fold from 1.0, so losing identities is not flattered against splitting them.`,
      [
        { head: "tracker" },
        { head: "identities at the gate", numeric: true },
        { head: "identities that crossed", numeric: true },
        { head: "fragmentation", numeric: true },
        { head: "identity deviation", numeric: true },
        { head: "crossing-id ratio", numeric: true },
        { head: "class agreement", numeric: true },
      ],
      data.clean.trackers.map((row): readonly Cell[] => {
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
    ]),
    h("div", { class: "block" }, [
      h("h3", {}, ["What is not claimed here"]),
      p(
        "Rendered from the benchmark's own record, word for word. A benchmark that says what it " +
          "cannot measure is worth more than one that does not.",
        "aside",
      ),
      h(
        "dl",
        { class: "claims" },
        data.claimsNotMade.flatMap((item) => [
          h("dt", {}, [item.claim]),
          h("dd", {}, [item.reason]),
        ]),
      ),
    ]),
    disclosure("What the metrics mean, exactly", [
      figures([
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
    ]),
    disclosure("Every level, every tracker", [
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
    ]),
    disclosure(`What these figures do not say (${data.caveats.length})`, [list(data.caveats)]),
  ];
}

// -- section: the honest negatives -------------------------------------------
//
// The claim sentences are authored in the markup. What lands here is the
// measurement behind each one, keyed by the slot it fills.

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
      ["engine, undegraded F1", rate(engineGate.f1)],
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
      ["F1 the looser association floor recovers here", rate(DROPOUT_FLOOR_GAIN)],
      ["so the floor explains it", DROPOUT_EXPLAINED ? "yes" : "no"],
    ],
    "negative-floor": [
      ["shipped association floor", String(robustness.associationFloor.shippedFloor)],
      ["floor compared against", String(robustness.associationFloor.floors[1])],
      [
        "most F1 the looser floor recovers",
        `${rate(robustness.associationFloor.largestGain.gain)} at ${robustness.associationFloor.largestGain.levelLabel}`,
      ],
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
      ["download saved", megabytes(model.bytesSaved)],
      ["detections lost", percent(model.detectionsLostFraction)],
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
      ["clips labelled", count(distinct([counting.clip, robustness.clip, REPORTS.tracking.clip]))],
      ["gates labelled", count(distinct([counting.gate.name]))],
      ["labelled crossings", count(counting.labels.total)],
      [
        "of which adjudicated as certain",
        `${count(counting.labels.certain)}; ${count(counting.labels.probable)} probable`,
      ],
      ["so every accuracy figure here is an", "UPPER bound"],
    ],
    "negative-scale": [
      ["anchor candidates searched for", count(speedReal.anchorCandidates.length)],
      ["usable ones found", count(USABLE_ANCHORS)],
      [
        "honest bracket on the along-road scale",
        `${fixed(speedReal.bracket.lowerM, 1)} m to ${fixed(speedReal.bracket.upperM, 1)} m`,
      ],
      [
        "which propagates to speed as",
        `${signed(speedReal.bracket.bandPercent[0] ?? 0)} % to ${signed(speedReal.bracket.bandPercent[1] ?? 0)} %`,
      ],
      ["km/h published from this clip", speedReal.absoluteSpeedPublished ? "yes" : "none"],
    ],
  };
}

/** The four identity claims, verbatim, as a list rather than a figure run.
 *
 * They are sentences, and a figure run is for a term and its number: putting a
 * sentence in the value column squeezes the term column to nothing and breaks it
 * one letter per line. Kept word for word from the benchmark's own record --
 * paraphrasing a refusal is how a refusal softens. */
function claimsNotMadeList(): HTMLElement {
  return list(
    REPORTS.tracking.claimsNotMade.map((item) => item.claim),
    "plain-list",
  );
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
    protocol([
      "one core, two implementations",
      `${parity.caseCount} committed parity cases`,
      `${parity.straddleKinds.length} boundary kinds`,
      `speeds agree to ${scientific(parity.speedToleranceKmh)} km/h`,
      `crossing decisions agree exactly`,
    ]),
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
    h("div", { class: "block" }, [
      h("h3", {}, ["The detector the page downloads, and the one it refused"]),
      table(
        `Both graphs decoded through identical letterboxing and class-wise NMS at ` +
          `confidence ${model.confidence} and IoU ${model.nmsIou}, over ${model.sampledFrames} ` +
          `frames sampled every ${model.sampleStride}th from the motorway clip.`,
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
      p(
        `int8 saves ${megabytes(model.bytesSaved)} of download and loses ` +
          `${percent(model.detectionsLostFraction)} of detections. In a counting product a ` +
          `missed detection is a missed count, so the trade is refused at any download size.`,
        "verdict",
      ),
    ]),
    disclosure("The detector noise the robustness sweep is calibrated against", [
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
      "One gate segment and four vehicle paths. A count fires where the step between two frame " +
        "samples meets the segment — which is why a path that clears the band in one step is " +
        "still counted, a path that sits inside the band without changing side is not, and a " +
        "path crossing the gate's line past its end never is.",
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
    slot(name).replaceChildren(figures(items));
  }
  slot("negative-tracking").replaceChildren(claimsNotMadeList());
}
