/** The two speed sections.
 *
 * Their own module because they are the pair a reader is most likely to check
 * line by line: one publishes the product's only absolute speed claim, and the
 * other publishes the reason no absolute speed is published for the flagship clip
 * at all.
 *
 * Nothing here derives a km/h from `motorway-a40.webm`. That clip has no
 * independent along-road anchor -- the survey looked for five and found none that
 * survives its own controls -- so its along-road scale is bracketed at -33 %/+0 %
 * and the engine returns nothing for speed on it. The tier-two section publishes
 * that negative result WITH its matched controls, which is what makes "not
 * measurable" a measurement rather than a shrug.
 *
 * Both sections used to argue their case in paragraphs. They now lead with the
 * figures and keep the argument -- the survey, its controls, the bracket and the
 * report's own words -- one disclosure down. Nothing was deleted; the reading
 * order changed. */

import { REPORTS } from "../generated/reports";
import { matchedControlsChart } from "./figures";
import type { ControlBand } from "./figures";
import {
  count,
  disclosure,
  figures,
  fixed,
  h,
  list,
  p,
  percent,
  plate,
  protocol,
  rate,
  scientific,
  signed,
  table,
  tiles,
} from "./kit";
import type { Child } from "./kit";

// -- which band is the candidate, and which are the controls -------------------
//
// The chart and the protocol strip both turn on this split, and it was decided by
// `band.startsWith("guardrail")` / `startsWith("positive control")`. A band
// renamed in the benchmark would have made the candidate a control and the
// positive control a control: the emphasis would move to nothing, the hairline
// would vanish, and "4 matched controls" would quietly become 6. Nothing would
// throw and no test would fail, because the sweep is addressed by index
// everywhere else.
//
// So the split is resolved once, at module load, against what the survey's own
// structure requires: exactly one candidate and exactly one positive control,
// with at least one matched control left over -- the controls are the whole point
// of the result.

/** The candidate band, the positive control, and the matched controls between
 * them. Throws rather than misclassifying: a wrong split here publishes the
 * opposite of the finding. */
export function classifyControlBands(
  bands: readonly { readonly band: string; readonly spreadPercent: number }[],
): readonly ControlBand[] {
  const out: ControlBand[] = bands.map((band) => ({
    band: band.band,
    spreadPercent: band.spreadPercent,
    candidate: band.band.startsWith("guardrail"),
    positive: band.band.startsWith("positive control"),
  }));
  const named = out.map((band) => band.band).join(", ");
  const candidates = out.filter((band) => band.candidate).length;
  const positives = out.filter((band) => band.positive).length;
  const matched = out.filter((band) => !band.candidate && !band.positive).length;
  if (candidates !== 1 || positives !== 1 || matched < 1) {
    throw new Error(
      `the scale survey's bands must be one candidate, one positive control and ` +
        `at least one matched control; found ${candidates}, ${positives} and ` +
        `${matched} in: ${named}`,
    );
  }
  return out;
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

const CONTROL_BANDS = classifyControlBands(REPORTS.speedReal.guardrailControls.bands);
const USABLE_ANCHORS = usableAnchorCount(REPORTS.speedReal.anchorCandidates);
const MATCHED_CONTROLS = CONTROL_BANDS.filter(
  (band) => !band.candidate && !band.positive,
).length;

// -- section: speed, tier one -------------------------------------------------

/** The fastest band in the full-chain sweep, found by its speed.
 *
 * The tile beside it says "at 130 km/h, the fastest band", and taking the LAST
 * row would make that sentence a statement about the report's row order rather
 * than about the sweep -- true today, silently false the first time a band is
 * appended out of order. The tile is the section's second headline figure, so
 * it is worth the six lines. */
export function fastestBand<T extends { readonly speedKmh: number }>(
  bands: readonly T[],
): T {
  let fastest = bands[0];
  if (fastest === undefined) {
    throw new Error("the tier-one full-chain sweep has no speed bands");
  }
  for (const band of bands) {
    if (band.speedKmh > fastest.speedKmh) {
      fastest = band;
    }
  }
  return fastest;
}

export function speedTierOneSection(): readonly Child[] {
  const data = REPORTS.speedSynthetic;
  const byBand = data.homographyOnly.byBand.map((row, index) => {
    const full = data.fullChain.byBand[index];
    return { band: row.speedKmh, chain: row, full };
  });
  const fastest = fastestBand(data.fullChain.byBand);

  return [
    tiles([
      {
        label: "plane and estimator alone",
        value: `${scientific(data.homographyOnly.settled.maxAbsErrorKmh)} km/h`,
        note: "worst absolute error on noise-free detections, settled",
        lead: true,
      },
      {
        label: "through the full tracker chain",
        value: `${rate(fastest.maxAbsErrorKmh)} km/h`,
        note: `worst absolute error at ${fixed(fastest.speedKmh, 0)} km/h, the fastest band`,
      },
      {
        label: "requirement",
        value: `${fixed(data.homographyOnly.requirementKmh, 1)} km/h`,
        note: `plane holdout recovers the camera to ${scientific(data.roadPlane.holdoutMaxErrorM)} m`,
      },
    ]),
    table(
      `Speed error by band against exact truth, on a simulated overpass camera. The first ` +
        `column pair is the homography and speed estimator alone; the second is the same ` +
        `detections through the full tracker chain. Calibration error is excluded by ` +
        `construction, so these are a ceiling and not a field prediction.`,
      [
        { head: "true speed km/h", numeric: true },
        { head: "samples", numeric: true },
        { head: "plane only, RMSE km/h", numeric: true },
        { head: "plane only, max abs km/h", numeric: true },
        { head: "full chain, RMSE km/h", numeric: true },
        { head: "full chain, max abs km/h", numeric: true },
      ],
      byBand.map((row) => [
        fixed(row.band, 0),
        count(row.chain.n),
        scientific(row.chain.rmseKmh),
        scientific(row.chain.maxAbsErrorKmh),
        scientific(row.full?.rmseKmh ?? Number.NaN),
        scientific(row.full?.maxAbsErrorKmh ?? Number.NaN),
      ]),
    ),
    disclosure(
      "Protocol, what this is an upper bound on, the independent check, the noise sweep and " +
        "the smoothing trade",
      [
        protocol([
          "synthetic scene, exact truth",
          `camera ${fixed(data.camera.heightM, 0)} m up, ${fixed(data.camera.pitchDeg, 0)}° down, ${fixed(data.camera.focalPx, 0)} px focal`,
          `plane holdout max error ${scientific(data.roadPlane.holdoutMaxErrorM)} m`,
          `${data.speedBandsKmh.length} speed bands ${fixed(data.speedBandsKmh[0] ?? 0, 0)}–${fixed(data.speedBandsKmh[data.speedBandsKmh.length - 1] ?? 0, 0)} km/h`,
          `settled = ${fixed(data.settled.windowS, 1)} s of observation`,
          `requirement ${fixed(data.homographyOnly.requirementKmh, 1)} km/h`,
        ]),
        p(data.whatThisIs, "verdict"),
        figures([
          ["settled samples, full chain", count(data.fullChain.settled.n)],
          ["full chain RMSE, settled", `${scientific(data.fullChain.settled.rmseKmh)} km/h`],
          [
            "full chain mean relative error, settled",
            percent(data.fullChain.settled.meanRelativePercent / 100),
          ],
          [
            "including start-up samples",
            `${scientific(data.fullChain.allSamples.rmseKmh)} km/h RMSE over ${count(data.fullChain.allSamples.n)}`,
          ],
          ["plane fit holdout, mean error", `${scientific(data.roadPlane.holdoutMeanErrorM)} m`],
        ]),
        p(data.limitations),
        p(
          `The plane here is fitted to the very camera that generated the boxes, and the fit ` +
            `recovers it to ${scientific(data.roadPlane.holdoutMaxErrorM)} m on held-out points. ` +
            `Calibration error is therefore excluded by construction, not measured — so this is ` +
            `the ceiling a perfectly surveyed camera would allow, and it says nothing about a ` +
            `survey done in the field.`,
        ),
        p(data.settled.meaning),
        p(REPORTS.speedSynthetic.checkC.what),
        figures([
          ["gate separation", `${fixed(REPORTS.speedSynthetic.checkC.gateSeparationM, 0)} m`],
          [
            "agreement required",
            `${fixed(REPORTS.speedSynthetic.checkC.agreementRequirementKmh, 1)} km/h`,
          ],
          [
            "worst disagreement measured",
            `${rate(REPORTS.speedSynthetic.checkC.maxAbsDifferenceKmh)} km/h`,
          ],
          ["run on", REPORTS.speedSynthetic.checkC.runOn],
        ]),
        p(REPORTS.speedSynthetic.checkC.whyNotOnRealFootage),
        p(data.noiseCalibration.why),
        p(data.noiseCalibration.sourceCaveat, "aside"),
        table(
          `Noise scaled from the residuals measured on the real clip ` +
            `(${data.noiseCalibration.source}), anchored on ${data.noiseCalibration.sweepAnchoredOn}.`,
          [
            { head: "× measured σ", numeric: true },
            { head: "σ centre-y px", numeric: true },
            { head: "σ box-width px", numeric: true },
            { head: "vehicles offered", numeric: true },
            { head: "lost by the tracker", numeric: true },
            { head: "samples", numeric: true },
            { head: "RMSE km/h", numeric: true },
            { head: "max abs km/h", numeric: true },
          ],
          data.noiseSweep.map((row) => [
            fixed(row.sigmaMultiple, 2),
            rate(row.sigmaCentreYPx),
            rate(row.sigmaBoxWidthPx),
            count(row.vehiclesOffered),
            count(row.vehiclesLost),
            count(row.overall.n),
            rate(row.overall.rmseKmh),
            rate(row.overall.maxAbsErrorKmh),
          ]),
        ),
        p(data.monotonicNote),
        p(REPORTS.speedSynthetic.smoothingTrade.what),
        p(REPORTS.speedSynthetic.smoothingTrade.whyThatMatters, "aside"),
        table(
          `At ${REPORTS.speedSynthetic.smoothingTrade.sigma}, over ` +
            `${REPORTS.speedSynthetic.smoothingTrade.seeds} seeds, with ` +
            `${REPORTS.speedSynthetic.smoothingTrade.vehiclesLost} vehicles lost — so the two ` +
            `columns differ by the anchor and nothing else.`,
          [
            { head: "true speed km/h", numeric: true },
            { head: "Kalman anchor, RMSE", numeric: true },
            { head: "raw detection anchor, RMSE", numeric: true },
            { head: "better" },
          ],
          REPORTS.speedSynthetic.smoothingTrade.byBand.map((row) => [
            fixed(row.speedKmh, 0),
            rate(row.kalmanRmseKmh),
            rate(row.rawRmseKmh),
            row.better === "kalman_anchor" ? "Kalman" : "raw detection",
          ]),
        ),
        p(REPORTS.speedSynthetic.smoothingTrade.finding, "verdict"),
        p(REPORTS.speedSynthetic.boxModel.what),
        figures([
          [
            "footprint anchor, RMSE",
            `${scientific(REPORTS.speedSynthetic.boxModel.footprintRmseKmh)} km/h`,
          ],
          [
            "solid-vehicle anchor, RMSE",
            `${scientific(REPORTS.speedSynthetic.boxModel.solidRmseKmh)} km/h`,
          ],
        ]),
        p(REPORTS.speedSynthetic.boxModel.whyItBarelyMoves),
        p(`Reproduce: ${data.reproduce}`, "reproduce"),
      ],
    ),
  ];
}

// -- section: speed, tier two -------------------------------------------------

export function speedTierTwoSection(): readonly Child[] {
  const data = REPORTS.speedReal;
  const bands = CONTROL_BANDS;
  const matched = MATCHED_CONTROLS;

  return [
    tiles([
      {
        label: "km/h published from this clip",
        value: data.absoluteSpeedPublished ? "yes" : "none",
        note: `the shipped config carries ${data.shippedConfigCalibrated ? "a" : "no"} calibration block`,
        lead: true,
      },
      {
        label: "usable along-road anchors",
        value: `${count(USABLE_ANCHORS)} of ${count(data.anchorCandidates.length)}`,
        note: `searched with ${count(matched)} matched controls and a positive control`,
      },
      {
        label: "bracket on every speed",
        value: `${signed(data.bracket.bandPercent[0] ?? 0)} % to ${signed(data.bracket.bandPercent[1] ?? 0)} %`,
        note: `along-road scale ${fixed(data.bracket.lowerM, 1)} m to ${fixed(data.bracket.upperM, 1)} m`,
      },
    ]),
    disclosure("Why: the survey, its controls, and the bracket", [
      protocol([
        data.clip,
        `${data.anchorCandidates.length} anchor candidates`,
        `${matched} matched controls and a positive control`,
        `assumed period ${fixed(data.dividerDisagreement.assumedPeriodM, 1)} m`,
        `bracket ${signed(data.bracket.bandPercent[0] ?? 0)} %/${signed(data.bracket.bandPercent[1] ?? 0)} % on every speed`,
        data.absoluteSpeedPublished ? "km/h published" : "no km/h published",
      ]),
      p(data.headline, "verdict"),
      plate(
        matchedControlsChart(bands),
        "Spread of the measured local period, one bar per band. Read what the picture shows " +
          "and nothing more: the guardrail candidate is the tightest of the bands that are not " +
          "known-periodic, and it is still nowhere near the positive control — a line whose " +
          "period IS known, which the hairline marks an order of magnitude to the left. " +
          "Beating featureless asphalt on spread is not evidence of a period. The evidence " +
          "that the posts are indistinguishable from asphalt is the full-span comb correlation " +
          "and the search-band-edge peaks, stated in the note above; neither is plotted here, " +
          "because neither is a spread.",
        [p(data.guardrailControls.whyTheControlsMatter, "aside")],
      ),
      table(
        "Every along-road anchor the footage was searched for, and what each one turned out to be.",
        [{ head: "candidate" }, { head: "verdict" }, { head: "what was measured" }],
        data.anchorCandidates.map((candidate) => [
          candidate.candidate,
          candidate.verdict,
          candidate.whatWasMeasured,
        ]),
      ),
      p(data.delineators.whyUnusable),
      h("h4", {}, ["The two markings disagree with each other"]),
      p(data.dividerDisagreement.what),
      table(
        `Under one road plane and one along-road scale both dividers must give the same step. ` +
          `They do not, at any plausible horizon row.`,
        [
          { head: "horizon row", numeric: true },
          { head: "divider 1 fit rms, m", numeric: true },
          { head: "divider 2 step, m", numeric: true },
          { head: "ratio", numeric: true },
        ],
        data.dividerDisagreement.byHorizonRow.map((row) => [
          fixed(row.horizonRow, 1),
          rate(row.divider1FitRmsM),
          rate(row.divider2StepM),
          rate(row.ratio),
        ]),
      ),
      figures([
        [
          "ratio across every plausible horizon row",
          `${rate(data.dividerDisagreement.ratioRange[0])} to ${rate(data.dividerDisagreement.ratioRange[1])}`,
        ],
        [
          "the row that would reconcile them",
          `${fixed(data.dividerDisagreement.escapeHatchClosed.horizonRow, 0)}, at which divider 1's own fit rms becomes ${rate(data.dividerDisagreement.escapeHatchClosed.divider1FitRmsM)} m`,
        ],
        [
          "guardrail meets divider 1, from the surveyed vanishing point",
          `${rate(data.dividerDisagreement.vanishingPoints.divider1xGuardrailPx)} px`,
        ],
        [
          "guardrail meets divider 2",
          `${rate(data.dividerDisagreement.vanishingPoints.divider2xGuardrailPx)} px`,
        ],
        [
          "the two dividers meet each other",
          `${rate(data.dividerDisagreement.vanishingPoints.divider1xDivider2Px)} px`,
        ],
      ]),
      p(data.dividerDisagreement.uniformityNote),
      p(data.dividerDisagreement.escapeHatchClosed.verdict),
      p(data.dividerDisagreement.cause.newEvidence),
      p(
        `The cause is ${data.dividerDisagreement.cause.identified ? "identified" : "not identified"}. ` +
          `Still open: ${data.dividerDisagreement.cause.candidatesStillOpen.join("; ")}.`,
      ),
      h("h4", {}, ["The one clean measurement in the survey"]),
      p(data.cleanMeasurement.what),
      figures([
        ["sub-pixel columns tracked", count(data.cleanMeasurement.trackedColumns)],
        ["robust weighted residual rms", `${rate(data.cleanMeasurement.robustWeightedRmsPx)} px`],
        ["plain residual rms", `${rate(data.cleanMeasurement.plainRmsPx)} px`],
        ["plain residual max", `${rate(data.cleanMeasurement.plainMaxPx)} px`],
        [
          "columns within half a pixel",
          `${count(data.cleanMeasurement.inliersWithinHalfPx)} of ${count(data.cleanMeasurement.trackedColumns)}`,
        ],
      ]),
      p(data.cleanMeasurement.honestyNote),
      p(data.cleanMeasurement.alsoShows),
      p(`Committed as a fixture at ${data.cleanMeasurement.fixture}.`, "aside"),
      h("h4", {}, ["Why nothing was refitted, and what the bracket is"]),
      p(`${data.whyRemoved.attemptedRepair} Outcome: ${data.whyRemoved.outcome}.`),
      p(data.whyRemoved.reason),
      p(data.bracket.propagation),
      p(data.notAnAnchor.theConvergence),
      p(data.notAnAnchor.whyItIsNotAnAnchor),
      p(data.notAnAnchor.roadworksHypothesis, "aside"),
      h("div", { class: "columns" }, [
        h("div", {}, [h("h4", {}, ["What this clip still supports"]), list(data.stillLicenses)]),
        h("div", {}, [
          h("h4", {}, ["What shipping it uncalibrated costs"]),
          list(data.consequences),
        ]),
        h("div", {}, [
          h("h4", {}, ["What would have to change first"]),
          p(data.whatWouldChangeThis),
          list(data.doNotResurrect),
        ]),
      ]),
      p(`Reproduce: ${data.reproduce}`, "reproduce"),
    ]),
  ];
}
