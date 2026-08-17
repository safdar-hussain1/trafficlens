/** The two speed sections.
 *
 * Their own module because they are the longest pair on the page and because they
 * are the pair a reader is most likely to check line by line: one publishes the
 * product's only absolute speed claim, and the other publishes the reason no
 * absolute speed is published for the flagship clip at all.
 *
 * Nothing here derives a km/h from `motorway-a40.webm`. That clip has no
 * independent along-road anchor -- the survey looked for five and found none that
 * survives its own controls -- so its along-road scale is bracketed at -33 %/+0 %
 * and the engine returns nothing for speed on it. The tier-two section publishes
 * that negative result WITH its matched controls, which is what makes "not
 * measurable" a measurement rather than a shrug. */

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

const CONTROL_BANDS = classifyControlBands(REPORTS.speedReal.guardrailControls.bands);
const MATCHED_CONTROLS = CONTROL_BANDS.filter(
  (band) => !band.candidate && !band.positive,
).length;

// -- section: speed, tier one -------------------------------------------------

export function speedTierOneSection(): readonly Child[] {
  const data = REPORTS.speedSynthetic;
  const byBand = data.homographyOnly.byBand.map((row, index) => {
    const full = data.fullChain.byBand[index];
    return { band: row.speedKmh, chain: row, full };
  });

  return [
    protocol([
      "synthetic scene, exact truth",
      `camera ${fixed(data.camera.heightM, 0)} m up, ${fixed(data.camera.pitchDeg, 0)}° down, ${fixed(data.camera.focalPx, 0)} px focal`,
      `plane holdout max error ${scientific(data.roadPlane.holdoutMaxErrorM)} m`,
      `${data.speedBandsKmh.length} speed bands ${fixed(data.speedBandsKmh[0] ?? 0, 0)}–${fixed(data.speedBandsKmh[data.speedBandsKmh.length - 1] ?? 0, 0)} km/h`,
      `settled = ${fixed(data.settled.windowS, 1)} s of observation`,
      `requirement ${fixed(data.homographyOnly.requirementKmh, 1)} km/h`,
    ]),
    p(data.whatThisIs, "verdict"),
    table(
      "Speed error by band, on noise-free detections. The first column pair is the " +
        "homography and speed estimator alone; the second is the same detections through the " +
        "full tracker chain.",
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
    figures([
      ["settled samples, full chain", count(data.fullChain.settled.n)],
      ["full chain RMSE, settled", `${scientific(data.fullChain.settled.rmseKmh)} km/h`],
      ["full chain mean relative error, settled", percent(data.fullChain.settled.meanRelativePercent / 100)],
      [
        "including start-up samples",
        `${scientific(data.fullChain.allSamples.rmseKmh)} km/h RMSE over ${count(data.fullChain.allSamples.n)}`,
      ],
      ["plane fit holdout, mean error", `${scientific(data.roadPlane.holdoutMeanErrorM)} m`],
    ]),
    h("div", { class: "block block--limit" }, [
      h("h3", {}, ["What this figure is an upper bound on"]),
      p(data.limitations),
      p(
        `The plane here is fitted to the very camera that generated the boxes, and the fit ` +
          `recovers it to ${scientific(data.roadPlane.holdoutMaxErrorM)} m on held-out points. ` +
          `Calibration error is therefore excluded by construction, not measured — so this is ` +
          `the ceiling a perfectly surveyed camera would allow, and it says nothing about a ` +
          `survey done in the field.`,
      ),
      p(data.settled.meaning),
    ]),
    disclosure("The independent check: two gates a known distance apart", [
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
    ]),
    disclosure("Under detector-like noise, and where it stops holding", [
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
    ]),
    disclosure("What the Kalman smoothing buys, measured rather than assumed", [
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
        ["footprint anchor, RMSE", `${scientific(REPORTS.speedSynthetic.boxModel.footprintRmseKmh)} km/h`],
        ["solid-vehicle anchor, RMSE", `${scientific(REPORTS.speedSynthetic.boxModel.solidRmseKmh)} km/h`],
      ]),
      p(REPORTS.speedSynthetic.boxModel.whyItBarelyMoves),
    ]),
    p(`Reproduce: ${data.reproduce}`, "reproduce"),
  ];
}

// -- section: speed, tier two -------------------------------------------------

export function speedTierTwoSection(): readonly Child[] {
  const data = REPORTS.speedReal;
  const bands = CONTROL_BANDS;
  const matched = MATCHED_CONTROLS;

  return [
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
    h("div", { class: "block" }, [
      h("h3", {}, ["The two markings disagree with each other"]),
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
    ]),
    h("div", { class: "block" }, [
      h("h3", {}, ["The one clean measurement in the survey"]),
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
    ]),
    h("div", { class: "block block--limit" }, [
      h("h3", {}, ["Why nothing was refitted, and what the bracket is"]),
      p(`${data.whyRemoved.attemptedRepair} Outcome: ${data.whyRemoved.outcome}.`, "prose-p"),
      p(data.whyRemoved.reason, "prose-p"),
      figures([
        ["assumed period, upper end", `${fixed(data.bracket.upperM, 1)} m`],
        ["lower end the evidence allows", `${fixed(data.bracket.lowerM, 1)} m`],
        [
          "band on every speed",
          `${signed(data.bracket.bandPercent[0] ?? 0)} % to ${signed(data.bracket.bandPercent[1] ?? 0)} %`,
        ],
        [
          "shipped config carries a calibration block",
          data.shippedConfigCalibrated ? "yes" : "no",
        ],
      ]),
      p(data.bracket.propagation),
      p(data.notAnAnchor.theConvergence),
      p(data.notAnAnchor.whyItIsNotAnAnchor),
      p(data.notAnAnchor.roadworksHypothesis, "aside"),
    ]),
    h("div", { class: "columns" }, [
      h("div", {}, [h("h3", {}, ["What this clip still supports"]), list(data.stillLicenses)]),
      h("div", {}, [
        h("h3", {}, ["What shipping it uncalibrated costs"]),
        list(data.consequences),
      ]),
      h("div", {}, [
        h("h3", {}, ["What would have to change first"]),
        p(data.whatWouldChangeThis),
        list(data.doNotResurrect),
      ]),
    ]),
    p(`Reproduce: ${data.reproduce}`, "reproduce"),
  ];
}

