/** How the results sections print a measurement, and where they must refuse to.
 *
 * The formatters are the page's precision policy in code: three significant
 * figures on a rate because seventeen labelled crossings cannot support more, an
 * em dash where there is no measurement rather than a plausible zero, and a real
 * minus sign on a bracket that runs one way only.
 *
 * The last group is the one that matters most: the resolution note and the
 * derived multiples must equal the BAKED values, not values retyped here. That is
 * the browser-side half of the rule that no figure on this page is typed --
 * `tests/test_site_data_sync.py` holds the other half, from the reports up. */

import { readFileSync } from "node:fs";

import { describe, expect, test } from "vitest";

import { REPORTS } from "../generated/reports";
import {
  count,
  fixed,
  fragmentationResolutionNote,
  megabytes,
  percent,
  rate,
  resolutionNote,
  scientific,
  signed,
} from "./kit";
import {
  COMMANDS,
  distinct,
  floorExplains,
  levelOf,
  negativeFigures,
  protocolNamed,
  slotNames,
  usableAnchorCount,
} from "./results";
import { classifyControlBands } from "./results-speed";

describe("count", () => {
  test("groups thousands with a narrow no-break space, not a comma", () => {
    // A comma is a decimal separator across most of Europe, including where the
    // flagship clip was filmed.
    expect(count(8901)).toBe("8 901");
    expect(count(735)).toBe("735");
    expect(count(1069)).toBe("1 069");
  });
});

describe("percent and megabytes", () => {
  test("a fraction becomes a percentage with one decimal", () => {
    expect(percent(0.1762)).toBe("17.6 %");
    expect(percent(1)).toBe("100.0 %");
  });

  test("bytes become megabytes to two places", () => {
    expect(megabytes(7691387)).toBe("7.69 MB");
  });

  test("a non-figure is an em dash", () => {
    expect(percent(Number.NaN)).toBe("—");
  });

  test("a measured value smaller than one decimal keeps three significant figures", () => {
    // The defect this exists for, on the product's ONLY absolute speed claim: the
    // tier-one settled mean relative error is 0.0390 %, and one fixed decimal
    // printed it as `0.0 %` -- a perfect score the measurement never recorded.
    expect(percent(0.00038956397751180726)).toBe("0.0390 %");
    expect(percent(0.0000123)).toBe("0.00123 %");
  });

  test("an exact zero stays a zero, because some measurements are zero", () => {
    // At 2 fps the engine really made no predictions at all. The rule is
    // "non-zero in, zero out", not "small".
    expect(percent(0)).toBe("0.0 %");
  });
});

describe("no formatter may print a measured non-zero as zero", () => {
  // A rounding policy that can do this is a policy that can publish a perfect
  // score the instrument never recorded, so it is asserted as a class over every
  // fixed-decimal formatter rather than patched at the one call site that had it.
  const tiny = 3.7e-5;

  test("percent, fixed, signed and megabytes all refuse to", () => {
    expect(percent(tiny / 100)).not.toBe("0.0 %");
    expect(Number(percent(tiny / 100).replace(" %", ""))).toBeGreaterThan(0);

    expect(fixed(tiny, 2)).not.toBe("0.00");
    expect(Number(fixed(tiny, 2))).toBeGreaterThan(0);

    expect(signed(-tiny, 0)).not.toBe("−0");
    expect(signed(tiny, 0)).not.toBe("+0");

    expect(megabytes(1)).not.toBe("0.00 MB");
    expect(Number(megabytes(1).replace(" MB", ""))).toBeGreaterThan(0);
  });

  test("but an exact zero still prints as one, in every one of them", () => {
    // The control, and it varies a different axis from the mutation above: same
    // formatters, a value that IS zero rather than a value that rounds to zero.
    // Without it, "never print zero" could be satisfied by never printing zero.
    expect(fixed(0, 2)).toBe("0.00");
    expect(signed(0)).toBe("+0");
    expect(megabytes(0)).toBe("0.00 MB");
    expect(percent(0)).toBe("0.0 %");
  });

  test("and the tier-one speed figure on the page is the measured one", () => {
    // Pinned to the bake rather than to a literal: this is the figure that was
    // published as 0.0 %.
    const measured = REPORTS.speedSynthetic.fullChain.settled.meanRelativePercent;
    expect(measured).toBeGreaterThan(0);
    expect(percent(measured / 100)).toBe(`${rate(measured)} %`);
    expect(percent(measured / 100)).not.toContain("0.0 %");
  });
});

describe("count, where there is nothing to count", () => {
  test("a cell the model card leaves empty is an em dash, not the string NaN", () => {
    // The architecture table reads `number | null` cells straight out of the
    // card. `count` was the one formatter without the finite guard its four
    // siblings have, so a card with an em-dashed cell put the literal text `NaN`
    // into a published table.
    expect(count(Number.NaN)).toBe("—");
    expect(count(Number.POSITIVE_INFINITY)).toBe("—");
    // The control: a real count still prints, so the guard cannot be passing by
    // refusing everything.
    expect(count(244)).toBe("244");
  });
});

describe("the fragmentation resolution note", () => {
  test("is one identity out of the labelled vehicles, in fragmentation's own units", () => {
    const denominator = REPORTS.tracking.metricDefinitions.fragmentationRatio.denominator;
    expect(fragmentationResolutionNote()).toContain(rate(1 / denominator));
    expect(fragmentationResolutionNote()).toContain("one identity");
    // Not the counting sentence reused: the metric is a ratio, not an F1.
    expect(fragmentationResolutionNote()).not.toContain("F1");
  });
});

describe("fixed and scientific", () => {
  test("fixed prints an em dash for a null rather than a zero", () => {
    expect(fixed(null, 2)).toBe("—");
    expect(fixed(30, 0)).toBe("30");
  });

  test("scientific reaches for an exponent only where decimals would be zeros", () => {
    expect(scientific(6.186672121032756e-6)).toBe("6.19e-6");
    expect(scientific(0.05566014192420679)).toBe("0.0557");
    expect(scientific(0)).toBe("0");
  });
});

describe("signed", () => {
  test("uses a real minus sign and an explicit plus", () => {
    // The bracket is -33 %/+0 %: the assumption is the CEILING, and a bare 0 at
    // the top end loses that.
    expect(signed(-33)).toBe("−33");
    expect(signed(0)).toBe("+0");
    expect(signed(1.456, 3)).toBe("+1.456");
  });
});

describe("the resolution note", () => {
  test("is the baked one-event step, not a number typed here", () => {
    expect(resolutionNote()).toContain(rate(REPORTS.counting.resolution.oneEventF1));
    expect(resolutionNote()).toContain("one event");
  });

  test("and the baked step is what one event actually moves F1 by", () => {
    // Recomputed from the engine's own operating point, independently of the
    // generator: if the bake's arithmetic drifted, the page would print a
    // resolution that does not match the measurement it is a resolution of.
    const engine = REPORTS.counting.methods.find((method) => method.method === "engine+gate");
    expect(engine).toBeDefined();
    const labels = REPORTS.counting.labels.total;
    const f1 = (tp: number, predicted: number) => (2 * tp) / (predicted + labels);
    const expected = Math.abs(
      f1(engine!.truePositives, engine!.nPredicted) -
        f1(engine!.truePositives, engine!.nPredicted - 1),
    );
    expect(REPORTS.counting.resolution.oneEventF1).toBeCloseTo(expected, 15);
  });
});

describe("what the bake must carry for the sections to render", () => {
  test("no absolute km/h from the flagship clip is reachable", () => {
    // The prohibition, asserted from the browser side as well as the Python side:
    // that clip has no independent along-road anchor, its along-road scale is
    // bracketed at -33 %/+0 %, and the engine returns null for speed on it.
    const serialised = JSON.stringify(REPORTS);
    expect(serialised).not.toContain("maxSpeedKmh");
    expect(serialised).not.toContain("106.96");
    expect(REPORTS.speedReal.absoluteSpeedPublished).toBe(false);
    expect(REPORTS.speedReal.shippedConfigCalibrated).toBe(false);
  });

  test("every section names the report it came from", () => {
    for (const [key, section] of Object.entries(REPORTS)) {
      expect(section, key).toHaveProperty("source");
    }
  });

  test("the tracking benchmark's refusals are carried, not summarised", () => {
    expect(REPORTS.tracking.claimsNotMade.length).toBeGreaterThanOrEqual(4);
    for (const item of REPORTS.tracking.claimsNotMade) {
      expect(item.claim.length).toBeGreaterThan(20);
      expect(item.reason.length).toBeGreaterThan(20);
    }
  });

  test("the int8 refusal carries both of its figures and its protocol", () => {
    expect(REPORTS.model.bytesSaved).toBeGreaterThan(0);
    expect(REPORTS.model.detectionsLostFraction).toBeGreaterThan(0);
    expect(REPORTS.model.sampledFrames).toBeGreaterThan(0);
    expect(REPORTS.model.confidence).toBeGreaterThan(0);
  });
});

describe("no figure in the negatives is typed into TS prose", () => {
  // The rule: a figure renders from the bake, and one that genuinely must be
  // inlined carries a test asserting it equals the baked value. These four were
  // inlined with neither -- `17` typed two lines from the baked label count it
  // restates, and `1`, `1`, `0` typed into figure-run values.
  const items = negativeFigures();

  test("the band-rule term states the baked label count, not a typed 17", () => {
    const term = items["negative-band"]?.[0]?.[0];
    expect(term).toBeDefined();
    expect(term).toContain(count(REPORTS.counting.labels.total));
    expect(term).toBe(`band rule predictions against ${count(REPORTS.counting.labels.total)} labels`);
  });

  test("clips and gates labelled are counted from the bake", () => {
    const values = new Map(items["negative-one-clip"]);
    expect(values.get("clips labelled")).toBe(
      count(distinct([REPORTS.counting.clip, REPORTS.robustness.clip, REPORTS.tracking.clip])),
    );
    expect(values.get("gates labelled")).toBe(count(distinct([REPORTS.counting.gate.name])));
    expect(values.get("labelled crossings")).toBe(count(REPORTS.counting.labels.total));
  });

  test("and `distinct` actually counts, so those rows are not constants", () => {
    // The control, on a different axis from the assertions above: the same helper
    // fed a set that is not of size one. Without it, "clips labelled = 1" could be
    // satisfied by a function that returns 1.
    expect(distinct(["a", "b", "a"])).toBe(2);
    expect(distinct(["a", "a"])).toBe(1);
    expect(distinct([])).toBe(0);
  });

  test("usable anchors are classified from the survey's own verdicts", () => {
    // Derived, not typed: this assertion is what moves if a candidate's verdict
    // moves. Kept separate from the reading below on purpose, so a bake that
    // changed would fail the reading rather than this.
    const values = new Map(items["negative-scale"]);
    expect(values.get("usable ones found")).toBe(
      count(usableAnchorCount(REPORTS.speedReal.anchorCandidates)),
    );
  });

  test("and on the committed survey that classification is none", () => {
    const values = new Map(items["negative-scale"]);
    expect(values.get("usable ones found")).toBe("0");
    expect(REPORTS.speedReal.absoluteSpeedPublished).toBe(false);
  });

  test("and a verdict that WOULD license a scale is counted, not assumed away", () => {
    // The control: the classifier is not constant-false. Its counterpart is that
    // an unknown verdict throws rather than being counted as another failure --
    // the failure mode where the survey finds an anchor and the page keeps saying
    // none was found.
    expect(usableAnchorCount([{ verdict: "PRESENT AND MEASURABLE AND CONSISTENT" }])).toBe(1);
    expect(usableAnchorCount([{ verdict: "ABSENT" }])).toBe(0);
    expect(() => usableAnchorCount([{ verdict: "SOMETHING NEW" }])).toThrow(/cannot classify/);
  });
});

describe("addressing the bake by name must fail loudly, not print a dash", () => {
  test("a sweep level the bake does not have throws instead of returning undefined", () => {
    // The failure being prevented: `.find(...)?.engine.f1 ?? NaN` printed an em
    // dash under an authored sentence that asserts the number, and no test could
    // see it because COVERAGE reaches sweep entries by index.
    expect(() => levelOf("box_jitter", "sigma=3 px")).toThrow(/has no level/);
    expect(() => levelOf("frame_rate_typo", "2 fps")).toThrow(/no robustness protocol/);
    expect(() => protocolNamed("detection_dropout_typo")).toThrow(/no robustness protocol/);
  });

  test("and the three levels the negatives name do resolve", () => {
    // The control, varying the label rather than the mechanism: a lookup that
    // threw on everything would satisfy the assertions above.
    expect(levelOf("frame_rate", "2 fps").engine.nPredicted).toBe(0);
    expect(levelOf("box_jitter", "sigma=0 px").engine.f1).toBeGreaterThan(0);
    expect(levelOf("box_jitter", "sigma=2 px").engine.f1).toBeGreaterThan(0);
    expect(protocolNamed("detection_dropout").entries.length).toBeGreaterThan(1);
  });

  test("a protocol the ablation classifies neither way throws", () => {
    // `doesNotExplain.includes(name)` on a renamed protocol is false, and false
    // there prints "the floor explains it: yes" -- the opposite of the finding.
    expect(() => floorExplains("detection_dropout_typo")).toThrow(/neither explained/);
    // The control: both real answers still come back, one of each.
    expect(floorExplains("detection_dropout")).toBe(false);
    expect(floorExplains("frame_rate")).toBe(true);
  });

  test("the scale survey's bands must be one candidate and one positive control", () => {
    const real = REPORTS.speedReal.guardrailControls.bands;
    const split = classifyControlBands(real);
    expect(split.filter((band) => band.candidate)).toHaveLength(1);
    expect(split.filter((band) => band.positive)).toHaveLength(1);
    expect(split.filter((band) => !band.candidate && !band.positive)).toHaveLength(
      real.length - 2,
    );
  });

  test("a renamed band throws instead of silently reclassifying the candidate", () => {
    // The failure being prevented: `startsWith("guardrail")` on a renamed band
    // makes the candidate a control. The chart's emphasis moves to nothing, the
    // hairline vanishes, and the protocol strip's control count grows by one --
    // and every existing test still passes, because the sweep is addressed by
    // index everywhere else.
    const renamed = REPORTS.speedReal.guardrailControls.bands.map((band) => ({
      ...band,
      band: band.band.replace(/^guardrail/, "crash barrier"),
    }));
    expect(() => classifyControlBands(renamed)).toThrow(/one candidate/);

    const noPositive = REPORTS.speedReal.guardrailControls.bands.map((band) => ({
      ...band,
      band: band.band.replace(/^positive control/, "reference line"),
    }));
    expect(() => classifyControlBands(noPositive)).toThrow(/one candidate/);
  });

  test("no surface captions the matched-controls figure as the opposite of its bars", () => {
    // The claim that was there, in the SVG `desc` and in the caption: "the
    // candidate sits among the controls". Its own bars draw the candidate as the
    // TIGHTEST of the five non-positive bands, which is the opposite. The
    // assertion on the measurement is what makes the text assertion non-vacuous:
    // it states why the phrase may not be used.
    const bands = classifyControlBands(REPORTS.speedReal.guardrailControls.bands);
    const candidate = bands.find((band) => band.candidate);
    const controls = bands.filter((band) => !band.candidate && !band.positive);
    const positive = bands.find((band) => band.positive);
    expect(candidate).toBeDefined();
    expect(positive).toBeDefined();
    expect(candidate!.spreadPercent).toBeLessThan(
      Math.min(...controls.map((band) => band.spreadPercent)),
    );
    // And still far from the band whose period is known, which is the comparison
    // the caption is allowed to draw.
    expect(candidate!.spreadPercent).toBeGreaterThan(positive!.spreadPercent * 5);
    // Comments are stripped first: the guard is about what SHIPS, and the
    // docstring on `matchedControlsChart` has to be free to record what the
    // description used to say and why it was wrong.
    for (const [module, marker] of [
      ["figures.ts", "spread of the measured period"],
      ["results-speed.ts", "Spread of the measured local period"],
    ] as const) {
      const shipped = readFileSync(new URL(module, import.meta.url), "utf8")
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/^\s*\/\/.*$/gm, "");
      expect(shipped, module).not.toContain("sits among the controls");
      // The control: stripping comments did not strip the strings the figure
      // actually renders, so this is not passing over an emptied file.
      expect(shipped, module).toContain(marker);
    }
  });

  test("and the controls the page counts are the ones the bake carries", () => {
    // The control, varying the number of bands rather than their names: a split
    // that threw on everything would pass the two assertions above.
    expect(() =>
      classifyControlBands([
        { band: "guardrail post band", spreadPercent: 9.1 },
        { band: "asphalt control A", spreadPercent: 31.8 },
        { band: "positive control: divider-1 dashes", spreadPercent: 1.3 },
      ]),
    ).not.toThrow();
  });
});

describe("the slots and the sections are two halves of one document", () => {
  const html = readFileSync(new URL("../../index.html", import.meta.url), "utf8");
  const inMarkup = [...html.matchAll(/data-results="([^"]+)"/g)].map((match) => match[1] ?? "");

  test("every slot in the markup is filled, and every section has a slot", () => {
    // Nothing compared these two sets before. `mountResults` throws on a missing
    // slot -- which, before the guard in `main.ts`, also took the live demo down
    // -- and `?selftest=1` returns before the results mount, so `verify_page.sh`
    // could not see a mismatch either. The guard keeps the demo alive; this is
    // what notices.
    expect([...inMarkup].sort()).toEqual([...slotNames()].sort());
  });

  test("and there are as many of them as the page actually has", () => {
    // The floor, so the comparison cannot be passing over two empty sets: an
    // emptied SECTIONS and an emptied markup would compare equal.
    expect(inMarkup.length).toBeGreaterThanOrEqual(17);
    expect(new Set(inMarkup).size).toBe(inMarkup.length);
  });

  // The four figures spelled out in the authored prose of the results half, found
  // by re-reading every honest-negative sentence against the numbers printed under
  // it. A word is as much a figure as a numeral: "Nine findings" over eight
  // findings is the same defect as a stale F1, and the markup carries no test of
  // its own. Each is pinned to what it counts.

  test("the negatives lede counts the negatives that are actually there", () => {
    const blocks = [...html.matchAll(/class="negative"/g)].length;
    expect(blocks).toBe(9);
    expect(html).toContain("Nine findings");
  });

  test("the identity-claims heading counts the claims the bake carries", () => {
    expect(REPORTS.tracking.claimsNotMade).toHaveLength(4);
    expect(html).toContain("Four identity claims this project does not make");
  });

  test("the jitter negative's prose names the sweep levels the bake has", () => {
    // "At two frames a second" and "jitter of two pixels" are the two levels the
    // figures beneath them report. A sweep that no longer ran 2 fps or sigma=2 px
    // would leave the sentences asserting conditions nobody measured.
    const rates = REPORTS.robustness.protocols.find((item) => item.name === "frame_rate");
    const jitter = REPORTS.robustness.protocols.find((item) => item.name === "box_jitter");
    expect(rates?.entries.map((entry) => entry.levelLabel)).toContain("2 fps");
    expect(jitter?.entries.map((entry) => entry.levelLabel)).toContain("sigma=2 px");
    expect(html).toContain("At two frames a second");
    expect(html).toContain("jitter of two pixels");
  });

  test("and the jitter negative does not claim a loss the figures deny", () => {
    // The claim that was there: "takes most of it" / "removes most of its
    // remaining accuracy", over a figure run reading 0.914 at sigma=0 px and
    // 0.643 at sigma=2 px. That is about 30 per cent lost and 70 per cent kept,
    // so "most" was the wrong side of the number.
    const jitter = REPORTS.robustness.protocols.find((item) => item.name === "box_jitter");
    const clean = jitter?.entries.find((entry) => entry.levelLabel === "sigma=0 px")?.engine.f1;
    const stressed = jitter?.entries.find((entry) => entry.levelLabel === "sigma=2 px")?.engine.f1;
    expect(clean).toBeDefined();
    expect(stressed).toBeDefined();
    // The assertion is on the measurement, not on the wording: while MORE than
    // half survives, the page may not say most of it is removed.
    expect(stressed! / clean!).toBeGreaterThan(0.5);
    expect(html).not.toContain("takes most of it");
    expect(html).not.toContain("removes most of its remaining accuracy");
  });
});

describe("the reference section", () => {
  test("marks the two placeholder commands as not built", () => {
    // The page must not claim a command works when --help says it does not.
    // `tests/test_reference_matches_the_cli.py` is what ties this list to the
    // real CLI, in both directions.
    const notBuilt = COMMANDS.filter((command) => !command.implemented).map((c) => c.name);
    expect(notBuilt.sort()).toEqual(["bench", "serve"]);
  });

  test("lists no command twice", () => {
    const names = COMMANDS.map((command) => command.name);
    expect(new Set(names).size).toBe(names.length);
  });
});
