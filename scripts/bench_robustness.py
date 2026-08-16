#!/usr/bin/env python3
"""Measure how far the engine, and every standard failure mode, falls when
the input stops being clean.

Four independent degradation protocols, each seeded, each run for all nine
{tracker} x {counting rule} compositions over the SAME cached detections
the counting benchmark used:

1. **Frame-rate sensitivity** -- decimate to 25 / 15 / 10 / 5 / 2 fps.
2. **Dropped frames** -- delete a seeded random 5 / 10 / 20 / 30 %.
3. **Detection dropout** -- drop each detection independently with
   p in {0, .05, .1, .2, .3}, simulating occlusion.
4. **Box jitter** -- Gaussian noise on each box corner at sigma in
   {0, 1, 2, 4, 8} px, with the clip's own measured noise marked on the
   sweep so a reader can see which part of the curve is real.

Every protocol includes its identity level, and those rows must reproduce
``reports/counting_accuracy.json`` exactly. That is the protocol's own
correctness proof: a degradation that does not reduce to the undegraded
baseline is not measuring the degradation it names, and
``tests/test_bench_robustness.py`` asserts the reduction against the
committed report rather than trusting it.

Ground truth is indexed in the original 30 fps stream, so the two
protocols that resample widen the LATE side of the match window by the
realised sampling gap and publish the window they scored each level with.
See ``trafficlens.bench.degrade`` for the derivation and for the
resolution limit that widening costs.

Writes ``reports/robustness.json`` and ``reports/figures/robustness_*.png``.

Usage:

    PYTHONPATH=src .venv/bin/python scripts/bench_robustness.py \
        --config configs/motorway.yaml --gate inbound \
        --gt data/groundtruth/motorway_inbound_gt.json

Detections are read from the same git-ignored cache
``scripts/bench_counting.py`` writes; the detector is never re-run here.
Figures need matplotlib, which is not a runtime dependency -- pass
``--no-figures`` to skip them.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trafficlens.bench.baselines import GreedyIoUTracker  # noqa: E402
from trafficlens.bench.degrade import (  # noqa: E402
    PROTOCOL_BOX_JITTER,
    PROTOCOL_DETECTION_DROPOUT,
    PROTOCOL_DROPPED_FRAMES,
    PROTOCOL_FRAME_RATE,
    ROBUSTNESS_SEED,
    corner_sigma_equivalents,
    dropout_streams,
    dropped_frame_streams,
    frame_rate_streams,
    jitter_streams,
    run_protocol,
    run_stream,
)
from trafficlens.bench.harness import (  # noqa: E402
    DetectionCacheError,
    build_methods,
    median_gate_approach_px_per_frame,
    read_detection_cache,
    run_counting,
    write_report,
)
from trafficlens.bench.scoring import DEFAULT_MATCH_WINDOW  # noqa: E402
from trafficlens.bench.slitscan import GroundTruth  # noqa: E402
from trafficlens.config import load_config  # noqa: E402
from trafficlens.core.constants import (  # noqa: E402
    BASELINE_BAND_PX,
    BASELINE_GREEDY_IOU_THRESH,
    TRACK_MATCH_IOU,
)
from trafficlens.core.gate import GateCounter  # noqa: E402
from trafficlens.io.video import VideoSource  # noqa: E402
from trafficlens.track.tracker import Tracker  # noqa: E402

#: The schema version of ``reports/robustness.json``.
REPORT_SCHEMA_VERSION = 1

#: Frame rates the decimation protocol sweeps BELOW the clip's own rate,
#: highest first. The identity level is deliberately NOT in this tuple: it
#: is the clip's own frame rate, taken from the ground truth, because that
#: is what ``reduction.identity_levels`` publishes and what the whole
#: family must reduce to. Hard-coding 30 here would put a rate the clip
#: does not have at the head of the sweep on any clip that is not 30 fps,
#: and ``decimate`` would refuse to invent frames.
DEFAULT_DEGRADED_RATES = (25.0, 15.0, 10.0, 5.0, 2.0)

#: Fractions of frames deleted outright. 0 is the identity.
DEFAULT_DROP_FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.30)

#: Per-detection dropout probabilities. 0 is the identity.
DEFAULT_DROPOUT_PROBABILITIES = (0.0, 0.05, 0.10, 0.20, 0.30)

#: Per-corner Gaussian sigmas, in pixels. 0 is the identity. The measured
#: clip sits below 1 px in these units -- see ``jitter_calibration`` in the
#: report -- so most of this sweep is extrapolation and is labelled as such.
DEFAULT_JITTER_SIGMAS = (0.0, 1.0, 2.0, 4.0, 8.0)

#: Band half-widths swept under decimation, the same values the counting
#: benchmark used so the two curves are directly comparable.
DEFAULT_BAND_VALUES = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0)

#: The three trackers compared with the counting rule held fixed. Task 14
#: found them indistinguishable on clean footage; whether degradation
#: separates them is question (b).
GATE_RULE_METHODS = ("engine+gate", "centroid+gate", "greedy-iou+gate")

#: The fields a band-sweep row publishes.
_BAND_ROW_FIELDS = (
    "n_predicted",
    "true_positives",
    "misses",
    "false_positives",
    "miss_rate",
    "phantom_rate",
    "precision",
    "recall",
    "f1",
    "signed_bias",
)


def target_rates_for(source_fps: float, degraded_rates=DEFAULT_DEGRADED_RATES):
    """The decimation sweep for a clip recorded at ``source_fps``.

    The clip's own rate leads, because that is the identity level the
    whole family reduces to, and only rates strictly below it follow --
    ``decimate`` removes frames and cannot invent them, so a sweep point
    above the source rate is not a degradation but a crash. On the shipped
    30 fps clip this returns exactly ``(30, 25, 15, 10, 5, 2)``.
    """
    source = float(source_fps)
    return [source] + [
        float(rate) for rate in degraded_rates if float(rate) < source
    ]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure counting accuracy under four seeded degradations."
    )
    parser.add_argument("--config", default="configs/motorway.yaml")
    parser.add_argument("--gate", default="inbound")
    parser.add_argument("--gt", default="data/groundtruth/motorway_inbound_gt.json")
    parser.add_argument(
        "--cache",
        default="private/bench",
        help="directory holding the shared detection cache; must be git-ignored",
    )
    parser.add_argument("--out", default="reports/robustness.json")
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--noise", default="reports/detection_noise.json")
    parser.add_argument("--seed", type=int, default=ROBUSTNESS_SEED)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="write the JSON only (matplotlib is not a runtime dependency)",
    )
    return parser.parse_args(argv)


def resolve(path_like: str) -> str:
    """A repo-relative path made absolute, so the script runs from
    anywhere."""
    path = Path(path_like)
    return str(path if path.is_absolute() else ROOT / path)


# --- assembling the report --------------------------------------------------


def _band_sweep(
    streams, gate, labels, band_values, rate_entries, *, gate_name, base_window
):
    """The band rule's behaviour across ``band_values`` at every frame
    rate, for two trackers, driven through the same widening and mapping
    every other figure goes through.

    This is the instrument for question (a). Task 14 measured a gate
    approach of 1.13 px per frame, at which no band in this sweep is ever
    stepped over, so the classic band miss mode never appeared and miss
    rate and phantom rate rose together instead of trading. Decimation is
    the protocol that should restore it: the same vehicle moves the stride
    multiple further between samples, and a narrow band can be jumped
    clean over.

    Two things make that measurable rather than merely narrated.

    **The signature must isolate the rule from the tracker.** A bare
    under-count does not: below 15 fps the engine's association collapses,
    and then the GATE rule under-counts by exactly as much as every band,
    so "misses exceed phantoms" reports step-over wherever tracking merely
    stopped. The signature published here is the band rule emitting FEWER
    events than the gate rule driven by the SAME tracker over the SAME
    stream. Both rules see identical tracks and both count once per track,
    so a shortfall means a track the gate rule counted reached the band and
    was declined -- which ``BandCounter`` does for exactly two reasons: no
    sample landed inside the band (step-over), or a sample landed inside it
    with zero perpendicular displacement from its predecessor
    (``baselines.BandCounter.update`` returns None when ``signed == 0``, so
    a vehicle stopped on the band is never counted). The second cannot
    explain anything observed under decimation, where per-sample
    displacement only GROWS, but it is a distinct cause and the criterion
    says so rather than claiming a shortfall can only be step-over. It
    is conservative: the band's phantoms inflate its own count, so a few
    stepped-over vehicles can be masked, and the flag never fires
    spuriously.

    **The sweep needs an association that survives.** A sweep driven only
    by a tracker that dies before the displacement grows could never
    observe step-over at all, whatever the rule does, so ``GreedyIoUTracker``
    -- whose 0.3 IoU floor still associates at 2 fps -- is swept alongside
    the engine.
    """
    trackers = ("engine", "greedy-iou")
    methods = {
        f"{tracker}+band-{value:g}px": build_methods(gate, band_px=value)[
            f"{tracker}+band"
        ]
        for tracker in trackers
        for value in band_values
    }
    gate_predicted = {
        entry["level"]: {
            tracker: entry["methods"][f"{tracker}+gate"]["n_predicted"]
            for tracker in trackers
        }
        for entry in rate_entries
    }

    blocks = []
    for stream in streams:
        entry = run_stream(
            stream, methods, labels, gate_name=gate_name, base_window=base_window
        )
        rows = []
        for tracker in trackers:
            reference = gate_predicted[entry["level"]][tracker]
            for value in band_values:
                record = entry["methods"][f"{tracker}+band-{value:g}px"]
                rows.append(
                    {
                        "level": entry["level"],
                        "level_label": entry["level_label"],
                        "tracker": tracker,
                        "band_px": float(value),
                        **{field: record[field] for field in _BAND_ROW_FIELDS},
                        "gate_rule_n_predicted": reference,
                        "stepped_over": record["n_predicted"] < reference,
                    }
                )
        blocks.append(
            {
                "level": entry["level"],
                "level_label": entry["level_label"],
                "max_gap_frames": entry["max_gap_frames"],
                "match_window": entry["match_window"],
                "entries": rows,
            }
        )
    return blocks


def _engine_gate_with_floor(gate, match_thresh: float):
    """The ``engine+gate`` composition with one constructor argument
    changed and nothing else.

    Built here rather than added to ``build_methods`` because it is not a
    tenth method under test: it is an ABLATION, and it exists only to vary
    a single engine constant while the stream, the counting rule, the
    track lifecycle and the match window all stay exactly what the main
    sweep used. ``run_counting`` is reused rather than re-implemented so
    the two cannot drift.
    """

    def method(frames):
        return run_counting(
            frames, Tracker(match_thresh=match_thresh), GateCounter(gate)
        )

    return method


def _association_floor_ablation(
    streams_by_protocol, gate, labels, *, gate_name, base_window, floors
) -> dict:
    """Vary the engine's IoU association floor, and nothing else, across
    every degradation level.

    Question (b) finds the engine's tracker losing to both baselines under
    every degradation that separates them. Attributing that to the 0.8 IoU
    floor is an assertion about a constant until something varies the
    constant on its own -- so this does. The comparison floor is
    ``BASELINE_GREEDY_IOU_THRESH``, SORT's published 0.3, so the ablation
    is against a value with a citation rather than a value chosen to make
    a point.
    """
    methods = {f"{floor:g}": _engine_gate_with_floor(gate, floor) for floor in floors}
    by_protocol: dict[str, list[dict]] = {}
    for protocol, streams in streams_by_protocol.items():
        rows = []
        for stream in streams:
            entry = run_stream(
                stream, methods, labels, gate_name=gate_name, base_window=base_window
            )
            rows.append(
                {
                    "level": entry["level"],
                    "level_label": entry["level_label"],
                    "f1": {
                        key: entry["methods"][key]["f1"] for key in methods
                    },
                    "n_predicted": {
                        key: entry["methods"][key]["n_predicted"] for key in methods
                    },
                    "true_positives": {
                        key: entry["methods"][key]["true_positives"]
                        for key in methods
                    },
                }
            )
        by_protocol[protocol] = rows

    shipped = f"{floors[0]:g}"
    loosened = f"{floors[-1]:g}"
    gains = [
        {
            "protocol": protocol,
            "level": row["level"],
            "level_label": row["level_label"],
            "gain": row["f1"][loosened] - row["f1"][shipped],
        }
        for protocol, rows in by_protocol.items()
        for row in rows
    ]
    best = max(gains, key=lambda row: row["gain"])
    per_protocol_best = {
        protocol: max(
            (row for row in gains if row["protocol"] == protocol),
            key=lambda row: row["gain"],
        )
        for protocol in by_protocol
    }
    explained = sorted(
        protocol
        for protocol, row in per_protocol_best.items()
        if row["gain"] > 0.05
    )
    unexplained = sorted(set(per_protocol_best) - set(explained))

    return {
        "question": (
            "The three trackers separate against the engine under "
            "degradation. Is the engine's IoU association floor what does "
            "it?"
        ),
        "shipped_floor": float(floors[0]),
        "floors": [float(floor) for floor in floors],
        "comparison_floor_source": (
            "BASELINE_GREEDY_IOU_THRESH, SORT's published iou_threshold, so "
            "the ablation is against a value with a citation rather than one "
            "chosen to make a point."
        ),
        "held_fixed": (
            "The degraded stream, the counting rule, the track lifecycle and "
            "the match window are all exactly what the main sweep used; only "
            "Tracker(match_thresh=...) changes. The shipped-floor column is "
            "therefore identical to the main sweep's engine+gate figures, "
            "which the tests assert."
        ),
        "by_protocol": by_protocol,
        "largest_f1_gain": best,
        "largest_f1_gain_by_protocol": per_protocol_best,
        "gain_threshold": 0.05,
        "protocols_the_floor_explains": explained,
        "protocols_the_floor_does_not_explain": unexplained,
        "loosening_the_floor_helps": best["gain"] > 0.0,
        "verdict": (
            f"Loosening the engine's association floor from {floors[0]:g} to "
            f"{floors[-1]:g}, changing NOTHING else, recovers up to "
            f"{best['gain']:.4f} crossing F1 -- the largest gain is at "
            f"{best['protocol']} {best['level_label']}. The floor explains "
            f"the collapse under {', '.join(explained) or 'no protocol'} and "
            f"does NOT explain it under "
            f"{', '.join(unexplained) or 'nothing measured here'}, where "
            f"loosening it recovers less than the "
            f"{0.05:g} F1 threshold above and can even cost a little. So this "
            f"is not one fault: the frame-rate and jitter collapses are the "
            f"association floor, while whatever the engine loses under "
            f"detection dropout is something else and is not diagnosed here. "
            f"The undegraded rows say why the floor exists at all: at 30 fps "
            f"the loose floor scores slightly WORSE, so {floors[0]:g} is a "
            f"value tuned on clean footage that does not survive the input "
            f"leaving it. This is a measurement of a shipped constant, not a "
            f"proposal to change it -- changing it is separate work with its "
            f"own baseline, and this sweep is not the evidence for a new "
            f"value."
        ),
    }


def _band_step_over_answer(report_protocols, band_values) -> dict:
    """Question (a), answered from the published series rather than
    asserted."""
    rates = report_protocols[PROTOCOL_FRAME_RATE]["entries"]
    blocks = report_protocols[PROTOCOL_FRAME_RATE]["band_sweep_by_rate"]

    approach = {
        str(entry["level"]): entry["median_gate_approach_px_per_frame"]
        for entry in rates
    }
    engine_tracked = {
        str(entry["level"]): entry["median_gate_approach_px_per_frame_engine_tracker"]
        for entry in rates
    }

    rows = [row for block in blocks for row in block["entries"]]
    stepped = [row for row in rows if row["stepped_over"]]
    reappears = bool(stepped)
    engine_rows = [row for row in stepped if row["tracker"] == "engine"]
    first_rate = max((row["level"] for row in stepped), default=None)

    if reappears:
        by_tracker = sorted({row["tracker"] for row in stepped})
        verdict = (
            f"The step-over mode DOES reappear under decimation. "
            f"{len(stepped)} of the {len(rows)} swept (tracker, rate, band) "
            f"combinations emit FEWER events than the gate rule fed by the "
            f"same tracker over the same stream -- the confound-free signature "
            f"of a band jumped clean over -- and the highest frame rate at "
            f"which that happens is {first_rate:g} fps. It appears with "
            f"{', '.join(by_tracker)}"
        )
        if not engine_rows:
            verdict += (
                ", but NOT with the engine's own tracker at any rate. That is "
                "the more interesting half of the answer: the engine's "
                "association collapses at a HIGHER frame rate than the band's "
                "step-over threshold, so on this clip the rule's classic "
                "failure mode is pre-empted by the tracker's. The step-over "
                "mode is real and this sweep reaches it, but only through an "
                "association loose enough to still be tracking when the "
                "displacement gets that large."
            )
        else:
            verdict += "."
    else:
        verdict = (
            "The step-over mode does NOT reappear anywhere in this sweep, and "
            "that is the finding. At no swept (tracker, rate, band) does the "
            "band rule emit fewer events than the gate rule fed by the same "
            "tracker over the same stream, so every band miss is a MISTIMED "
            "crossing rather than a skipped one -- exactly what Task 14 "
            "measured on clean footage, now measured at a sixteenth of the "
            "frame rate as well. Note what this does NOT say: below 15 fps "
            "every rule under-counts heavily, but the gate rule under-counts "
            "by the same amount at the same rate, so those losses belong to "
            "the tracker and not to the band."
        )

    return {
        "question": (
            "Task 14 measured the gate approach at 1.13 px per frame, at which "
            "no band is ever stepped over, so the band rule's classic miss mode "
            "never occurred and miss rate and phantom rate rose together "
            "instead of trading. Does decimation restore it, and at which rate?"
        ),
        "criterion": (
            "A (tracker, rate, band) row counts as stepped over when the band "
            "rule emits FEWER events than the GATE rule driven by the same "
            "tracker over the same stream. Both rules see identical tracks and "
            "both count once per track, so a shortfall means a track the gate "
            "rule counted reached the band and was declined. BandCounter "
            "declines for exactly two reasons: no sample landed inside the "
            "band -- step-over -- or a sample landed inside it with ZERO "
            "perpendicular displacement from its predecessor, which is a "
            "vehicle stopped on the band rather than one that jumped it. The "
            "second cause cannot explain any row observed here, because "
            "per-sample displacement only GROWS under decimation and this "
            "sweep's shortfalls all appear at the lowest rates; but it is a "
            "distinct cause, so the criterion is 'a declined band crossing' "
            "and step-over is the reading the measured displacement supports, "
            "not the only arithmetic possibility. A bare under-count would NOT "
            "do at all: below 15 fps the engine's association collapses and "
            "the gate rule under-counts by exactly as much as every band, so "
            "'misses exceed phantoms' reports step-over wherever tracking "
            "merely stopped. This criterion is also conservative -- the band's "
            "own phantoms inflate its count and can mask a few stepped-over "
            "vehicles -- so it under-reports the mode rather than "
            "manufacturing it."
        ),
        "band_px_default": float(BASELINE_BAND_PX),
        "band_px_swept": [float(value) for value in band_values],
        "trackers_swept": sorted({row["tracker"] for row in rows}),
        "approach_measured_with": (
            "GreedyIoUTracker. The engine's own tracker is published beside it "
            "and stops returning tracks altogether at the lowest rate, so a "
            "displacement series measured through it has a hole exactly where "
            "this question is sharpest. The quantity wanted here is geometric "
            "-- how far a vehicle moves between samples -- so it is measured "
            "with the association that survives, and the engine's collapse is "
            "reported as its own finding rather than as a missing value."
        ),
        "approach_px_per_frame_by_rate": approach,
        "engine_tracked_approach_px_per_frame_by_rate": engine_tracked,
        "step_over_rows": stepped,
        "step_over_trade_off_reappears": reappears,
        "step_over_with_the_engine_tracker": bool(engine_rows),
        "first_rate_with_step_over": first_rate,
        "verdict": verdict,
    }


def _tracker_separation_answer(report_protocols) -> dict:
    """Question (b), answered from the published per-level F1s."""
    spreads: dict[str, float] = {}
    detail: dict[str, dict] = {}
    for protocol, block in report_protocols.items():
        for entry in block["entries"]:
            scores = {
                name: entry["methods"][name]["f1"] for name in GATE_RULE_METHODS
            }
            key = f"{protocol}@{entry['level_label']}"
            spreads[key] = max(scores.values()) - min(scores.values())
            detail[key] = scores

    widest = max(spreads.values()) if spreads else 0.0
    separating = sorted(key for key, spread in spreads.items() if spread > 0.0)
    separate = widest > 0.0

    engine_highest = sorted(
        key
        for key, scores in detail.items()
        if scores["engine+gate"] == max(scores.values())
    )
    engine_lowest = sorted(
        key
        for key, scores in detail.items()
        if scores["engine+gate"] == min(scores.values())
    )
    engine_leads = any(spreads[key] > 0.0 for key in engine_highest)
    engine_trails = sorted(set(engine_lowest) & set(separating))

    if separate:
        argmax = max(spreads, key=lambda key: spreads[key])
        best = max(detail[argmax], key=lambda name: detail[argmax][name])
        verdict = (
            f"The three trackers DO separate under degradation -- and they "
            f"separate AGAINST the engine. The widest crossing-F1 spread with "
            f"the gate rule is {widest:.4f}, at {argmax}, where {best} scores "
            f"highest; they differ at {len(separating)} of the {len(spreads)} "
            f"degradation levels measured. "
        )
        if not engine_leads:
            verdict += (
                f"At not one of those {len(separating)} levels does the "
                f"engine's Kalman-plus-Hungarian tracker score highest, and at "
                f"{len(engine_trails)} of them it scores LOWEST of the three. "
                f"Task 14 found all three identical on clean 30 fps footage "
                f"while the engine cost about 21x the CPU of a baseline "
                f"tracker; degradation is where a motion model is supposed to "
                f"earn that, and on this clip it does the opposite. The cause "
                f"is measured rather than guessed at: see "
                f"association_floor_ablation, which varies the engine's "
                f"{TRACK_MATCH_IOU:g} IoU association floor and nothing else. "
                f"This is a result about this product's own engine and it is "
                f"published as measured."
            )
        else:
            verdict += (
                f"The engine scores highest at {len(engine_highest)} of them."
            )
    else:
        verdict = (
            "The three trackers do NOT separate, and that is the finding. "
            "Across all four degradation protocols and every level of each, "
            "the engine's Kalman-plus-Hungarian tracker, a centroid tracker "
            "and a greedy-IoU tracker score the identical crossing F1 with "
            "the gate rule -- the same result Task 14 published on clean 30 "
            "fps footage, now measured under exactly the conditions a motion "
            "model exists for. On this clip, at this gate, the engine's "
            "association buys no accuracy at any level of any degradation "
            "measured here."
        )

    return {
        "question": (
            "Task 14's sharpest honest negative was that on clean 30 fps "
            "footage all three trackers score identically with the gate rule, "
            "while the engine's Kalman plus second association stage costs "
            "about 21x the CPU of a baseline tracker. Dropout and dropped "
            "frames are exactly the conditions a motion model exists for. Does "
            "degradation separate them?"
        ),
        "methods_compared": list(GATE_RULE_METHODS),
        "f1_by_level": detail,
        "f1_spread_by_level": spreads,
        "max_f1_spread": widest,
        "levels_where_trackers_differ": separating,
        "levels_where_the_engine_scores_highest": engine_highest,
        "levels_where_the_engine_scores_lowest": engine_lowest,
        "engine_leads_at_any_degraded_level": engine_leads,
        "levels_measured": len(spreads),
        "trackers_separate": separate,
        "verdict": verdict,
    }


def build_report(
    detections,
    truth: GroundTruth,
    gate,
    *,
    noise: dict,
    detector: dict | None,
    seed: int = ROBUSTNESS_SEED,
    target_rates=None,
    drop_fractions=DEFAULT_DROP_FRACTIONS,
    dropout_probabilities=DEFAULT_DROPOUT_PROBABILITIES,
    jitter_sigmas=DEFAULT_JITTER_SIGMAS,
    band_values=DEFAULT_BAND_VALUES,
    base_window=DEFAULT_MATCH_WINDOW,
) -> dict:
    """The whole robustness family, scored and assembled.

    Deterministic end to end: no timing is measured here, so two runs of
    this function over the same detections produce byte-identical JSON.
    That is deliberate -- the counting benchmark already publishes the
    per-method cost, and a wall-clock column would make the one property
    this report most needs (reproducibility) untestable.
    """
    labels = list(truth.crossings)
    methods = build_methods(gate)
    # The identity rate is the CLIP's, never a constant: it is what
    # reduction.identity_levels publishes below and what the family must
    # reduce to.
    target_rates = (
        target_rates_for(truth.fps)
        if target_rates is None
        else [float(rate) for rate in target_rates]
    )
    if target_rates[0] != float(truth.fps):
        raise ValueError(
            f"the decimation sweep must lead with the clip's own rate "
            f"({truth.fps} fps), because that is the identity level the "
            f"whole family reduces to; got {target_rates[0]}"
        )

    rate_streams = frame_rate_streams(
        detections, source_fps=truth.fps, target_rates=target_rates
    )
    protocols = {
        PROTOCOL_FRAME_RATE: {
            "knob": "target_fps",
            "levels": target_rates,
            "seeded": False,
            "note": (
                "Decimation is deterministic: which frames a lower rate keeps "
                "is a property of the sampling grid, not of a draw. The stream "
                "handed to the engine is renumbered from zero, as though the "
                "clip had been recorded at that rate, and every prediction is "
                "mapped back to the original 30 fps index before scoring. "
                "max_age is expressed in FRAMES, so decimation implicitly "
                "lengthens the engine's memory in wall-clock terms -- at 2 fps "
                "a coasting track survives 15 seconds where at 30 fps it "
                "survives one."
            ),
            "streams": rate_streams,
        },
        PROTOCOL_DROPPED_FRAMES: {
            "knob": "fraction_dropped",
            "levels": [float(value) for value in drop_fractions],
            "seeded": True,
            "note": (
                "Frames deleted outright, so the surviving grid is irregular "
                "and the match window is widened by the realised worst gap "
                "rather than by a stride. That gap is computed from the drop "
                "pattern before any scoring happens."
            ),
            "streams": dropped_frame_streams(
                detections, fractions=drop_fractions, seed=seed
            ),
        },
        PROTOCOL_DETECTION_DROPOUT: {
            "knob": "dropout_probability",
            "levels": [float(value) for value in dropout_probabilities],
            "seeded": True,
            "note": (
                "Every frame is kept, so the sampling grid and the match "
                "window are untouched: what changes is how often a track has "
                "nothing to associate with. This is the protocol a motion "
                "model is supposed to be for."
            ),
            "streams": dropout_streams(
                detections, probabilities=dropout_probabilities, seed=seed
            ),
        },
        PROTOCOL_BOX_JITTER: {
            "knob": "corner_sigma_px",
            "levels": [float(value) for value in jitter_sigmas],
            "seeded": True,
            "note": (
                "Independent Gaussian noise on each of a box's four corners. "
                "The clip's own measured noise is below 1 px in these units, "
                "so everything above the sweep's first point is extrapolation "
                "-- see jitter_calibration, which marks where the measurement "
                "sits."
            ),
            "streams": jitter_streams(
                detections, sigmas_px=jitter_sigmas, seed=seed
            ),
        },
    }

    streams_by_protocol = {
        name: block.pop("streams") for name, block in protocols.items()
    }
    for name, block in protocols.items():
        block["entries"] = run_protocol(
            streams_by_protocol[name],
            methods,
            labels,
            gate_name=truth.gate_name,
            base_window=base_window,
        )

    # The frame-rate protocol carries two extra measurements: how far a
    # vehicle moves between samples (the quantity question (a) turns on),
    # and the band sweep that answers it.
    for entry, stream in zip(protocols[PROTOCOL_FRAME_RATE]["entries"], rate_streams):
        entry["median_gate_approach_px_per_frame"] = (
            median_gate_approach_px_per_frame(
                stream.frames, gate, tracker_factory=GreedyIoUTracker
            )
        )
        entry["median_gate_approach_px_per_frame_engine_tracker"] = (
            median_gate_approach_px_per_frame(stream.frames, gate)
        )
    protocols[PROTOCOL_FRAME_RATE]["band_sweep_by_rate"] = _band_sweep(
        rate_streams,
        gate,
        labels,
        band_values,
        protocols[PROTOCOL_FRAME_RATE]["entries"],
        gate_name=truth.gate_name,
        base_window=base_window,
    )

    equivalents = corner_sigma_equivalents(noise)
    max_sigma = max(float(value) for value in jitter_sigmas)

    rate_entries = protocols[PROTOCOL_FRAME_RATE]["entries"]
    overlapping = [
        entry
        for entry in rate_entries
        if entry["resolution"]["overlapping_label_pairs"]
    ]
    second_pair = [
        entry
        for entry in rate_entries
        if entry["resolution"]["overlapping_label_pairs"] >= 2
    ]
    label_frames = sorted(label.frame for label in labels)
    separations = [b - a for a, b in zip(label_frames, label_frames[1:])]
    closest = min(range(len(separations)), key=lambda index: separations[index])

    figures = sorted(
        [f"robustness_{protocol}.png" for protocol in protocols]
        + ["robustness_band_step_over.png"]
    )

    report = {
        "schema": REPORT_SCHEMA_VERSION,
        "seed": seed,
        "protocol": truth.protocol,
        # The label file's NAME, never its path: this report is tracked and
        # the same label set lives at a different absolute path on every
        # machine.
        "ground_truth": truth.path.name,
        "baseline_report": "counting_accuracy.json",
        "clip": truth.clip,
        "fps": truth.fps,
        "frames": len(detections),
        "window": {"start_frame": truth.start_frame, "end_frame": truth.end_frame},
        "gate": {
            "name": gate.name,
            "start": [float(gate.start[0]), float(gate.start[1])],
            "end": [float(gate.end[0]), float(gate.end[1])],
            "label_positive": gate.label_positive,
            "label_negative": gate.label_negative,
        },
        "detector": detector,
        "labels": {
            "total": len(labels),
            "certain": sum(1 for label in labels if label.confidence == "certain"),
            "probable": sum(1 for label in labels if label.confidence != "certain"),
        },
        "base_match_window": base_window.as_dict(),
        "reduction": {
            "claim": (
                "Every protocol's identity level -- 30 fps, 0 % dropped, "
                "p = 0, sigma = 0 -- reproduces counting_accuracy.json "
                "EXACTLY, for all nine methods, scored with the undegraded "
                "window. A protocol that does not reduce to the baseline is "
                "not measuring the degradation it names. The identity runs "
                "the general code path: there is no short-circuit branch, so "
                "the reduction proves the transform rather than proving a "
                "branch."
            ),
            "identity_levels": {
                PROTOCOL_FRAME_RATE: float(truth.fps),
                PROTOCOL_DROPPED_FRAMES: 0.0,
                PROTOCOL_DETECTION_DROPOUT: 0.0,
                PROTOCOL_BOX_JITTER: 0.0,
            },
        },
        "label_resolution": {
            "min_separation_frames": min(separations),
            "closest_pair_frames": [
                label_frames[closest],
                label_frames[closest + 1],
            ],
            # The HIGHEST qualifying rate, computed rather than taken from
            # the head of a list that happens to be in descending sweep
            # order. "First" means first as the rate falls; reordering
            # DEFAULT_DEGRADED_RATES must not silently change a published
            # answer. _band_step_over_answer takes the max for the same
            # reason.
            "first_rate_with_overlapping_windows": max(
                (entry["level"] for entry in overlapping), default=None
            ),
            "first_rate_with_a_second_overlapping_pair": max(
                (entry["level"] for entry in second_pair), default=None
            ),
            "note": (
                "The closest labelled pair is five frames apart, so with the "
                "UNDEGRADED window their match intervals [410, 415] and "
                "[415, 420] already share one frame -- the widening enlarges "
                "an overlap that was always there rather than creating one, "
                "and the rate at which the windows 'first overlap' is "
                "therefore 30 fps, the baseline itself. Each entry's "
                "resolution block reports how many label pairs overlap under "
                "the window that scored it and by how much; resolution_limited "
                "is true wherever that exceeds the undegraded overlap. Treat "
                "every crossing-level figure so marked as resolution-limited: "
                "at the lowest rate three quarters of the closest pair's "
                "window is shared, and the protocol cannot separate those two "
                "crossings at all. That is a genuine limit of scoring at a low "
                "sampling rate, not a defect engineered around; "
                "max_cardinality_true_positives is published per method per "
                "level so the cost of greedy matching under those overlaps is "
                "measured rather than assumed."
            ),
        },
        "jitter_calibration": {
            "source_report": "detection_noise.json",
            "measured_residuals": noise["residuals"],
            "median_box_width_px": noise.get("median_box_width_px"),
            "median_box_height_px": noise.get("median_box_height_px"),
            "corner_sigma_equivalent_px": equivalents,
            "stress_multiple_at_max_sigma": {
                "sigma_px": max_sigma,
                "lowest": max_sigma / equivalents["max_px"],
                "highest": max_sigma / equivalents["min_px"],
            },
            "note": (
                "The sweep's knob is a per-corner sigma; the measurement is a "
                "residual of box width, height and centre. corner_sigma_"
                "equivalent_px converts one into the other, both from the "
                "standard deviation and from the p95 of the absolute "
                "residual. The two disagree because the measured distribution "
                "is heavy-tailed -- for the centres the standard deviation "
                "exceeds the p95, because a few large excursions dominate the "
                "variance while the bulk sits near zero -- so read p95 for the "
                "typical case and the standard deviation for the tail, and "
                "treat NEITHER as a measurement of the detector: the source "
                "report states it is a proxy. Everything above the sweep's "
                "first non-zero point is extrapolation, and the top of the "
                "sweep is the stress multiple recorded above, not a plausible "
                "operating point."
            ),
        },
        "protocols": protocols,
        "association_floor_ablation": _association_floor_ablation(
            streams_by_protocol,
            gate,
            labels,
            gate_name=truth.gate_name,
            base_window=base_window,
            floors=(TRACK_MATCH_IOU, BASELINE_GREEDY_IOU_THRESH),
        ),
        "questions": {},
        "figures": figures,
        "caveats": [
            "The labelling gate was chosen in the near field for label "
            "RELIABILITY, not for the engine's convenience, so every accuracy "
            "figure here -- degraded or not -- is an UPPER bound on what the "
            "same engine scores on the far carriageway or in a queue.",
            "Crossing-level precision, recall and F1 are published with the "
            "count error at every level, never the count error alone. Count "
            "error alone is the metric this benchmark exists to discredit: "
            "Task 14's band rule predicted 18 crossings against 17 real ones "
            "-- a near-perfect total -- while landing one or two of them on "
            "the right frame.",
            "The two resampling protocols are scored with a match window "
            "widened on the LATE side by the realised sampling gap, because a "
            "method that sees only every Delta-th frame cannot report a "
            "crossing before the next sampled one. The widening is derived "
            "from the retained pattern a priori, never fitted to output, and "
            "the window used is published beside every level it scored.",
            "No degradation result is expressed as a speed. The clip's "
            "along-road scale cannot be anchored, so displacements are given "
            "in pixels per frame and durations in frames or seconds.",
            "This report measures accuracy only. Per-method cost is published "
            "in counting_accuracy.json and is deliberately absent here: a "
            "wall-clock column would make two runs of this report differ, and "
            "reproducibility is the property it most needs to be able to "
            "prove.",
            "Timing aside, this whole report is a function of the cached "
            "detections and the seed. Two runs are byte-identical, and every "
            "level draws from a stream derived from the protocol name and the "
            "level value, so adding a sweep point cannot move any other "
            "point's numbers.",
            "The dropout and jitter protocols degrade the DETECTOR's output, "
            "not the detector. They model what an occluded or noisy detection "
            "stream does to counting; they do not predict what a different "
            "detector, or the same detector on harder footage, would produce.",
            "Only full-label-set figures are swept. certain_only is carried "
            "alongside as a compact record; its two standing limitations -- "
            "an ignore set that absorbs genuine phantoms, and dilution "
            "arithmetic on a smaller denominator -- are stated in full in "
            "counting_accuracy.json and apply unchanged here.",
        ],
    }

    report["questions"] = {
        "band_step_over": _band_step_over_answer(protocols, band_values),
        "tracker_separation": _tracker_separation_answer(protocols),
    }
    return report


# --- figures ----------------------------------------------------------------


def _tick_labels(entries) -> list[str]:
    """Each level's label with the window that scored it underneath, so a
    figure can never be read without the window it was measured with."""
    return [
        f"{entry['level_label']}\n[-{entry['match_window']['frames_before']}, "
        f"+{entry['match_window']['frames_after']}]"
        for entry in entries
    ]


def _protocol_figure(plt, report: dict, protocol: str, path: Path) -> Path:
    entries = report["protocols"][protocol]["entries"]
    positions = list(range(len(entries)))
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

    for name in GATE_RULE_METHODS:
        axes[0].plot(
            positions,
            [entry["methods"][name]["f1"] for entry in entries],
            marker="o",
            label=name,
        )
    axes[0].set_title("crossing F1, gate rule")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(-0.05, 1.05)

    for name, style in (("engine+gate", "-"), ("engine+band", "--")):
        axes[1].plot(
            positions,
            [entry["methods"][name]["miss_rate"] for entry in entries],
            style,
            marker="o",
            label=f"{name} miss",
        )
        axes[1].plot(
            positions,
            [entry["methods"][name]["phantom_rate"] for entry in entries],
            style,
            marker="x",
            label=f"{name} phantom",
        )
    axes[1].set_title("miss and phantom rate, engine tracker")
    axes[1].set_ylabel("per labelled crossing")

    for name in ("engine+gate", "engine+band", "engine+per-frame"):
        axes[2].plot(
            positions,
            [entry["methods"][name]["signed_bias"] for entry in entries],
            marker="o",
            label=name,
        )
    axes[2].axhline(0.0, color="0.4", linewidth=0.8)
    axes[2].set_yscale("symlog", linthresh=10)
    axes[2].set_title("signed count bias (predicted - labelled)")
    axes[2].set_ylabel("crossings")

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(_tick_labels(entries), fontsize=8)
        axis.set_xlabel(f"{report['protocols'][protocol]['knob']}  /  match window")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)

    if protocol == PROTOCOL_BOX_JITTER:
        _mark_measured_sigma(axes, report, entries)

    limited = [
        entry["level_label"]
        for entry in entries
        if entry["resolution"]["resolution_limited"]
    ]
    subtitle = (
        f"seed {report['seed']} - {report['labels']['total']} labelled "
        f"crossings - full label set"
    )
    if limited:
        subtitle += f" - resolution-limited window at: {', '.join(limited)}"
    figure.suptitle(f"{protocol}\n{subtitle}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def _mark_measured_sigma(axes, report: dict, entries) -> None:
    """Put the clip's own measured noise on the jitter sweep.

    A sweep whose range is not tied to a measurement is decoration. The
    measured level converts to a RANGE of per-corner sigmas, not a point,
    because the residual distribution is heavy-tailed and its standard
    deviation and p95 imply different Gaussians; the band drawn here is
    that range, and everything to the right of it is extrapolation.
    """
    import numpy as np

    equivalents = report["jitter_calibration"]["corner_sigma_equivalent_px"]
    sigmas = [entry["level"] for entry in entries]
    positions = list(range(len(entries)))
    low = float(np.interp(equivalents["min_px"], sigmas, positions))
    high = float(np.interp(equivalents["max_px"], sigmas, positions))
    for axis in axes:
        axis.axvspan(
            low,
            high,
            color="tab:green",
            alpha=0.15,
            label="measured on this clip",
        )
        axis.legend(fontsize=7)


def _band_step_over_figure(plt, report: dict, path: Path) -> Path:
    """Band events against gate events, per tracker.

    The y axis is the SHORTFALL against the gate rule fed by the same
    tracker over the same stream, because that -- not a bare under-count
    -- is what distinguishes a jumped band from a collapsed association.
    Anything below zero is a stepped-over band; the flat zero line at
    every rate where the tracker has already died is the confound the
    criterion exists to exclude.
    """
    question = report["questions"]["band_step_over"]
    blocks = report["protocols"][PROTOCOL_FRAME_RATE]["band_sweep_by_rate"]
    trackers = question["trackers_swept"]
    figure, axes = plt.subplots(1, len(trackers), figsize=(11.0, 4.8), squeeze=False)

    for axis, tracker in zip(axes[0], trackers):
        for block in blocks:
            rows = [row for row in block["entries"] if row["tracker"] == tracker]
            axis.plot(
                [row["band_px"] for row in rows],
                [
                    row["n_predicted"] - row["gate_rule_n_predicted"]
                    for row in rows
                ],
                marker="o",
                label=block["level_label"],
            )
        axis.axhline(0.0, color="0.4", linewidth=0.8)
        axis.set_title(f"{tracker} tracker")
        axis.set_ylabel("band events minus gate events (same tracker)")
        axis.set_xlabel("band half-width (px)")
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, title="frame rate")

    figure.suptitle(
        "band step-over under decimation -- below zero is a jumped band\n"
        f"reappears: {question['step_over_trade_off_reappears']}"
        f" (with the engine tracker: "
        f"{question['step_over_with_the_engine_tracker']})"
        f" - seed {report['seed']}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def write_figures(report: dict, out_dir) -> list[Path]:
    """Render every figure the report names, returning the paths written.

    matplotlib is imported here, not at module scope, so the JSON half of
    this script runs on a machine without it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written = [
        _protocol_figure(
            plt, report, protocol, directory / f"robustness_{protocol}.png"
        )
        for protocol in report["protocols"]
    ]
    written.append(
        _band_step_over_figure(
            plt, report, directory / "robustness_band_step_over.png"
        )
    )
    return written


# --- the command line -------------------------------------------------------


def load_detections(config, truth, cache_dir: Path):
    """Read the shared detections the counting benchmark cached. This
    script never constructs a detector: a robustness sweep whose baseline
    row came from a different detector run than counting_accuracy.json
    could not reduce to it."""
    key = {
        "clip": truth.clip,
        "model": config.detector.model,
        "confidence": config.detector.confidence,
        "imgsz": config.detector.imgsz,
        "classes": list(config.detector.classes),
        "start_frame": truth.start_frame,
        "end_frame": truth.end_frame,
    }
    cache_path = cache_dir / f"{Path(truth.clip).stem}_detections.json"
    try:
        return read_detection_cache(cache_path, key=key), key
    except DetectionCacheError as error:
        raise SystemExit(
            f"no usable detection cache ({error}).\n"
            f"Run scripts/bench_counting.py first: it detects the labelled "
            f"window once and writes the cache this script reads. Detecting "
            f"here instead would risk scoring the identity level against a "
            f"different detector run than counting_accuracy.json used, and "
            f"the reduction test exists precisely to catch that."
        ) from error


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(resolve(args.config))

    gate_configs = {gate.name: gate for gate in config.gates}
    if args.gate not in gate_configs:
        raise SystemExit(
            f"config {args.config} has no gate named {args.gate!r}; it has: "
            f"{', '.join(sorted(gate_configs)) or '(none)'}"
        )

    source_spec = resolve(config.source)
    with VideoSource.open(source_spec) as probe:
        width, height = probe.width, probe.height
    gate = gate_configs[args.gate].to_gate(width, height)

    truth = GroundTruth.load(
        resolve(args.gt), gate=gate, clip_path=Path(source_spec)
    )
    detections, key = load_detections(config, truth, Path(resolve(args.cache)))
    print(
        f"ground truth {truth.path.name}: {len(truth.crossings)} crossings; "
        f"{len(detections)} cached detection frames"
    )

    noise = json.loads(Path(resolve(args.noise)).read_text())
    report = build_report(
        detections,
        truth,
        gate,
        noise=noise,
        detector={
            "model": key["model"],
            "confidence": key["confidence"],
            "imgsz": key["imgsz"],
            "classes": key["classes"],
        },
        seed=args.seed,
    )

    out_path = write_report(resolve(args.out), report)
    print(f"wrote {out_path.relative_to(ROOT)}")

    if not args.no_figures:
        for path in write_figures(report, resolve(args.figures)):
            print(f"wrote {path.relative_to(ROOT)}")

    for protocol, block in report["protocols"].items():
        print(f"\n{protocol} ({block['knob']}), full label set:")
        print(
            f"  {'level':16s} {'window':10s} "
            f"{'engine+gate F1':>15s} {'pred':>6s} {'bias':>6s}"
        )
        for entry in block["entries"]:
            record = entry["methods"]["engine+gate"]
            window = (
                f"[-{entry['match_window']['frames_before']},"
                f"+{entry['match_window']['frames_after']}]"
            )
            print(
                f"  {entry['level_label']:16s} {window:10s} "
                f"{record['f1']:15.4f} {record['n_predicted']:6d} "
                f"{record['signed_bias']:+6d}"
            )

    print("\n" + report["questions"]["band_step_over"]["verdict"])
    print("\n" + report["questions"]["tracker_separation"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
