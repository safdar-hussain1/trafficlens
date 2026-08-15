"""Running the engine and every standard failure mode over one shared
detection stream, and building the reports.

The scoring half lives in ``trafficlens.bench.scoring``: what a match is,
what the numbers mean, and the asymmetric match window are all decided
there, by a module that imports no tracker and no detector. This module
is the other half -- it composes trackers with counting rules, drives
them over cached detections, times them, and assembles the JSON. The two
were split at the whole-branch review's seam when the combined file
passed 1200 lines.

Timing
------
One method per measurement. A bracket that ran every method and then
indexed one out would report a sum in disguise -- and so would a single
bracket opened once and read after each method, which yields a running
total rather than all-equal values and slips past any check that only
compares methods against each other. Each method here is timed alone,
against its own injected cost in test. The measurement covers the tracker
and the counting rule ONLY: detections are read from a cache, so the
detector's cost is excluded and is identical for every method by
construction.

Identical detections
--------------------
Every method is handed the same decoded stream. A method that re-ran the
detector could differ from another by detector variance rather than by
its counting rule or its tracker, so the guarantee is structural: this
module never constructs a detector and never opens a model. The script
``scripts/bench_counting.py`` runs the detector once, caches it, and
passes the stream in.

Composition
-----------
``build_methods`` composes {tracker} x {counting rule}. That is the point
of the two-family split in ``trafficlens.bench.baselines``: holding the
tracker fixed and swapping the rule isolates the RULE, holding the rule
fixed and swapping the tracker isolates the TRACKER.

numpy + stdlib + ``trafficlens.core`` / ``.detect`` / ``.track`` /
``.bench`` / ``.pipeline`` only. No torch, no ultralytics: the benchmark
must run wherever the labels do.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np

from trafficlens.bench.baselines import (
    BandCounter,
    CentroidTracker,
    GreedyIoUTracker,
    PerFrameCounter,
    _signed_offset,
)
from trafficlens.bench.scoring import (
    DEFAULT_MATCH_WINDOW,
    MATCH_WINDOW_REASON,
    MatchResult,
    MatchWindow,
    match_crossings,
    max_cardinality_true_positives,
    restrict_to_adjudicated,
)
from trafficlens.bench.slitscan import Crossing, GroundTruth
from trafficlens.core.constants import BASELINE_BAND_PX
from trafficlens.core.gate import CrossingEvent, Gate, GateCounter
from trafficlens.detect.base import Detection
from trafficlens.pipeline import TrackReaper
from trafficlens.track.tracker import Tracker

#: One frame of the shared detection stream: ``(frame_index, timestamp,
#: detections)`` -- the same triple ``trafficlens.io.video.VideoSource``
#: yields, with the decoded frame replaced by its detections.
FrameDetections = tuple[int, float, list[Detection]]

#: The schema version of ``reports/counting_accuracy.json``.
REPORT_SCHEMA_VERSION = 1

#: Frames in the centred median filter ``measure_detection_noise`` takes a
#: trajectory's residual around. Must be odd. Five frames is a sixth of a
#: second at 30 fps: long enough that per-frame box jitter does not steer
#: the filter, short enough that genuine acceleration is not charged to
#: the residual as noise.
NOISE_MEDIAN_FILTER_FRAMES = 5

__all__ = [
    "DEFAULT_MATCH_WINDOW",
    "MATCH_WINDOW_REASON",
    "MatchResult",
    "MatchWindow",
    "match_crossings",
    "max_cardinality_true_positives",
    "restrict_to_adjudicated",
    "FrameDetections",
    "NOISE_MEDIAN_FILTER_FRAMES",
    "REPORT_SCHEMA_VERSION",
    "build_methods",
    "measure_detection_noise",
    "median_gate_approach_px_per_frame",
    "read_detection_cache",
    "run_counting",
    "run_counting_benchmark",
    "sweep_band_px",
    "write_detection_cache",
    "write_report",
    "DetectionCacheError",
]

# --- driving one tracker and one counting rule over cached detections -------


def run_counting(
    detections: Sequence[FrameDetections], tracker, counter
) -> list[CrossingEvent]:
    """Play ``detections`` through ``tracker`` and ``counter`` and return
    the ``CrossingEvent``s, with the pipeline's per-track lifecycle exactly.

    That lifecycle is not incidental detail. ``Tracker.update`` returns
    only confirmed tracks a detection updated THIS frame, so two things
    follow, both of which this reproduces from
    ``trafficlens.pipeline.run_session``:

    1. The previous anchor is ours to keep, and a track appearing for the
       first time has none, so it cannot cross. Seeding a missing previous
       anchor with anything at all would fabricate a swept segment across
       most of the frame and count crossings that never happened.
    2. Deaths are invisible, so tracks are reaped on a clock -- the same
       ``TrackReaper`` the pipeline uses, imported rather than
       re-implemented so the two can never drift. Reaping calls
       ``counter.forget`` and drops our own anchor entry; without it every
       counting rule accumulates a permanent per-track record for the
       length of the clip.

    ``max_age`` is read off the tracker, so a baseline tracker with a
    different lifetime is reaped on its own clock rather than the engine's.
    """
    events: list[CrossingEvent] = []
    reaper = TrackReaper(tracker.max_age)
    previous_anchor: dict[int, tuple[float, float]] = {}

    for frame_index, timestamp, frame_detections in detections:
        tracks = tracker.update(frame_detections, frame_index)
        for track in tracks:
            track_id = track.track_id
            anchor = track.anchor
            reaper.saw(track_id, frame_index)

            previous = previous_anchor.get(track_id)
            if previous is not None:
                event = counter.update(
                    track_id,
                    track.class_name,
                    previous,
                    anchor,
                    frame_index,
                    timestamp,
                )
                if event is not None:
                    events.append(event)
            previous_anchor[track_id] = anchor

        for dead_id in reaper.reap(frame_index):
            counter.forget(dead_id)
            previous_anchor.pop(dead_id, None)

    for dead_id in reaper.drain():
        counter.forget(dead_id)
    previous_anchor.clear()
    return events


#: The three trackers under test, by report name. ``engine`` is the
#: two-stage Kalman tracker; the other two are standard failure modes.
TRACKERS: dict[str, Callable[[], object]] = {
    "engine": Tracker,
    "centroid": CentroidTracker,
    "greedy-iou": GreedyIoUTracker,
}

#: The three counting rules under test, by report name. ``gate`` is the
#: engine's swept-segment rule; the other two are standard failure modes.
COUNTING_RULES: tuple[str, ...] = ("gate", "band", "per-frame")


def build_methods(
    gate: Gate, *, band_px: float = BASELINE_BAND_PX
) -> dict[str, Callable[[Sequence[FrameDetections]], list[CrossingEvent]]]:
    """Every {tracker} x {counting rule} composition, keyed
    ``"<tracker>+<rule>"``.

    The composition is the point of the two-family split in
    ``trafficlens.bench.baselines``: holding the tracker fixed and swapping
    the rule isolates the RULE, holding the rule fixed and swapping the
    tracker isolates the TRACKER. A single collapsed interface could not
    attribute a count error to either.

    Each returned callable builds a fresh tracker and a fresh counter on
    every call, so two runs of the same method are independent.
    """

    def make_counter(rule: str):
        if rule == "gate":
            return GateCounter(gate)
        if rule == "band":
            return BandCounter(gate, band_px)
        return PerFrameCounter(gate, band_px)

    def compose(tracker_factory, rule: str):
        def method(detections: Sequence[FrameDetections]) -> list[CrossingEvent]:
            return run_counting(detections, tracker_factory(), make_counter(rule))

        return method

    return {
        f"{tracker_name}+{rule}": compose(tracker_factory, rule)
        for tracker_name, tracker_factory in TRACKERS.items()
        for rule in COUNTING_RULES
    }


# --- the benchmark ----------------------------------------------------------


def _caveats(truth: GroundTruth) -> list[str]:
    """The statements that travel with every figure in the report.

    They are data, not prose in a README, because a number copied out of
    this file without them is a number that has been misread.
    """
    directions = {label.direction for label in truth.crossings}
    caveats = [
        "The labelling gate was chosen in the near field for label "
        "RELIABILITY, not for the engine's convenience: its traffic is the "
        "largest, best separated and least foreshortened in the frame. "
        "Every accuracy figure here is therefore an UPPER bound on what the "
        "same engine scores on the far carriageway, in the distance, or in "
        "a queue.",
        "Timing covers the tracker and the counting rule only. Detections "
        "are read from a cache, so the detector's cost -- by far the "
        "largest per-frame cost in a real session -- is excluded, and is "
        "identical for every method by construction.",
        "The per-frame counting rule's over-count is a LOWER bound: both "
        "band rules return no event on zero perpendicular displacement, so "
        "a vehicle stopped on the gate emits nothing at all where the "
        "failure mode would otherwise be at its worst.",
        "An identity swap does not have a fixed cost in counts. Whether a "
        "swap leaves the total unharmed, short by one, or short by two "
        "depends on where it falls relative to the gate, which is why this "
        "benchmark scores crossings one by one rather than totals.",
        "Every method publishes the signed frame offsets of its matched "
        "pairs so the match window itself can be audited. Mass sitting "
        "against the window's late edge means predictions run later than "
        "the labels' measured +0/-4 lead accounts for, and the response to "
        "that is to investigate the engine's anchor timing -- never to "
        "widen the window, which would also start matching predictions to "
        "the wrong vehicle.",
        "certain_only uses IGNORE-REGION semantics -- the standard "
        "treatment of an unadjudicated region (MOTChallenge distractor "
        "zones, COCO iscrowd) moved from space to the timeline. A probable "
        "label marks an interval the labeller could not adjudicate, so a "
        "prediction there is neither credited nor charged. It is scored by "
        "restricting the SAME joint match the full set reports, never by "
        "re-matching against the certain rows alone, so the two figures "
        "are directly comparable.",
        "certain_only precision is an UPPER bound within the certain "
        "subset: a genuine phantom landing inside a probable label's "
        "window is absorbed into the ignore set and leaves the denominator "
        "altogether. The ignore mass here is large -- see labels_ignored "
        "against the label total -- so predictions_moved_to_ignore is "
        "published with every figure.",
        "A prediction claimed by a probable label is removed from "
        "certain_only even when a certain label was also within window, "
        "because the joint match had already given it away. That can turn "
        "a certain label into a miss, so a recall gap between certain_only "
        "and full is not by itself evidence about the certain crossings.",
        "A gap between certain_only and full is dilution arithmetic before "
        "it is anything else: the same errors measured against a smaller "
        "label count move the rate further per error. It is not evidence "
        "that the certain crossings are harder.",
        "certain_only_naive is published only as the artefact that "
        "motivates the treatment above. It matches the certain rows alone, "
        "which turns every correct prediction of a probable crossing into "
        "a false alarm; its precision is not comparable with anything, and "
        "false_positives_matching_probable_labels is how much of it is "
        "manufactured that way.",
    ]
    if len(directions) < 2:
        only = next(iter(directions)) if directions else "one direction"
        caveats.append(
            f"Every one of the {len(truth.crossings)} labels is "
            f"{only!r}, so the per-direction breakdown has no "
            f"discriminating power on this clip. A prediction labelled "
            f"otherwise at this gate is simply a false positive; the table "
            f"is not evidence that direction is measured well."
        )
    return caveats


def _score_naive_subset(
    predictions: Sequence[CrossingEvent],
    labels: Sequence[Crossing],
    probable_labels: Sequence[Crossing],
    window: MatchWindow,
    gate_name: str,
) -> dict:
    """Score a confidence subset the NAIVE way -- match against the subset
    alone -- and count how many of its false alarms that manufactured.

    Published only as the artefact that motivates ignore-region
    semantics. Matching the certain rows alone turns every correct
    prediction of a ``probable`` crossing into a false alarm, so this
    precision figure is not comparable with anything; the count beside it
    is how much of it is manufactured.
    """
    result = match_crossings(predictions, labels, window, gate_name=gate_name)
    record = result.as_dict()
    leftovers = [predictions[index] for index in result.false_positives]
    explained = match_crossings(
        leftovers, probable_labels, window, gate_name=gate_name
    )
    record["false_positives_matching_probable_labels"] = explained.true_positives
    return record


def run_counting_benchmark(
    detections: Sequence[FrameDetections],
    gt: GroundTruth,
    methods: dict[str, Callable[[Sequence[FrameDetections]], list[CrossingEvent]]],
    window: MatchWindow = DEFAULT_MATCH_WINDOW,
    *,
    gate: Gate,
    detector: dict | None = None,
) -> dict:
    """Run every method over ONE shared detection stream and score each
    against ``gt``, on the full label set and on the ``certain`` rows.

    ``detections`` stands in for the brief's ``clip`` argument: every
    method must see the identical detections, or a measured difference
    could be detector variance rather than a difference in the counting
    rule or the tracker. Passing the decoded stream itself -- rather than a
    path each method would re-detect from -- makes that guarantee
    structural instead of a promise, and the report records the detector
    the stream came from.

    Both confidence subsets are published together because neither is
    honest alone: certain-only drops the hard cases and flatters recall,
    while the full set includes rows the labeller flagged as doubtful.
    They bracket the truth.
    """
    events: dict[str, list[CrossingEvent]] = {}
    timing: dict[str, dict] = {}
    frames = len(detections)

    for name, method in methods.items():
        # One method inside one bracket. Never a loop that runs them all
        # and indexes one out: that reports a sum in disguise.
        started = perf_counter()
        produced = method(detections)
        elapsed = perf_counter() - started
        events[name] = list(produced)
        timing[name] = {
            "seconds": elapsed,
            "frames": frames,
            "ms_per_frame": (elapsed * 1000.0 / frames) if frames else 0.0,
        }

    certain = [label for label in gt.crossings if label.confidence == "certain"]
    probable = [label for label in gt.crossings if label.confidence != "certain"]

    scored = {}
    for name, produced in events.items():
        joint = match_crossings(produced, gt.crossings, window, gate_name=gt.gate_name)
        full = joint.as_dict()
        # Greedy can under-report where two windows overlap; this is the
        # check that says whether it did, on this label set, today.
        full["max_cardinality_true_positives"] = max_cardinality_true_positives(
            produced, gt.crossings, window, gate_name=gt.gate_name
        )

        adjudicated, ignored = restrict_to_adjudicated(joint)
        certain_only = adjudicated.as_dict()
        certain_only["predictions_moved_to_ignore"] = ignored
        certain_only["labels_ignored"] = len(probable)

        scored[name] = {
            "full": full,
            "certain_only": certain_only,
            "certain_only_naive": _score_naive_subset(
                produced, certain, probable, window, gt.gate_name
            ),
        }

    return {
        "schema": REPORT_SCHEMA_VERSION,
        "protocol": gt.protocol,
        # The label file's NAME, never its path: this report is tracked,
        # and the same label set lives at a different absolute path on
        # every machine. It is the same rule GroundTruth._parse enforces
        # on the label file's own ``clip`` field, for the same reason.
        "ground_truth": gt.path.name,
        "match_window": window.as_dict(),
        "matching": {
            "rule": (
                "one-to-one, greedy nearest-frame-first, with a canonical "
                "tie-break on the label's own frame and id rather than on "
                "its position in the label file"
            ),
            "limitation": (
                "Greedy is not maximum-cardinality. Where two labels' "
                "windows overlap, taking the locally nearest pair first "
                "can consume a frame the other label needed, so greedy can "
                "score fewer true positives than the best possible "
                "assignment over the same eligible pairs. It errs only in "
                "the conservative direction -- it can under-report the "
                "engine, never flatter it -- and PROTOCOL.md fixed the rule "
                "before any scoring code existed, so it is kept and "
                "measured rather than amended after the fact."
            ),
            "greedy_equals_max_cardinality": all(
                record["full"]["true_positives"]
                == record["full"]["max_cardinality_true_positives"]
                for record in scored.values()
            ),
        },
        "clip": gt.clip,
        "fps": gt.fps,
        "frames": frames,
        "window": {"start_frame": gt.start_frame, "end_frame": gt.end_frame},
        "gate": {
            "name": gate.name,
            "start": [float(gate.start[0]), float(gate.start[1])],
            "end": [float(gate.end[0]), float(gate.end[1])],
            "label_positive": gate.label_positive,
            "label_negative": gate.label_negative,
        },
        "detector": detector,
        "labels": {
            "total": len(gt.crossings),
            "certain": len(certain),
            "probable": len(probable),
            "by_class": dict(
                sorted(Counter(label.class_name for label in gt.crossings).items())
            ),
            "by_direction": dict(
                sorted(Counter(label.direction for label in gt.crossings).items())
            ),
        },
        "caveats": _caveats(gt),
        "methods": scored,
        "timing": timing,
    }


def median_gate_approach_px_per_frame(
    detections: Sequence[FrameDetections],
    gate: Gate,
    *,
    within_px: float = 30.0,
    tracker_factory: Callable[[], object] = Tracker,
) -> float | None:
    """Median per-frame PERPENDICULAR anchor displacement of tracks while
    they are within ``within_px`` of the gate line, or ``None`` when no
    track ever comes that close.

    This is the number that makes a band result readable. A band of half
    width ``b`` fires when the anchor first comes within ``b`` of the line,
    which at ``v`` pixels per frame is roughly ``b / v`` frames before the
    line is actually reached -- so the band rule's crossing FRAME is wrong
    by about that much, whatever its crossing COUNT does. Without this
    figure, a reader cannot tell whether a band rule's mistiming is a
    property of the rule or of the footage.

    It uses ``baselines._signed_offset`` deliberately rather than
    re-deriving the geometry: it is explaining the band rule's behaviour,
    so it must measure with the band rule's own notion of distance to the
    gate, not with a second one that might disagree at the margin.
    """
    tracker = tracker_factory()
    previous: dict[int, float] = {}
    steps: list[float] = []
    for frame_index, _timestamp, frame_detections in detections:
        for track in tracker.update(frame_detections, frame_index):
            offset, position = _signed_offset(gate, track.anchor)
            last = previous.get(track.track_id)
            previous[track.track_id] = offset
            if last is None or not 0.0 <= position <= 1.0:
                continue
            if abs(offset) <= within_px or abs(last) <= within_px:
                steps.append(abs(offset - last))
    return float(np.median(steps)) if steps else None


def sweep_band_px(
    detections: Sequence[FrameDetections],
    labels: Sequence[Crossing],
    *,
    gate: Gate,
    band_values: Iterable[float],
    tracker_factory: Callable[[], object] = Tracker,
    window: MatchWindow = DEFAULT_MATCH_WINDOW,
) -> list[dict]:
    """The band rule's miss/phantom trade-off across a range of
    ``band_px``, as two separate series.

    A single "band versus gate" number is a choice of operating point
    presented as a result: a wider band cures misses and creates phantoms,
    and a narrower one trades back. Neither error can be tuned away, so the
    curve -- not one point on it -- is the finding.
    """
    entries: list[dict] = []
    for band_px in band_values:
        events = run_counting(
            detections, tracker_factory(), BandCounter(gate, band_px)
        )
        result = match_crossings(events, labels, window, gate_name=gate.name)
        entries.append(
            {
                "band_px": float(band_px),
                "n_predicted": result.n_predicted,
                "true_positives": result.true_positives,
                "misses": len(result.misses),
                "false_positives": len(result.false_positives),
                "miss_rate": result.miss_rate,
                "phantom_rate": result.phantom_rate,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
            }
        )
    return entries


# --- detection box noise ----------------------------------------------------


def _residual_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "std_px": 0.0, "mae_px": 0.0, "p95_abs_px": 0.0}
    array = np.asarray(values, dtype=np.float64)
    absolute = np.abs(array)
    return {
        "n": int(array.size),
        "std_px": float(np.std(array)),
        "mae_px": float(absolute.mean()),
        "p95_abs_px": float(np.percentile(absolute, 95)),
    }


def measure_detection_noise(
    detections: Sequence[FrameDetections],
    *,
    median_filter_frames: int = NOISE_MEDIAN_FILTER_FRAMES,
) -> dict:
    """Per-track residual of box width, height and centre around a
    median-filtered trajectory.

    This is a PROXY for detector noise and not ground truth, and the
    returned record says so in its own text. Three things it measures that
    are not the detector:

    - the median filter's own smoothing, which removes some of the real
      signal along with the noise and so understates fast genuine motion
      as well as overstating nothing;
    - the association, since a residual is only meaningful within one
      identity and a swapped identity contributes a step change as though
      the box had jumped;
    - genuine sub-filter motion, which is charged to the residual.

    The boxes measured are the RAW detection boxes: association runs
    through ``GreedyIoUTracker``, whose ``Track.box`` is always the last
    observed detection, never a predicted one. Grouping with the engine's
    own tracker would measure the Kalman filter's output instead, which is
    already smoothed, and would report a number far below the detector's
    real jitter.

    Only fully-centred windows contribute: the first and last
    ``median_filter_frames // 2`` samples of each track are dropped rather
    than filtered with a shrinking window, because a partial window is a
    different estimator and on a straight trajectory it produces a residual
    out of nothing at all.
    """
    if median_filter_frames < 3 or median_filter_frames % 2 == 0:
        raise ValueError(
            f"median_filter_frames must be an odd number of at least 3 so the "
            f"window is centred on a sample, got {median_filter_frames}"
        )

    tracker = GreedyIoUTracker()
    series: dict[int, list[tuple[float, float, float, float]]] = {}
    for frame_index, _timestamp, frame_detections in detections:
        for track in tracker.update(frame_detections, frame_index):
            x1, y1, x2, y2 = track.box
            series.setdefault(track.track_id, []).append(
                (x2 - x1, y2 - y1, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
            )

    quantities = ("box_width", "box_height", "centre_x", "centre_y")
    residuals: dict[str, list[float]] = {name: [] for name in quantities}
    widths: list[float] = []
    heights: list[float] = []
    half = median_filter_frames // 2
    contributing = 0

    for samples in series.values():
        if len(samples) < median_filter_frames:
            continue
        contributing += 1
        # Measured over the SAME tracks the residuals come from, so a
        # consumer normalising a residual by the box size is dividing two
        # figures drawn from one population rather than two.
        widths.extend(sample[0] for sample in samples)
        heights.extend(sample[1] for sample in samples)
        track_array = np.asarray(samples, dtype=np.float64)
        for axis, name in enumerate(quantities):
            column = track_array[:, axis]
            for index in range(half, column.size - half):
                smoothed = float(np.median(column[index - half : index + half + 1]))
                residuals[name].append(float(column[index]) - smoothed)

    return {
        "schema": REPORT_SCHEMA_VERSION,
        "caveat": (
            "A PROXY for detector box noise, not ground truth. The residual "
            "is measured against a median-filtered version of the track's "
            "own trajectory, so the filter's smoothing, any identity error "
            "in the association, and genuine sub-filter motion are all part "
            "of what is reported here. Treat it as the scale of the jitter, "
            "not as a measurement of the detector."
        ),
        "median_filter_frames": median_filter_frames,
        "association": (
            "GreedyIoUTracker, whose Track.box is the last OBSERVED "
            "detection box; the engine's tracker would report its own "
            "Kalman-smoothed boxes instead."
        ),
        "frames": len(detections),
        "tracks_seen": len(series),
        "tracks_contributing": contributing,
        "median_box_width_px": float(np.median(widths)) if widths else 0.0,
        "median_box_height_px": float(np.median(heights)) if heights else 0.0,
        "residuals": {
            name: _residual_stats(values) for name, values in residuals.items()
        },
    }


# --- the shared detection cache ---------------------------------------------

#: Schema of the git-ignored detection cache. Bumped whenever the record
#: layout below changes, so a stale file is refused rather than misread.
CACHE_SCHEMA_VERSION = 1


class DetectionCacheError(ValueError):
    """A detection cache could not be read, or does not describe the run
    that asked for it. Silently reusing a cache produced by a different
    detector would make every method's comparison meaningless in a way no
    downstream number would reveal."""


def write_detection_cache(
    path, detections: Sequence[FrameDetections], *, key: dict
) -> None:
    """Write the shared per-frame detections, stamped with the detector
    ``key`` that produced them."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA_VERSION,
                "key": key,
                "frames": [
                    {
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "detections": [
                            [
                                detection.x1,
                                detection.y1,
                                detection.x2,
                                detection.y2,
                                detection.score,
                                detection.class_id,
                                detection.class_name,
                            ]
                            for detection in frame_detections
                        ],
                    }
                    for frame_index, timestamp, frame_detections in detections
                ],
            }
        )
    )


def read_detection_cache(path, *, key: dict) -> list[FrameDetections]:
    """Read a cache written for exactly this detector ``key``.

    Raises ``DetectionCacheError`` when the file is absent, unreadable, of
    an unknown schema, or was produced by a different detector -- never
    repaired, never partially accepted.
    """
    cache_path = Path(path)
    if not cache_path.is_file():
        raise DetectionCacheError(f"no detection cache at {cache_path}")
    try:
        document = json.loads(cache_path.read_text())
    except json.JSONDecodeError as error:
        raise DetectionCacheError(f"{cache_path}: not valid JSON: {error}") from error
    if document.get("schema") != CACHE_SCHEMA_VERSION:
        raise DetectionCacheError(
            f"{cache_path}: cache schema {document.get('schema')!r}, this "
            f"build reads schema {CACHE_SCHEMA_VERSION}"
        )
    if document.get("key") != key:
        raise DetectionCacheError(
            f"{cache_path}: was produced by {document.get('key')!r} but this "
            f"run needs {key!r}; every method must see detections from the "
            f"detector the report names"
        )
    return [
        (
            frame["frame_index"],
            frame["timestamp"],
            [
                Detection(
                    x1=record[0],
                    y1=record[1],
                    x2=record[2],
                    y2=record[3],
                    score=record[4],
                    class_id=int(record[5]),
                    class_name=record[6],
                )
                for record in frame["detections"]
            ],
        )
        for frame in document["frames"]
    ]


def write_report(path, report: dict) -> Path:
    """Write a report as pretty JSON, refusing any non-finite number.

    A NaN or an infinity in a published accuracy figure is not a number a
    reader can act on, and JSON has no standard spelling for either --
    ``json`` would otherwise emit the bare tokens ``NaN`` and ``Infinity``,
    which are not valid JSON and which half the parsers in the world
    accept anyway. ``allow_nan=False`` is what does the refusing; it
    raises ``ValueError`` on the first such value, so nothing partial is
    ever written under a name that says it is a report.
    """
    report_path = Path(path)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)
    return report_path
