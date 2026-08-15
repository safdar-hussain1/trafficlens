"""Scoring the engine, and every standard failure mode, against ground
truth the pipeline did not produce.

This is the only measurement in the project where the yardstick shares
nothing with the thing measured: the labels come from
``trafficlens.bench.slitscan``'s raw-pixel review images, read by a human
under ``data/groundtruth/PROTOCOL.md``. Everything else here -- the
trackers, the counting rules, the gate geometry -- is under test.

What is scored, and at what level
---------------------------------
Crossings, one by one, not totals. A total-count metric structurally
under-reports tracker damage: measured on the crossing-paths fixture with
a vertical gate, the engine reads a total of 2 in every gate position
while ``CentroidTracker`` reads 1, 0 or 1 depending only on where its
identity swap falls relative to the gate. Whether a swap costs nothing,
one count or two is a property of the geometry, never a property of the
swap -- so nothing in this module, or in any report it writes, may claim
that an identity swap leaves the total unchanged.

Matching is therefore ONE-TO-ONE between predicted crossings and labels,
nearest-frame first, so a burst of predictions cannot all score against a
single label.

The match window is asymmetric: ``[label - 1, label + 4]``
-----------------------------------------------------------
``LABELLING_RECORD.md`` states the label frame is accurate to +0/-4
frames. The frame is machine-derived from the first blob row of a
stabilised slit-scan, and a vehicle's shadow reaches the gate band before
its tyres do, so labels are systematically EARLY. The engine fires on the
box's bottom-centre anchor -- the tyres -- and therefore lands 0 to 4
frames LATE relative to the label. A symmetric window scores a correct
3-frames-late prediction as a miss AND a false alarm, understating
accuracy twice over.

The asymmetry encodes a KNOWN bias in the LABELS. It is not a widening
chosen after seeing engine output, which ``PROTOCOL.md`` rightly warns
against: a symmetric +-4 window would put the closest real label pair
(411 and 416) at [407, 415] and [412, 420], genuinely overlapping, where
the asymmetric windows [410, 415] and [415, 420] touch at exactly one
frame and one-to-one nearest-first matching makes that touch harmless.

Nothing in this module takes a scalar ``tolerance_frames``. A single
symmetric number would misdescribe the scorer, and a report field named
that way would misdescribe it to every later reader.

Class and direction
-------------------
Matching is CLASS-BLIND and DIRECTION-AWARE. Requiring class equality
would charge a car detected as a truck twice -- one miss and one false
alarm -- and would conflate a detector class error with a counting error,
so class agreement is reported separately, as consistency among matched
pairs and as per-class predicted-vs-ground-truth counts. Direction is
different: a wrong-direction prediction at a gate IS a counting error,
so it never matches.

Timing
------
One method per measurement. A bracket that ran every method and then
indexed one out would report a sum in disguise; each method here is timed
alone, and the report's timing block is asserted by test to hold one
distinct entry per method. The measurement covers the tracker and the
counting rule ONLY -- detections are read from a cache, so the detector's
cost is excluded and is identical for every method by construction.

numpy + stdlib + ``trafficlens.core`` / ``.detect`` / ``.track`` /
``.bench`` / ``.pipeline`` only. No torch, no ultralytics: the scorer must
run wherever the labels do.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
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

#: Why the window is asymmetric, carried into every report that uses it so
#: the number can never be quoted without its justification.
MATCH_WINDOW_REASON = (
    "Labels are systematically early: the frame is the first slit-scan row "
    "of the vehicle's blob, and a vehicle's shadow reaches the gate band 1-4 "
    "frames before its tyres do, so LABELLING_RECORD.md states the label "
    "frame is accurate to +0/-4 frames. The engine fires on the box's "
    "bottom-centre anchor -- the tyres -- and so lands 0-4 frames late "
    "against the label. The asymmetry encodes that known label bias; it is "
    "not a widening chosen to absorb an engine offset."
)

#: The schema version of ``reports/counting_accuracy.json``.
REPORT_SCHEMA_VERSION = 1

#: Frames in the centred median filter ``measure_detection_noise`` takes a
#: trajectory's residual around. Must be odd. Five frames is a sixth of a
#: second at 30 fps: long enough that per-frame box jitter does not steer
#: the filter, short enough that genuine acceleration is not charged to
#: the residual as noise.
NOISE_MEDIAN_FILTER_FRAMES = 5


# --- the match window -------------------------------------------------------


@dataclass(frozen=True)
class MatchWindow:
    """The closed interval ``[label - frames_before, label + frames_after]``
    a prediction must land in to match a label.

    ``reason`` is not decoration. The interval is asymmetric, which is a
    claim about the LABELS, and a report that published the two numbers
    without the claim would look like a tolerance someone chose.
    """

    frames_before: int = 1
    frames_after: int = 4
    reason: str = MATCH_WINDOW_REASON

    def __post_init__(self) -> None:
        if self.frames_before < 0 or self.frames_after < 0:
            raise ValueError(
                f"a match window extends outward from the label in both "
                f"directions, so neither side may be negative; got "
                f"frames_before={self.frames_before}, "
                f"frames_after={self.frames_after}"
            )
        if not self.reason.strip():
            raise ValueError(
                "a match window must carry the reason for its width: a bare "
                "pair of numbers reads as a tolerance someone chose"
            )

    def contains(self, predicted_frame: int, label_frame: int) -> bool:
        """True when ``predicted_frame`` lies in this window around
        ``label_frame``. Both ends are CLOSED."""
        return (
            label_frame - self.frames_before
            <= predicted_frame
            <= label_frame + self.frames_after
        )

    def as_dict(self) -> dict:
        return {
            "frames_before": self.frames_before,
            "frames_after": self.frames_after,
            "reason": self.reason,
        }


DEFAULT_MATCH_WINDOW = MatchWindow()


# --- one-to-one crossing matching -------------------------------------------


def _rate(numerator: int, denominator: int) -> float:
    """A rate, or 0.0 when there is nothing to divide by.

    An empty prediction set has no precision and an empty label set has no
    recall; both report 0.0 rather than raising or returning NaN. The
    counts are always published alongside, so a 0.0 that means "no
    denominator" is never mistaken for a 0.0 that means "got everything
    wrong".
    """
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class MatchResult:
    """The one-to-one pairing of predicted crossings with labels, and the
    scores that follow from it.

    ``matches`` holds ``(prediction_index, label_index, frame_delta)``
    triples, where ``frame_delta`` is ``predicted_frame - label_frame`` --
    signed, so a systematic offset is visible rather than absorbed.
    ``false_positives`` and ``misses`` hold the indices of the predictions
    and labels nothing paired with.
    """

    predictions: tuple[CrossingEvent, ...]
    labels: tuple[Crossing, ...]
    matches: tuple[tuple[int, int, int], ...]
    false_positives: tuple[int, ...]
    misses: tuple[int, ...]

    @property
    def n_predicted(self) -> int:
        return len(self.predictions)

    @property
    def n_ground_truth(self) -> int:
        return len(self.labels)

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def precision(self) -> float:
        return _rate(self.true_positives, self.n_predicted)

    @property
    def recall(self) -> float:
        return _rate(self.true_positives, self.n_ground_truth)

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2.0 * self.precision * self.recall / total if total else 0.0

    @property
    def signed_bias(self) -> int:
        """Predicted minus labelled crossings: positive is an over-count."""
        return self.n_predicted - self.n_ground_truth

    @property
    def count_error(self) -> int:
        """The magnitude of ``signed_bias``. Reported alongside it, never
        instead of it: a rule that misses one vehicle and phantoms another
        has a count error of 0 and is still wrong twice."""
        return abs(self.signed_bias)

    @property
    def miss_rate(self) -> float:
        """Labels nothing predicted, over all labels."""
        return _rate(len(self.misses), self.n_ground_truth)

    @property
    def phantom_rate(self) -> float:
        """False alarms per labelled crossing -- NOT ``1 - precision``.

        Normalising by the label count keeps the magnitude visible: a rule
        emitting fifty times more crossings than exist would saturate any
        rate normalised by its own output and read as merely "bad", where
        this reads 50.0.
        """
        return _rate(len(self.false_positives), self.n_ground_truth)

    def _class_consistency(self) -> dict:
        """Class agreement among MATCHED pairs only.

        Matching is class-blind, so this is where a detector class error
        surfaces -- once, as a confusion, rather than twice as a miss plus
        a false alarm.
        """
        confusions: Counter[str] = Counter()
        same = 0
        for prediction_index, label_index, _delta in self.matches:
            predicted = self.predictions[prediction_index].class_name
            labelled = self.labels[label_index].class_name
            if predicted == labelled:
                same += 1
            else:
                confusions[f"{labelled} -> {predicted}"] += 1
        return {
            "matched": self.true_positives,
            "same_class": same,
            "rate": _rate(same, self.true_positives),
            "confusions": dict(sorted(confusions.items())),
        }

    def _per_class(self) -> dict:
        """Predicted against labelled counts, per class, unmatched.

        A separate view from the matching, deliberately: it answers "did
        the engine see four trucks?" without any pairing decision in the
        way.
        """
        predicted = Counter(event.class_name for event in self.predictions)
        labelled = Counter(label.class_name for label in self.labels)
        return {
            class_name: {
                "predicted": predicted.get(class_name, 0),
                "ground_truth": labelled.get(class_name, 0),
            }
            for class_name in sorted(set(predicted) | set(labelled))
        }

    def _per_direction(self) -> dict:
        """Predicted, labelled and matched counts per direction label.

        On a clip whose labels are all one direction this table has no
        discriminating power at all, and the report says so rather than
        letting two rows imply otherwise.
        """
        predicted = Counter(event.direction for event in self.predictions)
        labelled = Counter(label.direction for label in self.labels)
        matched = Counter(
            self.labels[label_index].direction
            for _prediction_index, label_index, _delta in self.matches
        )
        return {
            direction: {
                "predicted": predicted.get(direction, 0),
                "ground_truth": labelled.get(direction, 0),
                "true_positives": matched.get(direction, 0),
            }
            for direction in sorted(set(predicted) | set(labelled))
        }

    def _frame_deltas(self) -> dict:
        """Signed ``predicted - label`` frame offsets of matched pairs.

        Published because the match window is asymmetric: the asymmetry is
        a claim that predictions run late, and this is the evidence for or
        against it. A scorer that hid these could widen its window forever
        unobserved.
        """
        deltas = [delta for _prediction, _label, delta in self.matches]
        if not deltas:
            return {"mean": None, "min": None, "max": None, "histogram": {}}
        histogram = Counter(deltas)
        return {
            "mean": sum(deltas) / len(deltas),
            "min": min(deltas),
            "max": max(deltas),
            "histogram": {
                str(delta): histogram[delta] for delta in sorted(histogram)
            },
        }

    def as_dict(self) -> dict:
        """The JSON-shaped record of this match.

        ``false_positives`` and ``misses`` are published as FRAME NUMBERS,
        not as indices into arrays the report does not contain: a reader
        with the slit-scan open wants to know which frames went wrong.
        """
        return {
            "n_predicted": self.n_predicted,
            "n_ground_truth": self.n_ground_truth,
            "true_positives": self.true_positives,
            "false_positives": [
                self.predictions[index].frame_index
                for index in self.false_positives
            ],
            "misses": [self.labels[index].frame for index in self.misses],
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "count_error": self.count_error,
            "signed_bias": self.signed_bias,
            "miss_rate": self.miss_rate,
            "phantom_rate": self.phantom_rate,
            "matched_frame_delta": self._frame_deltas(),
            "class_consistency": self._class_consistency(),
            "per_class": self._per_class(),
            "per_direction": self._per_direction(),
        }


def match_crossings(
    predictions: Sequence[CrossingEvent],
    labels: Sequence[Crossing],
    window: MatchWindow = DEFAULT_MATCH_WINDOW,
    *,
    gate_name: str,
) -> MatchResult:
    """Pair predicted crossings with labels, one to one, nearest first.

    A pair is ELIGIBLE when the prediction names ``gate_name``, carries the
    label's direction, and its frame lies inside ``window`` around the
    label's frame. Class is deliberately NOT part of eligibility -- see the
    module docstring.

    Eligible pairs are then taken greedily in order of ascending
    ``abs(frame_delta)``, ties broken by ascending label index then
    ascending prediction index, so the same inputs always produce the same
    pairing. Each prediction and each label is consumed at most once;
    whatever is left over is a false alarm or a miss.

    ``gate_name`` is a required keyword because a label set carries no gate
    of its own: it is validated against one by ``GroundTruth.load``, and
    scoring a prediction from a different gate against it would be
    meaningless in a way no downstream number would reveal.
    """
    candidates: list[tuple[int, int, int, int]] = []
    for label_index, label in enumerate(labels):
        for prediction_index, prediction in enumerate(predictions):
            if prediction.gate != gate_name:
                continue
            if prediction.direction != label.direction:
                continue
            if not window.contains(prediction.frame_index, label.frame):
                continue
            delta = prediction.frame_index - label.frame
            candidates.append((abs(delta), label_index, prediction_index, delta))

    matches: list[tuple[int, int, int]] = []
    used_predictions: set[int] = set()
    used_labels: set[int] = set()
    for _distance, label_index, prediction_index, delta in sorted(candidates):
        if label_index in used_labels or prediction_index in used_predictions:
            continue
        used_labels.add(label_index)
        used_predictions.add(prediction_index)
        matches.append((prediction_index, label_index, delta))

    matches.sort()
    return MatchResult(
        predictions=tuple(predictions),
        labels=tuple(labels),
        matches=tuple(matches),
        false_positives=tuple(
            index for index in range(len(predictions)) if index not in used_predictions
        ),
        misses=tuple(
            index for index in range(len(labels)) if index not in used_labels
        ),
    )


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
        "certain-only precision is NOT comparable with the full-set "
        "figure: dropping the probable labels turns their correct "
        "predictions into false alarms. Each certain-only result records "
        "how many of its false alarms were manufactured that way.",
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


def _score_subset(
    predictions: Sequence[CrossingEvent],
    labels: Sequence[Crossing],
    probable_labels: Sequence[Crossing],
    window: MatchWindow,
    gate_name: str,
) -> dict:
    """Score one confidence subset, and say how many of its false alarms
    are artefacts of the labels the subset dropped.

    The second half matters for the certain-only subset: a prediction that
    correctly found a ``probable`` crossing becomes a false alarm the
    moment that label is excluded. Counting those is descriptive, not a
    second scoring rule -- the precision figure is left exactly as the
    subset produces it.
    """
    result = match_crossings(predictions, labels, window, gate_name=gate_name)
    record = result.as_dict()
    if probable_labels:
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

    scored = {
        name: {
            "full": _score_subset(
                produced, gt.crossings, (), window, gt.gate_name
            ),
            "certain_only": _score_subset(
                produced, certain, probable, window, gt.gate_name
            ),
        }
        for name, produced in events.items()
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
