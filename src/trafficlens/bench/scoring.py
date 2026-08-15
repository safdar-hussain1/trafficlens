"""Scoring predicted gate crossings against hand-labelled ones.

Split out of ``harness`` at the seam the whole-branch review named: this
module decides what a MATCH is and what the scores mean, and knows
nothing about how a prediction was produced. It imports no tracker, no
detector, no counting rule, and it never will -- hand it any source of
``CrossingEvent`` and the labels of ``trafficlens.bench.slitscan`` and it
will score them. ``harness`` is the other half: it runs the engine and
the baselines and calls in here.

What is scored, and at what level
---------------------------------
Crossings, one by one, not totals. A total-count metric structurally
under-reports tracker damage: measured on the crossing-paths fixture with
a vertical gate, the engine reads a total of 2 in every gate position
while ``CentroidTracker`` reads 1, 0 or 1 depending only on where its
identity swap falls relative to the gate. Whether a swap costs nothing,
one count or two is a property of the geometry, never a property of the
swap -- so nothing in this module, or in any report built from it, may
claim that an identity swap leaves the total unchanged.

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
(411 and 416) at [407, 415] and [412, 420], overlapping across four
frames, where the asymmetric windows [410, 415] and [415, 420] share
exactly one.

That one shared frame is NOT harmless. One-to-one matching stops a shared
frame being counted twice; it does not stop it being assigned to the
wrong label. Two correct predictions at 415 and 420 -- both +4, a delta
the real histogram contains -- score one true positive, one false alarm
and one miss, because greedy nearest-first lets label 416 take 415
(distance 1) before label 411 can take it (distance 4). The asymmetry
shrinks the ambiguous region from four frames to one and one-to-one
matching bounds the damage to a single crossing; neither eliminates it.
``max_cardinality_true_positives`` is the check that measures what greedy
costs on a given label set instead of assuming it away.

Nothing here takes a scalar ``tolerance_frames``. A single symmetric
number would misdescribe the scorer, and a report field named that way
would misdescribe it to every later reader.

Class and direction
-------------------
Matching is CLASS-BLIND and DIRECTION-AWARE. Requiring class equality
would charge a car detected as a truck twice -- one miss and one false
alarm -- and would conflate a detector class error with a counting error,
so class agreement is reported separately, as consistency among matched
pairs and as per-class predicted-vs-ground-truth counts. Direction is
different: a wrong-direction prediction at a gate IS a counting error,
so it never matches.

Unadjudicated labels
--------------------
``restrict_to_adjudicated`` implements ``PROTOCOL.md``'s ignore-region
rule for ``probable`` rows. Its docstring states the two limitations that
must travel with any figure it produces.

numpy + stdlib + ``trafficlens.core`` / ``.bench.slitscan`` only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from trafficlens.bench.slitscan import Crossing
from trafficlens.core.gate import CrossingEvent

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


def _pair_rank(
    distance: int, label: Crossing, prediction: CrossingEvent
) -> tuple[int, int, int, int, int]:
    """The sort key that decides which eligible pair greedy matching takes
    first: nearest in frame, then a canonical tie-break.

    Every component is an INTRINSIC property of the two records -- never a
    list index. A label's ``frame`` and ``id`` are unique together (the
    loader rejects duplicate ids), and a prediction's ``frame_index`` and
    ``track_id`` are unique together for every counting rule in this
    project, so this key totally orders the candidate pairs without ever
    consulting the order the caller happened to supply them in.
    """
    return (
        distance,
        label.frame,
        label.id,
        prediction.frame_index,
        prediction.track_id,
    )


def max_cardinality_true_positives(
    predictions: Sequence[CrossingEvent],
    labels: Sequence[Crossing],
    window: MatchWindow = DEFAULT_MATCH_WINDOW,
    *,
    gate_name: str,
) -> int:
    """The largest number of pairs ANY one-to-one matching of the same
    eligibility graph could achieve.

    ``match_crossings`` is greedy, so it can score fewer true positives
    than this where two labels' windows overlap. Publishing both is what
    turns that limitation from a hidden assumption into a checked one: if
    the two agree, greedy cost the engine nothing on this label set; if
    they diverge, the greedy figure is the conservative one and the size of
    the gap is on the record.

    Kuhn's augmenting-path algorithm. Recursion depth is bounded by the
    number of labels (seventeen here), not by the number of predictions.
    """
    eligible: dict[int, list[int]] = {}
    for prediction_index, prediction in enumerate(predictions):
        if prediction.gate != gate_name:
            continue
        for label_index, label in enumerate(labels):
            if prediction.direction != label.direction:
                continue
            if window.contains(prediction.frame_index, label.frame):
                eligible.setdefault(prediction_index, []).append(label_index)

    claimed_by: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for label_index in eligible.get(prediction_index, ()):
            if label_index in visited:
                continue
            visited.add(label_index)
            holder = claimed_by.get(label_index)
            if holder is None or augment(holder, visited):
                claimed_by[label_index] = prediction_index
                return True
        return False

    return sum(1 for index in sorted(eligible) if augment(index, set()))


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

    Eligible pairs are then taken GREEDILY in order of ascending
    ``abs(frame_delta)``. Each prediction and each label is consumed at
    most once; whatever is left over is a false alarm or a miss.

    **Greedy nearest-first, NOT maximum-cardinality.** Where two labels'
    windows overlap, taking the locally nearest pair first can consume a
    frame the other label needed, so greedy can score fewer true positives
    than a maximum-cardinality assignment over the same eligibility graph
    would. That is a deliberate choice, twice over: ``PROTOCOL.md`` fixed
    this rule before any scoring code existed, and greedy errs only in the
    conservative direction -- it can under-report the engine, never flatter
    it. ``max_cardinality_true_positives`` measures the gap so it is
    published rather than assumed away.

    **The tie-break is canonical and independent of input order.** Equal
    distances resolve by the label's own frame, then its id, then the
    prediction's frame and track id -- all intrinsic properties of the
    records. Ordering by list position instead would make the result
    depend on the order the label file happened to be written in, which
    holds today only because ``GroundTruth`` validates frame order and
    would break silently the moment anything handed this function a
    differently-sorted list.

    ``gate_name`` is a required keyword because a label set carries no gate
    of its own: it is validated against one by ``GroundTruth.load``, and
    scoring a prediction from a different gate against it would be
    meaningless in a way no downstream number would reveal.
    """
    candidates = []
    for label_index, label in enumerate(labels):
        for prediction_index, prediction in enumerate(predictions):
            if prediction.gate != gate_name:
                continue
            if prediction.direction != label.direction:
                continue
            if not window.contains(prediction.frame_index, label.frame):
                continue
            delta = prediction.frame_index - label.frame
            candidates.append(
                (
                    _pair_rank(abs(delta), label, prediction),
                    prediction_index,
                    label_index,
                    delta,
                )
            )

    matches: list[tuple[int, int, int]] = []
    used_predictions: set[int] = set()
    used_labels: set[int] = set()
    for _rank, prediction_index, label_index, delta in sorted(candidates):
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


def restrict_to_adjudicated(
    joint: MatchResult, confidence: str = "certain"
) -> tuple[MatchResult, int]:
    """Apply IGNORE-REGION semantics to an existing match, returning the
    restricted result and the number of predictions moved to the ignore
    set.

    A ``probable`` label marks an interval the labeller could not
    adjudicate. The standard treatment of an unadjudicated region --
    MOTChallenge distractor zones, COCO's ``iscrowd`` -- is that a
    prediction landing in one is neither credited nor charged. This is
    that rule transposed from space to the timeline.

    The restriction is applied to ONE joint match over ALL labels, never
    by re-matching against the surviving subset. That matters: it makes
    the restricted figures a strict subsetting of the same assignment the
    full set reports, so the two are directly comparable, rather than two
    numbers produced by two different runs of the matcher.

    Two consequences a report must state:

    - A prediction claimed by a ``probable`` label is removed **even when
      an adjudicated label was also within window**, because the joint
      match had already given it away. That can turn an adjudicated label
      into a miss, and it is why a recall gap here is not automatically
      evidence about the adjudicated crossings.
    - The resulting precision is an UPPER bound within the adjudicated
      subset: a genuine phantom that happens to land inside a ``probable``
      label's window is absorbed into the ignore set and disappears from
      the denominator entirely. The count returned alongside is how big
      that absorbed mass is.
    """
    ignored_predictions = {
        prediction_index
        for prediction_index, label_index, _delta in joint.matches
        if joint.labels[label_index].confidence != confidence
    }
    kept_labels = [
        index
        for index, label in enumerate(joint.labels)
        if label.confidence == confidence
    ]
    kept_predictions = [
        index
        for index in range(len(joint.predictions))
        if index not in ignored_predictions
    ]
    label_position = {old: new for new, old in enumerate(kept_labels)}
    prediction_position = {old: new for new, old in enumerate(kept_predictions)}

    restricted = MatchResult(
        predictions=tuple(joint.predictions[index] for index in kept_predictions),
        labels=tuple(joint.labels[index] for index in kept_labels),
        matches=tuple(
            (prediction_position[prediction_index], label_position[label_index], delta)
            for prediction_index, label_index, delta in joint.matches
            if label_index in label_position
        ),
        # A prediction the joint match paired with nothing at all is a
        # genuine phantom: no label claimed it, so no ignore region
        # absorbs it and it stays chargeable.
        false_positives=tuple(
            prediction_position[index] for index in joint.false_positives
        ),
        misses=tuple(
            label_position[index] for index in joint.misses if index in label_position
        ),
    )
    return restricted, len(ignored_predictions)
