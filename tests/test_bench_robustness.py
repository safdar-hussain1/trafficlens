"""Tests for the robustness benchmark family (``trafficlens.bench.degrade``
and ``scripts/bench_robustness.py``).

Four independent degradation protocols are measured here, and the way a
robustness benchmark quietly stops measuring anything is well understood,
so each failure mode has a test that can see it:

- **A protocol that does not reduce to the undegraded baseline.** If the
  identity level (30 fps / 0 % dropped / p = 0 / sigma = 0) does not
  reproduce the Task 14 numbers EXACTLY, the degradation is not the only
  thing that changed and every other row is uninterpretable. The identity
  level runs the general code path, never a short-circuit branch, so the
  reduction proves the transform rather than proving a branch.
- **A match window that is not widened with the sampling interval.** A
  method that only sees every Delta-th frame physically cannot report a
  crossing before the next sampled frame; scoring it against the
  undegraded window charges it for the sampling grid. The test below
  measures how much that manufactures.
- **A widening that is not derived from the realised sample pattern.** The
  late side must be wide enough that the first retained frame at or after
  every label is still inside it. That is the semantic property, and it
  is asserted against real drop patterns rather than assumed.
- **Randomness that leaks between levels or between methods.** Every
  method at one level must see the identical degraded stream, or a
  measured difference could be noise variance; and each level must draw
  from its own stream, or adding a sweep point silently moves every other
  point's numbers.
- **A published headline nobody checked against the series under it.** The
  report answers two questions with booleans; the test recomputes both
  from the published per-level numbers.

Tests that read the COMMITTED reports rather than a synthetic fixture are
deliberate, exactly as in ``test_bench_counting``: a schema asserted only
against a dict built in the test proves the test's dict is well-formed,
not that the published numbers are.
"""

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from trafficlens.bench.degrade import (
    FALSE_POSITIVE_FRAME_LIMIT,
    PROTOCOL_BOX_JITTER,
    PROTOCOL_DETECTION_DROPOUT,
    PROTOCOL_DROPPED_FRAMES,
    PROTOCOL_FRAME_RATE,
    ROBUSTNESS_SEED,
    corner_sigma_equivalents,
    decimate,
    drop_detections,
    drop_frames,
    dropout_streams,
    dropped_frame_streams,
    frame_rate_streams,
    jitter_boxes,
    jitter_streams,
    label_reachability,
    map_events_to_source,
    run_protocol,
    run_stream,
    widen_for_gap,
    window_resolution,
)
from trafficlens.bench.harness import build_methods
from trafficlens.bench.scoring import DEFAULT_MATCH_WINDOW, match_crossings
from trafficlens.bench.slitscan import Crossing, GroundTruth
from trafficlens.core.gate import Gate
from trafficlens.detect.base import Detection

ROOT = Path(__file__).resolve().parents[1]
COUNTING_REPORT = ROOT / "reports" / "counting_accuracy.json"
NOISE_REPORT = ROOT / "reports" / "detection_noise.json"
ROBUSTNESS_REPORT = ROOT / "reports" / "robustness.json"
FIGURE_DIR = ROOT / "reports" / "figures"

GATE_NAME = "inbound"
GATE_Y = 300.0
LAST_FRAME = 200


# -- fixtures ----------------------------------------------------------------


def _gate(name: str = GATE_NAME) -> Gate:
    """The same left-to-right gate at a constant image y the counting
    tests use: +1 (up the frame) is ``away``, -1 (down the frame, toward
    the camera) is ``toward``."""
    return Gate(
        name,
        (0.0, GATE_Y),
        (400.0, GATE_Y),
        label_positive="away",
        label_negative="toward",
        expected_direction="toward",
    )


def _det(
    x: float,
    y: float,
    score: float = 0.9,
    class_name: str = "car",
    width: float = 80.0,
    height: float = 80.0,
) -> Detection:
    """A detection whose bottom-centre anchor is exactly ``(x, y)``."""
    class_id = {"car": 2, "truck": 7}[class_name]
    return Detection(
        x1=x - width / 2.0,
        y1=y - height,
        x2=x + width / 2.0,
        y2=y,
        score=score,
        class_id=class_id,
        class_name=class_name,
    )


def _descending_vehicle(
    crossing_frame: int,
    lane_x: float,
    last_frame: int,
    *,
    speed_px: float = 1.0,
    class_name: str = "car",
) -> dict[int, Detection]:
    """One vehicle descending the frame at a constant speed, positioned so
    its anchor is strictly above the gate on ``crossing_frame - 1`` and
    strictly below it on ``crossing_frame``.

    Half a step of offset keeps the anchor off the gate line itself, so
    the engine's swept-segment test and both band rules all agree on which
    frame the crossing belongs to when nothing is degraded.

    One pixel per frame on an 80 px box is deliberately slow: at a stride
    of six the box still overlaps its own predicted position by more than
    the engine's 0.8 IoU floor, so these fixtures exercise the SCORING of
    a decimated run rather than the engine's association cliff. What that
    cliff does on the real clip is a measurement, and it belongs in the
    report, not in a fixture that would then be testing two things at once.
    """
    return {
        frame: _det(
            lane_x,
            GATE_Y + (frame - crossing_frame) * speed_px + speed_px / 2.0,
            class_name=class_name,
        )
        for frame in range(last_frame + 1)
    }


def _traffic(
    crossings: list[tuple[int, float]],
    last_frame: int = LAST_FRAME,
    *,
    speed_px: float = 1.0,
) -> list[tuple[int, float, list[Detection]]]:
    """A ``(frame_index, timestamp, detections)`` stream carrying one
    descending vehicle per ``(crossing_frame, lane_x)`` pair."""
    per_frame: dict[int, list[Detection]] = {
        frame: [] for frame in range(last_frame + 1)
    }
    for crossing_frame, lane_x in crossings:
        vehicle = _descending_vehicle(
            crossing_frame, lane_x, last_frame, speed_px=speed_px
        )
        for frame, detection in vehicle.items():
            per_frame[frame].append(detection)
    return [
        (frame, frame / 30.0, per_frame[frame]) for frame in range(last_frame + 1)
    ]


#: Three lanes far enough apart that an 80 px box never overlaps its
#: neighbour, and three crossing frames chosen against a stride-6 sampling
#: grid (0, 6, 12, ...): 31 lands 5 frames after its last sample, 54 lands
#: on one, 104 lands 4 after. The first is exactly the case the undegraded
#: +4 window cannot score.
SYNTHETIC_CROSSINGS = [(31, 60.0), (54, 160.0), (104, 260.0)]


def _synthetic_stream() -> list[tuple[int, float, list[Detection]]]:
    return _traffic(SYNTHETIC_CROSSINGS)


def _synthetic_labels() -> list[Crossing]:
    return [
        Crossing(1, 31, "car", "toward", "certain"),
        Crossing(2, 54, "car", "toward", "probable"),
        Crossing(3, 104, "car", "toward", "certain"),
    ]


def _load_script(name: str):
    """Import a file under ``scripts/`` by path: the directory is not a
    package, so the report-assembling code is otherwise untestable."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report() -> dict:
    if not ROBUSTNESS_REPORT.is_file():
        pytest.fail(
            f"{ROBUSTNESS_REPORT.relative_to(ROOT)} is missing: the tracked "
            f"report is the artefact these tests check, so it must be "
            f"regenerated by scripts/bench_robustness.py whenever the "
            f"harness changes"
        )
    return json.loads(ROBUSTNESS_REPORT.read_text())


def _entry(report: dict, protocol: str, level: float) -> dict:
    for entry in report["protocols"][protocol]["entries"]:
        if entry["level"] == level:
            return entry
    raise AssertionError(
        f"protocol {protocol!r} has no level {level!r}; it has "
        f"{[e['level'] for e in report['protocols'][protocol]['entries']]}"
    )


# -- the degradation module must not be able to see what it degrades ---------


def test_the_degradation_module_imports_no_tracker_and_no_counting_rule():
    """``degrade`` decides how the INPUT is spoiled; it must not know what
    consumes it.

    The same seam ``scoring`` is held to, for the same reason: a
    degradation protocol with a handle on the engine could be shaped
    against the engine's behaviour, and nothing downstream would show it.
    Methods arrive as opaque callables and labels as data, so the module's
    own import graph is asserted rather than trusted.

    Asserted as a TOTAL ALLOWLIST over every module the file reaches, not
    as a blacklist and not only over ``trafficlens.*``. A blacklist of
    dotted module paths is evaded by three forms a working programmer
    reaches for without thinking -- ``from trafficlens.track import
    tracker`` and ``from trafficlens.bench import harness`` name the
    PACKAGE and bind the module as an attribute, and
    ``from trafficlens.core.gate import GateCounter`` pulls the engine's
    own counting rule out of a module the seam does legitimately need for
    its ``CrossingEvent`` type.

    An allowlist keyed on ``node.module`` alone is evaded in turn by the
    two forms that name no absolute module at all:

    - **relative imports**, the most idiomatic way to reach a sibling.
      ``from . import harness`` and ``from ..track import tracker`` both
      leave ``node.module`` empty or partial while resolving straight
      into the package that imports ``Tracker``. They are rejected
      outright rather than resolved: this module has no legitimate use
      for one, and rejecting is not a rule anyone can get subtly wrong.
    - **dynamic imports**. ``importlib.import_module("trafficlens.track.
      tracker")`` is invisible to any check that reads the import graph,
      so the machinery itself is refused -- which the total allowlist
      does for free, because ``importlib`` is not on it.

    The allowlist and the module docstring's stated list are asserted
    against each other in BOTH directions, so neither an undocumented
    import nor a test-only widening can pass on its own.
    """
    import ast
    import re

    source = (ROOT / "src/trafficlens/bench/degrade.py").read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    bound: set[str] = set()
    star_from: set[str] = set()
    relative: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                target = star_from if alias.name == "*" else bound
                target.add(alias.name)
            if node.level:
                relative.append("." * node.level + (node.module or ""))
            elif node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    # A relative import names no absolute module, so it is invisible to
    # every check below. Rejected outright.
    assert relative == [], (
        f"degrade.py must not use relative imports -- a sibling-relative "
        f"form reaches trafficlens.bench.harness, and the tracker with it, "
        f"while naming no module this pin can see: {sorted(relative)}"
    )

    #: The four internal modules the seam may see, and nothing else.
    allowed_internal = {
        "trafficlens.bench.scoring",
        "trafficlens.bench.slitscan",
        "trafficlens.core.gate",
        "trafficlens.detect.base",
    }
    # The TOTAL import surface, stdlib included. Keeping the stdlib on the
    # allowlist is what refuses importlib -- and with it every dynamic
    # escape hatch -- without needing to enumerate escape hatches.
    allowed = allowed_internal | {
        "__future__",
        "dataclasses",
        "fractions",
        "hashlib",
        "math",
        "numpy",
        "typing",
    }
    assert imported <= allowed, sorted(imported - allowed)

    # An allowed MODULE is not a licence to import anything inside it:
    # trafficlens.core.gate carries both the CrossingEvent type this module
    # needs and the GateCounter rule it must never see.
    never_bound = {
        "Tracker",
        "GateCounter",
        "BandCounter",
        "PerFrameCounter",
        "CentroidTracker",
        "GreedyIoUTracker",
    }
    assert bound & never_bound == set(), sorted(bound & never_bound)
    # A star import would smuggle every one of those in unnamed.
    assert star_from == set(), sorted(star_from)
    # ... and __import__ would bypass the import graph altogether.
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "__import__"
    ], "degrade.py must not call __import__"

    assert "trafficlens.bench.scoring" in imported  # the scorer
    assert "trafficlens.detect.base" in imported  # the Detection type

    # The docstring's own statement of the allowlist, matched EXACTLY and
    # in both directions. A substring check on the last dotted component
    # would not do: "tracker" already appears in the docstring's prose, so
    # adding trafficlens.track.tracker to the set above would pass one.
    docstring = ast.get_docstring(tree) or ""
    stated = set(re.findall(r"trafficlens(?:\.[A-Za-z_][A-Za-z0-9_]*)+", docstring))
    assert stated == allowed_internal, {
        "documented but not allowed": sorted(stated - allowed_internal),
        "allowed but not documented": sorted(allowed_internal - stated),
    }


# -- the match window must widen with the sampling interval ------------------


def test_the_late_side_widens_by_the_gap_minus_one_and_the_early_side_never_moves():
    """A method sampling every Delta-th frame reports a crossing in
    ``[f, f + Delta - 1]``, so only the LATE side may move, and only by
    the quantisation. Widening the early side would start matching
    predictions to a vehicle that had not yet arrived."""
    for gap, expected_after in ((1, 4), (2, 5), (3, 6), (6, 9), (15, 18)):
        widened = widen_for_gap(DEFAULT_MATCH_WINDOW, gap)
        assert widened.frames_before == DEFAULT_MATCH_WINDOW.frames_before
        assert widened.frames_after == expected_after, gap


def test_a_gap_of_one_leaves_the_undegraded_window_exactly_unchanged():
    """The reduction proof for the window itself: at 30 fps, at 0 %
    dropped frames, and under both protocols that resample nothing, the
    scorer must be the one Task 14 fixed, to the frame."""
    widened = widen_for_gap(DEFAULT_MATCH_WINDOW, 1)
    assert widened.frames_before == 1
    assert widened.frames_after == 4
    published = json.loads(COUNTING_REPORT.read_text())["match_window"]
    assert widened.frames_before == published["frames_before"]
    assert widened.frames_after == published["frames_after"]


def test_the_widened_window_carries_the_gap_it_was_derived_from():
    """A pair of numbers with no reason reads as a tolerance someone
    chose. The widening is derived a priori from the sampling grid, and
    the window says so in its own text."""
    widened = widen_for_gap(DEFAULT_MATCH_WINDOW, 15)
    assert "15" in widened.reason
    assert DEFAULT_MATCH_WINDOW.reason in widened.reason


def test_the_widening_covers_the_first_retained_frame_after_every_label():
    """The semantic property the widening claims, checked against the
    realised drop patterns rather than assumed.

    A crossing labelled at frame ``f`` cannot be reported before the first
    RETAINED frame at or after ``f``. That wait must be covered by the
    WIDENING alone, not by the undegraded window's own +4: the +4 encodes
    the label's known early bias, and a prediction that is late by the
    sampling grid AND late by the anchor lag needs both allowances. A test
    that only checked the wait against the total late side would let the
    widening be a frame short and never see it.

    The tightness assertion at the end is what stops this being vacuous:
    some (level, label) pair must realise the worst case exactly, or the
    bound is never actually pressed and any narrowing would slip through.

    The property holds only for labels the retained pattern can still
    reach. A label past the LAST retained frame has no later sample at
    all, so there is nothing for the widening to cover and every method is
    charged a miss for it by construction -- which is what happens on the
    real clip, where 2 fps retains nothing after frame 720 and the last
    label sits at 726.

    That case is exercised rather than merely not tripped over, and the
    fixture is built to reach it: a fourth label sits at frame 199 of a
    stream ending at 200, which 10, 5 and 2 fps all retain nothing after.
    ``_synthetic_labels`` alone would not do -- its latest label is 104,
    every swept level keeps a frame after it, and the ``if not later``
    branch below would be dead code under a docstring claiming otherwise.
    The ``reached`` assertion at the end is what keeps that honest.
    """
    stream = _synthetic_stream()
    # The three shipped labels plus one in the stream's tail, past the last
    # frame the lower rates retain.
    labels = _synthetic_labels() + [Crossing(4, 199, "car", "toward", "certain")]
    label_frames = [label.frame for label in labels]
    streams = list(
        frame_rate_streams(stream, source_fps=30.0, target_rates=(30, 25, 15, 10, 5, 2))
    ) + list(
        dropped_frame_streams(stream, fractions=(0.0, 0.05, 0.10, 0.20, 0.30))
    )
    pressed = False
    reached = 0
    for degraded in streams:
        window = widen_for_gap(DEFAULT_MATCH_WINDOW, degraded.max_gap)
        widening = window.frames_after - DEFAULT_MATCH_WINDOW.frames_after
        reach = label_reachability(labels, degraded.source_frames)
        unreachable = set(reach["labels_after_the_last_retained_frame"])
        reached += bool(unreachable)
        for label_frame in label_frames:
            later = [s for s in degraded.source_frames if s >= label_frame]
            # The two halves must agree on which labels have no later
            # sample, or the ceiling the report publishes is not the one
            # this property is checked against.
            assert bool(later) == (label_frame not in unreachable), (
                degraded.level_label,
                label_frame,
            )
            if not later:
                continue
            wait = later[0] - label_frame
            assert wait <= widening, (
                degraded.protocol,
                degraded.level_label,
                label_frame,
                wait,
                widening,
            )
            pressed = pressed or wait == widening
    assert pressed, (
        "no label in any swept level waits the full quantisation, so this "
        "fixture never presses the bound and could not see it narrowed"
    )
    assert reached == 3, (
        f"exactly the three lowest rates must leave the tail label "
        f"unreachable, or the branch this test claims to exercise is dead "
        f"code; {reached} level(s) reached it"
    )


def test_a_label_past_the_last_retained_frame_caps_recall_below_one():
    """The tail case the widening cannot cover, on the real clip's own
    numbers.

    2 fps keeps nothing after frame 720 of 735 while the last label sits
    at 726, so that crossing has no sample at or after it: every method is
    charged a miss for it before any engine runs, and recall at that one
    level is capped at 16/17. The report must say so, because a ceiling it
    did not state would read as the engine missing a crossing it was never
    shown.
    """
    labels = _synthetic_labels() + [Crossing(4, 199, "car", "toward", "certain")]
    stream = _synthetic_stream()

    # A label inside the retained pattern is reachable at every rate.
    fast = decimate(stream, source_fps=30.0, target_fps=30.0)
    intact = label_reachability(labels, fast.source_frames)
    assert intact["labels_after_the_last_retained_frame"] == []
    assert intact["recall_ceiling"] == 1.0

    # ... and one past the last retained frame is not, at any widening.
    slow = decimate(stream, source_fps=30.0, target_fps=2.0)
    assert slow.source_frames[-1] < 199
    capped = label_reachability(labels, slow.source_frames)
    assert capped["labels_after_the_last_retained_frame"] == [199]
    assert capped["last_retained_source_frame"] == slow.source_frames[-1]
    assert capped["recall_ceiling"] == 3 / 4
    # The widening covers the wait for the NEXT sample; there is none.
    widened = widen_for_gap(DEFAULT_MATCH_WINDOW, slow.max_gap)
    assert 199 + widened.frames_after > slow.source_frames[-1]

    # No method can beat the ceiling, which is what makes it a ceiling.
    entry = run_stream(slow, build_methods(_gate()), labels, gate_name=GATE_NAME)
    assert entry["resolution"]["reachability"] == capped
    for name, record in entry["methods"].items():
        assert record["recall"] <= capped["recall_ceiling"] + 1e-12, name


def test_the_published_report_states_the_recall_ceiling_at_every_level():
    """The same property on the committed artefact, where the real clip
    puts label 726 past 2 fps's last retained frame at 720."""
    report = _report()
    labels = report["labels"]["total"]
    capped = []
    for protocol, block in report["protocols"].items():
        for entry in block["entries"]:
            reach = entry["resolution"]["reachability"]
            missing = len(reach["labels_after_the_last_retained_frame"])
            assert reach["recall_ceiling"] == (labels - missing) / labels, (
                protocol,
                entry["level"],
            )
            for frame in reach["labels_after_the_last_retained_frame"]:
                assert frame > reach["last_retained_source_frame"]
            if missing:
                capped.append(f"{protocol}@{entry['level_label']}")
                # No method may exceed a ceiling that is a ceiling.
                for name, record in entry["methods"].items():
                    assert record["recall"] <= reach["recall_ceiling"], (
                        protocol,
                        entry["level"],
                        name,
                    )

    # The one level where it bites, named so it cannot quietly spread.
    assert capped == ["frame_rate@2 fps"]
    slowest = _entry(report, PROTOCOL_FRAME_RATE, 2.0)["resolution"]["reachability"]
    assert slowest["last_retained_source_frame"] == 720
    assert slowest["labels_after_the_last_retained_frame"] == [726]
    assert slowest["recall_ceiling"] == 16 / 17


def test_scoring_a_decimated_run_against_the_undegraded_window_manufactures_errors():
    """Why the widening is load-bearing rather than decorative.

    The same decimated run is scored twice, against the undegraded window
    and against the widened one. With the undegraded window a crossing
    whose next sample falls 5 frames later is charged as a miss AND a
    false alarm -- two errors that measure the sampling grid, not the
    engine. If this test ever stops seeing a difference, the widening has
    stopped doing anything and every rate below 30 fps is meaningless.
    """
    gate = _gate()
    labels = _synthetic_labels()
    method = build_methods(gate)["engine+gate"]

    degraded = decimate(_synthetic_stream(), source_fps=30.0, target_fps=5.0)
    assert degraded.max_gap == 6
    events = map_events_to_source(method(list(degraded.frames)), degraded.source_frames)

    narrow = match_crossings(events, labels, DEFAULT_MATCH_WINDOW, gate_name=GATE_NAME)
    wide = match_crossings(
        events,
        labels,
        widen_for_gap(DEFAULT_MATCH_WINDOW, degraded.max_gap),
        gate_name=GATE_NAME,
    )
    assert wide.true_positives > narrow.true_positives
    assert len(narrow.misses) > 0
    assert len(narrow.false_positives) > len(wide.false_positives)


def test_run_stream_scores_a_decimated_level_with_the_widened_window():
    """The widening has to be applied by the thing that actually scores,
    not merely be available to it.

    ``run_stream`` is the single seam every published figure passes
    through, so this asserts the window it used, the widening it recorded,
    and -- the part a schema check would miss -- that the score is the
    widened one rather than the undegraded window's.
    """
    stream = frame_rate_streams(
        _synthetic_stream(), source_fps=30.0, target_rates=(5.0,)
    )[0]
    labels = _synthetic_labels()
    methods = build_methods(_gate())
    entry = run_stream(stream, methods, labels, gate_name=GATE_NAME)

    assert entry["max_gap_frames"] == 6
    assert entry["match_window"]["frames_before"] == 1
    assert entry["match_window"]["frames_after"] == 4 + 6 - 1
    assert entry["window_widened_by_frames"] == 5

    narrow = match_crossings(
        map_events_to_source(
            methods["engine+gate"](stream.frames), stream.source_frames
        ),
        labels,
        DEFAULT_MATCH_WINDOW,
        gate_name=GATE_NAME,
    )
    assert entry["methods"]["engine+gate"]["true_positives"] > narrow.true_positives


def test_events_are_mapped_back_to_the_original_frame_numbering():
    """Ground truth is indexed in the original 30 fps stream, so a
    decimated run's frame numbers mean nothing until they are mapped
    back. Without this the labels and the predictions are on two different
    clocks and every score is noise."""
    degraded = decimate(_synthetic_stream(), source_fps=30.0, target_fps=5.0)
    method = build_methods(_gate())["engine+gate"]
    raw = method(list(degraded.frames))
    assert raw, "the synthetic stream must produce crossings to map"
    mapped = map_events_to_source(raw, degraded.source_frames)
    assert [event.frame_index for event in mapped] == [
        degraded.source_frames[event.frame_index] for event in raw
    ]
    # Nothing but the frame index moves.
    for before, after in zip(raw, mapped):
        assert after.track_id == before.track_id
        assert after.timestamp == before.timestamp
        assert after.direction == before.direction


def test_the_window_resolution_report_measures_overlap_against_the_real_labels():
    """The widened window's cost, stated rather than hidden: how many
    label pairs' windows intersect and by how much."""
    labels = [
        Crossing(1, 411, "car", "toward", "probable"),
        Crossing(2, 416, "truck", "toward", "probable"),
        Crossing(3, 473, "truck", "toward", "probable"),
    ]
    undegraded = window_resolution(labels, DEFAULT_MATCH_WINDOW)
    assert undegraded["min_separation_frames"] == 5
    assert undegraded["closest_pair_frames"] == [411, 416]
    # [410, 415] and [415, 420] already share exactly one frame.
    assert undegraded["overlapping_label_pairs"] == 1
    assert undegraded["max_overlap_frames"] == 1
    # Measured against the UNDEGRADED overlap, not against zero: an
    # already-overlapping pair must not make the baseline itself read as
    # resolution-limited, or the flag says nothing at any level.
    assert undegraded["resolution_limited"] is False

    widened = window_resolution(labels, widen_for_gap(DEFAULT_MATCH_WINDOW, 15))
    assert widened["max_overlap_frames"] == 15
    assert widened["overlapping_label_pairs"] >= 1
    assert widened["max_overlap_fraction"] > undegraded["max_overlap_fraction"]
    assert widened["resolution_limited"] is True


# -- frame-rate decimation ---------------------------------------------------


def test_decimating_to_the_source_rate_returns_the_stream_untouched():
    stream = _synthetic_stream()
    degraded = decimate(stream, source_fps=30.0, target_fps=30.0)
    assert degraded.max_gap == 1
    assert degraded.source_frames == tuple(index for index, _t, _d in stream)
    assert [list(d) for _i, _t, d in degraded.frames] == [
        list(d) for _i, _t, d in stream
    ]


def test_decimation_keeps_the_sampling_grid_and_publishes_its_realised_gap():
    stream = _traffic([(50, 60.0)], last_frame=119)
    for target_fps, expected_kept, expected_gap in (
        (15.0, 60, 2),
        (10.0, 40, 3),
        (5.0, 20, 6),
        (2.0, 8, 15),
    ):
        degraded = decimate(stream, source_fps=30.0, target_fps=target_fps)
        assert len(degraded.frames) == expected_kept, target_fps
        assert degraded.max_gap == expected_gap, target_fps
        gaps = np.diff(np.asarray(degraded.source_frames))
        assert set(gaps.tolist()) == {expected_gap}, target_fps


def test_decimating_to_a_non_integer_stride_is_uniform_and_reports_its_worst_gap():
    """25 fps from 30 is not an integer stride: five frames in every six
    are kept, so the grid is irregular and the widening must be derived
    from the WORST gap it realises, never from a nominal stride of 1.2."""
    stream = _traffic([(50, 60.0)], last_frame=119)
    degraded = decimate(stream, source_fps=30.0, target_fps=25.0)
    assert len(degraded.frames) == 100  # 120 source frames at 25/30
    gaps = sorted(set(np.diff(np.asarray(degraded.source_frames)).tolist()))
    assert gaps == [1, 2]
    assert degraded.max_gap == 2


def test_the_decimation_grid_is_exact_rational_arithmetic_not_float_division():
    """The grid must be the same on every platform and every build.

    30 -> 11 fps is the case that separates the two: computed as a float,
    ``30 / 11`` rounds just below the true ratio, so slot 11 lands on
    source frame 29 instead of 30 and the whole grid shifts under it. The
    published rates happen not to expose that, which is exactly why it is
    pinned here -- a reproducibility guarantee only holds if something
    checks the case it was written for.
    """
    stream = _traffic([(50, 60.0)], last_frame=119)
    degraded = decimate(stream, source_fps=30.0, target_fps=11.0)
    assert degraded.source_frames[:12] == (0, 2, 5, 8, 10, 13, 16, 19, 21, 24, 27, 30)
    assert degraded.max_gap == 3


def test_decimation_renumbers_the_stream_it_hands_the_engine():
    """The engine is given a stream numbered from zero, exactly as it
    would be if the clip itself were recorded at the lower rate. Handing
    it the original indices instead would age tracks on the 30 fps clock
    while the tracker's own ``time_since_update`` still ticks once per
    call -- two clocks disagreeing inside one run."""
    degraded = decimate(_synthetic_stream(), source_fps=30.0, target_fps=5.0)
    assert [index for index, _t, _d in degraded.frames] == list(
        range(len(degraded.frames))
    )
    # The timestamps stay on the real clock: decimating a clip does not
    # change when anything happened.
    assert degraded.frames[1][1] == pytest.approx(6 / 30.0)


# -- dropped frames ----------------------------------------------------------


def test_dropping_no_frames_returns_the_stream_untouched():
    stream = _synthetic_stream()
    degraded = drop_frames(stream, fraction=0.0)
    assert degraded.max_gap == 1
    assert degraded.source_frames == tuple(index for index, _t, _d in stream)
    assert [list(d) for _i, _t, d in degraded.frames] == [
        list(d) for _i, _t, d in stream
    ]


def test_dropping_frames_removes_the_requested_fraction_and_keeps_the_order():
    stream = _synthetic_stream()
    total = len(stream)
    for fraction in (0.05, 0.10, 0.20, 0.30):
        degraded = drop_frames(stream, fraction=fraction)
        assert len(degraded.frames) == total - round(total * fraction), fraction
        assert list(degraded.source_frames) == sorted(degraded.source_frames)
        assert set(degraded.source_frames) <= set(range(total))
        assert degraded.max_gap >= 2, fraction


def test_dropped_frame_patterns_are_reproducible_and_level_specific():
    stream = _synthetic_stream()
    first = drop_frames(stream, fraction=0.20)
    again = drop_frames(stream, fraction=0.20)
    assert first.source_frames == again.source_frames
    other = drop_frames(stream, fraction=0.20, seed=ROBUSTNESS_SEED + 1)
    assert other.source_frames != first.source_frames
    assert drop_frames(stream, fraction=0.10).source_frames != first.source_frames


# -- detection dropout -------------------------------------------------------


def test_dropout_at_probability_zero_returns_the_detections_untouched():
    stream = _synthetic_stream()
    degraded = drop_detections(stream, probability=0.0)
    assert degraded.max_gap == 1
    assert [list(d) for _i, _t, d in degraded.frames] == [
        list(d) for _i, _t, d in stream
    ]
    assert degraded.detections_kept == degraded.detections_total


def test_dropout_keeps_a_subsequence_and_never_fabricates_a_detection():
    """Occlusion removes observations; it does not invent them, reorder
    them, or move a frame's detections to another frame. A transform that
    shuffled would still hit the right count and would change every
    tracker's association."""
    stream = _traffic([(50, 60.0), (80, 160.0), (120, 260.0)])
    degraded = drop_detections(stream, probability=0.30)
    assert len(degraded.frames) == len(stream)
    for (index, timestamp, kept), (source_index, source_timestamp, original) in zip(
        degraded.frames, stream
    ):
        assert index == source_index
        assert timestamp == source_timestamp
        remaining = list(original)
        for detection in kept:
            # A subsequence: every kept detection appears in the original,
            # in the original order.
            assert detection in remaining
            remaining = remaining[remaining.index(detection) + 1 :]
    assert 0 < degraded.detections_kept < degraded.detections_total


def test_dropout_removes_about_the_requested_share_of_detections():
    stream = _traffic([(50, 60.0), (80, 160.0), (120, 260.0)])
    for probability in (0.05, 0.10, 0.20, 0.30):
        degraded = drop_detections(stream, probability=probability)
        dropped = 1.0 - degraded.detections_kept / degraded.detections_total
        assert abs(dropped - probability) < 0.05, probability


def test_dropout_patterns_are_reproducible_and_level_specific():
    stream = _traffic([(50, 60.0), (80, 160.0), (120, 260.0)])
    first = drop_detections(stream, probability=0.20)
    again = drop_detections(stream, probability=0.20)
    assert [list(d) for _i, _t, d in first.frames] == [
        list(d) for _i, _t, d in again.frames
    ]
    other = drop_detections(stream, probability=0.20, seed=ROBUSTNESS_SEED + 1)
    assert [list(d) for _i, _t, d in other.frames] != [
        list(d) for _i, _t, d in first.frames
    ]


# -- box jitter --------------------------------------------------------------


def test_jitter_at_sigma_zero_returns_the_boxes_untouched():
    stream = _synthetic_stream()
    degraded = jitter_boxes(stream, sigma_px=0.0)
    assert [list(d) for _i, _t, d in degraded.frames] == [
        list(d) for _i, _t, d in stream
    ]


def test_jitter_moves_each_corner_independently_rather_than_translating_the_box():
    """Four independent draws, not one shared offset.

    A translation would leave every box the same size, so the detector's
    dominant measured residual -- box width -- would never be reproduced
    and the sweep would be calibrated against a quantity it does not
    perturb. The per-corner standard deviation is checked against sigma
    and the width residual against sigma * sqrt(2).
    """
    stream = _traffic([(50, 60.0), (80, 160.0), (120, 260.0)])
    sigma = 3.0
    degraded = jitter_boxes(stream, sigma_px=sigma)

    deltas = {"x1": [], "y1": [], "x2": [], "y2": []}
    widths = []
    for (_index, _timestamp, kept), (_i, _t, original) in zip(
        degraded.frames, stream
    ):
        for after, before in zip(kept, original):
            deltas["x1"].append(after.x1 - before.x1)
            deltas["y1"].append(after.y1 - before.y1)
            deltas["x2"].append(after.x2 - before.x2)
            deltas["y2"].append(after.y2 - before.y2)
            widths.append((after.x2 - after.x1) - (before.x2 - before.x1))

    assert len(widths) > 400
    for name, values in deltas.items():
        assert abs(float(np.std(values)) - sigma) < 0.4 * sigma, name
    # Independent corners: the width residual inflates by sqrt(2). A
    # shared per-box offset would leave it at zero.
    assert abs(float(np.std(widths)) - sigma * math.sqrt(2)) < 0.4 * sigma
    assert abs(float(np.corrcoef(deltas["x1"], deltas["x2"])[0, 1])) < 0.2


def test_jitter_never_produces_an_inverted_box():
    stream = _traffic([(50, 60.0), (80, 160.0), (120, 260.0)])
    degraded = jitter_boxes(stream, sigma_px=40.0)
    for _index, _timestamp, kept in degraded.frames:
        for detection in kept:
            assert detection.x1 <= detection.x2
            assert detection.y1 <= detection.y2


def test_jitter_degrades_geometry_only():
    """Confidence and class are a different degradation. Mixing them into
    the jitter protocol would make its curve unattributable."""
    stream = _traffic([(50, 60.0), (80, 160.0), (120, 260.0)])
    degraded = jitter_boxes(stream, sigma_px=4.0)
    for (index, timestamp, kept), (_i, source_timestamp, original) in zip(
        degraded.frames, stream
    ):
        assert timestamp == source_timestamp
        assert len(kept) == len(original)
        for after, before in zip(kept, original):
            assert after.score == before.score
            assert after.class_id == before.class_id
            assert after.class_name == before.class_name


def test_jitter_patterns_are_reproducible_and_level_specific():
    stream = _traffic([(50, 60.0)])
    first = jitter_boxes(stream, sigma_px=2.0)
    again = jitter_boxes(stream, sigma_px=2.0)
    assert [list(d) for _i, _t, d in first.frames] == [
        list(d) for _i, _t, d in again.frames
    ]
    other = jitter_boxes(stream, sigma_px=2.0, seed=ROBUSTNESS_SEED + 1)
    assert [list(d) for _i, _t, d in other.frames] != [
        list(d) for _i, _t, d in first.frames
    ]


# -- calibrating the sweep against the measured noise ------------------------


def test_the_corner_sigma_equivalent_inverts_the_jitter_the_sweep_applies():
    """The bridge between the measured residuals and the sweep's knob,
    checked by round trip against the jitter this module actually applies
    rather than against the algebra on its own.

    Getting the sqrt(2) the wrong way round would place the measured
    operating point four times off along the sweep, and every statement
    about which part of the curve is real would be wrong.
    """
    stream = _traffic([(50, 60.0), (80, 160.0), (120, 260.0)])
    sigma = 3.0
    degraded = jitter_boxes(stream, sigma_px=sigma)
    width_residual = []
    centre_residual = []
    for (_index, _timestamp, kept), (_i, _t, original) in zip(
        degraded.frames, stream
    ):
        for after, before in zip(kept, original):
            width_residual.append(
                (after.x2 - after.x1) - (before.x2 - before.x1)
            )
            centre_residual.append(
                (after.x1 + after.x2) / 2.0 - (before.x1 + before.x2) / 2.0
            )

    measured = {
        "residuals": {
            "box_width": {
                "std_px": float(np.std(width_residual)),
                "p95_abs_px": float(np.percentile(np.abs(width_residual), 95)),
            },
            "centre_x": {
                "std_px": float(np.std(centre_residual)),
                "p95_abs_px": float(np.percentile(np.abs(centre_residual), 95)),
            },
        }
    }
    recovered = corner_sigma_equivalents(measured)
    assert recovered["from_std"]["box_width"] == pytest.approx(sigma, rel=0.15)
    assert recovered["from_std"]["centre_x"] == pytest.approx(sigma, rel=0.15)
    assert recovered["from_p95"]["box_width"] == pytest.approx(sigma, rel=0.15)
    assert recovered["from_p95"]["centre_x"] == pytest.approx(sigma, rel=0.15)


def test_the_corner_sigma_equivalent_of_the_measured_clip_is_below_one_pixel():
    """The measured clip's noise, expressed in the sweep's own units. If
    this ever rises above the sweep's first non-zero point the range no
    longer brackets the measurement and must be re-chosen."""
    measured = json.loads(NOISE_REPORT.read_text())
    recovered = corner_sigma_equivalents(measured)
    values = list(recovered["from_std"].values()) + list(
        recovered["from_p95"].values()
    )
    assert values, "the noise report must yield at least one equivalent"
    assert max(values) < 1.0, recovered
    assert recovered["min_px"] == pytest.approx(min(values))
    assert recovered["max_px"] == pytest.approx(max(values))


# -- randomness must not leak between levels or between methods --------------


def test_each_level_draws_its_own_stream_so_adding_one_cannot_move_another():
    """Levels are seeded from the protocol and the level VALUE, never from
    a single sequence consumed in sweep order. Otherwise extending a sweep
    silently changes every published number after the insertion point,
    and a re-run would look like a measurement."""
    stream = _traffic([(50, 60.0), (80, 160.0)])
    short = {
        degraded.level: degraded
        for degraded in dropout_streams(stream, probabilities=(0.0, 0.05, 0.30))
    }
    long = {
        degraded.level: degraded
        for degraded in dropout_streams(
            stream, probabilities=(0.0, 0.05, 0.10, 0.20, 0.30)
        )
    }
    for level in (0.0, 0.05, 0.30):
        assert [list(d) for _i, _t, d in short[level].frames] == [
            list(d) for _i, _t, d in long[level].frames
        ], level

    short_drop = {
        degraded.level: degraded.source_frames
        for degraded in dropped_frame_streams(stream, fractions=(0.05, 0.30))
    }
    long_drop = {
        degraded.level: degraded.source_frames
        for degraded in dropped_frame_streams(
            stream, fractions=(0.05, 0.10, 0.20, 0.30)
        )
    }
    assert short_drop[0.05] == long_drop[0.05]
    assert short_drop[0.30] == long_drop[0.30]


def test_every_method_at_one_level_is_handed_the_identical_degraded_stream():
    """A measured difference between two methods must be a difference
    between the methods. Re-degrading per method would make it a
    difference between two draws of noise, and nothing downstream would
    say so."""
    stream = _traffic([(50, 60.0), (80, 160.0)])
    degraded = jitter_boxes(stream, sigma_px=4.0)
    seen = []

    def recorder(frames):
        seen.append(frames)
        return []

    run_protocol(
        [degraded],
        {"a": recorder, "b": recorder, "c": recorder},
        _synthetic_labels(),
        gate_name=GATE_NAME,
    )
    assert len(seen) == 3
    assert all(frames is seen[0] for frames in seen)
    # And what they saw is genuinely degraded, so an identity transform
    # could not satisfy this test by accident.
    assert [list(d) for _i, _t, d in seen[0]] != [list(d) for _i, _t, d in stream]


def test_two_runs_of_the_whole_family_are_byte_identical():
    stream = _traffic([(50, 60.0), (80, 160.0)])
    labels = _synthetic_labels()
    methods = build_methods(_gate())

    def family() -> str:
        entries = {}
        entries["frame_rate"] = run_protocol(
            frame_rate_streams(
                stream, source_fps=30.0, target_rates=(30, 15, 5)
            ),
            methods,
            labels,
            gate_name=GATE_NAME,
        )
        entries["dropped_frames"] = run_protocol(
            dropped_frame_streams(stream, fractions=(0.0, 0.2)),
            methods,
            labels,
            gate_name=GATE_NAME,
        )
        entries["detection_dropout"] = run_protocol(
            dropout_streams(stream, probabilities=(0.0, 0.2)),
            methods,
            labels,
            gate_name=GATE_NAME,
        )
        entries["box_jitter"] = run_protocol(
            jitter_streams(stream, sigmas_px=(0.0, 4.0)),
            methods,
            labels,
            gate_name=GATE_NAME,
        )
        return json.dumps(entries, sort_keys=True, allow_nan=False)

    assert family() == family()


# -- every published record must be arithmetically possible ------------------


def _ratio(numerator: int, denominator: int) -> float:
    """The scorer's own no-denominator convention, restated independently
    here so the check is not the implementation checking itself. Same
    helper, for the same reason, as ``test_bench_counting``."""
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )


def _assert_rates_follow_from_counts(record, n_ground_truth, where):
    """Every published rate recomputed from the record's own counts.

    Without this a report can carry cardinality-consistent counts and a
    fabricated headline. The engine at ``frame_rate@5 fps`` scores
    precision 1.0 on 2 predictions against 17 labels; an F1 of 0.9697
    pasted over its 0.2105 is arithmetically impossible on its face, and
    every summary above it can then be recomputed honestly from the
    fabricated record so that the whole report agrees with itself.
    """
    true_positives = record["true_positives"]
    false_positives = record["false_positives"]
    misses = record["misses"]
    n_predicted = record["n_predicted"]

    # Cardinality: the counts account for every label and every prediction.
    assert true_positives + false_positives == n_predicted, where
    assert true_positives + misses == n_ground_truth, where

    precision = _ratio(true_positives, n_predicted)
    recall = _ratio(true_positives, n_ground_truth)
    assert record["precision"] == pytest.approx(precision), where
    assert record["recall"] == pytest.approx(recall), where
    assert record["f1"] == pytest.approx(_f1(precision, recall)), where
    assert record["miss_rate"] == pytest.approx(
        _ratio(misses, n_ground_truth)
    ), where
    assert record["phantom_rate"] == pytest.approx(
        _ratio(false_positives, n_ground_truth)
    ), where
    assert record["signed_bias"] == n_predicted - n_ground_truth, where
    # Band-sweep rows publish the signed bias without the absolute count
    # error beside it; every full method record carries both.
    if "count_error" in record:
        assert record["count_error"] == abs(record["signed_bias"]), where


def test_every_published_record_is_arithmetically_self_consistent():
    """The layer under the headline: no published record may be impossible.

    The guards elsewhere anchor every SUMMARY to the per-level records.
    Nothing anchored the records to each other, so a single fabricated
    ``f1`` -- with its own precision and recall left untouched and every
    derived summary honestly recomputed from it -- inverted the session's
    strongest negative claim while the whole suite stayed green. This is
    the same treatment ``test_bench_counting`` gives counting_accuracy.json,
    applied to every method at every level, to the certain-only sub-record,
    and to every band-sweep row.
    """
    report = _report()
    total = report["labels"]["total"]
    certain = report["labels"]["certain"]
    checked = 0

    for protocol, block in report["protocols"].items():
        for entry in block["entries"]:
            window = entry["match_window"]
            for name, record in entry["methods"].items():
                where = f"{protocol}@{entry['level_label']}/{name}"
                assert record["n_ground_truth"] == total, where
                _assert_rates_follow_from_counts(record, total, where)
                checked += 1

                # The listed frames must agree with the counts beside them.
                assert len(record["miss_frames"]) == record["misses"], where
                frames = record["false_positive_frames"]
                if frames is None:
                    assert (
                        record["false_positives"] > FALSE_POSITIVE_FRAME_LIMIT
                    ), where
                else:
                    assert len(frames) == record["false_positives"], where

                # Greedy matching cannot beat the best possible assignment,
                # and no assignment can exceed either side's cardinality.
                best = record["max_cardinality_true_positives"]
                assert record["true_positives"] <= best <= min(
                    total, record["n_predicted"]
                ), where

                # A matched prediction lies inside the window that scored
                # it, which ties the deltas to the published window.
                delta = record["matched_frame_delta"]
                if record["true_positives"]:
                    assert -window["frames_before"] <= delta["min"], where
                    assert delta["max"] <= window["frames_after"], where
                    assert delta["min"] <= delta["mean"] <= delta["max"], where
                else:
                    assert delta == {"mean": None, "min": None, "max": None}, where

                # The certain-only sub-record, on its own denominator.
                sub = record["certain_only"]
                assert sub["n_ground_truth"] == certain, where
                sub_precision = _ratio(sub["true_positives"], sub["n_predicted"])
                sub_recall = _ratio(sub["true_positives"], certain)
                assert sub["precision"] == pytest.approx(sub_precision), where
                assert sub["recall"] == pytest.approx(sub_recall), where
                assert sub["f1"] == pytest.approx(
                    _f1(sub_precision, sub_recall)
                ), where
                assert sub["true_positives"] <= min(certain, sub["n_predicted"]), where

    # The band sweep publishes the same rates and must obey the same
    # arithmetic; it carries no denominator of its own, so it is checked
    # against the label total the rest of the report is scored on.
    for block in report["protocols"][PROTOCOL_FRAME_RATE]["band_sweep_by_rate"]:
        for row in block["entries"]:
            where = f"band {row['band_px']:g}px/{row['tracker']}@{row['level_label']}"
            _assert_rates_follow_from_counts(row, total, where)
            checked += 1

    # 21 levels x 9 methods + 6 rates x 2 trackers x 7 bands.
    assert checked == 21 * 9 + 6 * 2 * 7, checked


# -- the reduction test: every protocol's identity level ---------------------


def test_every_protocol_at_its_identity_level_reproduces_the_undegraded_score():
    """The protocol's own correctness proof.

    30 fps, 0 % dropped frames, p = 0 and sigma = 0 must each score
    EXACTLY what the undegraded stream scores, for every one of the nine
    methods. A protocol that does not reduce to the baseline is measuring
    something other than the degradation it names.

    The stream and the labels MUST correspond. A fixture whose traffic
    crosses at frames the labels never name scores TP = 0 and
    P = R = F1 = 0 on both sides of the comparison, so every
    crossing-level assertion below compares zero against zero and the
    test degenerates into a count check wearing a timing check's name --
    a protocol that shifted every prediction two frames late at every
    identity level would still pass it. ``_synthetic_stream`` crosses at
    exactly the frames ``_synthetic_labels`` names, so the gate
    compositions score F1 = 1.0 undegraded and any timing shift moves
    them.
    """
    stream = _synthetic_stream()
    labels = _synthetic_labels()
    methods = build_methods(_gate())

    undegraded = {
        name: match_crossings(
            method(list(stream)), labels, DEFAULT_MATCH_WINDOW, gate_name=GATE_NAME
        ).as_dict()
        for name, method in methods.items()
    }

    # The fixture must be able to SEE a timing shift, or the comparison
    # below is zero against zero. Every gate composition lands all three
    # crossings inside the undegraded window; a protocol that moved a
    # prediction off its label would drop these off 1.0 and the equality
    # assertions would then have something to disagree about.
    for name in ("engine+gate", "centroid+gate", "greedy-iou+gate"):
        assert undegraded[name]["true_positives"] == len(labels), name
        assert undegraded[name]["f1"] == 1.0, (name, undegraded[name]["f1"])

    identity = {
        PROTOCOL_FRAME_RATE: frame_rate_streams(
            stream, source_fps=30.0, target_rates=(30,)
        ),
        PROTOCOL_DROPPED_FRAMES: dropped_frame_streams(stream, fractions=(0.0,)),
        PROTOCOL_DETECTION_DROPOUT: dropout_streams(stream, probabilities=(0.0,)),
        PROTOCOL_BOX_JITTER: jitter_streams(stream, sigmas_px=(0.0,)),
    }
    for protocol, streams in identity.items():
        entries = run_protocol(streams, methods, labels, gate_name=GATE_NAME)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["match_window"]["frames_before"] == 1, protocol
        assert entry["match_window"]["frames_after"] == 4, protocol
        for name, record in entry["methods"].items():
            for field in (
                "n_predicted",
                "true_positives",
                "precision",
                "recall",
                "f1",
                "signed_bias",
                "count_error",
            ):
                assert record[field] == undegraded[name][field], (
                    protocol,
                    name,
                    field,
                )


# -- the published report ----------------------------------------------------


def test_the_published_report_records_the_seed_and_the_measured_jitter():
    report = _report()
    assert report["seed"] == ROBUSTNESS_SEED
    calibration = report["jitter_calibration"]
    assert calibration["source_report"] == "detection_noise.json"
    measured = json.loads(NOISE_REPORT.read_text())
    assert calibration["measured_residuals"] == measured["residuals"]
    equivalents = calibration["corner_sigma_equivalent_px"]
    assert equivalents == corner_sigma_equivalents(measured)
    # The sweep must bracket the measurement, or its range is decoration.
    sigmas = [entry["level"] for entry in report["protocols"]["box_jitter"]["entries"]]
    assert equivalents["max_px"] < max(sigmas)
    assert calibration["stress_multiple_at_max_sigma"]["lowest"] > 1.0


def test_the_published_report_reduces_exactly_to_the_counting_report():
    """The published artefact's own reduction proof, on the real clip.

    Every protocol's identity level must reproduce ``counting_accuracy``
    to the last digit, for all nine methods. This is what says the four
    degradations are the only thing that changed.
    """
    report = _report()
    baseline = json.loads(COUNTING_REPORT.read_text())["methods"]
    identity = {
        PROTOCOL_FRAME_RATE: 30.0,
        PROTOCOL_DROPPED_FRAMES: 0.0,
        PROTOCOL_DETECTION_DROPOUT: 0.0,
        PROTOCOL_BOX_JITTER: 0.0,
    }
    assert set(identity) == set(report["protocols"])
    for protocol, level in identity.items():
        entry = _entry(report, protocol, level)
        assert entry["match_window"]["frames_before"] == 1, protocol
        assert entry["match_window"]["frames_after"] == 4, protocol
        assert set(entry["methods"]) == set(baseline)
        for name, record in entry["methods"].items():
            full = baseline[name]["full"]
            assert record["n_predicted"] == full["n_predicted"], (protocol, name)
            assert record["true_positives"] == full["true_positives"], (
                protocol,
                name,
            )
            assert record["false_positives"] == len(full["false_positives"])
            assert record["misses"] == len(full["misses"])
            assert record["precision"] == full["precision"], (protocol, name)
            assert record["recall"] == full["recall"], (protocol, name)
            assert record["f1"] == full["f1"], (protocol, name)
            assert record["signed_bias"] == full["signed_bias"], (protocol, name)
            assert (
                record["certain_only"]["f1"]
                == baseline[name]["certain_only"]["f1"]
            ), (protocol, name)

    engine = _entry(report, PROTOCOL_FRAME_RATE, 30.0)["methods"]["engine+gate"]
    assert engine["true_positives"] == 16
    assert engine["false_positives"] == 1
    assert engine["misses"] == 1
    assert engine["n_predicted"] == 17
    assert engine["signed_bias"] == 0
    assert round(engine["f1"], 4) == 0.9412


def test_the_published_report_carries_the_window_it_scored_each_level_with():
    """Ruling: publish the window used at every rate. A widened window
    quoted nowhere is a scorer nobody can audit."""
    report = _report()
    for protocol, block in report["protocols"].items():
        for entry in block["entries"]:
            window = entry["match_window"]
            assert window["frames_before"] == 1, (protocol, entry["level"])
            assert (
                window["frames_after"]
                == DEFAULT_MATCH_WINDOW.frames_after + entry["max_gap_frames"] - 1
            ), (protocol, entry["level"])
            assert entry["window_widened_by_frames"] == entry["max_gap_frames"] - 1
            assert (
                f"gap of {entry['max_gap_frames']} original frames"
                in window["reason"]
            ), (protocol, entry["level"])


def test_the_published_report_marks_where_the_widened_window_loses_resolution():
    """The honest consequence of §1, on the record: adjacent labels'
    windows already share a frame undegraded, and every widening enlarges
    that overlap."""
    report = _report()
    resolution = report["label_resolution"]
    assert resolution["min_separation_frames"] == 5
    assert resolution["closest_pair_frames"] == [411, 416]
    assert resolution["first_rate_with_overlapping_windows"] == 30.0
    assert resolution["first_rate_with_a_second_overlapping_pair"] == 10.0

    for entry in report["protocols"][PROTOCOL_FRAME_RATE]["entries"]:
        limited = entry["resolution"]["resolution_limited"]
        assert limited == (entry["window_widened_by_frames"] > 0), entry["level"]
    slowest = _entry(report, PROTOCOL_FRAME_RATE, 2.0)
    assert slowest["resolution"]["max_overlap_frames"] == 15
    assert slowest["resolution"]["overlapping_label_pairs"] >= 3


def test_the_published_report_answers_the_band_step_over_question_from_its_series():
    """The headline must follow from the numbers under it.

    Task 14 measured 1.13 px/frame at the gate, at which no band is ever
    stepped over. The claim that decimation restores the step-over mode is
    checked against the published per-rate displacement and the published
    band sweep, not asserted.
    """
    report = _report()
    question = report["questions"]["band_step_over"]
    rates = report["protocols"][PROTOCOL_FRAME_RATE]["entries"]

    approach = {
        str(entry["level"]): entry["median_gate_approach_px_per_frame"]
        for entry in rates
    }
    assert question["approach_px_per_frame_by_rate"] == approach
    # Decimation is the protocol that is meant to restore the mode, so the
    # displacement must actually grow with the stride.
    assert approach["2.0"] > 10.0 * approach["30.0"]

    # The series is measured with GreedyIoUTracker because the engine's own
    # tracker stops producing tracks at the lowest rate -- which is itself a
    # published finding, pinned here so it cannot quietly stop being true.
    engine_tracked = {
        str(entry["level"]): entry[
            "median_gate_approach_px_per_frame_engine_tracker"
        ]
        for entry in rates
    }
    assert question["engine_tracked_approach_px_per_frame_by_rate"] == engine_tracked
    assert engine_tracked["30.0"] == pytest.approx(1.1326310188320008)
    assert engine_tracked["2.0"] is None
    assert approach["2.0"] is not None

    # The criterion must isolate the RULE from the TRACKER. A bare
    # under-count does not: below 15 fps the engine's association collapses
    # and the gate rule under-counts by exactly as much as every band, so a
    # "misses exceed phantoms" test reports step-over wherever tracking
    # merely stopped. The confound-free signature is the band rule emitting
    # FEWER events than the gate rule driven by the SAME tracker over the
    # SAME stream: both see identical tracks, so a shortfall means a track
    # the gate rule counted reached the band and was DECLINED. BandCounter
    # declines for two reasons -- no sample inside the band (step-over) and
    # a sample inside it with zero perpendicular displacement (a vehicle
    # stopped on the band) -- and the published criterion names both,
    # because only the measured displacement, not the arithmetic, rules the
    # second one out here.
    sweep = report["protocols"][PROTOCOL_FRAME_RATE]["band_sweep_by_rate"]
    rows = [row for block in sweep for row in block["entries"]]
    assert rows, "the band sweep must publish rows"
    for row in rows:
        gate_predicted = _entry(report, PROTOCOL_FRAME_RATE, row["level"])[
            "methods"
        ][f"{row['tracker']}+gate"]["n_predicted"]
        assert row["gate_rule_n_predicted"] == gate_predicted, row
        assert row["stepped_over"] == (row["n_predicted"] < gate_predicted), row

    stepped = [row for row in rows if row["stepped_over"]]
    observed = any(row["stepped_over"] for row in rows)
    assert question["step_over_trade_off_reappears"] == observed
    assert question["step_over_rows"] == stepped
    assert question["verdict"].strip() != ""

    # Both pins below derive from `stepped`, so an empty `stepped` would
    # reduce them to False == False and None == None and they would stop
    # biting without anything going red. The sweep observes 2 rows today;
    # if it ever observed none, that is a finding to be looked at rather
    # than a guard to be quietly satisfied.
    assert stepped, (
        "no swept (tracker, rate, band) row steps over, so the two pins "
        "below are vacuous -- if the sweep genuinely stopped reaching the "
        "mode, that is a change to the published answer, not to this test"
    )

    # The two fields carrying the INTERESTING half of the answer, both
    # recomputed rather than trusted. "NOT with the engine's own tracker at
    # any rate" is the whole point of the finding -- and it is printed into
    # the figure's own title -- while the rate is what says where the mode
    # begins. Neither follows from the rows above unless it is derived from
    # them here.
    assert question["step_over_with_the_engine_tracker"] == any(
        row["tracker"] == "engine" for row in stepped
    )
    assert question["first_rate_with_step_over"] == max(
        (row["level"] for row in stepped), default=None
    )
    assert question["trackers_swept"] == sorted({row["tracker"] for row in rows})

    # The criterion must state both ways BandCounter can decline a track
    # the gate rule counted, not just the interesting one -- see
    # test_a_band_shortfall_has_a_second_cause_besides_step_over.
    criterion = question["criterion"]
    assert "can only mean" not in criterion
    assert "ZERO" in criterion and "perpendicular displacement" in criterion

    # A sweep driven only by a tracker that dies before the displacement
    # grows could never observe step-over at all, so the question is also
    # asked of an association that survives to the lowest rate.
    assert {row["tracker"] for row in rows} == {"engine", "greedy-iou"}
    assert any(
        row["tracker"] == "greedy-iou" and row["level"] == 2.0 for row in rows
    )


def test_a_band_shortfall_has_a_second_cause_besides_step_over():
    """Why the published criterion no longer says "can only mean".

    A band shortfall against the gate rule means a track reached the band
    and was DECLINED. ``BandCounter.update`` declines for two reasons, not
    one: no sample landed inside the band -- step-over -- and a sample
    landed inside it with zero perpendicular displacement from its
    predecessor. The second is demonstrated here rather than argued, so
    the criterion's wording is pinned to a fact about the rule.

    It cannot explain either row this sweep observed, because per-sample
    displacement only GROWS under decimation while a stopped vehicle needs
    it to reach zero. That is a claim about the measurement, though, not
    about the arithmetic, and the criterion now says which.
    """
    from trafficlens.bench.baselines import BandCounter

    gate = _gate()
    counter = BandCounter(gate, band_px=20.0)

    # Sitting exactly on the band, having moved not at all across the gate.
    on_band = (100.0, GATE_Y)
    assert counter.update(1, "car", on_band, on_band, 10, 10 / 30.0) is None
    assert counter.total() == 0

    # The same track, once it actually moves through: counted.
    assert (
        counter.update(1, "car", (100.0, GATE_Y - 2.0), (100.0, GATE_Y + 2.0), 11, 0.36)
        is not None
    )
    assert counter.total() == 1

    # And the two causes are distinct: a track that never lands in the band
    # is declined for the OTHER reason, while moving perfectly well.
    stepped = BandCounter(gate, band_px=5.0)
    assert (
        stepped.update(
            2, "car", (200.0, GATE_Y - 40.0), (200.0, GATE_Y + 40.0), 12, 0.4
        )
        is None
    )
    assert stepped.total() == 0


def test_the_published_report_answers_the_tracker_separation_question():
    """Task 14's sharpest negative was that all three trackers scored
    identically on clean footage. Whether degradation separates them is
    recomputed here from the published per-level F1s rather than trusted.

    Everything the verdict rests on is recomputed from the PER-LEVEL
    RECORDS -- the same numbers the figures are drawn from -- and never
    from the question block's own summary of them. ``f1_by_level`` is a
    convenience copy; reading the engine's score out of it and comparing
    that against the records would let a fabricated copy agree with
    itself, and the strongest negative claim this session makes could be
    inverted by editing one number and two lists.
    """
    report = _report()
    question = report["questions"]["tracker_separation"]
    trackers = ("engine+gate", "centroid+gate", "greedy-iou+gate")

    records: dict[str, dict[str, float]] = {}
    spreads = {}
    for protocol, block in report["protocols"].items():
        for entry in block["entries"]:
            key = f"{protocol}@{entry['level_label']}"
            scores = {name: entry["methods"][name]["f1"] for name in trackers}
            records[key] = scores
            spreads[key] = max(scores.values()) - min(scores.values())

    # The convenience copy must BE the records, to the last digit, or
    # nothing derived from it means anything.
    assert set(question["f1_by_level"]) == set(records)
    for key, scores in records.items():
        assert question["f1_by_level"][key] == scores, key
    assert question["f1_spread_by_level"] == pytest.approx(spreads)
    assert question["levels_measured"] == len(records)
    assert question["methods_compared"] == list(trackers)

    assert question["max_f1_spread"] == pytest.approx(max(spreads.values()))
    assert question["trackers_separate"] == (max(spreads.values()) > 0.0)
    assert question["levels_where_trackers_differ"] == sorted(
        key for key, spread in spreads.items() if spread > 0.0
    )
    assert question["verdict"].strip() != ""

    # Separating is not the same as separating in the engine's favour, and
    # a headline that said only "they separate" would read as the opposite
    # of what the numbers show. The DIRECTION is recomputed here, from the
    # records.
    engine_best = sorted(
        key
        for key, scores in records.items()
        if scores["engine+gate"] == max(scores.values())
    )
    engine_worst = sorted(
        key
        for key, scores in records.items()
        if scores["engine+gate"] == min(scores.values())
    )
    assert question["levels_where_the_engine_scores_highest"] == engine_best
    assert question["levels_where_the_engine_scores_lowest"] == engine_worst
    engine_leads = any(spreads[key] > 0.0 for key in engine_best)
    assert question["engine_leads_at_any_degraded_level"] == engine_leads

    # The prose headline must point the same way as the booleans under it.
    # A reader sees the verdict, not the lists, so an inverted verdict is
    # the cheapest way to launder this result and the one worth pinning.
    verdict = question["verdict"]
    if engine_leads:
        assert "separate AGAINST the engine" not in verdict
    else:
        assert "separate AGAINST the engine" in verdict
        assert "LEADS" not in verdict
        # ... and the count it quotes must be the recomputed one.
        differing = sorted(key for key, spread in spreads.items() if spread > 0.0)
        trails = sorted(set(engine_worst) & set(differing))
        assert f"at {len(trails)} of them it scores LOWEST" in verdict
        assert f"{len(differing)} of the {len(spreads)} degradation levels" in verdict


def test_the_published_ablation_attributes_the_engine_collapse_to_its_iou_floor():
    """The mechanism behind question (b), measured rather than inferred.

    Claiming the engine's 0.8 IoU association floor is what breaks it under
    degradation is an assertion about a constant until something varies
    that constant and nothing else. The ablation does; this test pins its
    shipped-floor column to the main sweep, so it can never drift into
    being a separate, unrelated measurement whose comparison means nothing.
    """
    from trafficlens.core.constants import (
        BASELINE_GREEDY_IOU_THRESH,
        TRACK_MATCH_IOU,
    )

    report = _report()
    ablation = report["association_floor_ablation"]
    shipped = f"{TRACK_MATCH_IOU:g}"
    loosened = f"{BASELINE_GREEDY_IOU_THRESH:g}"
    assert ablation["shipped_floor"] == TRACK_MATCH_IOU
    assert ablation["floors"] == [TRACK_MATCH_IOU, BASELINE_GREEDY_IOU_THRESH]
    assert set(ablation["by_protocol"]) == set(report["protocols"])

    gains = []
    for protocol, rows in ablation["by_protocol"].items():
        entries = report["protocols"][protocol]["entries"]
        assert [row["level"] for row in rows] == [e["level"] for e in entries]
        for row, entry in zip(rows, entries):
            # Same stream, same window, same counting rule: only the floor
            # differs, so the shipped column must BE the main sweep's number.
            assert row["f1"][shipped] == entry["methods"]["engine+gate"]["f1"], (
                protocol,
                row["level"],
            )
            assert (
                row["n_predicted"][shipped]
                == entry["methods"]["engine+gate"]["n_predicted"]
            )
            gains.append(
                {
                    "protocol": protocol,
                    "level": row["level"],
                    "gain": row["f1"][loosened] - row["f1"][shipped],
                }
            )

    best = max(gains, key=lambda row: row["gain"])
    assert ablation["largest_f1_gain"]["gain"] == pytest.approx(best["gain"])
    assert ablation["largest_f1_gain"]["protocol"] == best["protocol"]
    assert ablation["loosening_the_floor_helps"] == (best["gain"] > 0.0)

    # "The floor explains it" must be a per-protocol claim, not a global
    # one: the same loosening that recovers 0.6 F1 under decimation buys
    # nothing under detection dropout, and a verdict that averaged the two
    # would misattribute the second failure to the first's cause.
    threshold = ablation["gain_threshold"]
    per_protocol = {
        protocol: max(
            row["gain"] for row in gains if row["protocol"] == protocol
        )
        for protocol in ablation["by_protocol"]
    }
    assert ablation["protocols_the_floor_explains"] == sorted(
        protocol for protocol, gain in per_protocol.items() if gain > threshold
    )
    assert ablation["protocols_the_floor_does_not_explain"] == sorted(
        protocol for protocol, gain in per_protocol.items() if gain <= threshold
    )
    assert PROTOCOL_DETECTION_DROPOUT in (
        ablation["protocols_the_floor_does_not_explain"]
    )
    # The undegraded row is what says the floor was tuned for clean footage
    # rather than being simply wrong: loosening it must not help there.
    identity = ablation["by_protocol"][PROTOCOL_FRAME_RATE][0]
    assert identity["level"] == 30.0
    assert identity["f1"][loosened] <= identity["f1"][shipped]


def test_the_published_report_states_no_speed_in_kilometres_per_hour():
    """Absolute along-road scale cannot be anchored on this clip, so no
    degradation result may be expressed as a speed. Displacements are
    published in pixels per frame."""
    text = ROBUSTNESS_REPORT.read_text().lower()
    assert "kmh" not in text
    assert "km/h" not in text
    assert "km_h" not in text


def test_the_published_report_publishes_crossing_level_metrics_not_only_counts():
    """§2: a count error alone is the metric Task 14 proved worthless --
    the band rule predicted 18 against 17 real crossings while landing one
    or two on the right frame."""
    report = _report()
    required = {
        "n_predicted",
        "n_ground_truth",
        "true_positives",
        "false_positives",
        "misses",
        "precision",
        "recall",
        "f1",
        "count_error",
        "signed_bias",
        "miss_rate",
        "phantom_rate",
        "max_cardinality_true_positives",
    }
    for protocol, block in report["protocols"].items():
        for entry in block["entries"]:
            for name, record in entry["methods"].items():
                missing = required - set(record)
                assert missing == set(), (protocol, entry["level"], name, missing)


def test_the_published_figures_exist_for_every_protocol():
    report = _report()
    expected = {f"robustness_{protocol}.png" for protocol in report["protocols"]}
    expected.add("robustness_band_step_over.png")
    for name in sorted(expected):
        path = FIGURE_DIR / name
        assert path.is_file(), f"missing figure {path.relative_to(ROOT)}"
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
        assert path.stat().st_size > 4096, name
    assert report["figures"] == sorted(expected)


def test_each_committed_figure_is_the_one_its_own_report_renders(tmp_path):
    """Ties every committed PNG to the JSON it claims to draw.

    Magic bytes and a size floor say a file is a PNG, not that it is a
    picture of THIS report: a figure and the numbers under it can drift
    apart silently, and the figure is what most readers actually look at.
    Re-rendering the committed report and comparing bytes is the only
    check that sees that, and matplotlib's Agg output is deterministic for
    a fixed build -- five figures reproduce byte-for-byte across repeated
    runs here.

    The cost is that the comparison is sensitive to the RENDERING BUILD as
    well as to the data: a different matplotlib or freetype lays glyphs
    out differently, and this test would then fail on committed artefacts
    that are perfectly honest. The failure message says so, and the fix in
    that case is to regenerate the figures rather than to weaken this.
    """
    pytest.importorskip("matplotlib")
    script = _load_script("bench_robustness")
    report = _report()

    for path in script.write_figures(report, tmp_path):
        committed = FIGURE_DIR / path.name
        assert path.read_bytes() == committed.read_bytes(), (
            f"{committed.relative_to(ROOT)} is not what robustness.json "
            f"renders. Either the figure is stale against the report -- "
            f"regenerate with scripts/bench_robustness.py -- or this "
            f"machine's matplotlib/freetype build lays the figure out "
            f"differently from the one that produced the committed file."
        )

    # ... and the comparison must be capable of failing: if the renderer
    # ignored the numbers it plots, every figure would match every report
    # and the assertion above would be decoration.
    perturbed = copy.deepcopy(report)
    entry = perturbed["protocols"][PROTOCOL_FRAME_RATE]["entries"][-1]
    entry["methods"]["engine+gate"]["f1"] = 0.5
    altered = tmp_path / "altered"
    rendered = {
        path.name: path.read_bytes()
        for path in script.write_figures(perturbed, altered)
    }
    name = f"robustness_{PROTOCOL_FRAME_RATE}.png"
    assert rendered[name] != (FIGURE_DIR / name).read_bytes(), (
        "changing a plotted F1 left the rendered figure identical, so the "
        "figure is not a function of the report and pinning its bytes "
        "proves nothing"
    )


# -- the report assembler ----------------------------------------------------


def _synthetic_ground_truth(labels) -> GroundTruth:
    return GroundTruth(
        path=Path("data/groundtruth/motorway_inbound_gt.json"),
        clip="motorway-a40.webm",
        fps=30.0,
        start_frame=0,
        end_frame=LAST_FRAME,
        gate_name=GATE_NAME,
        gate_start=(0.06, 0.80),
        gate_end=(0.46, 0.80),
        protocol="data/groundtruth/PROTOCOL.md",
        labeller="Safdar Hussain",
        labelled_on="2026-08-15",
        crossings=tuple(labels),
    )


def _built_report() -> dict:
    script = _load_script("bench_robustness")
    return script.build_report(
        # The traffic must cross where the labels say it does. Nothing
        # downstream of this helper reads a crossing-level score, so the
        # old mismatched pairing was harmless here -- but it was the last
        # copy in this file of the fixture bug the reduction test had, and
        # a harmless copy is what the next reader learns from.
        _synthetic_stream(),
        _synthetic_ground_truth(_synthetic_labels()),
        _gate(),
        noise=json.loads(NOISE_REPORT.read_text()),
        detector={"model": "yolo11s.pt"},
        target_rates=(30.0, 15.0, 5.0),
        drop_fractions=(0.0, 0.2),
        dropout_probabilities=(0.0, 0.2),
        jitter_sigmas=(0.0, 4.0),
        band_values=(5.0, 20.0),
    )


def test_the_assembled_report_is_byte_identical_across_two_builds():
    first = json.dumps(_built_report(), sort_keys=True, allow_nan=False)
    second = json.dumps(_built_report(), sort_keys=True, allow_nan=False)
    assert first == second


def test_the_assembled_report_names_the_label_file_and_never_a_filesystem_path():
    """``reports/`` is tracked, and the same label set lives at a
    different absolute path on every machine."""
    report = _built_report()
    assert report["ground_truth"] == "motorway_inbound_gt.json"
    text = json.dumps(report)
    assert "/Us" + "ers/" not in text
    assert str(ROOT) not in text


def test_the_decimation_sweep_takes_its_identity_rate_from_the_clip():
    """The identity level is the CLIP's frame rate, not the constant 30.

    ``reduction.identity_levels`` already derives it from ``truth.fps``, so
    a hard-coded 30 at the head of the sweep is a second source of truth
    for the same number -- and on any clip that is not 30 fps it is the
    wrong one. ``decimate`` removes frames and cannot invent them, so the
    old constant did not merely mislabel the identity: it killed the run.
    """
    script = _load_script("bench_robustness")

    # The shipped clip, unchanged.
    assert script.target_rates_for(30.0) == [30.0, 25.0, 15.0, 10.0, 5.0, 2.0]

    # A clip below some of the sweep points keeps only the ones it can
    # actually be decimated to, and still leads with its own rate.
    assert script.target_rates_for(24.0) == [24.0, 15.0, 10.0, 5.0, 2.0]
    assert script.target_rates_for(12.0) == [12.0, 10.0, 5.0, 2.0]
    for source in (30.0, 25.0, 24.0, 12.0, 59.94):
        rates = script.target_rates_for(source)
        assert rates[0] == source
        assert all(rate < source for rate in rates[1:])
        assert rates == sorted(rates, reverse=True)

    # And the assembled report agrees with itself: the sweep's top level
    # IS the identity level the reduction block names.
    truth = _synthetic_ground_truth(_synthetic_labels())
    report = script.build_report(
        _synthetic_stream(),
        truth,
        _gate(),
        noise=json.loads(NOISE_REPORT.read_text()),
        detector={"model": "yolo11s.pt"},
        drop_fractions=(0.0,),
        dropout_probabilities=(0.0,),
        jitter_sigmas=(0.0,),
        band_values=(20.0,),
    )
    levels = [e["level"] for e in report["protocols"][PROTOCOL_FRAME_RATE]["entries"]]
    assert levels[0] == truth.fps
    assert report["reduction"]["identity_levels"][PROTOCOL_FRAME_RATE] == truth.fps
    assert levels == script.target_rates_for(truth.fps)


def test_the_first_overlapping_rate_is_the_highest_one_not_the_first_listed():
    """"First" means first as the rate FALLS, so it must be computed from
    the qualifying levels rather than read off the head of a list that
    happens to be in descending sweep order. Reordering the sweep must not
    move a published answer.

    The fixture is chosen so both fields can actually SEE the difference:
    labels seven frames apart start overlapping at 10 fps and stay
    overlapping at 5, so each field has two qualifying levels and the head
    of the list differs between the two sweep orders while the maximum
    does not. On the shipped ``_synthetic_labels`` no level overlaps at
    all, both fields are ``None`` under every ordering, and a test built on
    them would agree with itself while measuring nothing.
    """
    script = _load_script("bench_robustness")
    labels = [
        Crossing(1, 31, "car", "toward", "certain"),
        Crossing(2, 38, "car", "toward", "certain"),
        Crossing(3, 45, "car", "toward", "certain"),
        Crossing(4, 104, "car", "toward", "certain"),
    ]
    stream = _traffic([(31, 60.0), (38, 160.0), (45, 260.0), (104, 360.0)])
    truth = _synthetic_ground_truth(labels)
    common = dict(
        noise=json.loads(NOISE_REPORT.read_text()),
        detector={"model": "yolo11s.pt"},
        drop_fractions=(0.0,),
        dropout_probabilities=(0.0,),
        jitter_sigmas=(0.0,),
        band_values=(20.0,),
    )
    descending = script.build_report(
        stream, truth, _gate(), target_rates=(30.0, 10.0, 5.0), **common
    )
    scrambled = script.build_report(
        stream, truth, _gate(), target_rates=(30.0, 5.0, 10.0), **common
    )

    # The fixture must qualify at more than one rate, or "highest" and
    # "first listed" cannot disagree and this test is vacuous.
    qualifying = [
        entry["level"]
        for entry in descending["protocols"][PROTOCOL_FRAME_RATE]["entries"]
        if entry["resolution"]["overlapping_label_pairs"] >= 2
    ]
    assert sorted(qualifying) == [5.0, 10.0], qualifying

    for field in (
        "first_rate_with_overlapping_windows",
        "first_rate_with_a_second_overlapping_pair",
    ):
        assert descending["label_resolution"][field] == 10.0, field
        assert scrambled["label_resolution"][field] == 10.0, field

    # Same property for the step-over rate, which already took the max.
    assert (
        descending["questions"]["band_step_over"]["first_rate_with_step_over"]
        == scrambled["questions"]["band_step_over"]["first_rate_with_step_over"]
    )


def test_the_ablation_actually_varies_the_association_floor():
    """The ablation's whole claim is that ONE constant moved.

    If both columns were built from the same tracker the report would show
    two identical series and the verdict would read as a measurement of
    nothing -- the published-artefact test could not see it, because it
    compares the shipped column against the main sweep and both would
    still agree. So the varying is asserted directly, on a stream chosen
    to be exactly the case that separates the two floors: 2 px per frame
    on an 80 px box sampled every sixth frame, where the box overlaps its
    own predicted position by about 0.74 -- under the shipped 0.8 floor,
    over SORT's 0.3.
    """
    from trafficlens.core.constants import (
        BASELINE_GREEDY_IOU_THRESH,
        TRACK_MATCH_IOU,
    )

    script = _load_script("bench_robustness")
    fast = _traffic(SYNTHETIC_CROSSINGS, speed_px=2.0)
    stream = decimate(fast, source_fps=30.0, target_fps=5.0)

    strict = script._engine_gate_with_floor(_gate(), TRACK_MATCH_IOU)(stream.frames)
    loose = script._engine_gate_with_floor(_gate(), BASELINE_GREEDY_IOU_THRESH)(
        stream.frames
    )
    assert len(strict) == 0, "the shipped floor must lose this stream entirely"
    assert len(loose) > 0, "the loosened floor must keep it"


def test_the_figure_writer_produces_one_png_per_protocol(tmp_path):
    pytest.importorskip("matplotlib")
    script = _load_script("bench_robustness")
    report = _built_report()
    written = script.write_figures(report, tmp_path)
    assert sorted(path.name for path in written) == report["figures"]
    for path in written:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
