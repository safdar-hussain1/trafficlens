"""Keep the cross-surface parity fixture honest.

``web/src/fixtures/parity.json`` is the artefact that turns "the browser
runs the same engine" from an assertion into a test: it carries the
inputs of a real clip window plus deliberately constructed boundary
cases, and beside them the outputs the PYTHON engine produced. The
Vitest suite replays those inputs through the TypeScript engine and
demands the same answers.

A fixture that silently drifted from the Python engine would make that
comparison meaningless, so the first test here regenerates it and
demands byte equality. On its own that assertion is a tautology -- a
generator always reproduces its own output, and it would pass just as
happily on an empty fixture. So it is paired with three independent
families of check that a degenerate fixture fails:

1. **The straddle inventory.** Every boundary the brief mandates must be
   present at least once, with a non-empty floor, and each is
   re-measured from the fixture's own recorded inputs -- the IoU is
   recomputed and compared to ``TRACK_MATCH_IOU``, the on-line anchor is
   pushed back through ``side_of_line``, the tied argmax column is
   re-read in float32. A case that stopped straddling its boundary fails
   here rather than passing quietly.
2. **Counterfactuals.** Where a case exists to pin one branch against
   another, the other branch is computed too and asserted to give a
   DIFFERENT answer. A case whose two branches agree is not testing the
   distinction it claims to.
3. **Internal consistency and non-triviality.** Recorded events must
   refer to frames and tracks that exist, the recorded counts must tally
   the recorded events, and the real-clip case must be large enough and
   busy enough to be worth replaying.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from trafficlens.analytics.speed import SpeedEstimator
from trafficlens.core.constants import (
    TRACK_HIGH_CONF,
    TRACK_LOW_CONF,
    TRACK_MATCH_IOU,
)
from trafficlens.core.geometry import side_of_line
from trafficlens.core.homography import RoadPlane

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "web" / "src" / "fixtures" / "parity.json"
REPORT = ROOT / "reports" / "parity.json"

#: Every boundary kind the fixture is required to carry. The first three
#: are the brief's mandate; the last two were added because this session
#: measured both live (see the task report).
REQUIRED_STRADDLES = (
    "anchorExactlyOnGate",
    "iouExactlyAtMatchThresh",
    "scoreExactlyAtHighThresh",
    "assignmentCostExactTie",
    "argmaxFloat32ClassTie",
    "deferredOnLineUsesLastOffLinePoint",
)


def _load_generator():
    """Import ``scripts/make_parity_fixtures.py`` by path: ``scripts/`` is
    not a package, so the generator is otherwise untestable."""
    path = ROOT / "scripts" / "make_parity_fixtures.py"
    spec = importlib.util.spec_from_file_location("_script_make_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fixture() -> dict:
    if not FIXTURE.is_file():
        pytest.fail(
            f"{FIXTURE.relative_to(ROOT)} is missing: regenerate it with "
            f"scripts/make_parity_fixtures.py"
        )
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.is_file():
        pytest.fail(
            f"{REPORT.relative_to(ROOT)} is missing: regenerate it with "
            f"scripts/make_parity_fixtures.py"
        )
    return json.loads(REPORT.read_text())


def _cases(fixture: dict) -> list[dict]:
    return [
        *fixture["trackerCases"],
        *fixture["gateCases"],
        *fixture["decodeCases"],
    ]


def _case(fixture: dict, name: str) -> dict:
    for case in _cases(fixture):
        if case["name"] == name:
            return case
    raise AssertionError(f"no case named {name!r} in the fixture")


def _iou(a: list[float], b: list[float]) -> float:
    """IoU of two xyxy boxes, spelled the way ``iou_matrix`` spells it."""
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# -- regenerability -----------------------------------------------------------


def test_the_committed_fixture_regenerates_byte_for_byte(tmp_path):
    """The fixture must be exactly what the Python engine produces today.

    This is the check that stops the fixture drifting into a record of
    what the engine USED to do -- which would leave the Vitest suite
    comparing the TypeScript engine against a stale answer and calling
    the agreement parity.
    """
    generator = _load_generator()
    generator.write_fixtures(tmp_path)
    regenerated = (tmp_path / "web" / "src" / "fixtures" / "parity.json").read_bytes()
    assert regenerated == FIXTURE.read_bytes(), (
        "web/src/fixtures/parity.json is not what the Python engine produces "
        "now: re-run scripts/make_parity_fixtures.py and review the diff"
    )


def test_the_committed_report_regenerates_byte_for_byte(tmp_path):
    generator = _load_generator()
    generator.write_fixtures(tmp_path)
    regenerated = (tmp_path / "reports" / "parity.json").read_bytes()
    assert regenerated == REPORT.read_bytes()


# -- the straddle inventory ---------------------------------------------------


def test_every_mandated_straddle_kind_appears_in_at_least_one_case(fixture):
    """The floor is non-empty on purpose: an inventory assertion that can
    be satisfied by finding nothing protects nothing."""
    seen: dict[str, list[str]] = {kind: [] for kind in REQUIRED_STRADDLES}
    for case in _cases(fixture):
        for kind in case["straddles"]:
            assert kind in seen, f"case {case['name']!r} declares unknown straddle {kind!r}"
            seen[kind].append(case["name"])
    for kind, names in seen.items():
        assert len(names) >= 1, f"no case in the fixture straddles {kind!r}"
    assert fixture["straddleKinds"] == list(REQUIRED_STRADDLES)


def test_the_on_gate_case_really_puts_an_anchor_exactly_on_the_gate_line(fixture):
    """Re-measure, do not trust the label: push every recorded position
    back through ``side_of_line`` and demand that at least one lands on
    the line, and that the case also has positions on BOTH sides (an
    anchor sitting on the line while the track never crosses would
    straddle nothing)."""
    for case in _cases(fixture):
        if "anchorExactlyOnGate" not in case["straddles"]:
            continue
        gate = case["gate"]
        a = tuple(gate["start"])
        b = tuple(gate["end"])
        sides = [side_of_line(a, b, tuple(step["curr"])) for step in case["steps"]]
        sides += [side_of_line(a, b, tuple(case["steps"][0]["prev"]))]
        assert 0 in sides, [case["name"], "no position lands on the gate line", sides]
        assert 1 in sides and -1 in sides, [
            case["name"],
            "the track never reaches both sides of the gate",
            sides,
        ]
        # Exactly on, not merely inside GEOMETRY_EPS: the cross product is
        # constructed to be identically zero, which is the only form of
        # "on the line" both languages are guaranteed to agree on.
        on_line = [
            step["curr"] for step in case["steps"]
            if side_of_line(a, b, tuple(step["curr"])) == 0
        ]
        for point in on_line:
            cross = (b[1] - a[1]) * (point[0] - a[0]) - (b[0] - a[0]) * (point[1] - a[1])
            assert cross == 0.0, [case["name"], "on-line only within eps", cross]


def test_the_deferred_on_line_case_would_answer_differently_from_prev(fixture):
    """The counterfactual: resolving the deferred crossing against the
    on-line ``prev`` instead of the stored last off-line point must give
    a DIFFERENT answer, or the case is not pinning the distinction it
    claims to."""
    case = _case(fixture, "gate_deferred_off_line_origin")
    assert "deferredOnLineUsesLastOffLinePoint" in case["straddles"]
    counter = case["counterfactualPrevOrigin"]
    assert len(case["expected"]["events"]) == 1, case["expected"]["events"]
    assert counter["events"] == [], (
        "using prev as the origin must NOT produce the event; if both "
        "branches agree this case tests nothing"
    )


def test_the_iou_straddle_case_sits_exactly_on_the_match_threshold(fixture):
    case = _case(fixture, "tracker_iou_at_match_thresh")
    assert "iouExactlyAtMatchThresh" in case["straddles"]
    measured = _iou(case["straddleTrackBox"], case["straddleDetectionBox"])
    assert measured == TRACK_MATCH_IOU, [
        "the constructed pair no longer sits exactly on the IoU floor",
        measured,
        TRACK_MATCH_IOU,
    ]
    # 1 - IoU must be exactly the ceiling assign() compares against, so
    # the pair is admitted by `<=` and barred by `<`.
    assert (1.0 - measured) == (1.0 - TRACK_MATCH_IOU)
    # And the Mahalanobis gate must not be what decides this pair: the
    # gate is the one comparison this project measured as language-
    # sensitive in its last bits, so the case is built to sit far from it.
    assert case["straddleGatingDistance"] < 0.5 * case["straddleGatingChi2"], [
        "the IoU straddle sits too close to the chi-square gate to be a "
        "clean IoU test",
        case["straddleGatingDistance"],
        case["straddleGatingChi2"],
    ]


def test_the_score_straddle_case_sits_exactly_on_the_confidence_thresholds(fixture):
    case = _case(fixture, "tracker_score_at_high_thresh")
    assert "scoreExactlyAtHighThresh" in case["straddles"]
    scores = {
        det["role"]: det["score"]
        for frame in case["frames"]
        for det in frame["detections"]
    }
    assert scores["at_high"] == TRACK_HIGH_CONF
    assert scores["below_high"] == math.nextafter(TRACK_HIGH_CONF, 0.0)
    assert scores["at_low"] == TRACK_LOW_CONF
    assert scores["below_low"] == math.nextafter(TRACK_LOW_CONF, 0.0)
    # The must-fail/must-survive pair: only the at-threshold detection
    # may become a track, because `score >= high_thresh` is inclusive.
    ids = {
        track["role"]
        for frame in case["expected"]["frames"]
        for track in frame["tracks"]
    }
    assert ids == {"at_high"}, [
        "exactly one of the four boundary detections may start a track",
        ids,
    ]


def test_the_argmax_tie_column_really_ties_two_kept_classes_in_float32(fixture):
    import numpy as np

    case = _case(fixture, "decode_argmax_class_tie")
    assert "argmaxFloat32ClassTie" in case["straddles"]
    columns = case["dims"][2]
    rows = case["dims"][1]
    raw = np.array(case["raw"], dtype=np.float32).reshape(rows, columns)
    column = case["tieColumn"]
    tied = case["tieClassIds"]
    assert len(tied) == 2, tied
    values = [raw[4 + class_id, column] for class_id in tied]
    assert values[0] == values[1], ["the tie is not exact in float32", values]
    assert float(values[0]) == float(np.float32(values[0]))
    others = [
        raw[4 + k, column] for k in range(rows - 4) if k not in tied
    ]
    assert max(others) < values[0], "the tied classes are not the column maximum"
    # numpy's argmax takes the FIRST maximum; the mirror scans with `>`.
    assert case["expectedDetections"][0]["classId"] == min(tied), [
        "the tie must resolve to the LOWER class id",
        case["expectedDetections"],
    ]


# -- internal consistency and non-triviality ----------------------------------


def test_the_real_clip_case_is_large_and_busy_enough_to_be_worth_replaying(fixture):
    case = _case(fixture, "tracker_real_clip_window")
    assert len(case["frames"]) >= 100, len(case["frames"])
    assert sum(len(f["detections"]) for f in case["frames"]) >= 500
    assert len(case["expected"]["events"]) >= 2, case["expected"]["events"]
    assert case["expected"]["tracksAllocated"] >= 20
    speeds = [
        track["speedKmh"]
        for frame in case["expected"]["frames"]
        for track in frame["tracks"]
    ]
    assert sum(1 for s in speeds if s is not None) >= 100, (
        "almost no speed was reported: the replay would not compare speeds "
        "at all"
    )
    assert any(s is None for s in speeds), (
        "no track was ever short of samples: the null-speed path is untested"
    )
    assert max(s for s in speeds if s is not None) > 10.0


def test_recorded_events_refer_to_frames_and_tracks_that_exist(fixture):
    for case in fixture["trackerCases"]:
        by_frame = {f["frameIndex"]: f for f in case["expected"]["frames"]}
        assert len(by_frame) == len(case["expected"]["frames"]), "duplicate frame index"
        assert list(by_frame) == sorted(by_frame), "frames are not in ascending order"
        assert set(by_frame) == {f["frameIndex"] for f in case["frames"]}
        for event in case["expected"]["events"]:
            frame = by_frame[event["frameIndex"]]
            live = {track["trackId"] for track in frame["tracks"]}
            assert event["trackId"] in live, [
                case["name"], "event fired for a track absent from its frame", event
            ]
            assert event["gate"] in {g["name"] for g in case["gates"]}
            gate = next(g for g in case["gates"] if g["name"] == event["gate"])
            assert event["direction"] in (gate["labelPositive"], gate["labelNegative"])
            assert event["signedDirection"] in (1, -1)


def test_recorded_counts_tally_the_recorded_events(fixture):
    for case in fixture["trackerCases"] + fixture["gateCases"]:
        tally: dict[str, dict[str, dict[str, int]]] = {}
        for event in case["expected"]["events"]:
            by_class = tally.setdefault(event["gate"], {})
            by_direction = by_class.setdefault(event["className"], {})
            by_direction[event["direction"]] = (
                by_direction.get(event["direction"], 0) + 1
            )
        assert case["expected"]["counts"] == tally, case["name"]


def test_no_track_is_counted_twice_at_one_gate(fixture):
    for case in fixture["trackerCases"] + fixture["gateCases"]:
        seen = set()
        for event in case["expected"]["events"]:
            key = (event["gate"], event["trackId"])
            assert key not in seen, [case["name"], "double count", event]
            seen.add(key)


def test_the_fixture_carries_a_fitted_plane_matrix_not_correspondences(fixture):
    """The browser has no cv2 and no SVD: it consumes an already-fitted
    3x3 matrix. A fixture carrying surveyed correspondences would be
    unusable on the other side."""
    plane = fixture["plane"]
    assert set(plane) == {"imageToWorld"}, plane.keys()
    matrix = plane["imageToWorld"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert matrix[2][2] == 1.0, "the matrix is not normalised"
    assert all(math.isfinite(v) for row in matrix for v in row)
    # It must be a usable plane, not a placeholder.
    world = RoadPlane(np.array(matrix), [], []).to_world((640.0, 600.0))
    assert all(math.isfinite(v) for v in world)


def test_the_report_agrees_with_the_fixture_it_describes(fixture, report):
    assert report["straddleKinds"] == list(REQUIRED_STRADDLES)
    assert report["caseCount"] == len(_cases(fixture))
    real = _case(fixture, "tracker_real_clip_window")
    assert report["realClip"]["frames"] == len(real["frames"])
    assert report["realClip"]["events"] == len(real["expected"]["events"])
    assert report["realClip"]["tracksAllocated"] == (
        real["expected"]["tracksAllocated"]
    )


# -- the timestamp-coercion hazard --------------------------------------------


def test_the_speed_estimator_coerces_integer_timestamps_to_float():
    """A fixture loaded from JSON can hand ``observe`` a Python ``int``.

    CPython's ``sum`` takes an UNCOMPENSATED path for ``int`` items, so a
    buffer holding one int timestamp fits a different slope from the same
    buffer holding the float. TypeScript has no int/float distinction and
    would always take the compensated path, so the two engines would
    disagree for a reason nothing in the failure would suggest. ``observe``
    therefore coerces.
    """
    identity = np.eye(3)
    plane = RoadPlane(identity, [], [])
    estimator = SpeedEstimator(plane, fps=30.0, min_samples=2)
    estimator.observe(1, (0.0, 0.0), 0)  # a bare int, as JSON `0` would give
    buffered = list(estimator._tracks[1])
    assert isinstance(buffered[0][0], float), (
        "observe buffered a raw int timestamp: sum() would take its "
        "uncompensated path for that item"
    )
