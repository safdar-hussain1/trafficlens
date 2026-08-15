"""Tests for the standard-failure-mode baselines
(``trafficlens.bench.baselines``).

These baselines exist to be beaten, so a test that merely showed one of
them working would prove nothing. Every failure test below is written as
a CONTRAST: the same input is put through the baseline and through the
engine, and the assertion is that the engine gets it right where the
baseline does not. A baseline that failed on input the engine also
failed on would measure nothing.

The reverse guard matters just as much and is tested too: each baseline
is asserted to get the EASY version of its own case right. A rule that
never counted anything would pass every "it misses the crossing"
assertion while being a straw man, so the miss tests are paired with a
well-sampled crossing the same rule counts correctly, and the tracker
fragmentation test is paired with an undipped run the same tracker holds
a single ID through.
"""

import pytest

from trafficlens.bench.baselines import (
    BandCounter,
    CentroidTracker,
    GreedyIoUTracker,
    PerFrameCounter,
)
from trafficlens.core.constants import (
    BASELINE_BAND_PX,
    BASELINE_CENTROID_MAX_DISTANCE_PX,
    BASELINE_GREEDY_IOU_THRESH,
    TRACK_HIGH_CONF,
    TRACK_MAX_AGE,
    TRACK_MIN_HITS,
)
from trafficlens.core.gate import CrossingEvent, Gate, GateCounter
from trafficlens.detect.base import Detection
from trafficlens.track.tracker import Track, Tracker

# A gate drawn left to right at a constant image y, exactly like both
# shipped motorway gates, so the direction labels below read the way a
# real deployment's do: side_of_line puts the side UP the frame (smaller
# y, further from the camera) at +1 = "away", and the side DOWN the frame
# at -1 = "toward".
GATE_Y = 300.0
GATE_X1, GATE_X2 = 0.0, 400.0
LANE_X = 200.0  # an anchor column comfortably inside the gate's x-span


def _gate() -> Gate:
    return Gate(
        "lane",
        (GATE_X1, GATE_Y),
        (GATE_X2, GATE_Y),
        label_positive="away",
        label_negative="toward",
    )


def _anchors(ys: list[float], x: float = LANE_X) -> list[tuple[float, float]]:
    """A one-track anchor stream at a fixed column, one point per frame."""
    return [(x, float(y)) for y in ys]


def _feed(rule, points, track_id: int = 1, class_name: str = "car") -> list:
    """Drive a counting rule over consecutive (prev, curr) anchor pairs,
    the way the pipeline drives ``GateCounter``. Returns the events."""
    events = []
    for i in range(1, len(points)):
        event = rule.update(
            track_id,
            class_name,
            points[i - 1],
            points[i],
            i,
            i / 30.0,
        )
        if event is not None:
            events.append(event)
    return events


def _det(
    x: float,
    y: float,
    score: float,
    cls: str = "car",
    w: float = 80.0,
    h: float = 80.0,
) -> Detection:
    """A Detection whose box has top-left (x, y) and size w x h."""
    class_ids = {"car": 2, "truck": 7}
    return Detection(
        x1=float(x),
        y1=float(y),
        x2=float(x + w),
        y2=float(y + h),
        score=float(score),
        class_id=class_ids[cls],
        class_name=cls,
    )


# -- construction contract ---------------------------------------------------


def test_baseline_defaults_come_from_constants():
    assert BandCounter(_gate()).band_px == BASELINE_BAND_PX
    assert PerFrameCounter(_gate()).band_px == BASELINE_BAND_PX

    centroid = CentroidTracker()
    assert centroid.max_distance_px == BASELINE_CENTROID_MAX_DISTANCE_PX
    # The lifecycle tunables are deliberately the ENGINE's own, so a
    # benchmark difference can never be blamed on a track living longer
    # or confirming sooner on one side than the other.
    assert centroid.max_age == TRACK_MAX_AGE
    assert centroid.min_hits == TRACK_MIN_HITS
    assert centroid.conf_thresh == TRACK_HIGH_CONF

    greedy = GreedyIoUTracker()
    assert greedy.iou_thresh == BASELINE_GREEDY_IOU_THRESH
    assert greedy.max_age == TRACK_MAX_AGE
    assert greedy.min_hits == TRACK_MIN_HITS
    assert greedy.conf_thresh == TRACK_HIGH_CONF


def test_counting_rules_expose_the_gate_counter_surface():
    # The harness swaps a rule in for GateCounter wholesale, so the whole
    # surface the pipeline touches has to be there, not just update().
    for rule in (BandCounter(_gate(), band_px=8.0), PerFrameCounter(_gate(), 8.0)):
        assert rule.update(1, "car", (LANE_X, 360.0), (LANE_X, 350.0), 1, 0.03) is None
        assert rule.total() == 0
        assert rule.totals == {}
        rule.forget(1)  # accepted for every rule, even one with no memory


# -- BandCounter: the miss mode ----------------------------------------------


def test_band_counter_misses_a_crossing_that_steps_over_the_band():
    # 40 px per frame past an 8 px band: the anchor is never once inside
    # the band, so the band rule has nothing to fire on. The engine reads
    # the SWEPT PATH between the two anchors instead of the anchor's
    # instantaneous position, so the step size cannot hide the crossing
    # from it.
    band = BandCounter(_gate(), band_px=8.0)
    engine = GateCounter(_gate())
    path = _anchors([360.0, 320.0, 280.0, 240.0])  # crosses between 320 and 280

    assert _feed(band, path) == []
    assert band.total() == 0

    engine_events = _feed(engine, path)
    assert len(engine_events) == 1
    assert engine_events[0].direction == "away"
    assert engine.total() == 1


def test_band_counter_counts_a_well_sampled_crossing_correctly():
    # The pair to the miss test: with the same rule, same band and same
    # gate, a crossing sampled finely enough to land in the band is
    # counted exactly once and in the same direction the engine gives it.
    # The band rule's weakness is the sampling rate, not the rule being
    # incapable of counting.
    band = BandCounter(_gate(), band_px=8.0)
    engine = GateCounter(_gate())
    path = _anchors([320.0, 310.0, 302.0, 298.0, 290.0])

    band_events = _feed(band, path)
    engine_events = _feed(engine, path)
    assert len(band_events) == 1
    assert len(engine_events) == 1
    assert band_events[0].direction == engine_events[0].direction == "away"
    assert band.total() == engine.total() == 1


# -- BandCounter: the phantom mode -------------------------------------------


def test_band_counter_fires_a_phantom_on_a_track_that_never_crosses():
    # A vehicle that drifts up to the gate, touches the band and falls
    # back -- a lane change along the line, or a box jittering on a slow
    # approach. Every anchor stays on the same side of the gate, so no
    # crossing happened. The band rule fires anyway, because entering the
    # band IS its trigger; the engine requires the swept path to actually
    # meet the gate segment, which it never does.
    band = BandCounter(_gate(), band_px=8.0)
    engine = GateCounter(_gate())
    path = _anchors([360.0, 340.0, 306.0, 340.0, 360.0])

    band_events = _feed(band, path)
    assert len(band_events) == 1
    assert band_events[0].direction == "away"  # a crossing that never happened
    assert band.total() == 1

    assert _feed(engine, path) == []
    assert engine.total() == 0


def test_band_counter_ignores_a_pass_beyond_the_gates_ends():
    # The obvious guard a real band implementation has: the band is
    # bounded by the drawn segment, so a vehicle on another carriageway
    # passing the gate's infinite line far outside its endpoints is not
    # counted. Leaving this out would have made the baseline a straw man
    # -- both rules agree here.
    band = BandCounter(_gate(), band_px=8.0)
    engine = GateCounter(_gate())
    path = _anchors([320.0, 310.0, 302.0, 298.0, 290.0], x=GATE_X2 + 100.0)

    assert _feed(band, path) == []
    assert _feed(engine, path) == []


def test_band_counter_counts_each_track_once_until_forgotten():
    band = BandCounter(_gate(), band_px=8.0)
    path = _anchors([320.0, 302.0, 298.0, 302.0, 298.0])  # loiters in the band
    assert len(_feed(band, path)) == 1

    band.forget(1)
    assert len(_feed(band, path)) == 1  # a recycled ID may count again


# -- PerFrameCounter: the dwell multiplier -----------------------------------


def test_per_frame_counter_multiplies_a_slow_vehicle_by_its_dwell():
    # A vehicle creeping through the gate at 2 px per frame. With a 10 px
    # band it is inside the band on the frames y = 290..310 -- 11 frames
    # -- and the per-frame rule counts every one of them, because it has
    # no track memory to notice they are all the same vehicle.
    band_px = 10.0
    ys = [288.0 + 2.0 * k for k in range(13)]  # 288 .. 312
    dwell_frames = sum(1 for y in ys if abs(GATE_Y - y) <= band_px)
    assert dwell_frames == 11  # pins the fixture, so the multiple below is exact

    per_frame = PerFrameCounter(_gate(), band_px=band_px)
    engine = GateCounter(_gate())
    path = _anchors(ys)

    per_frame_events = _feed(per_frame, path)
    engine_events = _feed(engine, path)

    assert len(engine_events) == 1
    assert engine.total() == 1
    assert len(per_frame_events) == dwell_frames
    assert per_frame.total() == dwell_frames * engine.total()
    # Every one of them is the same vehicle going the same way.
    assert {e.direction for e in per_frame_events} == {"toward"}
    assert engine_events[0].direction == "toward"


def test_per_frame_counter_counts_two_tracks_on_the_same_frame_separately():
    # It has no identity model at all, so it cannot dedupe within a frame
    # either -- each box on the gate is its own count.
    per_frame = PerFrameCounter(_gate(), band_px=10.0)
    prev, curr = (LANE_X, 310.0), (LANE_X, 296.0)
    assert per_frame.update(1, "car", prev, curr, 5, 0.16) is not None
    assert per_frame.update(2, "car", prev, curr, 5, 0.16) is not None
    # forget() is accepted but clears nothing: there is nothing to clear.
    per_frame.forget(1)
    assert per_frame.update(1, "car", prev, curr, 6, 0.2) is not None
    assert per_frame.total() == 3


def test_counting_rules_emit_engine_shaped_events():
    for rule in (BandCounter(_gate(), 10.0), PerFrameCounter(_gate(), 10.0)):
        # Descending the frame (y growing) is movement toward the camera,
        # which this gate's -1 side is named for.
        event = rule.update(
            7, "truck", (LANE_X, 290.0), (LANE_X, 304.0), 42, 1.4, 95.0, 80.0
        )
        assert isinstance(event, CrossingEvent)
        assert event.track_id == 7
        assert event.class_name == "truck"
        assert event.gate == "lane"
        assert event.direction == "toward"
        assert event.signed_direction == -1
        assert event.frame_index == 42
        assert event.timestamp == 1.4
        assert event.speed_kmh == 95.0
        assert event.is_violation is True
        assert rule.totals == {"truck": {"toward": 1}}


# -- CentroidTracker: the identity swap --------------------------------------


def _crossing_frames(t: int) -> list[Detection]:
    """The engine tracker's own crossing-paths fixture: two identical
    100x30 boxes on one row closing at 10 px/frame each, passing between
    t=9 and t=10 with a 10 px closest approach.

    Detection index 0 is the object moving RIGHT (cx = 100 + 10t) and
    index 1 the one moving LEFT (cx = 290 - 10t), so track 1 is born on
    the rightward object and track 2 on the leftward one.
    """
    w, h, cy = 100.0, 30.0, 300.0
    cx_a = 100.0 + 10.0 * t
    cx_b = 290.0 - 10.0 * t
    return [
        _det(cx_a - w / 2, cy - h / 2, 0.9, w=w, h=h),
        _det(cx_b - w / 2, cy - h / 2, 0.9, w=w, h=h),
    ]


def _final_centres(tracker, frames: int = 20) -> dict[int, float]:
    last: dict[int, float] = {}
    for t in range(frames):
        for track in tracker.update(_crossing_frames(t), t):
            last[track.track_id] = (track.box[0] + track.box[2]) / 2.0
    return last


def test_centroid_tracker_swaps_identities_where_the_engine_holds():
    # At t=10 the two detections sit at cx 200 (rightward) and 190
    # (leftward), while the last OBSERVED centroids from t=9 sit at 190
    # and 200. Nearest-centroid association therefore scores the SWAPPED
    # pairs at distance 0 and the correct pairs at distance 10, and takes
    # the swap. It has no velocity estimate to know that the rightward
    # object cannot suddenly be moving left.
    #
    # The engine survives the same frames twice over: its Kalman
    # prediction puts the correct pairs at IoU ~0.98, and the swapped
    # pairs fall outside the chi-square gate entirely.
    rightward_end = 100.0 + 10.0 * 19  # 290
    leftward_end = 290.0 - 10.0 * 19  # 100

    baseline = _final_centres(CentroidTracker(max_distance_px=60.0, min_hits=1))
    assert set(baseline) == {1, 2}
    # Track 1 was born on the RIGHTWARD object but ends up on the leftward
    # one's position, and vice versa: the identities are swapped.
    assert baseline[1] == pytest.approx(leftward_end, abs=1.0)
    assert baseline[2] == pytest.approx(rightward_end, abs=1.0)

    engine = _final_centres(Tracker(min_hits=1))
    assert set(engine) == {1, 2}
    assert engine[1] == pytest.approx(rightward_end, abs=4.0)
    assert engine[2] == pytest.approx(leftward_end, abs=4.0)


def test_centroid_tracker_keeps_identities_on_well_separated_paths():
    # The pair to the swap test: two objects that never come close stay
    # correctly associated, so the swap above is caused by the crossing,
    # not by the baseline being unable to track at all.
    tracker = CentroidTracker(max_distance_px=60.0, min_hits=1)
    last: dict[int, float] = {}
    for t in range(20):
        dets = [
            _det(100.0 + 10.0 * t, 100.0, 0.9, w=100.0, h=30.0),
            _det(100.0 + 10.0 * t, 500.0, 0.9, w=100.0, h=30.0),
        ]
        for track in tracker.update(dets, t):
            last[track.track_id] = track.box[1]
    assert last == {1: 100.0, 2: 500.0}


# -- GreedyIoUTracker: fragmentation through a confidence dip ----------------


def _dip_scores(frame: int, dip: bool) -> float:
    return 0.3 if dip and 5 <= frame <= 13 else 0.9


def _run_dip(tracker, dip: bool, frames: int = 20) -> list[int]:
    seen: list[int] = []
    for f in range(frames):
        for track in tracker.update(
            [_det(10.0 + 5.0 * f, 10.0, _dip_scores(f, dip))], f
        ):
            seen.append(track.track_id)
    return seen


def test_greedy_iou_tracker_fragments_through_a_confidence_dip():
    # Nine frames of low-confidence detections (score 0.3, below the 0.6
    # birth threshold both trackers use). The greedy baseline has one
    # association stage and therefore one confidence threshold: those
    # frames are simply dropped, the track coasts on a frozen box, and by
    # the time a high-confidence detection returns it has moved 50 px --
    # IoU 30/130 = 0.23, under the 0.3 floor. The track fragments into a
    # second ID, which downstream is a second counted vehicle.
    #
    # The engine's second stage matches those same low-confidence boxes
    # against already-confirmed tracks, so its ID never breaks.
    baseline_ids = _run_dip(GreedyIoUTracker(min_hits=1), dip=True)
    assert len(set(baseline_ids)) == 2, baseline_ids

    engine_ids = _run_dip(Tracker(min_hits=1), dip=True)
    assert set(engine_ids) == {1}, engine_ids


def test_greedy_iou_tracker_holds_one_id_without_the_dip():
    # The pair to the fragmentation test: the same motion at full
    # confidence keeps one ID, so the fragmentation above is the missing
    # low-confidence stage and not the matcher failing on ordinary motion.
    assert set(_run_dip(GreedyIoUTracker(min_hits=1), dip=False)) == {1}


# -- baseline trackers: the Tracker.update return convention -----------------


@pytest.mark.parametrize(
    "make_tracker",
    [
        lambda: CentroidTracker(),
        lambda: GreedyIoUTracker(),
    ],
    ids=["centroid", "greedy_iou"],
)
def test_baseline_trackers_return_only_confirmed_current_frame_tracks(make_tracker):
    tracker = make_tracker()
    # Tentative while it has fewer than min_hits: internal, not returned.
    assert tracker.update([_det(10.0, 10.0, 0.9)], 0) == []
    assert tracker.update([_det(15.0, 10.0, 0.9)], 1) == []
    tracks = tracker.update([_det(20.0, 10.0, 0.9)], 2)
    assert [t.track_id for t in tracks] == [1]
    assert isinstance(tracks[0], Track)
    assert tracks[0].state == "confirmed"
    assert tracks[0].time_since_update == 0
    assert tracks[0].hits == 3
    assert len(tracks[0].history) == 3

    # A coasting confirmed track is NOT returned while it coasts, but it
    # is still alive: the same ID comes back with the detection.
    assert tracker.update([], 3) == []
    reappeared = tracker.update([_det(25.0, 10.0, 0.9)], 4)
    assert [t.track_id for t in reappeared] == [1]
    assert reappeared[0].time_since_update == 0
    # History grows only on detection-backed frames: 3 + the return frame.
    assert len(reappeared[0].history) == 4

    # A low-confidence detection can never start a track.
    tracker.reset()
    for f in range(5):
        assert tracker.update([_det(10.0, 10.0, 0.3)], f) == []
    for f in range(5, 8):
        tracker.update([_det(10.0, 10.0, 0.9)], f)
    assert [t.track_id for t in tracker.update([_det(10.0, 10.0, 0.9)], 8)] == [1]


@pytest.mark.parametrize(
    "make_tracker",
    [
        lambda: CentroidTracker(min_hits=2),
        lambda: GreedyIoUTracker(min_hits=2),
    ],
    ids=["centroid", "greedy_iou"],
)
def test_baseline_trackers_are_deterministic(make_tracker):
    def scenario(f: int) -> list[Detection]:
        dets = _crossing_frames(f)
        dets.append(_det(10.0 + 4.0 * f, 600.0, _dip_scores(f, dip=True), w=60.0, h=60.0))
        if f < 10 or f >= 15:  # a dropout that returns inside max_age
            dets.append(_det(700.0, 10.0 + 6.0 * f, 0.9, w=70.0, h=70.0))
        dets.append(_det(50.0 + 10.0 * f, 350.0, 0.7, cls="truck", w=90.0, h=40.0))
        return dets

    runs = []
    for _ in range(2):
        tracker = make_tracker()
        runs.append(
            [
                [
                    (t.track_id, t.class_name, t.box, t.age, t.hits, tuple(t.history))
                    for t in tracker.update(scenario(f), f)
                ]
                for f in range(30)
            ]
        )
    assert runs[0] == runs[1]


def test_baseline_trackers_never_associate_across_classes():
    for tracker in (CentroidTracker(min_hits=1), GreedyIoUTracker(min_hits=1)):
        box = (100.0, 100.0, 0.9)
        assert [t.track_id for t in tracker.update([_det(*box, cls="car")], 0)] == [1]
        # A perfectly-overlapping truck box: same centroid, IoU 1.0. The
        # cross-class bar -- shared with the engine so it is never the
        # source of a benchmark difference -- refuses it.
        tracks = tracker.update([_det(*box, cls="truck")], 1)
        assert [(t.track_id, t.class_name) for t in tracks] == [(2, "truck")]


# -- composition: {tracker} x {counting rule} --------------------------------


def test_any_tracker_composes_with_any_counting_rule():
    # The harness pairs each tracker with each rule so a benchmark error
    # can be attributed to one or the other. This asserts the wiring:
    # every combination consumes the same detection stream through the
    # same Track.anchor property, and the engine pairing gets the fixture
    # exactly right.
    def frames(f: int) -> list[Detection]:
        # Two cars, one per lane column, descending 4 px/frame through the
        # gate: bottom anchor from y=340 down to y=260.
        return [
            _det(x - 30.0, 340.0 - 4.0 * f - 80.0, 0.9, w=60.0, h=80.0)
            for x in (120.0, 280.0)
        ]

    counts = {}
    for tracker_name, make_tracker in (
        ("engine", lambda: Tracker(min_hits=1)),
        ("centroid", lambda: CentroidTracker(min_hits=1)),
        ("greedy_iou", lambda: GreedyIoUTracker(min_hits=1)),
    ):
        for rule_name, make_rule in (
            ("engine", lambda: GateCounter(_gate())),
            ("band", lambda: BandCounter(_gate(), band_px=10.0)),
            ("per_frame", lambda: PerFrameCounter(_gate(), band_px=10.0)),
        ):
            tracker, rule = make_tracker(), make_rule()
            previous: dict[int, tuple[float, float]] = {}
            for f in range(21):
                for track in tracker.update(frames(f), f):
                    anchor = track.anchor
                    prev = previous.get(track.track_id)
                    if prev is not None:
                        rule.update(track.track_id, track.class_name, prev, anchor, f, f / 30.0)
                    previous[track.track_id] = anchor
            counts[(tracker_name, rule_name)] = rule.total()

    # Two vehicles crossed. Only the engine pairing is asserted correct;
    # the rest are recorded so a regression that broke the composition
    # (an exception, or a rule that stopped counting entirely) is caught.
    assert counts[("engine", "engine")] == 2
    assert all(count > 0 for count in counts.values()), counts
