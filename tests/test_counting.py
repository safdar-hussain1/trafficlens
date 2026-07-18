"""Gate counting vs the naive band baseline.

The baseline tests encode the two failure modes that motivated the
segment-intersection design; both must PASS for the baseline (i.e. the
baseline must demonstrably fail to count correctly) so the comparison
in the benchmark report stays honest.
"""

from trafficlens.baseline import NaiveBandCounter
from trafficlens.counting import Gate, GateCounter


def make_counter() -> GateCounter:
    return GateCounter(Gate("main", (100.0, 300.0), (600.0, 300.0),
                            label_positive="south", label_negative="north"))


def feed(counter: GateCounter, track_id: int, cls: str, path: list[tuple[float, float]]):
    """Run a track's anchor path through the counter, returning events."""
    events = []
    prev = None
    for i, p in enumerate(path):
        e = counter.update(track_id, cls, prev, p, frame_index=i, timestamp=i / 30.0)
        if e:
            events.append(e)
        prev = p
    return events


class TestGateCounter:
    def test_slow_crossing_counted_once(self):
        counter = make_counter()
        path = [(350, 290 + 5 * i) for i in range(5)]  # 5 px/frame through the gate
        events = feed(counter, 1, "car", path)
        assert len(events) == 1
        assert events[0].direction == "south"
        assert counter.total == 1

    def test_fast_crossing_counted(self):
        """200 px/frame — jumps any band, still counted by segment test."""
        counter = make_counter()
        events = feed(counter, 2, "car", [(350, 150), (350, 350), (350, 550)])
        assert len(events) == 1
        assert counter.total == 1

    def test_direction_labels(self):
        counter = make_counter()
        down = feed(counter, 3, "car", [(300, 250), (300, 350)])
        up = feed(counter, 4, "truck", [(400, 350), (400, 250)])
        assert down[0].direction == "south"
        assert up[0].direction == "north"
        assert counter.total_by_direction() == {"south": 1, "north": 1}

    def test_graze_without_crossing_not_counted(self):
        """Approaches the line, hovers 3 px above it, retreats: no count."""
        counter = make_counter()
        events = feed(counter, 5, "car", [(350, 250), (350, 297), (352, 297), (350, 250)])
        assert events == []
        assert counter.total == 0

    def test_lingering_on_gate_counts_once(self):
        counter = make_counter()
        path = [(350, 295), (350, 299), (350, 301), (350, 299), (350, 301), (350, 400)]
        events = feed(counter, 6, "car", path)
        assert len(events) == 1

    def test_crossing_outside_gate_extent_not_counted(self):
        counter = make_counter()
        events = feed(counter, 7, "car", [(700, 250), (700, 350)])
        assert events == []

    def test_per_class_tallies(self):
        counter = make_counter()
        feed(counter, 8, "car", [(300, 250), (300, 350)])
        feed(counter, 9, "car", [(320, 250), (320, 350)])
        feed(counter, 10, "bus", [(340, 250), (340, 350)])
        assert counter.totals["car"]["south"] == 2
        assert counter.totals["bus"]["south"] == 1

    def test_anchor_landing_exactly_on_line_counts_next_frame(self):
        counter = make_counter()
        events = feed(counter, 11, "car", [(350, 290), (350, 300), (350, 310)])
        assert len(events) == 1

    def test_speed_limit_flags_violation(self):
        counter = make_counter()
        e = counter.update(12, "car", (300, 250), (300, 350), 0, 0.0,
                           speed=92.0, speed_limit=80.0)
        assert e is not None and e.is_violation

    def test_reset_tracks_keeps_totals_but_allows_recount(self):
        """A looping file restarts tracker IDs; totals must keep climbing."""
        counter = make_counter()
        feed(counter, 1, "car", [(300, 250), (300, 350)])
        assert counter.total == 1
        counter.reset_tracks()  # stream rewound; ID 1 will be a new vehicle
        events = feed(counter, 1, "car", [(300, 250), (300, 350)])
        assert len(events) == 1
        assert counter.total == 2

    def test_forget_allows_id_reuse(self):
        counter = make_counter()
        feed(counter, 13, "car", [(300, 250), (300, 350)])
        counter.forget(13)
        events = feed(counter, 13, "car", [(300, 250), (300, 350)])
        assert len(events) == 1
        assert counter.total == 2


class TestNaiveBaselineFailureModes:
    """The band approach must fail exactly where the geometry says it does."""

    def make_band(self) -> NaiveBandCounter:
        # The classic tutorial band: line_y-15 < y < line_y+1
        return NaiveBandCounter(x_min=100, x_max=600, line_y=300)

    def test_band_misses_fast_mover(self):
        band = self.make_band()
        gate = make_counter()
        path = [(350.0, 150.0), (350.0, 350.0)]  # 200 px jump over the band
        assert not any(band.update(1, p) for p in path)          # baseline misses it
        assert len(feed(gate, 1, "car", path)) == 1              # gate counts it

    def test_band_counts_phantom_graze(self):
        band = self.make_band()
        gate = make_counter()
        path = [(350.0, 250.0), (350.0, 297.0), (350.0, 250.0)]  # in-band, never crosses
        assert any(band.update(2, p) for p in path)              # baseline counts a phantom
        assert feed(gate, 2, "car", path) == []                  # gate correctly ignores it

    def test_band_and_gate_agree_on_slow_clean_crossing(self):
        band = self.make_band()
        gate = make_counter()
        path = [(350.0, 270.0 + 5 * i) for i in range(15)]
        assert any(band.update(3, p) for p in path)
        assert len(feed(gate, 3, "car", path)) == 1
