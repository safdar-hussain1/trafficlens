"""The slit-scan ground-truth tooling, tested as a TOOL.

Nothing here asserts anything about real footage or about any real
crossing: the labels are a human's job, produced under
``data/groundtruth/PROTOCOL.md``. What is tested is that the instrument
the labeller reads is trustworthy -- that N vehicles physically crossing
a gate at known frames produce exactly N separable blobs at those rows,
that the same inputs always produce byte-identical images, and that the
label loader refuses every malformed label set it is given rather than
letting a bad one through into an accuracy number.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from trafficlens.bench.slitscan import (
    GroundTruth,
    GroundTruthError,
    build_slitscan,
    gate_strip,
    tile_windows,
)
from trafficlens.core.gate import Gate
from trafficlens.io.video import SourceError, VideoSource

WIDTH, HEIGHT = 640, 480
FPS = 30.0

# The same normalized geometry as the shipped inbound motorway gate, so
# the tooling is exercised on the gate shape it will actually be used on.
GATE_START = (0.06, 0.80)
GATE_END = (0.46, 0.80)


def make_gate(width: int = WIDTH, height: int = HEIGHT) -> Gate:
    return Gate.from_normalized(
        "inbound",
        GATE_START,
        GATE_END,
        width,
        height,
        label_positive="away",
        label_negative="toward",
    )


GATE_Y = GATE_START[1] * HEIGHT           # 384.0
GATE_X0 = GATE_START[0] * WIDTH           # 38.4
GATE_X1 = GATE_END[0] * WIDTH             # 294.4

BACKGROUND = 40
SQUARE = 20          # square side, px
SPEED = 6.0          # px per frame, straight down the frame

# Five vehicles crossing the gate, plus one decoy that crosses the gate
# line's infinite extension well past the gate's right-hand endpoint.
CROSSINGS = [(10, 60), (30, 110), (50, 160), (70, 210), (90, 260)]
DECOY = (110, 520)   # x = 520 px is beyond GATE_X1 = 294.4
CLIP_FRAMES = 130


def write_crossing_clip(path: Path) -> Path:
    """A synthetic clip: bright squares falling straight down the frame,
    each passing the gate line at a known frame.

    Square k's centre is exactly on the gate line at its crossing frame,
    so the blob it paints in the slit-scan is centred on that row.
    """
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
    )
    assert writer.isOpened(), "OpenCV could not open an MJPG writer"
    half = SQUARE // 2
    for frame_index in range(CLIP_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
        for cross_frame, x in [*CROSSINGS, DECOY]:
            cy = GATE_Y + (frame_index - cross_frame) * SPEED
            top, bottom = int(round(cy)) - half, int(round(cy)) + half
            left, right = x - half, x + half
            if bottom < 0 or top >= HEIGHT:
                continue
            frame[max(top, 0):bottom, left:right] = 255
        writer.write(frame)
    writer.release()
    return path


def blob_rows(scan: np.ndarray, threshold: int = 160) -> list[float]:
    """Centroid row of every run of bright rows in a slit-scan."""
    brightness = scan.reshape(scan.shape[0], -1).max(axis=1)
    bright = brightness > threshold
    runs, start = [], None
    for row, is_bright in enumerate(bright):
        if is_bright and start is None:
            start = row
        elif not is_bright and start is not None:
            runs.append((start, row))
            start = None
    if start is not None:
        runs.append((start, len(bright)))
    return [(a + b - 1) / 2.0 for a, b in runs]


# --- gate_strip -------------------------------------------------------------


def test_gate_strip_samples_only_the_band_along_the_gate():
    """A bright bar drawn exactly on the gate segment lights the strip;
    the same bar drawn past the gate's endpoint does not."""
    gate = make_gate()
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[int(GATE_Y) - 2:int(GATE_Y) + 3, 100:200] = 255

    strip = gate_strip(frame, gate, thickness_px=3, samples=256)

    assert strip.shape == (256, 3)
    # Sample i sits at x = GATE_X0 + i/255 * (GATE_X1 - GATE_X0).
    def sample_at(x: float) -> int:
        return int(round((x - GATE_X0) / (GATE_X1 - GATE_X0) * 255))

    assert strip[sample_at(150)].max() > 200      # inside the bright bar
    assert strip[sample_at(60)].max() < 40        # gate, but left of the bar
    assert strip[sample_at(280)].max() < 40       # gate, but right of the bar


def test_gate_strip_ignores_pixels_beyond_the_gate_endpoints():
    """The gate is a bounded segment: a bar drawn on its infinite
    extension, past the endpoint, must not appear in the strip at all."""
    gate = make_gate()
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[int(GATE_Y) - 2:int(GATE_Y) + 3, 400:600] = 255   # past GATE_X1

    strip = gate_strip(frame, gate, thickness_px=3, samples=256)

    assert strip.max() < 40


def test_gate_strip_averages_across_the_perpendicular_offsets():
    """thickness_px pixels perpendicular to the gate are averaged, so a
    bar covering half the band reads about half brightness."""
    gate = make_gate()
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    # thickness 4 straddles the line: offsets -1.5..1.5 round to rows
    # 383, 384, 385, 386 under half-away-from-zero rounding of
    # GATE_Y + offset. Light exactly two of them.
    frame[385:387, :] = 200

    strip = gate_strip(frame, gate, thickness_px=4, samples=64)

    assert np.all(strip == 100)


def test_gate_strip_clamps_offsets_that_leave_the_frame():
    """A gate on the bottom edge still yields a strip: offsets past the
    edge repeat the EDGE pixel -- not the opposite edge (a wrap) and not
    an interior row (a reflect).

    The frame's bottom row is given its own value, so the three
    behaviours give three different answers and only clamping passes.
    """
    gate = Gate.from_normalized("edge", (0.1, 1.0), (0.9, 1.0), WIDTH, HEIGHT)
    frame = np.full((HEIGHT, WIDTH, 3), 77, dtype=np.uint8)
    frame[HEIGHT - 1, :] = 200          # the distinctive bottom row

    # thickness 1: the single sample sits at y = 480, one past the last
    # row, so clamping reads the bottom row itself. A wrap would read row
    # 0 (77) and a reflect row 478 (77).
    strip = gate_strip(frame, gate, thickness_px=1, samples=32)
    assert strip.shape == (32, 3)
    assert np.all(strip == 200)

    # thickness 9: offsets -4..+4 land on rows 476..484. Clamped, that is
    # rows 476, 477, 478 (77 each) and row 479 six times (200 each), so
    # the average is (3*77 + 6*200) / 9 = 159 exactly. A wrap would read
    # rows 476-479 then 0-4, averaging (8*77 + 200) / 9 -> 91.
    strip = gate_strip(frame, gate, thickness_px=9, samples=32)
    assert strip.shape == (32, 3)
    assert np.all(strip == 159)


def test_gate_strip_matches_a_golden_strip_on_a_known_ramp():
    """The sampling rule, pinned to literal values on a tiny frame.

    ``frame[y, x] = x + y``, so every sampled pixel's value states both
    coordinates it was read at. Between them the two assertions fix:

    - the along-gate parameterisation ``t = i / (samples - 1)``, with
      BOTH endpoints included -- an endpoint-exclusive ``t = i / samples``
      would sample x = 0, 2.4, 4.8, 7.2, 9.6 and never reach x = 12;
    - half-up rounding of the perpendicular average -- the thickness-2
      averages are all exactly ``x + 4.5``, which half-up sends up to
      ``x + 5`` and numpy's half-to-even ``np.round`` would send down to
      ``x + 4`` wherever ``x + 4`` is even.
    """
    frame = np.fromfunction(
        lambda y, x: x + y, (8, 13), dtype=np.int64
    ).astype(np.uint8)
    gate = Gate("ramp", (0.0, 4.0), (12.0, 4.0))     # pixel coordinates

    # thickness 1: the gate row itself, sampled at x = 0, 3, 6, 9, 12.
    strip = gate_strip(frame, gate, thickness_px=1, samples=5)
    assert strip.tolist() == [4, 7, 10, 13, 16]

    # thickness 2 straddles the line: rows 4 and 5 at the same five x,
    # averaging (x + 4 + x + 5) / 2 = x + 4.5, rounded half UP.
    strip = gate_strip(frame, gate, thickness_px=2, samples=5)
    assert strip.tolist() == [5, 8, 11, 14, 17]


def test_gate_strip_handles_a_single_channel_frame():
    gate = make_gate()
    frame = np.full((HEIGHT, WIDTH), 77, dtype=np.uint8)

    strip = gate_strip(frame, gate, thickness_px=3, samples=32)

    assert strip.shape == (32,)
    assert np.all(strip == 77)


def test_gate_strip_rejects_a_degenerate_sample_count():
    gate = make_gate()
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="samples"):
        gate_strip(frame, gate, thickness_px=3, samples=1)


def test_gate_strip_rejects_a_zero_thickness():
    gate = make_gate()
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="thickness"):
        gate_strip(frame, gate, thickness_px=0, samples=32)


# --- build_slitscan ---------------------------------------------------------


def test_build_slitscan_paints_one_blob_per_crossing_at_the_right_row(tmp_path):
    """The whole point of the instrument: N vehicles crossing the gate at
    known frames produce exactly N separable blobs, each centred on its
    own crossing frame."""
    clip = write_crossing_clip(tmp_path / "crossings.avi")
    gate = make_gate()

    with VideoSource.open(str(clip)) as source:
        scan = build_slitscan(
            source, gate, range(CLIP_FRAMES), thickness_px=3, samples=256
        )

    assert scan.shape == (CLIP_FRAMES, 256, 3)
    rows = blob_rows(scan)
    assert len(rows) == len(CROSSINGS), rows
    # EXACT, not approximate. This is the only test that validates the
    # claim every hand-read crossing frame rests on -- that a blob's row
    # IS the frame index. A tolerance here would let a whole-image row
    # shift through, and a shifted slit-scan misdates every label read
    # off it while still looking perfectly plausible.
    for measured, (expected, _x) in zip(rows, CROSSINGS):
        assert measured == expected, (measured, expected)


def test_build_slitscan_ignores_traffic_past_the_gate_endpoints(tmp_path):
    """The decoy square crosses the gate line's extension, far past its
    right endpoint, and must paint nothing."""
    clip = write_crossing_clip(tmp_path / "crossings.avi")
    gate = make_gate()

    with VideoSource.open(str(clip)) as source:
        scan = build_slitscan(
            source, gate, range(CLIP_FRAMES), thickness_px=3, samples=256
        )

    decoy_frame = DECOY[0]
    assert not any(abs(row - decoy_frame) <= 3 for row in blob_rows(scan))


def test_build_slitscan_reads_only_the_requested_window(tmp_path):
    clip = write_crossing_clip(tmp_path / "crossings.avi")
    gate = make_gate()

    with VideoSource.open(str(clip)) as source:
        scan = build_slitscan(
            source, gate, range(40, 60), thickness_px=3, samples=128
        )

    assert scan.shape == (20, 128, 3)
    # Row r of this scan is frame 40 + r, so the crossing at frame 50
    # lands on row 10.
    rows = blob_rows(scan)
    assert len(rows) == 1
    assert rows[0] == 10


def test_build_slitscan_is_deterministic(tmp_path):
    """Same clip, same gate, same parameters -> byte-identical array."""
    clip = write_crossing_clip(tmp_path / "crossings.avi")
    gate = make_gate()

    scans = []
    for _ in range(2):
        with VideoSource.open(str(clip)) as source:
            scans.append(
                build_slitscan(
                    source, gate, range(CLIP_FRAMES), thickness_px=5, samples=200
                )
            )

    assert scans[0].dtype == scans[1].dtype
    assert scans[0].shape == scans[1].shape
    assert scans[0].tobytes() == scans[1].tobytes()


def test_build_slitscan_consumes_the_single_pass_source_exactly_once(tmp_path):
    """VideoSource hands out one iterator, ever. build_slitscan must take
    that one and not ask for another."""
    clip = write_crossing_clip(tmp_path / "crossings.avi")
    gate = make_gate()

    with VideoSource.open(str(clip)) as source:
        build_slitscan(source, gate, range(20), thickness_px=3, samples=64)
        with pytest.raises(SourceError, match="single-pass"):
            iter(source)


def test_build_slitscan_raises_when_the_clip_ends_before_the_window(tmp_path):
    clip = write_crossing_clip(tmp_path / "crossings.avi")
    gate = make_gate()

    with VideoSource.open(str(clip)) as source:
        with pytest.raises(ValueError, match="ended"):
            build_slitscan(
                source, gate, range(CLIP_FRAMES + 50), thickness_px=3, samples=64
            )


def test_build_slitscan_rejects_an_empty_window(tmp_path):
    gate = make_gate()
    with pytest.raises(ValueError, match="at least one frame"):
        build_slitscan([], gate, range(0), thickness_px=3, samples=64)


def test_build_slitscan_rejects_frames_that_go_backwards():
    gate = make_gate()
    with pytest.raises(ValueError, match="increasing"):
        build_slitscan([], gate, [5, 4, 3], thickness_px=3, samples=64)


def test_build_slitscan_rejects_a_negative_frame_index():
    gate = make_gate()
    with pytest.raises(ValueError, match="negative"):
        build_slitscan([], gate, [-1, 0], thickness_px=3, samples=64)


# --- tile_windows -----------------------------------------------------------


def test_tile_windows_cover_the_whole_range_with_the_asked_for_overlap():
    windows = tile_windows(0, 735, tile_frames=120, overlap=20)

    assert windows[0] == (0, 119)
    assert windows[-1][1] == 735
    for (a_start, a_end), (b_start, _b_end) in zip(windows, windows[1:]):
        assert b_start == a_end + 1 - 20      # 20 frames of overlap
    covered = set()
    for start, end in windows:
        covered.update(range(start, end + 1))
    assert covered == set(range(0, 736))


def test_tile_windows_returns_one_tile_when_the_range_is_short():
    assert tile_windows(10, 40, tile_frames=120, overlap=20) == [(10, 40)]


def test_tile_windows_rejects_an_overlap_that_never_advances():
    with pytest.raises(ValueError, match="overlap"):
        tile_windows(0, 500, tile_frames=120, overlap=120)


def test_tile_windows_rejects_a_backwards_range():
    with pytest.raises(ValueError, match="end_frame"):
        tile_windows(500, 100, tile_frames=120, overlap=20)


# --- GroundTruth loader -----------------------------------------------------


LABELLED_FPS = 30.0
LABELLED_FRAMES = 120


@pytest.fixture
def labelled_clip(tmp_path) -> Path:
    """A real, decodable clip for the loader to check `clip` and `fps`
    against -- the loader must never take those on trust."""
    path = tmp_path / "motorway-a40.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), LABELLED_FPS, (WIDTH, HEIGHT)
    )
    assert writer.isOpened()
    blank = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    for _ in range(LABELLED_FRAMES):
        writer.write(blank)
    writer.release()
    return path


def label_document(**overrides) -> dict:
    """A well-formed label document. Every rejection test starts from
    this and breaks exactly one thing."""
    document = {
        "schema": 1,
        "clip": "motorway-a40.avi",
        "fps": LABELLED_FPS,
        "window": {"start_frame": 0, "end_frame": 100},
        "gate": {"name": "inbound", "start": list(GATE_START), "end": list(GATE_END)},
        "protocol": "data/groundtruth/PROTOCOL.md",
        "labeller": "test fixture",
        "labelled_on": "2026-08-15",
        "crossings": [
            {"id": 1, "frame": 12, "class": "car", "direction": "toward",
             "confidence": "certain"},
            {"id": 2, "frame": 40, "class": "truck", "direction": "toward",
             "confidence": "probable"},
            {"id": 3, "frame": 40, "class": "car", "direction": "away",
             "confidence": "certain"},
        ],
    }
    document.update(overrides)
    return document


def load(tmp_path, clip: Path, document: dict) -> GroundTruth:
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(document))
    return GroundTruth.load(path, gate=make_gate(), clip_path=clip)


def test_groundtruth_loads_a_well_formed_label_set(tmp_path, labelled_clip):
    truth = load(tmp_path, labelled_clip, label_document())

    assert truth.clip == "motorway-a40.avi"
    assert truth.fps == LABELLED_FPS
    assert (truth.start_frame, truth.end_frame) == (0, 100)
    assert truth.gate_name == "inbound"
    assert len(truth.crossings) == 3
    assert truth.crossings[0].id == 1
    assert truth.crossings[0].frame == 12
    assert truth.crossings[0].class_name == "car"
    assert truth.crossings[0].direction == "toward"
    assert truth.crossings[0].confidence == "certain"
    # Two vehicles abreast may legitimately cross on the same frame.
    assert truth.crossings[1].frame == truth.crossings[2].frame
    assert truth.certain_count == 2
    assert len(truth.frames_by_direction("toward")) == 2


def test_groundtruth_accepts_a_window_with_no_crossings(tmp_path, labelled_clip):
    truth = load(tmp_path, labelled_clip, label_document(crossings=[]))
    assert truth.crossings == ()


def test_groundtruth_rejects_a_crossing_after_the_window(tmp_path, labelled_clip):
    document = label_document()
    document["crossings"][-1]["frame"] = 101       # window ends at 100
    with pytest.raises(GroundTruthError, match="window"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_a_crossing_before_the_window(tmp_path, labelled_clip):
    document = label_document(window={"start_frame": 20, "end_frame": 100})
    with pytest.raises(GroundTruthError, match="window"):
        load(tmp_path, labelled_clip, document)   # first crossing is frame 12


def test_groundtruth_rejects_a_duplicate_id(tmp_path, labelled_clip):
    document = label_document()
    document["crossings"][1]["id"] = 1
    with pytest.raises(GroundTruthError, match="duplicate"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_an_unknown_class(tmp_path, labelled_clip):
    document = label_document()
    document["crossings"][0]["class"] = "lorry"
    with pytest.raises(GroundTruthError, match="class"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_a_direction_the_gate_does_not_use(
    tmp_path, labelled_clip
):
    document = label_document()
    document["crossings"][0]["direction"] = "in"   # the Gate default, not this gate's
    with pytest.raises(GroundTruthError, match="direction"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_an_unknown_confidence(tmp_path, labelled_clip):
    document = label_document()
    document["crossings"][0]["confidence"] = "pretty sure"
    with pytest.raises(GroundTruthError, match="confidence"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_frames_that_go_backwards(tmp_path, labelled_clip):
    document = label_document()
    document["crossings"][2]["frame"] = 39        # after a frame-40 crossing
    with pytest.raises(GroundTruthError, match="order"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_a_clip_name_mismatch(tmp_path, labelled_clip):
    with pytest.raises(GroundTruthError, match="clip"):
        load(tmp_path, labelled_clip, label_document(clip="some-other-clip.webm"))


def test_groundtruth_rejects_an_fps_mismatch(tmp_path, labelled_clip):
    with pytest.raises(GroundTruthError, match="fps"):
        load(tmp_path, labelled_clip, label_document(fps=25.0))


def test_groundtruth_rejects_a_window_past_the_end_of_the_clip(
    tmp_path, labelled_clip
):
    document = label_document(window={"start_frame": 0, "end_frame": 9999})
    with pytest.raises(GroundTruthError, match="clip"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_accepts_a_window_ending_on_the_last_decodable_frame(
    tmp_path, labelled_clip
):
    document = label_document(
        window={"start_frame": 0, "end_frame": LABELLED_FRAMES - 1}
    )
    truth = load(tmp_path, labelled_clip, document)
    assert truth.end_frame == LABELLED_FRAMES - 1


def test_groundtruth_rejects_a_window_the_decoder_never_reaches(
    tmp_path, labelled_clip, monkeypatch
):
    """The container's advertised frame count is an UPPER BOUND, not a
    guarantee: the shipped motorway clip advertises 737 frames and
    decodes 735. A window checked only against the header would admit a
    frame no decoder produces, which is exactly what PROTOCOL.md forbids.

    The advertised count is faked high here so that a header-only check
    would pass; the loader must still refuse, because it decodes.
    """
    monkeypatch.setattr(VideoSource, "frame_count", property(lambda self: 9999))
    document = label_document(
        window={"start_frame": 0, "end_frame": LABELLED_FRAMES + 10}
    )
    with pytest.raises(GroundTruthError, match="decodes only"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_a_backwards_window(tmp_path, labelled_clip):
    document = label_document(window={"start_frame": 90, "end_frame": 10})
    with pytest.raises(GroundTruthError, match="window"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_a_gate_name_mismatch(tmp_path, labelled_clip):
    document = label_document()
    document["gate"]["name"] = "outbound"
    with pytest.raises(GroundTruthError, match="gate"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_gate_geometry_that_moved(tmp_path, labelled_clip):
    document = label_document()
    document["gate"]["end"] = [0.50, 0.80]
    with pytest.raises(GroundTruthError, match="gate"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_an_unknown_schema_version(tmp_path, labelled_clip):
    with pytest.raises(GroundTruthError, match="schema"):
        load(tmp_path, labelled_clip, label_document(schema=2))


def test_groundtruth_rejects_an_unknown_top_level_field(tmp_path, labelled_clip):
    with pytest.raises(GroundTruthError, match="unknown"):
        load(tmp_path, labelled_clip, label_document(counted_by="the detector"))


def test_groundtruth_rejects_an_unknown_crossing_field(tmp_path, labelled_clip):
    document = label_document()
    document["crossings"][0]["track_id"] = 7
    with pytest.raises(GroundTruthError, match="unknown"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_a_missing_field(tmp_path, labelled_clip):
    document = label_document()
    del document["labeller"]
    with pytest.raises(GroundTruthError, match="labeller"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_a_missing_crossing_field(tmp_path, labelled_clip):
    document = label_document()
    del document["crossings"][0]["confidence"]
    with pytest.raises(GroundTruthError, match="confidence"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_a_non_integer_frame(tmp_path, labelled_clip):
    document = label_document()
    document["crossings"][0]["frame"] = 12.5
    with pytest.raises(GroundTruthError, match="frame"):
        load(tmp_path, labelled_clip, document)


def test_groundtruth_rejects_a_file_that_is_not_json(tmp_path, labelled_clip):
    path = tmp_path / "gt.json"
    path.write_text("{not json")
    with pytest.raises(GroundTruthError, match="JSON"):
        GroundTruth.load(path, gate=make_gate(), clip_path=labelled_clip)


def test_groundtruth_rejects_a_missing_file(tmp_path, labelled_clip):
    with pytest.raises(GroundTruthError, match="not found"):
        GroundTruth.load(
            tmp_path / "absent.json", gate=make_gate(), clip_path=labelled_clip
        )


def test_groundtruth_error_names_the_label_file(tmp_path, labelled_clip):
    """Every rejection must say which file to fix."""
    document = label_document()
    document["crossings"][0]["class"] = "lorry"
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(document))
    with pytest.raises(GroundTruthError) as caught:
        GroundTruth.load(path, gate=make_gate(), clip_path=labelled_clip)
    assert "gt.json" in str(caught.value)


# --- the labelling protocol is a committed artefact -------------------------

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_document_forbids_detector_derived_labels():
    """The protocol's load-bearing sentence, asserted verbatim.

    Substring checks for words like "detector" cannot fail for any
    document that happens to contain those letters; the rule that makes
    the ground truth independent of the system under test is a specific
    sentence and is asserted as one.
    """
    protocol = ROOT / "data" / "groundtruth" / "PROTOCOL.md"
    assert protocol.is_file(), "the labelling protocol must be committed"
    text = " ".join(protocol.read_text().split())

    assert (
        "Labels are produced from slit-scan review plus full-frame "
        "confirmation. They are never produced, seeded, corrected or "
        '"sanity checked" against the detector, the tracker, or '
        "`GateCounter` output."
    ) in text


def test_protocol_document_fixes_the_scoring_tolerance():
    """A scorer needs a fixed frame window to match a prediction to a
    label, and that number must be settled before any scoring code
    exists -- for the same reason the labelling rules are settled before
    any labelling."""
    protocol = ROOT / "data" / "groundtruth" / "PROTOCOL.md"
    text = " ".join(protocol.read_text().split())

    assert (
        "A prediction matches a label when it names the same gate and its "
        "frame is within **2 frames** of the label's frame."
    ) in text


def test_protocol_document_states_the_anchor_tie_break_without_hedging():
    """The tie-break for a contact point sitting on the gate must match
    the engine's inclusive bounds, not be left to the labeller."""
    protocol = ROOT / "data" / "groundtruth" / "PROTOCOL.md"
    text = " ".join(protocol.read_text().split())

    assert (
        "The crossing frame is the **first** frame at which the contact "
        "point is on or beyond the gate segment."
    ) in text
    assert "unambiguously past" not in text


def test_generated_review_images_are_never_tracked():
    """private/ holds the generated review PNGs and must stay ignored."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", "private/gt/slitscan.png"], cwd=ROOT
    )
    assert result.returncode == 0, "private/gt/ is NOT git-ignored"
