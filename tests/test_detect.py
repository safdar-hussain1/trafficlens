"""Tests for trafficlens.detect: shared letterbox/NMS/decode preprocessing
(trafficlens.detect.base, torch-free) and the two detector adapters
(trafficlens.detect.ultralytics_yolo, trafficlens.detect.onnx_yolo).

Heavy-dependency tests (anything that loads a real checkpoint or runs a
real inference session) use ``pytest.importorskip`` for the optional
package and ``pytest.skip`` when the git-ignored weight files this repo
keeps on disk (``yolo11n.pt`` / ``yolo11s.pt``) are absent, so a clean
clone without the ``detect``/``onnx`` extras still collects and passes
every test in this file except those.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trafficlens.core.classes import COCO_CLASSES, VEHICLE_CLASSES, class_ids
from trafficlens.core.constants import LETTERBOX_PAD_VALUE
from trafficlens.detect.base import Detection, decode_yolo, letterbox, nms

ROOT = Path(__file__).resolve().parents[1]
YOLO11N = ROOT / "yolo11n.pt"


def _box_iou(a, b) -> float:
    """Plain-python IoU of two (x1, y1, x2, y2) boxes, independent of the
    nms() implementation under test -- used only to confirm the fixtures
    below actually have the overlap they claim to have."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --- letterbox: shape, dtype, value range -----------------------------------

def test_letterbox_returns_expected_shape_dtype_and_range():
    frame = np.random.randint(0, 256, (480, 854, 3), dtype=np.uint8)
    chw, scale, pad_x, pad_y = letterbox(frame, 480)
    assert chw.shape == (1, 3, 480, 480)
    assert chw.dtype == np.float32
    assert chw.min() >= 0.0 and chw.max() <= 1.0


def test_letterbox_wide_frame_pads_y_not_x():
    # 854x480 is wider than tall: the width drives the scale, so the
    # resized image fills the full 480px width and only the vertical axis
    # gets padding.
    frame = np.zeros((480, 854, 3), np.uint8)
    _, scale, pad_x, pad_y = letterbox(frame, 480)
    assert scale == pytest.approx(480 / 854)
    assert pad_x == 0.0
    assert pad_y > 0.0


def test_letterbox_tall_frame_pads_x_not_y():
    # The mirror image of the above: a taller-than-wide frame gets its
    # horizontal axis padded instead.
    frame = np.zeros((854, 480, 3), np.uint8)
    _, scale, pad_x, pad_y = letterbox(frame, 480)
    assert scale == pytest.approx(480 / 854)
    assert pad_y == 0.0
    assert pad_x > 0.0


def test_letterbox_scale_and_pad_match_documented_formula():
    h, w, size = 720, 1280, 640
    frame = np.zeros((h, w, 3), np.uint8)
    _, scale, pad_x, pad_y = letterbox(frame, size)
    expected_scale = min(size / w, size / h)
    new_w = int(round(w * expected_scale))
    new_h = int(round(h * expected_scale))
    assert scale == pytest.approx(expected_scale)
    assert pad_x == (size - new_w) // 2
    assert pad_y == (size - new_h) // 2


def test_letterbox_pads_with_the_constant_grey_value():
    frame = np.zeros((480, 854, 3), np.uint8)
    chw, _, _, pad_y = letterbox(frame, 480)
    # A row well inside the top pad band (pad_y is where real content
    # starts) must equal LETTERBOX_PAD_VALUE / 255 in every channel.
    assert pad_y > 1
    top_pad_row = chw[0, :, 0, :]
    assert np.allclose(top_pad_row, LETTERBOX_PAD_VALUE / 255.0)


def test_letterbox_converts_bgr_to_rgb():
    # cv2-style BGR frame: a pure-blue pixel (channel order B, G, R).
    frame = np.zeros((480, 480, 3), np.uint8)
    frame[..., 0] = 200  # B
    frame[..., 1] = 50   # G
    frame[..., 2] = 10   # R
    chw, _, _, _ = letterbox(frame, 480)
    # chw is (1, 3, H, W) in R, G, B order.
    assert chw[0, 0].max() == pytest.approx(10 / 255.0)
    assert chw[0, 1].max() == pytest.approx(50 / 255.0)
    assert chw[0, 2].max() == pytest.approx(200 / 255.0)


def test_letterbox_coordinates_invert_exactly():
    frame = np.zeros((480, 854, 3), np.uint8)
    _, scale, px, py = letterbox(frame, 480)
    x_model, y_model = 100.0, 200.0
    x_orig = (x_model - px) / scale
    y_orig = (y_model - py) / scale
    assert 0 <= x_orig <= 854 and 0 <= y_orig <= 480
    assert abs((x_orig * scale + px) - x_model) < 1e-9


# --- nms ---------------------------------------------------------------------

def test_nms_suppresses_a_near_duplicate_box():
    # Two same-size squares offset along x only by d: overlap width is
    # (100-d), full overlap in y, so IoU simplifies to (100-d)/(100+d).
    # Solved for IoU == 0.9 exactly: a near-perfect duplicate.
    d = 100 * (1 - 0.9) / (1 + 0.9)
    box_a = (0.0, 0.0, 100.0, 100.0)
    box_b = (d, 0.0, 100.0 + d, 100.0)
    assert _box_iou(box_a, box_b) == pytest.approx(0.9, abs=1e-6)

    boxes = np.array([box_a, box_b])
    scores = np.array([0.95, 0.80])
    kept = nms(boxes, scores, iou=0.5)
    assert kept == [0]


def test_nms_keeps_boxes_with_low_overlap():
    # Same construction (x-only offset), solved for IoU == 0.3: distinct
    # enough to both survive the same 0.5 threshold that collapsed the
    # 0.9-overlap pair above.
    d = 100 * (1 - 0.3) / (1 + 0.3)
    box_a = (200.0, 200.0, 300.0, 300.0)
    box_b = (200.0 + d, 200.0, 300.0 + d, 300.0)
    assert _box_iou(box_a, box_b) == pytest.approx(0.3, abs=1e-6)

    boxes = np.array([box_a, box_b])
    scores = np.array([0.9, 0.6])
    kept = nms(boxes, scores, iou=0.5)
    assert kept == [0, 1]


def test_nms_tie_break_is_deterministic_lower_index_wins():
    # Two identical, fully-overlapping boxes with an EXACTLY equal score:
    # the documented tie-break rule (stable sort on score, ties resolved
    # by ascending original index) must always keep index 0, never depend
    # on array order or hashing.
    boxes = np.array([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]])
    scores = np.array([0.5, 0.5])
    for _ in range(5):
        assert nms(boxes, scores, iou=0.5) == [0]


def test_nms_empty_input_returns_empty_list():
    boxes = np.zeros((0, 4))
    scores = np.zeros((0,))
    assert nms(boxes, scores, iou=0.5) == []


def test_nms_no_overlap_keeps_every_box():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
    scores = np.array([0.6, 0.9])
    # Higher score (index 1) should still appear, order reflects score-desc.
    assert sorted(nms(boxes, scores, iou=0.5)) == [0, 1]


# --- decode_yolo ---------------------------------------------------------------

def _build_raw_output(entries, n_classes=80):
    """entries: list of (cx, cy, w, h, class_id, score) -> (1, 4+n_classes, N)."""
    n = len(entries)
    out = np.zeros((1, 4 + n_classes, n), dtype=np.float32)
    for col, (cx, cy, w, h, class_id, score) in enumerate(entries):
        out[0, 0, col] = cx
        out[0, 1, col] = cy
        out[0, 2, col] = w
        out[0, 3, col] = h
        out[0, 4 + class_id, col] = score
    return out


def test_decode_yolo_hand_built_array_thresholds_and_converts_coordinates():
    # scale=2.0, pad_x=10, pad_y=20: nontrivial letterbox inversion.
    output = _build_raw_output(
        [
            (100.0, 100.0, 20.0, 20.0, 2, 0.9),   # car, passes conf + class filter
            (105.0, 105.0, 20.0, 20.0, 2, 0.1),   # car, below conf threshold
            (300.0, 300.0, 40.0, 40.0, 7, 0.8),   # truck, not in keep_class_ids
        ]
    )
    car_id = COCO_CLASSES.index("car")
    dets = decode_yolo(
        output, scale=2.0, pad_x=10.0, pad_y=20.0,
        conf=0.5, iou=0.45, keep_class_ids={car_id},
    )
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == car_id
    assert d.class_name == "car"
    assert d.score == pytest.approx(0.9)
    # model-space box: x1=90,y1=90,x2=110,y2=110 -> undo pad then scale.
    assert d.x1 == pytest.approx((90.0 - 10.0) / 2.0)
    assert d.y1 == pytest.approx((90.0 - 20.0) / 2.0)
    assert d.x2 == pytest.approx((110.0 - 10.0) / 2.0)
    assert d.y2 == pytest.approx((110.0 - 20.0) / 2.0)


def test_decode_yolo_returns_empty_list_when_nothing_passes():
    output = _build_raw_output([(50.0, 50.0, 10.0, 10.0, 2, 0.05)])
    dets = decode_yolo(
        output, scale=1.0, pad_x=0.0, pad_y=0.0,
        conf=0.25, iou=0.45, keep_class_ids={2},
    )
    assert dets == []


def test_decode_yolo_nms_is_class_wise():
    # Two heavily-overlapping boxes of DIFFERENT classes must both survive
    # (class-wise NMS never suppresses across classes); two heavily
    # overlapping boxes of the SAME class must collapse to one.
    car_id, bus_id = COCO_CLASSES.index("car"), COCO_CLASSES.index("bus")
    output = _build_raw_output(
        [
            (100.0, 100.0, 40.0, 40.0, car_id, 0.9),
            (102.0, 102.0, 40.0, 40.0, bus_id, 0.85),  # near-duplicate box, different class
            (102.0, 102.0, 40.0, 40.0, car_id, 0.6),   # near-duplicate box, SAME class as first
        ]
    )
    dets = decode_yolo(
        output, scale=1.0, pad_x=0.0, pad_y=0.0,
        conf=0.5, iou=0.5, keep_class_ids={car_id, bus_id},
    )
    class_names = sorted(d.class_name for d in dets)
    assert class_names == ["bus", "car"]
    # the lower-score duplicate car box must have been suppressed
    car_scores = [d.score for d in dets if d.class_name == "car"]
    assert car_scores == [pytest.approx(0.9)]


# --- core.classes: COCO_CLASSES / VEHICLE_CLASSES / class_ids -----------------

def test_coco_classes_has_80_unique_names():
    assert len(COCO_CLASSES) == 80
    assert len(set(COCO_CLASSES)) == 80


def test_vehicle_classes_are_all_valid_coco_names():
    assert set(VEHICLE_CLASSES) <= set(COCO_CLASSES)
    assert len(VEHICLE_CLASSES) > 0


def test_class_ids_resolves_known_names():
    ids = class_ids(["car", "bus"])
    assert ids == {COCO_CLASSES.index("car"), COCO_CLASSES.index("bus")}


def test_class_ids_raises_value_error_naming_the_offender():
    with pytest.raises(ValueError) as exc_info:
        class_ids(["car", "trucks"])  # typo: "trucks" is not a COCO class
    message = str(exc_info.value)
    assert "trucks" in message
    # and it must list valid options, so the typo is discoverable.
    assert "truck" in message


def test_coco_classes_matches_ultralytics_names_mapping():
    pytest.importorskip("ultralytics")
    if not YOLO11N.exists():
        pytest.skip("yolo11n.pt not present on disk")
    from ultralytics import YOLO

    yolo = YOLO(str(YOLO11N))
    names = yolo.names  # dict[int, str], the checkpoint's own class mapping
    expected = tuple(names[i] for i in range(len(names)))
    assert COCO_CLASSES == expected


# --- Detection dataclass -------------------------------------------------------

def test_detection_is_a_frozen_dataclass():
    d = Detection(x1=1.0, y1=2.0, x2=3.0, y2=4.0, score=0.5, class_id=2, class_name="car")
    assert d.x1 == 1.0 and d.class_name == "car"
    with pytest.raises(Exception):
        d.x1 = 99.0  # frozen: mutation must fail


# --- dependency layering: no torch/ultralytics/onnxruntime at import time -----

def _run_isolated_import(module: str) -> list[str]:
    code = (
        "import sys\n"
        f"import {module}\n"
        "leaked = sorted(m for m in ('torch', 'ultralytics', 'onnxruntime') "
        "if m in sys.modules)\n"
        "print(','.join(leaked))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    stdout = result.stdout.strip()
    return stdout.split(",") if stdout else []


def test_importing_detect_base_does_not_import_torch_ultralytics_or_onnxruntime():
    assert _run_isolated_import("trafficlens.detect.base") == []


def test_importing_detect_package_does_not_import_torch_ultralytics_or_onnxruntime():
    assert _run_isolated_import("trafficlens.detect") == []


def test_importing_ultralytics_adapter_module_does_not_import_torch_at_top_level():
    # Importing the MODULE (not instantiating the detector) must not pull
    # in torch/ultralytics: the import belongs inside __init__, per the
    # dependency-layering rule.
    assert _run_isolated_import("trafficlens.detect.ultralytics_yolo") == []


def test_importing_onnx_adapter_module_does_not_import_onnxruntime_at_top_level():
    assert _run_isolated_import("trafficlens.detect.onnx_yolo") == []


# --- real-weight adapter tests (skipped without optional extras/weights) ------

def test_ultralytics_detector_finds_vehicles_in_a_sample_frame():
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    if not YOLO11N.exists():
        pytest.skip("yolo11n.pt not present on disk")
    import cv2

    from trafficlens.detect.ultralytics_yolo import UltralyticsDetector

    video_path = ROOT / "data" / "samples" / "motorway-a40.webm"
    if not video_path.exists():
        pytest.skip("sample clip not present on disk")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ok, frame = cap.read()
    cap.release()
    assert ok

    detector = UltralyticsDetector(str(YOLO11N))
    detections = detector.detect(frame)
    assert len(detections) > 0
    # decode_yolo deliberately does not clip boxes to the frame -- a
    # vehicle genuinely at the image edge can have a raw model box that
    # extends a few pixels past it -- so bounds are checked loosely
    # (a generous margin), not as an exact frame-edge clip.
    margin = 25.0
    for d in detections:
        assert isinstance(d, Detection)
        assert d.class_name in VEHICLE_CLASSES
        assert d.x1 < d.x2 and d.y1 < d.y2
        assert -margin <= d.x1 and d.x2 <= frame.shape[1] + margin
        assert -margin <= d.y1 and d.y2 <= frame.shape[0] + margin


def test_ultralytics_and_onnx_adapters_agree_on_a_real_frame():
    """The route taken for the shared-decode requirement: both adapters
    obtain a raw (1, 84, N) tensor from their own backend and hand it to
    the exact same trafficlens.detect.base.decode_yolo. This test is the
    evidence that doing so makes them agree, on a real frame, with real
    weights -- not just structurally identical code paths."""
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    ort = pytest.importorskip("onnxruntime")
    if not YOLO11N.exists():
        pytest.skip("yolo11n.pt not present on disk")
    import shutil
    import tempfile

    import cv2

    from trafficlens.detect.onnx_yolo import OnnxDetector
    from trafficlens.detect.ultralytics_yolo import UltralyticsDetector

    video_path = ROOT / "data" / "samples" / "motorway-a40.webm"
    if not video_path.exists():
        pytest.skip("sample clip not present on disk")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ok, frame = cap.read()
    cap.release()
    assert ok

    ultra = UltralyticsDetector(str(YOLO11N))
    ultra_dets = ultra.detect(frame)

    # Export to ONNX with the source .pt itself inside an isolated tmp dir,
    # so ultralytics -- which always writes the .onnx next to the weights
    # file it's given -- writes directly into the tmp dir and the repo
    # root is never touched at all (not "touched then cleaned up": never
    # written to in the first place). The stray-cleanup in `finally` below
    # is kept purely as insurance against an unrelated ultralytics quirk,
    # not because this method is expected to touch the root.
    tmpdir = tempfile.mkdtemp(prefix="trafficlens_onnx_test_")
    try:
        from ultralytics import YOLO

        weights_copy = Path(tmpdir) / "yolo11n.pt"
        shutil.copy2(YOLO11N, weights_copy)

        yolo = YOLO(str(weights_copy))
        onnx_path = Path(
            yolo.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False, verbose=False)
        )
        assert onnx_path.parent == Path(tmpdir), (
            f"expected the ONNX export to land in {tmpdir}, got {onnx_path}"
        )

        onnx_detector = OnnxDetector(str(onnx_path))
        onnx_dets = onnx_detector.detect(frame)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        # Insurance only, per the comment above -- not expected to trigger.
        stray = ROOT / "yolo11n.onnx"
        if stray.exists():
            stray.unlink()

    assert len(ultra_dets) > 0
    assert len(onnx_dets) > 0
    # Equal counts in each direction rules out spurious extra detections
    # on either side that a one-directional "every X has a Y" check would
    # miss.
    assert len(ultra_dets) == len(onnx_dets)

    # Every ultralytics-adapter detection should have a matching
    # onnx-adapter detection of the same class, within a tight IoU AND a
    # tight score tolerance -- the numerical noise between a torch forward
    # pass and its ONNX export (observed empirically at <=0.01 max abs
    # diff on the raw (1, 84, 8400) tensor), not a structural difference.
    # Checked in both directions (ultra->onnx and onnx->ultra) so neither
    # side can have an unmatched, spurious extra detection.
    score_tol = 0.05

    def _assert_all_matched(dets_a, dets_b, label):
        for a in dets_a:
            candidates = [d for d in dets_b if d.class_name == a.class_name]
            assert candidates, f"no {label} detection of class {a.class_name!r} to match {a}"
            best = max(
                candidates,
                key=lambda c: _box_iou((a.x1, a.y1, a.x2, a.y2), (c.x1, c.y1, c.x2, c.y2)),
            )
            best_iou = _box_iou((a.x1, a.y1, a.x2, a.y2), (best.x1, best.y1, best.x2, best.y2))
            assert best_iou > 0.9, f"{a} has no close {label} match (best IoU {best_iou:.3f})"
            assert abs(best.score - a.score) < score_tol, (
                f"{a} vs closest {label} match {best}: score diff "
                f"{abs(best.score - a.score):.4f} exceeds tolerance {score_tol}"
            )

    _assert_all_matched(ultra_dets, onnx_dets, "onnx")
    _assert_all_matched(onnx_dets, ultra_dets, "ultralytics")
