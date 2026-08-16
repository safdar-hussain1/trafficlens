#!/usr/bin/env python3
"""Generate the cross-language fixtures the browser runtime is tested against.

Every byte written here comes from the PYTHON engine -- ``letterbox`` and
``decode_yolo`` in ``trafficlens.detect.base``, and, for the raw-output
fixture, the exported ONNX graph run through onnxruntime. That direction is
the whole point: a fixture produced by the TypeScript side would only prove
that the TypeScript side agrees with itself.

Writes into ``web/src/runtime/fixtures/``:

``letterbox_src_<case>.bin``   the source frame as the BROWSER sees it -- RGB,
                              row-major, uint8, no alpha. Python holds frames
                              in BGR and ``letterbox`` converts at step 7, so
                              handing the TypeScript side BGR bytes would make
                              it re-implement a conversion it never performs on
                              a real ``<video>`` frame.
``letterbox_want_<case>.bin`` the expected ``(1, 3, size, size)`` float32
                              tensor, little-endian, exactly as ``letterbox``
                              returns it.
``decode_<case>_raw.bin``     a ``(1, 84, N)`` float32 model output.
``manifest.json``             shapes, the scale/pad triples, the decoded
                              detections, and the structural facts each decode
                              case is asserted to exercise.

Usage:

    PYTHONPATH=src .venv/bin/python scripts/make_runtime_fixtures.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trafficlens.core.classes import VEHICLE_CLASSES, class_ids  # noqa: E402
from trafficlens.detect.base import decode_yolo, letterbox  # noqa: E402

OUT = ROOT / "web" / "src" / "runtime" / "fixtures"

#: Frame shapes chosen so the two letterbox hazards are both on the critical
#: path rather than adjacent to it.
#:
#: ``bankers`` -- 69 * 0.5 == 34.5 exactly, so step 3's round-half-to-even
#: gives 34 where ``Math.round`` would give 35. A mirror that reaches for
#: ``Math.round`` produces a 35-row image, a different pad, and every box in
#: the frame shifted; this case is the one that catches it.
#:
#: ``oddpad`` -- 64 - 35 == 29 is ODD, so step 5's floor division puts the
#: extra pad row at the BOTTOM (pad_y == 14, rows 49..63 grey) rather than
#: splitting it. A mirror that rounds the pad up, or centres by halving in
#: float, lands one row out.
#:
#: Both also sit at a downscale ratio between 1x and 3x on both axes, which is
#: where this project measured plain INTER_LINEAR being intercepted by the
#: platform's own resize HAL -- see the letterbox docstring. They are
#: therefore the shapes most likely to expose an interpolation that is not the
#: bit-exact one, which is exactly what they are here to pin.
LETTERBOX_CASES = {
    "bankers": (128, 69, 64),
    "oddpad": (100, 55, 64),
}

#: The shipped geometry: 1280x720 video into the 480px graph. Too large to
#: commit as a tensor (2.7 MB), so the manifest carries its scale/pad triple
#: only -- enough to pin the geometry at the size that actually ships.
SHIPPED_FRAME = (1280, 720, 480)

CONF = 0.35
IOU = 0.5
N_CLASSES = 80


def synthetic_frame(w: int, h: int) -> np.ndarray:
    """A deterministic BGR frame with both smooth and high-frequency content.

    The two ramps are what a "synthetic gradient" usually means, and they
    would pass under almost any interpolation. The third channel is a
    sawtooth that wraps every 256 steps, so it carries hard edges at
    non-integer resize positions -- an interpolator that is close but not
    identical shows up there and nowhere else.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    blue = (xx * 255 // max(w - 1, 1)).astype(np.uint8)
    green = (yy * 255 // max(h - 1, 1)).astype(np.uint8)
    red = ((xx * 7 + yy * 13) % 256).astype(np.uint8)
    return np.stack([blue, green, red], axis=2)


def write_letterbox_cases(manifest: dict) -> None:
    cases = {}
    for name, (w, h, size) in LETTERBOX_CASES.items():
        frame = synthetic_frame(w, h)
        tensor, scale, pad_x, pad_y = letterbox(frame, size)
        assert tensor.shape == (1, 3, size, size), tensor.shape
        assert tensor.dtype == np.float32, tensor.dtype

        # What a canvas hands JavaScript: RGB, no alpha.
        rgb = frame[:, :, ::-1].copy()
        (OUT / f"letterbox_src_{name}.bin").write_bytes(rgb.tobytes())
        (OUT / f"letterbox_want_{name}.bin").write_bytes(tensor.tobytes())

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        cases[name] = {
            "width": w,
            "height": h,
            "size": size,
            "scale": scale,
            "padX": pad_x,
            "padY": pad_y,
            "resizedWidth": new_w,
            "resizedHeight": new_h,
            # Recorded so the test can state which hazard each case covers and
            # fail loudly if a later edit makes a case stop covering it.
            "padXIsOdd": (size - new_w) % 2 == 1,
            "padYIsOdd": (size - new_h) % 2 == 1,
            "hasHalfwayRounding": (w * scale) % 1 == 0.5 or (h * scale) % 1 == 0.5,
        }
        print(f"letterbox {name}: {w}x{h} -> {size} "
              f"resized {new_w}x{new_h} pad ({pad_x}, {pad_y})")

    w, h, size = SHIPPED_FRAME
    _, scale, pad_x, pad_y = letterbox(synthetic_frame(w, h), size)
    manifest["letterbox"] = {
        "cases": cases,
        "shipped": {
            "width": w, "height": h, "size": size,
            "scale": scale, "padX": pad_x, "padY": pad_y,
        },
    }


#: A shape whose interpolation weights land exactly on a half-step, so the
#: coefficient rounding inside the resize is exercised rather than merely
#: present. `getCoeffs` quantises each weight with cvRound -- half-to-EVEN --
#: and at 3 -> 256 the weights include 2.5, 8.5, 14.5 ... where half-to-even
#: gives 2, 8, 14 and half-UP would give 3, 9, 15.
#:
#: Measured: over every (src, dst) pair in [2, 300] and every output index,
#: 32920 coefficients land on a half-step, and the letterbox cases above
#: happen to contain none of them -- which is exactly why this case exists
#: separately. Without it, swapping that one cvRound for Math.round changes
#: nothing any test can see.
RESIZE_HALFSTEP = {"srcWidth": 3, "srcHeight": 1, "dstWidth": 256, "dstHeight": 1}


def write_resize_case(manifest: dict) -> None:
    import cv2

    spec = RESIZE_HALFSTEP
    rng = np.random.default_rng(20)
    src = rng.integers(0, 256, size=(spec["srcHeight"], spec["srcWidth"], 3), dtype=np.uint8)
    want = cv2.resize(
        src,
        (spec["dstWidth"], spec["dstHeight"]),
        interpolation=cv2.INTER_LINEAR_EXACT,
    )
    if want.ndim == 2:
        want = want[:, :, None]
    (OUT / "resize_halfstep_src.bin").write_bytes(src.tobytes())
    (OUT / "resize_halfstep_want.bin").write_bytes(np.ascontiguousarray(want).tobytes())
    manifest["resize"] = {"halfstep": {**spec, "channels": 3}}
    print(f"resize halfstep: {spec['srcWidth']}x{spec['srcHeight']} -> "
          f"{spec['dstWidth']}x{spec['dstHeight']}")


def _put(raw: np.ndarray, col: int, box, class_id: int, score: float) -> None:
    """Write one prediction column: cx, cy, w, h then the class scores."""
    raw[0, 0:4, col] = np.array(box, dtype=np.float32)
    raw[0, 4 + class_id, col] = np.float32(score)


def build_boundary() -> tuple[np.ndarray, dict]:
    """A hand-built output whose columns sit ON the decode decision boundaries.

    A real model output is a poor probe for these: it never lands a score
    exactly on the threshold, never ties two scores exactly, and never places
    two boxes at exactly the suppression IoU. Every column below is placed
    deliberately, and ``write_decode_cases`` asserts each one had the effect it
    was built for -- so this cannot quietly stop testing anything.
    """
    ids = sorted(class_ids(VEHICLE_CLASSES))
    bicycle, car, motorcycle, bus, truck = 1, 2, 3, 5, 7
    assert {bicycle, car, motorcycle, bus, truck} <= set(ids), ids
    not_kept = 0  # person: a real class the product deliberately ignores

    n = 18
    raw = np.zeros((1, 4 + N_CLASSES, n), dtype=np.float32)
    notes = {}

    # 0/1: same class, heavy overlap -> the lower score must be suppressed.
    _put(raw, 0, (100, 100, 40, 40), car, 0.90)
    _put(raw, 1, (104, 100, 40, 40), car, 0.80)
    notes["suppressed_within_class"] = [0, 1]

    # 2/3: identical geometry, DIFFERENT classes -> class-wise NMS keeps both.
    # A single global NMS pass would drop one of these and nothing else in the
    # fixture would notice.
    _put(raw, 2, (300, 100, 50, 50), car, 0.70)
    _put(raw, 3, (300, 100, 50, 50), truck, 0.70)
    notes["cross_class_pair"] = [2, 3]

    # 4: score EXACTLY at conf -> kept, because the threshold is inclusive.
    # float32(0.35) is 0.3499999940395355, which is < 0.35 as a float64. A
    # mirror that compares in float64 drops this column.
    _put(raw, 4, (500, 100, 30, 30), bus, CONF)
    notes["exactly_at_conf"] = 4

    # 5: one float32 ULP below conf -> dropped.
    below = np.nextafter(np.float32(CONF), np.float32(0.0))
    _put(raw, 5, (560, 100, 30, 30), bus, float(below))
    notes["just_below_conf"] = 5

    # 6/7: identical scores, heavy overlap -> the stable sort must keep the
    # LOWER column index. An unstable sort passes or fails at random here.
    _put(raw, 6, (100, 300, 40, 40), motorcycle, 0.60)
    _put(raw, 7, (103, 300, 40, 40), motorcycle, 0.60)
    notes["score_tie"] = [6, 7]

    # 8/9: a class outside keep_class_ids, scoring higher than anything else.
    # If the class filter is dropped, these lead the output and the whole
    # ordering changes.
    _put(raw, 8, (700, 300, 40, 40), not_kept, 0.99)
    _put(raw, 9, (700, 300, 60, 60), not_kept, 0.95)
    notes["excluded_class"] = [8, 9]

    # 10/11/12: a suppression CHAIN, spaced so 10 suppresses 11 (IoU 0.6) but
    # NOT 12 (IoU 0.333), while 11 would suppress 12 if it were allowed to.
    # Greedy NMS must keep 12, because a box that has itself been suppressed
    # may not suppress anything. An implementation that walks the sorted list
    # without re-checking the suppressed flag drops 12.
    _put(raw, 10, (100, 500, 60, 40), car, 0.88)
    _put(raw, 11, (115, 500, 60, 40), car, 0.78)
    _put(raw, 12, (130, 500, 60, 40), car, 0.68)
    notes["suppression_chain"] = [10, 11, 12]

    # 13/14: overlap JUST below the IoU threshold (0.4815 at 14px apart; 13px
    # apart would be 0.509 and suppress) -> both survive. The must-survive
    # half of the suppression pair, placed close enough to the boundary that a
    # comparison drifting the wrong way is caught.
    _put(raw, 13, (400, 500, 40, 40), truck, 0.75)
    _put(raw, 14, (414, 500, 40, 40), truck, 0.65)
    notes["below_iou_pair"] = [13, 14]

    # 15/16: the float32-vs-float64 case, and the sharpest column here.
    #
    # 16 is nested inside 15 with exactly half its area, so in EXACT
    # arithmetic inter/union is exactly 0.5 -- and ``nms`` suppresses on
    # strict ``>``, so exact arithmetic would keep both. It does not work out
    # that way. ``decode_yolo`` undoes the pad and scale in float32 (numpy
    # keeps float32 under NEP 50 even where the scalars are Python floats),
    # and that rounding lands the IoU at 0.5000001788139343 -- 1.8e-7 ABOVE
    # the threshold -- so 16 is suppressed.
    #
    # A mirror that decodes in float64, as JavaScript does by default, gets
    # 0.5, does not suppress, and returns one more detection than Python. This
    # pair is therefore the fixture's proof that the mirror rounds to float32
    # at every step, and it fails by a whole detection rather than by an ULP.
    _put(raw, 15, (800, 300, 40, 40), bicycle, 0.72)
    _put(raw, 16, (800, 300, 40, 20), bicycle, 0.62)
    notes["float32_iou_boundary"] = [15, 16]

    # 17: an all-zero column. argmax picks class 0 (person) at score 0, which
    # is both below conf and outside the kept classes -- it must vanish
    # without upsetting the ordering.
    notes["empty_column"] = 17

    return raw, notes


def real_raw(n_cols: int) -> tuple[np.ndarray, dict]:
    """A slice of a genuine model output on a genuine frame.

    The boundary fixture above is built by hand and so can only test the cases
    someone thought of; this one carries the value distribution the graph
    actually produces -- near-duplicate boxes stacked on one vehicle, long
    tails just under threshold, class scores that are close but not equal.
    Columns are selected by score but kept in their ORIGINAL relative order,
    so the stable tie-break still means what it means in a full-size output.
    """
    import cv2
    import onnxruntime as ort

    clip = ROOT / "data" / "samples" / "motorway-a40.webm"
    capture = cv2.VideoCapture(str(clip))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"could not read a frame from {clip}")

    size = 480
    chw, scale, pad_x, pad_y = letterbox(frame, size)
    session = ort.InferenceSession(
        str(ROOT / "web" / "public" / "models" / "yolo11n-480.onnx"),
        providers=["CPUExecutionProvider"],
    )
    (out,) = session.run(None, {session.get_inputs()[0].name: chw})

    best = out[0, 4:, :].max(axis=0)
    chosen = np.sort(np.argsort(-best, kind="stable")[:n_cols])
    sliced = np.ascontiguousarray(out[:, :, chosen])
    return sliced, {
        "frameIndex": 200,
        "scale": float(scale), "padX": float(pad_x), "padY": float(pad_y),
        "sourceWidth": int(frame.shape[1]), "sourceHeight": int(frame.shape[0]),
    }


def decode_and_record(raw: np.ndarray, scale, pad_x, pad_y) -> list[dict]:
    keep = class_ids(VEHICLE_CLASSES)
    detections = decode_yolo(
        raw, scale, pad_x, pad_y, conf=CONF, iou=IOU, keep_class_ids=keep
    )
    return [
        {
            "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
            "score": d.score, "classId": d.class_id, "className": d.class_name,
        }
        for d in detections
    ]


def write_decode_cases(manifest: dict) -> None:
    keep = sorted(class_ids(VEHICLE_CLASSES))

    raw, notes = build_boundary()
    # The shipped geometry, so the pad/scale undo is exercised rather than
    # cancelled by an identity transform.
    scale, pad_x, pad_y = 0.375, 0.0, 105.0
    (OUT / "decode_boundary_raw.bin").write_bytes(raw.tobytes())
    boundary = decode_and_record(raw, scale, pad_x, pad_y)

    # Structural assertions: each constructed case must actually have had its
    # effect, or the fixture has silently stopped testing that path. These run
    # every time the fixtures are generated, so a geometry edit that quietly
    # stops exercising a branch fails here rather than shipping.
    classes = [d["classId"] for d in boundary]
    by_class = {c: [d for d in boundary if d["classId"] == c] for c in set(classes)}

    def undo(v_model: float) -> float:
        return float(np.float32(np.float32(v_model - pad_x) / np.float32(scale)))

    assert classes == sorted(classes), "output must be sorted by ascending class id"
    assert 0 not in classes, "the excluded class survived the keep filter"
    assert len(by_class.get(2, [])) == 4, ["cols 0,2,10,12 expected", by_class.get(2)]
    assert len(by_class.get(7, [])) == 3, ["cols 3,13,14 expected", by_class.get(7)]
    # Cols 15/16 are exactly 0.5 in exact arithmetic but 0.5000001788139343 in
    # the float32 the decode actually runs in, so the nested box IS
    # suppressed. Its control is the truck pair at 0.4815, which is not.
    assert len(by_class.get(1, [])) == 1, [
        "the float32 IoU boundary pair did not resolve as measured, cols 15,16",
        by_class.get(1)]
    assert len(by_class.get(5, [])) == 1 and (
        by_class[5][0]["score"] == float(np.float32(CONF))), [
        "the exactly-at-conf column was dropped, or the below-conf one survived",
        by_class.get(5)]
    tie = by_class.get(3, [])
    assert len(tie) == 1, ["the score tie must keep exactly one box", tie]
    assert abs(tie[0]["x1"] - undo(100 - 20)) < 1e-3, [
        "the score tie kept the HIGHER column index: the sort is not stable",
        tie[0]["x1"], undo(100 - 20), undo(103 - 20)]
    chain = sorted(d["score"] for d in by_class[2])
    assert len(chain) == 4 and len(set(chain)) == 4, ["chain scores", chain]
    assert len(boundary) == 10, [len(boundary), boundary]
    print(f"decode boundary: {len(boundary)} detections from {raw.shape[2]} columns "
          f"({ {c: len(v) for c, v in sorted(by_class.items())} })")

    # A third case, existing solely to sit ON the suppression threshold.
    #
    # The boundary fixture's nested pair lands at 0.5000001788139343 because
    # scale 0.375 does not divide evenly in float32, so `>` and `>=` agree
    # there and neither protects the distinction. Here scale is 0.5 and the pad
    # an integer, so undoing the letterbox is EXACT in float32 and the nested
    # box's IoU is exactly 0.5. `nms` suppresses on strict `>`, so both must
    # survive; a `>=` mirror drops one. The truck pair is the control on the
    # same axis at IoU 0.6, where both rules suppress.
    exact_raw = np.zeros((1, 4 + N_CLASSES, 4), dtype=np.float32)
    _put(exact_raw, 0, (200, 200, 40, 40), 2, 0.80)
    _put(exact_raw, 1, (200, 200, 40, 20), 2, 0.70)
    _put(exact_raw, 2, (600, 200, 40, 40), 7, 0.80)
    _put(exact_raw, 3, (600, 200, 40, 24), 7, 0.70)
    exact_scale, exact_pad_x, exact_pad_y = 0.5, 0.0, 64.0
    (OUT / "decode_iouexact_raw.bin").write_bytes(exact_raw.tobytes())
    exact = decode_and_record(exact_raw, exact_scale, exact_pad_x, exact_pad_y)
    assert len([d for d in exact if d["classId"] == 2]) == 2, [
        "the exactly-0.5 IoU pair must BOTH survive: nms suppresses on strict >",
        exact]
    assert len([d for d in exact if d["classId"] == 7]) == 1, [
        "the 0.6 IoU control pair must be suppressed", exact]
    print(f"decode iouexact: {len(exact)} detections from {exact_raw.shape[2]} columns")

    real, meta = real_raw(96)
    (OUT / "decode_real_raw.bin").write_bytes(real.tobytes())
    real_dets = decode_and_record(real, meta["scale"], meta["padX"], meta["padY"])
    assert len(real_dets) >= 3, ["the real slice decoded too little", real_dets]
    assert len(real_dets) < real.shape[2], "NMS suppressed nothing on the real slice"
    print(f"decode real: {len(real_dets)} detections from {real.shape[2]} columns")

    manifest["decode"] = {
        "conf": CONF,
        "iou": IOU,
        "keepClassIds": keep,
        "nClasses": N_CLASSES,
        "boundary": {
            "columns": int(raw.shape[2]),
            "scale": scale, "padX": pad_x, "padY": pad_y,
            "constructedCases": notes,
            "expected": boundary,
        },
        "iouexact": {
            "columns": int(exact_raw.shape[2]),
            "scale": exact_scale, "padX": exact_pad_x, "padY": exact_pad_y,
            "expected": exact,
        },
        "real": {
            "columns": int(real.shape[2]),
            "scale": meta["scale"], "padX": meta["padX"], "padY": meta["padY"],
            "sourceWidth": meta["sourceWidth"],
            "sourceHeight": meta["sourceHeight"],
            "frameIndex": meta["frameIndex"],
            "expected": real_dets,
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    write_letterbox_cases(manifest)
    write_resize_case(manifest)
    write_decode_cases(manifest)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote fixtures to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
