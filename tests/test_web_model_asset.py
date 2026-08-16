"""The digest the browser caches the model under must be the model's digest.

`web/src/model-asset.ts` carries a sha256 of the exported ONNX graph, and the
runtime keys Cache Storage on it. That key exists to fix a specific bug: Cache
Storage bypasses HTTP freshness entirely, so replacing the model at the same
path would leave every returning visitor running the old graph while the page
claimed the new graph's accuracy.

A digest that has drifted from the file is worse than no digest at all: the
cache would then key on a constant that no longer identifies anything, and a
genuine model swap would once again be invisible. So it is checked here, from
the bytes that are committed.
"""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_TS = ROOT / "web" / "src" / "model-asset.ts"
MODEL = ROOT / "web" / "public" / "models" / "yolo11n-480.onnx"


def _declared(name: str) -> str:
    match = re.search(
        rf"export const {name} =\s*\"?([^\";]+)\"?;", ASSET_TS.read_text()
    )
    assert match is not None, f"{name} is not declared in {ASSET_TS.name}"
    return match.group(1).strip().strip('"')


def test_the_model_the_page_loads_is_committed() -> None:
    assert MODEL.exists()
    assert _declared("MODEL_URL") == "models/yolo11n-480.onnx"


def test_the_declared_digest_is_the_model_s_digest() -> None:
    assert _declared("MODEL_SHA256") == hashlib.sha256(MODEL.read_bytes()).hexdigest()


def test_the_declared_size_is_the_model_s_size() -> None:
    assert int(_declared("MODEL_BYTES")) == MODEL.stat().st_size


def test_the_cache_version_is_derived_from_the_digest() -> None:
    """The cache key must be a slice of the digest rather than a hand-written
    label, so it cannot be forgotten when the bytes change -- which is the only
    moment it matters."""
    source = ASSET_TS.read_text()
    assert "export const MODEL_CONTENT_VERSION = MODEL_SHA256.slice(" in source


def test_the_input_size_is_the_export_size_not_the_python_default() -> None:
    """480, not `DETECT_DEFAULT_INPUT_SIZE`'s 640. Feeding 640 to this graph is
    an immediate shape error, and the two numbers are easy to confuse because
    both are 'the input size'."""
    assert int(_declared("MODEL_INPUT_SIZE")) == 480
    card = (ROOT / "web" / "public" / "models" / "MODEL_CARD.md").read_text()
    assert "[1, 3, 480, 480]" in card
