"""`docs/` must be the build output of the sources committed beside it.

`docs/` is the published GitHub Pages site and it is a build artefact living in
the repository. Nothing asserted that until now, which meant the site could
serve a bundle built from sources that no longer existed, indefinitely, with
every test green: the suite tests `web/src`, and the browser runs `docs/`.

`npm run build` writes `docs/BUILD_MANIFEST.json`, recording a digest of every
file the build reads and every file it writes. This module recomputes both
sides from the working tree. Change a source and forget to run the build again, hand-edit a
file in `docs/`, or delete an asset the site needs, and one of these fails.

What it deliberately does NOT do is run `npm run build` itself and diff the
result. That would need node in the test environment and would make the Python
suite depend on a JavaScript toolchain; more importantly it would be a slower
way of asking the same question, since a build from unchanged inputs produces
unchanged outputs.
"""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MANIFEST = DOCS / "BUILD_MANIFEST.json"
WEB = ROOT / "web"

#: Floors, so no assertion here can pass by finding nothing. An empty manifest
#: would otherwise satisfy every comparison below vacuously.
MINIMUM_INPUTS = 20
MINIMUM_OUTPUTS = 8


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.exists(), (
        "docs/BUILD_MANIFEST.json is missing: run `npm run build` in web/, which "
        "writes it as part of the build"
    )
    return json.loads(MANIFEST.read_text())


def test_the_manifest_records_a_real_build(manifest: dict) -> None:
    assert len(manifest["inputs"]) >= MINIMUM_INPUTS
    assert len(manifest["outputs"]) >= MINIMUM_OUTPUTS


def test_every_recorded_output_is_present_and_unchanged(manifest: dict) -> None:
    """Nothing in docs/ was hand-edited or lost after the build wrote it."""
    for relative, expected in manifest["outputs"].items():
        path = ROOT / relative
        assert path.exists(), f"{relative} is in the manifest but missing from the tree"
        assert _digest(path) == expected, (
            f"{relative} does not match the build that produced it; run "
            f"`npm run build`"
        )


def test_docs_contains_nothing_the_build_did_not_write(manifest: dict) -> None:
    """A stray file in docs/ is published to visitors, so it is a defect even
    when everything the manifest lists is intact."""
    on_disk = {
        str(path.relative_to(ROOT))
        for path in DOCS.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    assert on_disk == set(manifest["outputs"])


def test_every_build_input_still_hashes_to_what_was_built(manifest: dict) -> None:
    """The half that catches a stale site: a source edited after the last build.

    This is the assertion that ties docs/ to `npm run build` rather than merely
    to itself.
    """
    stale = []
    for relative, expected in manifest["inputs"].items():
        path = ROOT / relative
        if not path.exists():
            stale.append(f"{relative} (deleted)")
        elif _digest(path) != expected:
            stale.append(relative)
    assert not stale, (
        "docs/ was built from different sources than the ones committed; run "
        f"`npm run build` in web/. Changed since the build: {sorted(stale)}"
    )


def test_no_source_the_build_reads_is_missing_from_the_inputs(manifest: dict) -> None:
    """A file added to web/src or web/public without a fresh build would otherwise
    be invisible to the check above -- the manifest simply would not mention it,
    and every recorded input would still match."""
    tracked = set(manifest["inputs"])
    missing = []
    for base in (WEB / "src", WEB / "public"):
        for path in base.rglob("*"):
            if not path.is_file() or path.name.endswith(".test.ts"):
                continue
            if str(path.relative_to(ROOT)) not in tracked:
                missing.append(str(path.relative_to(ROOT)))
    assert not missing, (
        f"these files exist but the last build never saw them; build again: {sorted(missing)}"
    )


def test_the_published_index_references_only_files_that_shipped(manifest: dict) -> None:
    """The site's entry point, checked against what is actually in docs/."""
    index = (DOCS / "index.html").read_text()
    outputs = set(manifest["outputs"])
    referenced = [
        fragment.split('"')[0]
        for marker in ('src="./', 'href="./')
        for fragment in index.split(marker)[1:]
    ]
    assert referenced, "docs/index.html references no assets at all"
    for target in referenced:
        assert f"docs/{target}" in outputs, f"index.html points at {target}, which is not in docs/"
