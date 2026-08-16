import re, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [p for p in out.stdout.splitlines() if p]

_TEXT_SUFFIXES = {".py", ".md", ".ts", ".js", ".html", ".css", ".yaml", ".yml",
                  ".json", ".toml", ".txt", ".cfg", ".sh"}

def _looks_like_text(path: Path, sniff_bytes: int = 65536) -> bool:
    # Extension-less files (LICENSE, .gitignore, ...) have no suffix to check,
    # so decide by content: reject anything with a NUL byte or that isn't
    # valid UTF-8, accept the rest.
    chunk = path.read_bytes()[:sniff_bytes]
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True

def _text_files() -> list[Path]:
    files = []
    for p in _tracked():
        path = ROOT / p
        if not path.is_file():
            continue
        suffix = Path(p).suffix
        if suffix in _TEXT_SUFFIXES:
            files.append(path)  # fast path: known text extension
        elif suffix == "" and _looks_like_text(path):
            files.append(path)  # extension-less: only if content is text
    return files

# Paths that will legitimately contain banned substrings or absolute
# home-directory paths once the web build lands, and so are excluded from
# the two content guards below. Everything else stays fully in scope, in
# particular docs/*.html is still scanned.
_SKIP_PATTERNS = [
    # npm lockfile: resolved registry URLs, transitive package names and
    # integrity hashes can contain arbitrary substrings, and a local dev
    # install can bake an absolute filesystem path into a "file:" dependency.
    "package-lock.json",
    # bundled/minified JS+CSS output of the web build, published under the
    # docs/ site root: vendor code pulled in from third-party packages and
    # inlined sourcemaps can contain either banned substrings or absolute
    # build-machine paths.
    "docs/assets/",
]

# Vendored runtime assets: byte-identical copies of what npm shipped, never
# edited to appease a guard. Excluded by EXTENSION rather than by directory.
# Excluding all of web/public/ would buy nothing -- none of these extensions
# are in _TEXT_SUFFIXES, so they were never scanned anyway -- while covering
# the hand-authored files that live beside them: a model card, a licence
# note. Those are published straight to the site and are the likeliest route
# for a banned word or an absolute path onto a public page, so they stay in
# scope automatically rather than depending on anyone remembering.
_VENDORED_SUFFIXES = {".wasm", ".mjs", ".onnx"}

def _is_skipped(rel_path: str) -> bool:
    if Path(rel_path).suffix in _VENDORED_SUFFIXES:
        return True
    for pat in _SKIP_PATTERNS:
        if pat.endswith("/"):
            if rel_path == pat.rstrip("/") or rel_path.startswith(pat):
                return True
        elif rel_path == pat or rel_path.endswith("/" + pat):
            return True
    return False

# fragments so this file itself never matches
BANNED = [a + b for a, b in [
    ("re", "build"), ("re", "built"), ("re", "visit"), ("col", "lege"),
    ("course", "work"), ("origin", "ally"), ("clau", "de"), ("anthro", "pic"),
    ("co-auth", "ored"),
]]

def test_no_banned_words_in_tracked_files():
    hits = []
    for f in _text_files():
        rel = str(f.relative_to(ROOT))
        if _is_skipped(rel):
            continue
        text = f.read_text(errors="ignore").lower()
        for w in BANNED:
            if re.search(r"\b" + re.escape(w) + r"\b", text):
                hits.append(f"{rel}: {w}")
    assert hits == [], hits

def test_no_absolute_user_paths_in_tracked_files():
    needle = "/Us" + "ers/"
    hits = []
    for f in _text_files():
        rel = str(f.relative_to(ROOT))
        if _is_skipped(rel):
            continue
        if needle in f.read_text(errors="ignore"):
            hits.append(rel)
    assert hits == [], hits

def test_authored_files_under_web_public_are_scanned(tmp_path, monkeypatch):
    """web/public/ ships straight to docs/ and onto the published site, so
    anything hand-authored there must stay in scope. Only vendored binaries are
    excluded, and only by extension.

    Drives the real scan -- git ls-files, the text sniff, _is_skipped, the
    assertion -- against a throwaway repo rather than testing _is_skipped on its
    own, so re-broadening the exclusion anywhere along that path still fails
    here. Both content guards share _is_skipped, so covering one covers both.
    """
    banned = BANNED[0]
    card = tmp_path / "web" / "public" / "models" / "MODEL_CARD.md"
    card.parent.mkdir(parents=True)
    card.write_text(f"These weights were {banned} from the upstream export.\n")
    vendored = tmp_path / "web" / "public" / "ort-wasm-simd-threaded.jsep.mjs"
    vendored.write_text(f"// {banned} by the vendor toolchain\n")

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    # -f so a global excludesFile on the developer's machine cannot quietly
    # empty this fixture and leave the test passing on nothing.
    subprocess.run(["git", "add", "-f", "-A"], cwd=tmp_path, check=True)
    # _tracked() and _text_files() read ROOT at call time, so patching the
    # module global redirects the whole scan at the throwaway repo.
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)

    # The card is collected for scanning; the vendored .mjs is not, because
    # _TEXT_SUFFIXES filters it out one step earlier than _is_skipped ever
    # runs. That ordering is the point: vendored binaries were always out of
    # scope on extension alone, so excluding all of web/public/ bought nothing
    # and cost the card its coverage.
    collected = [str(p.relative_to(tmp_path)) for p in _text_files()]
    assert collected == ["web/public/models/MODEL_CARD.md"], collected

    with pytest.raises(AssertionError) as caught:
        test_no_banned_words_in_tracked_files()

    reported = str(caught.value)
    assert "web/public/models/MODEL_CARD.md" in reported, (
        "an authored file under web/public/ was not scanned"
    )
    assert ".mjs" not in reported, "vendored runtime assets should stay excluded"


def test_vendored_runtime_assets_stay_excluded_from_the_content_guards():
    """_VENDORED_SUFFIXES is belt-and-braces, and is tested as such.

    Today these extensions never reach _is_skipped -- _TEXT_SUFFIXES drops them
    first -- so this pins the second line of defence: if a later task adds .mjs
    to _TEXT_SUFFIXES (reasonable, it is JavaScript), the vendored files must
    still not be scanned, and must still never be edited to appease a guard.
    """
    for path in ["web/public/ort-wasm-simd-threaded.jsep.wasm",
                 "web/public/ort-wasm-simd-threaded.jsep.mjs",
                 "web/public/models/yolo11n-480.onnx"]:
        assert _is_skipped(path), f"{path} is vendored and must not be scanned"

    for path in ["web/public/models/MODEL_CARD.md",
                 "web/public/models/LICENCE.txt",
                 "web/index.html"]:
        assert not _is_skipped(path), f"{path} is authored and must be scanned"


def test_published_web_assets_are_not_git_ignored():
    """The browser model and its card ship from web/public/models/, and the
    build republishes them under docs/.

    A bare "models/" rule matches at any depth, which would ignore all three and
    silently never ship them to Pages -- or force `git add -f`, a habit worth
    not forming. Weights stay covered by *.pt.
    """
    for path in ["web/public/models/yolo11n-480.onnx",
                 "web/public/models/MODEL_CARD.md",
                 "docs/models/MODEL_CARD.md"]:
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
        assert r.returncode == 1, f"{path} IS git-ignored and would never ship"


def test_no_process_docs_tracked():
    bad = [p for p in _tracked()
           if re.search(r"(^|/)(plan|spec|design-doc|session|notes|handbook)[-_.]", p.lower())]
    assert bad == [], bad

def test_private_paths_are_git_ignored():
    for path in ["yolo11n.pt", "data/samples/motorway-a40.webm", "private/handbook.pdf",
                 ".superpowers/sdd/progress.md"]:
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
        assert r.returncode == 0, f"{path} is NOT git-ignored"

def test_no_attribution_in_git_history():
    out = subprocess.run(["git", "log", "--all", "--format=%B"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    pattern = "|".join(a + b for a, b in [("clau", "de"), ("anthro", "pic"), ("co-auth", "ored")])
    assert re.search(pattern, out.stdout, re.I) is None
