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

# Machine-produced binaries shipped from web/public/, never hand-edited and
# never edited to appease a guard: .wasm and .mjs are byte-identical copies of
# what npm shipped, .onnx is the exported browser model (a build artefact of
# this project, not an npm one). Excluded by EXTENSION, and only under
# web/public/. Excluding that whole directory would buy nothing -- none of
# these extensions are in _TEXT_SUFFIXES, so they were never scanned anyway --
# while covering the hand-authored files that live beside them: a model card,
# a licence note. Those are published straight to the site and are the
# likeliest route for a banned word or an absolute path onto a public page, so
# they stay in scope automatically rather than depending on anyone
# remembering. The directory scope matters for the same reason: an authored
# .mjs anywhere else in the tree must still be scanned.
_VENDORED_SUFFIXES = {".wasm", ".mjs", ".onnx"}
_VENDORED_DIR = "web/public/"

def _is_skipped(rel_path: str) -> bool:
    if rel_path.startswith(_VENDORED_DIR) and Path(rel_path).suffix in _VENDORED_SUFFIXES:
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

# Inflections of the banned stems, because the constraint is about the WORD,
# not about one spelling of it. Anchoring a stem with \b on both sides lets
# its -ing and -s forms through, which satisfies the letter of the rule while
# breaking its intent -- and that is exactly what happened: a tracked comment
# used the -ing form of BANNED[0] and this guard ran green over it.
#
# The suffixes are the regular English inflections that apply to a verb or a
# noun stem. They are OPTIONAL, so every bare stem still matches as before;
# this only ever widens. The two entries that are already terminal forms
# (the adverb and the hyphenated participle) pick up nothing from it, which
# is correct.
_INFLECTIONS = r"(?:s|es|ed|ing|d)?"


def _banned_pattern(word: str) -> str:
    return r"\b" + re.escape(word) + _INFLECTIONS + r"\b"


def test_no_banned_words_in_tracked_files():
    hits = []
    for f in _text_files():
        rel = str(f.relative_to(ROOT))
        if _is_skipped(rel):
            continue
        text = f.read_text(errors="ignore").lower()
        for w in BANNED:
            match = re.search(_banned_pattern(w), text)
            if match:
                hits.append(f"{rel}: {match.group(0)}")
    assert hits == [], hits


def test_the_word_guard_catches_inflections_and_not_unrelated_words():
    """A discriminating pair, varying one axis: whether the token is an
    inflection of a banned stem.

    The must-catch half is the hole this closes -- the bare-stem rule let
    every -ing/-s/-ed form through. The must-spare half is what stops the
    widening turning into a substring match: a longer word that merely
    STARTS with a banned stem is not that word, and a rule that had simply
    stopped anchoring on word boundaries would pass the first half alone.
    """
    stem_rebuild, stem_revisit, stem_college = BANNED[0], BANNED[2], BANNED[3]

    caught = [
        stem_rebuild + "ing", stem_rebuild + "s", stem_rebuild + "ed",
        stem_revisit + "ing", stem_revisit + "ed", stem_revisit + "s",
        stem_college + "s",
        stem_rebuild,  # the bare stem must still match
    ]
    for token in caught:
        assert re.search(_banned_pattern(stem_rebuild), token) or re.search(
            _banned_pattern(stem_revisit), token
        ) or re.search(_banned_pattern(stem_college), token), (
            f"{token!r} slipped past the widened guard"
        )

    # Unrelated longer words that begin with a banned stem, and a hyphenated
    # form that is a different word. None of these may fire.
    spared = [
        stem_rebuild + "able", stem_revisit + "ation", stem_college + "ial",
        stem_rebuild + "er",
    ]
    for token in spared:
        for stem in (stem_rebuild, stem_revisit, stem_college):
            assert not re.search(_banned_pattern(stem), token), (
                f"{token!r} is a false positive: it merely starts with a "
                f"banned stem"
            )

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
    vendored = tmp_path / "web" / "public" / "ort-wasm-simd-threaded.asyncify.mjs"
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

    Asserted as a discriminating pair. A rule that skips every .mjs everywhere
    would satisfy the must-skip half on its own while silently excusing an
    authored file elsewhere in the tree, so the must-scan half is what proves
    the exclusion is scoped rather than merely firing.
    """
    # Must be skipped: machine-produced, under web/public/.
    for path in ["web/public/ort-wasm-simd-threaded.asyncify.wasm",
                 "web/public/ort-wasm-simd-threaded.asyncify.mjs",
                 "web/public/models/yolo11n-480.onnx"]:
        assert _is_skipped(path), f"{path} is vendored and must not be scanned"

    # Must be scanned: authored, and/or outside web/public/. The .mjs and .onnx
    # entries are the negative controls for the extension rule -- same
    # extensions, different directory.
    for path in ["web/public/models/MODEL_CARD.md",
                 "web/public/models/LICENCE.txt",
                 "web/index.html",
                 "web/src/authored.mjs",
                 "src/trafficlens/thing.mjs",
                 "docs/a.onnx"]:
        assert not _is_skipped(path), f"{path} is authored and must be scanned"


def test_the_vendored_exclusion_still_discriminates_once_mjs_is_scannable(
    tmp_path, monkeypatch
):
    """Wake the dormant mechanism and check it discriminates, not just fires.

    _VENDORED_SUFFIXES does nothing today because _TEXT_SUFFIXES drops those
    extensions first. This exercises the exact future the comment above names --
    a later task adds .mjs to _TEXT_SUFFIXES because it is JavaScript -- and
    asserts the pair: the vendored copy under web/public/ stays excluded, and an
    authored .mjs anywhere else is scanned. An unscoped extension rule passes the
    first half and fails the second.
    """
    banned = BANNED[0]
    vendored = tmp_path / "web" / "public" / "ort-wasm-simd-threaded.asyncify.mjs"
    vendored.parent.mkdir(parents=True)
    vendored.write_text(f"// {banned} by the vendor toolchain\n")
    authored = tmp_path / "web" / "src" / "authored-helper.mjs"
    authored.parent.mkdir(parents=True)
    authored.write_text(f"// this helper was {banned} by hand\n")

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-f", "-A"], cwd=tmp_path, check=True)
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    monkeypatch.setattr(
        sys.modules[__name__], "_TEXT_SUFFIXES", _TEXT_SUFFIXES | {".mjs"}
    )

    collected = sorted(str(p.relative_to(tmp_path)) for p in _text_files())
    assert collected == ["web/public/ort-wasm-simd-threaded.asyncify.mjs",
                         "web/src/authored-helper.mjs"], collected

    with pytest.raises(AssertionError) as caught:
        test_no_banned_words_in_tracked_files()

    reported = str(caught.value)
    assert "web/src/authored-helper.mjs" in reported, (
        "an authored .mjs outside web/public/ was not scanned"
    )
    assert "web/public/" not in reported, (
        "a vendored .mjs under web/public/ should stay excluded"
    )


def test_published_web_assets_are_not_git_ignored():
    """The browser model and its card ship from web/public/models/, and the
    build republishes them under docs/.

    A bare "models/" rule matches at any depth, which would ignore all three and
    silently never ship them to Pages -- or force `git add -f`, a habit worth
    not forming. Weights stay covered by *.pt.

    `--no-index` is load-bearing, and this test could not fail without it. Plain
    `git check-ignore` consults the index first and reports any TRACKED path as
    not ignored, whatever the rules say -- and all three of these paths are
    tracked, so the question was answered by their being committed rather than by
    .gitignore. Replacing `/models/` with a bare `models/` left this green while
    the ignore rule really did swallow all three; the mutation battery found it.
    `--no-index` asks the question the test means: would the rules ignore this
    path if it were not already tracked -- which is exactly the situation of the
    next person to add a file here.
    """
    for path in ["web/public/models/yolo11n-480.onnx",
                 "web/public/models/MODEL_CARD.md",
                 "docs/models/MODEL_CARD.md"]:
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
        assert r.returncode == 1, f"{path} IS git-ignored and would never ship"


# Process docs are DOCUMENTS. Restricting the name rule to document
# extensions is what stops it firing on source: Task 20 adds
# web/src/runtime/session.ts, an execution-provider wrapper that has nothing to
# do with a work session, and the bare name rule matched it on "session.".
# Widening the exclusion to "anything under web/src/" would have been the wrong
# fix -- a process doc can be committed anywhere -- so the narrowing is by
# extension, and the pair below proves it still bites.
_PROCESS_DOC_SUFFIXES = {".md", ".markdown", ".mdx", ".txt", ".rst",
                         ".pdf", ".doc", ".docx", ".odt"}
_PROCESS_DOC_NAMES = r"(^|/)(plan|spec|design-doc|session|notes|handbook)[-_.]"


def _process_docs(paths) -> list[str]:
    return [p for p in paths
            if Path(p).suffix.lower() in _PROCESS_DOC_SUFFIXES
            and re.search(_PROCESS_DOC_NAMES, p.lower())]


def test_no_process_docs_tracked():
    assert _process_docs(_tracked()) == [], _process_docs(_tracked())


def test_the_process_doc_rule_still_catches_documents_and_spares_source():
    """A discriminating pair, varying one axis: the file's extension.

    The must-catch half is what the guard is for. The must-spare half is what
    the narrowing bought, and without it a rule that simply stopped matching
    anything would pass the first half on its own.
    """
    caught = ["plan-2026.md", "docs/session-notes.md", "spec.txt",
              "private/handbook.pdf", "a/b/notes.rst",
              # .markdown and .mdx are markdown too; the old, extension-blind
              # rule caught these and the narrowing must not lose them.
              "docs/plan-x.markdown", "plan-x.mdx"]
    assert _process_docs(caught) == caught, "the guard stopped catching process docs"

    spared = ["web/src/runtime/session.ts", "web/src/runtime/session.test.ts",
              "src/trafficlens/planner.py", "web/src/notes.ts", "specimen.py"]
    assert _process_docs(spared) == [], "the guard is still firing on source files"

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
