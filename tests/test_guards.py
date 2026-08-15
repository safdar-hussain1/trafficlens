import re, subprocess
from pathlib import Path

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
    # bundled/minified JS+CSS output of the web build: vendor code pulled
    # in from third-party packages and inlined sourcemaps can contain
    # either banned substrings or absolute build-machine paths.
    "web/public/",
    # GitHub Pages copy of the built web assets (same bundler output as
    # web/public/), published under the docs/ site root; same risk.
    "docs/assets/",
]

def _is_skipped(rel_path: str) -> bool:
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
