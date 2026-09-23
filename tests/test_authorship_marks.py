"""The page says who built it, in its source and in what is published.

Four marks, each written into the template or the app entry so that every build
carries them, and each asserted again in ``docs/``. A mark that is present in
``web/`` but lost by the build would pass a source-only check while the
published page went unsigned, so both halves are checked:

- ``<meta name="author" content="Safdar Hussain">`` in the head;
- an HTML comment naming the author and his GitHub profile;
- one ``console.info`` line from the app entry, which is the script the
  published ``index.html`` actually loads;
- a visible footer credit, "Built by Safdar Hussain", linking to the profile.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web" / "index.html"
ENTRY = ROOT / "web" / "src" / "main.ts"
PUBLISHED = ROOT / "docs" / "index.html"

PROFILE = "https://github.com/safdar-hussain1"
AUTHOR_META = re.compile(r'<meta\s+name="author"\s+content="Safdar Hussain"\s*/?>')
COMMENT = f"<!-- TrafficLens · built by Safdar Hussain · {PROFILE} -->"
CREDIT = f'<a href="{PROFILE}">Built by Safdar Hussain</a>'
CONSOLE_TEXT = f"TrafficLens — built by Safdar Hussain · {PROFILE}/trafficlens"
#: The bundler is free to re-quote a string literal (it emits backticks today),
#: so the quote character is matched, not assumed. The text inside is exact.
CONSOLE = re.compile(r"console\.info\(([\"'`])" + re.escape(CONSOLE_TEXT) + r"\1\)")


def _head(html: str) -> str:
    start, end = html.find("<head>"), html.find("</head>")
    assert 0 <= start < end, "no <head> element"
    return html[start:end]


def _footer(html: str) -> str:
    start, end = html.find("<footer"), html.find("</footer>")
    assert 0 <= start < end, "no <footer> element"
    return html[start:end]


def _published_entry() -> Path:
    """The module script docs/index.html loads -- the built app entry."""
    html = PUBLISHED.read_text(encoding="utf-8")
    scripts = re.findall(r'<script type="module"[^>]*\ssrc="\./(assets/[^"]+\.js)"', html)
    assert len(scripts) == 1, f"expected one entry script in docs/index.html, found {scripts}"
    path = ROOT / "docs" / scripts[0]
    assert path.is_file(), f"docs/index.html loads {scripts[0]}, which is not in docs/"
    return path


@pytest.mark.parametrize("page", [TEMPLATE, PUBLISHED], ids=["template", "published"])
def test_the_page_names_its_author_in_the_head(page: Path) -> None:
    html = page.read_text(encoding="utf-8")
    assert AUTHOR_META.search(_head(html)), f"{page.relative_to(ROOT)} has no author meta"


@pytest.mark.parametrize("page", [TEMPLATE, PUBLISHED], ids=["template", "published"])
def test_the_page_source_carries_the_authorship_comment(page: Path) -> None:
    assert COMMENT in page.read_text(encoding="utf-8")


@pytest.mark.parametrize("page", [TEMPLATE, PUBLISHED], ids=["template", "published"])
def test_the_footer_credits_the_author_visibly(page: Path) -> None:
    footer = _footer(page.read_text(encoding="utf-8"))
    assert CREDIT in footer, f"{page.relative_to(ROOT)}: no credit in the footer"
    opening_tag = footer[: footer.index(">") + 1]
    assert "hidden" not in opening_tag, "the footer carrying the credit is hidden"


def test_the_app_entry_signs_the_console_once() -> None:
    assert len(CONSOLE.findall(ENTRY.read_text(encoding="utf-8"))) == 1


def test_the_published_entry_bundle_signs_the_console_once() -> None:
    bundle = _published_entry().read_text(encoding="utf-8")
    assert len(CONSOLE.findall(bundle)) == 1, (
        "the script docs/index.html loads does not carry the console line; "
        "run `npm run build` in web/"
    )
