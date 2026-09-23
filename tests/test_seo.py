"""The page's search and sharing metadata, in the template and in what is published.

The head block is authored in ``web/index.html`` and its files in ``web/public/``;
``npm run build`` carries both into ``docs/``, which is what GitHub Pages serves.
Every assertion runs on both sides, because a tag present in the template but
dropped or rewritten by the build would pass a source-only check while the live
page went without it.

What is asserted is the portfolio's standard for every project page: a title of
at most 60 characters, a description of 120 to 160, the author, the Search
Console token, a canonical URL, a favicon that resolves, Open Graph and Twitter
tags that agree with the title and description, a 1200x630 ``og-image.png`` under
500 KB, JSON-LD describing the page as a ``WebApplication`` and the repository as
``SoftwareSourceCode`` with the author written out on both, exactly one ``<h1>``,
and a sitemap whose ``<lastmod>`` is a literal date rather than a build-time one.
"""

import datetime as dt
import json
import re
import struct
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://safdar-hussain1.github.io/trafficlens/"
REPO = "https://github.com/safdar-hussain1/trafficlens"
PROFILE = "https://github.com/safdar-hussain1"
LINKEDIN = "https://www.linkedin.com/in/safdar-hussain-a8a61b248"
SEARCH_CONSOLE_TOKEN = "0SIEfExLTSQj1qvnHWF5A5fY58KVl2lpIEnePP9CtI0"

#: (page, the directory its site-relative files resolve against)
SIDES = {
    "template": (ROOT / "web" / "index.html", ROOT / "web" / "public"),
    "published": (ROOT / "docs" / "index.html", ROOT / "docs"),
}


class _Page(HTMLParser):
    """Head metadata, JSON-LD blocks and the number of ``<h1>`` elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lang = None
        self.title = None
        self.meta: dict[str, list[str]] = {}
        self.links: list[dict] = []
        self.ld: list[str] = []
        self.h1 = 0
        self._in_head = self._in_title = self._in_ld = False
        self._buffer = ""

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "html":
            self.lang = attributes.get("lang")
        elif tag == "head":
            self._in_head = True
        elif tag == "title" and self._in_head and self.title is None:
            self._in_title, self._buffer = True, ""
        elif tag == "meta" and self._in_head:
            key = attributes.get("name") or attributes.get("property")
            if key:
                self.meta.setdefault(key.lower(), []).append(
                    (attributes.get("content") or "").strip()
                )
        elif tag == "link" and self._in_head:
            self.links.append(attributes)
        elif tag == "script" and attributes.get("type") == "application/ld+json":
            self._in_ld, self._buffer = True, ""
        elif tag == "h1":
            self.h1 += 1

    def handle_endtag(self, tag):
        if tag == "head":
            self._in_head = False
        elif tag == "title" and self._in_title:
            self._in_title, self.title = False, " ".join(self._buffer.split())
        elif tag == "script" and self._in_ld:
            self._in_ld = False
            self.ld.append(self._buffer)

    def handle_data(self, data):
        if self._in_title or self._in_ld:
            self._buffer += data

    def one(self, key: str) -> str:
        values = self.meta.get(key, [])
        assert len(values) == 1, f"expected exactly one {key!r} tag, found {values}"
        return values[0]


def _parse(side: str) -> _Page:
    page, _ = SIDES[side]
    parser = _Page()
    parser.feed(page.read_text(encoding="utf-8"))
    return parser


def _resolve(side: str, href: str) -> Path:
    """A site-relative or absolute on-site URL, as a file on that side."""
    _, base = SIDES[side]
    relative = href.replace(SITE, "")
    relative = relative[2:] if relative.startswith("./") else relative.lstrip("/")
    return base / relative


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"
    return struct.unpack(">II", data[16:24])


sides = pytest.mark.parametrize("side", sorted(SIDES))


@sides
def test_title_and_description_fit_what_a_search_result_shows(side):
    page = _parse(side)
    assert page.lang == "en"
    assert page.title and len(page.title) <= 60, (len(page.title or ""), page.title)
    assert page.title.startswith("TrafficLens — ")
    description = page.one("description")
    assert 120 <= len(description) <= 160, (len(description), description)
    assert "width=device-width" in page.one("viewport")


@sides
def test_author_search_console_and_canonical(side):
    page = _parse(side)
    assert page.one("author") == "Safdar Hussain"
    assert page.one("google-site-verification") == SEARCH_CONSOLE_TOKEN
    canonical = [link.get("href") for link in page.links if link.get("rel") == "canonical"]
    assert canonical == [SITE]


@sides
def test_the_favicon_is_a_file_that_ships(side):
    page = _parse(side)
    icons = [link for link in page.links if "icon" in (link.get("rel") or "").split()]
    assert len(icons) == 1, icons
    href = icons[0]["href"]
    assert not href.startswith("data:"), "the favicon must be a file a crawler can fetch"
    target = _resolve(side, href)
    assert target.is_file(), f"{href} does not resolve to a file on the {side} side"
    assert target.read_text(encoding="utf-8").lstrip().startswith("<svg")


@sides
def test_open_graph_and_twitter_agree_with_the_page(side):
    page = _parse(side)
    title, description = page.title, page.one("description")
    assert page.one("og:type") == "website"
    assert page.one("og:site_name") == "Safdar Hussain"
    assert page.one("og:url") == SITE
    for key in ("og:title", "twitter:title"):
        assert page.one(key) == title, key
    for key in ("og:description", "twitter:description"):
        assert page.one(key) == description, key
    assert page.one("og:image") == SITE + "og-image.png"
    assert page.one("twitter:image") == page.one("og:image")
    assert (page.one("og:image:width"), page.one("og:image:height")) == ("1200", "630")
    assert len(page.one("og:image:alt")) > 40
    assert page.one("twitter:image:alt") == page.one("og:image:alt")
    assert page.one("twitter:card") == "summary_large_image"


@sides
def test_the_share_image_is_1200_by_630_and_under_500_kb(side):
    image = _resolve(side, _parse(side).one("og:image"))
    data = image.read_bytes()
    assert _png_size(data) == (1200, 630)
    assert len(data) < 500 * 1024, f"{len(data)} bytes"


def test_the_published_share_image_is_the_authored_one():
    """docs/ must serve the file web/public/ holds, not a stale copy of it."""
    assert (ROOT / "docs" / "og-image.png").read_bytes() == (
        ROOT / "web" / "public" / "og-image.png"
    ).read_bytes()


@sides
def test_structured_data_names_the_application_the_source_and_the_author(side):
    page = _parse(side)
    assert len(page.ld) == 1, "expected one JSON-LD block"
    graph = json.loads(page.ld[0])["@graph"]
    nodes = {node["@type"]: node for node in graph}
    assert set(nodes) == {"WebApplication", "SoftwareSourceCode"}

    app = nodes["WebApplication"]
    assert app["name"] == "TrafficLens"
    assert app["url"] == SITE
    assert app["description"] == page.one("description")
    assert app["image"] == SITE + "og-image.png"
    assert app["applicationCategory"] == "BusinessApplication"
    assert app["isAccessibleForFree"] is True

    source = nodes["SoftwareSourceCode"]
    assert source["codeRepository"] == REPO
    assert source["license"] == "https://opensource.org/licenses/MIT"
    assert set(source["programmingLanguage"]) == {"Python", "TypeScript"}

    for node in graph:
        author = node["author"]
        assert author["@type"] == "Person"
        assert author["name"] == "Safdar Hussain"
        assert author["url"] == PROFILE
        assert set(author["sameAs"]) == {PROFILE, LINKEDIN}


@sides
def test_the_page_has_exactly_one_h1(side):
    assert _parse(side).h1 == 1


@pytest.mark.parametrize("base", [ROOT / "web" / "public", ROOT / "docs"], ids=["template", "published"])
def test_the_sitemap_lists_the_page_with_a_literal_lastmod(base):
    text = (base / "sitemap.xml").read_text(encoding="utf-8")
    assert f"<loc>{SITE}</loc>" in text
    dates = re.findall(r"<lastmod>([^<]*)</lastmod>", text)
    assert len(dates) == 1, dates
    # A real calendar date written into the file, so the build stays
    # byte-reproducible; the parse fails on anything else.
    dt.date.fromisoformat(dates[0])
