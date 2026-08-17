"""Every figure the documents publish must be the figure ``reports/`` measured.

The prose equivalent of ``tests/test_site_data_sync.py``. That module pins the
site's baked artefact to the reports; this one pins the four hand-written
documents -- ``README.md`` and the three cards published from ``web/public/`` --
to the same sources. Prose has no generator, so there is nothing to regenerate
and compare: the only way a sentence and a measurement can be held together is
to read the measurement out of the JSON at test time and look for it in the
document.

**The hole this module is built against.** A test that walks the numbers it finds
in a document and checks each one against the reports passes perfectly on a
document containing no numbers at all. A test that walks a hand-written list of
expected values stops protecting anything the day an entry is deleted. Both are
checks that cannot fail, and this repository has met that shape of defect
repeatedly.

So there are six independent mechanisms here:

1. ``PINNED`` -- pointers into the reports, one per published figure, with the
   VALUE read from the JSON at test time and never written here. A report that
   moves reddens the document that quotes it.
2. ``DERIVED`` -- the figures the documents compute rather than copy, with each
   formula written out again here, so a document's arithmetic is checked and not
   merely its transcription.
3. The containment sweep, the other direction: every number in a results table
   must be a value that actually occurs in the report that table declares as its
   source. A figure cannot be invented, mistyped, or taken from the wrong report.
4. Exact pins where a bare presence check is too weak: ``LABEL_PINS`` pins a figure
   to its own row's label, ``CONTEXT_PINS`` pins a prose figure inside the phrase
   that gives it its meaning, and the two counting tables and the degradation table
   are compared against their reports cell by cell.
5. Floors, so nothing above can pass by finding nothing: how many figures are
   pinned, how many numbers were swept, how many tables each document has, which
   reports contribute at all, and -- the strongest of them -- that no numeric pin is
   left where a one-step report move would already be satisfied by a number the
   document happens to contain.
6. The redaction: no absolute km/h derived from the flagship clip may appear in
   any of the four documents. That clip's along-road scale has no independent
   anchor, so as a published figure it would assert a scale the survey withdrew.

Mechanisms 1, 3 and 4 are deliberately different in kind, and each covers the
others' blind spot. 1 asks whether a document prints a figure at all, anywhere --
which is the only question that can be asked of prose. 3 is a net over the tables:
it cannot tell a figure taken from the wrong row of the right report, but it
catches every number no report supports at all, including one invented to fill a
gap. 4 is exact -- this cell, this value -- and is what catches the wrong row.
Mutation testing drove that division rather than taste: with only 1 and 3, moving
``caseCount`` from 8 to 9 left the pin on it green, because a document of this
length prints an 8 somewhere whatever the report says.

**That weakness is now bounded by a test rather than by judgement.**
``test_no_numeric_pin_can_be_satisfied_by_a_coincidence`` moves every numeric pin's
value one step in its own last published digit and asks whether that rendering is
ALREADY in the document. Where it is, mechanism 1 alone would stay green on a real
report move, so the pin must also be covered by something exact -- a ``LABEL_PINS``
row, a ``CONTEXT_PINS`` phrase, or one of the cell-pinned tables, whose covered
pointers are derived from the same builders the cell tests use rather than listed by
hand. Twelve pins failed that condition when it was first written, over nine report
pointers, and every one of them turned out to be a report move the module really did
survive; they are why ``CONTEXT_PINS`` exists.

**Scope, stated rather than left to be discovered.** The sweep covers the
results TABLES and the source-and-protocol paragraph above each one. It does not
sweep general prose, because prose carries counts of sections, dates, dependency
versions and licence identifiers that no report contains, and an allowlist wide
enough for those would be a hole wider than the mechanism. Prose figures are
covered by ``PINNED`` instead, which is exact. What stops the sweep's scope
quietly shrinking is ``test_every_table_declares_where_its_numbers_come_from``:
every table in all four documents must name its source, and the tables whose
source is not a ``reports/*.json`` are enumerated in ``TABLES_NOT_FROM_REPORTS``
with what they do come from.

The documents are read at their SOURCE paths. The three cards are authored under
``web/public/`` because ``npm run build`` empties ``docs/`` on every run, so a
file authored there is destroyed by the next build; Vite copies ``publicDir``
verbatim, which is how they reach the published site. That ``docs/`` carries the
same bytes is asserted by ``tests/test_docs_build_manifest.py``, not here.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

#: The documents this module pins, by the short key ``PINNED`` refers to them by.
DOCUMENTS = {
    "README": ROOT / "README.md",
    "DESIGN_CARD": ROOT / "web" / "public" / "DESIGN_CARD.md",
    "CALIBRATION": ROOT / "web" / "public" / "CALIBRATION.md",
    "ARCHITECTURE": ROOT / "web" / "public" / "ARCHITECTURE.md",
}

#: Floors. Every one of them exists because the assertion it guards compares
#: structures that are equal when both are empty, or searches text that is
#: trivially searchable when there is none.
MINIMUM_PINNED = 100
MINIMUM_SWEPT_NUMBERS = 450
MINIMUM_PINNED_PER_DOCUMENT = 5
#: Report-sourced tables only -- an undeclared table is not swept, so counting it
#: here would let the sweep's scope shrink while this floor still passed.
MINIMUM_SOURCED_TABLES = {
    "README": 10, "DESIGN_CARD": 1, "CALIBRATION": 4, "ARCHITECTURE": 1
}

#: The figure this project must never restate, and where it lives. It is the
#: maximum speed the parity fixture recorded on the flagship clip; the clip ships
#: with no calibration block because its along-road scale has no independent
#: anchor, so publishing this number would assert a scale that was withdrawn.
FORBIDDEN_SPEED_REPORT = "parity.json"
FORBIDDEN_SPEED_POINTER = "realClip/maxSpeedKmh"

#: Renderings a swept number is allowed to be. A report value counts as
#: published at any of these precisions -- the documents round for a reader, and
#: which precision suits a given table is the document's business.
_ROUNDINGS = ("int", "0dp", "1dp", "2dp", "3dp", "4dp", "g", "e1", "e2", "repr")


def _read(name: str) -> dict:
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def _document(key: str) -> str:
    path = DOCUMENTS[key]
    assert path.exists(), (
        f"{path.relative_to(ROOT)} is missing; this module must not pass by "
        f"absence -- every assertion below would be searching nothing"
    )
    return path.read_text(encoding="utf-8")


def _collapse(text: str) -> str:
    """Runs of whitespace as single spaces, so a phrase that markdown wrapped across
    two lines still matches. Case and punctuation are untouched."""
    return re.sub(r"\s+", " ", text)


def _flatten(text: str) -> str:
    """Markdown prose as one comparable line: blockquote markers dropped, runs of
    whitespace collapsed, case folded. Only used for quoted-sentence pins."""
    without_quotes = re.sub(r"(?m)^\s*>\s?", "", text)
    return re.sub(r"\s+", " ", without_quotes).strip().lower()


def _mentions_number(text: str, number: str) -> bool:
    """Whether ``text`` prints ``number`` as a number in its own right.

    A plain substring test is not good enough and this was measured, not
    theorised: the pin on 565 predicted crossings was being satisfied by the "565"
    inside an unrelated 0.5650, so deleting the row it was meant to protect left it
    green. So a digit or a decimal point before the match disqualifies it, and so
    does a digit after it -- but a full stop after it does not, because a figure at
    the end of a sentence is still that figure. Only ``.`` followed by a digit is
    rejected, which is the case where the match is the head of a longer number.
    """
    pattern = rf"(?<![\d.]){re.escape(number)}(?!\d)(?!\.\d)"
    return re.search(pattern, text) is not None


def _resolve(document, pointer: str):
    """The value at a ``/``-separated pointer.

    This module's own resolver. Report keys contain ``+`` and ``-`` and spaces
    (``engine+gate``, ``p=0.00``) but never ``/``, so a plain split is exact.
    """
    here = document
    for step in pointer.split("/"):
        if isinstance(here, list):
            here = here[int(step)]
        else:
            here = here[step]
    return here


def _render(value, kind: str) -> str:
    """One value, one rendering. ``kind`` is the document's chosen precision."""
    if kind == "int":
        assert float(value) == int(value), f"{value!r} is not a whole number"
        return str(int(value))
    if kind == "len":
        return str(len(value))
    if kind == "text":
        return str(value)
    if kind == "g":
        return f"{value:g}"
    if kind == "e1":
        return f"{value:.1e}"
    if kind == "e2":
        return f"{value:.2e}"
    if kind == "repr":
        # The exact literal. A control that is one floating-point step from its
        # partner is published AS that literal, because rounding it to anything
        # shorter is the same string as the partner and the pair stops being one.
        return repr(value)
    if kind.endswith("dp"):
        return f"{float(value):.{int(kind[:-2])}f}"
    raise AssertionError(f"unknown rendering {kind!r}")


# --- 1. The pinned set: a pointer per published figure ------------------------
#
# (document, report file, pointer into it, rendering). The VALUE is never here.
# Adding a figure to a document means adding its pointer; MINIMUM_PINNED and
# test_every_report_contributes_a_pinned_figure are what stop the table being
# quietly emptied instead.

PINNED = [
    # -- counting_accuracy.json ------------------------------------------------
    ("README", "counting_accuracy.json", "labels/total", "int"),
    ("README", "counting_accuracy.json", "labels/certain", "int"),
    ("README", "counting_accuracy.json", "labels/probable", "int"),
    ("README", "counting_accuracy.json", "frames", "int"),
    ("README", "counting_accuracy.json", "window/end_frame", "int"),
    ("README", "counting_accuracy.json", "detector/confidence", "2dp"),
    ("README", "counting_accuracy.json", "detector/imgsz", "int"),
    ("README", "counting_accuracy.json", "methods/engine+gate/full/precision", "3dp"),
    ("README", "counting_accuracy.json", "methods/engine+gate/full/recall", "3dp"),
    ("README", "counting_accuracy.json", "methods/engine+gate/full/f1", "3dp"),
    ("README", "counting_accuracy.json", "methods/engine+gate/full/n_predicted", "int"),
    ("README", "counting_accuracy.json", "methods/engine+gate/full/true_positives", "int"),
    ("README", "counting_accuracy.json", "methods/engine+gate/full/false_positives", "len"),
    ("README", "counting_accuracy.json", "methods/engine+gate/full/misses", "len"),
    ("README", "counting_accuracy.json", "methods/centroid+gate/full/f1", "3dp"),
    ("README", "counting_accuracy.json", "methods/greedy-iou+gate/full/f1", "3dp"),
    ("README", "counting_accuracy.json", "methods/engine+band/full/f1", "3dp"),
    ("README", "counting_accuracy.json", "methods/engine+per-frame/full/n_predicted", "int"),
    ("README", "counting_accuracy.json", "methods/engine+per-frame/full/false_positives", "len"),
    ("README", "counting_accuracy.json", "methods/engine+gate/certain_only/precision", "3dp"),
    ("README", "counting_accuracy.json", "methods/engine+gate/certain_only/recall", "3dp"),
    ("README", "counting_accuracy.json", "methods/engine+gate/certain_only/f1", "3dp"),
    ("README", "counting_accuracy.json", "methods/engine+gate/certain_only_naive/f1", "3dp"),
    ("README", "counting_accuracy.json", "methods/engine+gate/certain_only/predictions_moved_to_ignore", "int"),
    ("README", "counting_accuracy.json", "methods/engine+gate/full/class_consistency/rate", "3dp"),
    ("README", "counting_accuracy.json", "timing/engine+gate/ms_per_frame", "4dp"),
    ("README", "counting_accuracy.json", "timing/centroid+gate/ms_per_frame", "4dp"),
    ("README", "counting_accuracy.json", "band_sweep/median_gate_approach_px_per_frame", "3dp"),
    ("DESIGN_CARD", "counting_accuracy.json", "labels/total", "int"),
    ("DESIGN_CARD", "counting_accuracy.json", "labels/certain", "int"),
    ("DESIGN_CARD", "counting_accuracy.json", "labels/probable", "int"),
    ("DESIGN_CARD", "counting_accuracy.json", "methods/engine+gate/full/f1", "3dp"),
    ("DESIGN_CARD", "counting_accuracy.json", "methods/engine+gate/certain_only/f1", "3dp"),
    ("DESIGN_CARD", "counting_accuracy.json", "detector/imgsz", "int"),
    # -- robustness.json -------------------------------------------------------
    ("README", "robustness.json", "seed", "int"),
    ("README", "robustness.json", "protocols/frame_rate/entries/5/methods/engine+gate/f1", "3dp"),
    ("README", "robustness.json", "protocols/frame_rate/entries/5/methods/engine+gate/n_predicted", "int"),
    ("README", "robustness.json", "protocols/box_jitter/entries/2/methods/engine+gate/f1", "3dp"),
    ("README", "robustness.json", "protocols/box_jitter/entries/4/methods/engine+gate/f1", "3dp"),
    ("README", "robustness.json", "protocols/detection_dropout/entries/0/methods/engine+gate/f1", "3dp"),
    ("README", "robustness.json", "protocols/detection_dropout/entries/4/methods/engine+gate/f1", "3dp"),
    ("README", "robustness.json", "association_floor_ablation/shipped_floor", "1dp"),
    ("README", "robustness.json", "association_floor_ablation/largest_f1_gain/gain", "4dp"),
    ("README", "robustness.json", "association_floor_ablation/largest_f1_gain_by_protocol/detection_dropout/gain", "4dp"),
    ("README", "robustness.json", "association_floor_ablation/largest_f1_gain_by_protocol/box_jitter/gain", "4dp"),
    ("README", "robustness.json", "questions/tracker_separation/levels_measured", "int"),
    ("README", "robustness.json", "questions/tracker_separation/levels_where_trackers_differ", "len"),
    ("README", "robustness.json", "questions/tracker_separation/max_f1_spread", "4dp"),
    ("README", "robustness.json", "jitter_calibration/stress_multiple_at_max_sigma/lowest", "2dp"),
    ("README", "robustness.json", "jitter_calibration/stress_multiple_at_max_sigma/highest", "2dp"),
    ("README", "robustness.json", "questions/band_step_over/first_rate_with_step_over", "int"),
    ("DESIGN_CARD", "robustness.json", "protocols/frame_rate/entries/5/methods/engine+gate/f1", "3dp"),
    ("DESIGN_CARD", "robustness.json", "protocols/box_jitter/entries/2/methods/engine+gate/f1", "3dp"),
    ("DESIGN_CARD", "robustness.json", "questions/tracker_separation/levels_where_trackers_differ", "len"),
    # -- tracking.json ---------------------------------------------------------
    ("README", "tracking.json", "clean/trackers/engine/fragmentation_ratio", "4dp"),
    ("README", "tracking.json", "clean/trackers/centroid/fragmentation_ratio", "4dp"),
    ("README", "tracking.json", "clean/trackers/engine/class_consistency/rate", "3dp"),
    ("README", "tracking.json", "gate_region/half_width_px", "int"),
    ("README", "tracking.json", "questions/agreement_with_crossing_f1/levels_where_the_two_agree", "len"),
    ("README", "tracking.json", "questions/tracker_separation/max_spread", "4dp"),
    ("DESIGN_CARD", "tracking.json", "claims_not_made", "len"),
    ("DESIGN_CARD", "tracking.json", "claims_not_made/0/claim", "text"),
    ("DESIGN_CARD", "tracking.json", "claims_not_made/1/claim", "text"),
    ("DESIGN_CARD", "tracking.json", "claims_not_made/2/claim", "text"),
    ("DESIGN_CARD", "tracking.json", "claims_not_made/3/claim", "text"),
    # -- speed_synthetic.json --------------------------------------------------
    ("README", "speed_synthetic.json", "road_plane/holdout_max_error_m", "e1"),
    ("README", "speed_synthetic.json", "zero_noise/homography_chain_only/settled/max_abs_error_kmh", "e1"),
    ("README", "speed_synthetic.json", "zero_noise/homography_chain_only/requirement_kmh", "1dp"),
    ("README", "speed_synthetic.json", "zero_noise/full_chain/settled/max_abs_error_kmh", "3dp"),
    ("README", "speed_synthetic.json", "zero_noise/full_chain/by_band/5/max_relative_percent", "3dp"),
    ("README", "speed_synthetic.json", "zero_noise/smoothing_trade_off/by_band/5/kalman_anchor_rmse_kmh", "4dp"),
    ("README", "speed_synthetic.json", "zero_noise/smoothing_trade_off/by_band/5/raw_detection_anchor_rmse_kmh", "4dp"),
    ("README", "speed_synthetic.json", "check_c/max_abs_difference_kmh", "3dp"),
    ("README", "speed_synthetic.json", "check_c/agreement_requirement_kmh", "1dp"),
    ("DESIGN_CARD", "speed_synthetic.json", "zero_noise/full_chain/settled/max_abs_error_kmh", "3dp"),
    ("DESIGN_CARD", "speed_synthetic.json", "road_plane/holdout_max_error_m", "e1"),
    # -- speed_real.json -------------------------------------------------------
    ("CALIBRATION", "speed_real.json", "honest_bracket_on_the_along_road_scale/upper_m", "1dp"),
    ("CALIBRATION", "speed_real.json", "anchor_candidates", "len"),
    ("CALIBRATION", "speed_real.json", "anchor_candidates/0/matched_controls/0/spread_percent", "1dp"),
    ("CALIBRATION", "speed_real.json", "anchor_candidates/0/matched_controls/5/spread_percent", "1dp"),
    ("CALIBRATION", "speed_real.json", "divider_disagreement/ratio_range/0", "4dp"),
    ("CALIBRATION", "speed_real.json", "divider_disagreement/ratio_range/1", "4dp"),
    ("CALIBRATION", "speed_real.json", "what_this_clip_still_licenses/0", "text"),
    ("CALIBRATION", "speed_real.json", "what_this_clip_still_licenses/1", "text"),
    ("CALIBRATION", "speed_real.json", "what_this_clip_still_licenses/2", "text"),
    ("CALIBRATION", "speed_real.json", "what_this_clip_still_licenses/3", "text"),
    ("CALIBRATION", "speed_real.json", "what_this_clip_still_licenses/4", "text"),
    ("DESIGN_CARD", "speed_real.json", "honest_bracket_on_the_along_road_scale/lower_m", "1dp"),
    ("DESIGN_CARD", "speed_real.json", "honest_bracket_on_the_along_road_scale/band_percent/0", "int"),
    ("README", "speed_real.json", "honest_bracket_on_the_along_road_scale/upper_m", "1dp"),
    # -- detection_noise.json --------------------------------------------------
    ("README", "detection_noise.json", "residuals/box_width/std_px", "4dp"),
    ("README", "detection_noise.json", "residuals/box_width/p95_abs_px", "4dp"),
    ("README", "detection_noise.json", "residuals/centre_y/std_px", "4dp"),
    ("README", "detection_noise.json", "residuals/box_width/n", "int"),
    ("README", "detection_noise.json", "median_box_width_px", "2dp"),
    ("README", "detection_noise.json", "tracks_contributing", "int"),
    # -- parity.json -----------------------------------------------------------
]


@pytest.mark.parametrize(
    "entry", PINNED, ids=lambda entry: f"{entry[0]}|{entry[2]}"
)
def test_every_pinned_figure_appears_in_its_document(entry):
    """The load-bearing direction: a measurement that moves reddens its document.

    The expected string is computed from the committed report here, so nothing in
    this module states what any figure IS.
    """
    document_key, report, pointer, kind = entry
    value = _resolve(_read(report), pointer)
    expected = _render(value, kind)
    haystack = _document(document_key)
    if kind == "text":
        # A quoted sentence is wrapped by the document and may sit inside a
        # blockquote, so the comparison is over normalised text. Case is
        # normalised too: a quotation may legitimately open a sentence.
        found = _flatten(expected) in _flatten(haystack)
    else:
        found = _mentions_number(haystack, expected)
    assert found, (
        f"{DOCUMENTS[document_key].name} does not carry {expected!r}, which is "
        f"reports/{report}:{pointer} rendered as {kind}. Either the document is "
        f"stale or the report moved -- read the JSON, never an earlier draft."
    )


def _all_pins() -> list[tuple[str, str]]:
    """Every (document, report) pair either pinning mechanism covers.

    ``LABEL_PINS`` and ``CONTEXT_PINS`` are declared further down, beside the table parser it needs, so
    this is a function rather than a module-level constant.
    """
    return (
        [(key, report) for key, report, *_rest in PINNED]
        + [(key, report) for key, _label, _column, report, *_rest in LABEL_PINS]
        + [(key, report) for key, _phrase, report, *_rest in CONTEXT_PINS]
    )


def test_the_pinned_set_is_not_vacuously_small():
    pins = _all_pins()
    assert len(pins) >= MINIMUM_PINNED, (
        f"only {len(pins)} figures are pinned; expected at least "
        f"{MINIMUM_PINNED}. An emptied table satisfies every parametrised "
        f"assertion above by having nothing to run."
    )
    per_document = {key: 0 for key in DOCUMENTS}
    for document_key, _report in pins:
        per_document[document_key] += 1
    thin = {
        key: n for key, n in per_document.items() if n < MINIMUM_PINNED_PER_DOCUMENT
    }
    assert not thin, (
        f"these documents have almost nothing pinned in them: {thin}. A document "
        f"nothing points at is a document whose figures can drift freely."
    )


def test_every_report_contributes_a_pinned_figure():
    """A benchmark nobody quotes is a benchmark that can rot unnoticed.

    Set equality, so a new report file has to be published and pinned rather than
    silently ignored, and a pointer into a report that no longer exists fails
    here instead of erroring obscurely inside a parametrised case.
    """
    on_disk = {path.name for path in REPORTS_DIR.glob("*.json")}
    pinned = {report for _key, report in _all_pins()}
    assert pinned == on_disk, (
        "reports/ and the pinned set disagree; every report file must have at "
        "least one figure published and pinned"
    )


# --- 2. The figures the documents compute rather than copy ---------------------
#
# Each formula is written out here a second time, from the source reports, so a
# document's arithmetic is checked and not only its transcription. These are the
# ONLY numbers a document may print that no report contains, and the containment
# sweep below knows about exactly these.


def _derived_engine_cpu_multiple() -> float:
    timing = _read("counting_accuracy.json")["timing"]
    return timing["engine+gate"]["ms_per_frame"] / timing["centroid+gate"]["ms_per_frame"]


def _derived_one_event_f1() -> float:
    counting = _read("counting_accuracy.json")
    labels = counting["labels"]["total"]
    engine = counting["methods"]["engine+gate"]["full"]

    def f1(true_positives: int, predicted: int, truth: int) -> float:
        return 2.0 * true_positives / (predicted + truth)

    return abs(
        f1(engine["true_positives"], engine["n_predicted"], labels)
        - f1(engine["true_positives"], engine["n_predicted"] - 1, labels)
    )


def _derived_one_event_precision() -> float:
    return 1.0 / _read("counting_accuracy.json")["labels"]["total"]


def _derived_jitter_accuracy_lost_percent() -> float:
    """What sigma = 2 px costs, as a percentage of the undegraded score."""
    entries = _read("robustness.json")["protocols"]["box_jitter"]["entries"]
    identity = entries[0]["methods"]["engine+gate"]["f1"]
    sigma_2 = entries[2]["methods"]["engine+gate"]["f1"]
    return 100.0 * (1.0 - sigma_2 / identity)


def _derived_stress_at_sigma_2_lowest() -> float:
    stress = _read("robustness.json")["jitter_calibration"]["stress_multiple_at_max_sigma"]
    return stress["lowest"] * 2.0 / stress["sigma_px"]


def _derived_stress_at_sigma_2_highest() -> float:
    stress = _read("robustness.json")["jitter_calibration"]["stress_multiple_at_max_sigma"]
    return stress["highest"] * 2.0 / stress["sigma_px"]


def _derived_levels_the_engine_is_lowest_and_they_differ() -> float:
    """The published claim is an INTERSECTION, and the report publishes both sides.

    ``levels_where_the_engine_scores_lowest`` counts ties as lowest, so on its own
    it is larger than the number of levels where the comparison separates the
    trackers at all. Quoting either list's length alone would overstate or
    understate; the claim is "lowest wherever they differ", so the intersection is
    what the documents print and what is recomputed here.
    """
    separation = _read("robustness.json")["questions"]["tracker_separation"]
    lowest = set(separation["levels_where_the_engine_scores_lowest"])
    differ = set(separation["levels_where_trackers_differ"])
    return float(len(lowest & differ))


#: (document, name, formula, rendering).
DERIVED = [
    ("README", "engine tracker CPU against a baseline's", _derived_engine_cpu_multiple, "1dp"),
    ("README", "what one event is worth in F1", _derived_one_event_f1, "3dp"),
    ("README", "what one event is worth in precision", _derived_one_event_precision, "4dp"),
    ("README", "accuracy lost at sigma = 2 px, per cent", _derived_jitter_accuracy_lost_percent, "1dp"),
    ("README", "sigma = 2 px as a multiple of measured noise, low end", _derived_stress_at_sigma_2_lowest, "1dp"),
    ("README", "sigma = 2 px as a multiple of measured noise, high end", _derived_stress_at_sigma_2_highest, "1dp"),
    ("README", "levels where they differ and the engine is lowest",
     _derived_levels_the_engine_is_lowest_and_they_differ, "int"),
    ("DESIGN_CARD", "what one event is worth in F1", _derived_one_event_f1, "3dp"),
    ("DESIGN_CARD", "engine tracker CPU against a baseline's", _derived_engine_cpu_multiple, "1dp"),
    ("DESIGN_CARD", "levels where they differ and the engine is lowest",
     _derived_levels_the_engine_is_lowest_and_they_differ, "int"),
]


@pytest.mark.parametrize("entry", DERIVED, ids=lambda entry: f"{entry[0]}|{entry[1]}")
def test_every_derived_figure_recomputes_and_appears_in_its_document(entry):
    document_key, name, formula, kind = entry
    expected = _render(formula(), kind)
    assert _mentions_number(_document(document_key), expected), (
        f"{DOCUMENTS[document_key].name} does not carry {expected!r} for {name!r}, "
        f"recomputed here from the source reports"
    )


# --- 3. The containment sweep: no number a report does not support -------------

#: A markdown table's delimiter row, e.g. ``|---|---:|``.
_DELIMITER = re.compile(r"^\|[\s:|-]+\|\s*$")

#: The declaration every table carries. ``Source`` says which artefact the
#: numbers come from; a table sourced from a report must also state the protocol
#: that produced them, because a figure without its protocol is not a result.
_SOURCE_LINE = re.compile(r"^\*\*Source:\*\*\s+(.+)$")
_PROTOCOL_LINE = re.compile(r"^\*\*Protocol:\*\*\s+", re.M)
_REPORT_IN_SOURCE = re.compile(r"reports/([a-z_]+\.json)")

#: Numbers, including scientific notation. The lookbehind keeps the scan off
#: things that only look numeric because a number is embedded in a word or an
#: identifier -- ``AGPL-3.0``, ``yolo11s``, ``0_5px`` -- and off the tail of a
#: number that has already been matched.
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


class Table:
    """One markdown table, with the paragraph that declares it."""

    def __init__(self, document_key: str, preamble: list[str], rows: list[str]):
        self.document_key = document_key
        self.preamble = preamble
        self.rows = rows
        self.header = rows[0].strip()
        declarations = [
            match.group(1) for match in map(_SOURCE_LINE.match, preamble) if match
        ]
        self.source = declarations[-1] if declarations else None
        report = _REPORT_IN_SOURCE.search(self.source or "")
        self.report = report.group(1) if report else None

    @property
    def text(self) -> str:
        """What the sweep reads: the declaration and the table together.

        The declaration is in scope on purpose -- a protocol sentence carries
        figures too (a confidence threshold, a frame window), and they are as
        publishable and as capable of drifting as a cell.
        """
        return "\n".join(self.preamble + self.rows)


def _tables(document_key: str) -> list[Table]:
    lines = _document(document_key).splitlines()
    tables = []
    index = 0
    while index < len(lines) - 1:
        if lines[index].startswith("|") and _DELIMITER.match(lines[index + 1]):
            rows = []
            cursor = index
            while cursor < len(lines) and lines[cursor].startswith("|"):
                rows.append(lines[cursor])
                cursor += 1
            # The paragraph immediately above, blank lines skipped. Requiring the
            # declaration to be there rather than anywhere earlier is what stops
            # a second table free-riding on its neighbour's source line.
            end = index - 1
            while end >= 0 and not lines[end].strip():
                end -= 1
            start = end
            while start >= 0 and lines[start].strip():
                start -= 1
            tables.append(Table(document_key, lines[start + 1 : end + 1], rows))
            index = cursor
        else:
            index += 1
    return tables


#: Tables whose numbers are NOT from a ``reports/*.json``, with what they are
#: from. Enumerated rather than skipped by a rule, so a results table cannot slip
#: out of the sweep by losing its source line: the set equality below fails.
#:
#: The browser timings are the substantive entry. They have no report file -- they
#: come from a committed harness that drives a real headless Chrome, so they
#: cannot be regenerated by a Python test -- and this module therefore cannot
#: pin them. The documents state the command instead.
TABLES_NOT_FROM_REPORTS = [
    ("README", "| backend | ms per frame, median | fps | n | renderer |"),
    ("README", "| command | what it does |"),
    ("README", "| package | version range | what it is for |"),
    ("README", "| asset | licence | attribution and changes |"),
    ("README", "| document | what is in it |"),
    ("DESIGN_CARD", "| in scope | out of scope |"),
    ("ARCHITECTURE", "| module | what it owns |"),
    ("ARCHITECTURE", "| layer | guard | what it refuses |"),
]


def test_every_table_declares_where_its_numbers_come_from():
    """Exact set equality, in both directions.

    The must-be-declared half is what keeps the sweep honest: a new results table
    has to name the report it came from or fail here. The must-still-be-listed
    half is what stops the exception list growing: a table that starts being
    sourced from a report must come OFF this list, so a genuinely unsourced one
    cannot hide among entries that are now covered anyway.
    """
    undeclared = []
    not_from_reports = []
    for key in DOCUMENTS:
        for table in _tables(key):
            if table.source is None:
                undeclared.append((key, table.header))
            elif table.report is None:
                not_from_reports.append((key, table.header))
    assert undeclared == [], (
        f"these tables have no **Source:** line in the paragraph above them: "
        f"{undeclared}"
    )
    assert sorted(not_from_reports) == sorted(TABLES_NOT_FROM_REPORTS), (
        "the tables whose numbers do not come from a report changed. Add each "
        "new one here with what it does come from, or give it a reports/ source; "
        "remove any that is now sourced from a report."
    )


def test_every_report_sourced_table_states_its_protocol():
    """A figure without its protocol is not a result. Asserted per table."""
    missing = []
    for key in DOCUMENTS:
        for table in _tables(key):
            if table.report is None:
                continue
            if not _PROTOCOL_LINE.search("\n".join(table.preamble)):
                missing.append((key, table.header))
    assert missing == [], (
        f"these tables cite a report but state no protocol: {missing}"
    )


def test_each_document_carries_the_tables_it_owes():
    counts = {
        key: len([table for table in _tables(key) if table.report]) for key in DOCUMENTS
    }
    thin = {
        key: counts[key]
        for key, floor in MINIMUM_SOURCED_TABLES.items()
        if counts[key] < floor
    }
    assert not thin, (
        f"these documents lost report-sourced tables: {thin} against floors "
        f"{MINIMUM_SOURCED_TABLES}. The sweep below reads those tables, so a "
        f"document with none passes it without checking anything."
    )


def _scalars(value, out: list | None = None) -> list:
    out = [] if out is None else out
    if isinstance(value, dict):
        for item in value.values():
            _scalars(item, out)
    elif isinstance(value, list):
        for item in value:
            _scalars(item, out)
    else:
        out.append(value)
    return out


def _list_lengths(value, out: list | None = None) -> list:
    """How many things every list in the report holds.

    "Eight committed cases", "five candidate anchors", "four claims not made" are
    all counts of a list rather than a stored field, and they are as much the
    report's own measurement as any scalar in it.
    """
    out = [] if out is None else out
    if isinstance(value, dict):
        for item in value.values():
            _list_lengths(item, out)
    elif isinstance(value, list):
        out.append(len(value))
        for item in value:
            _list_lengths(item, out)
    return out


def _publishable(report: str, document_key: str) -> set[str]:
    """Every string a number in a table sourced from ``report`` may be.

    Built from the report's own scalars at each allowed precision, the length of
    every list in it, and the numbers inside its PROSE fields -- a measurement the
    benchmark recorded in a sentence rather than a field is still that report's
    measurement, and the scale investigation records several that way. Plus the
    declared derived figures for this document, which are the numbers the
    documents compute; each one is recomputed above.

    The redacted speed is removed at the end, so a number this project must not
    publish cannot be waved through merely because it is in a report.
    """
    allowed: set[str] = set()
    for value in _scalars(_read(report)):
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, str):
            for match in _NUMBER.finditer(value):
                allowed.add(match.group(0))
            continue
        for kind in _ROUNDINGS:
            if kind == "int" and float(value) != int(value):
                continue
            allowed.add(_render(value, kind))
    for length in _list_lengths(_read(report)):
        allowed.add(str(length))
    for key, _name, formula, kind in DERIVED:
        if key == document_key:
            allowed.add(_render(formula(), kind))
    forbidden = _resolve(_read(FORBIDDEN_SPEED_REPORT), FORBIDDEN_SPEED_POINTER)
    for kind in _ROUNDINGS:
        if kind == "int" and float(forbidden) != int(forbidden):
            continue
        allowed.discard(_render(forbidden, kind))
    return allowed


def test_every_number_in_a_results_table_occurs_in_its_own_report():
    """The other direction: a figure no report supports is as much a defect as a
    report figure the documents omit.

    This is what catches a number invented to fill a gap, a digit mistyped, and a
    figure lifted from the wrong benchmark.
    """
    problems = []
    swept = 0
    for key in DOCUMENTS:
        for table in _tables(key):
            if table.report is None:
                continue
            allowed = _publishable(table.report, key)
            found = _NUMBER.findall(table.text)
            assert found, (
                f"{key}: the table {table.header!r} cites {table.report} and "
                f"contains no numbers at all, so it is checking nothing"
            )
            swept += len(found)
            for token in found:
                if token not in allowed:
                    problems.append(f"{key} :: {table.header} :: {token}")
    assert problems == [], (
        f"{len(problems)} published numbers are not in the report their table "
        f"cites: {problems[:12]}"
    )
    assert swept >= MINIMUM_SWEPT_NUMBERS, (
        f"only {swept} numbers were swept; expected at least "
        f"{MINIMUM_SWEPT_NUMBERS}. Emptied tables pass the sweep above by "
        f"having nothing in them."
    )


# --- 4. The headline tables, cell by cell -------------------------------------
#
# The sweep says a number occurs somewhere in the report its table cites; the
# pinned set says a number occurs somewhere in the document. Neither can catch a
# figure taken from the WRONG ROW of the right report and printed in a document
# that legitimately carries that string elsewhere -- and the counting table's own
# rows differ from each other by exactly one event, which is the confusion most
# worth preventing.
#
# So the three tables the whole project's headline claims rest on are pinned cell
# by cell against the report, keyed by the row's own label. Row ORDER is not
# pinned: how a document orders its rows is the document's business, and a
# dictionary comparison reports a wrong cell rather than a shifted table.


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _table_by_header(document_key: str, header: str) -> Table:
    for table in _tables(document_key):
        if table.header == header:
            return table
    raise AssertionError(
        f"{DOCUMENTS[document_key].name} has no table with the header {header!r}; "
        f"this pin must not pass by absence"
    )


def _published_rows(document_key: str, header: str, key_columns: int) -> dict:
    table = _table_by_header(document_key, header)
    rows = {}
    for row in table.rows[2:]:
        cells = _cells(row)
        key = tuple(cells[:key_columns])
        assert key not in rows, f"duplicate row {key} in {header}"
        rows[key] = tuple(cells[key_columns:])
    assert rows, f"{header} in {document_key} has no body rows to check"
    return rows


def _expected_counting_rows(subset: str) -> dict:
    methods = _read("counting_accuracy.json")["methods"]
    return {
        (f"`{name}`",): (
            _render(method[subset]["precision"], "3dp"),
            _render(method[subset]["recall"], "3dp"),
            _render(method[subset]["f1"], "3dp"),
            _render(method[subset]["n_predicted"], "int"),
            _render(method[subset]["true_positives"], "int"),
            _render(method[subset]["false_positives"], "len"),
            _render(method[subset]["misses"], "len"),
        )
        for name, method in methods.items()
    }


def _expected_robustness_rows() -> dict:
    robustness = _read("robustness.json")
    separation = robustness["questions"]["tracker_separation"]
    expected = {}
    for name, protocol in robustness["protocols"].items():
        for entry in protocol["entries"]:
            level = entry["level_label"]
            by_tracker = separation["f1_by_level"][f"{name}@{level}"]
            expected[(level,)] = (
                _render(by_tracker["engine+gate"], "3dp"),
                _render(by_tracker["centroid+gate"], "3dp"),
                _render(by_tracker["greedy-iou+gate"], "3dp"),
                _render(separation["f1_spread_by_level"][f"{name}@{level}"], "4dp"),
                _render(entry["methods"]["engine+gate"]["n_predicted"], "int"),
            )
    return expected


#: Rows of the form "| what it is | the figure |", pinned by their label:
#: (document, row label, which cell after the label, report, pointer, rendering).
#:
#: These exist because a pin on a short figure is a weak pin. "8 committed parity
#: cases" renders as "8", and a document of this length prints an 8 somewhere
#: whatever the report says -- measured: moving caseCount from 8 to 9 left the
#: plain pin green. Anchoring the figure to its own row's label removes the
#: coincidence.
LABEL_PINS = [
    ("README", "committed parity cases", 1, "parity.json", "caseCount", "int"),
    ("README", "boundary kinds every case must straddle", 1, "parity.json", "straddleKinds", "len"),
    ("README", "speed agreement tolerance, km/h", 1, "parity.json", "speedToleranceKmh", "g"),
    ("README", "association floor the fixtures straddle", 1, "parity.json", "iouStraddle/matchThresh", "1dp"),
    ("README", "Mahalanobis gate the fixtures straddle", 1, "parity.json", "iouStraddle/gatingChi2", "4dp"),
    ("README", "frames replayed from the real clip", 1, "parity.json", "realClip/frames", "int"),
    ("README", "detections in them", 1, "parity.json", "realClip/detections", "int"),
    ("README", "track identities allocated", 1, "parity.json", "realClip/tracksAllocated", "int"),
    ("README", "crossings emitted", 1, "parity.json", "realClip/events", "int"),
    ("ARCHITECTURE", "committed parity cases", 1, "parity.json", "caseCount", "int"),
    ("ARCHITECTURE", "boundary kinds every case must straddle", 1, "parity.json", "straddleKinds", "len"),
    ("ARCHITECTURE", "speed agreement tolerance, km/h", 1, "parity.json", "speedToleranceKmh", "g"),
    ("ARCHITECTURE", "association floor the fixtures straddle", 1, "parity.json", "iouStraddle/matchThresh", "1dp"),
    ("ARCHITECTURE", "Mahalanobis gate the fixtures straddle", 1, "parity.json", "iouStraddle/gatingChi2", "4dp"),
    ("ARCHITECTURE", "frames replayed from the real clip", 1, "parity.json", "realClip/frames", "int"),
    ("ARCHITECTURE", "detections in them", 1, "parity.json", "realClip/detections", "int"),
    ("ARCHITECTURE", "track identities allocated", 1, "parity.json", "realClip/tracksAllocated", "int"),
    ("CALIBRATION", "assumed along-road period, m", 1, "speed_real.json",
     "divider_disagreement/assumed_period_m", "1dp"),
    ("CALIBRATION", "lower end of the honest bracket, m", 1, "speed_real.json",
     "honest_bracket_on_the_along_road_scale/lower_m", "1dp"),
    ("CALIBRATION", "implied band on every speed, %", 1, "speed_real.json",
     "honest_bracket_on_the_along_road_scale/band_percent/0", "int"),
    ("CALIBRATION", "tracked columns", 1, "speed_real.json",
     "the_one_clean_measurement/tracked_columns", "int"),
    ("CALIBRATION", "robust weighted residual RMS, px", 1, "speed_real.json",
     "the_one_clean_measurement/robust_weighted_residual_rms_px", "3dp"),
    ("CALIBRATION", "plain residual RMS, px", 1, "speed_real.json",
     "the_one_clean_measurement/plain_residual_rms_px", "3dp"),
    ("CALIBRATION", "worst plain residual, px", 1, "speed_real.json",
     "the_one_clean_measurement/plain_residual_max_px", "3dp"),
    ("CALIBRATION", "columns within 0.5 px of the fit", 1, "speed_real.json",
     "the_one_clean_measurement/inliers_within_0_5px", "int"),
    ("CALIBRATION", "agreement with the surveyed vanishing point, px", 1, "speed_real.json",
     "the_one_clean_measurement/parallel_to_the_road/agreement_px", "3dp"),
    ("CALIBRATION", "self-fit, against its own correspondences", 1, "speed_real.json",
     "why_the_calibration_was_removed_rather_than_refitted/"
     "removed_calibration_residuals_for_the_record/self_fit_mean_m", "4dp"),
    ("CALIBRATION", "self-fit, against its own correspondences", 2, "speed_real.json",
     "why_the_calibration_was_removed_rather_than_refitted/"
     "removed_calibration_residuals_for_the_record/self_fit_max_m", "4dp"),
    ("CALIBRATION", "holdout, against points kept out of the fit", 1, "speed_real.json",
     "why_the_calibration_was_removed_rather_than_refitted/"
     "removed_calibration_residuals_for_the_record/holdout_mean_m", "4dp"),
    ("CALIBRATION", "holdout, against points kept out of the fit", 2, "speed_real.json",
     "why_the_calibration_was_removed_rather_than_refitted/"
     "removed_calibration_residuals_for_the_record/holdout_max_m", "4dp"),
    ("DESIGN_CARD", "labelled crossings", 1, "counting_accuracy.json", "labels/total", "int"),
    ("DESIGN_CARD", "of those, adjudicated certain", 1, "counting_accuracy.json",
     "labels/certain", "int"),
    ("DESIGN_CARD", "of those, adjudicated probable", 1, "counting_accuracy.json",
     "labels/probable", "int"),
]

#: Prose figures pinned to the phrase that gives them their meaning:
#: (document, phrase with ``{}`` where the figure goes, report, pointer, rendering).
#:
#: The phrase is matched literally apart from the figure, over whitespace-collapsed
#: text so markdown wrapping does not matter, and the figure keeps the same
#: standalone-number boundaries ``_mentions_number`` applies. This is what a table
#: row's label does for a table, done for a sentence -- and it exists because a bare
#: presence pin on a short figure is satisfiable by coincidence: each pin below was
#: measured surviving a real one-step move of its own report field.
CONTEXT_PINS = [
    ("README", "{} adjudicated certain and", "counting_accuracy.json", "labels/certain", "int"),
    ("README", "adjudicated certain and {} probable", "counting_accuracy.json",
     "labels/probable", "int"),
    ("README", "over frames 0 to {} of", "counting_accuracy.json", "window/end_frame", "int"),
    ("README", "{} predictions move to the ignore set", "counting_accuracy.json",
     "methods/engine+gate/certain_only/predictions_moved_to_ignore", "int"),
    ("README", "The three trackers differ at {} of the 21 levels", "robustness.json",
     "questions/tracker_separation/levels_where_trackers_differ", "len"),
    ("README", "differ at 19 of the {} levels", "robustness.json",
     "questions/tracker_separation/levels_measured", "int"),
    ("README", "{} differ by more than one event", "robustness.json",
     "questions/tracker_separation/levels_where_trackers_differ", "differ_above_one_event"),
    ("README", "the highest frame rate at which it happens is {} fps", "robustness.json",
     "questions/band_step_over/first_rate_with_step_over", "int"),
    ("README", "a band of half-width {} px about the gate segment", "tracking.json",
     "gate_region/half_width_px", "int"),
    ("README", "The two metrics agree at {} of the 19 levels", "tracking.json",
     "questions/agreement_with_crossing_f1/levels_where_the_two_agree", "len"),
    ("README", "against a {} km/h requirement", "speed_synthetic.json",
     "check_c/agreement_requirement_kmh", "1dp"),
    ("README", "scored against {} hand-labelled crossings", "counting_accuracy.json",
     "labels/total", "int"),
    ("README", "per method over the {} decoded frames", "counting_accuracy.json",
     "frames", "int"),
    ("README", "{} of 48 tracks contributing", "detection_noise.json",
     "tracks_contributing", "int"),
    ("CALIBRATION", "{} candidate anchors were measured", "speed_real.json",
     "anchor_candidates", "len"),
    ("DESIGN_CARD", "record carries {} refusals", "tracking.json", "claims_not_made", "len"),
    ("DESIGN_CARD", "at all **{}** levels where they\ndiffer at all", "robustness.json",
     "questions/tracker_separation/levels_where_trackers_differ", "len"),
    ("DESIGN_CARD", "lowest at all **{}** levels where they differ by more", "robustness.json",
     "questions/tracker_separation/levels_where_trackers_differ", "differ_above_one_event"),
]


def _levels_above_one_event() -> int:
    """How many of the levels where the trackers differ differ by MORE than one event.

    The published claim the resolution rule licenses. Computed here from both
    reports -- the step from ``counting_accuracy.json``, the spreads from
    ``robustness.json`` -- so it cannot be typed.
    """
    separation = _read("robustness.json")["questions"]["tracker_separation"]
    step = _derived_one_event_f1()
    differ = set(separation["levels_where_trackers_differ"])
    lowest = set(separation["levels_where_the_engine_scores_lowest"])
    spreads = separation["f1_spread_by_level"]
    return len([key for key in differ & lowest if spreads[key] > step])


@pytest.mark.parametrize(
    "entry", CONTEXT_PINS, ids=lambda entry: f"{entry[0]}|{entry[3]}|{entry[1][:28]}"
)
def test_every_prose_figure_appears_inside_its_own_phrase(entry):
    document_key, phrase, report, pointer, kind = entry
    if kind == "differ_above_one_event":
        # A count over two reports rather than a stored field; the pointer names the
        # list it is a subset of, so a moved list still reddens this pin.
        expected = str(_levels_above_one_event())
    else:
        expected = _render(_resolve(_read(report), pointer), kind)
    before, _, after = phrase.partition("{}")
    assert after or before, f"{phrase!r} has no literal context around the figure"
    pattern = (
        re.escape(_collapse(before))
        + rf"(?<![\d.]){re.escape(expected)}(?!\d)(?!\.\d)"
        + re.escape(_collapse(after))
    )
    haystack = _collapse(_document(document_key))
    assert re.search(pattern, haystack), (
        f"{DOCUMENTS[document_key].name} does not contain "
        f"{_collapse(phrase.replace('{}', expected))!r}, which is "
        f"reports/{report}:{pointer} in the sentence that gives it its meaning"
    )


def _labelled_cell(document_key: str, label: str, column: int) -> str:
    """The cell ``column`` places right of the row whose first cell is ``label``.

    Searched across every table in the document, with exactly one row allowed to
    carry the label, so a duplicated label is a failure rather than an ambiguity
    resolved by whichever table came first.
    """
    matches = [
        _cells(row)
        for table in _tables(document_key)
        for row in table.rows[2:]
        if _cells(row)[0] == label
    ]
    assert len(matches) == 1, (
        f"{DOCUMENTS[document_key].name} has {len(matches)} rows labelled {label!r}; "
        f"this pin needs exactly one"
    )
    return matches[0][column]


@pytest.mark.parametrize(
    "entry", LABEL_PINS, ids=lambda entry: f"{entry[0]}|{entry[1]}|{entry[2]}"
)
def test_every_labelled_row_carries_its_measured_value(entry):
    document_key, label, column, report, pointer, kind = entry
    expected = _render(_resolve(_read(report), pointer), kind)
    published = _labelled_cell(document_key, label, column)
    assert published == expected, (
        f"{DOCUMENTS[document_key].name}, row {label!r}, column {column}: prints "
        f"{published!r} but reports/{report}:{pointer} is {expected!r}"
    )


def test_the_counting_table_is_pinned_cell_by_cell():
    """Both counting tables: every method, every column, against the report.

    The two tables share a header, so they are told apart by which subset their
    cells actually match -- and a document carrying only one of them fails, because
    each expected set has to be found.
    """
    header = "| method | precision | recall | F1 | predicted | matched | false positives | missed |"
    tables = [table for table in _tables("README") if table.header == header]
    assert len(tables) == 2, (
        f"expected the full-set and certain-only counting tables under one header; "
        f"found {len(tables)}"
    )
    published = []
    for table in tables:
        rows = {}
        for row in table.rows[2:]:
            cells = _cells(row)
            rows[(cells[0],)] = tuple(cells[1:])
        assert rows, "a counting table has no body rows to check"
        published.append(rows)

    for subset in ("full", "certain_only"):
        expected = _expected_counting_rows(subset)
        assert expected in published, (
            f"no counting table in README.md matches the {subset} figures in "
            f"reports/counting_accuracy.json cell for cell.\nexpected: {expected}\n"
            f"found: {published}"
        )
    assert published[0] != published[1], (
        "both counting tables carry identical cells, so one of them is not the "
        "subset it claims to be"
    )


def test_the_degradation_table_is_pinned_cell_by_cell():
    header = (
        "| degradation | level | engine+gate | centroid+gate | greedy-iou+gate | "
        "spread | engine predicted |"
    )
    published = _published_rows("README", header, 2)
    # The protocol column is prose and is not pinned; level labels are unique
    # across the four protocols, so the level alone identifies a row.
    by_level = {(key[1],): value for key, value in published.items()}
    assert by_level == _expected_robustness_rows(), (
        "README.md's degradation table disagrees with reports/robustness.json"
    )


# --- 4b. No pin may be satisfiable by a coincidence ---------------------------
#
# The hole mechanism 1 has by construction: it asks whether a figure is printed
# anywhere in the document, and in a document of this length a short figure's
# neighbour is usually printed somewhere too. So a real report move to that
# neighbouring value leaves the pin green.
#
# This is not theory. Twelve pins over nine report pointers were measured surviving a
# one-step move of their own report field -- `labels/certain` 7, `labels/probable` 10,
# `window/end_frame` 734, `predictions_moved_to_ignore` 10,
# `levels_where_trackers_differ` 19, `levels_measured` 21,
# `first_rate_with_step_over` 5, `gate_region/half_width_px` 20,
# `levels_where_the_two_agree` 18 and `check_c/agreement_requirement_kmh` 1.0. The
# design card would have kept printing "adjudicated certain | 7" against a report
# saying 6.
#
# The rule adopted: a numeric pin whose one-step neighbour is already in its document
# must ALSO be covered by something exact. The exemptions are not a hand-written list
# -- they are derived from the mechanisms themselves, so an exemption cannot be
# claimed for coverage that does not exist.


def _one_step(value, kind: str) -> list[str] | None:
    """The renderings one step either side of ``value`` in its own last digit.

    ``None`` for the scientific and exact renderings, where "one step" is not a
    transcription a reader could confuse: those figures are either label-pinned or
    have no neighbour a document of this kind would print.
    """
    if kind == "len":
        length = len(value)
        return [str(length - 1), str(length + 1)]
    if kind == "int":
        whole = int(value)
        return [str(whole - 1), str(whole + 1)]
    if kind.endswith("dp"):
        places = int(kind[:-2])
        step = 10.0**-places
        return [_render(value - step, kind), _render(value + step, kind)]
    return None


def _cell_pinned_pointers(document_key: str) -> set[str]:
    """Pointers the cell-by-cell table tests compare, built the way those tests build
    their expected rows -- from the reports, not from a list kept in step by hand.

    Every pointer returned is resolved here, so a pointer that has stopped existing
    cannot go on granting an exemption.
    """
    if document_key != "README":
        return set()
    pointers = set()
    counting = _read("counting_accuracy.json")
    for name in counting["methods"]:
        for subset in ("full", "certain_only"):
            for field in ("precision", "recall", "f1", "n_predicted",
                          "true_positives", "false_positives", "misses"):
                pointers.add(f"methods/{name}/{subset}/{field}")
    robustness = _read("robustness.json")
    for name, protocol in robustness["protocols"].items():
        for index, entry in enumerate(protocol["entries"]):
            level = entry["level_label"]
            pointers.add(f"protocols/{name}/entries/{index}/methods/engine+gate/n_predicted")
            pointers.add(f"questions/tracker_separation/f1_by_level/{name}@{level}")
            pointers.add(f"questions/tracker_separation/f1_spread_by_level/{name}@{level}")
    for pointer in pointers:
        report = ("counting_accuracy.json" if pointer.startswith("methods/")
                  else "robustness.json")
        _resolve(_read(report), pointer)  # raises if the pointer has gone
    return pointers


def _exactly_pinned(document_key: str) -> set[str]:
    return (
        {pointer for key, _label, _column, _report, pointer, _kind in LABEL_PINS
         if key == document_key}
        | {pointer for key, _phrase, _report, pointer, _kind in CONTEXT_PINS
           if key == document_key}
        | _cell_pinned_pointers(document_key)
    )


def _coincidence_prone(document_key: str, report: str, pointer: str, kind: str) -> list[str]:
    neighbours = _one_step(_resolve(_read(report), pointer), kind)
    if neighbours is None:
        return []
    text = _document(document_key)
    return [rendering for rendering in neighbours if _mentions_number(text, rendering)]


def test_no_numeric_pin_can_be_satisfied_by_a_coincidence():
    """Every pin whose neighbour the document already prints must be exactly pinned."""
    examined = 0
    prone = 0
    unbackstopped = []
    for document_key, report, pointer, kind in PINNED:
        if kind == "text":
            continue
        collisions = _coincidence_prone(document_key, report, pointer, kind)
        if _one_step(_resolve(_read(report), pointer), kind) is not None:
            examined += 1
        if not collisions:
            continue
        prone += 1
        if pointer not in _exactly_pinned(document_key):
            unbackstopped.append(
                f"{document_key} :: {pointer} = "
                f"{_render(_resolve(_read(report), pointer), kind)} :: a report move to "
                f"{collisions} would leave this pin green"
            )
    assert unbackstopped == [], (
        f"{len(unbackstopped)} numeric pins could be satisfied by a number the "
        f"document already contains. Add a CONTEXT_PINS phrase or a LABEL_PINS row "
        f"for each:\n" + "\n".join(unbackstopped)
    )
    assert examined >= 60, (
        f"only {examined} numeric pins were examined; this test would be passing over "
        f"almost nothing"
    )
    assert prone >= 10, (
        f"only {prone} pins were found coincidence-prone. That is suspiciously few for "
        f"documents this long -- check _one_step and _mentions_number still work "
        f"rather than assuming the documents improved"
    )


def test_the_coincidence_check_fires_on_a_pin_that_is_not_exactly_covered():
    """The control on the test above, varying one axis: whether the pin is exactly
    covered. Without it, a broken ``_one_step`` or ``_mentions_number`` would make the
    assertion pass by finding no collisions at all.

    ``labels/certain`` is the reviewer's own example: the README prints both 6 and 8,
    so a bare presence pin on 7 survives a report saying either.
    """
    collisions = _coincidence_prone("README", "counting_accuracy.json", "labels/certain", "int")
    assert collisions, (
        "the collision detector no longer fires on the pin that motivated it"
    )
    # And the must-survive half: a figure whose neighbours the document does not print.
    assert _coincidence_prone(
        "README", "counting_accuracy.json", "methods/engine+gate/full/f1", "3dp"
    ) == [], "0.913/0.915 should not be in README.md; the detector is over-firing"
    # The exemption is real rather than asserted: this pointer IS exactly covered.
    assert "labels/certain" in _exactly_pinned("README")


# --- 5. The redaction ---------------------------------------------------------


def test_no_document_carries_a_speed_for_the_uncalibrated_clip():
    """A discriminating pair.

    The must-not-publish half is the rule. The must-still-exist half is what
    proves the guard is live: the figure is in reports/ for the cross-surface
    comparison's own purposes, and if it were simply gone this test would be
    passing over nothing.
    """
    parity = _read(FORBIDDEN_SPEED_REPORT)
    assert "maxSpeedKmh" in parity["realClip"], (
        "the redacted field is no longer in reports/parity.json, so this guard "
        "is passing over nothing; re-derive the redaction"
    )
    forbidden = _resolve(parity, FORBIDDEN_SPEED_POINTER)
    for key in DOCUMENTS:
        text = _document(key)
        assert "maxSpeedKmh" not in text, key
        # The digits alone, not only the exact literal: it is the FIGURE that is
        # prohibited and not one spelling of it.
        assert f"{forbidden:.2f}"[:6] not in text, key
        assert f"{forbidden:.1f}" not in text, key
