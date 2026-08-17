"""The mutation battery's own credibility test.

``scripts/mutation_battery.py`` decides whether this project's published claims
are defended. Its report is only worth reading if the instrument can be shown to
FAIL when it should -- because "0 survivors because everything is protected" and
"0 survivors because nothing was mutated" print the same sentence, and this
repository has met the second thing five times in mutation harnesses alone.

So the load-bearing test here is not that the real table passes. It is that the
battery, handed a claim with genuinely no test behind it, exits 1 and NAMES that
claim; and that handed a stale anchor it errors instead of quietly reporting the
claim as protected.

Everything is driven against a throwaway git repository built in ``tmp_path``,
holding a two-function "engine" with one defended rule and one undefended one.
That is deliberate. Proving the battery can report a survivor by pointing it at
permanently-unprotected code in THIS repository would mean shipping
permanently-unprotected code, and the real table would then exit 1 forever. The
sandbox gives the proof without the hostage.

The one thing measured against the real table is a set of floors: how many rows
it has, that it carries must-survive controls and covers the claim families the
task requires, and that every row's anchor still resolves uniquely. Those exist
so the table cannot quietly shrink to something that passes by covering nothing.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "scripts" / "mutation_battery.py"


def _load_battery():
    """Import the battery by path: ``scripts/`` is not a package.

    Registered in ``sys.modules`` before execution because the module defines
    dataclasses, which resolve their own annotations through the module entry.
    """
    spec = importlib.util.spec_from_file_location("_mutation_battery", BATTERY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENGINE = '''"""A two-rule engine: one rule has a test, the other has none."""


def defended_sum(a, b):
    return a + b


def undefended_product(a, b):
    return a * b
'''

ENGINE_TEST = '''from engine import defended_sum


def test_defended_sum_adds():
    assert defended_sum(2, 3) == 5
'''

#: The mutation that breaks the DEFENDED rule. Its named test must go red.
DEFENDED = {
    "claim": "the defended rule adds its arguments",
    "path": "engine.py",
    "find": "    return a + b",
    "replace": "    return a - b",
    "runner": "pytest",
    "test": "test_engine.py::test_defended_sum_adds",
    "expect": "must_fail",
    "note": "subtracts instead of adding",
}

#: The canary: a mutation to code no test touches, whose row nonetheless claims
#: the defended rule's test defends it. This MUST be reported as a survivor.
CANARY = {
    "claim": "the undefended rule multiplies its arguments",
    "path": "engine.py",
    "find": "    return a * b",
    "replace": "    return a / b",
    "runner": "pytest",
    "test": "test_engine.py::test_defended_sum_adds",
    "expect": "must_fail",
    "note": "divides instead of multiplying, and nothing looks",
}


@pytest.fixture
def sandbox(tmp_path) -> Path:
    """A committed throwaway repository holding the two-rule engine."""
    (tmp_path / "engine.py").write_text(ENGINE)
    (tmp_path / "test_engine.py").write_text(ENGINE_TEST)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    # -f so a developer's global excludesFile cannot empty the fixture, and an
    # explicit identity so a machine with none configured still commits.
    subprocess.run(["git", "add", "-f", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", "sandbox"],
        cwd=tmp_path, check=True,
    )
    assert _porcelain(tmp_path) == "", "the sandbox did not start clean"
    return tmp_path


def _porcelain(root: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _run(sandbox: Path, rows: list[dict]) -> subprocess.CompletedProcess:
    # The table lives OUTSIDE the sandbox repository: an untracked file inside it
    # would make the tree dirty, which the battery is meant to refuse, and every
    # test here would then measure that refusal instead of what it means to.
    table = sandbox.parent / "table.json"
    table.write_text(json.dumps(rows))
    return subprocess.run(
        [sys.executable, str(BATTERY), "--root", str(sandbox), "--table", str(table)],
        capture_output=True, text=True,
    )


# --- the load-bearing tests ---------------------------------------------------


def test_the_battery_reports_a_survivor_and_names_the_claim(sandbox):
    """THE test of this whole task.

    A claim whose named test does not notice the mutation must exit 1 and print
    the claim, the file, the byte change and the test that stayed green -- so a
    reader can act on the finding rather than being told a number.
    """
    result = _run(sandbox, [CANARY])

    assert result.returncode == 1, (
        f"the battery exited {result.returncode} on an undefended claim; a "
        f"survivor must be a non-zero exit:\n{result.stdout}\n{result.stderr}"
    )
    assert "SURVIVORS" in result.stdout
    assert CANARY["claim"] in result.stdout, (
        f"the battery did not name the unprotected claim:\n{result.stdout}"
    )
    assert CANARY["test"] in result.stdout, "the report does not name the test that stayed green"
    assert "bytes" in result.stdout, "the report carries no byte-change proof"


def test_the_battery_errors_on_a_stale_anchor_instead_of_reporting_protection(sandbox):
    """A ``find`` that is not in the file must be a hard error, not a pass.

    This is the exact defect five earlier harnesses in this project shipped: a
    bare ``str.replace`` that matched nothing ran green while mutating nothing.
    The file must also be left untouched -- an error is not a licence to write.
    """
    before = (sandbox / "engine.py").read_bytes()
    stale = dict(DEFENDED, find="    return a + b + 1")

    result = _run(sandbox, [stale])

    assert result.returncode == 2, (
        f"a stale anchor exited {result.returncode}; it must be a hard error:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "NOT PRESENT" in result.stderr
    assert stale["claim"] in result.stderr, "the error does not name the claim"
    assert "SURVIVORS" not in result.stdout
    assert (sandbox / "engine.py").read_bytes() == before
    assert _porcelain(sandbox) == ""


def test_the_battery_reports_no_survivor_when_the_claim_is_defended(sandbox):
    """The other half of the pair.

    Without this, a battery that always exited 1 would pass the survivor test
    above while being useless. The clean run must be reachable.
    """
    result = _run(sandbox, [DEFENDED])

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "SURVIVORS" not in result.stdout
    assert "PASSED" in result.stdout


def test_the_battery_refuses_to_start_on_a_dirty_tree(sandbox):
    """It edits tracked files in place; a crash mid-run must not be able to
    take uncommitted work with it."""
    (sandbox / "engine.py").write_text(ENGINE + "\n# an uncommitted edit\n")
    dirty_bytes = (sandbox / "engine.py").read_bytes()

    result = _run(sandbox, [DEFENDED])

    assert result.returncode == 2
    assert "REFUSING TO RUN" in result.stderr
    # The uncommitted edit is still exactly where it was.
    assert (sandbox / "engine.py").read_bytes() == dirty_bytes


def test_the_battery_errors_when_the_named_test_matches_nothing(sandbox):
    """A node id that collects nothing would make every mutation look caught.

    pytest exits non-zero when a node id does not resolve, which is
    indistinguishable from a failing test unless the battery checks -- so it
    checks, before mutating anything.
    """
    result = _run(sandbox, [dict(DEFENDED, test="test_engine.py::test_absent")])

    assert result.returncode == 2, f"{result.stdout}\n{result.stderr}"
    assert "HARD ERROR" in result.stderr
    assert "SURVIVORS" not in result.stdout


def test_the_battery_errors_on_an_ambiguous_anchor(sandbox):
    """An anchor matching twice mutates more than its row describes."""
    result = _run(sandbox, [dict(DEFENDED, find="def ")])

    assert result.returncode == 2
    assert "occurs 2 times" in result.stderr


def test_the_battery_leaves_the_tree_byte_identical(sandbox):
    """Restoration is from a saved byte copy, and it is checked.

    Run several rows, including one whose mutation is caught and one whose
    mutation is not, then compare the whole repository.
    """
    before = (sandbox / "engine.py").read_bytes()

    result = _run(sandbox, [DEFENDED, CANARY, DEFENDED])

    assert result.returncode == 1  # the canary survives, as it must
    assert (sandbox / "engine.py").read_bytes() == before
    assert _porcelain(sandbox) == ""


def test_the_battery_flags_a_must_survive_control_that_reddens(sandbox):
    """A control is an assertion too.

    A semantically-equivalent mutation that DOES redden its test means the test
    is pinning a spelling rather than a claim, and that must be reported rather
    than passed over.
    """
    result = _run(sandbox, [dict(DEFENDED, expect="must_survive")])

    assert result.returncode == 1
    assert "CONTROLS THAT REDDENED" in result.stdout
    assert DEFENDED["claim"] in result.stdout


def test_the_battery_flags_a_known_open_finding_that_has_been_closed(sandbox):
    """``known_open`` must bite in the other direction.

    A row documenting an open finding asserts the test stays green. If the
    finding is fixed, the row is wrong and must be promoted, not left claiming a
    hole that no longer exists -- otherwise "known open" would be an
    unfalsifiable exemption.
    """
    result = _run(sandbox, [dict(DEFENDED, expect="known_open")])

    assert result.returncode == 1
    assert "NOW CAUGHT" in result.stdout
    assert DEFENDED["claim"] in result.stdout


def test_the_battery_accepts_a_genuinely_open_known_finding(sandbox):
    """And the must-survive half of that pair: a still-open finding is not a
    failure, and is reported as a finding."""
    result = _run(sandbox, [dict(CANARY, expect="known_open")])

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "KNOWN-OPEN FINDINGS" in result.stdout
    assert CANARY["claim"] in result.stdout


# --- floors on the real table -------------------------------------------------
#
# No mutation is applied here. These are the assertions that stop the real table
# shrinking into something that passes by covering nothing.


#: Every claim family the task requires the table to cover, keyed by a fragment
#: that must appear in some row's claim string. Written as fragments rather than
#: whole claim names so a reworded row does not need this list touched, while a
#: DELETED family does.
REQUIRED_CLAIM_FAMILIES = [
    "bounded to the drawn segment",
    "once-per-track",
    "on-line deferral",
    "last OFF-LINE POINT",
    "uncalibrated",
    "second association stage",
    "Mahalanobis",
    "crosses classes",
    "reaping boundary",
    "determinism",
    "component-wise least squares",
    "SPEED_MAX_STEP_M",
    "homography validation",
    "4-point fit with no holdout",
    "constants-sync",
    "parity",
    "docs-numbers pin",
    "counting report",
    "robustness report",
    "tracking report",
    "PROTOCOL.md",
    "timing guard",
    "banned words",
    "absolute user path",
    "check-ignore",
    "process-doc",
    "vendored exclusion",
]

MINIMUM_ROWS = 45
MINIMUM_CONTROLS = 6
MINIMUM_MUST_FAIL = 35


def test_the_real_table_covers_every_claim_family_and_is_not_vacuously_small():
    battery = _load_battery()
    claims = battery.CLAIMS

    assert len(claims) >= MINIMUM_ROWS, (
        f"the table has only {len(claims)} rows; a battery this small is not "
        f"covering the published claims"
    )
    must_fail = [c for c in claims if c.expect == battery.MUST_FAIL]
    controls = [c for c in claims if c.expect == battery.MUST_SURVIVE]
    assert len(must_fail) >= MINIMUM_MUST_FAIL, len(must_fail)
    assert len(controls) >= MINIMUM_CONTROLS, (
        f"only {len(controls)} must-survive controls; a table of must-fails "
        f"alone proves the tests fire but not that they discriminate"
    )
    # Every control has to say which axis it varies, or it is not a control.
    for control in controls:
        assert "axis:" in control.note, control.claim
    # Every documented open finding has to say where it is routed.
    for known in [c for c in claims if c.expect == battery.KNOWN_OPEN]:
        assert "notes section 5" in known.note, known.claim

    text = " || ".join(c.claim for c in claims)
    missing = [family for family in REQUIRED_CLAIM_FAMILIES if family not in text]
    assert missing == [], f"the table no longer covers: {missing}"


def test_every_row_of_the_real_table_still_resolves_to_a_unique_anchor():
    """A refactor that moves a mutated line must fail HERE, in the suite, rather
    than only when someone next remembers to run the battery."""
    battery = _load_battery()
    battery.preflight_anchors(ROOT, battery.CLAIMS)


def test_every_named_test_in_the_real_table_exists():
    """Cheap, mutation-free existence check on the other half of every row.

    The battery itself refuses to run a row whose named test collects nothing,
    but that only helps someone who runs the battery. This catches a renamed or
    deleted test in the ordinary suite.
    """
    battery = _load_battery()
    pytest_rows = [c for c in battery.CLAIMS if c.runner == "pytest"]
    assert len(pytest_rows) >= 40, len(pytest_rows)

    missing = []
    for node_id in sorted({c.test for c in pytest_rows}):
        file_part, _, name = node_id.partition("::")
        source = (ROOT / file_part).read_text()
        if f"def {name}(" not in source:
            missing.append(node_id)
    assert missing == [], f"the table names tests that no longer exist: {missing}"
