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
import os
import py_compile
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

ENGINE_TEST = '''import pytest

from engine import defended_sum


def test_defended_sum_adds():
    assert defended_sum(2, 3) == 5


@pytest.mark.parametrize("case", [])
def test_defended_sum_over_cases(case):
    """Collected, never run.

    pytest turns an empty parameter set into a single ``[NOTSET]`` placeholder,
    prints "1 test collected", skips it and exits 0 -- so a battery row naming
    this id has a test that cannot go red. ``NEVER_RUNS`` points at it.
    """
    assert defended_sum(*case) == 0
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

#: A row whose named test pytest COLLECTS and never RUNS. Not a survivor and not
#: a stale anchor: an instrument fault, which the battery must refuse to run on
#: rather than report as a finding either way.
NEVER_RUNS = dict(DEFENDED, test="test_engine.py::test_defended_sum_over_cases")

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


@pytest.fixture
def sandbox_with_stale_bytecode(sandbox: Path) -> Path:
    """The sandbox with a pre-existing ``__pycache__`` beside the engine.

    THE HALF OF THE FALSE-SURVIVOR FIX THAT THE PLAIN SANDBOX CANNOT REACH.
    ``_pytest_env`` sets two things: ``PYTHONDONTWRITEBYTECODE``, which stops a
    run leaving bytecode behind, and ``PYTHONPYCACHEPREFIX``, which stops a run
    READING bytecode that was already there. Removing both reddens the tests
    above, but only incidentally -- the sandbox then goes dirty because bytecode
    lands in it. Removing only the prefix leaves every one of them green, and
    that is the half that matters: this repository has ``__pycache__``
    directories beside its sources right now, and CPython will happily use a
    cached ``.pyc`` whose source has since been mutated.

    The staleness is made deterministic rather than raced. In the wild the
    mechanism is CPython validating a ``.pyc`` against the source's size and its
    mtime TRUNCATED TO WHOLE SECONDS, so a length-preserving mutation written in
    the same second as the last compile is invisible -- which is a real hazard
    and an unreliable fixture. An UNCHECKED_HASH ``.pyc`` is the same defect
    without the stopwatch: CPython never validates it against the source at all.
    So if the cache beside the source is consulted, the mutation cannot be seen;
    if it is not consulted, it must be.

    ``.gitignore`` is committed first so the cache directory does not make the
    tree dirty -- the battery refuses a dirty tree, and this repository ignores
    ``__pycache__`` for the same reason.
    """
    (sandbox / ".gitignore").write_text("__pycache__/\n")
    subprocess.run(["git", "add", "-f", ".gitignore"], cwd=sandbox, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", "ignore bytecode"],
        cwd=sandbox, check=True,
    )
    py_compile.compile(
        str(sandbox / "engine.py"),
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        doraise=True,
    )
    cached = sorted((sandbox / "__pycache__").glob("engine.*.pyc"))
    assert cached, "the fixture wrote no bytecode, so it probes nothing"
    # And it really is unconditionally valid: read through it and the ORIGINAL
    # source's behaviour comes back even though the file on disk says otherwise.
    mutated = ENGINE.replace("return a + b", "return a - b")
    assert mutated != ENGINE
    (sandbox / "engine.py").write_text(mutated)
    try:
        proof = subprocess.run(
            [sys.executable, "-c", "import engine; print(engine.defended_sum(2, 3))"],
            cwd=sandbox, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(sandbox),
                 "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert proof.stdout.strip() == "5", (
            f"the stale .pyc is not masking the source, so this fixture cannot "
            f"probe anything: {proof.stdout!r} {proof.stderr!r}"
        )
    finally:
        (sandbox / "engine.py").write_text(ENGINE)
    assert _porcelain(sandbox) == "", "the fixture left the sandbox dirty"
    return sandbox


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


def test_the_battery_is_not_fooled_by_bytecode_left_beside_the_source(
    sandbox_with_stale_bytecode,
):
    """A mutation must be scored against the bytes on disk, not a cached compile.

    The same DEFENDED row as the test above, run in a sandbox that already has a
    ``__pycache__``. If the battery's environment lets CPython read it, the
    mutated source is never compiled, the named test passes on the ORIGINAL
    code, and a defended claim is reported as an undefended one -- the
    false-survivor failure that produced a phantom survivor on this table's
    first run. Exit 0 here means the mutation was seen.
    """
    result = _run(sandbox_with_stale_bytecode, [DEFENDED])

    assert result.returncode == 0, (
        f"the battery scored a DEFENDED claim as a survivor with a stale "
        f".pyc beside the source; it is reading cached bytecode instead of "
        f"the file it mutated:\n{result.stdout}\n{result.stderr}"
    )
    assert "SURVIVORS" not in result.stdout
    assert "PASSED" in result.stdout


def test_the_battery_errors_when_the_named_test_is_collected_but_never_run(sandbox):
    """"The test failed" and "the test never ran" must never be confused --
    and neither must "the test ran" and "the test was collected".

    ``test_the_battery_errors_when_the_named_test_matches_nothing`` above cannot
    make this distinction: a node id that does not exist makes pytest exit 4,
    which ``run_pytest`` turns into a HardError all by itself, so that test
    passes whether or not the precheck runs at all. This row's named test does
    exist, is collected, prints "1 test collected", and then never runs -- it is
    parametrised over an empty set. pytest exits 0, ``run_pytest`` says "pass",
    and without the precheck's passed-count check the battery would go on to
    report the mutation as a survivor: a hole invented by the instrument.
    """
    result = _run(sandbox, [NEVER_RUNS])

    assert result.returncode == 2, (
        f"a named test that is collected and never run exited "
        f"{result.returncode}; it must be a hard error, not a survivor:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "HARD ERROR" in result.stderr
    assert "without a single test PASSING" in result.stderr, result.stderr
    assert "SURVIVORS" not in result.stdout


def test_the_battery_hard_errors_when_root_is_not_a_git_repository(tmp_path):
    """Exit 1 means "survivors found". Nothing else may exit 1.

    ``git status`` in a directory that is not a repository used to escape as an
    uncaught CalledProcessError, which Python exits 1 for -- indistinguishable,
    to a script or a reader, from the battery's one real finding.
    """
    root = tmp_path / "not-a-repo"
    root.mkdir()
    (root / "engine.py").write_text(ENGINE)
    (root / "test_engine.py").write_text(ENGINE_TEST)

    result = _run(root, [DEFENDED])

    assert result.returncode == 2, (
        f"a non-repository --root exited {result.returncode}, which collides "
        f"with 'survivors found':\n{result.stdout}\n{result.stderr}"
    )
    assert "HARD ERROR" in result.stderr
    assert "not a usable git repository" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_dirty_tree_refusal_states_what_it_cannot_see(sandbox):
    """The refusal must say what it means.

    ``git status --porcelain`` is blind to git-IGNORED paths, so the refusal
    protects tracked work only. Stated rather than fixed: ``--ignored`` would
    make the battery refuse to start on this repository at all, whose detection
    cache, sample clips, models and process notes all live under ignored paths.
    """
    (sandbox / "engine.py").write_text(ENGINE + "\n# an uncommitted edit\n")

    result = _run(sandbox, [DEFENDED])

    assert result.returncode == 2
    assert "REFUSING TO RUN" in result.stderr
    assert "IGNORED" in result.stderr, (
        f"the refusal does not say that ignored paths are outside it:\n"
        f"{result.stderr}"
    )


def test_the_battery_runs_with_uncommitted_work_under_an_ignored_path(sandbox):
    """The other half of the pair above: the stated limit is the real one.

    This is the behaviour the refusal message documents, pinned so that it
    cannot drift away from the sentence without something going red. If a later
    change adds ``--ignored`` to the porcelain call, this reddens and the
    message has to be rewritten -- which is the point.
    """
    (sandbox / ".gitignore").write_text("scratch/\n")
    subprocess.run(["git", "add", "-f", ".gitignore"], cwd=sandbox, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", "ignore scratch"],
        cwd=sandbox, check=True,
    )
    (sandbox / "scratch").mkdir()
    (sandbox / "scratch" / "notes.txt").write_text("uncommitted, and ignored\n")
    assert _porcelain(sandbox) == "", "the fixture did not actually ignore it"

    result = _run(sandbox, [DEFENDED])

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "REFUSING TO RUN" not in result.stderr
    assert (sandbox / "scratch" / "notes.txt").read_text() == (
        "uncommitted, and ignored\n"
    )


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

#: Floors, not targets. Raised with the table each time it grows, so that a
#: later edit cannot quietly delete the rows this wave added and still pass.
#: Measured when last raised: 57 rows, 46 must-fail, 11 must-survive.
MINIMUM_ROWS = 55
MINIMUM_CONTROLS = 10
MINIMUM_MUST_FAIL = 44


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
    # Every documented open finding has to say where it is routed. There are
    # none in the table today -- the last three were promoted to must_fail when
    # the findings behind them were closed -- so this loop is a rule waiting
    # for the next one rather than a check on the current table. Asserted
    # explicitly so that "0 open findings" is a statement rather than a silently
    # empty loop; the known_open MACHINERY is covered by the sandbox tests above.
    known = [c for c in claims if c.expect == battery.KNOWN_OPEN]
    assert len(known) == 0, (
        f"{len(known)} known-open rows are back in the table. That is allowed, "
        f"but each has to say where it is routed and this floor has to be "
        f"updated to say how many there are: {[c.claim for c in known]}"
    )
    for entry in known:
        assert "notes section 5" in entry.note, entry.claim

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
    assert len(pytest_rows) >= 50, len(pytest_rows)

    missing = []
    for node_id in sorted({c.test for c in pytest_rows}):
        file_part, _, name = node_id.partition("::")
        source = (ROOT / file_part).read_text()
        if f"def {name}(" not in source:
            missing.append(node_id)
    assert missing == [], f"the table names tests that no longer exist: {missing}"
