"""The reference section on the site must describe the software that exists.

`web/src/ui/results.ts` publishes a command table and an API table. Those are the
one part of the page that is authored rather than baked -- a command name is not a
measurement, so there is no report to bake it from -- which means they are the one
part that can drift. This module is what stops that, and it fails in both
directions on purpose:

* a command the page lists that the CLI does not have is a page that documents
  software nobody can run;
* a command the CLI grows without the page listing it is a reference that is
  quietly incomplete, which is the direction a one-way check always misses.

It also pins the honesty of the `state` column. Two commands are placeholders and
`--help` says so; a page that showed them as working would be the page's only
untrue claim about itself.
"""

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS_TS = ROOT / "web" / "src" / "ui" / "results.ts"

#: Floors, so neither table can pass by being empty. Two empty lists compare
#: equal, which would leave every assertion below vacuously true.
MINIMUM_COMMANDS = 5
MINIMUM_API_ENTRIES = 5

_COMMAND = re.compile(
    r'name:\s*"(?P<name>[a-z][a-z-]*)",\s*'
    r"what:\s*\n?\s*\"(?P<what>[^\"]*)\",\s*"
    r"implemented:\s*(?P<implemented>true|false)",
    re.DOTALL,
)

_PYTHON_ENTRY = re.compile(r'python:\s*"(?P<path>[A-Za-z_][\w.]*)"')

#: The phrase click's own help uses for a command that is a placeholder. Read from
#: the CLI rather than hard-coded per command, so a command that gets implemented
#: fails here until the page stops calling it unbuilt.
_NOT_BUILT = "not built yet"


def _page_commands() -> dict[str, bool]:
    """Every command the page lists, mapped to whether it claims it works."""
    text = RESULTS_TS.read_text(encoding="utf-8")
    found = {
        match.group("name"): match.group("implemented") == "true"
        for match in _COMMAND.finditer(text)
    }
    assert len(found) >= MINIMUM_COMMANDS, (
        f"only found {len(found)} commands in {RESULTS_TS.relative_to(ROOT)}; the "
        f"parser has stopped matching the source, so this module is checking "
        f"nothing"
    )
    return found


def _cli_help() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "trafficlens.cli", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _cli_commands() -> dict[str, str]:
    """Every command click reports, mapped to its one-line help."""
    help_text = _cli_help()
    _, _, listing = help_text.partition("Commands:")
    found = {}
    for line in listing.splitlines():
        match = re.match(r"\s{2,}(?P<name>[a-z][a-z-]*)\s\s+(?P<help>\S.*)$", line)
        if match:
            found[match.group("name")] = match.group("help")
    assert len(found) >= MINIMUM_COMMANDS, (
        f"only parsed {len(found)} commands out of `--help`; the parser has "
        f"stopped matching click's output"
    )
    return found


def test_the_page_and_the_cli_list_exactly_the_same_commands():
    page = set(_page_commands())
    cli = set(_cli_commands())
    assert page == cli, (
        f"the reference section and the CLI disagree. On the page only: "
        f"{sorted(page - cli)}. In the CLI only: {sorted(cli - page)}."
    )


def test_a_command_the_cli_calls_unbuilt_is_not_shown_as_working():
    """The page's `state` column, checked against the CLI's own words."""
    page = _page_commands()
    wrong = []
    for name, help_line in _cli_commands().items():
        unbuilt = _NOT_BUILT in help_line.lower()
        if page.get(name) is (not unbuilt):
            continue
        wrong.append(
            f"{name}: --help says {'not built' if unbuilt else 'implemented'}, "
            f"the page says {'works' if page.get(name) else 'not built'}"
        )
    assert wrong == [], wrong


def test_the_cli_really_does_carry_placeholders():
    """The control on the test above.

    If every command were implemented, the state column would be trivially
    correct and the check would prove nothing about it. It is worth failing when
    that becomes true, because then the column can be dropped.
    """
    unbuilt = [
        name for name, help_line in _cli_commands().items() if _NOT_BUILT in help_line.lower()
    ]
    assert unbuilt, (
        "no command reports itself unbuilt any more, so the page's state column "
        "has nothing left to be honest about; drop it"
    )


@pytest.mark.parametrize("dotted", _PYTHON_ENTRY.findall(RESULTS_TS.read_text(encoding="utf-8")))
def test_every_python_object_the_page_names_can_be_imported(dotted: str):
    """A reference is only a reference if the names in it resolve.

    Imported rather than grepped for: a symbol can be present in a file and not
    exported, and a module can be present and not importable.
    """
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        target = module
        for attribute in parts[split:]:
            assert hasattr(target, attribute), f"{module_name} has no {attribute}"
            target = getattr(target, attribute)
        return
    pytest.fail(f"no importable module prefix of {dotted}")


def test_the_api_table_is_not_empty():
    entries = _PYTHON_ENTRY.findall(RESULTS_TS.read_text(encoding="utf-8"))
    assert len(entries) >= MINIMUM_API_ENTRIES, (
        f"only found {len(entries)} API entries; the parser has stopped matching "
        f"the source, so the parametrised test above is running on nothing"
    )
