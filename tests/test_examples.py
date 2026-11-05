"""The examples are run here, so the README and the files in examples/ cannot drift from the code."""

import runpy
from pathlib import Path

from polisys import cli

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_the_facilities_desk_package_validates_and_counts(tmp_path, capsys):
    folder = EXAMPLES / "facilities_desk"
    assert cli.main(["validate", str(folder / "package.json"), "--tools", str(folder / "tools.json")]) == 0
    assert cli.main(["counts", str(folder / "package.json")]) == 0
    assert "must_without_outcome" in capsys.readouterr().out


def test_the_facilities_desk_package_packs_again_from_its_draft(tmp_path):
    folder = EXAMPLES / "facilities_desk"
    out = tmp_path / "package.json"
    assert cli.main(["convert", str(folder / "policy.md"), "-o", str(tmp_path / "set.json")]) == 0
    assert (
        cli.main(
            ["pack", str(tmp_path / "set.json"), str(folder / "draft.json"), "-o", str(out), "--harness", "t", "--model", "t",
             "--tools", str(folder / "tools.json"), "--tests", str(folder / "tests.json")]
        )
        == 0
    )
    assert cli.main(["validate", str(out), "--tools", str(folder / "tools.json")]) == 0


def test_the_quickstart_blocks_exactly_the_two_bad_calls(capsys):
    runpy.run_path(str(EXAMPLES / "quickstart.py"), run_name="__main__")
    out = capsys.readouterr().out
    baseline, rules = out.split("== profile rules_only")
    assert "BLOCKED" not in baseline
    assert rules.count("BLOCKED") == 2 and rules.count("-> allowed") == 1
