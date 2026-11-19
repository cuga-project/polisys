"""``cli explain``: what a profile turns on, read from the profile alone."""

from pathlib import Path

from polisys import cli
from polisys.runtime.registry import COMPONENTS, status

PROFILES = Path(__file__).parent.parent / "examples" / "profiles"


def _rows(text):
    """component -> [state, status], from the table that ``explain`` prints."""
    rows = [line.split() for line in text.splitlines()]
    return {row[0]: row[1:3] for row in rows if row and row[0] in COMPONENTS}


def test_explain_lists_every_component_with_its_state_and_status(capsys):
    assert cli.main(["explain", str(PROFILES / "rules_only.yaml")]) == 0
    out = capsys.readouterr().out
    rows = _rows(out)
    assert set(rows) == set(COMPONENTS)
    assert rows["rules"] == ["on", "core"] and rows["procedure_monitor"] == ["on", "core"]
    assert rows["gate"] == ["off", "experimental"]
    assert "on: 2 (2 core, 0 experimental); off: 15" in out
    assert "left at auto" not in out  # the profile lists every component


def test_a_profile_with_nothing_on_has_nothing_on(capsys):
    assert cli.main(["explain", str(PROFILES / "baseline.yaml")]) == 0
    assert all(state == "off" for state, _ in _rows(capsys.readouterr().out).values())


def test_a_setting_left_at_auto_is_named_and_counts_as_off(tmp_path, capsys):
    profile = tmp_path / "auto.yaml"
    profile.write_text("policy:\n  organizers:\n    guide: auto\n")
    assert cli.main(["explain", str(profile)]) == 0
    out = capsys.readouterr().out
    assert _rows(out)["guide"] == ["off", "experimental"]
    assert "left at auto" in out and "organizers.guide.mode" in out


def test_a_profile_that_does_not_load_gives_the_reason(tmp_path, capsys):
    assert cli.main(["explain", str(tmp_path / "missing.yaml")]) == 1
    assert "cannot read the file" in capsys.readouterr().out


def test_the_status_of_a_component_is_where_its_code_is():
    core = {name for name in COMPONENTS if status(name) == "core"}
    assert core == {"rules", "procedure_monitor", "intake", "requirements_tracker", "work_method"}
