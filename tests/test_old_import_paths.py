"""Every import path that existed before the files moved into folders still works.

``tests/data/old_import_paths.json`` lists, for each module of the flat layout, the names it had at module level (its own
functions, classes and constants, and the names it imported from other polisys modules). The list was written from the code
before the move and is not written again: a name that is deliberately dropped is deleted from the file by hand, in the commit
that drops it.
"""

import importlib
import json
from pathlib import Path

import pytest

from polisys._aliases import OLD_TO_NEW

OLD = json.loads((Path(__file__).parent / "data" / "old_import_paths.json").read_text())


@pytest.mark.parametrize("module", sorted(OLD))
def test_an_old_module_path_imports_and_has_its_names(module):
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        if (exc.name or "").split(".")[0] != "polisys":  # a host framework or an extra that is not installed here
            pytest.skip(f"{module} needs {exc.name}")
        raise
    missing = [name for name in OLD[module] if not hasattr(mod, name)]
    assert not missing, f"{module} lost {missing}"


def _import(module):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if (exc.name or "").split(".")[0] != "polisys":
            pytest.skip(f"{module} needs {exc.name}")
        raise


@pytest.mark.parametrize("old", sorted(OLD_TO_NEW))
def test_an_old_path_gives_the_module_at_the_new_path_itself(old):
    new = _import(OLD_TO_NEW[old])
    assert importlib.import_module(old) is new
    assert new.__name__ == new.__spec__.name == OLD_TO_NEW[old]  # the old path leaves no mark on the module


def test_a_patch_through_an_old_path_changes_the_code_that_runs(monkeypatch):
    from polisys.checks import rules

    marker = object()
    monkeypatch.setattr("polisys.ruleset.is_write_tool", marker)
    assert rules.is_write_tool is marker


def test_every_module_of_the_flat_layout_is_in_the_table_or_kept_its_path():
    kept = {"polisys", "polisys.cli", "polisys.hooks", "polisys.llm", "polisys.tool_kind"}
    folders_now = {"polisys.package", "polisys.runtime"}  # their __init__ answers for the names of the old module
    assert set(OLD) - kept - folders_now == set(OLD_TO_NEW)
