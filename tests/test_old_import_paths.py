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
