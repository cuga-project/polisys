"""The CUGA agent-loop test helper (``ScriptedModel``, ``MARKER``) that the real-agent-loop tests run on.

It is taken from an installed ``cuga`` that ships it (``cuga.backend.hooks.tests.test_agent_loop``), else from a
cuga-agent checkout named by ``CUGA_REPO`` (``tests/unit/hooks/test_agent_loop.py``). With neither, the test skips and
says why, so a run on a new CUGA layout does not skip these tests without a word.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

NAME = "cuga.backend.hooks.tests.test_agent_loop"
HELP = (
    f"needs the CUGA agent-loop test helper: install a cuga that ships {NAME}, "
    "or set CUGA_REPO to a cuga-agent checkout that has tests/unit/hooks/test_agent_loop.py"
)


def load_agent_loop():
    try:
        return importlib.import_module(NAME)
    except ImportError:
        pass
    repo = os.environ.get("CUGA_REPO")
    path = Path(repo) / "tests" / "unit" / "hooks" / "test_agent_loop.py" if repo else None
    if path is None or not path.is_file():
        pytest.skip(HELP)
    spec = importlib.util.spec_from_file_location(NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # the helper needs the agent's own dependencies
        sys.modules.pop(NAME, None)
        pytest.skip(f"the CUGA agent-loop helper cannot be imported: {exc}")
    return module
