"""Old module paths. Until 2026-11 every module sat in one flat folder; this table names where each one is now.

``import polisys.ruleset`` still works and gives the module ``polisys.checks.rules`` itself (the same object, not a
copy), so a patch through an old path changes the code that runs. New code uses the new paths. ``polisys.runtime`` and
``polisys.package`` are folders now and answer for their old names themselves (their ``__init__``).
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from typing import Optional

OLD_TO_NEW = {
    "polisys.adapters": "polisys.hosts",
    "polisys.adapters.cuga": "polisys.hosts.cuga",
    "polisys.builder": "polisys.compile.builder",
    "polisys.cedar_engine": "polisys.checks.cedar",
    "polisys.classify": "polisys.compile.classify",
    "polisys.counts": "polisys.compile.counts",
    "polisys.decisions": "polisys.checks.decisions",
    "polisys.default_values": "polisys.experimental.checks.default_values",
    "polisys.deliverables": "polisys.experimental.finish.request_checklist",
    "polisys.detectors": "polisys.sources.detectors",
    "polisys.documents": "polisys.compile.documents",
    "polisys.dry_run": "polisys.runtime.dry_run",
    "polisys.facts": "polisys.checks.facts",
    "polisys.followups": "polisys.finish.followups",
    "polisys.guard_code": "polisys.experimental.checks.guard_code",
    "polisys.handlers": "polisys.compile.handlers",
    "polisys.handlers.links": "polisys.compile.handlers.links",
    "polisys.handlers.procedure_rules": "polisys.compile.handlers.procedure_rules",
    "polisys.handlers.procedures": "polisys.compile.handlers.procedures",
    "polisys.handlers.prose": "polisys.compile.handlers.prose",
    "polisys.handlers.tables": "polisys.compile.handlers.tables",
    "polisys.intake": "polisys.sources.intake",
    "polisys.levels": "polisys.runtime.levels",
    "polisys.listing": "polisys.sources.listing",
    "polisys.model": "polisys.package.model",
    "polisys.obligations": "polisys.experimental.finish.ledger",
    "polisys.policy_config": "polisys.runtime.profile",
    "polisys.policy_lines": "polisys.experimental.guidance.policy_lines",
    "polisys.procedure_compiler": "polisys.experimental.compile.procedure_compiler",
    "polisys.procedure_guide": "polisys.experimental.guidance.procedure_guide",
    "polisys.procedure_monitor": "polisys.checks.procedure_monitor",
    "polisys.procedure_push": "polisys.experimental.guidance.procedure_push",
    "polisys.procedure_store": "polisys.experimental.guidance.procedure_store",
    "polisys.read_first": "polisys.experimental.guidance.read_first",
    "polisys.references": "polisys.sources.references",
    "polisys.requirements": "polisys.finish.requirements",
    "polisys.router": "polisys.checks.router",
    "polisys.rule_repair": "polisys.compile.rule_repair",
    "polisys.ruleset": "polisys.checks.rules",
    "polisys.sop_loop": "polisys.experimental.organizers.sop_loop",
    "polisys.sop_navigator": "polisys.experimental.organizers.sop_navigator",
    "polisys.sop_planner": "polisys.experimental.organizers.sop_planner",
    "polisys.sop_retrieval": "polisys.checks.retrieval",
    "polisys.sop_tracker": "polisys.experimental.organizers.sop_tracker",
    "polisys.store_gate": "polisys.experimental.checks.gate",
    "polisys.structure": "polisys.compile.structure",
    "polisys.tool_notes": "polisys.experimental.guidance.tool_notes",
    "polisys.trace": "polisys.runtime.trace",
    "polisys.tracker": "polisys.finish.requirements_tracker",
    "polisys.validate": "polisys.compile.validate",
    "polisys.verifier": "polisys.checks.verifier",
    "polisys.work_method": "polisys.guidance.work_method",
}
_FOLDERS = ("polisys.adapters", "polisys.handlers")  # old folders: a module under one moved with it


def new_path(name: str) -> Optional[str]:
    """The new path of an old module path, or ``None`` when ``name`` is not an old path."""
    if name in OLD_TO_NEW:
        return OLD_TO_NEW[name]
    for folder in _FOLDERS:
        if name.startswith(folder + "."):
            return OLD_TO_NEW[folder] + name[len(folder) :]
    return None


class _OldPaths(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        return importlib.machinery.ModuleSpec(name, self) if new_path(name) else None

    def create_module(self, spec):
        module = importlib.import_module(new_path(spec.name))
        spec.loader_state = module.__spec__  # the import system puts this spec on the module: keep the module's own
        return module

    def exec_module(self, module):
        module.__spec__ = module.__spec__.loader_state


def install() -> None:
    if not any(isinstance(finder, _OldPaths) for finder in sys.meta_path):
        sys.meta_path.insert(0, _OldPaths())
