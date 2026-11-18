"""The components of the policy layer: a name, when the profile turns it on, and the function that builds it.

The order of the entries is the order of the hooks in a run. A new component is one entry here, a ``build`` function in its
own file and, when it has options, a ``Settings`` class there that ``runtime/profile.py`` gives a place in the profile.
Components under ``polisys.experimental`` have no gain shown on a held-out task set (docs/components.md).
"""

from __future__ import annotations

from typing import Dict

from polisys.checks import procedure_monitor, rules
from polisys.experimental.checks import default_values, gate
from polisys.experimental.finish import ledger
from polisys.experimental.guidance import (
    policy_lines,
    procedure_guide,
    procedure_push,
    procedure_store,
    read_first,
    tool_notes,
)
from polisys.experimental.organizers import sop_loop, sop_navigator, sop_tracker
from polisys.finish import requirements_tracker
from polisys.guidance import work_method
from polisys.runtime.context import Component
from polisys.runtime.profile import ledger_on, rules_on_package
from polisys.sources import intake

COMPONENTS: Dict[str, Component] = {
    "rules": Component(rules_on_package, rules.build),
    "loop": Component(lambda cfg: cfg.organizers.loop.mode == "on", sop_loop.build),
    "procedure_monitor": Component(rules_on_package, procedure_monitor.build),
    "intake": Component(lambda cfg: cfg.sources.data, intake.build),
    "requirements_tracker": Component(
        lambda cfg: cfg.organizers.requirements_tracker.mode == "on", requirements_tracker.build
    ),
    "guide": Component(lambda cfg: cfg.organizers.guide.mode == "on", procedure_guide.build),
    "procedure_tools": Component(lambda cfg: cfg.organizers.procedure_tools == "on", procedure_store.build),
    "gate": Component(lambda cfg: cfg.checkers.gate.mode == "on", gate.build),
    "procedure_push": Component(lambda cfg: cfg.organizers.procedure_push.mode == "on", procedure_push.build),
    "read_first": Component(lambda cfg: cfg.notes.read_first, read_first.build),
    "work_method": Component(lambda cfg: bool(cfg.notes.work_method), work_method.build),
    "tool_notes": Component(lambda cfg: cfg.notes.tool_notes.mode == "on", tool_notes.build),
    "default_values": Component(lambda cfg: cfg.checkers.default_values.mode == "on", default_values.build),
    "policy_lines": Component(lambda cfg: cfg.notes.policy_lines.mode == "on", policy_lines.build),
    "sop_tracker": Component(lambda cfg: cfg.organizers.sop_tracker.mode == "on", sop_tracker.build),
    "navigator": Component(lambda cfg: cfg.organizers.navigator.mode == "on", sop_navigator.build),
    "obligations": Component(ledger_on, ledger.build),
}
