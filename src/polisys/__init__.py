"""polisys: policy and SOP adherence for tool-using agents.

A policy document becomes a **package** (typed rules, procedures with their must-not rules, obligations, questions and
tests). A **profile** (YAML) says which components check it, and ``runtime.build_runtime`` turns profile and package into
hook objects that any agent host can run at its lifecycle hooks (``polisys.hooks``).

| Hook object            | Policy source                         | Made by                          |
|------------------------|---------------------------------------|----------------------------------|
| ``RuleSetHooks``       | typed rules of a package or build     | ``build_runtime`` or ``from_package`` |
| ``StoreGate``          | procedures and their must-not rules   | ``build_runtime``                |
| ``ObligationLedger``   | what must exist at the final reply    | ``build_runtime``                |
| ``OnlinePolicyIntake`` | policies found in tool output         | itself, during the run           |

Cedar needs the ``cedar`` extra (``cedarpy``). Model calls need the ``llm`` extra. Design: docs/design.md.
"""

from .cedar_engine import CedarDecision, CedarEngine
from .builder import BuildResult, build, openai_json_model, split_policy
from .intake import OnlinePolicyIntake
from .ruleset import RULE_TYPES, RuleSetHooks, is_write_tool

__all__ = [
    "BuildResult",
    "OnlinePolicyIntake",
    "RULE_TYPES",
    "RuleSetHooks",
    "build",
    "is_write_tool",
    "openai_json_model",
    "split_policy",
    "CedarDecision",
    "CedarEngine",
]
