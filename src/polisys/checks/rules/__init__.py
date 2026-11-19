"""Typed policy rules and the hooks that enforce them at run time.

A rule set is a list of rule dicts made by ``compile/builder.py`` from policy text (offline) or by online intake. Each rule
has a ``type`` with fixed slots, a ``source`` (section and line) and a verbatim ``quote``. ``RuleSetHooks`` runs
them at the agent's lifecycle hooks; nothing in this file is specific to one benchmark or one policy.

| Type             | Slots                                              | Checker | Hook             |
|------------------|----------------------------------------------------|---------|------------------|
| required_args    | tool, args, when?                                  | code    | before_tool_call |
| allowed_values   | tool, arg, values, when?                           | code    | before_tool_call |
| derived_value    | tool, arg, inputs, table                           | code    | before_tool_call |
| transition       | tool, arg, id_arg, edges                           | code    | before_tool_call |
| reference_status | tool, arg, field, allowed                          | code    | before_tool_call |
| membership       | tool, arg, group_arg                               | code    | before_tool_call |
| role_permission  | tools, roles, effect (forbid)                      | cedar   | before_tool_call |
| grounded_values  | (global) optional arguments must come from data    | code    | before_tool_call |
| id_provenance    | (global) IDs must come from data                   | code    | before_tool_call |
| duty             | trigger {tool, when}, then {tool, when}            | monitor | after_tool_call, before_reply |
| guidance         | text, tools                                        | brief   | before_llm_call  |
| record_condition | tools, field, op, value, effect (forbid), when?    | code    | before_tool_call |
| threshold        | tool, arg, op, value, then (forbid|requires_approval), approval_tool? | code | before_tool_call |
| format           | tool, arg, pattern                                 | code    | before_tool_call |
| verify           | statement, tools                                   | verifier | before_tool_call |

``record_condition`` looks at the records that the call names (identifiers in its arguments, e.g. ``FE-103``) and
denies when one of them meets the condition ("Notes contains FSR-"). If no named record was seen, it does not decide.

**The parts of this folder.** This file holds ``build``.

- ``rule_types``: ``RULE_TYPES``, ``CHECKER``, ``Check``.
- ``seen``: ``Facts``, what the agent has seen.
- ``rule_set_hooks``: ``RuleSetHooks``: state, the level of a rule, the decision on a call.
- ``code_checks``, ``role_check``, ``brief``, ``duties``: the other parts of ``RuleSetHooks``.
- ``duty_obligations``: ``obligations_from_duties``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polisys._aliases import names_of
from polisys.runtime.context import Piece

if TYPE_CHECKING:
    from polisys.runtime.assembly import Assembly


def build(a: Assembly) -> Piece:
    from polisys.checks.rules.rule_set_hooks import RuleSetHooks
    from polisys.package.files import runtime_rules
    from polisys.checks.verifier import Verifier

    pkg = a.package("rules")
    rules = runtime_rules(pkg)
    if not rules:
        raise a.fail("rules", "the package has no rules")
    if not a.ctx.tool_specs:
        raise a.fail("rules", "the task gives no tool specs, so no rule can be matched to a call")
    choice = a.method_choice()
    verifier = verifier_for = None
    check = a.cfg.checkers.verifier
    specific = check.scope == "specific_statements"
    # specific_statements: only a judgment rule of the package, on a tool it names (spec section 4: a general statement never
    # goes to a per-call model); a package with none builds no verifier.
    if check.mode == "on" and any(
        r.get("type") == "verify"
        and (not specific or (not r.get("routed_from") and (r.get("tools") or r.get("tool"))))
        for r in rules
    ):
        verifier = Verifier(a.model("verifier", check.model))
        verifier_for = a.verifier_for("verifier") if choice else None
    hooks = RuleSetHooks.from_package(
        pkg,
        a.ctx.tool_specs,
        principal=a.ctx.principal,
        tools_in_run=a.ctx.tools_in_run,
        method_choice=choice,
        verifier=verifier,
        verifier_for=verifier_for,
        verifier_may_deny=check.may_deny,
        brief=a.brief(),
        brief_guidance=a.cfg.notes.brief_guidance,
        # a rule list shows each rule with its exceptions (spec section 8); other shapes keep the brief as measured
        with_exceptions="rules" in a.resolution.shapes,
        note_channel=a.note_channel(),
        cap_effect=a.cfg.decisions.after_repeated_denials,
        duty_follow_up=not a.ledger_on,  # with a producer, only the ledger sends follow-ups
        verify_scope=check.scope,
    )
    return Piece(hooks=[hooks])

_PARTS = ("rule_types", "seen", "code_checks", "role_check", "brief", "duties", "rule_set_hooks", "duty_obligations")

# A name of one of the parts, read on first use: ``polisys.checks.rules`` was one module before 2026-11.
__getattr__ = names_of(__name__, *(f"{__name__}.{part}" for part in _PARTS), "polisys.tool_kind", "polisys.values")
