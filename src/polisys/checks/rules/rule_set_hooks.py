"""``RuleSetHooks``: the rules of a run, the level of each rule, and the decision on a tool call. Its other parts are in
``code_checks``, ``role_check``, ``brief`` and ``duties``.

State the parts share (set in ``__init__`` and ``session_start``):

| Attribute | What it holds |
|---|---|
| ``rules``, ``by_tool``, ``global_rules``, ``global_verify`` | the rules, and the rules by trigger tool |
| ``facts``, ``store``, ``called`` | what the run has seen, and the tools called so far |
| ``open_duties`` | duties whose trigger was called and that no call has met |
| ``records`` | one line per decision, for the audit |
| ``notes`` | notes for the model that wait for the next channel |
| ``specs``, ``tools_in_run``, ``principal`` | the tool schemas, the tools of this run, the acting user |"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from polisys.hooks import AgentHooks, HookResult
from polisys.tool_kind import case_of, is_write_tool
from polisys.values import _when
from polisys.runtime.trace import audited_failure, emit, fresh
from polisys.checks.rules.rule_types import Check, MAX_SAME_DENY, RULE_TYPES, SCOPE_MIN
from polisys.checks.rules.seen import Facts
from polisys.checks.rules.code_checks import _CodeChecks
from polisys.checks.rules.role_check import _RoleCheck
from polisys.checks.rules.brief import _Brief
from polisys.checks.rules.duties import _Duties


class RuleSetHooks(_CodeChecks, _RoleCheck, _Brief, _Duties, AgentHooks):
    """Enforce a typed rule set at the lifecycle hooks.

    ``tool_specs`` are the MCP/JSON-schema tool definitions (used to tell required from optional arguments and
    free text from values). ``principal`` is the logged-in user as the platform knows it (``id``, ``role``).
    """

    def __init__(
        self,
        rules: List[Dict[str, Any]],
        tool_specs: List[Dict[str, Any]],
        principal: Optional[Dict[str, Any]] = None,
        mode: str = "enforce",
        tools_in_run: Optional[List[str]] = None,
        method_choice: Optional[Dict[str, Dict[str, Any]]] = None,
        verifier: Any = None,
        verifier_for: Optional[Any] = None,
        verifier_may_deny: bool = False,
        brief: bool = True,
        note_channel: str = "tool",
        scope_check: Optional[Callable[[str, str], Optional[float]]] = None,
        cap_effect: str = "allow",
        duty_follow_up: bool = True,
        verify_scope: str = "all_statements",
        brief_guidance: bool = False,
        exception_links: Optional[Dict[str, List[str]]] = None,
    ):
        """``brief_guidance``: the brief also lists the ``guidance`` rules that name no tool (default off: a measured run never
        showed them, and they were not part of what was measured). ``exception_links``: rule id -> ids of the rules that are
        an exception of it (``package.exception_links``); the brief shows each of those under the rule it is an exception of,
        and not again as its own item. ``None``: the brief lists every rule alone, as before.
        ``verify_scope``: ``all_statements`` sends the verifier every ``verify`` rule, also those that name no tool and
        those routed from a typed rule (the measured runs); ``specific_statements`` sends it the package's judgment rules on
        the tools they name and nothing else (design spec section 4: a general statement never goes to a per-call model).
        ``duty_follow_up``: ``False`` when the obligation ledger sends the follow-ups of the run (design section 5): an
        open duty is then recorded and closed as before, and neither shown in the brief nor sent at the reply.
        ``method_choice``: per statement or rule type, ``{"checker": "code|cedar|verifier|kev", "may_deny": bool}``
        (measured in the private evaluation repo). A type routed to ``verifier`` is checked from its text; a type with
        ``may_deny: false`` only reminds. ``verifier``: a ``verifier.Verifier`` for ``verify`` rules; without one,
        ``verify`` rules only appear in the brief."""
        self._duty_follow_up = (
            duty_follow_up  # private: a flag of the wiring, so a run with none is described as before
        )
        if verify_scope not in ("all_statements", "specific_statements"):
            raise ValueError(f"verify_scope {verify_scope!r} is not all_statements or specific_statements")
        self._verify_scope = verify_scope
        self._last = ""  # what the last check of a stage did (see ``last_decision``)
        self.verifier_may_deny = verifier_may_deny
        self.brief = brief  # list the checked rules in the system prompt at each model call
        # private, like ``_duty_follow_up``: settings of the brief that a run with none is described without
        self._brief_guidance = brief_guidance
        self._exception_links = {k: list(v) for k, v in (exception_links or {}).items()}
        # reminders: "tool" = attached to the result of the call they are about (stays in the conversation);
        # "system" = appended to the system prompt of the next model call only (the earlier behaviour)
        self.note_channel = note_channel
        self.method_choice = method_choice or {}
        self.verifier = verifier
        self.verifier_for = (
            verifier_for  # model name -> Verifier, for types whose measured best model differs
        )
        # scope_check(rule text, action) -> p that the rule applies; a code denial with p < SCOPE_MIN is dropped
        self.scope_check = scope_check
        self.cap_effect = (
            cap_effect  # after the repeated-denial cap: "allow" (let through) or "skip" (keep blocked)
        )
        self.rules: List[Dict[str, Any]] = []
        self.specs = {t["name"]: t for t in tool_specs}
        self.mode = mode
        self.tools_in_run = set(tools_in_run or self.specs)
        self.by_tool: Dict[str, List[Dict[str, Any]]] = {}
        self.global_rules: List[Dict[str, Any]] = []
        self.global_verify: List[Dict[str, Any]] = []
        self._global_guidance: List[Dict[str, Any]] = []
        self.cedar = None
        self.add_rules(rules)
        self.principal = dict(principal or {})
        self.session_start(None)

    @classmethod
    def from_package(
        cls, pkg: Any, tool_specs: List[Dict[str, Any]], with_exceptions: bool = False, **kwargs: Any
    ) -> "RuleSetHooks":
        """Enforce the rules of a ``PolicyPackage``, each with its ``level`` (see ``_level``). The other arguments are
        those of the constructor. ``with_exceptions``: the package's ``exception_of`` links reach the hooks as
        ``exception_links`` (``package.exception_links``: a link joins statements, and the rules that hold them are
        joined), so the brief shows each rule with its exceptions. Off, a package's links are not read; an
        ``exception_links`` argument wins."""
        from polisys.package.files import exception_links, runtime_rules

        if with_exceptions:
            kwargs.setdefault("exception_links", exception_links(pkg))
        return cls(runtime_rules(pkg), tool_specs, **kwargs)

    def add_rules(self, rules: List[Dict[str, Any]]) -> int:
        """Add rules (also during the run, e.g. from online intake). Returns how many were new."""
        known = {r.get("id") for r in self.rules}
        new = [self._route(r) for r in rules if r.get("type") in RULE_TYPES and r.get("id") not in known]
        for r in new:
            self.rules.append(r)
            tools = r.get("tools") or ([r["tool"]] if r.get("tool") else [])
            if r["type"] == "duty":
                tools = [r["trigger"]["tool"]]
            if r["type"] in ("grounded_values", "id_provenance"):
                self.global_rules.append(r)
            if r["type"] == "verify" and not tools:
                self.global_verify.append(r)
            if r["type"] == "guidance" and not tools:
                self._global_guidance.append(r)
            for t in tools:
                self.by_tool.setdefault(t, []).append(r)
        if any(r["type"] == "role_permission" for r in new):
            self.cedar = self._cedar([r for r in self.rules if r["type"] == "role_permission"])
        return len(new)

    def remove_rules_for(self, statement_text: str) -> int:
        """Drop the rules compiled from one statement (it was replaced by a later rule). Returns how many."""
        keys = ("statement_text", "quote", "statement")
        drop = {id(r) for r in self.rules if any(r.get(k) == statement_text for k in keys)}
        if not drop:
            return 0
        removed = [r for r in self.rules if id(r) in drop]
        self.rules = [r for r in self.rules if id(r) not in drop]
        self.by_tool = {t: [r for r in rs if id(r) not in drop] for t, rs in self.by_tool.items()}
        self.global_rules = [r for r in self.global_rules if id(r) not in drop]
        self.global_verify = [r for r in self.global_verify if id(r) not in drop]
        self._global_guidance = [r for r in self._global_guidance if id(r) not in drop]
        if any(r["type"] == "role_permission" for r in removed):
            self.cedar = self._cedar([r for r in self.rules if r["type"] == "role_permission"])
        return len(removed)

    def session_start(self, thread_id: Optional[str]) -> HookResult:
        self.facts = Facts(principal=self.principal)
        self.open_duties: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        # verdict log for the report; a tapped list stays the same list, so the audit keeps reading it
        self.records: List[Dict[str, Any]] = fresh(getattr(self, "records", None))
        self._denials: Dict[str, int] = {}
        # (rule id, case) of "block_once" rules that have stopped a call in this session: a rule stops once for each case
        self._blocked_once: set = set()
        self.called: List[str] = []
        from polisys.checks.facts import FactStore

        self.store = FactStore()
        self.request = ""
        self.notes: List[str] = []  # reminders for the next model call (a reminder never blocks the write)
        return HookResult(
            details=[
                {"rule": "ruleset", "outcome": "info", "reason": f"{len(self.rules)} rules, mode {self.mode}"}
            ]
        )

    def _record(self, hook: str, tool: str, check: Check, outcome: str) -> Dict[str, Any]:
        rec = {
            "hook": hook,
            "tool": tool,
            "rule_id": check.rule.get("id"),
            "type": check.rule.get("type"),
            "source": check.rule.get("source"),
            "outcome": outcome,
            "reason": check.reason,
            "remedy": check.remedy,
        }
        self.records.append(rec)
        return rec

    def on_user_message(self, message: str, thread_id: Optional[str]) -> HookResult:
        self.facts.add_text(message)
        self.request = message
        return HookResult()

    def _choice(self, r: Dict[str, Any]) -> Dict[str, Any]:
        for key in [r.get("type")] + list(r.get("types") or []):
            if key in self.method_choice:
                return self.method_choice[key]
        return {}

    def _route(self, r: Dict[str, Any]) -> Dict[str, Any]:
        """A rule whose type is routed to the verifier becomes a ``verify`` rule with the rule's quote as text."""
        if r.get("type") in ("verify", "guidance", "duty", "grounded_values", "id_provenance"):
            return r
        choice = self._choice(r)
        if choice.get("checker") != "verifier" or choice.get("code_may_deny", True) is not False:
            return r  # code stays the checker unless measurement says code is not reliable for this type
        tools = r.get("tools") or ([r["tool"]] if r.get("tool") else [])
        return {
            "id": r.get("id"),
            "type": "verify",
            "statement": r.get("quote") or self._describe(r),
            "tools": tools,
            "types": [r.get("type")] + list(r.get("types") or []),
            "source": r.get("source"),
            "unless": r.get("unless"),
            "routed_from": r.get("type"),
        }

    def _level(self, r: Dict[str, Any]) -> str:
        """The rule's ``level`` when it decides, else "" and the code after it decides. A level is the default when no
        measured setting applies, so it does not decide for:

        - a rule with no level (old inputs);
        - a rule whose type has a method choice, or a ``verify`` rule with ``verifier_may_deny``;
        - a duty (it has its own follow-up path) or guidance (shown in the brief, never checked);
        - a "no assumed data" rule with ``may_deny: false``, which stays observe-only.

        ``schema_conflict`` and ``may_deny: false`` are limits a level cannot raise. ``verifier_may_deny=False`` is the
        default and means no setting: a ``verify`` rule then never denies, and level ``observe`` still lowers it."""
        rule_type = r.get("type")
        if rule_type in ("duty", "guidance") or self._choice(r):
            return ""
        if rule_type == "verify" and self.verifier_may_deny:
            return ""
        if rule_type in ("grounded_values", "id_provenance") and r.get("may_deny") is False:
            return ""
        return r.get("level") or ""

    def _may_deny(self, r: Dict[str, Any], args: Optional[Dict[str, Any]] = None) -> bool:
        """False for a rule marked remind-only (``may_deny: false``, e.g. by a tuning filter), or when the measured
        method choice says this checker may not deny for the type (``may_deny`` for the verifier side,
        ``code_may_deny`` for code and Cedar rules). Otherwise the rule's level decides, if it has one: ``deny`` denies,
        ``block_once`` denies until the rule has stopped one call of the same case in this session and then reminds
        (the case is the call's identifying arguments, see ``tool_kind.case_of``; a call that names none has the case "")."""
        if r.get("may_deny", True) is False:
            return False
        level = self._level(r)
        if level and r.get("type") != "verify":
            return level == "deny" or (
                level == "block_once" and (r.get("id"), case_of(args or {})) not in self._blocked_once
            )
        if (
            r.get("type") in ("grounded_values", "id_provenance")
            and r.get("may_deny") is not True
            and "code_may_deny" not in self._choice(r)
        ):
            return False  # "no assumed data" is a judgment on every write and was never measured: remind only
        if r.get("type") == "verify" and not self.verifier_may_deny:
            return False  # model verdicts only remind (replay 2026-11-02: 0 violations stopped, 8 of 181 correct writes denied)
        key = "may_deny" if r.get("type") == "verify" else "code_may_deny"
        return self._choice(r).get(key, True) is not False

    def _observe_only(self, r: Dict[str, Any]) -> bool:
        """Recorded for audit, never shown to the agent: rules where the policy conflicts with the tool's own allowed
        values (the tool stands), and unmeasured "no assumed data" judgments unless the method choice enables them.
        Measured 2026-11-02 (DeepSeek agent): these notes fired on correct values and one cost a verifier. A rule
        with level ``observe`` is also recorded only."""
        if r.get("schema_conflict"):
            return True
        level = self._level(r)
        if level:
            return level == "observe"
        return (
            r.get("type") in ("grounded_values", "id_provenance")
            and r.get("may_deny") is not True
            and not self._choice(r)
        )

    @audited_failure
    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> HookResult:
        if not is_write_tool(tool):
            self._last = "n/a"
            return HookResult()
        return self._decide(tool, args, self._checks(tool, args))

    @audited_failure
    def check_code(self, tool: str, args: Dict[str, Any]) -> HookResult:
        """The checks that code decides (typed rules and Cedar), with no model call: the first stage of the write checks."""
        if not is_write_tool(tool):
            self._last = "n/a"
            return HookResult()
        return self._decide(tool, args, self._checks(tool, args, verify=False))

    @audited_failure
    def check_verifier(self, tool: str, args: Dict[str, Any]) -> HookResult:
        """The verifier on the ``verify`` rules of this call: the last stage of the write checks. Reminds unless
        ``verifier_may_deny``."""
        if not is_write_tool(tool):
            self._last = "n/a"
            return HookResult()
        return self._decide(tool, args, self._verify(tool, args))

    @property
    def last_decision(self) -> str:
        """What the last check of a stage did: ``n/a`` (nothing applied), ``allow``, ``remind``, ``block_once``, ``deny`` or
        ``error`` (a model check that failed and let the call through)."""
        return self._last

    def _decide(self, tool: str, args: Dict[str, Any], checks: List[Check]) -> HookResult:
        quiet = [c for c in checks if not c.ok and self._observe_only(c.rule)]
        soft = [c for c in checks if not c.ok and not self._may_deny(c.rule, args) and c not in quiet]
        denied = [c for c in checks if not c.ok and self._may_deny(c.rule, args) and c not in quiet]
        out_of_scope = []
        if denied and self.scope_check is not None:
            act = f"{tool}({json.dumps({k: v for k, v in args.items() if v not in (None, '', [], {})}, default=str)[:1500]})"
            for c in denied:
                text = c.rule.get("quote") or c.rule.get("statement_text") or c.rule.get("statement")
                if not text:
                    continue
                p = self.scope_check(str(text), act)
                if p is not None and p < SCOPE_MIN:
                    out_of_scope.append(c)
            denied = [c for c in denied if c not in out_of_scope]
        details = [
            self._record(
                "before_tool_call",
                tool,
                c,
                "allow"
                if c.ok
                else (
                    "observe"
                    if c in quiet
                    else (
                        "out_of_scope"
                        if c in out_of_scope
                        else ("deny" if self._may_deny(c.rule, args) else "remind")
                    )
                ),
            )
            for c in checks
        ]
        failed = any(c.ok and str(c.reason).startswith("verifier_error") for c in checks)
        self._last = (
            "n/a"
            if not checks
            else "error"
            if failed and not denied
            else "deny"
            if any(self._level(c.rule) != "block_once" for c in denied)
            else "block_once"
            if denied
            else "remind"
            if soft
            else "allow"
        )
        if soft and not denied and self.mode != "shadow":
            # A before_tool_call message blocks the call (HookBlocked), so a reminder is not returned here: it is
            # queued and shown at the next model call, and the write runs.
            lines = [f"POLICY NOTE about your `{tool}` call (it was run):"] + [
                f"- [{c.rule.get('source', '')}] {c.reason}" for c in soft
            ]
            self.notes.append("\n".join(lines))
            return HookResult(details=details)
        if not denied or self.mode == "shadow":
            return HookResult(details=details)
        self._blocked_once.update(
            (c.rule.get("id"), case_of(args)) for c in denied if self._level(c.rule) == "block_once"
        )
        # a block_once rule stops each case once, so its denials are counted for each case too (else the 4th case of a rule
        # would meet the cap on its first try)
        case = case_of(args)
        key = (
            tool
            + "|"
            + ",".join(
                sorted(
                    str(c.rule.get("id"))
                    + (f"@{case}" if self._level(c.rule) == "block_once" and case else "")
                    for c in denied
                )
            )
        )
        self._denials[key] = self._denials.get(key, 0) + 1
        if self._denials[key] > MAX_SAME_DENY:
            capped = {  # the detail shows the cap with the rules it concerns; the audit names it deny_after_cap
                "hook": "before_tool_call",
                "tool": tool,
                "rule": "deny_cap",
                "rule_id": ",".join(sorted(str(c.rule.get("id")) for c in denied)),
                "outcome": "deny" if self.cap_effect == "skip" else "allow",
                "reason": f"same denial {self._denials[key]} times",
            }
            details.append(capped)
            emit(self, capped)  # no record holds this decision
            if self.cap_effect == "skip":
                return HookResult(
                    message=(
                        f"POLICY: `{tool}` is still not allowed after {self._denials[key]} tries. Do not try this "
                        "action again. Continue with the rest of the task without it, and say in your answer that it "
                        "was skipped because of the rule."
                    ),
                    details=details,
                )
            self._last = "allow"
            return HookResult(details=details)
        lines = [f"POLICY: `{tool}` was not called."]
        for c in denied:
            lines.append(
                f"- [{c.rule.get('source', '')}] {c.reason}."
                + (f" Do instead: {c.remedy}" if c.remedy else "")
            )
        return HookResult(message="\n".join(lines), details=details)

    def _checks(self, tool: str, args: Dict[str, Any], verify: bool = True) -> List[Check]:
        out: List[Check] = []
        role_check = self._role_check(tool)
        if role_check:
            out.append(role_check)
        demanded = set()
        for r in self.by_tool.get(tool, []):
            fn = getattr(self, f"_check_{r['type']}", None)
            if fn is None:
                continue
            res = fn(r, tool, args)
            if res is not None:
                out.append(res)
            if r["type"] in ("required_args",) and _when(r.get("when"), args):
                demanded.update(r.get("args", []))
            if r["type"] in ("allowed_values", "derived_value"):
                demanded.add(r.get("arg"))
        for r in self.global_rules:
            res = getattr(self, f"_check_{r['type']}")(r, tool, args, demanded)
            if res is not None:
                out.append(res)
        if verify:
            out.extend(self._verify(tool, args))
        return out

    def _verify(self, tool: str, args: Dict[str, Any]) -> List[Check]:
        """One verifier call for all ``verify`` rules that apply to this write (tool-specific and global)."""
        rules = [r for r in self.by_tool.get(tool, []) if r["type"] == "verify"]
        if self._verify_scope == "specific_statements":
            rules = [
                r for r in rules if not r.get("routed_from")
            ]  # a judgment rule of the package, on a tool it names
        else:
            rules += self.global_verify
        if not rules or self.verifier is None:
            return []
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in rules:  # one verifier call per chosen model (method_choice), default model otherwise
            groups.setdefault(self._choice(r).get("model") or "", []).append(r)
        facts = self.store.for_call(args)
        verdicts = []
        for model_name, group in groups.items():
            verifier = self.verifier_for(model_name) if model_name and self.verifier_for else self.verifier
            statements = [
                {
                    "n": i,
                    "text": r.get("statement", ""),
                    "source": r.get("source", ""),
                    "rule_id": r.get("id"),
                    "unless": r.get("unless"),
                }
                for i, r in enumerate(group, 1)
            ]
            props = list(((self.specs.get(tool) or {}).get("inputSchema") or {}).get("properties") or {})
            verdicts += verifier.check(self.request, statements, tool, args, facts, props)
        by_id = {r.get("id"): r for r in rules}
        out = []
        for v in verdicts:
            if v.outcome == "deny":
                out.append(Check(by_id.get(v.rule_id, rules[0]), False, v.reason, v.remedy))
            else:
                out.append(
                    Check(
                        {"id": "verifier", "type": "verify", "source": "verifier"},
                        True,
                        v.reason or "no statement broken",
                    )
                )
        return out
