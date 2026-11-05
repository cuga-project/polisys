"""Typed policy rules and the hooks that enforce them at run time.

A rule set is a list of rule dicts made by ``builder.py`` from policy text (offline) or by online intake. Each rule
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
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from polisys.hooks import AgentHooks, HookResult

from .tool_kind import READ_FIRST, case_of
from .trace import audited_failure, emit, fresh

RULE_TYPES = (
    "required_args",
    "allowed_values",
    "derived_value",
    "transition",
    "reference_status",
    "membership",
    "role_permission",
    "grounded_values",
    "id_provenance",
    "duty",
    "guidance",
    "record_condition",
    "threshold",
    "format",
    "verify",
)
CHECKER = {"role_permission": "cedar", "duty": "monitor", "guidance": "brief"}
WRITE_VERBS = {
    "create", "update", "add", "delete", "link", "register", "remove", "send", "map", "set", "assign", "close",
    "resolve", "publish", "approve", "reject", "cancel", "escalate", "attach", "submit", "post", "transfer",
    "pay", "refund", "book", "reset", "move", "archive", "upload", "insert", "patch", "put", "write",
    "schedule", "enroll", "invite", "notify", "reply", "share", "grant", "revoke", "release", "allocate", "purchase",
    "edit", "modify", "save",
}  # fmt: skip
FREE_TEXT = re.compile(r"description|notes?$|comment|message|subject|body|text|summary|title|worknote", re.I)
ID_VALUE = re.compile(r"^(?=.*\d)[A-Za-z]{2,}[A-Za-z0-9]*([_-][A-Za-z0-9]+)+$|^[A-Z]{2,}\d{3,}$")
SCOPE_MIN = 0.35  # decision-model answer below which a code rule is taken as not applying to the call
MAX_SAME_DENY = 3  # repeated-denial cap: the 4th identical denial is let through and recorded


READ_VERBS = READ_FIRST  # one list for the whole layer (tool_kind.py)


def _name_tokens(name: str) -> List[str]:
    """Words of a tool name: snake_case, kebab-case and camelCase ("recruitee_candidateCreate" -> candidate, create)."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return [t for t in re.split(r"[_\-.\s]+", spaced.lower()) if t]


def is_write_tool(name: str) -> bool:
    """The first verb in the name decides: ``update_incident``, ``zoho_desk_update_ticket``, ``salesforce_case_create``
    are writes; ``find_incident_by_id``, ``google_sheets_get_many_rows`` are reads."""
    for token in _name_tokens(name):
        if token in READ_VERBS:
            return False
        if token in WRITE_VERBS:
            return True
    return (
        True  # no known verb: check it (a missed write skips every check; checking a read only costs a call)
    )


_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"), "-")
_DASHES.update(dict.fromkeys(map(ord, "\u00a0\u202f\u2007"), " "))


def canon(value: Any) -> str:
    """Text as compared by checks: NFKC, Unicode dashes as '-', non-breaking spaces as spaces (policy text and
    model output often differ only in these: 'product\u2011voc@' vs 'product-voc@')."""
    import unicodedata

    return unicodedata.normalize("NFKC", str(value)).translate(_DASHES)


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]", " ", canon(value).lower())).strip()


def _walk(obj: Any) -> Iterable[Any]:
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _is_id_key(key: str) -> bool:
    k = "_".join(_name_tokens(str(key)))
    return k == "id" or k.endswith("_id")


def _own_id(d: Dict[str, Any]) -> Optional[str]:
    for k, v in d.items():
        if "_".join(_name_tokens(str(k))) == "id" and isinstance(v, (str, int)):
            return str(v)
    for k, v in d.items():
        if _is_id_key(k) and isinstance(v, (str, int)):
            return str(v)
    return None


def _primary_id(d: Dict[str, Any]) -> Optional[str]:
    for k, v in d.items():
        if isinstance(v, str) and (k == "id" or k.endswith("_id") or k in ("number", "sys_id")):
            return v
    return None


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "d"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value))
    return float(m.group(0).replace(",", "")) if m else None


ISO_MOMENT = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}:?\d{2})?)?\s*$"
)


def _iso_moment(value: Any) -> Optional[datetime]:
    """An ISO 8601 date (``2025-02-25``) or date and time (``2025-02-25T12:08:00-05:00``) as a UTC moment, else None. Only
    a whole string in that form counts: a date inside other text, ``02/25/2025`` and a number are not ISO dates. A date
    with no time is midnight, and a time with no offset is read as UTC."""
    if not isinstance(value, str) or not ISO_MOMENT.match(value):
        return None
    try:
        moment = datetime.fromisoformat(value.strip().replace(" ", "T", 1).replace("Z", "+00:00"))
    except ValueError:  # 2025-13-45
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


def _condition(actual: Any, op: str, value: Any) -> bool:
    """The record-condition operators: contains, equals, in, gt, ge, lt, le, empty."""
    if op == "empty":
        return actual in (None, "", [], {})
    if op == "contains":
        return value is not None and str(value).lower() in str(actual or "").lower()
    if op == "equals":
        return norm(actual) == norm(value)
    if op == "in":
        return norm(actual) in {norm(v) for v in (value or [])}
    when_a, when_v = _iso_moment(actual), _iso_moment(value)
    if (
        when_a is not None and when_v is not None
    ):  # two ISO dates are compared as dates, not by their first number
        return {
            "gt": when_a > when_v,
            "ge": when_a >= when_v,
            "lt": when_a < when_v,
            "le": when_a <= when_v,
        }.get(op, False)
    a, v = (
        _number(actual),
        _number(value),
    )  # a number, or a value of another form: the first number, as before
    if a is None or v is None:
        return False
    return {"gt": a > v, "ge": a >= v, "lt": a < v, "le": a <= v}.get(op, False)


DUTY_VERBS = {
    "create",
    "update",
    "link",
    "add",
    "set",
    "new",
    "details",
    "to",
    "for",
    "get",
    "send",
    "record",
    "by",
    "incident",
}


def _values(slot: Any) -> Any:
    """A rule slot that should hold a list. A model may write one value alone (``"when": {"is_read": true}``): it counts
    as one value. Nothing counts as no value. A string stays as it is, so the text and checks of measured builds do not
    change."""
    if slot is None:
        return []
    return slot if isinstance(slot, (list, tuple, set, dict, str)) else [slot]


def _joined(separator: str, slot: Any) -> str:
    """The values of a rule slot as text. They need not be strings (``"allowed": [true]``)."""
    return separator.join(str(v) for v in _values(slot))


def _when(when: Optional[Dict[str, List[Any]]], args: Dict[str, Any]) -> bool:
    if not when:
        return True
    return all(
        (args.get(k) not in (None, ""))
        if "*" in [str(v) for v in _values(vals)]
        else norm(args.get(k)) in {norm(v) for v in _values(vals)}
        for k, vals in when.items()
    )


def duty_tools(then: Dict[str, Any], specs: Dict[str, Any]) -> set:
    """Tools that fulfil a duty: the named tool, or a tool for the same kind of record (a shared name word other than a
    verb) whose schema has every required argument ("pause the SLA" by updating it or by linking a new one)."""
    name = then.get("tool", "")
    need = set(then.get("when") or {})
    kind = set(re.findall(r"[a-z]+", name.lower())) - DUTY_VERBS
    out = {name}
    if not need or not kind:
        return out
    for t, spec in specs.items():
        props = set(((spec.get("inputSchema") or {}).get("properties") or {}))
        if need <= props and kind <= set(re.findall(r"[a-z]+", t.lower())):
            out.add(t)
    return out


@dataclass
class Facts:
    """What the agent has seen: the request, tool results, records by ID, and the principal."""

    text: str = ""
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    nodes: List[Dict[str, Any]] = field(default_factory=list)  # every record seen, for record_condition
    principal: Dict[str, Any] = field(default_factory=dict)

    def add_text(self, value: Any) -> None:
        self.text += " " + norm(value if isinstance(value, str) else json.dumps(value, default=str))

    def add_result(self, result: Any) -> None:
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                pass
        self.add_text(result)
        for node in _walk(result):
            if isinstance(node, dict):
                pid = _primary_id(node)
                if pid:
                    self.records.setdefault(pid, {}).update(node)
                if any(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in node.values()):
                    self.nodes.append(node)

    def named(self, token: str) -> List[Dict[str, Any]]:
        """Records that a call names with ``token``: by the record's own primary id first ("id", or its first "... id"
        field such as "Unit ID"); otherwise by an exact value in a field that is not an id of another record
        (``owner_id`` = 7 does not name every record owned by 7)."""
        own = [n for n in self.nodes if _own_id(n) == token]
        if own:
            return own
        exact = [
            n
            for n in self.nodes
            if any(str(v) == token for k, v in n.items() if isinstance(v, (str, int)) and not _is_id_key(k))
        ]
        if exact:
            return exact
        return [
            n
            for n in self.nodes
            if any(
                isinstance(v, str) and token in v and len(v) < 200 and not _is_id_key(k) for k, v in n.items()
            )
        ]

    def seen(self, value: Any) -> bool:
        v = norm(value)
        return bool(v) and v in self.text


@dataclass
class Check:
    rule: Dict[str, Any]
    ok: bool
    reason: str
    remedy: str = ""


class RuleSetHooks(AgentHooks):
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
        from .package import exception_links, runtime_rules

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

    # -- state ------------------------------------------------------------------------------------------------------
    def session_start(self, thread_id: Optional[str]) -> HookResult:
        self.facts = Facts(principal=self.principal)
        self.open_duties: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        # verdict log for the report; a tapped list stays the same list, so the audit keeps reading it
        self.records: List[Dict[str, Any]] = fresh(getattr(self, "records", None))
        self._denials: Dict[str, int] = {}
        # (rule id, case) of "block_once" rules that have stopped a call in this session: a rule stops once for each case
        self._blocked_once: set = set()
        self.called: List[str] = []
        from .facts import FactStore

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

    # -- hooks ------------------------------------------------------------------------------------------------------
    def on_user_message(self, message: str, thread_id: Optional[str]) -> HookResult:
        self.facts.add_text(message)
        self.request = message
        return HookResult()

    # -- routing by measured method choice ---------------------------------------------------------------------------
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

    @property
    def brief_guidance(self) -> bool:
        return self._brief_guidance

    @property
    def exception_links(self) -> Dict[str, List[str]]:
        return self._exception_links

    def pop_notes(self) -> List[str]:
        notes, self.notes = list(getattr(self, "notes", [])), []
        return notes

    def before_llm_call(self, thread_id: Optional[str]) -> HookResult:
        notes = self.pop_notes()
        if not self.brief:
            return HookResult(message="\n".join(notes) or None)
        lines = ["## Policy rules checked before each write (from the policy text)"]
        shown = set()
        entries: List[Dict[str, Any]] = []  # the rules of the brief, in order
        for t in sorted(self.tools_in_run):
            for r in self.by_tool.get(t, []):
                if r["id"] in shown or self._observe_only(r):
                    continue
                shown.add(r["id"])
                entries.append(r)
        for r in self.global_rules:
            if not self._observe_only(r):
                entries.append(r)
        if self._brief_guidance:
            entries += self._global_guidance
        shown_exceptions = 0
        nested = self._nested(entries)
        reached: set = set()  # the rules written under another rule
        # A rule that is the exception of another rule of the brief is written under each rule it is an exception of, and not
        # as an item of its own. The rules of a cycle of exceptions have no first rule, so they come last, each once.
        top = [e for e in entries if id(e) not in nested]
        for first in (True, False):
            for r in top if first else [e for e in entries if id(e) in nested]:
                if not first and id(r) in reached:
                    continue  # a cycle that a rule above leads to
                lines.append(f"- [{r.get('source', '')}] {self._describe(r)}")
                reached.add(id(r))
                shown_exceptions += self._exception_lines(r, nested, lines, reached, {id(r)}, "  ")
        for trig, duty in self.open_duties if self._duty_follow_up else ():
            lines.append(f"- OPEN DUTY [{duty.get('source', '')}]: {self._describe(duty)}")
        if len(lines) == 1 and not notes:
            return HookResult()
        brief = "\n".join(notes + (lines if len(lines) > 1 else []))
        detail: Dict[str, Any] = {"rule": "brief", "lines": len(lines) - 1, "notes": len(notes)}
        if shown_exceptions:
            detail["exceptions"] = shown_exceptions
        return HookResult(message=brief, details=[detail])

    def _nested(self, entries: List[Dict[str, Any]]) -> set:
        """``id()`` of the rules written under another rule of the brief: the exceptions of a rule of the brief
        (``exception_links``), and theirs in turn. A rule the agent cannot see (observe-only) is not shown as an
        exception either."""
        if not self._exception_links:
            return set()
        by_id = {r.get("id"): r for r in self.rules}
        nested: set = set()
        work = list(entries)
        while work:
            r = work.pop()
            for child in self._exception_links.get(r.get("id"), []):
                c = by_id.get(child)
                if c is None or c is r or id(c) in nested or self._observe_only(c):
                    continue
                nested.add(id(c))
                work.append(c)
        return nested

    def _exception_lines(
        self, rule: Dict[str, Any], nested: set, lines: List[str], reached: set, path: set, indent: str
    ) -> int:
        """The exceptions of one rule under it, each with its own exceptions: under every rule it is an exception of. ``path``
        holds the rules above this one, so a cycle ends. Returns how many lines it wrote."""
        by_id = {r.get("id"): r for r in self.rules}
        count = 0
        for child in self._exception_links.get(rule.get("id"), []):
            c = by_id.get(child)
            if c is None or id(c) not in nested or id(c) in path:
                continue
            reached.add(id(c))
            lines.append(f"{indent}- exception [{c.get('source', '')}]: {self._describe(c)}")
            count += 1 + self._exception_lines(c, nested, lines, reached, path | {id(c)}, indent + "  ")
        return count

    @audited_failure
    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> HookResult:
        if not is_write_tool(tool):
            self._last = "n/a"
            return HookResult()
        return self._decide(tool, args, self._checks(tool, args))

    # -- the stages of ``WriteChecks`` (levels.py): the same decision as ``before_tool_call``, with the verifier apart ---------
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

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> HookResult:
        self.facts.add_result(result)
        self.store.add(tool, result)
        self.called.append(tool)
        failed = isinstance(result, dict) and "error" in result
        if failed:
            return HookResult(message=self._tool_notes())
        details = []
        # a call may close an open duty
        still = []
        for trig, duty in self.open_duties:
            then = duty["then"]
            if tool in self._duty_tools(then) and _when(then.get("when"), args):
                details.append(
                    self._record("after_tool_call", tool, Check(duty, True, "duty done"), "duty_done")
                )
            else:
                still.append((trig, duty))
        self.open_duties = still
        for r in self.by_tool.get(tool, []):
            if r["type"] == "duty" and _when(r["trigger"].get("when"), args):
                if (
                    not self._duty_tools(r["then"]) & self.tools_in_run
                ):  # the agent has no tool to do it in this run
                    details.append(
                        self._record(
                            "after_tool_call", tool, Check(r, True, "duty tool not available"), "duty_skipped"
                        )
                    )
                    continue
                self.open_duties.append(({"tool": tool, "args": args}, r))
                details.append(
                    self._record("after_tool_call", tool, Check(r, True, "duty opened"), "duty_open")
                )
        return HookResult(message=self._tool_notes(), details=details)

    def _tool_notes(self) -> Optional[str]:
        """Queued reminders, handed to the tool result when notes go there (see ``note_channel``)."""
        if self.note_channel != "tool":
            return None
        return "\n".join(self.pop_notes()) or None

    def _duty_tools(self, then: Dict[str, Any]) -> set:
        return duty_tools(then, self.specs)

    @audited_failure
    def before_reply(self, answer: str) -> HookResult:
        if not self.open_duties or self.mode == "shadow" or not self._duty_follow_up:
            return HookResult()
        lines = ["POLICY: before you finish, these duties from the policy are still open:"]
        details = []
        for trig, duty in self.open_duties:
            unless = "; ".join(u.get("condition", "") for u in duty.get("unless") or [])
            if unless:  # the policy has an exception: the agent decides whether it applies
                lines.append(f"- [{duty.get('source', '')}] {self._describe(duty)} (not needed if: {unless})")
                details.append(
                    self._record(
                        "before_reply", trig["tool"], Check(duty, False, "duty with exception"), "remind"
                    )
                )
                continue
            lines.append(f"- [{duty.get('source', '')}] {self._describe(duty)}")
            details.append(
                self._record(
                    "before_reply", trig["tool"], Check(duty, False, "duty open at reply"), "followup"
                )
            )
        return HookResult(message="\n".join(lines), details=details)

    # -- checks -----------------------------------------------------------------------------------------------------
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

    def _check_record_condition(self, r, tool, args) -> Optional[Check]:
        if not _when(r.get("when"), args):
            return None
        from .facts import call_tokens  # identifiers in the call's arguments

        fld, op, value = r.get("field", ""), r.get("op", "contains"), r.get("value")
        seen_any = False
        for tok in call_tokens(args):
            for rec in self.facts.named(tok):
                key = next((k for k in rec if k.lower() == str(fld).lower()), None)
                if key is None:
                    continue
                seen_any = True
                if _condition(rec.get(key), op, value):
                    return Check(
                        r,
                        False,
                        f"record {tok} has {key} {op} {value!r}",
                        f"do not act on {tok}; leave it out (policy condition on {key})",
                    )
        return Check(r, True, f"named records do not meet {fld} {op} {value!r}") if seen_any else None

    def _check_threshold(self, r, tool, args) -> Optional[Check]:
        arg = r.get("arg")
        num = _number(args.get(arg))
        if num is None or not _condition(num, r.get("op", "gt"), r.get("value")):
            return None if num is None else Check(r, True, f"{arg} within the limit")
        if r.get("then") == "requires_approval" and r.get("approval_tool") in self.called:
            return Check(r, True, f"{arg} above the limit, approval requested before")
        if r.get("then") == "requires_approval":
            return Check(
                r,
                False,
                f"`{arg}={args.get(arg)}` needs approval ({r.get('op')} {r.get('value')})",
                "request approval first"
                + (f" with `{r['approval_tool']}`" if r.get("approval_tool") else ""),
            )
        return Check(
            r,
            False,
            f"`{arg}={args.get(arg)}` is over the limit ({r.get('op')} {r.get('value')})",
            f"keep {arg} within the limit or stop and explain",
        )

    def _check_format(self, r, tool, args) -> Optional[Check]:
        arg, pattern = r.get("arg"), r.get("pattern", "")
        if arg not in args or args[arg] in (None, ""):
            return None
        try:  # free text (a message, a note) must contain the pattern; a value field must match it whole
            text, pattern = canon(args[arg]), canon(pattern)
            free = bool(FREE_TEXT.search(arg)) or len(text.split()) > 3
            ok = (
                re.search(pattern, text) if free else re.fullmatch(pattern, text, flags=re.DOTALL)
            ) is not None
        except re.error:
            return None
        return (
            Check(r, True, f"{arg} format ok")
            if ok
            else Check(
                r,
                False,
                f"`{arg}={args[arg]}` does not have the required format",
                f"write {arg} in the form {pattern}",
            )
        )

    def _check_required_args(self, r, tool, args) -> Optional[Check]:
        if not _when(r.get("when"), args):
            return None
        missing = [a for a in r.get("args", []) if args.get(a) in (None, "", [])]
        if missing:
            return Check(
                r,
                False,
                f"missing mandatory field(s) {_joined(', ', missing)}",
                f"look up the value(s) and pass {_joined(', ', missing)}; if the data does not exist, stop and "
                f"state which field is required",
            )
        return Check(r, True, "mandatory fields present")

    def _check_allowed_values(self, r, tool, args) -> Optional[Check]:
        arg = r.get("arg")
        if arg not in args or not _when(r.get("when"), args):
            return None
        allowed = {norm(str(v).lstrip("#@")) for v in r.get("values", [])}
        if norm(str(args[arg]).lstrip("#@")) not in allowed and not self._names_allowed(args[arg], allowed):
            if r.get("schema_conflict"):
                return Check(
                    r,
                    False,
                    f"the policy lists {_joined(', ', r.get('values'))} for `{arg}`, but the "
                    f"tool accepts other values",
                    f"keep a value the tool accepts for {arg}; do not drop {arg}",
                )
            return Check(
                r,
                False,
                f"`{arg}={args[arg]}` is not allowed here; allowed: {_joined(', ', r.get('values'))}",
                f"use one of the allowed values for {arg}",
            )
        return Check(r, True, f"{arg} allowed")

    def _names_allowed(self, value: Any, allowed: set) -> bool:
        """An id whose record carries an allowed name ("C_SALES_ALERTS" with name "sales-alerts") is allowed."""
        for rec in self.facts.named(str(value)):
            for k, v in rec.items():
                if (
                    isinstance(v, str)
                    and re.search(r"name|title|label|handle", k, re.I)
                    and norm(v.lstrip("#@")) in allowed
                ):
                    return True
        return False

    def _check_derived_value(self, r, tool, args) -> Optional[Check]:
        arg = r.get("arg")
        if arg not in args:
            return None
        inputs = r.get("inputs", [])
        vals = []
        for i in inputs:
            v = args.get(i)
            if v is None:
                rec = self._record_for(args)
                v = rec.get(i) if rec else None
            vals.append(norm(v) if v is not None else None)
        if None in vals:
            return Check(
                r,
                False,
                f"`{arg}` is computed from {_joined(', ', inputs)}, which are not known for this call",
                f"do not set {arg}; the system computes it",
            )
        want = (r.get("table") or {}).get("|".join(vals))
        if want is not None and norm(args[arg]) != norm(want):
            return Check(
                r,
                False,
                f"`{arg}={args[arg]}` does not match the policy table ({' x '.join(vals)} → {want})",
                f"do not set {arg}, or set it to {want}",
            )
        return Check(r, True, f"{arg} matches the table")

    def _record_for(self, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for k, v in args.items():
            if isinstance(v, str) and (k == "id" or k.endswith("_id")) and v in self.facts.records:
                return self.facts.records[v]
        return None

    def _check_transition(self, r, tool, args) -> Optional[Check]:
        arg, id_arg = r.get("arg"), r.get("id_arg")
        if arg not in args:
            return None
        rec = self.facts.records.get(str(args.get(id_arg))) if id_arg else None
        current = rec.get(arg) if rec else None
        if current is None:
            return Check(
                r,
                False,
                f"the current `{arg}` of {args.get(id_arg)} is not known, so the change to "
                f"`{args[arg]}` cannot be checked",
                f"read the record first, then change {arg}",
            )
        edges = {norm(k): {norm(x) for x in v} for k, v in (r.get("edges") or {}).items()}
        if norm(current) == norm(args[arg]):
            return Check(r, True, "no change")
        if norm(args[arg]) not in edges.get(norm(current), set()):
            allowed = ", ".join(sorted(edges.get(norm(current), set()))) or "none"
            return Check(
                r,
                False,
                f"`{arg}` cannot go from {current} to {args[arg]}; allowed from {current}: {allowed}",
                "use an allowed transition (possibly through an intermediate state)",
            )
        return Check(r, True, f"{current} → {args[arg]} allowed")

    def _check_reference_status(self, r, tool, args) -> Optional[Check]:
        arg = r.get("arg")
        if arg not in args:
            return None
        rec = self.facts.records.get(str(args[arg]))
        status = rec.get(r.get("field")) if rec else None
        if status is None:
            return None  # unknown: id_provenance covers lookups
        if norm(status) not in {norm(v) for v in r.get("allowed", [])}:
            return Check(
                r,
                False,
                f"{args[arg]} has {r.get('field')}={status}; allowed: {_joined(', ', r.get('allowed'))}",
                f"do not use {args[arg]} here",
            )
        return Check(r, True, f"{arg} status allowed")

    def _check_membership(self, r, tool, args) -> Optional[Check]:
        """The record named by ``arg`` (e.g. the assignee) must be a member of the one named by ``group_arg``, as the
        data read shows: allowed when one record links both; denied when the data lists other members of that group
        and none is this one; not decided when the group's members were never read."""
        member, group = args.get(r.get("arg")), args.get(r.get("group_arg"))
        if member in (None, "") or group in (None, ""):
            return None
        m, g = str(member), str(group)

        def values(n: Dict[str, Any]) -> List[str]:
            out = []
            for v in n.values():
                out += (
                    [str(x) for x in v]
                    if isinstance(v, list)
                    else [str(v)]
                    if isinstance(v, (str, int))
                    else []
                )
            return out

        names_group = [n for n in self.facts.nodes if g in values(n)]
        if any(m in values(n) for n in names_group):
            return Check(r, True, f"{m} is a member of {g}")
        others = [
            n
            for n in names_group
            if any(
                ("user" in k.lower() or "member" in k.lower()) and v not in (None, "", m)
                for k, v in n.items()
            )
        ]
        if not others:
            return None
        return Check(
            r,
            False,
            f"the members of {g} that were read do not include {m}",
            f"choose a member of {g}, or add {m} to the group first if the request asks for it",
        )

    def _check_grounded_values(self, r, tool, args, demanded) -> Optional[Check]:
        props = (self.specs.get(tool, {}).get("inputSchema") or {}).get("properties") or {}
        required = set((self.specs.get(tool, {}).get("inputSchema") or {}).get("required") or [])
        bad = []
        for k, v in args.items():
            if k in required or k in demanded or isinstance(v, (bool, dict, list)) or v in (None, ""):
                continue
            prop = props.get(k, {})
            if not prop.get("enum") and (FREE_TEXT.search(k) or len(str(v).split()) > 3):
                continue  # free text is written by the agent
            if ID_VALUE.match(str(v)):
                continue  # IDs are checked by id_provenance
            if not self.facts.seen(v) and not (prop.get("enum") and self._stem_seen(v)):
                bad.append(f"{k}={v}")
        if bad:
            return Check(
                r,
                False,
                f"optional value(s) {', '.join(bad)} do not come from the request or from any "
                f"tool result (no assumed data)",
                "leave these arguments out, or use values stated in the request or returned by a tool",
            )
        return Check(r, True, "optional values grounded")

    def _stem_seen(self, value: Any) -> bool:
        """A tool option counts as stated when its word stem appears in the request or data ("retire" -> "retired")."""
        target = _stem(norm(value))
        return bool(target) and target in {_stem(w) for w in re.findall(r"[a-z]+", self.facts.text)}

    def _check_id_provenance(self, r, tool, args, demanded) -> Optional[Check]:
        bad = [
            f"{k}={v}"
            for k, v in args.items()
            if isinstance(v, str) and ID_VALUE.match(v) and not self.facts.seen(v)
        ]
        if bad:
            return Check(
                r,
                False,
                f"ID(s) {', '.join(bad)} were not returned by any tool and are not in the request",
                "look the record up first and use the ID it returns",
            )
        return Check(r, True, "IDs come from data")

    def _role_check(self, tool: str) -> Optional[Check]:
        rules = [r for r in self.by_tool.get(tool, []) if r["type"] == "role_permission"]
        if not rules or self.cedar is None:
            return None
        role = self.principal.get("role")
        if not role:
            return Check(rules[0], False, "the user's role is not known", "check the user information first")
        principal = f'User::"{self.principal.get("id", "me")}"'
        dec = self.cedar.is_authorized(
            principal,
            f'Action::"{tool}"',
            'Resource::"any"',
            [
                {
                    "uid": {"type": "User", "id": self.principal.get("id", "me")},
                    "attrs": {"role": role},
                    "parents": [],
                }
            ],
        )
        decided = next((r for r in rules if r["id"] in dec.policy_ids), rules[0])
        if not dec.allowed:
            return Check(
                decided,
                False,
                f"role `{role}` may not call `{tool}`",
                "stop and state the policy reason; do not try another way to make this change",
            )
        return Check(decided, True, f"role {role} allowed")

    @staticmethod
    def _cedar(rules: List[Dict[str, Any]]):
        if not rules:
            return None
        try:
            from .cedar_engine import CedarEngine
        except ImportError:  # pragma: no cover
            return None
        text = ['@id("base_permit")\npermit(principal, action, resource);']
        for r in rules:
            acts = ", ".join(f'Action::"{t}"' for t in r.get("tools", []))
            roles = ", ".join(json.dumps(x) for x in r.get("roles", []))
            text.append(
                f'@id("{r["id"]}")\nforbid(principal, action in [{acts}], resource) '
                f"when {{ [{roles}].contains(principal.role) }};"
            )
        try:
            return CedarEngine("\n".join(text))
        except ImportError:  # pragma: no cover
            return None

    @staticmethod
    def _describe(r: Dict[str, Any]) -> str:
        t = r["type"]
        when = r["trigger"].get("when") if t == "duty" else r.get("when")
        cond = (" when " + ", ".join(f"{k} is {_joined('/', v)}" for k, v in when.items())) if when else ""
        if t == "required_args":
            return f"`{r['tool']}`{cond}: must include {_joined(', ', r.get('args'))}"
        if t == "allowed_values":
            return f"`{r['tool']}`{cond}: {r['arg']} must be one of {_joined(', ', r.get('values'))}"
        if t == "derived_value":
            return f"`{r['tool']}`: {r['arg']} is computed from {_joined(' x ', r.get('inputs'))}; do not guess it"
        if t == "transition":
            edges = "; ".join(f"{k} → {_joined('/', v)}" for k, v in (r.get("edges") or {}).items())
            return f"`{r['tool']}`: {r['arg']} transitions allowed: {edges}"
        if t == "reference_status":
            return (
                f"`{r['tool']}`: {r['arg']} must have {r.get('field')} in {_joined(', ', r.get('allowed'))}"
            )
        if t == "membership":
            return f"`{r['tool']}`: {r['arg']} must be a member of the {r['group_arg']} given"
        if t == "role_permission":
            return f"roles {_joined(', ', r.get('roles'))} may not call {_joined(', ', r.get('tools'))}"
        if t == "grounded_values":
            return "do not set optional arguments unless the request or a tool result gives the value"
        if t == "id_provenance":
            return "use only IDs returned by tools or given in the request"
        if t == "duty":
            th = r["then"]
            tw = (
                (" with " + ", ".join(f"{k}={_joined('/', v)}" for k, v in th.get("when", {}).items()))
                if th.get("when")
                else ""
            )
            return f"after `{r['trigger']['tool']}`{cond}: also call `{th['tool']}`{tw}"
        if t == "record_condition":
            return (
                f"{', '.join('`%s`' % x for x in r.get('tools', []))}{cond}: not for a record whose "
                f"{r.get('field')} {r.get('op')} {r.get('value')!r}"
            )
        if t == "threshold":
            then = "needs approval first" if r.get("then") == "requires_approval" else "is not allowed"
            return f"`{r['tool']}`: {r['arg']} {r.get('op')} {r.get('value')} {then}"
        if t == "format":
            return f"`{r['tool']}`: {r['arg']} must match {r.get('pattern')}"
        if t == "verify":
            return r.get("statement", "")
        return r.get("text", "")


def obligations_from_duties(
    rules: Iterable[Dict[str, Any]],
    specs: Optional[Iterable[Dict[str, Any]]] = None,
    level: str = "observe",
) -> List[Any]:
    """The ``duty`` rules as obligations of the producer ``duty`` (design section 5): "after this write, also do that".

    An item opens at its trigger write (``scope.trigger``: ``tools`` and ``when``, tested on the trigger's arguments) and is
    met by a later successful call of the ``then`` tool or of a tool for the same kind of record (``duty_tools``). A value of
    ``then.when`` that the rule names alone becomes a ``literal`` that the call must carry in that argument; a list of
    values or ``"*"`` is not compared. An exception of the rule (``unless``) stays in the text of the item, as in the old
    follow-up: the agent decides whether it applies. Nothing here reads data or secrets: only the rule. ``specs``: the tool
    specs of the run, for the tools of the same kind; ``level``: the level of the items."""
    from .model import Obligation

    by_name = {t["name"]: t for t in specs or [] if isinstance(t, dict) and t.get("name")}
    out: List[Any] = []
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("type") != "duty":
            continue
        trigger, then = rule.get("trigger") or {}, rule.get("then") or {}
        if not trigger.get("tool") or not then.get("tool"):
            continue
        tools = sorted(duty_tools(then, by_name), key=lambda t: (t != then["tool"], t))
        match: Dict[str, Any] = {}
        values: Dict[str, Any] = {}
        for arg, wanted in (then.get("when") or {}).items():
            alone = [str(v) for v in ([wanted] if isinstance(wanted, str) else _values(wanted))]
            if len(alone) != 1 or alone[0] == "*" or not alone[0].strip():
                continue
            name = re.sub(r"\W", "_", str(arg))
            if not re.match(r"[A-Za-z_]", name):
                name = f"v_{name}"
            match[str(arg)] = {"pattern": "{" + name + "}"}
            values[name] = {"from": "literal", "value": alone[0]}
        text = RuleSetHooks._describe(rule)
        unless = "; ".join(
            str(u.get("condition", "")) for u in rule.get("unless") or [] if isinstance(u, dict)
        )
        if unless:
            text = f"{text} (not needed if: {unless[:160]})"
        out.append(
            Obligation(
                id=f"duty_{rule.get('id') or len(out)}",
                origin="duty",
                scope={"trigger": {"tools": [trigger["tool"]], "when": trigger.get("when") or {}}},
                effect={"kind": "write", "tools": tools, "match": match},
                values=values,
                check="code",
                level=level,
                source={"section": str(rule.get("source") or ""), "quote": text},
            )
        )
    return out
