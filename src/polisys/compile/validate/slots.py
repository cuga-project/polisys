"""The slots of each rule type in a policy package: which slots a type has, what each may hold, and what is wrong with the
slots of one rule (``slot_problem``). Also the names shared by the package checks: ``PackageError``, the outcomes and the
warning codes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from polisys.checks.rules import _iso_moment, _number
from polisys.package.model import RULE_CLASSES, RULE_LEVELS, Rule
from polisys.compile.validate.build import ValidationError

COMPLETENESS = {"coverage_incomplete", "rule_no_trigger"}  # gaps: a warning for a converted legacy package
OUTCOMES = (
    "hard_rule",
    "judgment_rule",
    "obligation",
    "procedure_step",
    "guidance",
    "definition",
    "out_of_scope",
    "question",
)  # design section 9: what can become of a statement
# Coverage kinds written before the package, and the outcome each stands for. Others (dangling_link, link,
# classify_fallback, conflict, ...) are notes about a statement, not outcomes.
LEGACY_OUTCOME = {
    "rule": "hard_rule",
    "verify": "judgment_rule",
    "guidance": "guidance",
    "definition": "definition",
    "step": "procedure_step",
    "procedure_rule": "procedure_step",  # a precondition, exception, template or escalation of a procedure
    "superseded": "out_of_scope",  # each has a reason: "replaced by ..."
    "untrusted": "out_of_scope",
}
# A quote of at least this many words ties an obligation, or a part of a procedure, to the statement it was taken from.
MIN_QUOTE_WORDS = 4
GLOBAL_TYPES = ("grounded_values", "id_provenance", "guidance")  # no trigger tool by design
KNOWN_EXTRA = {"stats", "id"}  # package ``extra`` keys that ``package/files.py`` writes back to a source format
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ].*)?$")
# Codes whose findings are always warnings. ``rule_conflict`` and the legacy gaps have their severity set where they arise.
WARNING_CODES = {
    "schema_conflict",
    "unknown_priority_class",
    "unknown_field",
    "empty_value",
    "coverage_conflict",
    "unbound_value",
    "verify_without_why",
    "not_in_text",
    "bare_must_not",
    "date_compare",
    "exception_without_rule",
}
PLACEHOLDER = re.compile(r"^\s*(todo|tbd|fixme|xxx|\?+|\.\.\.|<[^<>]*>|\{\{.*\}\})\s*$", re.I)
OBLIGATION_ENUMS = {
    "origin": ("package", "request", "data", "duty"),
    "check": ("code", "kev", "model"),
    "level": ("observe", "show", "follow_up"),
}
EFFECT_KINDS = ("write", "file", "message", "reply")
VALUE_ORIGINS = ("literal", "template", "computed", "case_fact", "request")


@dataclass
class PackageError(ValidationError):
    """A finding of ``validate_package``. ``severity``: "error" fails ``validate``; "warning" is only reported."""

    severity: str = "error"


def _show(value: Any) -> str:
    return f"{type(value).__name__} {json.dumps(value, default=str, ensure_ascii=False)[:40]}"


# Slot kinds: each takes a value and returns what follows the slot name in the message when it has the wrong type
# (": must be ..." or ".key: must be ..."), or None.
def _is_str(v: Any) -> Optional[str]:
    return None if isinstance(v, str) else f": must be a string, got {_show(v)}"


def _is_text(v: Any) -> Optional[str]:
    """A string in the policy's own words (a sentence, a term): ``<main_policy>`` can be real text there."""
    return _is_str(v)


def _is_strs(v: Any) -> Optional[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        return f": must be a list of strings, got {_show(v)}"
    return None


def _is_scalar(v: Any) -> Optional[str]:
    return None if isinstance(v, (str, int, float, bool)) else f": must be one value, got {_show(v)}"


def _is_scalars(v: Any) -> Optional[str]:
    if not isinstance(v, list) or not all(isinstance(x, (str, int, float, bool)) for x in v):
        return f": must be a list of values, got {_show(v)}"
    return None


def _is_number(v: Any) -> Optional[str]:
    """A number, or text that holds one: what ``ruleset._number`` reads."""
    if isinstance(v, (int, float, str)) and _number(v) is not None:
        return None
    return f": must be a number, got {_show(v)}"


def _is_lists_by_name(v: Any) -> Optional[str]:
    """``when`` and ``edges``: an object whose every value is a list. A bare value there counts as one value or as
    its characters, depending on its type, so it is never what the author meant."""
    if not isinstance(v, dict):
        return f": must be an object of name -> list of values, got {_show(v)}"
    for key, items in v.items():
        if not isinstance(items, list):
            return f".{key}: must be a list of values, got {_show(items)}"
    return None


def _is_dict(v: Any) -> Optional[str]:
    return None if isinstance(v, dict) else f": must be an object, got {_show(v)}"


def _is_unless(v: Any) -> Optional[str]:
    """Exceptions as ``fold_exceptions`` writes them: objects with a text ``condition`` (``before_reply`` and the verifier
    join the conditions as text)."""
    if not isinstance(v, list) or not all(
        isinstance(x, dict) and isinstance(x.get("condition", ""), str) for x in v
    ):
        return f": must be a list of objects with a text condition, got {_show(v)}"
    return None


def _is_level(v: Any) -> Optional[str]:
    return (
        None
        if isinstance(v, str) and v in RULE_LEVELS
        else f": must be one of {', '.join(RULE_LEVELS)}, got {_show(v)}"
    )


KINDS: Dict[str, Callable[[Any], Optional[str]]] = {
    "str": _is_str,
    "text": _is_text,
    "strs": _is_strs,
    "scalar": _is_scalar,
    "values": _is_scalars,
    "number": _is_number,
    "when": _is_lists_by_name,
    "edges": _is_lists_by_name,
    "dict": _is_dict,
    "unless": _is_unless,
    "level": _is_level,
    "any": lambda v: None,
}
# Slots shared by every rule type: ``_describe``, ``_choice``, ``add_rules``, ``before_llm_call`` and ``fold_exceptions``
# read them whatever the type is. A tool name is never "required" here: an empty one is ``rule_no_trigger``.
_WHEN = ("when", False)
COMMON_SLOTS: Dict[str, Tuple[Any, bool]] = {
    "id": (
        "str",
        True,
    ),  # ``package.runtime_rules`` copies the rule dict as it is, and the hooks read ``r["id"]``
    "quote": ("text", False),
    "types": ("strs", False),
    "when": _WHEN,
    "unless": ("unless", False),
    "tool": ("str", False),
    "tools": ("strs", False),
}
GLOBAL_RULES = ("grounded_values", "id_provenance")  # apply to every write: a tool would make them per tool
_TRIGGER = {"tool": ("str", False), "when": _WHEN}
_THEN = {"tool": ("str", True), "when": _WHEN}
# What each type adds to ``COMMON_SLOTS``: name -> (kind, required). A nested table is an object with its own slots.
# Types as in ``checks/rules.py``.
_OWN_SLOTS: Dict[str, Dict[str, Tuple[Any, bool]]] = {
    "required_args": {"args": ("strs", True)},
    "allowed_values": {"arg": ("str", True), "values": ("values", True)},
    "derived_value": {"arg": ("str", True), "inputs": ("strs", True), "table": ("dict", True)},
    "transition": {"arg": ("str", True), "id_arg": ("str", True), "edges": ("edges", True)},
    "reference_status": {"arg": ("str", True), "field": ("str", True), "allowed": ("values", True)},
    "membership": {"arg": ("str", True), "group_arg": ("str", True)},
    "role_permission": {"roles": ("strs", True), "effect": ("str", False)},
    "grounded_values": {},
    "id_provenance": {},
    "duty": {"trigger": (_TRIGGER, False), "then": (_THEN, True)},
    "guidance": {"text": ("text", True)},
    "record_condition": {
        "field": ("str", True),
        "op": ("str", True),
        "value": ("any", False),  # by op: see ``_value_findings``
        "effect": ("str", False),
    },
    "threshold": {
        "arg": ("str", True),
        "op": ("str", True),
        "value": ("number", True),
        "then": ("str", True),
        "approval_tool": ("str", False),
    },
    "format": {"arg": ("str", True), "pattern": ("str", True)},
    "verify": {"statement": ("text", True)},
}
RULE_SLOTS: Dict[str, Dict[str, Tuple[Any, bool]]] = {
    t: {
        **{k: v for k, v in COMMON_SLOTS.items() if t not in GLOBAL_RULES or k not in ("tool", "tools")},
        **own,
    }
    for t, own in _OWN_SLOTS.items()
}
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
DATE_LIKE = re.compile(  # a value that reads as a date: 2025-02-25, 2025-02, 02/25/2025, 25.02.2025, Feb 25, 25 February
    rf"\b\d{{4}}-\d{{1,2}}(-\d{{1,2}})?\b|\b\d{{1,2}}[/.]\d{{1,2}}[/.]\d{{2,4}}\b|\b\d{{4}}[/.]\d{{1,2}}[/.]\d{{1,2}}\b"
    rf"|\b({MONTHS})[a-z]*\.?\s+\d{{1,2}}\b|\b\d{{1,2}}\s+({MONTHS})[a-z]*\b",
    re.I,
)
# What ``record_condition`` needs as ``value``, by ``op`` (``ruleset._condition``). ``empty`` ignores it.
VALUE_KIND = {
    "in": "values",
    "contains": "scalar",
    "equals": "scalar",
    "gt": "number",
    "ge": "number",
    "lt": "number",
    "le": "number",
}


def _severity(code: str) -> str:
    return "warning" if code in WARNING_CODES else "error"


def _is_broken(findings: List[Tuple[str, str]]) -> bool:
    """A rule with a finding that is not a warning cannot be read safely by the checks that follow."""
    return any(code not in WARNING_CODES for code, _ in findings)


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, (str, list, dict)) and not value)


def _blank(value: Any) -> Optional[str]:
    """Why a text value is empty or a placeholder (a skill that left a slot to fill), or None."""
    if isinstance(value, str):
        if not value.strip():
            return "is empty"
        if PLACEHOLDER.match(value):
            return f"is a placeholder: {value!r}"
    return None


def _unfilled(kind: str, value: Any, here: str) -> List[Tuple[str, str]]:
    """(code, message) for empty or placeholder elements of a list, of a ``when`` or of a table, in a value of the
    right type. An empty string among ``values`` can be the value: "the description starts empty" is a real rule. It is
    an ``empty_value`` warning there, and an ``unfilled_slot`` error anywhere else."""
    out: List[Tuple[str, str]] = []
    items: List[Tuple[str, Any]] = []
    if kind in ("strs", "values"):
        items = [(f"{here}[{i}]", x) for i, x in enumerate(value)]
    elif kind in ("when", "edges"):
        for key, values in value.items():
            if kind == "when" and not values:
                out.append(("unfilled_slot", f"{here}.{key} is empty, so the rule would never apply"))
            items += [(f"{here}.{key}[{i}]", x) for i, x in enumerate(values)]
    elif kind == "dict":
        items = [(f"{here}.{key}", x) for key, x in value.items()]
    for where, x in items:
        why = _blank(x)
        if why == "is empty" and kind == "values":
            out.append(("empty_value", f"{where} is an empty string: right only if the value must be empty"))
        elif why:
            out.append(("unfilled_slot", f"{where} {why}"))
    return out


def _check_slots(
    slots: Dict[str, Tuple[Any, bool]], data: Dict[str, Any], path: str
) -> List[Tuple[str, str]]:
    """(code, message) for each slot that is empty or a placeholder (``unfilled_slot``) or has the wrong type
    (``invalid_slots``)."""
    out: List[Tuple[str, str]] = []
    for name, (kind, required) in slots.items():
        value, here = data.get(name), f"{path}{name}"
        if _empty(value):
            if required:
                out.append(("unfilled_slot", f"{here} is empty"))
            continue
        if isinstance(kind, dict):
            if not isinstance(value, dict):
                out.append(("invalid_slots", f"{here}: must be an object, got {_show(value)}"))
            else:
                out += _check_slots(kind, value, here + ".")
            continue
        why = KINDS[kind](value)
        if why:
            out.append(("invalid_slots", f"{here}{why}"))
            continue
        if kind == "str" and _blank(value):
            out.append(("unfilled_slot", f"{here} {_blank(value)}"))
        else:
            out += _unfilled(kind, value, here)
    return out


def _trigger_tools(rule: Dict[str, Any]) -> List[str]:
    """The tools that make a rule apply, as ``RuleSetHooks.add_rules`` reads them (slots of the right type)."""
    if rule.get("type") == "duty":
        trigger = rule.get("trigger")
        tool = trigger.get("tool") if isinstance(trigger, dict) else None
        return [tool] if isinstance(tool, str) and tool else []
    tools, tool = rule.get("tools"), rule.get("tool")
    if isinstance(tools, list) and tools:
        return [t for t in tools if isinstance(t, str)]
    return [tool] if isinstance(tool, str) and tool else []


def _value_findings(rule: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``record_condition``: ``value`` has the type its ``op`` reads (a list for ``in``, a number for ``gt``, ...)."""
    kind = VALUE_KIND.get(rule.get("op")) if isinstance(rule.get("op"), str) else None
    if kind is None:  # ``empty`` ignores the value; an unknown op is reported by check_rule
        return []
    out = _check_slots({"value": (kind, True)}, rule, "")
    value = rule.get("value")
    if kind == "number" and isinstance(value, str) and DATE_LIKE.search(value) and _iso_moment(value) is None:
        out.append(
            (
                "date_compare",
                f"`{rule.get('op')}` compares two ISO dates (2025-02-25) as dates, but {value!r} is not one, so it is read by "
                "its first number (the year or the day): write the date in ISO form, or do not compare it in a record_condition",
            )
        )
    return out


def _threshold_findings(rule: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``threshold``: it reads the call's argument as a number (the first number of the text), so a limit that is a date is
    compared by its year or day only, in ISO form or not."""
    value = rule.get("value")
    if isinstance(value, str) and DATE_LIKE.search(value):
        return [
            (
                "date_compare",
                f"threshold reads the argument as a number, so the date {value!r} is compared by its first number only. Dates "
                "are compared only by a record_condition, in ISO form",
            )
        ]
    return []


def _slot_findings(r: Rule) -> List[Tuple[str, str]]:
    """Wrong-typed or empty slots of one rule, and a missing trigger tool. A rule of an unknown type has no table:
    ``check_rule`` reports it."""
    rule = r.rule
    if not isinstance(rule, dict):
        return [("invalid_slots", f"rule must be an object, got {_show(rule)}")]
    out: List[Tuple[str, str]] = []
    if rule.get("type", r.type) != r.type:
        out.append(
            ("invalid_slots", f"type {r.type!r} differs from the type in the rule: {rule.get('type')!r}")
        )
    slots = RULE_SLOTS.get(r.type)
    if slots is None:
        return out
    out += _check_slots(slots, rule, "")
    if r.type in GLOBAL_RULES:
        for name in ("tool", "tools"):
            if not _empty(rule.get(name)):
                out.append(
                    ("invalid_slots", f"{name}: a {r.type} rule applies to every write and takes no tool")
                )
    if r.type == "record_condition":
        out += _value_findings(rule)
    if r.type == "threshold":
        out += _threshold_findings(rule)
    if not any(code == "invalid_slots" for code, _ in out) and r.type not in GLOBAL_TYPES:
        if not _trigger_tools(rule):
            out.append(("rule_no_trigger", f"{r.type} rule names no tool, so nothing triggers it"))
    return out


def slot_problem(rule: Dict[str, Any]) -> Optional[str]:
    """The first empty or wrong-typed slot of a rule that a model wrote (``unfilled_slot`` or ``invalid_slots`` of
    ``validate_package``), or None. A rule with no id yet gets a stand-in."""
    stand_in = {**rule, "id": rule.get("id") or "new"}
    found = _slot_findings(Rule(id=stand_in["id"], type=str(rule.get("type") or ""), rule=stand_in))
    return next((msg for code, msg in found if code in ("unfilled_slot", "invalid_slots")), None)


def _label_findings(r: Rule) -> List[Tuple[str, str]]:
    """``class`` and ``level`` must be set and be one of the known values (also the ``level`` inside the rule dict,
    which ``package.runtime_rules`` reads when the rule has no level of its own)."""
    out = []
    for label, value, known, code in (
        ("class", r.cls, RULE_CLASSES, "unknown_class"),
        ("level", r.level, RULE_LEVELS, "unknown_level"),
    ):
        if value in ("", None):
            out.append(("unfilled_slot", f"{label} is not set"))
        elif value not in known:
            out.append((code, f"{label} {value!r} is not one of {', '.join(known)}"))
    inner = r.rule.get("level") if isinstance(r.rule, dict) and not r.level else None
    if inner not in ("", None) and inner not in RULE_LEVELS:
        out.append(("unknown_level", f"level {inner!r} in the rule is not one of {', '.join(RULE_LEVELS)}"))
    return out
