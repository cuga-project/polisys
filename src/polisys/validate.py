"""Checks that a built rule is well-formed and matches the tool schemas and the policy text.

``check_rule`` is used on every rule a model or a person writes (pipeline and skill). It returns the reason a rule
is rejected, or None. ``validate_build`` checks a whole build. ``validate_package`` checks a whole policy package
(design sections 9 and 10): rule slots and types, coverage, rules with no trigger tool, conflicts between rules.
It only reports; ``resolve_conflicts`` is the one function that changes a package, and only a new one. Nothing here
knows any benchmark: it only compares rules with tool schemas, quotes and each other.
"""

from __future__ import annotations

import copy
import dataclasses
import re
import typing
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .documents import clean as _clean
from .handlers.tables import _props
from .ruleset import FREE_TEXT, RULE_TYPES, _condition, _iso_moment, _number, canon, norm

_FOLD = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


def _fold(text: Any) -> str:
    """Text as it is compared with the policy: emphasis marks and spaces cleaned, lower case, every dash a hyphen and every
    curly quote straight, on both sides (a reader may fold them, and a quote may be typed with a plain one)."""
    return _clean(text if isinstance(text, str) else "").lower().translate(_FOLD)


DELAYED = re.compile(r"\bafter\s+\d+\s*(minutes?|hours?|days?|weeks?|business days?)\b", re.I)
OPS = {"contains", "equals", "in", "gt", "ge", "lt", "le", "empty"}
NO_QUOTE_TYPES = ("grounded_values", "id_provenance", "guidance", "verify")


PROMPT_PREFIX = re.compile(
    r"^\s*(S\d+\.\s*)?(\[[^\]]*\]\s*)?"
)  # our own numbering and type tags, copied by the model


def quote_ok(quote: str, statement: str, min_share: float = 0.9) -> bool:
    """The quote is the statement's text: verbatim, or with at most one word in ten changed (a model copying slips)."""
    q = PROMPT_PREFIX.sub("", _clean(quote)).lower().strip()
    st = _clean(statement).lower()
    if not q:
        return False
    if q in st:
        return True
    qw, sw = re.findall(r"\w+", q), re.findall(r"\w+", st)
    if len(qw) < 4:
        return False
    import difflib

    same = sum(b.size for b in difflib.SequenceMatcher(None, qw, sw, autojunk=False).get_matching_blocks())
    return same / len(qw) >= min_share


ID_DESCRIPTION = re.compile(r"\b(id|identifier)\b", re.I)


def holds_identifier(arg: str, prop: Any, statement: str) -> bool:
    """The argument has no enum and is a record identifier: its name ends in ``_id`` or its description says ``id`` or
    ``identifier``. A statement that writes the argument's name itself may mean the identifiers (an allow list of ids)."""
    description = prop.get("description") if isinstance(prop, dict) else ""
    named = isinstance(arg, str) and bool(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(arg)}(?![A-Za-z0-9_])", statement or "", re.I)
    )
    return not named and bool(
        re.search(r"_id$", str(arg), re.I) or ID_DESCRIPTION.search(str(description or ""))
    )


def check_rule(
    rule: Dict[str, Any], specs: Optional[Dict[str, Dict[str, Any]]], statement: str
) -> Optional[str]:
    """Return the reason a model-made rule is rejected, or None. A rule with the wrong shape is rejected, never raised.
    ``specs`` None: there is no tool schema to compare with, so tool names, arguments and enums are not checked."""
    try:
        return _check_rule(rule, specs, statement)
    except (TypeError, KeyError, AttributeError, ValueError) as e:
        return f"malformed rule ({type(e).__name__})"


def _check_rule(
    rule: Dict[str, Any], specs: Optional[Dict[str, Dict[str, Any]]], statement: str
) -> Optional[str]:
    t = rule.get("type")
    if t not in RULE_TYPES:
        return f"unknown type {t}"
    if t not in NO_QUOTE_TYPES and not quote_ok(rule.get("quote", ""), statement):
        return "quote not verbatim"
    tools = rule.get("tools") or ([rule["tool"]] if rule.get("tool") else [])
    if t == "duty":
        tools = [rule.get("trigger", {}).get("tool"), rule.get("then", {}).get("tool")]
    for name in tools:
        if specs is not None and name not in specs:
            return f"unknown tool {name}"

    def _check_args(tool: str, names: List[str]) -> Optional[str]:
        if specs is None:
            return None
        props = _props(specs[tool])
        for a in names:
            if a not in props:
                return f"unknown argument {tool}.{a}"
        return None

    def _check_when(tool: str, when: Optional[Dict[str, Any]]) -> Optional[str]:
        if not when:
            return None
        bad = _check_args(tool, list(when))
        if bad or specs is None:
            return bad
        for a, vals in when.items():
            enum = {norm(v) for v in _props(specs[tool])[a].get("enum") or []}
            if (
                enum and "*" not in [str(v) for v in vals] and not {norm(v) for v in vals} <= enum
            ):  # "*" = given
                return f"value not in schema enum for {tool}.{a}"
        return None

    if t == "required_args":
        return _check_args(rule["tool"], rule.get("args", [])) or _check_when(rule["tool"], rule.get("when"))
    if t == "allowed_values":
        bad = _check_args(rule["tool"], [rule.get("arg")]) or _check_when(rule["tool"], rule.get("when"))
        if bad or specs is None:
            return bad
        prop = _props(specs[rule["tool"]])[rule["arg"]]
        enum = {norm(v) for v in prop.get("enum") or []}
        if enum and not {norm(v) for v in rule.get("values", [])} <= enum:
            return "values not in schema enum"
        if not enum and holds_identifier(rule["arg"], prop, statement):
            # no real identifier is one of a few words: the words are a property of the record the argument names
            return f"{rule['arg']} holds a record identifier, so a list of values cannot fit it: use reference_status"
        return None
    if t == "reference_status":
        return _check_args(rule["tool"], [rule.get("arg")])
    if t == "membership":
        return _check_args(rule["tool"], [rule.get("arg"), rule.get("group_arg")])
    if t == "duty":
        trig, then = rule.get("trigger") or {}, rule.get("then") or {}
        text = " ".join(
            [rule.get("quote", "")]
            + [str(u.get("condition", "")) for u in rule.get("unless") or [] if isinstance(u, dict)]
        )
        if DELAYED.search(text):
            return "time-delayed change: done by the system, not a duty of the agent"
        if trig.get("tool") == then.get("tool") and set(then.get("when") or {}) <= set(
            trig.get("when") or {}
        ):
            return "same tool and fields as the trigger: a state transition, not a duty"
        return _check_when(trig["tool"], trig.get("when")) or _check_when(then["tool"], then.get("when"))
    if t == "role_permission" and rule.get("effect", "forbid") != "forbid":
        return "only forbid role rules are supported"
    if t == "record_condition":
        if rule.get("op") not in OPS or not rule.get("field"):
            return f"record_condition needs field and op in {sorted(OPS)}"
        return _check_when(tools[0], rule.get("when")) if tools else "record_condition needs tools"
    if t == "threshold":
        if rule.get("op") not in ("gt", "ge", "lt", "le") or rule.get("value") is None:
            return "threshold needs op (gt, ge, lt, le) and value"
        if rule.get("then") not in ("forbid", "requires_approval"):
            return "threshold needs then: forbid or requires_approval"
        if specs is not None and rule.get("approval_tool") and rule["approval_tool"] not in specs:
            return f"unknown tool {rule['approval_tool']}"
        return _check_args(rule["tool"], [rule.get("arg")])
    if t == "format":
        try:
            re.compile(rule.get("pattern") or "")
        except re.error:
            return "bad_pattern"
        if not rule.get("pattern"):
            return "bad_pattern"
        return _check_args(rule["tool"], [rule.get("arg")])
    return None


# -- whole builds ----------------------------------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
import json  # noqa: E402

from .model import (  # noqa: E402
    PACKAGE_SCHEMA,
    RULE_CLASSES,
    RULE_LEVELS,
    CoverageItem,
    Link,
    LinkType,
    PolicyBuild,
    PolicyPackage,
    PolicySet,
    Rule,
    Statement,
    json_key,
    resolve_conflict,
)

CODES = {
    "unknown type": "unknown_rule_type",
    "quote not verbatim": "quote_not_verbatim",
    "unknown tool": "unknown_tool",
    "unknown argument": "unknown_arg",
    "not in schema enum": "value_not_in_enum",
    "holds a record identifier": "identifier_values",
    "bad_pattern": "bad_pattern",
}


@dataclass
class ValidationError:
    statement_id: str
    code: str
    message: str


def _code(reason: str) -> str:
    return next((c for k, c in CODES.items() if k in reason), "invalid_slots")


def _ids(ps: PolicySet) -> set:
    out = set()
    for d in ps.documents:
        out.add(d.id)
        for sec in d.sections:
            out.add(f"{d.id}:{sec.id}")
            out.update(st.id for st in sec.statements)
    return out


def _rule_error(
    r: Rule, by_name: Optional[Dict[str, Dict[str, Any]]], texts: Dict[str, str]
) -> Optional[ValidationError]:
    """The first problem of one rule (``check_rule``), as an error that names the rule's first statement."""
    sid = r.statement_ids[0] if r.statement_ids else r.id
    statement = " ".join(texts.get(i, "") for i in r.statement_ids) if texts else r.rule.get("quote", "")
    by_code = r.rule.get("by") == "code"
    if by_code:
        statement = r.rule.get("quote", "")  # code rules quote a table header, not one statement
    why = check_rule(r.rule, by_name, statement or r.rule.get("quote", ""))
    if not why:
        return None
    code = _code(why)
    if by_code and code == "value_not_in_enum":
        code = "schema_conflict"  # the policy lists values the tool does not accept: report, not a bad rule
    return ValidationError(sid, code, why)


def _cedar_error(rules: List[Rule]) -> Optional[ValidationError]:
    """The Cedar text of the role rules does not parse."""
    roles = [r.rule for r in rules if r.type == "role_permission"]
    if not roles:
        return None
    try:
        from .ruleset import RuleSetHooks

        RuleSetHooks._cedar(roles)
    except Exception as exc:  # cedar text that does not parse
        return ValidationError(roles[0].get("id", ""), "cedar_invalid", str(exc)[:200])
    return None


def _dangling(links: List[Link], ids: set) -> List[ValidationError]:
    out = []
    for lk in links:
        for end in (lk.src, lk.dst):
            if end not in ids:
                kind = getattr(lk.type, "value", lk.type)  # a package may hold the plain string
                out.append(ValidationError(lk.src, "link_target_missing", f"{kind} {lk.src} -> {lk.dst}"))
                break
    return out


def validate_build(
    build: PolicyBuild, specs: List[Dict[str, Any]], policy_set: Optional[PolicySet] = None
) -> List[ValidationError]:
    """Every problem in a build: bad rules, invalid Cedar, links without a target, statements not accounted for."""
    by_name = {t["name"]: t for t in specs}
    texts = {st.id: st.text for d in (policy_set.documents if policy_set else []) for st in d.statements}
    errors: List[ValidationError] = []
    for r in build.rules:
        error = _rule_error(r, by_name, texts)
        if error:
            errors.append(error)
    cedar = _cedar_error(build.rules)
    if cedar:
        errors.append(cedar)
    if policy_set is not None:
        errors += _dangling(build.links, _ids(policy_set))
        covered = {i for r in build.rules for i in r.statement_ids} | {c.statement_id for c in build.coverage}
        covered |= {s.id for p in build.procedures for s in p.steps}
        covered |= {d.source.get("statement") for d in build.definitions if d.source.get("statement")}
        for sid in texts:
            if sid not in covered:
                errors.append(ValidationError(sid, "statement_uncovered", texts[sid][:120]))
    return errors


def _signature(rule: Dict[str, Any]) -> str:
    return json.dumps(
        {k: v for k, v in rule.items() if k not in ("id", "by", "source", "quote")},
        sort_keys=True,
        default=str,
    )


def diff_builds(a: PolicyBuild, b: PolicyBuild) -> List[str]:
    """Statement ids whose rules differ between two builds of the same policy (build twice and compare)."""

    def per_statement(build: PolicyBuild) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for r in build.rules:
            for sid in r.statement_ids:
                out.setdefault(sid, []).append(_signature(r.rule))
        return {k: sorted(v) for k, v in out.items()}

    pa, pb = per_statement(a), per_statement(b)
    return sorted(sid for sid in set(pa) | set(pb) if pa.get(sid) != pb.get(sid))


# -- policy packages ---------------------------------------------------------------------------------------------------

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
KNOWN_EXTRA = {"stats", "id"}  # package ``extra`` keys that ``package.py`` writes back to a source format
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
# Types as in ``ruleset.py``.
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


# -- the fields of the package -----------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _hints(cls: type) -> Dict[str, Any]:
    return typing.get_type_hints(cls)


def _conforms(value: Any, tp: Any) -> bool:
    """``value`` has the shape of the annotation ``tp``. Nested model objects only need to be of the right class:
    their own fields are checked when the walk reaches them."""
    if tp is Any:
        return True
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is typing.Union:
        return any(_conforms(value, a) for a in args)
    if tp is type(None):
        return value is None
    if origin in (list, List):
        return isinstance(value, list) and (not args or all(_conforms(v, args[0]) for v in value))
    if origin in (dict, Dict):
        return isinstance(value, dict) and (
            not args or all(_conforms(k, args[0]) and _conforms(v, args[1]) for k, v in value.items())
        )
    if isinstance(tp, type) and issubclass(tp, Enum):
        return isinstance(value, tp) or (isinstance(value, str) and value in {m.value for m in tp})
    if tp is bool:
        return isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tp is str:
        return isinstance(value, str)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return isinstance(value, tp)
    return True


def _model_errors(obj: Any, path: str, out: List[PackageError]) -> None:
    """Fields whose value does not have the annotated type (``from_json`` lets them through), and keys that went to
    ``extra`` because the model does not have them. Walks every model object under ``obj``."""
    hints = _hints(type(obj))
    own = getattr(obj, "id", "")
    label = own if isinstance(own, str) else ""
    for f in dataclasses.fields(obj):
        value, key = getattr(obj, f.name), json_key(f)
        if not _conforms(value, hints[f.name]):
            want = str(hints[f.name]).replace("typing.", "").replace("polisys.model.", "")
            out.append(
                PackageError(label, "invalid_field", f"{path}{key}: expected {want}, got {_show(value)}")
            )
            continue
        items = value if isinstance(value, list) else [value]
        for i, item in enumerate(items):
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                _model_errors(item, f"{path}{key}[{i}]." if isinstance(value, list) else f"{path}{key}.", out)
    known = KNOWN_EXTRA if isinstance(obj, PolicyPackage) else set()
    extra = getattr(obj, "extra", None)
    for key in extra if isinstance(extra, dict) else {}:  # a wrong-typed ``extra`` was reported above
        if isinstance(key, str) and not key.startswith("_") and key not in known:
            out.append(
                PackageError(
                    label, "unknown_field", f"{path}extra.{key}: the model has no field {key!r}", "warning"
                )
            )


# The parts of a package that the model types only as ``Dict[str, Any]``, and what the checks below read from them.
QUESTION_SLOTS: Dict[str, Tuple[Any, bool]] = {
    "statement_ids": ("strs", True),
    "rule_ids": ("strs", False),  # the rules a question is about, when it is about a pair of rules
    "question": ("text", True),
    "open_values": (
        "strs",
        False,
    ),  # "<obligation id>.<value name>" of a literal that the question leaves open
    "default_level": ("level", False),
}
DEFINITION_SLOTS: Dict[str, Tuple[Any, bool]] = {
    "term": ("text", False),
    "meaning": ("text", False),
    "source": ({"statement": ("str", False)}, False),
}
OBLIGATION_SLOTS: Dict[str, Tuple[Any, bool]] = {
    "scope": ("dict", False),
    "effect": ({"kind": ("str", False), "tools": ("strs", False)}, False),
    "values": ("dict", False),
    "source": ({"quote": ("str", False)}, False),
}


def _table_errors(
    label: str, where: str, slots: Dict[str, Tuple[Any, bool]], data: Any
) -> List[PackageError]:
    """``_check_slots`` for a part of the package: a wrong type is an ``invalid_field`` (the checks below read it)."""
    if not isinstance(data, dict):
        return [PackageError(label, "invalid_field", f"{where}: must be an object, got {_show(data)}")]
    return [
        PackageError(
            label, "invalid_field" if code == "invalid_slots" else code, f"{where}.{msg}", _severity(code)
        )
        for code, msg in _check_slots(slots, data, "")
    ]


def _shape_errors(pkg: PolicyPackage) -> List[PackageError]:
    out: List[PackageError] = []
    for i, q in enumerate(pkg.policy_questions):
        out += _table_errors("", f"policy_questions[{i}]", QUESTION_SLOTS, q)
    definitions = pkg.global_.get("definitions")
    if definitions is not None and not isinstance(definitions, list):
        out.append(
            PackageError("", "invalid_field", f"global.definitions: must be a list, got {_show(definitions)}")
        )
    for i, d in enumerate(definitions if isinstance(definitions, list) else []):
        out += _table_errors("", f"global.definitions[{i}]", DEFINITION_SLOTS, d)
    for o in pkg.obligations:
        fields = {name: getattr(o, name) for name in OBLIGATION_SLOTS}
        out += _table_errors(o.id, f"obligation {o.id}", OBLIGATION_SLOTS, fields)
    return out


# What ``package.to_store_graph`` keeps of ``global``, and the shape ``ProcedureStore`` reads from each key. A key outside
# this list is dropped without an error, and a value of another shape crashes the store or loses its content.
GLOBAL_KEYS = (
    "global_rules",
    "triage",
    "routing",
    "housekeeping",
    "priority",
    "escalation",
    "templates",
    "calendar",
    "definitions",
)
GRAPH_NODES = ("triage", "housekeeping", "escalation")  # {name, src, steps, must_not}


def _str_items(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _global_findings(g: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(code, message) for each ``global`` value that ``ProcedureStore`` would crash on or not read."""
    import datetime as dt

    out: List[Tuple[str, str]] = []

    def bad(where: str, want: str, value: Any) -> None:
        out.append(("invalid_field", f"global.{where}: must be {want}, got {_show(value)}"))

    for key in g:
        if key not in GLOBAL_KEYS:
            out.append(
                (
                    "unknown_field",
                    f"global.{key}: the runtime reads only {', '.join(GLOBAL_KEYS)}; this key is dropped",
                )
            )
    rules = g.get("global_rules")
    if rules is not None:
        if not isinstance(rules, list):
            bad("global_rules", "a list of {id, src, rule}", rules)
        for i, r in enumerate(rules if isinstance(rules, list) else []):
            if not isinstance(r, dict) or not all(
                isinstance(r.get(k), str) and r[k] for k in ("id", "src", "rule")
            ):
                bad(f"global_rules[{i}]", "an object with text id, src and rule", r)
            elif r.get("attach_when") is not None:
                try:
                    re.compile(r["attach_when"])
                except (re.error, TypeError):
                    bad(f"global_rules[{i}].attach_when", "a regular expression", r["attach_when"])
    for key in GRAPH_NODES:
        node = g.get(key)
        if node is None:
            continue
        if not isinstance(node, dict):
            bad(key, "an object {name, src, steps, must_not}", node)
            continue
        for field_name in ("name", "src"):
            if field_name in node and not isinstance(node[field_name], str):
                bad(f"{key}.{field_name}", "a string", node[field_name])
        for field_name in ("steps", "must_not"):
            if field_name in node and not _str_items(node[field_name]):
                bad(f"{key}.{field_name}", "a list of strings", node[field_name])
    priority = g.get("priority")
    if priority is not None:
        if not isinstance(priority, dict):
            bad("priority", "an object {name, src, levels}", priority)
        else:
            for field_name in ("name", "src"):
                if field_name in priority and not isinstance(priority[field_name], str):
                    bad(f"priority.{field_name}", "a string", priority[field_name])
            if "levels" in priority and not _str_items(priority["levels"]):
                bad("priority.levels", "a list of strings", priority["levels"])
    routing = g.get("routing")
    if routing is not None:
        if not isinstance(routing, dict) or not all(isinstance(v, dict) for v in routing.values()):
            bad("routing", "an object: category -> {field: value}", routing)
    templates = g.get("templates")
    if templates is not None:
        if not isinstance(templates, dict):
            bad("templates", "an object keyed by the number the text cites", templates)
        for key, t in (templates if isinstance(templates, dict) else {}).items():
            if not isinstance(t, dict) or not all(
                isinstance(t.get(k), str) and t[k] for k in ("name", "text")
            ):
                bad(f"templates.{key}", "an object with text name and text (and number, slots)", t)
            elif str(t.get("number")) != str(key):
                bad(f"templates.{key}.number", f"the key {key!r}", t.get("number"))
            elif "slots" in t and not _str_items(t["slots"]):
                bad(f"templates.{key}.slots", "a list of strings", t["slots"])
    calendar = g.get("calendar")
    if calendar is not None:
        if not isinstance(calendar, dict):
            bad("calendar", "an object {workdays, holidays}", calendar)
        else:
            days = calendar.get("workdays")
            if days is not None and not (
                isinstance(days, list)
                and all(isinstance(d, int) and not isinstance(d, bool) and 0 <= d <= 6 for d in days)
            ):
                bad("calendar.workdays", "a list of weekday numbers 0 (Monday) to 6", days)
            holidays = calendar.get("holidays")
            if holidays is not None:
                ok = _str_items(holidays)
                if ok:
                    try:
                        [dt.date.fromisoformat(h) for h in holidays]
                    except ValueError:
                        ok = False
                if not ok:
                    bad("calendar.holidays", "a list of ISO dates (YYYY-MM-DD)", holidays)
    definitions = g.get("definitions")
    for i, d in enumerate(definitions if isinstance(definitions, list) else []):
        if isinstance(d, dict) and not (isinstance(d.get("term"), str) and d["term"]):
            out.append(("invalid_field", f"global.definitions[{i}].term: must be a non-empty string"))
    return out


def _structure_errors(pkg: PolicyPackage, legacy: bool = False) -> List[PackageError]:
    """The schema name, fields of the wrong type, keys the model does not have, and the shape of what the model types
    only as a dict."""
    out: List[PackageError] = []
    if pkg.schema != PACKAGE_SCHEMA:
        out.append(
            PackageError("", "invalid_field", f"schema is {pkg.schema!r}, expected {PACKAGE_SCHEMA!r}")
        )
    _model_errors(pkg, "", out)
    if not any(
        e.code == "invalid_field" for e in out
    ):  # the dicts are only known to be dicts if the model walk passed
        out += _shape_errors(pkg)
        if not legacy:  # a converted package keeps the global part of its source as it was
            out += [
                PackageError("", code, msg, _severity(code)) for code, msg in _global_findings(pkg.global_)
            ]
    return out


def _content_errors(pkg: PolicyPackage) -> List[PackageError]:
    """Values of the right type that are still wrong: a date that is not a date, a rule id used twice."""
    out: List[PackageError] = []
    for d in pkg.documents:
        if isinstance(d.effective_date, str) and not ISO_DATE.match(d.effective_date):
            out.append(
                PackageError(
                    d.id,
                    "invalid_field",
                    f"document {d.id}: effective_date {d.effective_date!r} is not an ISO date",
                )
            )
    wrapper_ids: set = set()
    inner_ids: set = set()
    for r in pkg.rules:
        inner = r.rule.get("id") if isinstance(r.rule, dict) else None
        twice = r.id in wrapper_ids or (isinstance(inner, str) and inner in inner_ids)
        if not r.id:
            out.append(PackageError(r.id, "duplicate_rule_id", "a rule has no id"))
        elif twice:
            # ``RuleSetHooks.add_rules`` keeps every rule of one load, but the brief lists an id once, and a block_once or a
            # repeated denial is counted by id
            where = f"id {r.id!r}" if r.id in wrapper_ids else f"inner id {inner!r}"
            out.append(
                PackageError(
                    r.id,
                    "duplicate_rule_id",
                    f"rule {where} is used by another rule: both load, but the brief lists only the first"
                    " and block_once and repeated denials count them together",
                )
            )
        wrapper_ids.add(r.id)
        if isinstance(inner, str) and inner:
            inner_ids.add(inner)
    return out


# -- obligations -------------------------------------------------------------------------------------------------------


def _obligation_errors(
    pkg: PolicyPackage, by_name: Optional[Dict[str, Dict[str, Any]]]
) -> List[PackageError]:
    out: List[PackageError] = []
    for o in pkg.obligations:
        for field_name, known in OBLIGATION_ENUMS.items():
            value = getattr(o, field_name)
            if isinstance(value, str) and value not in known:
                out.append(
                    PackageError(
                        o.id,
                        "invalid_field",
                        f"obligation {o.id}: {field_name} {value!r} is not one of {', '.join(known)}",
                    )
                )
        effect = o.effect if isinstance(o.effect, dict) else {}
        if effect.get("kind") not in EFFECT_KINDS:
            out.append(
                PackageError(
                    o.id,
                    "unfilled_slot",
                    f"obligation {o.id}: effect.kind {effect.get('kind')!r} is not one of {', '.join(EFFECT_KINDS)}",
                )
            )
        tools = effect.get("tools") if isinstance(effect.get("tools"), list) else []
        if effect.get("kind") == "write" and not tools:
            out.append(
                PackageError(
                    o.id, "unfilled_slot", f"obligation {o.id}: a write names no tool in effect.tools"
                )
            )
        for tool in tools:
            if by_name is not None and tool not in by_name:
                out.append(PackageError(o.id, "unknown_tool", f"obligation {o.id}: unknown tool {tool}"))
        for name, spec in (o.values or {}).items():
            origin = spec.get("from") if isinstance(spec, dict) else None
            if origin not in VALUE_ORIGINS:
                out.append(
                    PackageError(
                        o.id,
                        "invalid_field",
                        f"obligation {o.id}: value {name} has origin {origin!r}, expected one of {', '.join(VALUE_ORIGINS)}",
                    )
                )
    return out


# The contract of an obligation's values (design section 5, "Value binding"). ``effect.match`` maps a tool argument to
# ``{"pattern": text}``; a pattern holds ``{name}`` or ``{name:format}`` for the values of the obligation (``{{`` and
# ``}}`` are braces). A value is bound when a match pattern names it, directly or through a bound ``template`` value.
# Only a bound value is compared with the call; an unbound one is shown and never compared. Only a bound ``literal``,
# ``template`` or ``computed`` value can be called wrong.
CODE_ARGS = (
    "code",
    "command",
)  # the argument that holds the code of a code tool (``executePython``, ``executeBash``)
PLACEHOLDER_NAME = re.compile(r"\{([A-Za-z_]\w*)(?::[^{}]*)?\}")
VALUE_KEY = {  # the key each origin must carry
    "literal": "value",
    "template": "pattern",
    "computed": "fn",
    "case_fact": "field",
    "request": "about",
}


def _placeholders(text: Any) -> List[str]:
    if not isinstance(text, str):
        return []
    return PLACEHOLDER_NAME.findall(text.replace("{{", "").replace("}}", ""))


def _binding_errors(pkg: PolicyPackage, by_name: Optional[Dict[str, Dict[str, Any]]]) -> List[PackageError]:
    """Value binding of each obligation: what ``effect.match`` and the values say about each other."""
    out: List[PackageError] = []
    asked: set = set()  # (obligation id, value name) that a question calls open
    known = {o.id: o.values if isinstance(o.values, dict) else {} for o in pkg.obligations}
    for q in pkg.policy_questions:
        for ref in (
            q.get("open_values") or []
            if isinstance(q, dict) and isinstance(q.get("open_values"), list)
            else []
        ):
            oid, _, name = ref.rpartition(".") if isinstance(ref, str) else ("", "", "")
            if name in known.get(oid, {}):
                asked.add((oid, name))
            else:
                out.append(
                    PackageError(
                        oid,
                        "unknown_open_value",
                        f"policy question: open_values {ref!r} is not `<obligation id>.<value name>` of an obligation",
                    )
                )
    for o in pkg.obligations:
        values = o.values if isinstance(o.values, dict) else {}
        effect = o.effect if isinstance(o.effect, dict) else {}
        match = effect.get("match") if isinstance(effect.get("match"), dict) else {}
        listed = effect.get("tools")
        tools = [t for t in listed if isinstance(t, str)] if isinstance(listed, list) else []

        def err(code: str, message: str, severity: str = "error") -> None:
            out.append(PackageError(o.id, code, f"obligation {o.id}: {message}", severity))

        for name, spec in values.items():
            origin = spec.get("from") if isinstance(spec, dict) else None
            if not isinstance(origin, str) or origin not in VALUE_KEY:
                continue
            key = VALUE_KEY[origin]
            if _empty(spec.get(key)) and not (origin == "literal" and key in spec):
                err("unfilled_slot", f"value {name} ({origin}) has no {key}")
        for arg, m in match.items():
            if not isinstance(m, dict) or not isinstance(m.get("pattern"), str) or not m["pattern"]:
                err("invalid_field", f"match.{arg} must be an object with a text pattern")
                continue
            if by_name is not None and tools and all(t in by_name for t in tools):
                lacking = [t for t in tools if arg not in _props(by_name[t])]
                if arg in CODE_ARGS and len(lacking) < len(tools):
                    pass  # a code argument is `code` for one code tool and `command` for another: see below
                elif lacking:
                    err("unknown_arg", f"match.{arg} is not an argument of {', '.join(lacking)}")
            for ph in _placeholders(m["pattern"]):
                if ph not in values:
                    err("unknown_value_name", f"match.{arg} names {{{ph}}}, which is not in values")
        if (
            by_name is not None
            and tools
            and all(t in by_name for t in tools)
            and any(a in match for a in CODE_ARGS)
        ):
            for t in tools:  # one match for each code tool: the argument that holds its code
                own = [a for a in CODE_ARGS if a in _props(by_name[t])]
                if own and own[0] not in match:
                    err(
                        "unknown_arg",
                        f"{t} takes its code in `{own[0]}`, and match has no pattern for it: bind the code argument "
                        f"of every code tool ({', '.join(CODE_ARGS)})",
                    )
        bound: set = set()
        todo = [ph for m in match.values() if isinstance(m, dict) for ph in _placeholders(m.get("pattern"))]
        while todo:
            name = todo.pop()
            if name in bound or name not in values:
                continue
            bound.add(name)
            spec = values[name]
            if isinstance(spec, dict) and spec.get("from") == "template":
                for ph in _placeholders(spec.get("pattern")):
                    if ph not in values:
                        err("unknown_value_name", f"value {name} names {{{ph}}}, which is not in values")
                    todo.append(ph)
        for name, spec in values.items():
            if name not in bound:
                err(
                    "unbound_value",
                    f"unmatched value {name}: it is in no effect.match pattern, so it is shown and never compared",
                    "warning",
                )
        for name, spec in values.items():
            if isinstance(spec, dict) and spec.get("from") == "literal" and (o.id, name) in asked:
                err(
                    "literal_open_question",
                    f"value {name} is a literal and a policy question lists it as open: leave the value out until the "
                    "owner answers",
                )
    return out


# -- text the package must come from --------------------------------------------------------------------------------------

MIN_PART_WORDS = 3  # a must-not with a condition is two parts of verbatim words, each of at least this many


def _flat(text: Any) -> str:
    """Words only, lower case, one space between them, a space at both ends: for finding a run of words in the text."""
    return " " + " ".join(re.findall(r"\w+", _fold(text))) + " "


class _Text:
    """The words of each document, statements joined in order (a sentence the converter split is whole again)."""

    def __init__(self, pkg: PolicyPackage):
        docs = [[st.text for st in d.statements] for d in pkg.documents]
        self.exact = [" ".join(_fold(t) for t in texts) for texts in docs if texts]
        self.words = [_flat(" ".join(texts)) for texts in docs if texts]

    def __bool__(self) -> bool:
        return bool(self.exact)

    def has(self, text: Any) -> bool:
        """The text is in a document, word for word (spaces and emphasis marks aside)."""
        t = _fold(text).strip(" .;:,\"'")
        return bool(t) and any(t in e for e in self.exact)

    def has_words(self, words: str) -> bool:
        return any(words in w for w in self.words)

    def in_parts(self, text: Any, most: int) -> bool:
        """The text is verbatim, or is at most ``most`` runs of words that are each verbatim and at least ``MIN_PART_WORDS``
        long. A run that is in the text keeps every shorter run that starts at the same word in the text, so ``longest[i]``
        tells which runs from word ``i`` are allowed, and a split with the fewest runs is found by trying every allowed
        length (the longest first run can leave a rest that is too short, so it is not the only choice)."""
        words = _flat(text).split()
        if not words:
            return False
        if self.has_words(f" {' '.join(words)} "):
            return True
        n = len(words)
        longest = []
        for i in range(n):
            j = i + 1
            while j <= n and self.has_words(f" {' '.join(words[i:j])} "):
                j += 1
            longest.append(j - 1 - i)
        fewest = [0] * (n + 1)  # fewest runs that cover words[i:]; n + 1 when none does
        for i in range(n - 1, -1, -1):
            best = n + 1
            for size in range(MIN_PART_WORDS, longest[i] + 1):
                if i + size <= n:
                    best = min(best, 1 + fewest[i + size])
            fewest[i] = best
        return fewest[0] <= most

    def must_not_ok(self, text: Any) -> bool:
        """A must-not is verbatim, or a condition and a prohibition that are each verbatim."""
        return self.in_parts(text, 2)


PRONOUN_START = re.compile(
    r"^\W*(they|them|their|it|its|he|she|his|her|we|this|these|those|that|such)\b", re.I
)


def bare_must_not(text: str) -> bool:
    """A must-not that cannot be read alone: under 5 words, or one that starts with a pronoun."""
    return len(re.findall(r"\w+", text)) < 5 or bool(PRONOUN_START.match(text))


def _must_nots(pkg: PolicyPackage) -> Iterator[Tuple[str, str]]:
    """(where, text) of every must-not the runtime shows or checks: those of procedures and of the global nodes."""
    for p in pkg.procedures:
        for m in p.must_not if isinstance(p.must_not, list) else []:
            yield f"procedure {p.id}", m
    for key in GRAPH_NODES:
        node = pkg.global_.get(key)
        listed = node.get("must_not") if isinstance(node, dict) else None
        for m in listed if isinstance(listed, list) else []:
            if isinstance(m, str):
                yield f"global.{key}", m


def _tool_typed(prop: Any) -> bool:
    """The argument takes a value of the tool itself: an enum or a boolean, also as one of the forms of ``anyOf``,
    ``oneOf`` or a list of types (an optional boolean is ``anyOf`` a boolean and null)."""
    if not isinstance(prop, dict):
        return False
    kind = prop.get("type")
    if prop.get("enum") or kind == "boolean" or (isinstance(kind, list) and "boolean" in kind):
        return True
    return any(_tool_typed(alt) for key in ("anyOf", "oneOf", "allOf") for alt in prop.get(key) or [])


def _unfounded_texts(
    pkg: PolicyPackage, text: _Text, by_name: Optional[Dict[str, Dict[str, Any]]]
) -> List[PackageError]:
    """Warnings for what the runtime shows or compares and the text does not contain: global rules, decisions, triage,
    housekeeping and escalation steps, definitions, notes, ``applies_when`` (verbatim, or up to three verbatim runs of words),
    template text (verbatim), a literal value and the fixed parts of a template value (verbatim). A literal on an argument
    with an enum, or a boolean, is a value of the tool and is not looked up."""
    out: List[PackageError] = []

    def warn(where: str, value: Any, strict: bool = False, most: int = 3) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        if not (text.has(value) if strict else text.in_parts(value, most)):
            out.append(
                PackageError(
                    where,
                    "not_in_text",
                    f"{where}: not found in the policy text{'' if strict else ' (as it is, or in up to 3 runs of its words)'}: "
                    f"{_show(value)}",
                    "warning",
                )
            )

    g = pkg.global_
    for i, r in enumerate(g.get("global_rules") if isinstance(g.get("global_rules"), list) else []):
        warn(f"global.global_rules[{i}]", r.get("rule") if isinstance(r, dict) else None)
    for key in GRAPH_NODES:
        node = g.get(key)
        for i, step in enumerate(
            node.get("steps") if isinstance(node, dict) and isinstance(node.get("steps"), list) else []
        ):
            warn(f"global.{key}.steps[{i}]", step)
    templates = g.get("templates")
    for key, t in (templates if isinstance(templates, dict) else {}).items():
        warn(f"global.templates.{key}", t.get("text") if isinstance(t, dict) else None, strict=True)
    definitions = g.get("definitions")
    for i, d in enumerate(definitions if isinstance(definitions, list) else []):
        if isinstance(d, dict):
            warn(f"global.definitions[{i}].term", d.get("term"), strict=True)
            warn(f"global.definitions[{i}].meaning", d.get("meaning"))
    for p in pkg.procedures:
        warn(f"procedure {p.id}: applies_when", p.applies_when)
        for i, note in enumerate(p.notes):
            warn(f"procedure {p.id}: notes[{i}]", note)
        for i, d in enumerate(p.decisions):
            # only the condition: `then` and `else` are mostly flow text ("continue with step 2")
            warn(f"procedure {p.id}: decisions[{i}].rule", d.get("rule") if isinstance(d, dict) else None)
    for o in pkg.obligations:
        values = o.values if isinstance(o.values, dict) else {}
        effect = o.effect if isinstance(o.effect, dict) else {}
        match = effect.get("match") if isinstance(effect.get("match"), dict) else {}
        listed = effect.get("tools")
        tools = (
            [t for t in listed if isinstance(t, str) and by_name and t in by_name]
            if isinstance(listed, list)
            else []
        )
        for name, spec in values.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("from") == "literal" and isinstance(spec.get("value"), str):
                arg_names = [
                    arg
                    for arg, m in match.items()
                    if isinstance(m, dict) and name in _placeholders(m.get("pattern"))
                ]
                typed = False
                for arg in arg_names:
                    for t in tools:
                        prop = _props(by_name[t]).get(arg)
                        typed = typed or _tool_typed(prop)
                if not typed:
                    warn(f"obligation {o.id}: value {name}", spec["value"], strict=True)
            elif spec.get("from") == "template" and isinstance(spec.get("pattern"), str):
                fixed = PLACEHOLDER_NAME.split(spec["pattern"].replace("{{", "").replace("}}", ""))[::2]
                for piece in fixed:
                    if len(re.findall(r"[A-Za-z0-9]", piece)) >= 4:
                        warn(f"obligation {o.id}: value {name}", piece.strip(), strict=True)
    return out


def _invented_errors(pkg: PolicyPackage, by_name: Optional[Dict[str, Dict[str, Any]]]) -> List[PackageError]:
    """Content that the policy text does not contain: quotes, step text and must-nots that are not its words, a step
    action that is no tool, a role the text never names. Skipped when the package carries no text."""
    text = _Text(pkg)
    if not text:
        return []
    out: List[PackageError] = []

    def not_verbatim(label: str, what: str, value: Any) -> None:
        out.append(
            PackageError(
                "", "quote_not_verbatim", f"{label}: {what} is not the words of the policy: {_show(value)}"
            )
        )

    for o in pkg.obligations:
        quote = o.source.get("quote") if isinstance(o.source, dict) else None
        if quote and not text.has(quote):
            not_verbatim(f"obligation {o.id}", "source.quote", quote)
    for p in pkg.procedures:
        for kind in ("preconditions", "exceptions", "templates", "escalations"):
            for i, item in enumerate(getattr(p, kind)):
                quote = item.get("quote") if isinstance(item, dict) else None
                if not quote:
                    out.append(
                        PackageError(p.id, "unfilled_slot", f"procedure {p.id}: {kind}[{i}] has no quote")
                    )
                elif not text.has(quote):
                    not_verbatim(f"procedure {p.id}", f"{kind}[{i}].quote", quote)
        for i, step in enumerate(p.steps):
            words = step.text or step.quote
            if words and not text.has(words):
                not_verbatim(f"procedure {p.id}", f"steps[{i}]", words)
            if step.action and step.action != "reply" and by_name is not None and step.action not in by_name:
                out.append(
                    PackageError(
                        p.id,
                        "unknown_tool",
                        f"procedure {p.id}: steps[{i}].action {step.action!r} is not a tool",
                    )
                )
    for where, must_not in _must_nots(pkg):
        if not text.must_not_ok(must_not):
            not_verbatim(where, "must_not", must_not)
    for where, must_not in _must_nots(pkg):
        if isinstance(must_not, str) and bare_must_not(must_not):
            out.append(
                PackageError(
                    where,
                    "bare_must_not",
                    f"{where}: the must-not {_show(must_not)} cannot be read alone (under 5 words, or it starts with a "
                    "pronoun): write the condition first",
                    "warning",
                )
            )
    out += _unfounded_texts(pkg, text, by_name)
    roles = [
        (r.id, role)
        for r in pkg.rules
        if r.type == "role_permission" and isinstance(r.rule, dict) and isinstance(r.rule.get("roles"), list)
        for role in r.rule["roles"]
    ]
    for rule_id, role in roles:
        if isinstance(role, str) and not text.has_words(_flat(role)):
            out.append(
                PackageError(
                    rule_id, "unknown_role", f"rule {rule_id}: the role {role!r} is not named in the policy"
                )
            )
    if (pkg.generator or {}).get("kind") == "skill":
        for r in pkg.rules:
            if r.type == "verify" and not r.why:
                out.append(
                    PackageError(
                        r.id,
                        "verify_without_why",
                        f"rule {r.id}: a verify rule needs `why`: what keeps a typed rule from fitting",
                        "warning",
                    )
                )
    return out


# -- coverage ----------------------------------------------------------------------------------------------------------


def _outcomes(
    pkg: PolicyPackage, statements: List[Any], legacy: bool
) -> Tuple[Dict[str, set], Dict[str, set], set, List[PackageError]]:
    """``(shown, said, out_of_scope, errors)``: for each statement id the outcomes that an object of the package gives, the
    kinds of the coverage items that did not count, the ids that are ``out_of_scope`` with a reason, and the errors of the
    coverage items themselves (``out_of_scope`` with no reason)."""
    out: List[PackageError] = []
    shown: Dict[str, set] = defaultdict(set)  # statement id -> outcomes that an object of the package gives
    for r in pkg.rules:
        kind = {"hard": "hard_rule", "judgment": "judgment_rule", "guidance": "guidance"}.get(r.cls)
        for sid in r.statement_ids if kind else []:
            shown[sid].add(kind)
    for p in pkg.procedures:
        for step in p.steps:
            if step.id:
                shown[step.id].add("procedure_step")
    for d in pkg.global_.get("definitions") or []:
        if (d.get("source") or {}).get("statement"):
            shown[d["source"]["statement"]].add("definition")
    for q in pkg.policy_questions:
        for sid in q.get("statement_ids") or []:
            shown[sid].add("question")
    texts = {st.id: _fold(st.text) for st in statements}

    def quoted_by(quote: Any, outcome: str) -> None:
        """``outcome`` for each statement that contains ``quote`` (at least ``MIN_QUOTE_WORDS`` words)."""
        quote = _fold(quote).strip()
        if len(re.findall(r"\w+", quote)) >= MIN_QUOTE_WORDS:
            for sid, text in texts.items():
                if quote in text:
                    shown[sid].add(outcome)

    for o in pkg.obligations:
        quoted_by((o.source or {}).get("quote"), "obligation")
    for (
        p
    ) in pkg.procedures:  # a condition, exception, template or escalation of a procedure quotes its statement
        for part in (p.preconditions, p.exceptions, p.templates, p.escalations):
            for item in part:
                quoted_by(item.get("quote"), "procedure_step")
    said: Dict[str, set] = defaultdict(set)  # statement id -> kinds of the coverage items that did not count
    out_of_scope = set()
    for c in pkg.coverage:
        kind = c.kind if c.kind in OUTCOMES else (LEGACY_OUTCOME.get(c.kind) if legacy else None)
        if kind == "out_of_scope" and not c.reason:
            out.append(PackageError(c.statement_id, "coverage_incomplete", "out_of_scope needs a reason"))
        elif kind == "out_of_scope":
            out_of_scope.add(c.statement_id)
        elif legacy and kind:
            shown[c.statement_id].add(kind)
        else:
            said[c.statement_id].add(c.kind)
    return shown, said, out_of_scope, out


def statement_outcomes(pkg: PolicyPackage, legacy: bool = False) -> Dict[str, List[str]]:
    """For each statement id of the package, the outcomes it has (``OUTCOMES``), sorted; ``[]`` for a statement with none.
    The same reading as ``coverage_incomplete``: an object of the package that names the statement, or an
    ``out_of_scope`` coverage item with a reason."""
    statements = [st for d in pkg.documents for st in d.statements]
    shown, _, out_of_scope, _ = _outcomes(pkg, statements, legacy)
    return {
        st.id: sorted(shown.get(st.id) or []) + (["out_of_scope"] if st.id in out_of_scope else [])
        for st in statements
    }


def _coverage_errors(pkg: PolicyPackage, legacy: bool) -> List[PackageError]:
    """Every must or must-not statement has an outcome (design section 9). An outcome is an object in the package that
    names the statement: a rule (its class gives ``hard_rule``, ``judgment_rule`` or ``guidance``), a procedure step with
    its id, a definition, a policy question, or an obligation, or a precondition, exception, template or escalation of a
    procedure, whose quote (at least ``MIN_QUOTE_WORDS`` words) is taken from it. A coverage item counts for ``out_of_scope`` only, and needs a reason: another kind is a claim, and a claim
    with nothing behind it does not cover a statement. A statement may have more than one outcome (one statement can
    give two rules, and a tie adds a question beside the rules); ``out_of_scope`` beside another one is a
    ``coverage_conflict`` warning.

    ``legacy``: the coverage items of older builds count as written, by ``LEGACY_OUTCOME``. A converted flat rule can
    have lost its statement ids, so a ``rule`` item is taken as true there."""
    statements = [st for d in pkg.documents for st in d.statements]
    if not statements:
        if (pkg.rules or pkg.procedures or pkg.obligations) and not legacy:
            return [
                PackageError(
                    "",
                    "coverage_incomplete",
                    "the package has rules but no statements, so coverage cannot be shown",
                )
            ]
        return []
    shown, said, out_of_scope, out = _outcomes(pkg, statements, legacy)
    for st in statements:
        covered = shown.get(st.id) or st.id in out_of_scope
        if st.modality in ("must", "must_not") and not covered:
            modality = getattr(st.modality, "value", st.modality)
            claim = (
                f" (coverage says {', '.join(sorted(said[st.id]))}, but nothing in the package shows it)"
                if said.get(st.id)
                else ""
            )
            out.append(
                PackageError(
                    st.id,
                    "coverage_incomplete",
                    f"{modality} statement has no outcome: {st.text[:100]}{claim}",
                )
            )
        if not legacy and st.id in out_of_scope and shown.get(st.id):
            out.append(
                PackageError(
                    st.id,
                    "coverage_conflict",
                    f"statement is out_of_scope and also has {', '.join(sorted(shown[st.id]))}",
                    "warning",
                )
            )
    return out


# -- conflicts ---------------------------------------------------------------------------------------------------------


@dataclass
class Conflict:
    """Two rules that cannot both hold. ``winner``: the one that trust, a ``supersedes`` link, the newer effective date
    or the higher authority rank prefers (``model.resolve_conflict``), or None when nothing decides.
    ``blocking``: the conflict can stop a call that the policy allows (a rule that cannot deny never blocks).
    ``kind``: ``values`` (rules with values that exclude each other), ``duty`` (a duty asks for what a rule forbids) or
    ``role`` (a duty needs a tool that a role rule forbids for some roles: it depends on the role, so it is reported and
    never settled by ``resolve_conflicts``)."""

    a: Rule
    b: Rule
    detail: str
    blocking: bool
    winner: Optional[Rule] = None
    kind: str = "values"

    @property
    def loser(self) -> Optional[Rule]:
        return None if self.winner is None else (self.b if self.winner is self.a else self.a)


def _can_deny(r: Rule) -> bool:
    if r.type in ("verify", "guidance", "duty") or r.may_deny is False or r.rule.get("may_deny") is False:
        return False
    return (r.level or "deny") in ("deny", "block_once")


def _values(items: List[Any]) -> set:
    """Values as ``RuleSetHooks._check_allowed_values`` compares them."""
    return {norm(str(v).lstrip("#@")) for v in items}


def _overlap(a: Optional[Dict[str, List[Any]]], b: Optional[Dict[str, List[Any]]]) -> bool:
    """Some call can meet both ``when`` conditions: no argument they both name has disjoint values. ``*`` (a value is
    given) fits every value. An argument only one of them names is free in the other."""
    a, b = a or {}, b or {}
    for key in set(a) & set(b):
        va, vb = {norm(v) for v in a[key]}, {norm(v) for v in b[key]}
        if "*" not in va and "*" not in vb and not va & vb:
            return False
    return True


def _format_ok(arg: str, pattern: str, value: Any) -> bool:
    """Whether a value passes a ``format`` rule, as ``RuleSetHooks._check_format`` decides it. A pattern that does not
    compile passes: it is reported as ``bad_pattern``, not as a conflict."""
    try:
        text, pattern = canon(value), canon(pattern)
        free = bool(FREE_TEXT.search(arg)) or len(text.split()) > 3
        return (
            re.search(pattern, text) if free else re.fullmatch(pattern, text, flags=re.DOTALL)
        ) is not None
    except re.error:
        return True


def _forbids(r: Rule, tool: str, arg: str, wanted: List[Any], when: Dict[str, List[Any]]) -> Optional[str]:
    """Why rule ``r`` does not allow any of the values a duty asks ``tool.arg`` to have, or None."""
    d, want = r.rule, sorted(_values(wanted))
    if r.type == "allowed_values":
        if _overlap(d.get("when"), when) and not set(want) & _values(d["values"]):
            return f"the duty requires {tool}.{arg} {want}, which {r.id} does not allow"
    elif r.type == "threshold" and d.get("then") == "forbid":
        if all(_condition(v, d["op"], d["value"]) for v in wanted):
            return (
                f"the duty requires {tool}.{arg} {want}, which {r.id} forbids ({arg} {d['op']} {d['value']})"
            )
    elif r.type == "format":
        if not any(_format_ok(arg, d["pattern"], v) for v in wanted):
            return f"the duty requires {tool}.{arg} {want}, which does not match the format of {r.id} ({d['pattern']})"
    return None


def _pairs(rules: List[Rule]) -> Iterator[Tuple[Rule, Rule, str, bool, str]]:
    """(rule, rule, what differs, can it block a call, kind) for:

    - rules that restrict the same tool and argument to values with nothing in common, for calls that can meet both
      ``when`` conditions (``allowed_values``, ``reference_status``);
    - a duty that asks for a value that another rule forbids: not in its ``allowed_values``, over a ``threshold`` that
      forbids, or outside a ``format`` (a duty never blocks, the rule that forbids its value can);
    - a duty that needs a tool that a ``role_permission`` rule forbids for some roles, when those roles can still do what
      triggers the duty (a warning: it depends on the role).
    """
    groups: Dict[tuple, List[Tuple[Rule, set]]] = defaultdict(list)
    limits: Dict[Tuple[str, str], List[Rule]] = defaultdict(
        list
    )  # rules that forbid some values of (tool, argument)
    roles: List[Rule] = []
    for r in rules:
        d = r.rule
        if r.type == "role_permission" and d.get("effect") in (None, "", "forbid"):
            roles.append(r)
        for tool in _trigger_tools(d):
            if r.type == "allowed_values":
                groups[(r.type, tool, d["arg"])].append((r, _values(d["values"])))
            elif r.type == "reference_status":
                groups[(r.type, tool, d["arg"], d["field"])].append((r, _values(d["allowed"])))
            if r.type in ("allowed_values", "threshold", "format"):
                limits[(tool, d["arg"])].append(r)
    seen = set()
    for key, members in groups.items():
        for i, (a, va) in enumerate(members):
            for b, vb in members[i + 1 :]:
                if (id(a), id(b)) in seen or va & vb or not _overlap(a.rule.get("when"), b.rule.get("when")):
                    continue
                seen.add((id(a), id(b)))
                detail = (
                    f"both restrict {key[1]}.{key[2]} to values with nothing in common: "
                    f"{sorted(va)} and {sorted(vb)}"
                )
                yield a, b, detail, _can_deny(a) and _can_deny(b), "values"
    for duty in (r for r in rules if r.type == "duty"):
        then = duty.rule["then"]
        for arg, wanted in (then.get("when") or {}).items():
            if "*" in _values(wanted):
                continue
            for r in limits.get((then["tool"], arg), []):
                why = _forbids(r, then["tool"], arg, wanted, then["when"])
                if why:
                    yield duty, r, why, _can_deny(r), "duty"
        trigger = duty.rule.get("trigger") or {}
        for r in roles:
            forbidden = r.rule.get("tools", [])
            if (
                then["tool"] in forbidden and trigger.get("tool") not in forbidden
            ):  # else that role never owes the duty
                why = (
                    f"the duty requires {then['tool']}, which {r.id} forbids for the roles {r.rule['roles']}"
                )
                yield duty, r, why, False, "role"


def _scope(ps: PolicySet, st: Statement) -> set:
    """Ids a ``supersedes`` or ``exception_of`` link can name for a statement: itself, its section, its document."""
    doc = ps.document_of(st.id)
    if doc is None:
        return {st.id}
    return {st.id, f"{doc.id}:{st.source.get('section')}", doc.id}


def _conflicts(pkg: PolicyPackage, rules: List[Rule], priority_classes: Sequence[str] = ()) -> List[Conflict]:
    """The conflicts among ``rules`` (rules with well-formed slots), by code. A pair joined by an ``exception_of`` link
    is not a conflict: the exception is folded into the rule (design section 10)."""
    ps = PolicySet(documents=pkg.documents, links=pkg.links)
    index = {st.id: st for d in pkg.documents for st in d.statements}

    def statement(r: Rule) -> Optional[Statement]:
        return next((index[i] for i in r.statement_ids if i in index), None)

    out = []
    for a, b, detail, blocking, kind in _pairs(rules):
        sa, sb = statement(a), statement(b)
        winner = None
        if sa is not None and sb is not None:
            ids_a, ids_b = _scope(ps, sa), _scope(ps, sb)
            if any(
                lk.type == LinkType.EXCEPTION_OF
                and (lk.src in ids_a and lk.dst in ids_b or lk.src in ids_b and lk.dst in ids_a)
                for lk in pkg.links
            ):
                continue
            won = resolve_conflict(sa, sb, ps, priority_classes) if kind != "role" else None
            winner = None if won is None else (a if won is sa else b)
        out.append(Conflict(a, b, detail, blocking, winner, kind))
    return out


def _weaker(level: str, cap: str) -> str:
    """``level`` lowered to at most ``cap``; a rule already below it stays."""
    if level in RULE_LEVELS and RULE_LEVELS.index(level) <= RULE_LEVELS.index(cap):
        return level
    return cap


def _removed_by_survivors(decided: List[Conflict]) -> Tuple[Dict[int, Conflict], List[Conflict]]:
    """The conflicts that remove a rule, by the id of the removed rule, and the decided conflicts left over. A rule is
    removed only by a winner that stays: a winner that another conflict defeats waits until that one is settled.
    What is left over cannot be ordered (the winners beat each other in a cycle)."""
    removed: Dict[int, Conflict] = {}
    pending = list(decided)
    while pending:
        defeated = {id(c.loser) for c in pending}
        ready = [c for c in pending if id(c.winner) not in defeated]
        if not ready:
            break
        for c in ready:
            removed.setdefault(id(c.loser), c)
        pending = [c for c in pending if id(c.loser) not in removed and id(c.winner) not in removed]
    return removed, pending


def resolve_conflicts(
    pkg: PolicyPackage, priority_classes: Sequence[str] = ()
) -> Tuple[PolicyPackage, List[str]]:
    """A copy of a new package with its conflicts settled (design section 10), and a note for each change.

    - Trust, a ``supersedes`` link, the newer ``effective_date`` or the higher ``authority_rank`` decides (with
      ``priority_classes``, a document's rank is that of its class: ``model.authority_of``): the losing
      rule is removed, and a statement left with no rule is recorded as ``out_of_scope`` with that reason. A rule is
      removed only by a winner that stays: with A over B and C over A, A goes and B stays (B and C do not conflict).
    - Nothing decides (a tie): never decided here. Both rules drop to ``remind`` at most, and one policy question names
      the pair (``rule_ids``) and offers "each holds in a different case". A second run adds no second question.
    - A ``role`` conflict depends on the role: it is left as it is.

    A package that was converted from a legacy artifact (it has a ``_layout``), or has a field of the wrong type, is
    returned unchanged. The input is never changed."""
    from .package import LAYOUT  # package.py reads handlers that import this module

    out = copy.deepcopy(pkg)
    if any(e.code == "invalid_field" for e in _structure_errors(out)):
        return out, ["package has fields of the wrong type, not changed: run validate_package"]
    sound = [r for r in out.rules if not _is_broken(_slot_findings(r))]
    conflicts = [c for c in _conflicts(out, sound, priority_classes) if c.kind != "role"]
    if LAYOUT in out.extra:
        return out, [
            f"converted package, not changed: {len(conflicts)} conflicts are reported by validate_package"
        ]
    removed, cycle = _removed_by_survivors([c for c in conflicts if c.winner is not None])
    out.rules = [r for r in out.rules if id(r) not in removed]
    notes: List[str] = []
    left = {sid for r in out.rules for sid in r.statement_ids}
    for c in removed.values():
        notes.append(f"removed {c.loser.id}: {c.winner.id} wins over it ({c.detail})")
        for sid in c.loser.statement_ids:
            if sid not in left:
                left.add(sid)
                out.coverage.append(CoverageItem(sid, "out_of_scope", f"superseded by {c.winner.id}"))
    ties = [c for c in conflicts if c.winner is None] + cycle
    for c in ties:
        if id(c.a) in removed or id(c.b) in removed:
            continue
        for r in (c.a, c.b):
            r.level = _weaker(r.level, "remind")
        pair = {c.a.id, c.b.id}
        asked = any(set(q.get("rule_ids") or []) == pair for q in out.policy_questions)
        if not asked:
            out.policy_questions.append(
                {
                    "statement_ids": list(dict.fromkeys(c.a.statement_ids + c.b.statement_ids)),
                    "rule_ids": [c.a.id, c.b.id],
                    "question": (
                        f"Rules {c.a.id} and {c.b.id} conflict ({c.detail}), and nothing decides which one applies. "
                        "Does one replace the other, or does each hold in a different case (which one)?"
                    ),
                    "default_level": "remind",
                }
            )
        notes.append(
            f"tie between {c.a.id} and {c.b.id}: both set to at most remind, "
            + ("policy question already open" if asked else "policy question added")
        )
    return out, notes


# -- the whole package -------------------------------------------------------------------------------------------------


def _spec_map(
    tool_specs: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], List[PackageError]]:
    """Tool specs by name, with the schema under ``inputSchema`` (also read from ``input_schema`` and ``parameters``),
    and a finding for each spec that has no name. None: there are no specs, and tool checks are skipped."""
    if tool_specs is None:
        return None, []
    if not isinstance(tool_specs, list):
        return {}, [PackageError("", "invalid_field", f"tool specs: must be a list, got {_show(tool_specs)}")]
    specs: Dict[str, Dict[str, Any]] = {}
    bad: List[PackageError] = []
    for i, t in enumerate(tool_specs):
        name = t.get("name") if isinstance(t, dict) else None
        if not isinstance(name, str) or not name:
            bad.append(PackageError("", "invalid_field", f"tool spec {i} has no name: {_show(t)}"))
            continue
        schema = t.get("inputSchema") or t.get("input_schema") or t.get("parameters") or {}
        specs[name] = {**t, "inputSchema": schema if isinstance(schema, dict) else {}}
    return specs, bad


def _enum_detail(rule: Dict[str, Any], by_name: Optional[Dict[str, Dict[str, Any]]]) -> str:
    """For a rule whose values the tool does not accept: the tool, argument and values the tool does not list."""
    try:
        tool, arg = rule["tool"], rule["arg"]
        enum = {norm(v) for v in _props(by_name[tool])[arg].get("enum") or []}
        return f" ({tool}.{arg}: {[v for v in rule['values'] if norm(v) not in enum]})"
    except (KeyError, TypeError, AttributeError):
        return ""


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [] if value is None else [value]


def _pattern_conflicts(rule: Dict[str, Any], by_name: Optional[Dict[str, Dict[str, Any]]]) -> List[str]:
    """Literals of a rule that the tool's own argument ``pattern`` rejects: the values of an ``allowed_values`` rule and the
    values of a ``when`` (JSON Schema reads a pattern as a search). Like a value outside an enum, the policy then names what
    the tool does not accept: reported, not a bad rule."""
    if not by_name:
        return []
    tools = [rule.get("tool")] if isinstance(rule.get("tool"), str) else rule.get("tools") or []
    found = []
    literals: List[Tuple[str, List[Any]]] = []
    if rule.get("type") == "allowed_values" and isinstance(rule.get("arg"), str):
        literals.append((rule["arg"], _as_list(rule.get("values"))))
    if isinstance(rule.get("when"), dict):
        literals += [(arg, _as_list(vals)) for arg, vals in rule["when"].items() if isinstance(arg, str)]
    for tool in tools:
        props = _props(by_name[tool]) if tool in by_name else {}
        for arg, values in literals:
            pattern = (props.get(arg) or {}).get("pattern") if isinstance(props.get(arg), dict) else None
            if not isinstance(pattern, str):
                continue
            try:
                bad = [v for v in values if isinstance(v, str) and re.search(pattern, v) is None]
            except re.error:
                continue
            if bad:
                found.append(f"{tool}.{arg} has the pattern {pattern}, which {bad[:5]} do not match")
    return found


def _rule_findings(
    r: Rule, by_name: Optional[Dict[str, Dict[str, Any]]], texts: Dict[str, str]
) -> Tuple[List[PackageError], bool]:
    """The findings of one rule, and whether its slots are sound (the rule can take part in the conflict check)."""
    sid = r.statement_ids[0] if r.statement_ids else r.id
    slots = _slot_findings(r)
    out = [
        PackageError(sid, code, f"rule {r.id}: {msg}", _severity(code))
        for code, msg in slots + _label_findings(r)
    ]
    if _is_broken(slots):  # check_rule would only report the same problem as "malformed rule"
        return out, False
    error = _rule_error(r, by_name, texts)
    if error:
        code = (
            "schema_conflict"
            if error.code == "value_not_in_enum" and r.rule.get("schema_conflict")
            else error.code
        )
        detail = _enum_detail(r.rule, by_name) if code in ("value_not_in_enum", "schema_conflict") else ""
        message = f"rule {r.id}: {error.message}{detail}"
        out.append(PackageError(error.statement_id, code, message, _severity(code)))
    out += [
        PackageError(sid, "schema_conflict", f"rule {r.id}: {msg}", "warning")
        for msg in _pattern_conflicts(r.rule, by_name)
    ]
    return out, True


def exception_findings(pkg: PolicyPackage) -> List[PackageError]:
    """A warning for an ``exception_of`` link with no rule at one end: the brief of a rule list cannot show the exception under
    its rule (``package.exception_links``). Only a package with rules is checked. A statement that is a step of a procedure
    counts as held: the exception is then in the procedure, which another shape of the run shows."""
    if not pkg.rules:
        return []
    from .package import _rules_by_statement

    held = set(_rules_by_statement(pkg)) | {
        getattr(step, "id", "") for proc in pkg.procedures for step in proc.steps
    }
    out = []
    for lk in pkg.links:
        if str(getattr(lk.type, "value", lk.type)) != "exception_of":
            continue
        missing = [
            f"{end} ({role})" for end, role in ((lk.src, "exception"), (lk.dst, "rule")) if end not in held
        ]
        if missing:
            out.append(
                PackageError(
                    lk.src,
                    "exception_without_rule",
                    f"exception_of {lk.src} -> {lk.dst}: no rule holds {' and '.join(missing)}, so a rule list cannot "
                    "show the exception with its rule",
                    "warning",
                )
            )
    return out


def priority_findings(pkg: PolicyPackage, priority_classes: Sequence[str]) -> List[PackageError]:
    """A warning for a document whose ``priority_class`` the config does not list. A listed class that no document has is
    fine: a config may name classes for many policies. With no list there is nothing to compare, so no finding."""
    if not priority_classes:
        return []
    listed = ", ".join(priority_classes)
    return [
        PackageError(
            d.id,
            "unknown_priority_class",
            f"document {d.id} has the priority class {d.priority_class!r}, and the config lists only {listed}: "
            "its rank is its own authority_rank",
            "warning",
        )
        for d in pkg.documents
        if d.priority_class and d.priority_class not in priority_classes
    ]


def _conflict_findings(
    pkg: PolicyPackage, rules: List[Rule], priority_classes: Sequence[str] = ()
) -> List[PackageError]:
    asked = [set(q.get("rule_ids") or []) for q in pkg.policy_questions]
    out = []
    for c in _conflicts(pkg, rules, priority_classes):
        if c.kind == "role":
            how = "it applies only to those roles; resolve_conflicts leaves it"
        elif c.winner is not None:
            how = (
                f"{c.winner.id} wins (trust, supersedes link, effective_date or authority_rank); "
                f"resolve_conflicts removes {c.loser.id}"
            )
        else:
            how = "nothing decides (same trust, no supersedes link, same effective_date and authority_rank); "
            how += (
                "a policy question about this pair is open"
                if {c.a.id, c.b.id} in asked
                else "resolve_conflicts sets both to remind and adds a policy question"
            )
        out.append(
            PackageError(
                (c.a.statement_ids or [c.a.id])[0],
                "rule_conflict",
                f"{c.a.id} and {c.b.id}: {c.detail}; {how}",
                "error" if c.blocking else "warning",
            )
        )
    return out


def validate_package(
    pkg: PolicyPackage,
    tool_specs: Optional[List[Dict[str, Any]]],
    *,
    legacy: bool = False,
    priority_classes: Sequence[str] = (),
) -> List[ValidationError]:
    """Every problem in a package, with a severity. It only reports and never changes the package.

    Fields of the wrong type (``invalid_field``; the other checks wait until there are none), slots of the wrong type
    (``invalid_slots``), empty slots and placeholders (``unfilled_slot``), unknown ``level`` and ``class``, rules with
    the problems ``check_rule`` finds (tool, argument, enum, quote), rules with no trigger tool (``rule_no_trigger``),
    must and must-not statements with no outcome (``coverage_incomplete``), and conflicts between rules
    (``rule_conflict``). A conflict is an error when it can block a call and a warning otherwise. It does not raise on
    a malformed package: it reports.

    ``tool_specs`` None: there is no tool schema, and the checks that need one are skipped.
    ``priority_classes``: the classes of the config, highest first. They set the rank that decides a conflict when dates tie,
    and a document with a class that is not listed is a warning (``unknown_priority_class``).
    ``legacy``: the package was converted from an older artifact. Its coverage kinds count as outcomes (see
    ``LEGACY_OUTCOME``) and the gaps it has, ``coverage_incomplete`` and ``rule_no_trigger``, are warnings."""
    by_name, findings = _spec_map(tool_specs)
    structure = _structure_errors(pkg, legacy)
    findings += structure
    if any(f.code == "invalid_field" for f in structure):
        return findings  # the checks below read these fields: a wrong-typed one has to be fixed first
    findings += _content_errors(pkg)
    texts = {st.id: st.text for d in pkg.documents for st in d.statements}
    sound: List[Rule] = []
    for r in pkg.rules:
        found, ok = _rule_findings(r, by_name, texts)
        findings += found
        if ok:
            sound.append(r)
    cedar = _cedar_error(sound)
    if cedar:
        findings.append(PackageError(cedar.statement_id, cedar.code, cedar.message))
    if pkg.documents:
        dangling = _dangling(pkg.links, _ids(PolicySet(documents=pkg.documents)))
        findings += [PackageError(e.statement_id, e.code, e.message) for e in dangling]
    findings += _obligation_errors(pkg, by_name)
    if not legacy:
        findings += _binding_errors(pkg, by_name)
        findings += _invented_errors(pkg, by_name)
    findings += _coverage_errors(pkg, legacy)
    findings += _conflict_findings(pkg, sound, priority_classes)
    findings += priority_findings(pkg, priority_classes)
    findings += exception_findings(pkg)
    if legacy:
        for f in findings:
            if f.code in COMPLETENESS:
                f.severity = "warning"
    return findings
