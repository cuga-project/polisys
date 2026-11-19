"""One rule against the policy text and the tool schemas: ``check_rule`` gives the reason a rule is rejected, or None.
``quote_ok`` and ``holds_identifier`` compare text with the policy as every check here does (case, spaces and dashes folded)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from polisys.compile.documents import clean as _clean
from polisys.compile.handlers.tables import _props
from polisys.checks.rules import RULE_TYPES
from polisys.values import norm

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
