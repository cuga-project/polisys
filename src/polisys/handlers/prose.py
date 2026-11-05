"""Rules from prose statements: a model fills the slots of the fixed rule types, code checks every rule.

Input: typed statements (``classify.py``) that code did not already turn into rules. One model call per section,
with the statements numbered and their types shown. A bare value in a ``when`` is first made a list
(``rule_repair.normalize_when``). Every rule the model writes is then checked by ``validate.check_rule`` (tool and
argument names, schema values, verbatim quote) and by ``rule_repair.partial_update_overreach``. What happens to a
statement:

- at least one valid rule → those rules;
- "rule" but no valid rule, no answer, or a model error → one ``verify`` rule: the verifier checks writes against
  the statement text at run time, so nothing is dropped;
- guidance → a ``guidance`` rule (shown in the brief);
- definition or out of scope → a coverage item only, except for a statement that says "must" or "must not": a
  definition, or an out-of-scope call with no reason, would leave it with no outcome, so it becomes a ``verify`` rule
  (``rule_repair`` then looks for tools and typed rules, and what no tool fits becomes guidance with a question).

Nothing here is specific to a benchmark or a document: the prompt only lists rule types and the tool catalog.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..llm import parse_json
from ..model import CoverageItem, Modality, Rule, Statement, StatementType
from ..rule_repair import normalize_when, partial_update_overreach
from ..validate import check_rule, slot_problem
from .tables import tool_catalog

CHECKER = {"role_permission": "cedar", "duty": "monitor", "guidance": "brief", "verify": "verifier"}

RULE_GUIDE = """Rule types and their slots (use only tool and argument names from the catalog):
- required_args: {"tool", "args": [..], "when": {"arg": [values]} (optional)} - the call must include these arguments
- allowed_values: {"tool", "arg", "values": [..], "when": {..} (optional)} - the argument must be one of the values
- reference_status: {"tool", "arg", "field", "allowed": [..]} - the record whose ID is passed in arg must have field in allowed
- membership: {"tool", "arg", "group_arg"} - the record named by arg must be a member of the record named by group_arg
- role_permission: {"tools": [..], "roles": [..], "effect": "forbid"} - users with these roles may not call these tools
- duty: {"trigger": {"tool", "when": {..}}, "then": {"tool", "when": {..}}, "unless": [{"condition": "<text>"}] (optional)} - after trigger, the agent must also call then. Use it for every statement that says an action must follow an event or a state ("when X, Y must be done", "before closing", "automatically", "must be created/linked/notified"), also when part of the condition cannot be written as argument values: put that part in "unless" as text
- record_condition: {"tools": [..], "field", "op": "contains|equals|in|gt|ge|lt|le|empty", "value", "effect": "forbid", "when": {..} (optional)} - do not call these tools for a record (named in the call) whose field meets the condition
- threshold: {"tool", "arg", "op": "gt|ge|lt|le", "value": <number>, "then": "forbid|requires_approval", "approval_tool": "<tool>" (optional)} - a number limit on an argument
- format: {"tool", "arg", "pattern": "<Python regex for the whole value>"} - the argument must have this exact form
- grounded_values: {} - global: optional arguments may only use values from the request or from tool results
- id_provenance: {} - global: IDs must come from tool results or the request
- guidance: {"text", "tools": [..]} - advice worth showing the agent that cannot be checked by code
Write a rule only when the statement clearly says it; do not invent tools, arguments, values or conditions.
"quote" must be copied word for word from the statement. If no rule type fits but the statement is a rule,
return class "rule" with an empty rules list: it will be checked at run time from its text. The same holds for a
statement that applies only in a case that the call's arguments and the records read cannot tell (it depends on how or
why the call is made): a rule without that condition would apply to every call."""

SYSTEM = (
    "You turn company policy statements into typed rules for an automatic checker. Use only tool and argument "
    "names from the tool catalog. Copy quotes word for word. Output JSON only."
)
USER = """{guide}

Tool catalog (name(args); * = required; [a|b] = allowed values):
{catalog}

Policy section {section}, statements (with their types):
{statements}
{feedback}
Return {{"statements": [{{"s": <number>, "class": "rule|guidance|definition|out_of_scope", "reason": "<if not a rule>",
"rules": [{{"type": "...", "quote": "...", <slots>}}]}}]}} with one entry per statement."""

SKIP_TYPES = {StatementType.DEFINITION, StatementType.PROCEDURE}  # handled by links.py and procedures.py


def _rid(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _why_rejected(raw: Dict[str, Any], by_name: Dict[str, Dict[str, Any]], text: str) -> Optional[str]:
    """Why a rule the model wrote is rejected, or None. The reason goes back to the model in a repair round."""
    normalize_when(raw)
    why = check_rule(raw, by_name, text)
    if why is None and slot_problem(raw):
        return f"a slot is empty or has the wrong type ({slot_problem(raw)})"
    if why is None and partial_update_overreach(raw, text):
        return (
            f"a statement about what a record must contain binds the call that creates it, not {raw.get('tool')}: "
            "write the rule on the create tool, or leave the statement to a verify rule"
        )
    return why


def _binding(st: Statement) -> bool:
    """The statement says that something must or must not be done."""
    return st.modality in (Modality.MUST, Modality.MUST_NOT)


def _tool_names(raw: Dict[str, Any]) -> List[str]:
    """The tool names a rejected rule names, whether it wrote them as one name or as a list (``tool`` or ``tools``)."""
    named = raw.get("tools") or raw.get("tool") or []
    return [t for t in (named if isinstance(named, list) else [named]) if isinstance(t, str)]


def _verify_rule(st: Statement, tools: Optional[List[str]] = None) -> Rule:
    rule = {
        "id": _rid("verify", st.id, st.text),
        "type": "verify",
        "statement": st.text,
        "tools": tools or [],
        "types": [t.value for t in st.types],
        "source": f"§{st.source.get('section', '')}",
        "quote": st.text,
    }
    return Rule(
        id=rule["id"],
        type="verify",
        rule=rule,
        statement_ids=[st.id],
        checker="verifier",
        source=dict(st.source),
        quote=st.text,
    )


def _rule(raw: Dict[str, Any], st: Statement) -> Rule:
    raw = dict(raw)
    raw.setdefault("source", f"§{st.source.get('section', '')}")
    raw["by"] = "model"
    raw["id"] = _rid(raw.get("type"), {k: v for k, v in raw.items() if k not in ("id", "by")})
    return Rule(
        id=raw["id"],
        type=raw["type"],
        rule=raw,
        statement_ids=[st.id],
        checker=CHECKER.get(raw["type"], "code"),
        source=dict(st.source),
        quote=raw.get("quote", st.text),
    )


def _section_rules(
    sid: str,
    sts: List[Statement],
    specs: List[Dict[str, Any]],
    by_name: Dict[str, Dict[str, Any]],
    model: Callable[[str, str], str],
    catalog: str,
    feedback: Optional[Dict[str, str]] = None,
) -> Tuple[List[Rule], List[CoverageItem]]:
    lines = "\n".join(
        f"S{i}. [{', '.join(t.value for t in st.types) or 'untyped'}] {st.text}"
        for i, st in enumerate(sts, 1)
    )
    fb = [f"S{i}: {feedback[st.id]}" for i, st in enumerate(sts, 1) if feedback and st.id in feedback]
    fb_text = (
        (
            "\nYour previous rules for these statements were rejected by the checker; fix them:\n"
            + "\n".join(fb)
            + "\n"
        )
        if fb
        else ""
    )
    try:
        reply = parse_json(
            model(
                SYSTEM,
                USER.format(
                    guide=RULE_GUIDE, catalog=catalog, section=sid, statements=lines, feedback=fb_text
                ),
            )
        )
    except Exception as exc:  # a model error must not lose statements
        return (
            [_verify_rule(st) for st in sts],
            [
                CoverageItem(
                    st.id, "prose_fallback", f"model error: {type(exc).__name__}", st.source, st.text
                )
                for st in sts
            ],
        )
    answers = {}
    for a in reply.get("statements") or []:
        if isinstance(a, dict):
            try:
                answers[int(str(a.get("s", "")).lstrip("S"))] = a
            except ValueError:
                continue
    rules: List[Rule] = []
    cov: List[CoverageItem] = []
    for i, st in enumerate(sts, 1):
        a = answers.get(i)
        if a is None:
            rules.append(_verify_rule(st))
            cov.append(CoverageItem(st.id, "prose_fallback", "no answer from the model", st.source, st.text))
            continue
        cls = str(a.get("class", "rule")).lower()
        reason = str(a.get("reason", ""))
        if cls in ("definition", "out_of_scope") and _binding(st) and (cls == "definition" or not reason):
            # a statement that says "must" and has no check, no definition and no reason would have no outcome
            # (design section 9): it is a ``verify`` rule, so that the repair step tries tools and typed rules first;
            # what no tool fits ends as guidance and a policy question
            rules.append(_verify_rule(st))
            why = f"model class {cls}: {reason}".strip(": ")
            cov.append(CoverageItem(st.id, "verify", why, st.source, st.text))
            continue
        if cls in ("definition", "out_of_scope"):
            cov.append(CoverageItem(st.id, cls, reason, st.source, st.text))
            continue
        good, bad = [], []
        for raw in a.get("rules") or []:
            if not isinstance(raw, dict):
                continue
            why = _why_rejected(raw, by_name, st.text)
            (bad if why else good).append((raw, why))
        if good:
            rules.extend(_rule(raw, st) for raw, _ in good)
            cov.append(CoverageItem(st.id, "rule", f"{len(good)} rule(s)", st.source, st.text))
        elif cls == "guidance":
            g = {"type": "guidance", "text": st.text, "tools": [], "quote": st.text}
            rules.append(_rule(g, st))
            cov.append(CoverageItem(st.id, "guidance", "", st.source, st.text))
        else:
            tools = sorted({t for raw, _ in bad for t in _tool_names(raw) if t in by_name})
            rules.append(_verify_rule(st, tools))
            cov.append(
                CoverageItem(
                    st.id,
                    "verify",
                    "; ".join(why for _, why in bad) or "no rule type fits",
                    st.source,
                    st.text,
                )
            )
    return rules, cov


def prose_rules(
    statements: List[Statement],
    specs: List[Dict[str, Any]],
    model: Optional[Callable[[str, str], str]],
    catalog: Optional[str] = None,
    feedback: Optional[Dict[str, str]] = None,
    skip_types=SKIP_TYPES,
) -> Tuple[List[Rule], List[CoverageItem]]:
    """Rules for prose statements, one model call per section. Without a model every statement becomes ``verify``."""
    by_name = {t["name"]: t for t in specs}
    catalog = catalog if catalog is not None else tool_catalog(specs)
    groups: "OrderedDict[str, List[Statement]]" = OrderedDict()
    for st in statements:
        types = set(st.types or [])
        if st.source.get("step") is None:
            types.discard(
                StatementType.PROCEDURE
            )  # one sentence, not a numbered list: procedures.py skips it
        if types and types <= set(skip_types):
            continue
        groups.setdefault(str(st.source.get("section", "")), []).append(st)
    rules: List[Rule] = []
    cov: List[CoverageItem] = []
    if model is None:
        for sts in groups.values():
            rules.extend(_verify_rule(st) for st in sts)
            cov.extend(CoverageItem(st.id, "verify", "no model", st.source, st.text) for st in sts)
        return rules, cov
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:  # sections in parallel; results kept in document order
        results = list(
            pool.map(
                lambda kv: _section_rules(kv[0], kv[1], specs, by_name, model, catalog, feedback),
                groups.items(),
            )
        )
    for r, c in results:
        rules.extend(r)
        cov.extend(c)
    return rules, cov
