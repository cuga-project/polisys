"""Repairs of compiled rule sets, found by the rule audit (the rule audit of the evaluation repo, 2026-11-04). The
builder runs them as a step of ``build_policy_set`` (``repair_rules``), after the rules from prose are known, so a
rebuild gives them with no extra step.

1. ``normalize_when``: a bare value in a ``when`` (``"General"``, ``true``) becomes a list, and a placeholder
   (``"<record_id>"``: the value of the trigger call, which a slot cannot hold) becomes ``"*"`` (the argument is given).
   The runtime reads a string as its characters, so a bare string never matched.
2. ``attach_verify_tools``: a ``verify`` rule compiled with no tools never runs. Give it the write tools its statement
   names: an argument name written in the statement (``assigned_to``), or the record noun of a tool name (``problem``
   in ``create_problem``).
3. ``grant_rules``: a role table lists what each role MAY do ("Full CRUD on orders ..."). The rule type
   ``role_permission`` only forbids, so grant rows were dropped. One model call per row picks the write tools the row
   grants; the rule forbids the other write tools for that role (checked by Cedar). Rows that grant everything give
   no rule (out of scope: nothing to forbid). A row that has an outcome needs no ``verify`` rule of its own. The rows are
   those of a table whose header starts with a role, and table rows typed ``role_permission`` that have no such header.
   ``may_deny`` sets the level: the default is False (remind only), because a grant table is often broader or
   narrower than what the platform's own tasks do, and a deny on it blocked passed runs.
4. ``retype_verify``: one model call per ``verify`` rule that is left proposes rules of the code-checkable types. A
   proposal that ``validate.check_rule`` accepts, that is not a ``partial_update_overreach`` and that names only
   arguments its statement names (``ungrounded_names``), replaces the verify rule (code decides it, with no verifier
   call). A statement about a change after a delay ("after 72 hours") is not retyped: the system does it later, and no
   call of the agent breaks it (``check_rule`` holds the same for duties).
5. A ``verify`` rule that has no tool after steps 2 to 4 cannot be triggered by any call. It becomes ``guidance``: the
   statement stays in the brief, and no verifier runs on every write for it. If it says "must" or "must not", it also
   gets a policy question (outcome ``question``): the owner says whether it is guidance, a behaviour of the system or a
   rule for a tool the build did not find. A ``verify`` rule about a change after a delay becomes ``guidance`` at step 4.

A call that fails (it raises, or its answer has no JSON object with the key asked for) is counted, as ``grant_errors`` and
``retype_errors``: "nothing to type" and "every call failed" give the same rule set otherwise.

Nothing here is specific to a policy: the prompts only list rule types and the tool catalog.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from .llm import parse_json
from .model import CoverageItem, Rule, Statement
from .ruleset import is_write_tool

VERBS = {
    "add",
    "create",
    "update",
    "delete",
    "remove",
    "register",
    "link",
    "map",
    "send",
    "new",
    "set",
    "submit",
}
WORKERS = 4  # model calls of one repair step at a time (as ``handlers/prose.py``)


def _props(spec: Dict[str, Any]) -> Dict[str, Any]:
    return (spec.get("input_schema") or spec.get("inputSchema") or spec.get("parameters") or {}).get(
        "properties"
    ) or {}


def _nouns(tool: str) -> List[str]:
    return [w for w in tool.lower().split("_") if w and w not in VERBS]


def _st(statement: Any, key: str) -> Any:
    """A field of a ``Statement`` or of a statement dict."""
    return statement.get(key) if isinstance(statement, dict) else getattr(statement, key, None)


# -- 1. bare values in ``when`` -----------------------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"^\s*<[^<>]*>\s*$")


def _list_form(value: Any) -> Any:
    """One ``when`` value as the runtime reads it: a list. Only a string, a number or a boolean is changed."""
    if isinstance(value, str) and _PLACEHOLDER.match(value):
        return ["*"]
    if isinstance(value, (str, int, float, bool)):
        return [value]
    return value


def normalize_when(rule: Dict[str, Any]) -> int:
    """In place: every ``when`` of a rule (its own, and the ``trigger`` and ``then`` of a duty) maps names to lists.
    Returns how many values were changed."""
    n = 0
    for holder in (rule, rule.get("trigger"), rule.get("then")):
        when = holder.get("when") if isinstance(holder, dict) else None
        if not isinstance(when, dict):
            continue
        for key, value in list(when.items()):
            fixed = _list_form(value)
            if fixed is not value:
                when[key], n = fixed, n + 1
    return n


# -- 2. tools for a verify rule -----------------------------------------------------------------------------------------


def tools_named_by(text: str, specs: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    """Write tools a statement is about: argument names written in it (snake_case, exact), else tools whose record
    noun(s) all appear in it as words (singular or plural)."""
    low = text.lower()
    words = set(re.findall(r"[a-z]+", low))
    words |= {w[:-1] for w in words if w.endswith("s")}
    writes = [s for s in specs if is_write_tool(s["name"])]
    by_arg = [s["name"] for s in writes if any("_" in k and k.lower() in low for k in _props(s))]
    if by_arg:
        return by_arg[:limit]
    by_noun = [s["name"] for s in writes if _nouns(s["name"]) and all(n in words for n in _nouns(s["name"]))]
    return by_noun[:limit]


def attach_verify_tools(rules: List[Dict[str, Any]], specs: List[Dict[str, Any]]) -> int:
    """In place. Returns how many verify rules got tools."""
    n = 0
    for r in rules:
        if r.get("type") == "verify" and not r.get("tools"):
            tools = tools_named_by(r.get("statement") or "", specs)
            if tools:
                r["tools"], r["tools_by"] = tools, "statement text"
                n += 1
    return n


# -- 3. grants ----------------------------------------------------------------------------------------------------------

GRANT_SYSTEM = "You read one row of a role permission table. Output JSON only."
GRANT_USER = """Role table header: {header}
Row: {row}

Write tools of the system (name: description):
{tools}

Which of these write tools does this row ALLOW the role to call? Read capability words literally ("Full CRUD on
invoices" allows every write tool of invoices: create, update, delete, and the tools that link or remove an invoice
from another record; "READ access" allows no write tool). If the row grants every table, set "all" to true.
Return {{"role": "<role id as written, e.g. clerk>", "all": true|false, "allowed": ["<tool>", ...]}}"""


def _answer(model: Callable[[str, str], str], system: str, user: str, key: str) -> Optional[Dict[str, Any]]:
    """The JSON object a model answers with, or None when the call failed: it raised, or there is no object with ``key``
    in the answer. An answer of ``{"rules": []}`` is an answer."""
    try:
        v = parse_json(model(system, user))
    except Exception:  # noqa: BLE001 - a failed call gives no rule; the caller counts it
        return None
    return v if isinstance(v, dict) and key in v else None


def _one_grant(
    row: Dict[str, Any],
    catalog: str,
    names: set,
    model: Callable[[str, str], str],
    may_deny: bool,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """(rule or None, why there is none): ``"all"`` the row grants every write tool, ``"failed"`` the call failed or named
    no role, or ``""``."""
    v = _answer(
        model,
        GRANT_SYSTEM,
        GRANT_USER.format(header=" | ".join(row.get("header") or []), row=row["text"], tools=catalog),
        "role",
    )
    if v is None:
        return None, "failed"
    role = str(v.get("role") or "").strip().lower()
    m = re.search(r"\(([a-z_]+)\)", row["text"])  # "Name (id) | ..." -> the id in brackets wins
    role = m.group(1) if m else role
    if not role:
        return None, "failed"
    if v.get("all"):
        return None, "all"
    allowed = {t for t in v.get("allowed") or [] if t in names}
    forbidden = sorted(names - allowed)
    if not forbidden:
        return None, "all"
    return {
        "id": f"grant_{role}",
        "type": "role_permission",
        "tools": forbidden,
        "roles": [role],
        "effect": "forbid",
        "from_grant": sorted(allowed),
        "source": row.get("source") or "",
        "quote": row["text"][:300],
        "statement_ids": [row["id"]],
        "by": "model",
        "may_deny": may_deny,
    }, ""


def grant_rules(
    rows: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    model: Callable[[str, str], str],
    may_deny: bool = False,
) -> List[Dict[str, Any]]:
    """``rows``: role-table statements ``{"id", "text", "header", "source"}``. Returns role_permission rules."""
    return [rule for _, rule, _ in _grants(rows, specs, model, may_deny) if rule]


def _grants(
    rows: List[Dict[str, Any]], specs: List[Dict[str, Any]], model: Callable[[str, str], str], may_deny: bool
) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]]:
    """(row, rule, why no rule) for each row, in the order of ``rows``; the calls run in parallel."""
    writes = [s for s in specs if is_write_tool(s["name"])]
    catalog = "\n".join(f"{s['name']}: {(s.get('description') or '')[:120]}" for s in writes)
    names = {s["name"] for s in writes}
    if not rows:
        return []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda row: _one_grant(row, catalog, names, model, may_deny), rows))
    return [(row, rule, why) for row, (rule, why) in zip(rows, results)]


def role_rows(statements: List[Any]) -> List[Dict[str, Any]]:
    """Statements (objects or dicts) that are rows of a role table: the first column of its header is a role, or the
    statement is typed ``role_permission`` and is a table row (a parser that takes a first row for the header, or finds
    no header, leaves the rows without one)."""
    out = []
    for st in statements:
        source = _st(st, "source") or {}
        header = [str(h) for h in source.get("table_header") or []]
        types = {getattr(t, "value", t) for t in _st(st, "types") or []}
        by_header = bool(header) and "role" in header[0].lower()
        if by_header or ("role_permission" in types and str(_st(st, "text") or "").count("|") >= 2):
            out.append(
                {
                    "id": _st(st, "id"),
                    "text": _st(st, "text"),
                    "header": header if by_header else [],
                    "source": f"§{source.get('section', '')} L{source.get('line', '')}",
                }
            )
    return out


# -- 4. typed rules for verify rules -------------------------------------------------------------------------------------

TYPE_SYSTEM = (
    "You turn one policy statement into code-checkable rules when that is possible. Output JSON only."
)
TYPE_USER = """Policy statement (section {source}): "{text}"

Write tools (name(arguments)):
{catalog}

Can ONE call of one of these tools break this statement in a way code can check from the call's arguments and the
records already read? If yes, express it with these rule types (tool and argument names exactly from the list):
- required_args: {{"type": "required_args", "tool": "...", "args": ["..."], "when": {{"<arg>": ["<value>" or "*"]}}}}
  the call must give these arguments ("when" optional; "*" = that argument is given)
- allowed_values: {{"type": "allowed_values", "tool": "...", "arg": "...", "values": ["..."], "when": {{...}}}}
  the argument must be one of the values (e.g. when a coupon is given, tier must be promo)
- reference_status: {{"type": "reference_status", "tool": "...", "arg": "...", "field": "...", "allowed": ["..."]}}
  the record that the argument names must have this field in the allowed values (e.g. the customer named by
  customer_id must have active = true)
- membership: {{"type": "membership", "tool": "...", "arg": "...", "group_arg": "..."}}
  the record named by arg must be a member of the record named by group_arg
- record_condition: {{"type": "record_condition", "tools": ["..."], "field": "...", "op": "eq|ne|contains", "value": "...",
  "effect": "forbid"}}  deny when a record that the call names meets the condition
Use only what the statement says: every argument or field you use must be one that the statement names. If the
statement holds only in a case that the call's arguments and the records read cannot tell (it depends on how or why the
call is made), return {{"rules": []}}: a rule without that condition would apply to every call. If no type fits
exactly (a heading, a definition, a general duty that names no argument, a step spread over several calls, system
behaviour), return {{"rules": []}}. One statement may give several rules.
Return {{"rules": [...]}}"""


def _write_catalog(specs: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{s['name']}({', '.join(_props(s))})" for s in specs if is_write_tool(s["name"]))


UPDATE_TOOL = re.compile(r"^(update|patch|edit|modify|set)_")
UPDATE_WORDS = re.compile(r"\b(updat|modif|edit)", re.I)


def partial_update_overreach(rule: Dict[str, Any], statement: str) -> bool:
    """A statement about what a record must contain ("every record must have an owner") binds the call that creates the
    record, not every partial update of it: a required_args rule on an update tool needs a statement about updates. A
    rule with a ``when`` binds only the calls that meet it ("cannot become resolved without a note": ``when`` status
    resolved), so it is no overreach."""
    return (
        rule.get("type") == "required_args"
        and bool(UPDATE_TOOL.match(str(rule.get("tool") or "")))
        and not rule.get("when")
        and not UPDATE_WORDS.search(statement or "")
    )


def _named(name: str, text: str) -> bool:
    """Whether a statement names an argument or a field: the name as written, or each word of it as a word (``owner_id``
    for "a recorded owner"; the word ``id`` alone does not count)."""
    low = text.lower()
    if name.lower() in low:
        return True
    words = set(re.findall(r"[a-z]+", low))
    words |= {w[:-1] for w in words if w.endswith("s")}
    parts = [w for w in re.split(r"[_\W]+", name.lower()) if w and w != "id"]
    return bool(parts) and all(w in words or w[:-1] in words for w in parts)


def ungrounded_names(rule: Dict[str, Any], statement: str) -> List[str]:
    """The argument and field names of a typed rule that its statement does not name. A general statement ("all
    mandatory fields must be valid") gives the model no argument, and it fills in every argument of every tool."""
    names = list(rule.get("args") or []) + [rule.get(k) for k in ("arg", "group_arg", "field")]
    for holder in (rule, rule.get("trigger"), rule.get("then")):
        names += list((holder.get("when") if isinstance(holder, dict) else None) or {})
    return sorted({n for n in names if isinstance(n, str) and n and n != "*" and not _named(n, statement)})


def retype_rule(
    rule: Dict[str, Any],
    by_name: Dict[str, Dict[str, Any]],
    catalog: str,
    model: Callable[[str, str], str],
) -> Tuple[List[Dict[str, Any]], str]:
    """(the code-checkable rules that replace one ``verify`` rule, status). Ids are ``<verify id>_t<n>``. Status: ``"ok"``
    (empty list: none fits), ``"delayed"`` (a change after a delay: the system does it later, and no call of the agent
    breaks it; no call is made) or ``"failed"`` (the call raised or its answer has no ``rules``)."""
    from .validate import DELAYED, check_rule, slot_problem

    text = rule.get("statement") or ""
    if DELAYED.search(text):
        return [], "delayed"
    v = _answer(
        model,
        TYPE_SYSTEM,
        TYPE_USER.format(source=rule.get("source", ""), text=text, catalog=catalog),
        "rules",
    )
    if v is None:
        return [], "failed"
    good = []
    for i, nr in enumerate(v.get("rules") or []):
        if not isinstance(nr, dict):
            continue
        nr = {
            **nr,
            "id": f"{rule.get('id')}_t{i}",
            "source": rule.get("source", ""),
            "quote": text,
            "by": "model",
            "retyped_from": rule.get("id"),
        }
        normalize_when(nr)
        if nr.get("type") in ("verify", "guidance") or check_rule(nr, by_name, text) is not None:
            continue
        if slot_problem(nr):
            continue  # an empty list of values, for one
        if partial_update_overreach(nr, text) or ungrounded_names(nr, text):
            continue
        if rule.get("unless"):
            nr["unless"] = rule["unless"]  # an exception of the statement still applies to what came from it
        nr["may_deny"] = rule.get("may_deny", True)
        good.append(nr)
    return good, "ok"


def _wrap(rule: Dict[str, Any], statement_ids: List[str], source: Dict[str, Any], checker: str) -> Rule:
    return Rule(
        id=rule["id"],
        type=rule["type"],
        rule=rule,
        statement_ids=list(statement_ids),
        checker=checker,
        may_deny=rule.get("may_deny", True),
        source=dict(source),
        quote=rule.get("quote", ""),
    )


def _as_guidance(r: Rule) -> Rule:
    """A ``verify`` rule shown to the agent instead of checked: the same statement, as a ``guidance`` rule."""
    from .handlers.prose import CHECKER

    text = r.rule.get("statement") or r.quote
    g = {"type": "guidance", "text": text, "tools": [], "quote": text, "by": "model", "from_verify": r.id}
    g["id"] = "g_" + r.id
    return _wrap(g, r.statement_ids, r.source, CHECKER["guidance"])


def retype_verify(
    rules: List[Rule], specs: List[Dict[str, Any]], model: Callable[[str, str], str]
) -> Tuple[List[Rule], Dict[str, int], Dict[str, List[str]]]:
    """Each ``verify`` rule is replaced by the typed rules ``retype_rule`` finds for it, or stays as it is; one about a
    change after a delay becomes ``guidance``. Returns (the rules, counts, statement ids by outcome: ``typed`` and
    ``delayed``). The calls run in parallel."""
    from .handlers.prose import CHECKER

    by_name = {s["name"]: s for s in specs}
    catalog = _write_catalog(specs)
    verify = [r for r in rules if r.type == "verify"]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        found = {
            id(r): res
            for r, res in zip(
                verify, pool.map(lambda r: retype_rule(r.rule, by_name, catalog, model), verify)
            )
        }
    out: List[Rule] = []
    ids: Dict[str, List[str]] = {"typed": [], "delayed": []}
    counts = {"verify_retyped": 0, "typed_rules_made": 0, "retype_errors": 0, "delayed_to_guidance": 0}
    for r in rules:
        good, status = found.get(id(r), ([], "ok"))
        counts["retype_errors"] += status == "failed"
        if status == "delayed":
            out.append(_as_guidance(r))
            counts["delayed_to_guidance"] += 1
            ids["delayed"] += r.statement_ids
        elif good:
            out += [_wrap(g, r.statement_ids, r.source, CHECKER.get(g["type"], "code")) for g in good]
            counts["verify_retyped"] += 1
            counts["typed_rules_made"] += len(good)
            ids["typed"] += r.statement_ids
        else:
            out.append(r)
    return out, counts, ids


# -- the pipeline step --------------------------------------------------------------------------------------------------

QUESTION = (
    "The statement says what must or must not be done, and no write tool names it or fits a typed rule: "
    '"{text}" Is it guidance for the agent, a behaviour of the system, or a rule for a tool that the build did not find?'
)


def repair_rules(
    rules: List[Rule],
    statements: List[Statement],
    specs: List[Dict[str, Any]],
    model: Optional[Callable[[str, str], str]],
    *,
    grant_may_deny: bool = False,
) -> Tuple[List[Rule], List[CoverageItem], List[Dict[str, Any]], Dict[str, int]]:
    """The repairs above on the rules of a build, in this order: ``when`` values, tools of verify rules, grants of role
    rows (a row with a rule needs no verify rule), typed rules for the verify rules that are left, guidance for those
    that no tool can trigger.

    Returns (the rules, the coverage items of the statements whose outcome changed, policy questions, counts). A coverage
    item replaces any earlier one of the statement. A statement that says "must" or "must not" and ends as guidance with
    no check also gets a policy question. Without a ``model`` only the steps that need none run."""
    from .handlers.prose import CHECKER

    by_id = {s.id: s for s in statements}
    stats = {
        "when_normalized": 0,
        "verify_tools_added": 0,
        "verify_retyped": 0,
        "typed_rules_made": 0,
        "retype_errors": 0,
        "delayed_to_guidance": 0,
        "role_rows": 0,
        "grant_rules": 0,
        "role_rows_all": 0,
        "grant_errors": 0,
        "verify_to_guidance": 0,
        "questions": 0,
    }
    changed: Dict[str, CoverageItem] = {}
    questions: List[Dict[str, Any]] = []

    def note(statement_id: str, kind: str, reason: str) -> None:
        st = by_id.get(statement_id)
        changed[statement_id] = CoverageItem(
            statement_id, kind, reason, dict(st.source) if st else {}, st.text if st else ""
        )

    rules = list(rules)
    stats["when_normalized"] = sum(normalize_when(r.rule) for r in rules)
    stats["verify_tools_added"] = attach_verify_tools([r.rule for r in rules if r.type == "verify"], specs)

    if model is not None:
        # 3. grants first: a role row that has a rule (or grants every tool) needs no verify rule and no typed rules
        rows = role_rows(statements)
        stats["role_rows"] = len(rows)
        have = {r.id for r in rules}
        settled = set()
        for row, grant, why in _grants(rows, specs, model, grant_may_deny):
            if grant is None:
                if why == "all":
                    settled.add(row["id"])
                    stats["role_rows_all"] += 1
                    if not any(r.type != "verify" and row["id"] in r.statement_ids for r in rules):
                        note(
                            row["id"], "out_of_scope", "the role may call every write tool: nothing to forbid"
                        )
                else:
                    stats["grant_errors"] += 1
                continue
            settled.add(row["id"])
            if grant["id"] in have:
                continue
            have.add(grant["id"])
            ids = grant.pop("statement_ids")
            rules.append(_wrap(grant, ids, by_id[row["id"]].source, CHECKER["role_permission"]))
            stats["grant_rules"] += 1
            note(
                row["id"],
                "rule",
                f"role rule: forbids {len(grant['tools'])} write tools for {grant['roles'][0]}",
            )
        rules = [
            r
            for r in rules
            if not (r.type == "verify" and r.statement_ids and set(r.statement_ids) <= settled)
        ]

        # 4. typed rules for the verify rules that are left
        rules, counts, ids = retype_verify(rules, specs, model)
        stats.update(counts)
        for sid in ids["typed"]:
            note(sid, "rule", "typed rules, replacing a verify rule")
        for sid in ids["delayed"]:
            note(
                sid, "guidance", "a change after a delay: the system does it, shown to the agent, not checked"
            )

    # 5. a verify rule no write can trigger is guidance
    out = []
    for r in rules:
        if r.type == "verify" and not r.rule.get("tools"):
            out.append(_as_guidance(r))
            stats["verify_to_guidance"] += 1
            for sid in r.statement_ids:
                note(sid, "guidance", "no write tool names this statement: shown to the agent, not checked")
                st = by_id.get(sid)
                if st is not None and getattr(st.modality, "value", st.modality) in ("must", "must_not"):
                    questions.append(
                        {
                            "statement_ids": [sid],
                            "question": QUESTION.format(text=" ".join(st.text.split())[:200]),
                            "default_level": "observe",
                        }
                    )
        else:
            out.append(r)
    stats["questions"] = len(questions)
    return out, list(changed.values()), questions, stats
