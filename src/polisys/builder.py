"""Offline builder: policy text + tool schemas → typed rule set (see ``ruleset.py``) and a coverage report.

Steps (policy-flow.html, "Graph methodology"):
1. split    — sections by numbered headings; preamble kept; statements are bullets, paragraphs and table rows
2. tables   — by code: enumeration tables, input→output matrices, "a → b / c" transition lists
3. classify — by a model, per section: each statement → rule(s), guidance, definition or out of scope
4. validate — tool and argument names must exist in the tool schemas; values must be in the schema enum;
              each quote must be verbatim in the statement; invalid rules go to the coverage report
5. publish  — rule ids are content hashes; results are cached per section hash, so a new policy version only
              re-asks the model for the sections that changed

Nothing here is specific to one policy: the code parsers only know table shapes, and the model only fills the
slots of the fixed rule types.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .documents import LineSection as Section, split_policy  # noqa: F401 (re-exported)
from .llm import openai_json_model  # noqa: F401 (re-exported)
from .validate import check_rule as validate  # noqa: F401 (moved; re-exported)
from .handlers.tables import (  # noqa: F401 (moved; re-exported)
    ARROW,
    _props,
    _required,
    _rid,
    _tables,
    _tools_for_field,
    _values,
    code_rules,
    tool_catalog,
)

from .ruleset import RULE_TYPES

PROMPT_VERSION = "v2"
HEADING = re.compile(r"^#{1,4}\s*\**\s*(\d+(?:\.\d+)*)\.?\s*(.*?)\**\s*$")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("**", "").replace("`", "")).strip()


SYSTEM = (
    "You turn one section of a company policy into typed rules for an automatic checker. Use only tool and argument "
    "names from the tool catalog. Copy quotes word for word from the statement. Output JSON only."
)
TYPES_HELP = """Rule types and their slots:
- required_args: {"tool", "args": [..], "when": {"arg": [values]} (optional)} - the call must include these arguments (when the condition holds)
- allowed_values: {"tool", "arg", "values": [..], "when": {..} (optional)} - the argument must be one of the values
- reference_status: {"tool", "arg", "field", "allowed": [..]} - the record whose ID is passed in arg must have field in allowed
- role_permission: {"tools": [..], "roles": [..], "effect": "forbid"} - users with these roles may not call these tools
- duty: {"trigger": {"tool", "when": {..}}, "then": {"tool", "when": {..}}} - after trigger succeeds, the agent must also call then before it finishes
- grounded_values: {} - global: optional arguments may only use values from the request or from tool results
- id_provenance: {} - global: IDs must come from tool results or the request
- guidance: {"text", "tools": [..]} - an instruction worth showing the agent that cannot be checked automatically
Statement classes: rule, guidance, definition, out_of_scope (with a short reason).
Write a rule only when the statement clearly says it. Do not invent conditions or values.
Code already reads exactly three kinds of content; skip only these: "Table.Field | Allowed Values" lists, state
transition lists ("a -> b / c") and input-to-output lookup matrices. Every other table (for example a role or
permission table) is made of statements: turn each row into rules (role_permission for what a role may not do).
When a statement says something "must" happen and a catalog tool performs that action, it is a duty or a rule for
the assistant, not automatic system behavior. Roles are the role IDs used in the policy (e.g. admin, agent)."""


def _prompt(sec: Section, catalog: str, skip: Dict[int, str]) -> str:
    body = "\n".join(f"L{no}: {_clean(t)}" for no, t in sec.lines if no not in skip)
    return (
        f"{TYPES_HELP}\n\nTool catalog (name(args); * = required; [a|b] = allowed values):\n{catalog}\n\n"
        f"Policy section §{sec.sid} {sec.title}:\n{body}\n\n"
        'Return {"statements": [{"line": <n>, "class": "rule|guidance|definition|out_of_scope", '
        '"reason": "<if not a rule>", "rules": [{"type": "...", "quote": "...", <slots>}]}]}'
    )


def _parse_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


@dataclass
class BuildResult:
    policy_hash: str
    rules: List[Dict[str, Any]]
    coverage: List[Dict[str, Any]]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "prompt_version": PROMPT_VERSION,
            "stats": self.stats,
            "rules": self.rules,
            "coverage": self.coverage,
        }


def build(
    policy: str,
    specs: List[Dict[str, Any]],
    model: Optional[Callable[[str, str], str]] = None,
    cache_dir: Optional[Path] = None,
) -> BuildResult:
    sections = split_policy(policy)
    rules, used = code_rules(sections, specs)
    by_name = {t["name"]: t for t in specs}
    catalog = tool_catalog(specs)
    coverage: List[Dict[str, Any]] = [
        {"source": f"L{n}", "class": "rule", "by": "code", "how": how} for n, how in sorted(used.items())
    ]
    model_name = getattr(model, "model_name", "none")
    calls = cached = 0
    for sec in sections:
        todo = [(n, t) for n, t in sec.lines if n not in used]
        if not todo:
            continue
        key = hashlib.sha256(
            f"{PROMPT_VERSION}|{model_name}|{sec.hash}|{hashlib.sha256(catalog.encode()).hexdigest()}".encode()
        ).hexdigest()[:20]
        path = cache_dir / f"{key}.json" if cache_dir else None
        if path and path.exists():
            data, cached = json.loads(path.read_text()), cached + 1
            # cached answers carry the line numbers of the version that made them: map them by statement text
            by_text = {_clean(t): n for n, t in sec.lines}
            old_text = {int(k): v for k, v in (data.get("_lines") or {}).items()}
            for st in data.get("statements", []):
                st["line"] = by_text.get(old_text.get(st.get("line"), ""), st.get("line"))
        elif model is None:
            data = {
                "statements": [{"line": n, "class": "unprocessed", "reason": "no model"} for n, _ in todo]
            }
        else:
            data, calls = _parse_json(model(SYSTEM, _prompt(sec, catalog, used))), calls + 1
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({**data, "_lines": {n: _clean(t) for n, t in sec.lines}}, indent=1)
                )
        lines = dict(sec.lines)
        for st in data.get("statements", []):
            no = st.get("line")
            text = lines.get(no, "")
            entry = {
                "source": f"§{sec.sid} L{no}",
                "class": st.get("class"),
                "reason": st.get("reason"),
                "rules": [],
                "rejected": [],
            }
            for r in st.get("rules") or []:
                why = validate(r, by_name, text)
                if why:
                    entry["rejected"].append({"type": r.get("type"), "why": why})
                    continue
                r = {**r, "source": f"§{sec.sid} L{no}", "by": "model"}
                r["id"] = _rid(r["type"], {k: v for k, v in r.items() if k not in ("source", "quote", "by")})
                if r["type"] in ("grounded_values", "id_provenance") and any(
                    x["type"] == r["type"] for x in rules
                ):
                    entry["rules"].append(next(x["id"] for x in rules if x["type"] == r["type"]))
                    continue  # one global rule of each kind
                rules.append(r)
                entry["rules"].append(r["id"])
            coverage.append(entry)
    stats = {
        "sections": len(sections),
        "statements": sum(len(s.lines) for s in sections),
        "rules": len(rules),
        "by_type": {
            t: sum(r["type"] == t for r in rules) for t in RULE_TYPES if any(r["type"] == t for r in rules)
        },
        "rejected": sum(len(c.get("rejected", [])) for c in coverage),
        "model_calls": calls,
        "cached_sections": cached,
        "model": model_name,
    }
    return BuildResult(hashlib.sha256(policy.encode()).hexdigest()[:10], rules, coverage, stats)


# -- policy sets: the full pipeline (documents → types → links → handlers → validation) -----------------------------

PIPELINE_VERSION = "p1"
# coverage kinds of a statement that a repair can change: the repair writes the new outcome in their place
REPLACED_COVERAGE = ("verify", "prose_fallback", "rule", "guidance", "out_of_scope")


def outcome_kinds(coverage: List[Any], rules: List[Any], questions: List[Dict[str, Any]]) -> List[Any]:
    """Coverage items with the outcomes of design section 9 where the item says what became of a statement: ``rule`` is
    ``hard_rule``, ``judgment_rule`` or ``guidance`` by the class of the rules that name the statement, ``verify`` is
    ``judgment_rule``, ``step`` and ``procedure_rule`` are ``procedure_step``. Notes (``classify_fallback``,
    ``dangling_link``, ...) stay as they are. A statement that a policy question names also gets a ``question`` item."""
    from .model import CoverageItem
    from .package import rule_class

    classes: Dict[str, set] = {}
    for r in rules:
        for sid in r.statement_ids:
            classes.setdefault(sid, set()).add(rule_class(r.type))
    outcome = {"hard": "hard_rule", "judgment": "judgment_rule", "guidance": "guidance"}

    def kind_of(c: Any) -> str:
        if c.kind == "rule":
            have = classes.get(c.statement_id, {"hard"})
            return outcome[next(k for k in ("hard", "judgment", "guidance") if k in have)]
        return {"verify": "judgment_rule", "step": "procedure_step", "procedure_rule": "procedure_step"}.get(
            c.kind, c.kind
        )

    out = [CoverageItem(c.statement_id, kind_of(c), c.reason, c.source, c.quote, c.extra) for c in coverage]
    asked = {sid for q in questions for sid in q.get("statement_ids") or []}
    by_id = {c.statement_id: c for c in out}
    for sid in sorted(asked):
        c = by_id.get(sid)
        out.append(
            CoverageItem(sid, "question", "policy question open", c.source if c else {}, c.quote if c else "")
        )
    return out


def cached_model(
    model: Optional[Callable[[str, str], str]], cache_dir: Optional[Path]
) -> Optional[Callable[[str, str], str]]:
    """Wrap a model so the same (system, user) prompt is answered from disk. Keeps ``usage`` and ``model_name``."""
    if model is None or cache_dir is None:
        return model
    name = getattr(model, "model_name", "model")
    stats = {"hits": 0}

    def call(system: str, user: str) -> str:
        key = hashlib.sha256(f"{PIPELINE_VERSION}|{name}|{system}|{user}".encode()).hexdigest()[:24]
        path = Path(cache_dir) / f"{key}.json"
        if path.exists():
            stats["hits"] += 1
            return json.loads(path.read_text())["answer"]
        answer = model(system, user)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"answer": answer}))
        return answer

    call.model_name = name  # type: ignore[attr-defined]
    call.usage = getattr(model, "usage", {})  # type: ignore[attr-defined]
    call.cache = stats  # type: ignore[attr-defined]
    return call


def build_policy_set(
    policy_set,
    specs: List[Dict[str, Any]],
    model: Optional[Callable[[str, str], str]] = None,
    cache_dir: Optional[Path] = None,
    repair_rounds: int = 2,
    generator: Optional[Dict[str, Any]] = None,
    link_model: bool = True,
    repair: bool = True,
    grant_may_deny: bool = False,
):
    """Build a ``PolicyBuild`` from a ``PolicySet`` with code first and the model only where code cannot decide.

    1. classify statement types (tables by code; prose by the model, per section);
    2. find links (references, exceptions, supersedes, defined terms; the model may propose more);
    3. rules from tables and transition lists (code);
    4. procedures from numbered steps (step graphs);
    5. definitions;
    6. rules from prose (model slot filling; invalid or missing rules become ``verify`` rules);
    7. repair: statements whose rules were rejected are sent again with the checker's messages;
    8. fold exceptions into rules; drop rules of superseded statements;
    9. repair (``rule_repair.repair_rules``): ``when`` values as lists, tools for ``verify`` rules, typed rules for
       ``verify`` rules, role rules from grant tables (``remind`` unless ``grant_may_deny``), guidance and a policy
       question for a ``verify`` rule that no write tool can trigger;
    10. coverage kinds as the outcomes of design section 9 (``outcome_kinds``);
    11. validate the whole build.

    ``repair=False`` gives the build as it was before step 9 existed.
    """
    from .classify import classify
    from .handlers.links import drop_superseded, find_links, fold_exceptions
    from .handlers.procedure_rules import enrich_procedures, general_rules
    from .handlers.procedures import procedures
    from .handlers.prose import prose_rules
    from .handlers.tables import table_rules
    from .model import CoverageItem, Definition, PolicyBuild, StatementType
    from .rule_repair import repair_rules
    from .validate import validate_build

    m = cached_model(model, cache_dir)
    catalog = tool_catalog(specs)
    coverage: List[Any] = []
    for doc in policy_set.documents:
        coverage += classify(doc, m, catalog)
    statements = [s for d in policy_set.documents for s in d.statements]
    links, cov = find_links(policy_set, m if link_model else None)
    coverage += cov
    rules, consumed = table_rules(statements, specs)
    coverage += [CoverageItem(sid, "rule", "read by code from a table or list") for sid in consumed]
    titles = {}
    for d in policy_set.documents:  # a subsection heading may have been parsed as its first statement
        by_id = {sec.id: sec for sec in d.sections}
        for sec in d.sections:
            if sec.title:
                titles[f"{d.id}:{sec.id}"] = sec.title
                continue
            parent = by_id.get(getattr(sec, "parent", None) or "")
            head = sec.statements[0].text if sec.statements and len(sec.statements[0].text) <= 80 else ""
            name = " - ".join(x for x in ((parent.title if parent else ""), head) if x)
            if name:
                titles[f"{d.id}:{sec.id}"] = name
    procs = procedures(statements, specs, m, titles)
    procs, pr_cov = enrich_procedures(
        procs, statements, links, m
    )  # preconditions, exceptions, templates, escalations
    coverage += pr_cov
    procs, g_cov = general_rules(procs, statements, m)  # case rules outside the procedures
    coverage += g_cov
    step_ids = {s.id for p in procs for s in p.steps}
    definitions = []
    for st in statements:
        if StatementType.DEFINITION in st.types:
            term, _, meaning = st.text.partition("|") if "|" in st.text else (st.text, "", "")
            definitions.append(
                Definition(
                    term=term.strip(),
                    meaning=meaning.strip() or st.text,
                    used_by=[lk.dst for lk in links if lk.src == st.id],
                    source={**st.source, "statement": st.id},
                    quote=st.text,
                )
            )
    done = set(consumed) | step_ids | {d.source["statement"] for d in definitions}
    prose = [s for s in statements if s.id not in done]
    p_rules, p_cov = prose_rules(prose, specs, m, catalog, skip_types=())  # every statement no handler took
    for _ in range(repair_rounds):
        retry = {
            c.statement_id: c.reason
            for c in p_cov
            if c.kind == "verify"
            and c.reason not in ("no rule type fits", "no model")
            and not c.reason.startswith("model class")  # said by the model, not rejected by the checker
        }
        if not retry or m is None:
            break
        again = [s for s in prose if s.id in retry]
        r2, c2 = prose_rules(again, specs, m, catalog, feedback=retry, skip_types=())
        fixed = {c.statement_id for c in c2 if c.kind != "verify"}
        if not fixed:
            break
        p_rules = [r for r in p_rules if not (r.type == "verify" and r.statement_ids[0] in fixed)] + [
            r for r in r2 if r.statement_ids and r.statement_ids[0] in fixed
        ]
        p_cov = [c for c in p_cov if c.statement_id not in fixed] + [c for c in c2 if c.statement_id in fixed]
    coverage += p_cov
    coverage += [CoverageItem(s.id, "step", "procedure step") for s in statements if s.id in step_ids]
    coverage += [CoverageItem(d.source["statement"], "definition", d.term) for d in definitions]
    rules += p_rules
    unique: Dict[str, Any] = {}
    for r in rules:  # one global rule of each kind; same rule from two statements kept once
        key = r.type if r.type in ("grounded_values", "id_provenance") else r.id
        if key in unique:
            unique[key].statement_ids = sorted(set(unique[key].statement_ids) | set(r.statement_ids))
        else:
            unique[key] = r
    rules = fold_exceptions(list(unique.values()), links, policy_set)
    rules, sup_cov = drop_superseded(rules, links, policy_set)
    coverage += [  # a replaced statement is out of scope, with the reason (design section 9)
        CoverageItem(c.statement_id, "out_of_scope", c.reason, c.source, c.quote) for c in sup_cov
    ]
    repairs: Dict[str, int] = {}
    questions: List[Dict[str, Any]] = []
    if repair:
        rules, changed, questions, repairs = repair_rules(
            rules, statements, specs, m, grant_may_deny=grant_may_deny
        )
        known = {c.statement_id: c for c in changed}
        coverage = [c for c in coverage if not (c.statement_id in known and c.kind in REPLACED_COVERAGE)]
        coverage += changed
    coverage = outcome_kinds(coverage, rules, questions)
    name = getattr(model, "model_name", "none") if model is not None else "none"
    build = PolicyBuild(
        version=policy_set.version,
        rules=rules,
        procedures=procs,
        links=links,
        definitions=definitions,
        coverage=coverage,
        generator=generator or {"kind": "pipeline", "harness": "builder", "model": name},
        policy_questions=questions,
    )
    build.validation = [vars(e) for e in validate_build(build, specs, policy_set)]
    conflicted = {e["statement_id"] for e in build.validation if e["code"] == "schema_conflict"}
    fields = set()
    for r in build.rules:  # a value list the tool rejects can never be followed: remind, never deny
        if r.rule.get("by") == "code" and set(r.statement_ids) & conflicted and r.type == "allowed_values":
            r.may_deny, r.rule["may_deny"], r.rule["schema_conflict"] = False, False, True
            fields.add((r.rule.get("tool"), r.rule.get("arg")))
    for r in build.rules:  # other rules on the same field (e.g. "mandatory") would leave the agent stuck
        args = [r.rule.get("arg")] + list(r.rule.get("args") or [])
        if any((r.rule.get("tool"), a) in fields for a in args if a):
            r.may_deny, r.rule["may_deny"] = False, False
    kinds: Dict[str, int] = {}
    for c in coverage:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    build.stats = {
        "documents": len(policy_set.documents),
        "statements": len(statements),
        "rules": len(rules),
        "by_type": {t: sum(r.type == t for r in rules) for t in sorted({r.type for r in rules})},
        "procedures": len(procs),
        "links": len(links),
        "coverage": kinds,
        "validation_errors": len(build.validation),
        "model": name,
        **({"repair": repairs} if repair else {}),
        "model_calls": (getattr(model, "usage", {}) or {}).get("calls"),
        "usage": dict(getattr(model, "usage", {}) or {}),
        "cache_hits": (getattr(m, "cache", {}) or {}).get("hits"),
    }
    return build
