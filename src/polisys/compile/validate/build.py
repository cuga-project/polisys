"""A whole build of rules: ``validate_build`` lists what is wrong with it, ``diff_builds`` compares two builds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from polisys.package.model import Link, PolicyBuild, PolicySet, Rule
from polisys.compile.validate.rule import check_rule

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
        from polisys.checks.rules import RuleSetHooks

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
