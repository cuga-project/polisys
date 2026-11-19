"""``validate_package``: every check of this folder on one policy package, as one list of findings."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from polisys.compile.handlers.tables import _props
from polisys.checks.rules import norm
from polisys.package.model import PolicyPackage, PolicySet, Rule
from polisys.compile.validate.build import ValidationError, _cedar_error, _dangling, _ids, _rule_error
from polisys.compile.validate.slots import (
    COMPLETENESS, PackageError, _is_broken, _label_findings, _severity, _show, _slot_findings,
)
from polisys.compile.validate.fields import _content_errors, _structure_errors
from polisys.compile.validate.obligations import _binding_errors, _obligation_errors
from polisys.compile.validate.invented import _invented_errors
from polisys.compile.validate.coverage import _coverage_errors
from polisys.compile.validate.conflicts import _conflicts


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
    from polisys.package.files import _rules_by_statement

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
