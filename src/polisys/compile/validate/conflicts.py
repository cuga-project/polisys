"""Conflicts: two rules that cannot both hold. ``resolve_conflicts`` is the one function that changes a package, and only a
new one."""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from polisys.checks.rules import FREE_TEXT
from polisys.values import _condition, canon, norm
from polisys.package.model import (
    RULE_LEVELS, CoverageItem, LinkType, PolicyPackage, PolicySet, Rule, Statement, resolve_conflict,
)
from polisys.compile.validate.slots import _is_broken, _slot_findings, _trigger_tools
from polisys.compile.validate.fields import _structure_errors


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
    from polisys.package.files import LAYOUT  # package/files.py reads handlers that import this module

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
