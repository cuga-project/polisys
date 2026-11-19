"""Coverage: every statement of the policy has an outcome (a rule, a reason it is none), and every outcome has its statement."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from polisys.package.model import PolicyPackage
from polisys.compile.validate.rule import _fold
from polisys.compile.validate.slots import LEGACY_OUTCOME, MIN_QUOTE_WORDS, OUTCOMES, PackageError


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
