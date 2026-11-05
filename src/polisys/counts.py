"""What a policy package holds, in numbers: the count behind a row of the coverage matrix (``cli counts``).

No model and no tool schema is needed. Statements are counted by modality and by outcome (``validate.statement_outcomes``, the
same reading as ``coverage_incomplete``); rules by class, level and type; ``verify`` rules apart, because a verify rule
checks nothing at run time without a verifier model. Everything is a plain count of the package, never a judgment of it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from .model import PolicyPackage
from .validate import OUTCOMES, statement_outcomes


def package_counts(pkg: PolicyPackage) -> Dict[str, Any]:
    statements = [st for d in pkg.documents for st in d.statements]
    outcomes = statement_outcomes(pkg)
    modality = Counter(str(getattr(st.modality, "value", st.modality) or "none") for st in statements)
    by_outcome = Counter(o for kinds in outcomes.values() for o in kinds)
    must = [
        st for st in statements if str(getattr(st.modality, "value", st.modality)) in ("must", "must_not")
    ]
    rules = pkg.rules
    return {
        "documents": len(pkg.documents),
        "statements": len(statements),
        "by_modality": dict(sorted(modality.items())),
        "must_statements": len(must),
        "must_without_outcome": sum(1 for st in must if not outcomes.get(st.id)),
        "statements_with_outcome": sum(1 for kinds in outcomes.values() if kinds),
        "by_outcome": {k: by_outcome[k] for k in OUTCOMES if by_outcome[k]},
        "rules": len(rules),
        "rules_by_class": dict(sorted(Counter(r.cls or "none" for r in rules).items())),
        "rules_by_level": dict(sorted(Counter(r.level or "none" for r in rules).items())),
        "rules_by_type": dict(sorted(Counter(r.type for r in rules).items())),
        "verify_rules": sum(1 for r in rules if r.type == "verify"),
        "procedures": len(pkg.procedures),
        "steps": sum(len(p.steps) for p in pkg.procedures),
        "steps_with_action": sum(1 for p in pkg.procedures for s in p.steps if s.action),
        "must_nots": sum(len(p.must_not) for p in pkg.procedures),
        "obligations": len(pkg.obligations),
        "obligations_by_level": dict(sorted(Counter(o.level for o in pkg.obligations).items())),
        "policy_questions": len(pkg.policy_questions),
        "tests": len([c for c in pkg.tests if isinstance(c, dict) and c.get("tool") and c.get("expected")]),
        "generator": pkg.generator,
    }
