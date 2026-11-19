"""The values of one obligation for one case (``resolve_values``) and the tokens that each argument test compares
(``plan_of``)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from polisys.package.model import Obligation
from polisys.experimental.finish.ledger.text import _as_text
from polisys.experimental.finish.ledger.value_patterns import Token, _compute_plan, _parse, _read_date, _tokens
from polisys.experimental.finish.ledger.cases import Case, fact_id, stated_id


def _compute(spec: Dict[str, Any], case: Optional[Case], store: Any, today: str) -> Optional[dt.date]:
    plan = _compute_plan(spec)
    if plan is None:
        return None
    days, unit, base = plan
    start = None
    if not base:
        start = _read_date(today)
    else:
        args = spec.get("args") if isinstance(spec.get("args"), list) else []
        if len(args) == 1 and case is not None:
            start = _read_date((case.facts.get(fact_id(args[0])) or {}).get("value"))
    if start is None:
        return None
    if unit == "calendar":
        return start + dt.timedelta(days=days)
    if store is None or not hasattr(store, "is_business_day"):
        return None  # with no calendar, the business days are not known
    current, left = start, days
    while left:
        current += dt.timedelta(days=1)
        left -= 1 if store.is_business_day(current) else 0
    return current


def resolve_values(
    ob: Obligation, case: Optional[Case], store: Any = None, today: str = ""
) -> Dict[str, Dict[str, Any]]:
    """Each value of the obligation for this case: ``{name: {"value": text or None, "from": origin, "matched": bool}}``. A
    value with no answer is ``None``, and is then not asserted. A ``template`` gets its text when all its parts are known."""
    specs = ob.values if isinstance(ob.values, dict) else {}
    effect = ob.effect if isinstance(ob.effect, dict) else {}
    match = effect.get("match") if isinstance(effect.get("match"), dict) else {}
    named = {
        tok[1]
        for m in match.values()
        if isinstance(m, dict) and isinstance(m.get("pattern"), str)
        for tok in _parse(m["pattern"])
        if tok[0] == "name"
    }
    todo, matched = list(named), set()
    while todo:  # a value is matched through a matched template too
        name = todo.pop()
        if name in matched or name not in specs:
            continue
        matched.add(name)
        spec = specs[name]
        if isinstance(spec, dict) and spec.get("from") == "template":
            todo += [t[1] for t in _parse(str(spec.get("pattern") or "")) if t[0] == "name"]
    plain: Dict[str, Any] = {}
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            continue
        origin = spec.get("from")
        value: Any = None
        if origin == "literal" and "value" in spec:
            value = _as_text(spec["value"])
        elif origin == "case_fact" and case is not None:
            value = (case.facts.get(fact_id(spec.get("field"), spec.get("record") or "")) or {}).get("value")
        elif origin == "request" and case is not None:
            value = (case.stated.get(stated_id(spec.get("about"))) or {}).get("value")
        elif origin == "computed":
            day = _compute(spec, case, store, today)
            value = day.isoformat() if day else None
        plain[name] = value
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            continue
        value = plain.get(name)
        if spec.get("from") == "template":
            toks = _tokens(str(spec.get("pattern") or ""), specs, plain)
            value = "".join(t[0] for t in toks) if all(t[1] != "gap" for t in toks) else None
        out[name] = {
            "value": value,
            "from": spec.get("from"),
            "matched": name in matched,
            "record": spec.get("record") or "",
        }
    return out


def plan_of(
    ob: Obligation, case: Optional[Case], store: Any = None, today: str = ""
) -> Dict[str, List[Token]]:
    """Each matched argument with its pattern, values filled in."""
    effect = ob.effect if isinstance(ob.effect, dict) else {}
    match = effect.get("match") if isinstance(effect.get("match"), dict) else {}
    specs = ob.values if isinstance(ob.values, dict) else {}
    resolved = resolve_values(ob, case, store, today)
    plain = {name: entry["value"] for name, entry in resolved.items() if entry["from"] != "template"}
    return {
        arg: _tokens(m["pattern"], specs, plain)
        for arg, m in match.items()
        if isinstance(m, dict) and isinstance(m.get("pattern"), str)
    }
