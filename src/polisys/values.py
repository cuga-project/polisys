"""Values as the checks compare them: text with dashes and spaces folded (``canon``, ``norm``), a number or an ISO date
read from a value, the operators of a record condition (``_condition``) and the ``when`` of a rule (``_when``). The
compile step and the run-time checks both read values through this file, so they agree."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"), "-")
_DASHES.update(dict.fromkeys(map(ord, "\u00a0\u202f\u2007"), " "))


def canon(value: Any) -> str:
    """Text as compared by checks: NFKC, Unicode dashes as '-', non-breaking spaces as spaces (policy text and
    model output often differ only in these: 'product\u2011voc@' vs 'product-voc@')."""
    import unicodedata

    return unicodedata.normalize("NFKC", str(value)).translate(_DASHES)


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]", " ", canon(value).lower())).strip()


def _walk(obj: Any) -> Iterable[Any]:
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value))
    return float(m.group(0).replace(",", "")) if m else None


ISO_MOMENT = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}:?\d{2})?)?\s*$"
)


def _iso_moment(value: Any) -> Optional[datetime]:
    """An ISO 8601 date (``2025-02-25``) or date and time (``2025-02-25T12:08:00-05:00``) as a UTC moment, else None. Only
    a whole string in that form counts: a date inside other text, ``02/25/2025`` and a number are not ISO dates. A date
    with no time is midnight, and a time with no offset is read as UTC."""
    if not isinstance(value, str) or not ISO_MOMENT.match(value):
        return None
    try:
        moment = datetime.fromisoformat(value.strip().replace(" ", "T", 1).replace("Z", "+00:00"))
    except ValueError:  # 2025-13-45
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


def _condition(actual: Any, op: str, value: Any) -> bool:
    """The record-condition operators: contains, equals, in, gt, ge, lt, le, empty."""
    if op == "empty":
        return actual in (None, "", [], {})
    if op == "contains":
        return value is not None and str(value).lower() in str(actual or "").lower()
    if op == "equals":
        return norm(actual) == norm(value)
    if op == "in":
        return norm(actual) in {norm(v) for v in (value or [])}
    when_a, when_v = _iso_moment(actual), _iso_moment(value)
    if (
        when_a is not None and when_v is not None
    ):  # two ISO dates are compared as dates, not by their first number
        return {
            "gt": when_a > when_v,
            "ge": when_a >= when_v,
            "lt": when_a < when_v,
            "le": when_a <= when_v,
        }.get(op, False)
    a, v = (
        _number(actual),
        _number(value),
    )  # a number, or a value of another form: the first number, as before
    if a is None or v is None:
        return False
    return {"gt": a > v, "ge": a >= v, "lt": a < v, "le": a <= v}.get(op, False)


def _values(slot: Any) -> Any:
    """A rule slot that should hold a list. A model may write one value alone (``"when": {"is_read": true}``): it counts
    as one value. Nothing counts as no value. A string stays as it is, so the text and checks of measured builds do not
    change."""
    if slot is None:
        return []
    return slot if isinstance(slot, (list, tuple, set, dict, str)) else [slot]


def _joined(separator: str, slot: Any) -> str:
    """The values of a rule slot as text. They need not be strings (``"allowed": [true]``)."""
    return separator.join(str(v) for v in _values(slot))


def _when(when: Optional[Dict[str, List[Any]]], args: Dict[str, Any]) -> bool:
    if not when:
        return True
    return all(
        (args.get(k) not in (None, ""))
        if "*" in [str(v) for v in _values(vals)]
        else norm(args.get(k)) in {norm(v) for v in _values(vals)}
        for k, vals in when.items()
    )
