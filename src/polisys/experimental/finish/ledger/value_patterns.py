"""How an obligation writes a value: a pattern with value names (``{name:format}``) read into tokens, and a calculation in
words that code can compute (today, or a date plus days)."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, List, Optional, Tuple

from polisys.experimental.guidance.procedure_store import _parse_date

_ADD_DAYS = (
    re.compile(r"^(?P<n>\d+) (?P<unit>calendar|business) days? (?:from|after) (?P<base>[^;,]+)$", re.I),
    re.compile(
        r"^(?P<base>[^;,]+?) (?:plus|\+) (?P<n>\d+) (?P<unit>calendar|business) days?(?: \([^;,]*\))?$", re.I
    ),
)
_TODAY = re.compile(r"^(?:(?:the )?current (?:day|date)|today|today's date)$", re.I)


def _compute_plan(spec: Dict[str, Any]) -> Optional[Tuple[int, str, str]]:
    """What code can compute from the calculation in words: today, or a date plus days, and only when the whole text says
    this. Anything else has no value, and is not asserted. Gives (days, unit, base), with base "" for today."""
    text = re.sub(r"\s+", " ", str(spec.get("fn") or "")).strip()
    if _TODAY.match(text):
        return 0, "calendar", ""
    for shape in _ADD_DAYS:
        m = shape.match(text)
        if m:
            base = m.group("base")
            return int(m.group("n")), m.group("unit").lower(), "" if _TODAY.match(base) else base
    return None


def _read_date(text: Any) -> Optional[dt.date]:
    value = str(text or "").strip()
    return _parse_date(value[:10] if re.match(r"^\d{4}-\d{2}-\d{2}[T ]", value) else value)


_TOKEN = re.compile(r"\{\{|\}\}|\{([A-Za-z_]\w*)(?::([^{}]*))?\}")


def _parse(pattern: str) -> List[Tuple[str, str, str]]:
    """A pattern as parts: ("text", text, "") and ("name", value name, format)."""
    out: List[Tuple[str, str, str]] = []
    pos = 0
    for m in _TOKEN.finditer(pattern):
        if m.start() > pos:
            out.append(("text", pattern[pos : m.start()], ""))
        pos = m.end()
        if m.group(0) in ("{{", "}}"):
            out.append(("text", m.group(0)[0], ""))
        else:
            out.append(("name", m.group(1), m.group(2) or ""))
    if pos < len(pattern):
        out.append(("text", pattern[pos:], ""))
    return out


FIXED, INFERRED, GAP = "fixed", "inferred", "gap"
Token = Tuple[str, str, str]  # text, trust, value name


def _tokens(
    pattern: str, specs: Dict[str, Any], plain: Dict[str, Any], seen: Tuple[str, ...] = ()
) -> List[Token]:
    """A pattern with the known values filled in. Trust: ``fixed`` text can be called wrong (SOP text, ``literal``,
    ``computed`` with a format); ``inferred`` text is compared and never called wrong (``case_fact``, ``request``, a computed
    date in a form the pattern does not say); a ``gap`` is not compared."""
    out: List[Token] = []
    for kind, text, fmt in _parse(pattern):
        if kind == "text":
            out.append((text, FIXED, ""))
            continue
        spec = specs.get(text)
        origin = spec.get("from") if isinstance(spec, dict) else None
        if origin == "template" and text not in seen:
            out += _tokens(str(spec.get("pattern") or ""), specs, plain, seen + (text,))
            continue
        value = plain.get(text)
        if origin not in ("literal", "computed", "case_fact", "request") or value is None:
            out.append(("", GAP, text))
            continue
        if fmt:
            day = _read_date(value)
            if day is None:
                out.append(("", GAP, text))
                continue
            value = day.strftime(fmt)
        trust = FIXED if origin == "literal" or (origin == "computed" and fmt) else INFERRED
        out.append((str(value), trust, text))
    return out
