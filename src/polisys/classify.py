"""Statement type classification: each statement gets one or two ``StatementType`` values and a modality.

Code first, model second:

1. Table rows are typed by the shape of their table (no model call): a term/meaning table → ``definition``; a
   role/may table → ``role_permission``; rows with arrows → ``state_transition``; a table with two or more short
   input columns and one output column → ``lookup_matrix``; a two-column value list → ``allowed_values``.
2. Prose lines with an arrow ("new → in progress") → ``state_transition``.
3. Everything else: one model call per section, all statements of the section numbered in one prompt.

A type the model invents, a missing answer, a model error or invalid JSON all give ``open_judgment`` and a
coverage note, so nothing is lost and the statement can still be checked by the verifier.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional

from .llm import parse_json
from .model import CoverageItem, Document, Modality, Section, Statement, StatementType

ARROW = re.compile(r"(→|->)")
SHORT_CELL_WORDS = 4

TYPE_GUIDE = "\n".join(
    f"- {t.value}: {g}"
    for t, g in [
        (StatementType.ALLOWED_VALUES, "a field may only take listed values"),
        (StatementType.LOOKUP_MATRIX, "a value is looked up from other values (a table or matrix)"),
        (StatementType.STATE_TRANSITION, "which status changes are allowed"),
        (StatementType.REQUIRED_FIELD, "a field or argument must be present"),
        (StatementType.DERIVED_VALUE, "a value is computed from other values"),
        (StatementType.THRESHOLD, "a number limit (amount, count, days) and what happens above or below it"),
        (StatementType.ROLE_PERMISSION, "who may or may not do an action"),
        (StatementType.APPROVAL, "an action needs approval by a named role or person"),
        (StatementType.PROHIBITION, "something must never be done"),
        (
            StatementType.ELIGIBILITY_FILTER,
            "which records an action applies to or must skip (by a field or condition)",
        ),
        (StatementType.ROUTING, "where work goes: a group, queue, person or channel"),
        (StatementType.DEADLINE, "a time limit or time window"),
        (StatementType.NOTIFICATION_DUTY, "someone must be told when something happens"),
        (StatementType.PROCEDURE, "steps that must be done in order"),
        (StatementType.EXCEPTION_OVERRIDE, "an exception to, change of, or replacement for another rule"),
        (StatementType.DEFINITION, "the meaning of a term"),
        (StatementType.FORMAT, "the exact form of a value, text, ID or template"),
        (StatementType.SOURCE_OF_TRUTH, "which data source wins when sources disagree"),
        (StatementType.OPEN_JUDGMENT, "needs judgment; none of the above"),
    ]
)

SYSTEM = "You classify company policy statements for an assistant that must follow them. Output JSON only."
USER = """Statement types:
{guide}

Tools the assistant can call (for context):
{catalog}

Section {sid} {title}, statements:
{statements}

For each statement give one type, or two if it clearly has both, and its modality (must, must_not, may, should).
Return {{"statements": [{{"n": <number>, "types": ["<type>"], "modality": "<modality>"}}]}} with one entry per statement."""

_TYPES = {t.value: t for t in StatementType}
_MODALITIES = {m.value: m for m in Modality}


def _short(cells: List[str]) -> bool:
    return all(len(c.split()) <= SHORT_CELL_WORDS for c in cells if c)


def _table_type(st: Statement) -> Optional[StatementType]:
    header = [h.lower() for h in (st.source.get("table_header") or []) if h is not None]
    if not header:
        return None
    cells = [c.strip() for c in st.text.split("|")]
    joined = " ".join(header)
    if ARROW.search(st.text):
        return StatementType.STATE_TRANSITION
    if header[0] in ("term", "word", "name") or "meaning" in joined or "definition" in joined:
        return StatementType.DEFINITION
    if "role" in header[0]:  # a table that lists roles states what each role may or may not do
        return StatementType.ROLE_PERMISSION
    if len(header) >= 3 and _short(cells):
        return StatementType.LOOKUP_MATRIX
    if (
        len(header) == 2
        and _short(cells[:1])
        and any(w in joined for w in ("value", "code", "status", "option"))
    ):
        return StatementType.ALLOWED_VALUES
    return None


def _code_type(st: Statement) -> Optional[StatementType]:
    if st.source.get("table_header"):
        return _table_type(st)
    if ARROW.search(st.text) and len(st.text) < 200:
        return StatementType.STATE_TRANSITION
    return None


def _fallback(st: Statement, reason: str) -> CoverageItem:
    st.types = [StatementType.OPEN_JUDGMENT]
    return CoverageItem(
        statement_id=st.id, kind="classify_fallback", reason=reason, source=st.source, quote=st.text
    )


def classify_section(
    section: Section, model: Optional[Callable[[str, str], str]], tool_catalog: str = ""
) -> List[CoverageItem]:
    notes: List[CoverageItem] = []
    todo: List[Statement] = []
    for st in section.statements:
        t = _code_type(st)
        if t is not None:
            st.types = [t]
        else:
            todo.append(st)
    if not todo:
        return notes
    if model is None:
        return [_fallback(st, "no model for prose statements") for st in todo]
    body = "\n".join(f"{i}. {st.text}" for i, st in enumerate(todo, 1))
    try:
        reply = parse_json(
            model(
                SYSTEM,
                USER.format(
                    guide=TYPE_GUIDE,
                    catalog=tool_catalog or "(none)",
                    sid=section.id,
                    title=section.title,
                    statements=body,
                ),
            )
        )
    except Exception as exc:  # a model error must not stop the build
        return [_fallback(st, f"model error: {type(exc).__name__}") for st in todo]
    answers: Dict[int, dict] = {}
    for a in reply.get("statements") or []:
        if isinstance(a, dict) and isinstance(a.get("n"), int):
            answers[a["n"]] = a
    for i, st in enumerate(todo, 1):
        a = answers.get(i)
        if a is None:
            notes.append(_fallback(st, "no answer from the model"))
            continue
        raw = [str(x).strip().lower() for x in (a.get("types") or [])][:2]
        types = [_TYPES[x] for x in raw if x in _TYPES]
        if not types:
            notes.append(_fallback(st, f"unknown type from the model: {json.dumps(raw)}"))
        else:
            st.types = types
            if len(types) < len(raw):
                notes.append(
                    CoverageItem(
                        statement_id=st.id,
                        kind="classify_note",
                        reason=f"dropped unknown type(s): {json.dumps(raw)}",
                        source=st.source,
                    )
                )
        st.modality = _MODALITIES.get(str(a.get("modality", "")).strip().lower())
    return notes


def classify(
    document: Document, model: Optional[Callable[[str, str], str]], tool_catalog: str = "", workers: int = 4
) -> List[CoverageItem]:
    """Set ``types`` and ``modality`` on every statement of ``document`` (in place), sections in parallel.
    Returns coverage notes."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda sec: classify_section(sec, model, tool_catalog), document.sections))
    return [n for r in results for n in r]
