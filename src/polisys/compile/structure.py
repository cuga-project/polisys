"""How a document is structured: its heading and numbering styles, and how it refers to its own parts.

Documents number their parts in many ways: ``4.2``, ``Article II``, ``(a)``, ``I.``, ``Rule 7:``, unnumbered
bold titles. A **structure profile** lists the patterns of one document as regular expressions:

- ``headings``: ``{"pattern", "level", "kind"}``; the pattern has a named group ``id`` (and maybe ``title``);
  ``kind`` is ``heading`` (the rest of the line is a title) or ``clause`` (the rest of the line is a statement);
  ``level`` 1 is the top; a level-2 part gets its parent's id in front (``II`` + ``(a)`` → ``II.a``);
- ``references``: patterns with a named group ``ref`` that point to a part ("clause (d)", "Article IV").

``DEFAULT_PROFILE`` covers common styles with code only. ``learn_structure`` asks a small model once per document
to read candidate lines and add the document's own patterns; every learned pattern must compile, have the named
group, match at least one line and not match most lines. Code then applies the patterns, so parsing stays fast and
repeatable. Nothing here knows a benchmark or a document.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from .llm import parse_json

DEFAULT_PROFILE: Dict[str, Any] = {
    "headings": [
        {
            "pattern": r"^(?:ARTICLE|Article|PART|Part|CHAPTER|Chapter|SECTION|Section|CLAUSE|Clause|SCHEDULE|Schedule|"
            r"ANNEX|Annex|EXHIBIT|Exhibit|RULE|Rule)\s+(?P<id>[0-9]+[A-Za-z]?|[IVXLC]+)\b[.:\-–]?\s*(?P<title>.*)$",
            "level": 1,
            "kind": "heading",
        },
        {"pattern": r"^(?P<id>[IVX]{1,5})\.\s+(?P<title>[A-Z][^.]{0,80})$", "level": 1, "kind": "heading"},
        {"pattern": r"^\((?P<id>[a-z])\)\s+", "level": 2, "kind": "clause"},
    ],
    "references": [
        r"\b(?:Article|Part|Chapter|Clause|Schedule|Annex|Exhibit|Rule|Item)\s+(?P<ref>[0-9]+(?:\.[0-9]+)*[A-Za-z]?|[IVXLCDM]+|[A-Z])\b",
        r"\bclause\s+\((?P<ref>[a-z])\)",
        r"\bparagraph\s+(?P<ref>\d+(?:\.\d+)*)",
    ],
}

CANDIDATE_HEADING = re.compile(
    r"^\s*(#{1,6}\s|\(?[0-9]{1,3}[.)]|\(?[A-Za-z][.)]\s|\([ivx]{1,4}\)|[IVXLC]{1,6}[.)]\s|"
    r"(ARTICLE|Article|SECTION|Section|PART|Part|CHAPTER|Chapter|CLAUSE|Clause|RULE|Rule|SCHEDULE|Schedule|"
    r"ANNEX|Annex|APPENDIX|Appendix|EXHIBIT|Exhibit)\b|\*\*[^*]{2,80}\*\*\s*$|[A-Z][A-Z0-9 ,&/-]{4,80}$)"
)
CANDIDATE_REF = re.compile(
    r"\b(see|refer|per|under|pursuant|according to|as (?:set out|described|defined) in|"
    r"clause|section|article|paragraph|rule|appendix|schedule|annex|above|below)\b",
    re.I,
)
MAX_HEADING_LINES, MAX_REF_LINES = 120, 40
MAX_SHARE = 0.6  # a heading pattern that matches more than this share of lines is too broad

SYSTEM = "You describe how a policy document is structured. Output JSON only."
USER = """Candidate heading lines (line number: text):
{headings}

Lines that may refer to other parts:
{refs}

Describe the numbering and heading styles of this document, and how it refers to its own parts.
Write Python regular expressions:
- each heading pattern matches a whole line start and has a named group "id" (and "title" if the line has one);
  give its level (1 = top) and kind: "heading" if the rest of the line is a title, "clause" if the rest is a rule;
- each reference pattern has a named group "ref" holding the id that is pointed to.
Return {{"headings": [{{"pattern": "...", "level": 1, "kind": "heading|clause"}}], "references": ["..."]}}"""


def candidate_lines(text: str):
    lines = [ln.strip() for ln in text.splitlines()]
    heads = [(i, ln) for i, ln in enumerate(lines, 1) if ln and CANDIDATE_HEADING.match(ln) and len(ln) < 160]
    refs = [(i, ln) for i, ln in enumerate(lines, 1) if ln and CANDIDATE_REF.search(ln)]
    return lines, heads[:MAX_HEADING_LINES], refs[:MAX_REF_LINES]


def _valid_heading(h: Any, lines: List[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(h, dict) or not isinstance(h.get("pattern"), str):
        return None
    try:
        rx = re.compile(h["pattern"])
    except re.error:
        return None
    if "id" not in rx.groupindex:
        return None
    nonempty = [ln for ln in lines if ln]
    hits = sum(1 for ln in nonempty if rx.match(ln))
    if hits < 2:  # a numbering scheme repeats; one match is usually the document title
        return None
    if len(nonempty) >= 10 and hits > MAX_SHARE * len(nonempty):
        return None
    ids = [m.group("id") for m in (rx.match(ln) for ln in nonempty) if m and m.group("id")]
    if len(ids) >= 4 and len(set(ids)) < 0.8 * len(
        ids
    ):  # section numbers are unique; list numbers restart at 1
        return None
    tops = [i for i in ids if i.isdigit()]
    if any(
        int(b) == 1 and int(a) >= 2 and b in seen
        for a, b, seen in ((tops[k - 1], tops[k], set(tops[:k])) for k in range(1, len(tops)))
    ):
        return None  # top-level numbers that start again at 1 after a higher number: a numbered list, not sections
    level = h.get("level") if isinstance(h.get("level"), int) and 1 <= h["level"] <= 4 else 1
    kind = h.get("kind") if h.get("kind") in ("heading", "clause") else "heading"
    return {"pattern": h["pattern"], "level": level, "kind": kind}


def _ids(pattern: str, lines: List[str]) -> set:
    rx = re.compile(pattern)
    return {
        m.group("id")
        for m in (rx.match(ln) for ln in lines if ln)
        if m and "id" in rx.groupindex and m.group("id")
    }


def _drop_colliding_clauses(headings: List[Dict[str, Any]], lines: List[str]) -> List[Dict[str, Any]]:
    """A clause pattern whose numbers are already the numbers of headings ("# 1 Purpose" and "1. Review the
    application") is numbering list items or steps, not sections: drop it so the items stay steps."""
    heading_ids = set().union(
        *[_ids(h["pattern"], lines) for h in headings if h["kind"] == "heading"] or [set()]
    )
    return [h for h in headings if h["kind"] != "clause" or len(_ids(h["pattern"], lines) & heading_ids) < 2]


def _valid_reference(p: Any, text: str, ref_lines: int) -> Optional[str]:
    """A reference pattern needs the ``ref`` group, literal words around it, and must not match everywhere."""
    if not isinstance(p, str):
        return None
    try:
        rx = re.compile(p, re.I)
    except re.error:
        return None
    if "ref" not in rx.groupindex:
        return None
    context = re.sub(r"\(\?P<ref>(?:[^()]|\([^()]*\))*\)", "", p)
    if len(re.findall(r"[A-Za-z§]", re.sub(r"\\[A-Za-z]|\?[a-zA-Z]*:|\(\?[a-zA-Z]+\)", "", context))) < 1:
        return None
    if len(rx.findall(text)) > 3 * max(ref_lines, 1):
        return None
    return p


def learn_structure(text: str, model: Optional[Callable[[str, str], str]]) -> Dict[str, Any]:
    """The document's structure profile: code defaults plus the patterns a model reads from candidate lines."""
    profile = {
        "headings": list(DEFAULT_PROFILE["headings"]),
        "references": list(DEFAULT_PROFILE["references"]),
        "by": "code",
    }
    if model is None:
        return profile
    lines, heads, refs = candidate_lines(text)
    if not heads and not refs:
        return profile
    try:
        reply = parse_json(
            model(
                SYSTEM,
                USER.format(
                    headings="\n".join(f"{i}: {ln}" for i, ln in heads) or "(none)",
                    refs="\n".join(f"{i}: {ln[:200]}" for i, ln in refs) or "(none)",
                ),
            )
        )
    except Exception:
        return profile
    learned_h = [v for v in (_valid_heading(h, lines) for h in reply.get("headings") or []) if v]
    learned_h = _drop_colliding_clauses(learned_h, lines)
    learned_r = [
        v for v in (_valid_reference(p, text, len(refs)) for p in reply.get("references") or []) if v
    ]
    known = {h["pattern"] for h in profile["headings"]}
    profile["headings"] = [h for h in learned_h if h["pattern"] not in known] + profile["headings"]
    profile["references"] = list(dict.fromkeys(learned_r + profile["references"]))
    profile["by"] = getattr(model, "model_name", "model")
    return profile
