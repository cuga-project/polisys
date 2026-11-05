"""Policy references: the request or the data names a policy source ("per our regional meeting policy",
"See ws_onboarding_policy worksheet", "exclude per editorial calendar guidelines").

``ReferenceLedger`` keeps a short record of the run for the "read before write" check: which policy sources were
referred to, which a later tool output read, and which searches for them found nothing. The layer does not search
for the sources itself; finding and reading them is the agent's job. The layer reminds the agent before a write
while a referred source is still unread.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

NOUNS = r"(?:polic(?:y|ies)|rules?|guidelines?|requirements?|procedures?|SOPs?|playbooks?|settings|thresholds|worksheet)"
CUES = r"(?:our|per|follow(?:ing)?|see|using|according\s+to|in\s+line\s+with|refer\s+to|apply)"
# A cue, then up to 6 words, then a policy noun: "per our regional meeting policy", "follow our amortization rules"
REF = re.compile(rf"\b{CUES}\s+((?:[\w'’\-]+\s+){{0,6}}?{NOUNS})\b", re.IGNORECASE)
IDENT = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"
)  # any snake_case identifier (a sheet, file or record id)
STOP = {
    "our",
    "the",
    "any",
    "all",
    "from",
    "with",
    "this",
    "that",
    "your",
    "these",
    "those",
    "team",
    "per",
    "follow",
    "following",
    "according",
    "using",
    "apply",
    "see",
    "separate",
    "sheet",
    "worksheet",
    "applicable",
    "relevant",
    "standard",
    "current",
    "latest",
    "new",
    "recent",
    "some",
    "there",
    "have",
    "been",
    "changes",
    "data",
}
NOUN_RE = re.compile(rf"\b{NOUNS}\b", re.IGNORECASE)
RULE_LIKE = re.compile(
    r"\b(?:must|only|do not|don't|never|required|require[sd]?|exclude[sd]?|at least|no more than|"
    r"at most|unless|within \d+|threshold|if\b[^.\n]{0,80}\bthen)\b|[≥≤]|>=|<=",
    re.IGNORECASE,
)
EMPTY = re.compile(
    r'"error"|not found|no (?:results|matches|files|rows)|"count":\s*"?0\b|"result_count":\s*0\b|'
    r'"rows":\s*\[\]|"files":\s*"\[\]"',
    re.IGNORECASE,
)


@dataclass
class Reference:
    text: str
    origin: str  # "the request" or the tool whose output named it
    keys: List[str]
    status: str = "unread"  # unread | read
    read_by: Optional[str] = None
    empty_searches: List[str] = field(default_factory=list)

    def named_in(self, text: str) -> bool:
        """The source is named: its identifier, or its key words close to a policy word ("Regional Meeting Policy")."""
        low = text.lower()
        idents = [k.lower() for k in self.keys if IDENT.fullmatch(k)]
        if idents:
            return any(k in low for k in idents)
        need = max(1, round(0.6 * len(self.keys)))
        for m in NOUN_RE.finditer(low):
            window = low[max(0, m.start() - 80) : m.end() + 80]
            if sum(1 for k in self.keys if k in window) >= need:
                return True
        return False

    def read_in(self, text: str) -> bool:
        """Read = the output names the source and holds its rules (a file list or a title alone is not a read)."""
        return self.named_in(text) and bool(RULE_LIKE.search(text))


def _keys(phrase: str) -> List[str]:
    idents = IDENT.findall(phrase)
    if idents:
        return idents
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", phrase)]
    nouns = re.compile(rf"^{NOUNS}$", re.IGNORECASE)
    return [w for w in dict.fromkeys(words) if w not in STOP and not nouns.match(w)]


def find_references(text: str) -> List[str]:
    out = []
    for m in REF.finditer(text or ""):
        phrase = m.group(0).strip()
        if _keys(phrase):
            out.append(phrase)
    return out


class ReferenceLedger:
    def __init__(self):
        self.refs: List[Reference] = []

    def add_from(self, text: str, origin: str) -> List[Reference]:
        new = []
        for phrase in find_references(text):
            keys = _keys(phrase)
            if any(set(keys) == set(r.keys) for r in self.refs):
                continue
            ref = Reference(text=phrase, origin=origin, keys=keys)
            self.refs.append(ref)
            new.append(ref)
        return new

    def observe(self, tool: str, args: Dict[str, Any], result: Any, is_write: bool) -> None:
        """A tool output reads a referred source when it holds the source's key words; an empty or failed search
        whose arguments name the source is kept as a dead end. A write never reads a source."""
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        arg_text = json.dumps(args, default=str)
        empty = bool(EMPTY.search(text[:400]))
        found_in_data = [] if is_write else self.add_from(_text_values(result), tool)
        for (
            ref
        ) in found_in_data:  # the output that names a source may also hold it (rows of several worksheets)
            if ref.read_in(re.sub(re.escape(ref.text), " ", text, flags=re.IGNORECASE)):
                ref.status, ref.read_by = "read", tool
        for ref in self.refs:
            if ref.status == "read" or ref in found_in_data:
                continue
            if empty and not is_write and ref.named_in(arg_text + " " + _arg_nouns(args)):
                ref.empty_searches.append(f"{tool}({_short(args)})")
            elif not is_write and not empty and ref.read_in(text):
                ref.status, ref.read_by = "read", tool

    def unread(self) -> List[Reference]:
        return [r for r in self.refs if r.status != "read"]

    def summary(self, max_lines: int = 8) -> str:
        lines = []
        for r in self.refs[:max_lines]:
            state = f"read ({r.read_by})" if r.status == "read" else "NOT READ"
            dead = (
                f"; searches that found nothing: {', '.join(r.empty_searches[-2:])}"
                if r.empty_searches
                else ""
            )
            lines.append(f"- '{r.text}' (from {r.origin}): {state}{dead}")
        return "\n".join(lines)


def _text_values(result: Any) -> str:
    """Free-text values in a tool output (notes, descriptions, bodies), where references to other sources appear."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return result
    out: List[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str) and len(v.split()) >= 3:
            out.append(v)

    walk(result)
    return "\n".join(out)


def _arg_nouns(args: Dict[str, Any]) -> str:
    """A search argument rarely repeats the policy word ("title='Amortization Rules'" does; "query='amortization'"
    does not), so a search counts as one for the source when its key words appear in the arguments."""
    return "policy rules guidelines" if args else ""


def _short(args: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={str(v)[:40]!r}" for k, v in list(args.items())[:3] if v not in (None, "", []))
