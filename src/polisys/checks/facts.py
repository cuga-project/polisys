"""What the agent has seen during the run, kept for the checks.

``call_tokens`` finds the identifiers in a call's arguments (emails; words with a letter and a digit or underscore,
such as ``FE-103``, ``INC_030``, ``@frustrated_ops``). Checks use them to pick the records that the call names.
"""

from __future__ import annotations

import re
from typing import Any, List

# Identifiers in call arguments: emails, and words of 3+ characters with a letter and a digit or underscore
# (FE-103, INC_030, @frustrated_ops). Plain numbers and dates are not selective, so they are left out.
TOKEN = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|@?(?<![\w-])(?=[\w-]*[A-Za-z])(?=[\w-]*[\d_])[A-Za-z0-9][\w-]{2,}"
)


PHRASE_MIN, PHRASE_MAX = (
    8,
    80,
)  # a text argument this long is matched as a whole phrase (subjects, names, titles)


def call_phrases(args: Any) -> List[str]:
    """Short text values of a call (a subject, a title, a name) that identify a record by content, lowercased.

    A create or copy call often carries no identifier of the record it comes from; its subject or name does. Long
    texts (message bodies) and plain numbers or dates are skipped.
    """
    out: List[str] = []
    stack = [args]
    while stack:
        x = stack.pop(0)
        if isinstance(x, dict):
            stack[:0] = list(x.values())
        elif isinstance(x, list):
            stack[:0] = list(x)
        elif isinstance(x, str):
            v = " ".join(x.split()).lower()
            if PHRASE_MIN <= len(v) <= PHRASE_MAX and re.search(r"[a-z]{3}", v) and "\n" not in x:
                out.append(v)
    return list(dict.fromkeys(out))


def call_tokens(args: Any) -> List[str]:
    """Identifiers in a call's arguments (nested values included), in order, without repeats."""
    out: List[str] = []
    stack = [args]
    while stack:
        x = stack.pop(0)
        if isinstance(x, dict):
            stack[:0] = list(x.values())
        elif isinstance(x, list):
            stack[:0] = list(x)
        elif isinstance(x, str):
            out.extend(t.rstrip(".") for t in TOKEN.findall(x))
    return list(dict.fromkeys(out))


# -- the fact store ---------------------------------------------------------------------------------------------------

import json  # noqa: E402
from typing import Dict, Iterator  # noqa: E402

FACTS_CHARS = 8000  # data in a verifier prompt
MATCH_CHARS = 5000  # of which: records that mention an identifier from the call
KEEP_FACTS_CHARS = 60000  # recent tool output kept per thread
MAX_MATCHES_PER_TOKEN = 20  # an identifier found in more records than this is not selective


def _scalars(node: Dict[str, Any]) -> str:
    return " ".join(
        str(v) for v in node.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)
    )


def dicts(x: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from dicts(v)
    elif isinstance(x, list):
        for v in x:
            yield from dicts(v)


class FactStore:
    """Tool output seen in one thread: the recent text, and every record (dict) with its scalar values.

    ``for_call(args)`` gives a check the records that the call names first (identifiers from its arguments), then the
    most recent data, within a character budget. This is what keeps a check from judging a write without the
    record it is about (the 2026-11-01 false-deny fix).
    """

    def __init__(self, match_phrases: bool = False) -> None:
        self.text = ""
        self.index: List[Dict[str, Any]] = []
        self.match_phrases = (
            match_phrases  # also find records by short text values of the call (call_phrases)
        )

    def add(self, tool: str, result: Any) -> None:
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                pass
        self.text = (self.text + f"\n{tool}: {json.dumps(result, default=str)}")[-KEEP_FACTS_CHARS:]
        for node in dicts(result):
            scalars = _scalars(node)
            if scalars:
                self.index.append(
                    {"scalars": scalars, "text": json.dumps(node, default=str)[:1500], "node": node}
                )

    def records_for(self, args: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tok in call_tokens(args):
            hits = [r for r in self.index if tok in r["scalars"]]
            if 0 < len(hits) <= MAX_MATCHES_PER_TOKEN:
                out.extend(r for r in hits if r not in out)
        if self.match_phrases:
            for ph in call_phrases(args):
                hits = [r for r in self.index if ph in r["scalars"].lower()]
                if 0 < len(hits) <= MAX_MATCHES_PER_TOKEN:
                    out.extend(r for r in hits if r not in out)
        return out

    def recent(self, chars: int) -> str:
        return self.text[-chars:]

    def for_call(self, args: Any, budget: int = FACTS_CHARS) -> str:
        matched: List[str] = []
        for r in self.records_for(args):
            if r["text"] not in matched:
                matched.append(r["text"])
        head = "\n".join(matched)[: min(MATCH_CHARS, budget)]
        tail = self.text[-(budget - len(head)) :] if budget > len(head) else ""
        return (
            f"Records that mention values in this call:\n{head}\n\nMost recent data:\n{tail}"
            if head
            else tail
        )
