"""Work method (``notes.work_method``): general rules for doing SOP work on many records, given to the agent in the system
prompt, and an optional read-back turn before the final answer.

Each rule holds for any enterprise task that follows a written procedure over a set of records (lines, claims, tickets,
people). None names a domain, a system or a value. A profile lists the rules it wants, so a run can measure each one.

| Name | Rule |
|---|---|
| ``worksheet`` | one row per record before any write: the decision inputs with their source, the decision, the outputs it needs |
| ``actors`` | a rule that needs a role or an authority is met only by a person whose role is confirmed, never by the acting user |
| ``request_scope`` | a request to follow a procedure authorizes its internal writes; names the request gives win over defaults |
| ``cells`` | a flag column holds one value of its set; reasons and source values go to a notes column |
| ``idempotent`` | look for an existing record before creating one; resolve an account once after an account error |
| ``readback`` | before the final answer, one turn that reads back each record's required outputs from the systems |
| ``sop_literal`` | templates, log fields, names, formats and locations come word for word from the SOP document |
| ``readback_sop`` | ``readback``, but against the SOP: the procedures the agent opened, each record's branch and its outputs |
| ``lookup_retry`` | after a failed lookup of a name the request gives, a note in that tool result: search again before giving up |
| ``finish_check`` | before the final answer, one turn: each asked item not done needs a quoted policy line that forbids it |

``readback``, ``readback_sop`` and ``finish_check`` each send the agent back once before the reply; a profile lists one of them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from polisys.hooks import AgentHooks, HookResult

METHODS: Dict[str, str] = {
    "worksheet": (
        "Before any write, build a worksheet with one row per record the procedure covers (each line, item, claim, "
        "ticket or person). In each row write: the inputs every SOP decision needs, each with where you read it; the "
        "decision the SOP gives for those inputs; and the outputs the SOP requires for that decision (records, comments, "
        "files, messages). Compare amounts, codes, dates and names in code, not by eye: two values are equal only if "
        "they are exactly equal. Then make the writes from the rows, one row at a time, and mark each output done."
    ),
    "actors": (
        "When an SOP rule needs a role or an authority (a manager, an approver, a reviewer, two different people), look "
        "up each person in the evidence in the directory or user profiles and note their name and title. The account you "
        "act as, and the person who prepared the work, never meet such a rule. If the role cannot be confirmed, take the "
        "SOP's path for a missing or unclear approval."
    ),
    "request_scope": (
        "A request to do work under an SOP authorizes the internal writes of that procedure (records, comments, files, "
        "internal tickets): make them, and do not end with an offer to make them. Names the request gives (file names, "
        "column names, field labels) win over the SOP's defaults; use the SOP's names only where the request gives none."
    ),
    "cells": (
        "A flag or status column holds exactly one value from its set (for example Y or N). Put reasons, both source "
        "values and references in a notes column, never in the flag column."
    ),
    "idempotent": (
        "Before you create a record (a ticket, a comment, a file), check whether one already exists for the same item. "
        "After an error that a user or account is unknown, look the account up once and reuse it for every later write."
    ),
    "sop_literal": (
        "The procedure tools summarize the SOP. For templates, log and audit fields, names, formats and where files and "
        "records are kept, use the SOP document's own text word for word; where the summary and the document differ, the "
        "document wins. Save each file and record where the request or the SOP keeps it, not in a working folder."
    ),
}

READBACK = (
    "Before the final answer, check your work against the systems, not against your notes. For each record the "
    "procedure covered, read back from the systems each output the SOP requires for its decision (the record, comment, "
    "ticket, file row or message), and check that it exists exactly once, on the right record, with the right values. "
    "Create what is missing and correct what is wrong. Then give the final answer."
)

READBACK_SOP = (
    "Before the final answer, check your work against the SOP, not against your own plan. For each record you handled, "
    "take the branch the SOP gives for the decision you made, and check in the systems that every output of that branch "
    "exists exactly once, in the place the request or the SOP names (not in a working folder), with the SOP's exact "
    "template text and fields. Correct or complete only the outputs of decisions you already made; do not send new "
    "messages to people as corrections unless a step of that branch requires them. Then give the final answer."
)
SOP_CHARS = 12000  # the procedures shown with READBACK_SOP

# Sent once before the reply (``finish_check``). A stop must rest on the policy, not on the agent's own caution.
FINISH_CHECK = (
    "Before the final answer, compare the request with what you did. List each thing the request asked for. For each one "
    "that you did not do, or did only in part, quote the line of the policy that forbids it for that operation. Your own "
    "caution, a search that found nothing at the first try, input that looks unusual, or a condition that one of your "
    "tools can make true is not such a line. Where you have no such line, do the work now with your tools. Then give the "
    "final answer."
)

# Added to the result of a failed lookup (``lookup_retry``). {name}: the argument value that the request also holds.
LOOKUP_RETRY = (
    'The request names "{name}", and this call did not find it. A name in a request is often not the exact name in the '
    "system. Before you conclude that it does not exist, search again with a shorter or a partial form of the name, or "
    "with another search or list tool. Skip the work that depends on it only after those tries fail."
)
# Added to FINISH_CHECK with ``authority.value_lists: tool``.
FINISH_CHECK_TOOL_LISTS = (
    " A list of allowed values in the policy that differs from the tool's own list of allowed values is not such a line "
    "either: the tool's list is the one in force."
)
MIN_NAME_CHARS = 4

NAMES = (*METHODS, "readback", "readback_sop", "lookup_retry", "finish_check", "literal_values", "finish_ledger",
         "literal_coverage")
REPLY_TURNS = ("readback", "readback_sop", "finish_check", "finish_ledger")  # each sends the agent back once before the reply


def note(names: List[str]) -> str:
    """The system-prompt text of the listed rules (``readback`` acts at the reply, so it adds no text here)."""
    rules = [METHODS[n] for n in names if n in METHODS]
    if not rules:
        return ""
    return "How to do SOP work on many records:\n" + "\n".join(f"- {r}" for r in rules)


class ReadbackHooks(AgentHooks):
    """One read-back turn per run: the first final answer is sent back once with ``READBACK``. With a ``store``
    (``readback_sop``) the message is ``READBACK_SOP`` followed by the text of the procedures the agent opened."""

    def __init__(self, store: Any = None) -> None:
        self.store = store
        self.sent = False
        self.records: List[Dict[str, Any]] = []

    def _message(self) -> str:
        if self.store is None:
            return READBACK
        opened: List[str] = []
        for call in getattr(self.store, "calls", []) or []:
            opened += [i for i in call.get("expanded") or [] if i not in opened]
        text = "\n\n".join(self.store.procedure_text(self.store.procs[i]) for i in opened if i in self.store.procs)
        if not text:
            return READBACK_SOP
        return READBACK_SOP + "\n\nThe SOP procedures you opened:\n\n" + text[:SOP_CHARS]

    def session_start(self, thread_id: Optional[str]) -> None:
        self.sent = False

    def before_reply(self, answer: str) -> Optional[HookResult]:
        if self.sent:
            return None
        self.sent = True
        message = self._message()
        self.records.append({"component": "work_method", "outcome": "readback_sent", "chars": len(message)})
        return HookResult(message=message)


class FinishCheckHooks(AgentHooks):
    """One turn before the reply: the first final answer is sent back once with ``FINISH_CHECK``."""

    def __init__(self, value_lists: str = "policy") -> None:
        self.sent = False
        self.message = FINISH_CHECK + (FINISH_CHECK_TOOL_LISTS if value_lists == "tool" else "")
        self.records: List[Dict[str, Any]] = []

    def session_start(self, thread_id: Optional[str]) -> None:
        self.sent = False

    def before_reply(self, answer: str) -> Optional[HookResult]:
        if self.sent:
            return None
        self.sent = True
        self.records.append({"component": "work_method", "outcome": "finish_check_sent", "chars": len(self.message)})
        return HookResult(message=self.message)


def named_in(value: Any, request: str) -> bool:
    """Is this argument value a name the request gives: text of ``MIN_NAME_CHARS`` or more, found as whole words?"""
    if not isinstance(value, str):
        return False
    value = " ".join(value.split())
    if len(value) < MIN_NAME_CHARS:
        return False
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", " ".join((request or "").split()), re.I) is not None


def failed(result: Any) -> bool:
    return isinstance(result, dict) and "error" in result


class LookupRetryHooks(AgentHooks):
    """After a failed call of a read tool whose argument is a name the request gives, a note in that tool result
    (``LOOKUP_RETRY``). At most ``max_notes`` per run, and one per name."""

    def __init__(self, request: str, is_write: Callable[[str], bool], max_notes: int = 3) -> None:
        self.request, self.is_write, self.max_notes = request or "", is_write, max_notes
        self.told: List[str] = []
        self.records: List[Dict[str, Any]] = []

    def session_start(self, thread_id: Optional[str]) -> None:
        self.told = []

    def on_user_message(self, message: str, thread_id: Optional[str]) -> None:
        if message and not self.request:
            self.request = message

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> Optional[HookResult]:
        if len(self.told) >= self.max_notes or not failed(result) or self.is_write(tool):
            return None
        names = [v for v in (args or {}).values() if named_in(v, self.request)]
        names = [" ".join(v.split()) for v in names if " ".join(v.split()).lower() not in self.told]
        if not names:
            return None
        self.told.append(names[0].lower())
        self.records.append({"component": "work_method", "outcome": "lookup_retry_note", "tool": tool})
        return HookResult(message=LOOKUP_RETRY.format(name=names[0]))



LITERAL = (
    "The request gives this value in quotes: {quoted!r}. This call passes {given!r} for {argument}. Pass the quoted text "
    "exactly as the request gives it, with no character added, removed or changed, then make the call again."
)
MIN_QUOTED_CHARS = 12
# Text between a pair of quotation marks that are not inside a word (so an apostrophe does not open a quote).
QUOTED = re.compile(r"""(?<![A-Za-z0-9])['"‘“]([^'"‘’“”\n]{12,400})['"’”](?![A-Za-z0-9])""")
EDGE = " \t.,;:!?"


def quoted_values(request: str) -> List[str]:
    """The texts the request gives between quotation marks, ``MIN_QUOTED_CHARS`` or longer."""
    return [m.group(1) for m in QUOTED.finditer(request or "") if len(m.group(1).strip()) >= MIN_QUOTED_CHARS]


def _loose(text: str) -> str:
    """The text without spaces and punctuation at its two ends, and with one space between words. Case is kept."""
    return " ".join(text.split()).strip(EDGE)


def altered_quote(value: Any, quoted: List[str]) -> Optional[str]:
    """The quoted text of the request that this value is a changed copy of: equal apart from spaces and punctuation at
    the ends, and not equal as written. ``None`` when the value is an exact copy, or is no copy of a quoted text."""
    if not isinstance(value, str) or value in quoted:
        return None
    loose = _loose(value)
    return next((q for q in quoted if _loose(q) == loose), None)


def call_key(tool: str, args: Any) -> str:
    """One call: the tool and its arguments. A block that is given once per call uses it."""
    return tool + json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)


class LiteralValueHooks(AgentHooks):
    """A write call that passes a changed copy of a text the request gives in quotes is blocked (``LITERAL``): the quoted
    text is passed as written. Decided by code. Each call is blocked once: the same call made again goes through, and
    two calls of one turn that carry the same changed text are both blocked. At most ``max_blocks`` per run."""

    def __init__(self, request: str, is_write: Callable[[str], bool], max_blocks: int = 8) -> None:
        self.request, self.is_write, self.max_blocks = request or "", is_write, max_blocks
        self.told: List[str] = []
        self.records: List[Dict[str, Any]] = []

    def session_start(self, thread_id: Optional[str]) -> None:
        self.told = []

    def on_user_message(self, message: str, thread_id: Optional[str]) -> None:
        if message and not self.request:
            self.request = message

    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> Optional[HookResult]:
        if len(self.told) >= self.max_blocks or not self.is_write(tool):
            return None
        quoted = quoted_values(self.request)
        key = call_key(tool, args)
        for argument, value in (args or {}).items():
            source = altered_quote(value, quoted)
            if source is None or key in self.told:
                continue
            self.told.append(key)
            self.records.append({"component": "work_method", "outcome": "literal_value_block", "tool": tool, "argument": argument})
            return HookResult(message=LITERAL.format(quoted=source, given=value, argument=argument))
        return None


LEDGER_HEAD = "Record of this run, taken from the tool calls (not from memory). Write calls:"
LEDGER_LINES = 30


class FinishLedgerHooks(FinishCheckHooks):
    """The finish check with a record of the run added by code: each write call with its arguments and whether it
    succeeded, and the number of read calls. The agent compares the request with what was really done."""

    def __init__(self, value_lists: str, is_write: Callable[[str], bool]) -> None:
        super().__init__(value_lists)
        self.is_write = is_write
        self.calls: List[tuple] = []

    def session_start(self, thread_id: Optional[str]) -> None:
        super().session_start(thread_id)
        self.calls = []

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> Optional[HookResult]:
        self.calls.append((tool, args, result))
        return None

    def ledger(self) -> str:
        writes = [(t, a, r) for t, a, r in self.calls if self.is_write(t)]
        reads = [(t, a, r) for t, a, r in self.calls if not self.is_write(t)]
        lines = [LEDGER_HEAD]
        for tool, args, result in writes[-LEDGER_LINES:]:
            shown = json.dumps(args or {}, ensure_ascii=False, default=str)
            shown = shown if len(shown) <= 300 else shown[:300] + " ..."
            if failed(result):
                state = "FAILED: " + " ".join(str(result.get("error")).split())[:160]
            else:
                state = "ok"
            lines.append(f"- {tool}({shown}) -> {state}")
        if not writes:
            lines.append("- (no write call was made)")
        lines.append(f"Read calls: {len(reads)}, of which {sum(1 for _, _, r in reads if failed(r))} failed.")
        return "\n".join(lines)

    def before_reply(self, answer: str) -> Optional[HookResult]:
        if self.sent:
            return None
        self.sent = True
        message = self.message + "\n\n" + self.ledger()
        self.records.append({"component": "work_method", "outcome": "finish_ledger_sent", "chars": len(message)})
        return HookResult(message=message)


EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# A record code: capitals and digits, with at least one of each, of 5 characters or more (parts joined by - or _).
CODE = re.compile(r"(?<![\w-])(?=[A-Z0-9_-]*[0-9])(?=[A-Z0-9_-]*[A-Z])[A-Z0-9]+(?:[-_][A-Z0-9]+)*(?![\w-])")
COVERAGE = (
    "These values of the request appear in no tool call and in no tool result of this run: {values}. If the request asks "
    "for work on one of them, that work is not done: do it now with your tools. If the request only mentions it, go on."
)


def request_literals(request: str) -> List[str]:
    """Exact values the request gives: e-mail addresses, record codes and texts in quotation marks."""
    text = request or ""
    found = EMAIL.findall(text) + [c for c in CODE.findall(text) if len(c) >= 5] + quoted_values(text)
    out: List[str] = []
    for value in found:
        value = " ".join(value.split()).strip(EDGE)
        if value and value.lower() not in [o.lower() for o in out]:
            out.append(value)
    return out


class LiteralCoverageHooks(AgentHooks):
    """At the first final answer, the exact values of the request (e-mail addresses, record codes, quoted texts) that
    appear in no tool call and no tool result are named once (``COVERAGE``). Decided by code."""

    def __init__(self, request: str) -> None:
        self.request = request or ""
        self.seen = ""
        self.sent = False
        self.records: List[Dict[str, Any]] = []

    def session_start(self, thread_id: Optional[str]) -> None:
        self.seen, self.sent = "", False

    def on_user_message(self, message: str, thread_id: Optional[str]) -> None:
        if message and not self.request:
            self.request = message

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> Optional[HookResult]:
        blob = json.dumps([args, result], ensure_ascii=False, default=str)
        self.seen += " " + " ".join(blob.lower().split())
        return None

    def before_reply(self, answer: str) -> Optional[HookResult]:
        if self.sent:
            return None
        self.sent = True
        missing = [v for v in request_literals(self.request) if " ".join(v.lower().split()) not in self.seen]
        self.records.append({"component": "work_method", "outcome": "literal_coverage", "missing": len(missing)})
        if not missing:
            return None
        return HookResult(message=COVERAGE.format(values="; ".join(repr(v) for v in missing[:8])))
