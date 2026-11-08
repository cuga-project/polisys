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
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .hooks import AgentHooks, HookResult

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

NAMES = (*METHODS, "readback", "readback_sop")


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
