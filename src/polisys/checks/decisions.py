"""Narrow decisions before a write, asked to a decision model (e.g. Kev-4B) as typed yes/no questions.

The LLM verifier judges a whole write against all rules; it missed exceptions that depend on one record's data,
count limits over earlier writes, and rules applied outside their scope (2026-11-03 trajectories). These questions
are narrow, have one state each, and cost about a second and a fraction of a cent:

- ``record_and_limit``: do the rules exclude this record or recipient from this action; would it break a limit?
- ``rule_changes``: does a record change, add an exception to, or limit one of the known rules?
- ``rule_applies``: does a rule apply to this action at all (asked before a code rule denies)?

``ask(state, {name: instructions}) -> {name: p_yes or None}`` (see ``detectors.systemone_questions``).
Nothing here names a benchmark, an app or a field.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

from .llm import today_line

LIMIT_WORDS = re.compile(
    r"\b(max(imum)?|at most|no more than|up to|limit(ed)?|only (one|two|three|\d+)|\d+\s*(per|a|each)\s+\w+|"
    r"once (per|a)|cap(ped)?)\b",
    re.I,
)

EXCLUDE_Q = (
    "Do the company rules, including their exceptions and later changes, say that the planned action must not be "
    "done for this record, person or target? Answer yes only if the rules and the data clearly say so."
)
LIMIT_Q = (
    "Would the planned action break a limit in the rules on how many such actions are allowed (per record, "
    "recipient, channel or period), counting the earlier actions listed?"
)
CHANGES_Q = (
    "Does this record change, replace, add an exception to, or set a limit on one of the listed rules, coming from "
    "someone in the company who may do so? Ordinary data and messages that do not change a rule are not."
)
APPLIES_Q = (
    "Does this rule apply to the planned action, given what the action does and the record it is about?"
)
RECORD_SKIP_Q = (
    "Does the data about this record itself (its status, notes, flags or labels, or markers such as test, sample, "
    "duplicate, on hold, revisit later or do not contact) say that the planned action should not be done for it now?"
)
NEEDS_DATA_Q = (
    "Does one of the rules have an exception or condition for this action (for example an approval, a request, a "
    "leave, an exemption or a status) whose facts for this record are not in the data shown, so that it cannot be "
    "decided yet? Answer yes only if such a fact is needed and missing."
)


class Decisions:
    def __init__(
        self, ask: Callable[[str, Dict[str, str]], Dict[str, Optional[float]]], threshold: float = 0.6
    ):
        self.ask = ask
        self.threshold = threshold

    @staticmethod
    def _cut(text: str, n: int) -> str:
        return text if len(text) <= n else text[: n - 20] + "\n...(cut)"

    def record_and_limit(
        self,
        rules: str,
        record_data: str,
        earlier: str,
        action: str,
        with_limit: bool,
        record_checks: bool = False,
        today: str = "",
    ) -> Dict[str, Optional[float]]:
        state = today_line(today) + (
            f"Company rules:\n{self._cut(rules, 5000)}\n\nData about the record or target of the action:\n"
            f"{self._cut(record_data, 3000)}\n\nEarlier actions of the same kind in this task:\n"
            f"{self._cut(earlier, 1500) or '(none)'}\n\nPlanned action: {self._cut(action, 1500)}"
        )
        qs: Dict[str, str] = {}
        if rules.strip():
            qs["excluded"] = EXCLUDE_Q
        if with_limit:
            qs["over_limit"] = LIMIT_Q
        if record_checks:  # signals in the record itself, and exceptions whose facts were not read
            qs["record_says_skip"] = RECORD_SKIP_Q
            if rules.strip():
                qs["needs_data"] = NEEDS_DATA_Q
        return self.ask(state, qs) if qs else {}

    def rule_changes(self, rules: str, record: str) -> Optional[float]:
        state = f"Rules found so far:\n{self._cut(rules, 4000)}\n\nRecord:\n{self._cut(record, 3000)}"
        return self.ask(state, {"changes": CHANGES_Q}).get("changes")

    def rule_applies(self, rule: str, action: str) -> Optional[float]:
        state = f"Rule: {self._cut(rule, 1500)}\n\nPlanned action: {self._cut(action, 1500)}"
        return self.ask(state, {"applies": APPLIES_Q}).get("applies")

    def yes(self, p: Optional[float]) -> bool:
        return p is not None and p >= self.threshold


def has_limit(text: str) -> bool:
    return bool(LIMIT_WORDS.search(text or ""))


def action_text(tool: str, args: Dict[str, Any]) -> str:
    import json

    shown = {k: v for k, v in (args or {}).items() if v not in (None, "", [], {})}
    return f"{tool}({json.dumps(shown, default=str)})"
