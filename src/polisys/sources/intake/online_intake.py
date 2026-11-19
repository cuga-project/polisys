"""``OnlinePolicyIntake``: the settings of the intake and the state of one run. ``session_start`` lists that state, with a
comment on each field; the parts of the class share it. The parts are in ``found_rules``, ``discovery``, ``write_checks``
and ``reply``."""

from __future__ import annotations

import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from polisys.hooks import AgentHooks, HookResult
from polisys.checks.facts import FACTS_CHARS, FactStore
from polisys.package.model import PolicySet
from polisys.checks.decisions import Decisions
from polisys.experimental.checks.guard_code import GuardCode
from polisys.sources.references import ReferenceLedger
from polisys.runtime.trace import fresh
from polisys.checks.rules.rule_set_hooks import RuleSetHooks
from polisys.tool_kind import is_write_tool
from polisys.sources.intake.found_rules import _FoundRules
from polisys.sources.intake.discovery import _Discovery
from polisys.sources.intake.write_checks import _WriteChecks
from polisys.sources.intake.reply import _Reply


class OnlinePolicyIntake(_FoundRules, _Discovery, _WriteChecks, _Reply, AgentHooks):
    def __init__(
        self,
        model: Callable[[str, str], str],
        wait_s: float = 45.0,
        verify_writes: bool = True,
        check_reply: bool = True,
        is_write: Callable[[str], bool] = is_write_tool,
        detector: Optional[Callable[[str, Sequence[str]], List[float]]] = None,
        detect_threshold: float = 0.47,
        tool_specs: Optional[List[Dict[str, Any]]] = None,
        method_choice: Optional[Dict[str, Any]] = None,
        read_before_write: Any = True,
        verifier_may_deny: bool = False,
        guard_code: str = "off",
        brief: bool = True,
        note_channel: str = "tool",
        today: Optional[str] = None,
        track_supersedes: bool = False,
        discover: Optional[List[Any]] = None,
        listing: Optional[Callable[[], Dict[str, List[Any]]]] = None,
        listing_data: bool = False,
        decisions: Optional[Decisions] = None,
        model_effect: str = "remind",
        cap_effect: str = "allow",
        record_checks: bool = False,
        retry_effect: str = "allow",
        voice: str = "order",
        evidence: str = "off",
    ):
        """``tool_specs``: when given, trusted policies are also compiled (code for rule tables, the model for prose)
        and enforced by code; statements that do not compile stay with the verifier. ``method_choice``: as in
        ``RuleSetHooks``. ``detect_threshold`` 0.47 is the Kev-4B threshold from the detector probe."""
        self.model = model
        self.tool_specs = tool_specs
        self.method_choice = method_choice
        self.detector = detector
        self.detect_threshold = detect_threshold
        self.wait_s = wait_s
        self.verify_writes = verify_writes
        self.check_reply = check_reply
        self.note_channel = note_channel  # see RuleSetHooks
        # True / "remind": remind while a referred policy source is unread. "block_once": also stop the first write once
        # (the agent reads, then retries); later writes only get the reminder. False / "off": no check.
        self.read_before_write = "remind" if read_before_write is True else (read_before_write or "off")
        self.verifier_may_deny = verifier_may_deny  # model verdicts only remind unless measured otherwise
        self.brief = brief  # list the found policies in the system prompt at each model call
        self.guard_mode = guard_code  # off | remind | deny: generated code checks per (statement, write tool)
        # Replacement tracking: a rule that a later trusted record replaces leaves the current list (brief, verifier,
        # reply check, code rules), whichever of the two records was read first.
        self.track_supersedes = track_supersedes
        # (tool name, search(query) -> result) pairs run once per thread at the request (see DISCOVERY_QUERIES)
        self.discover = list(discover or [])
        # () -> {source: [record, ...]}: every record of every app, read once per thread at the request (sources/listing.py)
        self.listing = listing
        self.listing_data = listing_data  # the listing also hands corrections and examples to the agent
        # Narrow decision-model questions (checks/decisions.py): rule changes without policy words, record exclusions and
        # limits before writes, and whether a code rule applies before it denies.
        self.decisions = decisions
        # What a model verdict does: "remind" = a note after the write (it runs); "block_once" = the write is
        # stopped once with the reason, and the same call repeated goes through.
        self.model_effect = model_effect
        self.voice, self.evidence = voice, evidence  # finish/followups.py
        # After the repeated-denial cap: "allow" lets the write through; "skip" keeps it blocked and tells the agent
        # to continue without it.
        self.cap_effect = cap_effect
        # record_checks: decision questions about the record itself (status, notes, flags) and about exceptions
        # whose facts were not read; the fact store also finds records by short text values of the call.
        self.record_checks = record_checks
        # A call stopped once and repeated unchanged: "allow" lets it through; "hold" checks it again and keeps it
        # stopped (the agent is told to skip it) when the same check flags it again.
        self.retry_effect = retry_effect
        self.today = today  # the run's date (benchmarks: the task's world time); else the request's "Today is", else now
        self.guard = GuardCode(model)
        self.is_write = is_write
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.session_start(None)

    def session_start(self, thread_id: Optional[str]) -> HookResult:
        self.request = ""
        self.seen: set = set()
        self.flagged_ids: set = set()
        self.pending: List[Future] = []
        self.policies: List[Dict[str, Any]] = []
        self.store = FactStore(match_phrases=self.record_checks)  # tool output seen in this thread (checks/facts.py)
        self.writes: List[str] = []
        self.own_texts: List[str] = []
        self.data_notes: List[Tuple[str, str, str]] = []  # (label, source, record text) from the listing  # text the agent wrote in this run (long string arguments of its writes)
        self.records: List[Dict[str, Any]] = fresh(getattr(self, "records", None))
        self._denials: Dict[str, int] = {}
        self._reply_checked = False
        self.overlay = PolicySet()  # online policies in the policy model (one document per record)
        self.notes: List[str] = []  # reminders for the next model call (a reminder never blocks the write)
        self._noted_statements: set = set()  # verifier notes are shown once per statement
        self.refs = (
            ReferenceLedger()
        )  # policy sources the request or data refers to, and whether one was read
        self.observations: List[Dict[str, Any]] = []  # read outputs as records, for generated checks
        self.called: List[str] = []  # tool names in call order (for the trajectory summary in reminders)
        self._ref_noted: set = set()  # unread sources already named in a reminder
        self.compiled: set = set()  # statement texts enforced by code rules
        self.superseded: Dict[str, str] = {}  # replaced statement text -> what replaced it
        self._discovered = False
        self._listed = False
        self.listed: Any = None  # listing.Listed: what the listing read; later reads of it skip the online intake
        self._blocked_once: set = set()  # calls already stopped once (model verdicts and decisions)
        self.enforcer = (
            RuleSetHooks(
                [],
                self.tool_specs,
                method_choice=self.method_choice,
                scope_check=(self.decisions.rule_applies if self.decisions is not None else None),
                cap_effect=self.cap_effect,
                note_channel=self.note_channel,
            )
            if self.tool_specs is not None
            else None
        )
        return HookResult()

    def _facts_for(self, args: Dict[str, Any]) -> str:
        """Records that mention identifiers from the call (innermost record per match), then the most recent data."""
        return self.store.for_call(args, FACTS_CHARS)

    @property
    def facts(self) -> str:
        return self.store.text

    @facts.setter
    def facts(self, value: str) -> None:
        self.store.text = value

    def _task_date(self) -> str:
        """The task's own date: the run's world time first, then "today is YYYY-MM-DD" in the request; "" when the
        task gives none. Model checks get only this, never the clock: a world with its own dates and the real date
        disagree, and a wrong date in the prompt is worse than none."""
        if self.today:
            return str(self.today)[:10]
        m = re.search(r"\btoday is (\d{4}-\d{2}-\d{2})", self.request or "", re.IGNORECASE)
        return m.group(1) if m else ""
