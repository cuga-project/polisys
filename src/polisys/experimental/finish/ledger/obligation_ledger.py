"""``ObligationLedger``: the hooks that hold the items of one run, settle them at the final reply and send the follow-up."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from polisys.hooks import AgentHooks, HookResult
from polisys.package.model import Obligation
from polisys.values import _when
from polisys.tool_kind import is_write_tool
from polisys.tool_kind import ID_ARG
from polisys.runtime.trace import HOOK_FAILED, audited_failure, fresh
from polisys.experimental.finish.ledger.constants import (
    CODE_ARGS, CODE_TOOL, CONVERSATION, DATA_CHARS, FILE_TOOL, FOLLOW_UPS_BY_LEVEL, HELD, IDENTITY_CHARS, MET,
    MIN_KEY_CHARS, NOTE_PREFIX, NOT_APPLICABLE, ONE_REQUEST, OPEN, POLICY_TOOL_PREFIX, READ_CHARS, SEND_BACKS, SOP_LOOP,
    UNBOUND, UNKNOWN, _is_free_text,
)
from polisys.experimental.finish.ledger.text import (
    _as_text, _failed_call, _names, _one_line, _readable_call, _value_text, _without_secrets, fold,
)
from polisys.experimental.finish.ledger.value_patterns import GAP
from polisys.experimental.finish.ledger.cases import BindingError, Case, bind_cases, cards_of, known_from_loop
from polisys.experimental.finish.ledger.values import plan_of, resolve_values
from polisys.experimental.finish.ledger.call_check import Checked, FAIL, _schema_of, arg_outcome, check_calls
from polisys.experimental.finish.ledger.follow_up import Item, follow_up_text


class ObligationLedger(AgentHooks):
    """The items of one run. ``level``: 0 and 1 record only; 2 sends one follow-up; 3 up to two, and never more than
    ``max_follow_ups``. ``item_level``: the level of every item; ``None`` uses each obligation's own. Only ``follow_up`` is
    sent. ``handoff_tools`` and ``handoff_text`` are for ``run_type="conversation"``."""

    def __init__(
        self,
        obligations: Iterable[Obligation],
        model: Optional[Callable[[str, str], str]],
        *,
        cards: Optional[List[Dict[str, Any]]] = None,
        tool_specs: Sequence[Dict[str, Any]] = (),
        tools_in_run: Optional[Iterable[str]] = None,
        run_type: str = ONE_REQUEST,
        level: int = 2,
        max_follow_ups: int = 1,
        wrong_values: str = "known_origin_only",
        item_level: Optional[str] = None,
        request: str = "",
        store: Any = None,
        today: str = "",
        policy_text: str = "",
        handoff_tools: Iterable[str] = (),
        handoff_text: Iterable[str] = (),
        message_tool: str = "",
        is_write: Callable[[str], bool] = is_write_tool,
        request_producer: Optional[Callable[[str], Iterable[Obligation]]] = None,
    ):
        self._base = list(obligations)
        self.obligations = list(self._base)
        # the producer ``request`` (design section 5): the items of the request, made once at the user message
        self.request_producer = request_producer
        self.model = model
        self.cards = cards if cards is not None else cards_of(self.obligations)
        self.specs = {t["name"]: t for t in tool_specs}
        # A tool with no spec is not the agent's (a tool of the customer's device, say): it is never asked for.
        in_run = tools_in_run if tools_in_run is not None else (self.specs or None)
        self.tools_in_run = set(in_run) if in_run is not None else None
        self.run_type, self.level = run_type, level
        self.max_follow_ups = max_follow_ups
        self.wrong_values = wrong_values
        self.item_level = item_level
        self.given_request = request
        self.store, self.today = store, today
        self.policy_text = policy_text
        self.handoff_tools = sorted(set(handoff_tools))
        self.handoff_text = [m.strip().lower() for m in handoff_text if m.strip()]
        self.message_tool = message_tool
        self.is_write = is_write
        self.records: List[Dict[str, Any]] = []
        self._ignore_blocks = False  # set by the loop before its first turn, so a new session keeps it
        self.session_start(None)

    def session_start(self, thread_id: Optional[str]) -> HookResult:
        self.records = fresh(self.records)
        self.obligations = list(self._base)
        self._requested = False
        self.request = self.given_request
        self.reads: List[str] = []
        self.calls: List[Dict[str, Any]] = []
        self.blocks: List[Dict[str, Any]] = []
        # a block could not be recorded: an open item may be the work it stopped, so no follow-up in this run
        self._silent = False
        self.items: List[Item] = []
        self.follow_ups = 0
        self._read_chars = 0
        self._replies = 0
        self._bound = False
        self._failed_binding = False
        self._final = False  # the loop's finish turn
        self._known: Optional[List[Dict[str, Any]]] = None
        self._loop_log: Optional[Sequence[Dict[str, Any]]] = None
        self._handoff_blocked = False
        self._own_block = False
        self._opened: Set[Any] = set()
        return HookResult()

    # -- what the run does ---------------------------------------------------------------------------------------------

    def on_user_message(self, message: str, thread_id: Optional[str]) -> HookResult:
        self._replies = 0
        text = message or ""
        if self.run_type == CONVERSATION:
            if not text.startswith(NOTE_PREFIX):
                self.request = f"{self.request}\n{text}".strip()
        elif not self.request:
            self.request = text
        if self.request_producer is not None and not self._requested and not text.startswith(NOTE_PREFIX):
            self._requested = True
            try:
                made = list(self.request_producer(self.given_request or text))
            except Exception as exc:  # noqa: BLE001 - no checklist is better than a wrong one
                made = []
                self._record(
                    "on_user_message",
                    "",
                    "request",
                    "error",
                    f"no request item: {type(exc).__name__}: {exc}",
                    "model",
                )
            self.obligations += made
        for ob in self.obligations:
            if ob.origin == "request" and ob.id not in self._opened:
                self._opened.add(ob.id)
                self._add_item(ob, Case(name="", decision="proceed"))
        return HookResult()

    def tool_blocked(self, tool: str, args: Dict[str, Any], message: str) -> None:
        if self._own_block:
            self._own_block = False  # the ledger's own stop of a hand-off holds no work
            return
        if self._ignore_blocks:
            return
        try:
            self._keep_block(tool, args, message)
        except Exception as exc:  # noqa: BLE001 - see ``_silent``
            self._silent = True
            self._record(
                "tool_blocked",
                tool,
                HOOK_FAILED,
                "error",
                f"{type(exc).__name__}: {exc}: the block is not recorded, so this run sends no follow-up",
                "code",
            )

    def _keep_block(self, tool: str, args: Dict[str, Any], message: str) -> None:
        text, readable = _readable_call(args)
        if readable is not args:
            self._record(
                "tool_blocked",
                tool,
                HOOK_FAILED,
                "error",
                f"the text of the call could not be read in full: the block holds {sorted(map(str, readable))} and the "
                "other arguments by name",
                "code",
            )
        for block in self.blocks:
            if (
                block["tool"] == tool and block["text"] == text
            ):  # stopped again: what counts as later starts here
                block["at"], block["message"] = len(self.calls), str(message or "")
                return
        self.blocks.append(
            {
                "tool": tool,
                "args": readable,
                "text": text,
                "message": str(message or ""),
                "at": len(self.calls),
            }
        )

    @audited_failure
    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> HookResult:
        if _failed_call(result):
            return HookResult()
        self.calls.append({"tool": tool, "args": args})
        if tool.startswith(POLICY_TOOL_PREFIX):
            return HookResult()
        if not self.is_write(tool) or CODE_TOOL.search(tool) or tool == self.message_tool:
            self._keep_read(tool, args, result)
        self._open_duties(tool, args)
        return HookResult()

    def _keep_read(self, tool: str, args: Dict[str, Any], result: Any) -> None:
        text = result if isinstance(result, str) else _as_text(result)
        line = f"{tool}({_as_text(_without_secrets(args))[:200]}) -> {text[:READ_CHARS]}"
        if self._read_chars + len(line) <= DATA_CHARS:  # earliest first: listings, then documents and records
            self.reads.append(line)
            self._read_chars += len(line)

    def _open_duties(self, tool: str, args: Dict[str, Any]) -> None:
        for ob in self.obligations:
            if ob.origin != "duty":
                continue
            trigger = (ob.scope or {}).get("trigger") if isinstance(ob.scope, dict) else None
            tools = trigger.get("tools") if isinstance(trigger, dict) else None
            key = (ob.id, tool, _as_text(_identity(args)))
            if not (isinstance(tools, list) and tool in tools) or key in self._opened:
                continue
            if isinstance(trigger.get("when"), dict) and not _when(trigger["when"], args or {}):
                continue  # the trigger call does not carry the values the rule names
            self._opened.add(key)
            self._add_item(ob, Case(name=f"after {tool}", decision="proceed"), since=len(self.calls))

    # -- the final reply -----------------------------------------------------------------------------------------------

    def loop_phase(self, phase: str, loop: Any) -> None:
        """Called by the SOP loop. A stop in its explore phase says that writes wait, and holds no work. At ``finish`` the
        loop's log gives the cases."""
        self._ignore_blocks = phase == "explore"
        if phase == "finish":
            self._final = True
            self._loop_log = list(getattr(loop, "log", []) or [])

    @audited_failure
    def before_reply(self, answer: str) -> HookResult:
        slot = self._replies < SEND_BACKS
        self._replies += 1
        if self.run_type == CONVERSATION or (self.run_type == SOP_LOOP and not self._final):
            return HookResult()
        text = self._final_check("before_reply", "", answer, slot)
        return HookResult(message=text) if text else HookResult()

    @audited_failure
    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> HookResult:
        self._own_block = False
        if self.run_type != CONVERSATION or tool not in self.handoff_tools or self._handoff_blocked:
            return HookResult()
        text = self._final_check("before_tool_call", tool, None, True)
        if not text:
            return HookResult()
        self._handoff_blocked = True
        self._own_block = True
        return HookResult(message=text)

    def after_reply(
        self, answer: str, turns: List[Dict[str, Any]], live: List[Dict[str, Any]]
    ) -> Optional[str]:
        """The new prompt for the agent when a reply is a hand-off with open items, or ``None`` to send the reply."""
        low = (answer or "").lower()
        if self.run_type != CONVERSATION or not any(mark in low for mark in self.handoff_text):
            return None
        try:
            text = self._final_check("after_reply", "", answer, True)
        except Exception as exc:  # noqa: BLE001 - a failed check must not stop the conversation
            self._record("before_reply", "", "hook_failed", "error", f"{type(exc).__name__}: {exc}", "code")
            return None
        if not text:
            return None
        turns.extend(t for t in live if t.get("role") != "tool")
        return f"{NOTE_PREFIX} — your hand-off message was not sent]\n{text}"

    def _final_check(self, hook: str, tool: str, answer: Optional[str], slot: bool) -> Optional[str]:
        """Bind if that is not done yet, check the items by code, and give the follow-up, if one is due."""
        self._bind(
            hook
        )  # a failed binding gives no bound item; the items of the request and of duties need no binding
        self._refresh(answer, hook)
        ready = [i for i in self.items if i.status == OPEN and self._may_send(i)]
        if not ready:
            return None
        if self._silent:
            names = ", ".join(sorted({i.ob.id for i in ready}))
            why = "a blocked call could not be recorded, so an open item may be the work it stopped"
            self._record(hook, tool, "follow_up", "not_sent", f"{why}; open: {names}", "code")
            return None
        limit = min(self.max_follow_ups, FOLLOW_UPS_BY_LEVEL.get(self.level, 0))
        names = ", ".join(sorted({i.ob.id for i in ready}))
        if self.follow_ups >= limit or not slot:
            why = (
                f"the limit of {limit} follow-up(s) is reached"
                if self.follow_ups >= limit
                else "CUGA's send-backs for this request are used up"
            )
            self._record(hook, tool, "follow_up", "not_sent", f"{why}; open: {names}", "code")
            return None
        self.follow_ups += 1
        held = [i for i in self.items if i.status == HELD and self._may_send(i)]
        text = follow_up_text(ready, held)
        self._record(
            hook,
            tool,
            "follow_up",
            "followup",
            f"open: {names}; held: {', '.join(sorted({i.ob.id for i in held})) or 'none'}",
            "code",
        )
        return text

    def _may_send(self, item: Item) -> bool:
        tools = (item.ob.effect or {}).get("tools") or []
        if set(tools) & set(self.handoff_tools):
            return False  # the hand-off itself is not asked for before a hand-off
        return (self.item_level or item.ob.level) == "follow_up"

    # -- binding -----------------------------------------------------------------------------------------------------

    def _bind(self, hook: str) -> None:
        if self._bound:
            return
        self._bound = True
        bindable = [o for o in self.obligations if o.origin in ("package", "data")]
        if not bindable:
            return
        data = "\n\n".join(self.reads)
        known = None
        if self._loop_log is not None:
            known = known_from_loop(self._loop_log, self.cards, f"{self.request}\n{data}", self.policy_text)
        try:
            if self.model is None:
                raise BindingError("no model")
            cases = bind_cases(
                self.model, self.request, data, self.cards, known=known, policy=self.policy_text
            )
        except BindingError as exc:
            self._failed_binding = True
            self._record(hook, "", "binding", "error", f"no follow-up: {exc}", "model")
            return
        grounded = [c for c in cases if c.grounded]
        for case in cases:
            if not case.grounded:
                self._record(
                    hook,
                    "",
                    "binding",
                    UNBOUND,
                    f"case {case.name}: its evidence is not a quote of the data, so it binds nothing",
                    "model",
                )
        bound: Set[str] = set()
        for ob in bindable:
            scope = ob.scope if isinstance(ob.scope, dict) else {}
            procedure = str(scope.get("procedure") or "")
            for case in grounded:
                if not procedure or procedure == case.procedure:
                    self._add_item(ob, case)
                    bound.add(ob.id)
            if ob.id not in bound:
                self._audit_item(ob, "", UNBOUND, "no case of this procedure", hook)

    def _add_item(self, ob: Obligation, case: Case, since: int = 0) -> None:
        gate, why = _when_gate(ob, case)
        self.items.append(
            Item(
                ob=ob,
                case=case,
                gate=gate,
                gate_reason=why,
                since=since,
                plan=plan_of(ob, case, self.store, self.today),
                values=resolve_values(ob, case, self.store, self.today),
            )
        )

    # -- the check ---------------------------------------------------------------------------------------------------

    def _refresh(self, answer: Optional[str], hook: str) -> None:
        checked = [self._check(i, (), answer) for i in self.items]
        met_by: Dict[int, Set[int]] = {}
        for n, result in enumerate(checked):
            for call in result.met_calls:
                met_by.setdefault(call, set()).add(n)
        for n, item in enumerate(self.items):
            taken = {call for call, who in met_by.items() if who - {n}}
            result = self._check(item, taken, answer) if taken else checked[n]
            self._settle(item, result, hook)

    def _check(self, item: Item, taken: Iterable[int], answer: Optional[str]) -> Checked:
        # a call before the item opened is blanked, not cut, so the indexes stay the ones of the whole log
        log = (
            self.calls if not item.since else [{} if n < item.since else c for n, c in enumerate(self.calls)]
        )
        return check_calls(
            item.ob,
            log,
            item.plan,
            specs=self.specs,
            wrong_values=self.wrong_values,
            taken=taken,
            answer=answer,
            tools_in_run=self.tools_in_run,
            # a request or a duty item has no case: keys None, so every call of its tool counts as about it (a call that omits
            # an argument is `unknown`, not no evidence, and the item is not asked for as if nothing had been done)
            keys=item.case.keys if item.ob.origin not in ("request", "duty") else None,
            # a request or a duty item has no case to tell from another: no call is another case's. A package item whose case has no keys keeps the keys of the others
            other_keys=[k for i in self.items if i.case is not item.case for k in i.case.keys]
            if item.ob.origin not in ("request", "duty")
            else [],
        )

    def _settle(self, item: Item, result: Checked, hook: str) -> None:
        """The order decides: a branch that excludes the item; a call that did it; a hold of the case; a blocked call; the
        rest of the code check. A follow-up can then never ask for what a block or a hold stopped."""
        case = item.case
        status, reason = result.status, result.reason
        item.hold = ""
        if item.gate == NOT_APPLICABLE:
            status, reason = NOT_APPLICABLE, item.gate_reason
        elif status != MET or not result.met_calls:  # a call with nothing compared does not outrank a block
            block = self._block_on(item)
            if status != MET and case.decision == "hold":
                status, reason = HELD, f"the case is on hold: {case.decision_evidence}"
                item.hold = f'the records say "{_one_line(case.decision_evidence)}"'
            elif status != MET and case.decision == "unknown":
                status, reason = UNKNOWN, "the binding gave no clear decision for this case"
            elif status != MET and item.gate == UNKNOWN:
                status, reason = UNKNOWN, item.gate_reason
            if block is not None and status != HELD:
                status, reason = HELD, f"blocked call of {block['tool']}: {block['message']}"
                item.hold = block["message"] or f"a check stopped {block['tool']}"
        item.status, item.reason = status, reason
        if item.recorded != status:
            item.recorded = status
            self._audit_item(item.ob, case.name, status, reason, hook)

    def _block_on(self, item: Item) -> Optional[Dict[str, Any]]:
        if not (item.ob.effect or {}).get("tools"):
            return None
        for block in self.blocks:
            if self._attaches(item, block) and not self._cleared(item, block):
                return block
        return None

    def _attaches(self, item: Item, block: Dict[str, Any]) -> bool:
        """Does a blocked call stop this item? The keys of the case come first: a block that names one holds the case's write
        items, on any tool, because a first try in the wrong format is the common case. A block that names another case
        does not. On the item's own tool it does, with nothing to tell them apart, unless a fixed text of an identifying
        argument differs. ``not_sure`` is never another target: a follow-up must not ask for it."""
        text = block["text"]
        tools = (item.ob.effect or {}).get("tools") or []
        if _names(item.case.keys, text):
            return True
        others = {id(i.case): i.case for i in self.items if i.case is not item.case}
        if any(_names(c.keys, text) for c in others.values()):
            return False
        if block["tool"] not in tools:
            return self._same_work_elsewhere(item, block)
        for arg, tokens in item.plan.items():
            value = block["args"].get(arg)
            if value is None or _is_free_text(arg):
                continue
            schema = _schema_of(self.specs, block["tool"], arg)
            if arg_outcome(block["tool"], arg, tokens, value, schema) == FAIL:
                return False
        return True

    def _same_work_elsewhere(self, item: Item, block: Dict[str, Any]) -> bool:
        """A block on a tool outside the item's tools, naming no case: the same work, when it holds a text of the item's
        pattern, or when a tool that writes files was stopped and the item is a file or names only code tools."""
        for tokens in item.plan.values():
            for piece, trust, _ in tokens:
                if trust != GAP and len(fold(piece)) > MIN_KEY_CHARS and fold(piece) in block["text"]:
                    return True
        effect = item.ob.effect or {}
        on_code_only = all(CODE_TOOL.search(str(t)) for t in effect.get("tools") or [])
        writes_files = FILE_TOOL.search(block["tool"]) or CODE_TOOL.search(block["tool"])
        return (effect.get("kind") == "file" or on_code_only) and bool(writes_files)

    def _cleared(self, item: Item, block: Dict[str, Any]) -> bool:
        """A block stays until the same work went through. A later call of the blocked tool clears it for an item:

        - the same call, in the same words: for every item;
        - naming the same keys of the case as the blocked call: for an item on that tool only if it has no pattern on an
          argument that is not an identifier (a hold notice for the same person is not the approval that was stopped; if
          the pattern fits, the item is ``met`` and is not here); for an item on another tool only if every argument that is
          not an identifier is the same in the two calls, because the item's own pattern says nothing about this call.

        With no key named there is no way to tell two calls apart, so only the same call clears. Identifiers are named by a
        list and all other arguments count, so that an argument with a new name cannot reopen this."""
        later = [c for c in self.calls[block["at"] :] if c["tool"] == block["tool"]]
        if any(_readable_call(c.get("args"))[0] == block["text"] for c in later):
            return True
        named = [k for k in item.case.keys if _names([k], block["text"])]
        if not named:
            return False
        same_case = [c for c in later if all(_names([k], _readable_call(c.get("args"))[0]) for k in named)]
        if not same_case:
            return False
        if block["tool"] in ((item.ob.effect or {}).get("tools") or []):
            return all(ID_ARG.search(arg) for arg in item.plan)

        def rest(args: Dict[str, Any]) -> Dict[str, str]:
            return {k: _value_text(v) for k, v in (args or {}).items() if not ID_ARG.search(k)}

        return any(rest(c.get("args")) == rest(block["args"]) for c in same_case)

    # -- the audit ---------------------------------------------------------------------------------------------------

    def _audit_item(self, ob: Obligation, case: str, status: str, reason: str, hook: str) -> None:
        source = ob.source if isinstance(ob.source, dict) else {}
        self.records.append(
            {
                "hook": hook,
                "tool": ",".join(t for t in (ob.effect or {}).get("tools") or [] if isinstance(t, str)),
                "rule_id": ob.id,
                "outcome": status,
                "reason": f"{case}: {reason}" if case else reason,
                "source": source.get("section") or "",
                "level": self.item_level or ob.level,
                "checker": "code",
            }
        )

    def _record(self, hook: str, tool: str, rule: str, outcome: str, reason: str, checker: str) -> None:
        self.records.append(
            {
                "hook": hook,
                "tool": tool,
                "rule_id": rule,
                "outcome": outcome,
                "reason": reason,
                "checker": checker,
            }
        )


def _when_gate(ob: Obligation, case: Case) -> Tuple[str, str]:
    """What ``scope.when`` says for this case. Every key is answered with quoted evidence, or the item is ``unknown``."""
    scope = ob.scope if isinstance(ob.scope, dict) else {}
    when = scope.get("when")
    if not when:
        return OPEN, ""
    if not isinstance(when, dict):
        return UNKNOWN, "scope.when is not a list of keys"
    missing: List[str] = []
    for key, want in when.items():
        wants = [fold(_as_text(w)) for w in (want if isinstance(want, list) else [want])]
        if key == "case_decision":
            if case.decision in ("proceed", "hold") and fold(case.decision) not in wants:
                return NOT_APPLICABLE, f"case_decision is {case.decision}"
            continue
        entry = case.answers.get(key)
        answer = fold((entry or {}).get("answer"))
        if not answer or answer == "unknown":
            missing.append(key)
        elif answer not in wants:
            return NOT_APPLICABLE, f"{key} is {answer}: {(entry or {}).get('evidence', '')}"
    if missing:
        return UNKNOWN, f"no answer with quoted evidence for {', '.join(missing)}"
    return OPEN, ""


def _identity(args: Dict[str, Any]) -> Dict[str, str]:
    """The arguments that say which target a call is about, for opening one duty per target. Code is identified by its text,
    and a call with only free text by all of it. Never for clearing a block: free text is not part of it."""
    code = {k: fold(_as_text(v)) for k, v in (args or {}).items() if k in CODE_ARGS}
    if code:
        return code
    short = {
        k: fold(_as_text(v))
        for k, v in (args or {}).items()
        if not _is_free_text(k) and len(_as_text(v)) <= IDENTITY_CHARS
    }
    return short or {k: fold(_as_text(v)) for k, v in (args or {}).items()}
