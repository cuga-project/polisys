"""SOP-aware loop (spike): an outer controller that walks cases and procedures, and asks narrow typed questions.

ReAct / CodeAct loops interleave thinking and acting in one model. This controller runs the same CUGA agent in one
thread, but in phases, and hands typed questions to a separate model at fixed decision points:

0. explore  - CUGA reads the inputs of the request; writes are stopped (``LoopHooks.mode = "explore"``).
1. cases    - which cases does the request cover in the data read (one email, person, ticket, item each)?
2. rules    - with no SOP graph: which rules in the data read apply to this work (policies found in emails, sheets)?
   per case:
3. choose   - which SOP procedure applies (a typed choice among the procedures, or none)?
4. check    - does each precondition hold, does each exception apply (yes / no / unknown, with the data's words)?
5. handled  - is the case already handled in the data (an existing ticket, record or message)?
6. decide   - follow / skip / hold / escalate, by code from the answers (a "no" on a hard precondition or a "yes" on an
              exception decides), never by the sub-task model alone.
7. do       - one scoped turn for CUGA: the decision, the procedure's steps with templates, the checks and the rules.
8. finish   - a last turn: the parts of the request that are not cases, then the final answer.
The agent keeps the final say inside its turn (it may state why it deviates); checks before writes stay with the
procedure guide when it is attached. Nothing here is specific to a benchmark: inputs are the request, the procedure
graph (or none) and what the agent read.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from polisys.hooks import AgentHooks, HookResult

from .llm import parse_json
from .model import step_text
from .ruleset import is_write_tool
from .trace import audited_failure, emit

READ_CHARS, DATA_CHARS, CASE_DATA_CHARS = 20_000, 120_000, 40_000
MAX_CASES = 12
CODE_TOOL = re.compile(r"execute|bash|python|shell|code|script", re.I)
HARD = re.compile(r"^\s*(stop|hold|do not|don't|halt|escalate|skip)", re.I)
HELD_STOPS = 2  # a held case's normal-procedure write is stopped at most this many times
PROOF = re.compile(
    r"authori[sz]|approv|written|signed|signature|consent", re.I
)  # preconditions that need proof


class LoopHooks(AgentHooks):
    """Records reads and writes; in "explore" mode it stops writes (code tools are allowed: they also read)."""

    def __init__(self, is_write: Callable[[str], bool] = is_write_tool):
        self.mode = "free"
        self.is_write = is_write
        self.reads: List[str] = []
        self.writes: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []
        self.held: List[Dict[str, Any]] = []  # cases decided hold / stop / exception (set by SopLoop)
        self.which: Optional[Callable[[str, Dict[str, Any]], List[Any]]] = (
            None  # write -> procedures it is a step of
        )

    @audited_failure
    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> HookResult:
        if self.mode == "explore" and self.is_write(tool) and not CODE_TOOL.search(tool):
            emit(  # a stop with no record of its own: the audit has it, the verdict records stay as they were
                self,
                {
                    "hook": "before_tool_call",
                    "tool": tool,
                    "rule": "explore_phase",
                    "outcome": "deny",
                    "reason": "writes wait until the request's inputs are read",
                },
            )
            return HookResult(
                message="EXPLORE PHASE: do not change anything yet. Only read what the request needs; "
                "the work itself comes in the next steps."
            )
        if self.held and self.which and self.is_write(tool) and not CODE_TOOL.search(tool):
            msg = self._held_stop(tool, args)
            if msg:
                return HookResult(message=msg)
        return HookResult()

    def _held_stop(self, tool: str, args: Dict[str, Any]) -> Optional[str]:
        """A write that carries a held case's normal-procedure template and names that case is stopped (twice at most).
        Writes of other procedures and the SOP's own actions for the situation (notices, forwards) pass."""
        text = json.dumps(args, default=str, ensure_ascii=False).lower()
        procs = {
            p.get("id") for p, t in (self.which(tool, args) or []) if t is not None
        }  # only by the step's template text
        for h in self.held:
            keys = [k.lower() for k in h["keys"] if k and len(k) > 2]
            if h["proc_id"] in procs and any(k in text for k in keys) and h["stops"] < HELD_STOPS:
                h["stops"] += 1
                self.records.append(
                    {
                        "hook": "before_tool_call",
                        "rule": "case_decision",
                        "case": h["case"],
                        "tool": tool,
                        "outcome": "deny",
                    }
                )
                return (
                    f"SOP DECISION FOR {h['case']}: {h['decision'].upper()} ({h['reason'][:200]}). This write is a step "
                    f"of the normal procedure \"{h['proc_name']}\", which must not run for this case. Do only what the "
                    "SOP says for this situation (for example post the hold or stop notice, forward, or escalate)."
                )
        return None

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> HookResult:
        text = result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False)
        if self.is_write(tool) and not CODE_TOOL.search(tool):
            self.writes.append({"tool": tool, "args": args})
        else:
            self.reads.append(
                f"{tool}({json.dumps(args, default=str, ensure_ascii=False)[:200]}) -> {text[:READ_CHARS]}"
            )
        return HookResult()

    def data(self, keys: Optional[List[str]] = None, limit: int = DATA_CHARS) -> str:
        reads = self.reads
        if keys:
            low = [k.lower() for k in keys if k and len(k) > 2]
            hit = [r for r in reads if any(k in r.lower() for k in low)]
            reads = hit or reads
        return "\n\n".join(reads)[-limit:]


EXPLORE = """{request}

Before doing anything, first read what this request needs: the inputs it mentions (messages, files, records, tickets)
and any policy or SOP document that applies. Do not change anything in this step. When you have read enough, reply
with a short list of what you found."""

ADVISORY = """Here is the work request again:
<<<
{request}
>>>

The SOP was checked for each case in the data. Use this as your plan and do the whole request now, with the tools.
For each case: follow its procedure and steps; a HOLD / STOP / EXCEPTION below has evidence from the data, so do only
what the SOP says for that situation; a "no" check that did not change the decision is for you to verify. If the data
shows the SOP needs something different, do what the SOP says and state why in one line.

{plans}"""
DISCOVERY = """Here is the work request again:
<<<
{request}
>>>

For each case in the data, the SOP sections that apply were found for you: the procedure, its preconditions,
exceptions, templates and related rules. Use them as your checklist and do the whole request now, with the tools.
Decide from the data whether a precondition fails or an exception applies (then do only what the SOP says for that
situation). Use template text exactly.

{plans}"""
Q_SYSTEM = "You answer one narrow question about some work, using only the given data. Output JSON only."
Q_CASES = """Work request:
<<<
{request}
>>>

Data read so far:
<<<
{data}
>>>

List the separate cases this request covers in the data. A case is something that needs its OWN separate actions (its
own tickets, messages or records), such as one email to handle or one person to onboard. When the request turns a list
of items into ONE combined result (one report, workbook, digest, summary email or reconciliation), the whole list is
ONE case: do not split it per item. If the request is one single piece of work, return one case. For each: {{"case": "<short name with its
identifying value>", "keys": ["<words that identify it in the data and in actions, e.g. a last name, an id, an
address>"], "summary": "<what the data says about it, one sentence>", "combined": <true if the case is a list of items
turned into one result, else false>}}. At most {max_cases}. Return {{"cases": [...]}}"""
Q_RULES = """Work request:
<<<
{request}
>>>

Data read so far:
<<<
{data}
>>>

List the rules in this data that apply to this work (policies, instructions or SOP text found in messages, sheets or
documents, not the request itself): {{"rule": "<the rule, short>", "quote": "<its words, copied exactly>"}}.
Return {{"rules": [...]}} (empty if there are none)."""
Q_CHOOSE = """Case: {case} - {summary}

SOP procedures:
{procedures}

Data about this case:
<<<
{data}
>>>

Which one procedure does this case need? Return {{"procedure": <number, or 0 if none applies>, "evidence": "<the
data's words that decide it>"}}"""
Q_CHECK = """Case: {case} - {summary}
Condition from the SOP: {condition}
SOP text: "{quote}"

Data about this case:
<<<
{data}
>>>

Does the condition hold for this case? Be literal: if the SOP requires something written, signed or from a specific
person, only such a document or a message from that person counts (check the sender). If the condition is a
requirement for some kinds of cases only (for example "if X, then Y is required") and this case is not of that kind,
answer "not_applicable". Return {{"answer": "yes|no|not_applicable|unknown", "evidence": "<the data's words, or what
is missing>"}}"""
Q_GENERAL = """Case: {case} - {summary}

General SOP rules (numbered):
{rules}

Data about this case:
<<<
{data}
>>>

Which of these rules apply to this case, by what the data shows? Return {{"apply": [{{"n": <number>, "evidence":
"<the data's words>"}}]}} (empty if none applies)."""
Q_HANDLED = """Case: {case} - {summary}
Planned work for it: {work}

Data read so far (including existing tickets, records and sent messages):
<<<
{data}
>>>

Is this work already done in the data (for example an existing ticket, record or message for the same case and
purpose)? Return {{"answer": "yes|no|unknown", "evidence": "<the data's words>"}}"""


def _grounded(evidence: Any, data: str, min_len: int = 15) -> bool:
    """The evidence quotes the case data: some fragment of it (between quotes, or the whole) of at least ``min_len``
    characters appears in the data, ignoring case and spacing."""
    norm = lambda t: " ".join(str(t or "").lower().split())  # noqa: E731
    d, ev = norm(data), norm(evidence)
    if not ev or not d:
        return False
    parts = [x for x in re.findall(r"[\"“”']([^\"“”']{%d,})[\"“”']" % min_len, str(evidence or ""))] or [ev]
    parts += [x.strip() for x in re.split(r"[;|]|\.\.\.|…", ev) if len(x.strip()) >= min_len]
    return any(norm(x) in d for x in parts if len(norm(x)) >= min_len)


def _proc_line(i: int, p: Dict[str, Any]) -> str:
    steps = " / ".join(step_text(s)[:90] for s in (p.get("steps") or [])[:3])
    pre = "; ".join(str(c.get("condition"))[:80] for c in (p.get("preconditions") or [])[:2])
    return f"{i}. {p.get('name')}\n   first steps: {steps}" + (f"\n   needs: {pre}" if pre else "")


class SopLoop:
    def __init__(
        self,
        model: Callable[[str, str], str],
        procedures: Optional[List[Dict[str, Any]]] = None,
        rules_from_data: bool = True,
        mode: str = "control",
        require_evidence: bool = False,
        index: Optional[Any] = None,
    ):
        """``mode``: "control" runs one agent turn per case; "advisory" (for agents stronger than the question model)
        runs one work turn with every case's procedure, checks and decision, and the agent plans the work itself.
        ``require_evidence``: a stop / hold / exception decision counts only when its evidence quotes the case data;
        otherwise it stays a note and the case is followed."""
        self.mode = mode
        self.require_evidence = require_evidence
        # mode "discovery": per case, retrieve the procedure from the SOP index (sop_retrieval.SopIndex) and show
        # its expanded checklist; no model decisions; one work turn for all cases.
        self.index = index
        self.model = model
        self.procs = [p for p in procedures or [] if p.get("steps")]
        self.general = next((p for p in procedures or [] if not p.get("steps")), None)
        self.rules_from_data = rules_from_data
        self.log: List[Dict[str, Any]] = []
        self.receipts: List[Dict[str, Any]] = []  # one per agent turn (each turn is its own invoke)
        self._errors: List[
            str
        ] = []  # questions that failed, in order (private: it is not part of the wiring)
        # told of each phase ("explore", "work", "finish") with this loop (private: it is not part of the wiring)
        self._phase_listener: Optional[Callable[[str, Any], None]] = None

    @classmethod
    def from_package(cls, pkg: Any, model: Callable[[str, str], str], **kwargs: Any) -> "SopLoop":
        """A loop over the procedures of a ``PolicyPackage``. The other arguments are those of the constructor."""
        from .package import procedure_dicts

        return cls(model, procedure_dicts(pkg), **kwargs)

    def _phase(self, phase: str) -> None:
        if self._phase_listener is not None:
            self._phase_listener(phase, self)

    def _ask(self, user: str) -> Dict[str, Any]:
        """One question to the loop's model. An unanswered question decides nothing: the case goes on at its default (follow).
        The failure is kept for the audit, which says ``error`` for that case, where the run itself does not change."""
        try:
            return parse_json(self.model(Q_SYSTEM, user)) or {}
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"{type(exc).__name__}: {exc}"[:200])
            return {}

    async def _invoke(self, agent: Any, msg: str, thread_id: str, timeout: float) -> Any:
        import asyncio

        r = await asyncio.wait_for(agent.invoke(msg, thread_id=thread_id), timeout=timeout)
        if getattr(r, "receipt", None) is not None:
            self.receipts.append(r.receipt.model_dump(exclude={"tool_timings"}))
        return r

    async def run(self, agent: Any, request: str, thread_id: str, rec: LoopHooks, timeout: float) -> Any:
        if self.procs:
            from .procedure_guide import ProcedureGuide

            rec.which = ProcedureGuide(
                self.procs, gate=False, templates=False
            )._which  # same identification as the guide
        rec.mode = "explore"
        self._phase("explore")
        await self._invoke(agent, EXPLORE.format(request=request), thread_id, timeout)
        rec.mode = "free"
        self._phase("work")
        data = rec.data()
        cases = (
            self._ask(Q_CASES.format(request=request[:8000], data=data, max_cases=MAX_CASES)).get("cases")
            or []
        )[:MAX_CASES]
        cases = [c for c in cases if isinstance(c, dict) and c.get("case")]
        rules = []
        if self.rules_from_data and not self.procs:
            rules = [
                r
                for r in self._ask(Q_RULES.format(request=request[:8000], data=data)).get("rules") or []
                if isinstance(r, dict)
            ]
        plan: Dict[str, Any] = {"phase": "plan", "cases": cases, "rules": rules}
        if self._errors:
            plan["errors"] = list(self._errors)
        self.log.append(plan)
        plist = "\n".join(_proc_line(i, p) for i, p in enumerate(self.procs, 1))
        plans: List[str] = []
        if self.mode == "discovery" and self.index is not None:
            for k, c in enumerate(cases, 1):
                q = f"{c['case']}. {c.get('summary', '')}"
                proc = self.index.choose(c["case"], c.get("summary", ""))
                self.log.append(
                    {
                        "phase": "case",
                        "case": c.get("case"),
                        "procedure": proc.get("name") if proc else None,
                        "decision": "discovery",
                    }
                )
                body = (
                    self.index.checklist(proc, q)
                    if proc
                    else "No matching SOP procedure was found; follow the SOP as you read it."
                )
                plans.append(f"Case {k} of {len(cases)}: {c['case']} - {c.get('summary', '')}\n{body}")
            if plans:
                await self._invoke(
                    agent,
                    DISCOVERY.format(request=request[:6000], plans="\n\n".join(plans)),
                    thread_id,
                    timeout,
                )
            cases = []  # the per-case questions below are not used in discovery mode
        for k, c in enumerate(cases, 1):
            keys = [str(x) for x in c.get("keys") or []]
            cdata = rec.data(keys, CASE_DATA_CHARS)
            entry: Dict[str, Any] = {"phase": "case", "case": c.get("case")}
            failed_before = len(self._errors)
            proc = None
            if self.procs:
                ch = self._ask(
                    Q_CHOOSE.format(
                        case=c["case"], summary=c.get("summary", ""), procedures=plist, data=cdata
                    ),
                )
                n = ch.get("procedure")
                proc = self.procs[n - 1] if isinstance(n, int) and 1 <= n <= len(self.procs) else None
                entry["procedure"] = proc.get("name") if proc else None
                entry["choose_evidence"] = ch.get("evidence")
            decision, reasons, checks = "follow", [], []
            conds = []
            if proc:
                conds += [
                    ("pre", x.get("condition"), x.get("quote"), x.get("if_not") or "stop")
                    for x in proc.get("preconditions") or []
                ]
                conds += [
                    ("exc", x.get("when"), x.get("quote"), x.get("then") or "")
                    for x in proc.get("exceptions") or []
                ]
            general_hits = []
            if self.general:
                g = (self.general.get("exceptions") or []) + (self.general.get("preconditions") or [])
                text = "\n".join(
                    f"{i}. If {x.get('when') or x.get('condition')} then {x.get('then') or x.get('if_not')}"
                    for i, x in enumerate(g, 1)
                )
                for a in (
                    self._ask(
                        Q_GENERAL.format(
                            case=c["case"], summary=c.get("summary", ""), rules=text, data=cdata
                        ),
                    ).get("apply")
                    or []
                ):
                    n = a.get("n") if isinstance(a, dict) else None
                    if isinstance(n, int) and 1 <= n <= len(g):
                        x = g[n - 1]
                        general_hits.append(
                            {
                                "kind": "general",
                                "condition": x.get("when") or x.get("condition"),
                                "answer": "yes",
                                "evidence": str(a.get("evidence"))[:240],
                                "then": x.get("then") or x.get("if_not") or "",
                            }
                        )
            for kind, cond, quote, then in conds[:12]:
                a = self._ask(
                    Q_CHECK.format(
                        case=c["case"], summary=c.get("summary", ""), condition=cond, quote=quote, data=cdata
                    ),
                )
                ans = a.get("answer")
                checks.append(
                    {
                        "kind": kind,
                        "condition": cond,
                        "answer": ans,
                        "evidence": str(a.get("evidence"))[:240],
                        "then": then,
                    }
                )
                ok = not self.require_evidence or _grounded(a.get("evidence"), cdata)
                if not ok and ans in ("yes", "no"):
                    entry.setdefault("ungrounded", []).append(f"{kind}: {cond} -> {ans}")
                if (
                    ok
                    and kind == "pre"
                    and ans == "no"
                    and HARD.search(str(then))
                    and PROOF.search(f"{cond} {quote}")
                ):
                    if decision == "follow":  # other failed preconditions stay notes for the agent
                        decision = HARD.search(str(then)).group(1).lower()
                    reasons.append(f"precondition not met: {cond}; SOP: {then} ({a.get('evidence')})")
                if ok and kind == "exc" and ans == "yes" and str(then).strip():
                    decision = "exception" if decision == "follow" else decision
                    reasons.append(f"exception applies: if {cond} then {then} ({a.get('evidence')})")
            checks += general_hits
            for x in general_hits:
                if self.require_evidence and not _grounded(x["evidence"], cdata):
                    entry.setdefault("ungrounded", []).append(f"general: {x['condition']}")
                    continue
                if HARD.search(str(x["then"])):
                    decision = "exception" if decision == "follow" else decision
                    reasons.append(
                        f"SOP rule applies: if {x['condition']} then {x['then']} ({x['evidence']})"
                    )
            if (
                c.get("combined") is True and decision != "follow"
            ):  # one result from many items: rules apply per item
                entry["per_item"] = reasons
                decision, reasons = "follow", []
            work = "; ".join(step_text(s)[:80] for s in (proc.get("steps") if proc else []) or [])[
                :600
            ] or c.get("summary", "")
            h = self._ask(
                Q_HANDLED.format(case=c["case"], summary=c.get("summary", ""), work=work, data=cdata),
            )
            if len(self._errors) > failed_before:
                entry["errors"] = self._errors[failed_before:]
            entry.update(
                {
                    "decision": decision,
                    "reasons": reasons,
                    "checks": checks,
                    "handled": h.get("answer"),
                    "handled_evidence": str(h.get("evidence"))[:240],
                }
            )
            self.log.append(entry)
            if proc and decision != "follow":
                rec.held.append(
                    {
                        "case": c["case"],
                        "keys": keys,
                        "proc_id": proc.get("id"),
                        "proc_name": proc.get("name"),
                        "decision": decision,
                        "reason": "; ".join(reasons)[:500],
                        "stops": 0,
                    }
                )
            msg = self._case_turn(
                k, len(cases), c, proc, decision, reasons, checks, h, rules, entry.get("per_item")
            )
            if self.mode == "advisory":
                plans.append(msg)
            else:
                await self._invoke(agent, msg, thread_id, timeout)
        if self.mode == "advisory" and plans:  # one work turn: the agent plans and does all cases itself
            await self._invoke(
                agent, ADVISORY.format(request=request[:6000], plans="\n\n".join(plans)), thread_id, timeout
            )
        decided = [
            e
            for e in self.log
            if e.get("phase") == "case" and e.get("decision") not in ("follow", "discovery")
        ]
        keep = ""
        if decided:
            keep = (
                "\nThese case decisions stand - do not run the normal procedure for them now; only the SOP's actions "
                "for their situation (hold, stop, forward, escalate, notify) belong to them:\n"
                + "\n".join(
                    f"- {e['case']}: {e['decision'].upper()} ({'; '.join(e.get('reasons') or [])[:300]})"
                    for e in decided
                )
            )
        finish = (
            "All cases above are handled. Here is the original request again:\n<<<\n"
            + request[:6000]
            + "\n>>>\n"
            "Check it line by line: do every output it asks for that does not exist yet (for example a summary, "
            "report, file or message), without repeating work already done."
            + keep
            + "\nThen give the final answer."
        )
        self._phase("finish")
        return await self._invoke(agent, finish, thread_id, timeout)

    def _case_turn(
        self,
        k: int,
        n: int,
        c: Dict[str, Any],
        proc: Optional[Dict[str, Any]],
        decision: str,
        reasons: List[str],
        checks: List[Dict[str, Any]],
        handled: Dict[str, Any],
        rules: List[Dict[str, Any]],
        per_item: Optional[List[str]] = None,
    ) -> str:
        lines = [f"Case {k} of {n}: {c['case']} - {c.get('summary', '')}"]
        if proc:
            lines.append(f"SOP procedure: {proc.get('name')}")
        if decision == "follow":
            lines.append("Decision from the checks: FOLLOW the procedure for this case.")
        else:
            lines.append(
                f"Decision from the checks: {decision.upper()} - "
                + "; ".join(r[:300] for r in reasons[:6])
                + ". Do only what the SOP says for this situation (for example stop, hold, forward or escalate as "
                "it states); do not run the normal procedure for this case."
            )
        if per_item:
            lines.append(
                "This case is one result built from many items. These SOP rules hold for SOME items; apply each "
                "rule's own action to the items it fits (do not stop the whole case):"
            )
            lines += [f"- {r[:300]}" for r in per_item[:8]]
        if checks:
            lines.append("Checks (from the data):")
            lines += [
                f"- {x['condition']}: {x['answer']} ({x['evidence']})"
                for x in checks
                if x.get("answer") in ("yes", "no")
            ][:10]
            lines.append(
                "A 'no' above that did not change the decision is for you to check in the data before you act."
            )
        if proc and decision == "follow":
            tpl = {str(t.get("step")): t for t in proc.get("templates") or []}
            lines.append("Steps:")
            for i, s in enumerate(proc.get("steps") or [], 1):
                t = tpl.get(str(i))
                lines.append(
                    f"{i}. {step_text(s)[:220]}" + (f"  [template: {t.get('template')}]" if t else "")
                )
        if handled.get("answer") == "yes":
            lines.append(
                f"Note: this may already be done ({handled.get('evidence')}). Check before you repeat anything; "
                f"do not create duplicates."
            )
        if rules:
            lines.append("Rules found in the data that apply to this work:")
            lines += [f"- {r.get('rule')}" for r in rules[:10]]
        if self.mode != "advisory":
            lines.append(
                "Do this case now with the tools. Handle only this case in this step. If the data shows the SOP "
                "needs something different, do what the SOP says and state why in one line."
            )
        return "\n".join(lines)
