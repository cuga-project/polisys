"""SOP planner (spike): puts the SOP procedures that apply to each case into the agent's plan.

From the SOP's procedure graph (offline intake) and the data the agent has read, a model lists the cases in the data
that the request covers (an email, a person, a suspense item), the procedure each one needs, its preconditions and
exceptions, and the procedure's steps with the templates filled in for that case. Cases the SOP says to skip, hold or
forward are listed as such. The plan is shown at every model call (``before_llm_call``, re-sent each call, so it stays
in view) and steps are ticked off when a write matches them (the step's tool and the case's name in the arguments).

``directive=True`` (level 3, procedure-driven) adds a "NEXT" line: the first open step of the first open case.
The planner never acts and never blocks; checks stay with ``procedure_guide.ProcedureGuide``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from polisys.hooks import AgentHooks, HookResult

from .llm import parse_json
from .model import step_text
from .ruleset import is_write_tool

READ_CHARS, DATA_CHARS, PROC_CHARS = 8_000, 60_000, 16_000
REPLAN_AFTER_READS, MAX_PLANS = 3, 3

SYSTEM = "You plan how a company SOP applies to the cases in some work. Output JSON only."
USER = """Work request:
<<<
{request}
>>>

SOP procedures (from the SOP; numbered):
{procedures}

Data the agent has read so far:
<<<
{data}
>>>

List every case in the data that the request covers (one email, person, ticket or item each). For each case:
{{"case": "<short name with the identifying value, e.g. 'Daniel Yoon (new hire)'>", "key": "<a word that will appear in
the actions for this case, e.g. a last name or item id>", "procedure": <procedure number, or 0 if none applies>,
"action": "follow|skip|hold|forward|escalate", "why": "<the SOP reason when not 'follow'>",
"check_first": ["<precondition or exception to confirm in the data, with the SOP's words>"],
"steps": [{{"step": "<what to do, with this case's values and the SOP template filled in>", "tool": "<tool name if known>"}}]}}
Use only what the SOP says. A case whose exception applies gets action skip/hold/forward/escalate and only the steps
the SOP gives for that. Return {{"cases": [...]}}"""


def _proc_summary(i: int, p: Dict[str, Any]) -> str:
    lines = [f"{i}. {p.get('name')}"]
    lines += [
        f"   step {k}: {(s.get('quote') or step_text(s))[:160]} [tool: {s.get('action') or '-'}]"
        for k, s in enumerate(p.get("steps") or [], 1)
    ]
    lines += [
        f"   must hold: {c.get('condition')} (if not: {c.get('if_not')})"
        for c in p.get("preconditions") or []
    ]
    lines += [f"   exception: if {x.get('when')} then {x.get('then')}" for x in p.get("exceptions") or []]
    lines += [f"   template: {t.get('artifact')}: {t.get('template')}" for t in p.get("templates") or []]
    return "\n".join(lines)


class SopPlanner(AgentHooks):
    def __init__(
        self,
        procedures: List[Dict[str, Any]],
        model: Callable[[str, str], str],
        directive: bool = False,
        policy_doc_hint: str = "",
        is_write: Callable[[str], bool] = is_write_tool,
    ):
        self.procs = procedures
        self.model = model
        self.directive = directive
        self.doc_hint = policy_doc_hint.lower()
        self.is_write = is_write
        self.request = ""
        self.reads: List[str] = []
        self.reads_at_plan = 0
        self.plans = 0
        self.cases: List[Dict[str, Any]] = []
        self.done: set = set()  # (case index, step index)
        self.records: List[Dict[str, Any]] = []
        self.proc_text = "\n".join(_proc_summary(i, p) for i, p in enumerate(procedures, 1))[:PROC_CHARS]

    def on_user_message(self, message: str, thread_id: Optional[str]) -> HookResult:
        if not self.request:
            self.request = message or ""
        return HookResult()

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> HookResult:
        text = result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False)
        if self.is_write(tool) and not re.search(r"execute|bash|python", tool, re.I):
            flat = json.dumps(args, default=str, ensure_ascii=False).lower()
            for ci, c in enumerate(self.cases):
                key = str(c.get("key") or "").lower()
                if not key or key not in flat:
                    continue
                for si, s in enumerate(c.get("steps") or []):
                    if (ci, si) not in self.done and (not s.get("tool") or s.get("tool") == tool):
                        self.done.add((ci, si))
                        break
            return HookResult()
        if self.doc_hint and self.doc_hint in json.dumps(args).lower():
            return HookResult()  # the SOP itself: already in the procedure graph
        self.reads.append(f"{tool}({json.dumps(args, default=str)[:160]}) -> {text[:READ_CHARS]}")
        return HookResult()

    def _plan(self) -> None:
        data = "\n\n".join(self.reads)[-DATA_CHARS:]
        try:
            reply = parse_json(
                self.model(
                    SYSTEM, USER.format(request=self.request[:6000], procedures=self.proc_text, data=data)
                )
            )
        except Exception as e:  # noqa: BLE001 - no plan is better than a broken one
            self.records.append(
                {"hook": "before_llm_call", "rule": "sop_plan", "outcome": "error", "reason": str(e)[:200]}
            )
            return
        cases = [c for c in reply.get("cases") or [] if isinstance(c, dict) and c.get("case")]
        if cases:
            old = {
                (c.get("case"), si)
                for ci, c in enumerate(self.cases)
                for si in range(len(c.get("steps") or []))
                if (ci, si) in self.done
            }
            self.cases = cases
            self.done = {
                (ci, si)
                for ci, c in enumerate(cases)
                for si in range(len(c.get("steps") or []))
                if (c.get("case"), si) in old
            }
        self.records.append(
            {"hook": "before_llm_call", "rule": "sop_plan", "outcome": "plan", "cases": cases}
        )

    def before_llm_call(self, thread_id: Optional[str]) -> HookResult:
        if self.plans < MAX_PLANS and len(self.reads) - self.reads_at_plan >= REPLAN_AFTER_READS:
            self.plans += 1
            self.reads_at_plan = len(self.reads)
            self._plan()
        if not self.cases:
            return HookResult()
        lines = ["## SOP plan for this work (from the SOP and the data read; tick by doing)"]
        nxt = None
        for ci, c in enumerate(self.cases):
            proc = c.get("procedure")
            name = (
                self.procs[proc - 1].get("name")
                if isinstance(proc, int) and 1 <= proc <= len(self.procs)
                else "no procedure"
            )
            head = f"- {c.get('case')}: {c.get('action', 'follow').upper()} ({name})"
            if c.get("action", "follow") != "follow" and c.get("why"):
                head += f" - {c.get('why')}"
            lines.append(head)
            for x in c.get("check_first") or []:
                lines.append(f"    check first: {x}")
            for si, s in enumerate(c.get("steps") or []):
                mark = "x" if (ci, si) in self.done else " "
                lines.append(
                    f"    [{mark}] {s.get('step')}" + (f" ({s.get('tool')})" if s.get("tool") else "")
                )
                if nxt is None and mark == " ":
                    nxt = (c, s)
        if self.directive and nxt is not None:
            c, s = nxt
            lines.append(
                f"NEXT: {c.get('case')} - {s.get('step')}"
                + (f" ({s.get('tool')})" if s.get("tool") else "")
                + ". Do it now unless the data shows the SOP says otherwise for this case."
            )
        return HookResult(message="\n".join(lines))


def planner_from_build(path: str, model: Callable[[str, str], str], **kw: Any) -> SopPlanner:
    data = json.loads(open(path).read())
    build = data.get("build", data)
    sop = (data.get("stats") or {}).get("sop", "")
    return SopPlanner(build.get("procedures") or [], model, policy_doc_hint=sop.rsplit(".", 1)[0][:30], **kw)
