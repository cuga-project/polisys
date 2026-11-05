"""Follow procedures (step graphs) during the run: remind the next step, list missing steps before the reply.

A procedure becomes active when the agent calls the tool of one of its steps. Then:

- ``after_tool_call`` marks the step done; a step done before an earlier required step is recorded as an
  ``order_warning`` (never denied: the order may be fine for reasons the monitor cannot see);
- ``before_llm_call`` adds "Next required step: ..." to the brief;
- ``before_reply`` (once per thread) asks the agent to finish the required steps that are still missing.

Steps without a tool, or whose tool is not in this run, are skipped: the agent could not do them. So is a step with a
``condition``, a ``reply`` or a read tool, when it carries ``text`` (a package written from a skill has a condition on about a
third of its steps, and a read tool is in several procedures): the monitor reads no case, so it has no evidence that the
condition holds, and a line would ask for work that may not apply. A read call, or a tool that is a step of several procedures, does not start a procedure of such a package. A
step of a build (``quote``, no ``text``) is handled as before, byte for byte. A line shows the step in words (``step_text``),
never an empty description.

``follow_up=False`` (the ledger sends the follow-ups of the run, design section 5) keeps the records and sends nothing: no
"Next required step" line and no message at the reply.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from polisys.hooks import AgentHooks, HookResult

from .model import Procedure, Step, step_text
from .ruleset import is_write_tool
from .trace import fresh

REPLY = "reply"  # the action of a step that is an answer to the user, in a package written from a skill


class ProcedureMonitor(AgentHooks):
    def __init__(
        self, procedures: List[Procedure], tools_in_run: Optional[List[str]] = None, *, follow_up: bool = True
    ):
        self.procedures = procedures
        self.tools_in_run = set(tools_in_run) if tools_in_run is not None else None
        # skill-style steps: how many procedures have a step with this tool (a shared tool cannot say which one starts)
        self._shared: Dict[str, int] = {}
        for proc in procedures:
            for tool in {st.action for st in proc.steps if st.text and st.action}:
                self._shared[tool] = self._shared.get(tool, 0) + 1
        self._follow_up = follow_up  # private: a flag of the wiring, not a setting of the monitor's own
        self.session_start(None)

    def session_start(self, thread_id: Optional[str]) -> HookResult:
        self.done: Dict[str, set] = {p.id: set() for p in self.procedures}
        self.active: set = set()
        self.records: List[Dict[str, Any]] = fresh(getattr(self, "records", None))
        self._replied = False
        return HookResult()

    def _usable(self, step: Step) -> bool:
        """A step the agent can do and that holds in every case: it has a tool in this run. A step that carries ``text`` (a
        package written from a skill; a build has ``quote`` and no ``text``) is also not listed when it has a ``condition``
        (the monitor reads no case, so it has no evidence that it holds), when it is a ``reply`` (the agent's answer, no
        tool call shows it done), or when its tool only reads (not work that can be left undone)."""
        if not step.action or (self.tools_in_run is not None and step.action not in self.tools_in_run):
            return False
        if step.text:
            return not step.condition and step.action != REPLY and is_write_tool(step.action)
        return True

    def _missing(self, p: Procedure) -> List[Step]:
        return [s for s in p.steps if s.required and self._usable(s) and s.id not in self.done[p.id]]

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> HookResult:
        if isinstance(result, dict) and "error" in result:
            return HookResult()
        details = []
        for p in self.procedures:
            for i, s in enumerate(p.steps):
                if s.action != tool or s.id in self.done[p.id]:
                    continue
                if s.text and not is_write_tool(
                    tool
                ):  # a read of a skill-style step is done, and starts no procedure
                    self.done[p.id].add(s.id)
                    break
                if s.text and self._shared.get(tool, 0) > 1:  # which procedure is it for? it starts none
                    self.done[p.id].add(s.id)
                    break
                self.active.add(p.id)
                earlier = [
                    e for e in p.steps[:i] if e.required and self._usable(e) and e.id not in self.done[p.id]
                ]
                self.done[p.id].add(s.id)
                rec = {
                    "hook": "after_tool_call",
                    "procedure": p.id,
                    "step": s.id,
                    "tool": tool,
                    "outcome": "order_warning" if earlier else "step_done",
                    "before": [e.id for e in earlier],
                }
                self.records.append(rec)
                details.append(rec)
                break
        return HookResult(details=details)

    def before_llm_call(self, thread_id: Optional[str]) -> HookResult:
        if not self._follow_up:
            return HookResult()
        lines = []
        for p in self.procedures:
            if p.id not in self.active:
                continue
            missing = self._missing(p)
            if missing:
                s = missing[0]
                lines.append(
                    f"- Next required step of {p.name} [§{s.source.get('section', '')}]: {step_text(s)} (`{s.action}`)"
                )
        if not lines:
            return HookResult()
        return HookResult(message="## Procedures in progress\n" + "\n".join(lines))

    def before_reply(self, answer: str) -> HookResult:
        if self._replied or not self._follow_up:
            return HookResult()
        self._replied = True
        lines, details = [], []
        for p in self.procedures:
            if p.id not in self.active:
                continue
            for s in self._missing(p):
                lines.append(f"- [{p.name} §{s.source.get('section', '')}] {step_text(s)} (`{s.action}`)")
                rec = {"hook": "before_reply", "procedure": p.id, "step": s.id, "outcome": "followup"}
                self.records.append(rec)
                details.append(rec)
        if not lines:
            return HookResult()
        return HookResult(
            message="POLICY: before you finish, these required procedure steps are still missing:\n"
            + "\n".join(lines),
            details=details,
        )
