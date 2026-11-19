"""After a call and at the reply: what the run has seen, and duties (part of ``RuleSetHooks``). ``duty_tools`` names the
tools that fulfil a duty."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from polisys.hooks import HookResult
from polisys.values import _when
from polisys.runtime.trace import audited_failure
from polisys.checks.rules.rule_types import Check

DUTY_VERBS = {
    "create",
    "update",
    "link",
    "add",
    "set",
    "new",
    "details",
    "to",
    "for",
    "get",
    "send",
    "record",
    "by",
    "incident",
}


def duty_tools(then: Dict[str, Any], specs: Dict[str, Any]) -> set:
    """Tools that fulfil a duty: the named tool, or a tool for the same kind of record (a shared name word other than a
    verb) whose schema has every required argument ("pause the SLA" by updating it or by linking a new one)."""
    name = then.get("tool", "")
    need = set(then.get("when") or {})
    kind = set(re.findall(r"[a-z]+", name.lower())) - DUTY_VERBS
    out = {name}
    if not need or not kind:
        return out
    for t, spec in specs.items():
        props = set(((spec.get("inputSchema") or {}).get("properties") or {}))
        if need <= props and kind <= set(re.findall(r"[a-z]+", t.lower())):
            out.add(t)
    return out


class _Duties:
    """After a call: the result joins what the run has seen, and the call may open or close a duty. At the reply: one
    follow-up while a duty is open."""

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> HookResult:
        self.facts.add_result(result)
        self.store.add(tool, result)
        self.called.append(tool)
        failed = isinstance(result, dict) and "error" in result
        if failed:
            return HookResult(message=self._tool_notes())
        details = []
        # a call may close an open duty
        still = []
        for trig, duty in self.open_duties:
            then = duty["then"]
            if tool in self._duty_tools(then) and _when(then.get("when"), args):
                details.append(
                    self._record("after_tool_call", tool, Check(duty, True, "duty done"), "duty_done")
                )
            else:
                still.append((trig, duty))
        self.open_duties = still
        for r in self.by_tool.get(tool, []):
            if r["type"] == "duty" and _when(r["trigger"].get("when"), args):
                if (
                    not self._duty_tools(r["then"]) & self.tools_in_run
                ):  # the agent has no tool to do it in this run
                    details.append(
                        self._record(
                            "after_tool_call", tool, Check(r, True, "duty tool not available"), "duty_skipped"
                        )
                    )
                    continue
                self.open_duties.append(({"tool": tool, "args": args}, r))
                details.append(
                    self._record("after_tool_call", tool, Check(r, True, "duty opened"), "duty_open")
                )
        return HookResult(message=self._tool_notes(), details=details)

    def _tool_notes(self) -> Optional[str]:
        """Queued reminders, handed to the tool result when notes go there (see ``note_channel``)."""
        if self.note_channel != "tool":
            return None
        return "\n".join(self.pop_notes()) or None

    def _duty_tools(self, then: Dict[str, Any]) -> set:
        return duty_tools(then, self.specs)

    @audited_failure
    def before_reply(self, answer: str) -> HookResult:
        if not self.open_duties or self.mode == "shadow" or not self._duty_follow_up:
            return HookResult()
        lines = ["POLICY: before you finish, these duties from the policy are still open:"]
        details = []
        for trig, duty in self.open_duties:
            unless = "; ".join(u.get("condition", "") for u in duty.get("unless") or [])
            if unless:  # the policy has an exception: the agent decides whether it applies
                lines.append(f"- [{duty.get('source', '')}] {self._describe(duty)} (not needed if: {unless})")
                details.append(
                    self._record(
                        "before_reply", trig["tool"], Check(duty, False, "duty with exception"), "remind"
                    )
                )
                continue
            lines.append(f"- [{duty.get('source', '')}] {self._describe(duty)}")
            details.append(
                self._record(
                    "before_reply", trig["tool"], Check(duty, False, "duty open at reply"), "followup"
                )
            )
        return HookResult(message="\n".join(lines), details=details)
