"""Requirements tracker hook: at the reply, compile what the finished work must contain and check it.

From the request (``on_user_message``), the policy text the runner gives (for example a system-prompt policy) and
everything the agent has read (results of non-write calls; this includes policies found in emails, sheets and SOP
files the agent opened), ``requirements.compile_requirements`` lists the required effects, steps, guards, formats and
per-record decisions. ``requirements.gate`` then checks them against the agent's successful writes. Items that are
missing or wrong go back to the agent as one specific follow-up (``before_reply``), at most ``max_rounds`` times;
each later round re-checks the same checklist against the new writes.

The hook reads only what the agent itself read; it never opens a source.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from polisys.hooks import AgentHooks, HookResult

from .followups import CHECK_HEADER, required_actions
from .requirements import compile_requirements, gate
from .ruleset import is_write_tool

PER_RESULT_CHARS = 80_000  # a whole SOP or handbook file the agent opened
DATA_CHARS = 160_000


def _failed(res: Any) -> bool:
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except ValueError:
            return res.lower().startswith("error")
    return isinstance(res, dict) and ("error" in res or res.get("success") is False)


class RequirementsTracker(AgentHooks):
    def __init__(
        self,
        model: Callable[[str, str], str],
        policy: str = "",
        max_rounds: int = 2,
        is_write: Callable[[str], bool] = is_write_tool,
        wrong_mode: str = "off",
        voice: str = "order",
        evidence: str = "off",
    ):
        self.model, self.policy, self.max_rounds, self.is_write = model, policy, max_rounds, is_write
        self.request = ""
        self.reads: List[str] = []
        self.writes: List[Dict[str, Any]] = []
        self.items: Optional[List[Dict[str, Any]]] = None
        self.rounds = 0
        self.records: List[Dict[str, Any]] = []
        # items judged "wrong": "off" = not shown (default: such judgments come from values the compiler inferred and
        # led the agent to change correct records); "verify" = ask to check by a read first; "fix"
        self.wrong_mode = wrong_mode
        self.voice, self.evidence = voice, evidence  # followups.py

    def on_user_message(self, message: str, thread_id: Optional[str]) -> HookResult:
        if not self.request:  # later turns are this hook's own follow-ups
            self.request = message or ""
        return HookResult()

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> HookResult:
        text = result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False)
        if self.is_write(tool):
            if not _failed(result):
                self.writes.append({"tool": tool, "args": args, "result": result})
        else:
            self.reads.append(f"{tool}({json.dumps(args, default=str)[:200]}) -> {text[:PER_RESULT_CHARS]}")
        return HookResult()

    def _data(self) -> str:
        data, total = [], 0
        for r in self.reads:  # earliest first: listings, then documents and records
            if total + len(r) > DATA_CHARS:
                break
            data.append(r)
            total += len(r)
        return "\n".join(data)

    def before_reply(self, answer: str) -> HookResult:
        if self.rounds >= self.max_rounds or not self.request:
            return HookResult()
        try:
            if self.items is None:
                cl = compile_requirements(self.model, self.request, self.policy, data=self._data())
                self.items = cl["items"]
                self.records.append(
                    {
                        "hook": "before_reply",
                        "rule": "requirements",
                        "outcome": "compiled",
                        "items": self.items,
                        "dropped": len(cl["dropped"]),
                    }
                )
            status = gate(self.model, self.items, self.writes, answer=answer if self.voice == "check" else "")
        except Exception as e:  # noqa: BLE001 - a failed check must not stop the agent
            self.records.append(
                {"hook": "before_reply", "rule": "requirements", "outcome": "error", "reason": str(e)[:200]}
            )
            return HookResult()
        missing = [it for it in self.items if (status.get(it["id"]) or {}).get("status") == "missing"]
        wrong = [it for it in self.items if (status.get(it["id"]) or {}).get("status") == "wrong"]
        if self.wrong_mode == "off":
            wrong = []
        left_out = []
        if self.evidence == "on" and missing:
            try:
                keep, left_out = required_actions(
                    self.model, [it["text"] for it in missing], self.request, self.policy, self._data(), answer
                )
                missing = [missing[i] for i in keep]
            except Exception as e:  # noqa: BLE001 - a failed check leaves the items as they are
                self.records.append(
                    {"hook": "before_reply", "rule": "requirements", "outcome": "evidence_error", "reason": str(e)[:200]}
                )
        rec = {
            "hook": "before_reply",
            "rule": "requirements",
            "outcome": "followup" if missing or wrong else "complete",
            "round": self.rounds + 1,
            "status": status,
        }
        if left_out:
            rec["left_out"] = left_out
        self.records.append(rec)
        if not missing and not wrong:
            return HookResult(details=[rec])
        self.rounds += 1
        if self.voice == "check":
            lines = [CHECK_HEADER]
        else:
            lines = ["REQUIREMENTS CHECK before you finish (from the request and the policy):"]
        if missing:
            lines.append(
                "These may not be done yet:"
                if self.voice == "check"
                else "Not done yet - do these now, or say briefly why one does not apply:"
            )
            lines += [
                f"- {it['text']} ({status[it['id']]['why']}) [source: \"{it['quote'][:120]}\"]"
                for it in missing
            ]
        if wrong:
            if self.wrong_mode == "fix":
                lines.append("Possibly done differently from what is required - fix them:")
            else:  # a "wrong" judgment from call arguments is often a false alarm: verify before changing anything
                lines.append(
                    "Possibly done differently from what is required - first read the record or message to "
                    "check; change it only if what you read shows it really differs:"
                )
            lines += [
                f"- {it['text']} ({status[it['id']]['why']}) [source: \"{it['quote'][:120]}\"]"
                for it in wrong
            ]
        return HookResult(message="\n".join(lines), details=[rec])
