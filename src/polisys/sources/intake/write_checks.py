"""The checks before a write (part of ``OnlinePolicyIntake``)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from polisys.hooks import HookResult
from polisys.checks.decisions import action_text, has_limit
from polisys.runtime.trace import audited_failure, emit
from polisys.llm import today_line
from polisys.sources.intake.prompts import (
    MAX_SAME_DENY, PROVEN_ONLY, REQUEST_CONDITIONS, RETRY_RULE, SKIP_AFTER_CAP, VERIFY_SYSTEM, VERIFY_USER, _json,
    _with_statements,
)  # fmt: skip


class _WriteChecks:
    """Before a write: the two message checks when a profile turns them on (message_checks.py); then the pending intake is
    awaited, then guard code, the read-before-write reminder, the write decisions and the verifier call on the trusted
    statements."""

    def _may_deny(self, types: List[str]) -> bool:
        """A statement may block only if none of its types is measured as remind-only (method_choice)."""
        if not self.verifier_may_deny:
            return False
        choice = self.method_choice or {}
        return all(choice.get(t, {}).get("may_deny", True) is not False for t in types)

    def _tool_args(self, tool: str, args: Optional[Dict[str, Any]] = None) -> str:
        """The argument names of the tool's spec. ``args``: with no spec, the names of this call's own arguments."""
        spec = next((t for t in self.tool_specs or [] if t.get("name") == tool), None)
        props = ((spec or {}).get("inputSchema") or {}).get("properties") or {}
        return ", ".join(props) or ", ".join(args or {}) or "(unknown)"

    def _today(self) -> str:
        """For guard code, which needs a date: the task's date, else the clock."""
        import datetime as _dt

        return self._task_date() or _dt.date.today().isoformat()

    def _guard_code(
        self, tool: str, args: Dict[str, Any], statements: List[Dict[str, Any]]
    ) -> Optional[HookResult]:
        """Generated code checks (experimental/checks/guard_code.py) for the statements that decide this tool. Sets ``self._guarded`` to
        the statements whose check ran (they leave the model verifier's list). Returns a blocking result only in
        ``deny`` mode; in ``remind`` mode violations become a note for the next model call."""
        self._guarded: set = set()
        checks = self.guard.checks_for(statements, tool, args, self.observations)
        found = []
        for n, check in checks.items():
            out = self.guard.evaluate(check, tool, args, self.observations, self._today())
            if out is None:
                continue
            self._guarded.add(n)
            found += [{"statement": n, "reason": r} for r in out]
        if not checks:
            return None
        by_statement = {s["n"]: s for s in statements}
        if not found:
            self.records.append(
                {
                    "hook": "before_tool_call",
                    "tool": tool,
                    "outcome": "allow",
                    "rule": "guard_code",
                    "checked": sorted(self._guarded),
                }
            )
            return None
        by_n = {s["n"]: s for s in statements}
        deny = self.guard_mode == "deny"
        key = tool + "|guard|" + ",".join(sorted(str(v["statement"]) for v in found))
        over_cap = False
        if deny:
            self._denials[key] = self._denials.get(key, 0) + 1
            over_cap = self._denials[key] > MAX_SAME_DENY
        rec = {  # final when it is appended: the audit reads a record at that moment
            "hook": "before_tool_call",
            "tool": tool,
            "outcome": "allow_after_cap" if over_cap else ("deny" if deny else "remind"),
            "rule": "guard_code",
            "violations": _with_statements(found, by_statement),
        }
        self.records.append(rec)
        lines = [
            f"- [{by_n.get(v['statement'], {}).get('source', 'policy')}] "
            f"{by_n.get(v['statement'], {}).get('text', '')} — {v['reason']}"
            for v in found
        ]
        if not deny:
            self.notes.append(
                f"POLICY NOTE about your `{tool}` call (it was run; checked by code):\n" + "\n".join(lines)
            )
            return None
        if over_cap:
            return None
        return HookResult(
            message=f"POLICY: `{tool}` was not called (checked by code).\n"
            + "\n".join(lines)
            + "\nFix the call so it follows these rules.",
            details=[rec],
        )

    @staticmethod
    def _call_key(tool: str, args: Dict[str, Any]) -> str:
        return tool + "|" + hashlib.sha1(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()

    def _decide_write(
        self, tool: str, args: Dict[str, Any], statements: List[Dict[str, Any]]
    ) -> Optional[HookResult]:
        """Decision-model questions for one write: is its record or target excluded by a rule; would it break a
        limit (asked only when a rule states one); with ``record_checks``, does the record's own data say not to
        act on it now, and does a rule's exception depend on facts not read yet. A yes stops the call once. The
        same call again goes through, or with ``retry_effect="hold"`` is checked again and held when still
        flagged (a missing-facts flag never holds: the agent was asked to read them)."""
        key = self._call_key(tool, args)
        retry = key in self._blocked_once
        if retry and self.retry_effect != "hold":
            return None
        rules = self._fmt(statements) if statements else ""
        earlier = "\n".join(w for w in self.writes if w.startswith(tool + "("))
        with_limit = any(has_limit(st["text"]) for st in statements)
        ans = self.decisions.record_and_limit(
            rules,
            self._facts_for(args),
            earlier,
            action_text(tool, args),
            with_limit,
            record_checks=self.record_checks,
            today=self._task_date(),
        )
        if not ans:
            return None
        hits = [
            n
            for n in ("excluded", "over_limit", "record_says_skip", "needs_data")
            if self.decisions.yes(ans.get(n))
        ]
        rec = {
            "hook": "before_tool_call",
            "tool": tool,
            "rule": "decisions",
            "outcome": "allow",
            "answers": ans,
        }
        if retry:
            held = [h for h in hits if h != "needs_data"]
            rec["outcome"] = "hold_after_recheck" if held else "allow_after_block"
            self.records.append(rec)
            return HookResult(message=SKIP_AFTER_CAP.format(tool=tool, n=2), details=[rec]) if held else None
        if not hits:
            self.records.append(rec)
            return None
        self._blocked_once.add(key)
        rec["outcome"] = "block_once"
        self.records.append(rec)
        why = {
            "excluded": "a rule (or its exception or a later change) may exclude this record, person or target",
            "over_limit": "this may go over a limit in the rules on how many such actions are allowed",
            "record_says_skip": "the record's own data (its status, notes or flags) suggests not acting on it now",
            "needs_data": "a rule's exception or condition depends on facts you have not read yet: read them "
            "first (for example approvals, requests, leave or status records)",
        }
        return HookResult(
            message=f"POLICY CHECK: `{tool}` was not called: "
            + "; ".join(why[h] for h in hits)
            + ". " + RETRY_RULE + " Otherwise skip it and continue with the task.",
            details=[rec],
        )

    def _unused_read_tools(self, limit: int = 8) -> List[str]:
        specs = [t for t in self.tool_specs or [] if t.get("name") and not self.is_write(t["name"])]
        # With a large catalogue (specs carry their "app"), suggest only apps the agent already works in.
        app_of = {t.get("name"): t.get("app") for t in self.tool_specs or [] if t.get("app")}
        used_apps = {app_of[c] for c in self.called if c in app_of}
        if used_apps:
            specs = [t for t in specs if t.get("app") in used_apps or not t.get("app")]
        return [t["name"] for t in specs if t["name"] not in set(self.called)][:limit]

    def _where_to_look(self) -> str:
        unused = self._unused_read_tools()
        return f"\nRead tools you have not used yet: {', '.join(unused)}." if unused else ""

    def _read_before_write(self, tool: str) -> Optional[HookResult]:
        """A write while a source the request or data refers to is unread: record it and remind the agent once per
        source at the next model call. Finding and reading the source is the agent's job; this never blocks."""
        unread = self.refs.unread()
        if not unread:
            return None
        if self.read_before_write == "block_once" and not getattr(self, "_rbw_blocked", False):
            self._rbw_blocked = True
            self._ref_noted.update(r.text for r in unread)
            self.records.append(
                {
                    "hook": "before_tool_call",
                    "tool": tool,
                    "outcome": "deny",
                    "rule": "read_before_write",
                    "unread": [r.text for r in unread],
                }
            )
            return HookResult(
                message=f"POLICY: `{tool}` was not run. This task refers to policy sources that no tool "
                f"output has shown yet:\n{self.refs.summary()}{self._where_to_look()}\nFind and read each one first "
                "(search the connected applications, including parts of a source you opened but did not read), apply "
                "its rules, then make this call again if it still follows them. This check stops a "
                "write only once."
            )
        self.records.append(
            {
                "hook": "before_tool_call",
                "tool": tool,
                "outcome": "remind",
                "rule": "read_before_write",
                "unread": [r.text for r in unread],
            }
        )
        new = [r for r in unread if r.text not in self._ref_noted]
        if new:
            self._ref_noted.update(r.text for r in new)
            self.notes.append(
                "POLICY NOTE (read before write): this task refers to policy sources that no tool output "
                f"has shown yet. You wrote with `{tool}` before reading them:\n{self.refs.summary()}"
                f"{self._where_to_look()}\n"
                "Find and read each source (search the connected apps) before more writes, and apply "
                "its rules."
            )
        return None

    @audited_failure
    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> HookResult:
        import time as _time

        t0, n = _time.monotonic(), len(self.records)
        res = self._before_tool_call(tool, args)
        for r in self.records[n:]:  # the decision records of this call carry the check time
            if r.get("hook") == "before_tool_call":
                r["check_ms"] = round((_time.monotonic() - t0) * 1000)
        return res

    def _before_tool_call(self, tool: str, args: Dict[str, Any]) -> HookResult:
        if not self.is_write(tool):
            return HookResult()
        stop = self._message_checks(tool, args)
        if stop is not None:
            return stop
        if not self.verify_writes:
            return HookResult()
        if self.read_before_write != "off":
            stop = self._read_before_write(tool)
            if stop is not None:
                return stop
        self._drain(self.wait_s)
        if self.enforcer is not None:  # compiled online rules first (code)
            res = self.enforcer.before_tool_call(tool, args)
            details = [d for d in (res.details or []) if isinstance(d, dict)]
            if res.message:  # a deny: the call is blocked
                self.records.extend(details)
                return res
            kept = [d for d in details if d.get("outcome") == "remind"]
            self.records.extend(kept)
            for (
                d
            ) in details:  # a rule that denied and was let through (the cap, a scope check) is a decision too
                if not any(d is k for k in kept):
                    emit(self, d)
        statements = self._statements(True)
        # request_conditions: the verifier also checks the call against the request's own conditions, also with no rule found
        extra = (
            [
                {
                    "n": max([s["n"] for s in statements], default=0) + 1,
                    "source": "user request",
                    "text": REQUEST_CONDITIONS,
                    "types": [],
                }
            ]
            if self.request_conditions
            else []
        )
        if not statements:
            if (
                self.decisions is not None and self.record_checks
            ):  # no rules: the record's own signals still count
                stop = self._decide_write(tool, args, [])
                if stop is not None:
                    return stop
            if not extra:
                return HookResult()
        elif self.decisions is not None:
            stop = self._decide_write(tool, args, statements)
            if stop is not None:
                return stop
        if statements and self.guard_mode != "off":
            blocked = self._guard_code(tool, args, statements)
            if blocked is not None:
                return blocked
            statements = [s for s in statements if s["n"] not in self._guarded]
            if not statements and not extra:  # every statement has a code check for this tool: no model verdict needed
                return HookResult()
        statements = statements + extra
        narrow = self.request_conditions or self.proven_only  # as tested: the call's own argument names when no spec is known
        data = _json(
            self.model(
                VERIFY_SYSTEM,
                today_line(self._task_date())
                + VERIFY_USER.format(
                    request=self.request,
                    statements=self._fmt(statements),
                    facts=self._facts_for(args),
                    tool=tool,
                    args=json.dumps(args, default=str)[:3000],
                    tool_args=self._tool_args(tool, args if narrow else None),
                )
                + (PROVEN_ONLY if self.proven_only else ""),
            )
        )
        found = [v for v in data.get("violations") or [] if isinstance(v, dict)]
        by_n = {s["n"]: s for s in statements}
        soft = [v for v in found if not self._may_deny(by_n.get(v.get("statement"), {}).get("types") or [])]
        found = [v for v in found if v not in soft]  # measured remind-only types: a note, not a block
        if soft and not found and self.model_effect == "block_once":
            key = self._call_key(tool, args)
            if key not in self._blocked_once:
                self._blocked_once.add(key)
                rec = {
                    "hook": "before_tool_call",
                    "tool": tool,
                    "outcome": "block_once",
                    "violations": _with_statements(soft, by_n),
                }
                self.records.append(rec)
                lines = [f"POLICY CHECK: `{tool}` was not called. It may break these rules:"]
                for v in soft:
                    st = by_n.get(v.get("statement"), {})
                    lines.append(
                        f"- [{st.get('source', 'policy')}] {st.get('text', '')} — {v.get('reason', '')}"
                        + (f" Do instead: {v.get('fix')}" if v.get("fix") else "")
                    )
                lines.append(
                    RETRY_RULE + " Otherwise change or skip it and continue with the task."
                )
                return HookResult(message="\n".join(lines), details=[rec])
            if self.retry_effect == "hold":  # flagged again on the unchanged retry: keep it stopped
                rec = {
                    "hook": "before_tool_call",
                    "tool": tool,
                    "outcome": "hold_after_recheck",
                    "violations": _with_statements(soft, by_n),
                }
                self.records.append(rec)
                return HookResult(message=SKIP_AFTER_CAP.format(tool=tool, n=2), details=[rec])
            rec = {
                "hook": "before_tool_call",
                "tool": tool,
                "outcome": "allow_after_block",
                "violations": _with_statements(soft, by_n),
            }
            self.records.append(rec)
            return HookResult(details=[rec])
        if soft and not found:
            rec = {
                "hook": "before_tool_call",
                "tool": tool,
                "outcome": "remind",
                "violations": _with_statements(soft, by_n),
            }
            self.records.append(rec)
            # not returned as a message: a before_tool_call message blocks the call. Shown at the next model call,
            # once per statement (the same note on every write only distracts).
            new = [
                v for v in soft if by_n.get(v.get("statement"), {}).get("text") not in self._noted_statements
            ]
            self._noted_statements.update(by_n.get(v.get("statement"), {}).get("text") for v in new)
            if new:
                self.notes.append(
                    f"POLICY NOTE about your `{tool}` call (it was run): "
                    + "; ".join(str(v.get("reason", "")) for v in new)
                )
            return HookResult(details=[rec])
        outcome, key = "allow", ""
        if found:
            key = tool + "|" + ",".join(sorted(str(v.get("statement")) for v in found))
            self._denials[key] = self._denials.get(key, 0) + 1
            outcome = "deny"
            if self._denials[key] > MAX_SAME_DENY:
                outcome = "deny_after_cap" if self.cap_effect == "skip" else "allow_after_cap"
        rec = {  # final when it is appended: the audit reads a record at that moment
            "hook": "before_tool_call",
            "tool": tool,
            "outcome": outcome,
            "violations": _with_statements(found, by_n),
        }
        self.records.append(rec)
        if not found:
            return HookResult(details=[rec])
        if outcome == "deny_after_cap":
            return HookResult(message=SKIP_AFTER_CAP.format(tool=tool, n=self._denials[key]), details=[rec])
        if outcome == "allow_after_cap":
            return HookResult(details=[rec])
        by_n = {s["n"]: s for s in statements}
        lines = [f"POLICY: `{tool}` was not called."]
        for v in found:
            s = by_n.get(v.get("statement"), {})
            lines.append(
                f"- [{s.get('source', 'policy')}] {s.get('text', '')} — {v.get('reason', '')}. "
                f"Do instead: {v.get('fix', '')}"
            )
        return HookResult(message="\n".join(lines), details=[rec])
