"""What the model is told about the rules (part of ``RuleSetHooks``)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from polisys.hooks import HookResult
from polisys.values import _joined


class _Brief:
    """Before a model call: the rules of the tools in the run, guidance, open duties and waiting notes, as text.
    ``_describe`` gives one rule as one line."""

    @property
    def brief_guidance(self) -> bool:
        return self._brief_guidance

    @property
    def exception_links(self) -> Dict[str, List[str]]:
        return self._exception_links

    def pop_notes(self) -> List[str]:
        notes, self.notes = list(getattr(self, "notes", [])), []
        return notes

    def before_llm_call(self, thread_id: Optional[str]) -> HookResult:
        notes = self.pop_notes()
        if not self.brief:
            return HookResult(message="\n".join(notes) or None)
        lines = ["## Policy rules checked before each write (from the policy text)"]
        shown = set()
        entries: List[Dict[str, Any]] = []  # the rules of the brief, in order
        for t in sorted(self.tools_in_run):
            for r in self.by_tool.get(t, []):
                if r["id"] in shown or self._observe_only(r):
                    continue
                shown.add(r["id"])
                entries.append(r)
        for r in self.global_rules:
            if not self._observe_only(r):
                entries.append(r)
        if self._brief_guidance:
            entries += self._global_guidance
        shown_exceptions = 0
        nested = self._nested(entries)
        reached: set = set()  # the rules written under another rule
        # A rule that is the exception of another rule of the brief is written under each rule it is an exception of, and not
        # as an item of its own. The rules of a cycle of exceptions have no first rule, so they come last, each once.
        top = [e for e in entries if id(e) not in nested]
        for first in (True, False):
            for r in top if first else [e for e in entries if id(e) in nested]:
                if not first and id(r) in reached:
                    continue  # a cycle that a rule above leads to
                lines.append(f"- [{r.get('source', '')}] {self._describe(r)}")
                reached.add(id(r))
                shown_exceptions += self._exception_lines(r, nested, lines, reached, {id(r)}, "  ")
        for trig, duty in self.open_duties if self._duty_follow_up else ():
            lines.append(f"- OPEN DUTY [{duty.get('source', '')}]: {self._describe(duty)}")
        if len(lines) == 1 and not notes:
            return HookResult()
        brief = "\n".join(notes + (lines if len(lines) > 1 else []))
        detail: Dict[str, Any] = {"rule": "brief", "lines": len(lines) - 1, "notes": len(notes)}
        if shown_exceptions:
            detail["exceptions"] = shown_exceptions
        return HookResult(message=brief, details=[detail])

    def _nested(self, entries: List[Dict[str, Any]]) -> set:
        """``id()`` of the rules written under another rule of the brief: the exceptions of a rule of the brief
        (``exception_links``), and theirs in turn. A rule the agent cannot see (observe-only) is not shown as an
        exception either."""
        if not self._exception_links:
            return set()
        by_id = {r.get("id"): r for r in self.rules}
        nested: set = set()
        work = list(entries)
        while work:
            r = work.pop()
            for child in self._exception_links.get(r.get("id"), []):
                c = by_id.get(child)
                if c is None or c is r or id(c) in nested or self._observe_only(c):
                    continue
                nested.add(id(c))
                work.append(c)
        return nested

    def _exception_lines(
        self, rule: Dict[str, Any], nested: set, lines: List[str], reached: set, path: set, indent: str
    ) -> int:
        """The exceptions of one rule under it, each with its own exceptions: under every rule it is an exception of. ``path``
        holds the rules above this one, so a cycle ends. Returns how many lines it wrote."""
        by_id = {r.get("id"): r for r in self.rules}
        count = 0
        for child in self._exception_links.get(rule.get("id"), []):
            c = by_id.get(child)
            if c is None or id(c) not in nested or id(c) in path:
                continue
            reached.add(id(c))
            lines.append(f"{indent}- exception [{c.get('source', '')}]: {self._describe(c)}")
            count += 1 + self._exception_lines(c, nested, lines, reached, path | {id(c)}, indent + "  ")
        return count

    @staticmethod
    def _describe(r: Dict[str, Any]) -> str:
        t = r["type"]
        when = r["trigger"].get("when") if t == "duty" else r.get("when")
        cond = (" when " + ", ".join(f"{k} is {_joined('/', v)}" for k, v in when.items())) if when else ""
        if t == "required_args":
            return f"`{r['tool']}`{cond}: must include {_joined(', ', r.get('args'))}"
        if t == "allowed_values":
            return f"`{r['tool']}`{cond}: {r['arg']} must be one of {_joined(', ', r.get('values'))}"
        if t == "derived_value":
            return f"`{r['tool']}`: {r['arg']} is computed from {_joined(' x ', r.get('inputs'))}; do not guess it"
        if t == "transition":
            edges = "; ".join(f"{k} → {_joined('/', v)}" for k, v in (r.get("edges") or {}).items())
            return f"`{r['tool']}`: {r['arg']} transitions allowed: {edges}"
        if t == "reference_status":
            return (
                f"`{r['tool']}`: {r['arg']} must have {r.get('field')} in {_joined(', ', r.get('allowed'))}"
            )
        if t == "membership":
            return f"`{r['tool']}`: {r['arg']} must be a member of the {r['group_arg']} given"
        if t == "role_permission":
            return f"roles {_joined(', ', r.get('roles'))} may not call {_joined(', ', r.get('tools'))}"
        if t == "grounded_values":
            return "do not set optional arguments unless the request or a tool result gives the value"
        if t == "id_provenance":
            return "use only IDs returned by tools or given in the request"
        if t == "duty":
            th = r["then"]
            tw = (
                (" with " + ", ".join(f"{k}={_joined('/', v)}" for k, v in th.get("when", {}).items()))
                if th.get("when")
                else ""
            )
            return f"after `{r['trigger']['tool']}`{cond}: also call `{th['tool']}`{tw}"
        if t == "record_condition":
            return (
                f"{', '.join('`%s`' % x for x in r.get('tools', []))}{cond}: not for a record whose "
                f"{r.get('field')} {r.get('op')} {r.get('value')!r}"
            )
        if t == "threshold":
            then = "needs approval first" if r.get("then") == "requires_approval" else "is not allowed"
            return f"`{r['tool']}`: {r['arg']} {r.get('op')} {r.get('value')} {then}"
        if t == "format":
            return f"`{r['tool']}`: {r['arg']} must match {r.get('pattern')}"
        if t == "verify":
            return r.get("statement", "")
        return r.get("text", "")
