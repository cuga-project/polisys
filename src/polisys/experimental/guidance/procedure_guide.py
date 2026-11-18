"""Procedure guide: uses the SOP's procedure graph (offline intake) while the agent works.

Input: the ``procedures`` of a build (``handlers/procedures.py`` + ``handlers/procedure_rules.py``): steps with the tool
that does them, preconditions, exceptions, templates and escalations, each with a verbatim quote; plus the "General
rules" entry for rules outside the procedures.

- A write belongs to a procedure when the tool is one of its steps. When several procedures use the same tool (a Jira
  ticket for onboarding and for offboarding), the fixed text of a step template decides ("ONBOARD-" vs "OFFBOARD-"),
  or an address or #channel in the call that the procedure's text names. A code or shell tool identifies a procedure
  only through a template or an address (it is used for reading too).
- Precondition gate: the first write of a procedure is stopped once with the procedure's preconditions and exceptions
  (and the general rules), so the agent confirms them for the case before it starts. It calls the tool again when
  they hold, or follows the exception.
- Template check: a write whose text starts like a template of its step but does not match the template is stopped
  once, with the expected format.
- Evidence check (with a model): when the agent calls again after the gate, preconditions that need proof
  (authorization, approval, signature) are checked on the data it read; without evidence the call stays stopped,
  at most twice per procedure.
- General rules: the first write that belongs to no procedure is stopped once with the SOP's general rules.
A stop is a ``before_tool_call`` message (the call does not run). Each stop happens at most once per procedure (gate)
or per template, so a wrong judgment costs one round. Nothing here is specific to a document or a benchmark.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from polisys.hooks import AgentHooks, HookResult

from .llm import parse_json
from .model import step_text
from .trace import audited_failure, emit

PLACEHOLDER = re.compile(r"\[[^\]]+\]|\{[^}]+\}|<[^>]+>")
TAG = re.compile(
    r"^\s*\[[A-Z]{2,}(?:\s+[A-Z]{2,})+\]"
)  # a leading "[ORDER REQUEST]" is fixed text; "[CASE_ID]" is a field
FRAGMENT_MIN = 8  # a fixed fragment this long that no other template has also identifies the template
GENERIC_TOOL = re.compile(
    r"execute|bash|python|shell|code|script", re.I
)  # runs anything: never identifies a procedure alone
TARGET_ARG = re.compile(r"^(to|cc|bcc|recipients?|channel|channel_name|channel_id|email|assignee)$", re.I)
ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|#[\w-]{3,}")  # e-mail addresses and #channels
MAX_ITEMS = 8
EVIDENCE_WORDS = re.compile(
    r"authori[sz]|approv|written|signed|signature|consent", re.I
)  # proof from someone
HARD = re.compile(
    r"^\s*(stop|hold|do not|don't|halt|escalate)", re.I
)  # the SOP's answer when the precondition fails
EVIDENCE_DENIES = 2  # a precondition without evidence keeps the call stopped at most this often (no deadlock)
READ_CHARS, EVIDENCE_CHARS = 20_000, 24_000
NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
EV_SYSTEM = (
    "You check whether a precondition from a company SOP is met, using only the given data. Output JSON only."
)
EV_USER = """Precondition: {condition}
SOP text: "{quote}"
Action about to happen: {action}

Data the agent has read:
<<<
{data}
>>>

Is there evidence in this data that the precondition holds for this case? Be literal: if the SOP requires something
written, signed or from a specific person, the evidence must be such a document or a message from that person (check
the sender), not a report that the person was involved.
Return {{"answer": "yes|no|unclear", "evidence": "<the data's words that show it, or what is missing>"}}"""


def _strings(v: Any) -> List[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [s for x in v.values() for s in _strings(x)]
    if isinstance(v, list):
        return [s for x in v for s in _strings(x)]
    return []


def _placeholders(template: str) -> List[re.Match]:
    tag = TAG.match(template)
    return [m for m in PLACEHOLDER.finditer(template) if not (tag and m.start() < tag.end())]


def literal_head(template: str) -> str:
    """The fixed text before the first placeholder ("ONBOARD-" of "ONBOARD-[Last Name]-[Start Date MMDDYYYY]";
    "[ORDER REQUEST] Acme |" of "[ORDER REQUEST] Acme | [Customer Last Name] ...")."""
    ms = _placeholders(template)
    head = (template[: ms[0].start()] if ms else template).strip()
    return head if len(head) >= 4 else ""


def fixed_parts(template: str) -> List[str]:
    """The fixed text between placeholders, stripped ("| MemberID:" ...)."""
    out, pos = [], 0
    for m in _placeholders(template) + [None]:
        part = template[pos : m.start() if m else len(template)].strip()
        if part:
            out.append(part)
        pos = m.end() if m else pos
    return out


def template_regex(template: str) -> Optional[re.Pattern]:
    """A template as a pattern: fixed text literally, placeholders as any text; dates by their written form."""
    if not _placeholders(template):
        return None
    out, pos = [], 0
    for m in _placeholders(template):
        out.append(re.escape(template[pos : m.start()]))
        ph = m.group(0).upper()
        if "MMDDYYYY" in ph:
            out.append(r"\d{8}")
        elif "MM/DD/YYYY" in ph:
            out.append(r"\d{1,2}/\d{1,2}/\d{4}")
        elif "YYYY-MM-DD" in ph:
            out.append(r"\d{4}-\d{2}-\d{2}")
        else:
            out.append(r".+?")
        pos = m.end()
    out.append(re.escape(template[pos:]))
    try:
        return re.compile("".join(out).replace(r"\ ", r"\s*"), re.I | re.S)
    except re.error:
        return None


class ProcedureGuide(AgentHooks):
    def __init__(
        self,
        procedures: List[Dict[str, Any]],
        gate: bool = True,
        templates: bool = True,
        model: Optional[Any] = None,
        is_write: Optional[Any] = None,
    ):
        """``model``: a (system, user) -> text callable for evidence checks of preconditions (off when None)."""
        self.procs = [p for p in procedures if p.get("steps")]
        self.general = next((p for p in procedures if not p.get("steps")), None)
        self.gate_on, self.templates_on = gate, templates
        self.model = model
        if is_write is None:
            from .ruleset import is_write_tool as is_write
        self.is_write = is_write
        self.reads: List[str] = []
        self.evidence: Dict[str, int] = {}  # procedure id -> evidence stops so far
        self.general_gated = False
        self.gated: set = set()
        self.template_stopped: set = set()
        self.records: List[Dict[str, Any]] = []

    @classmethod
    def from_package(cls, pkg: Any, **kwargs: Any) -> "ProcedureGuide":
        """A guide over the procedures of a ``PolicyPackage``. The guide finds a write's procedure by the tool of a step,
        so a package from a store graph (its steps name no tool) gives no procedure stops."""
        from .package import procedure_dicts

        return cls(procedure_dicts(pkg), **kwargs)

    def _step_templates(self, p: Dict[str, Any], tool: str) -> List[Dict[str, Any]]:
        idx = {i for i, s in enumerate(p.get("steps") or [], 1) if s.get("action") == tool}
        out = []
        for t in p.get("templates") or []:
            step = str(t.get("step") or "").strip()
            if (step.isdigit() and int(step) in idx) or not step.isdigit():
                out.append(t)
        return out

    def _which(
        self, tool: str, args: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
        """Procedures this write belongs to, with the template whose fixed text it carries (if any)."""
        texts = [s.lower() for s in _strings(args)]
        cands = [p for p in self.procs if any(s.get("action") == tool for s in p.get("steps") or [])]
        hit = []
        for p in cands:
            for t in self._step_templates(p, tool):
                head = literal_head(str(t.get("template") or "")).lower()
                if head and any(head in x for x in texts):
                    hit.append((p, t))
                    break
        if hit:
            return hit
        for p in cands:  # no fixed head: a long fixed fragment that only this template has
            for t in self._step_templates(p, tool):
                frags = [
                    f.lower() for f in fixed_parts(str(t.get("template") or "")) if len(f) >= FRAGMENT_MIN
                ]
                if any(f in x for f in frags for x in texts) and all(
                    self._frag_count(f) == 1 for f in frags if any(f in x for x in texts)
                ):
                    hit.append((p, t))
                    break
        if hit:
            return hit
        target = [
            str(v).lower() for k, v in args.items() if TARGET_ARG.search(str(k))
        ]  # recipients and channels only
        addrs = {a.lower() for x in target for a in ADDRESS.findall(x)}
        if addrs:  # a shared tool: the procedure whose text names the same address or channel
            named = [p for p in self.procs if addrs & {a.lower() for a in ADDRESS.findall(self._text(p))}]
            if len(named) == 1 and named[0] in cands:  # an address several procedures name decides nothing
                return [(named[0], None)]
        if GENERIC_TOOL.search(tool):
            return []
        return [(cands[0], None)] if len(cands) == 1 else []

    def _frag_count(self, frag: str) -> int:
        """In how many templates (of all procedures) a fixed fragment occurs."""
        return sum(
            frag in str(t.get("template") or "").lower() for p in self.procs for t in p.get("templates") or []
        )

    @staticmethod
    def _text(p: Dict[str, Any]) -> str:
        parts = [step_text(s) for s in p.get("steps") or []]
        for k in ("templates", "preconditions", "exceptions", "escalations"):
            parts += [json.dumps(x, ensure_ascii=False) for x in p.get(k) or []]
        return " ".join(parts)

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> HookResult:
        if not self.is_write(tool) or GENERIC_TOOL.search(
            tool
        ):  # what the agent read (code tools often read)
            text = result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False)
            self.reads.append(f"{tool}({json.dumps(args, default=str)[:200]}) -> {text[:READ_CHARS]}")
        return HookResult()

    def _evidence_check(self, p: Dict[str, Any], tool: str, args: Dict[str, Any]) -> Optional[HookResult]:
        """On a retry after the gate: preconditions that need proof are checked on the data read."""
        if self.model is None or self.evidence.get(p["id"], 0) >= EVIDENCE_DENIES:
            return None
        needs = [
            c
            for c in p.get("preconditions") or []
            if HARD.search(str(c.get("if_not") or "stop"))
            and EVIDENCE_WORDS.search(f"{c.get('condition')} {c.get('quote')}")
        ]
        if not needs:
            return None
        names = {n for x in _strings(args) for n in NAME.findall(x)}
        rel = [r for r in self.reads if any(n in r for n in names)] or self.reads[-10:]
        data = "\n\n".join(rel)[-EVIDENCE_CHARS:]
        action = f"{tool}({json.dumps(args, default=str, ensure_ascii=False)[:300]})"
        for c in needs:
            try:
                ans = parse_json(
                    self.model(
                        EV_SYSTEM,
                        EV_USER.format(
                            condition=c.get("condition"),
                            quote=c.get("quote"),
                            action=action,
                            data=data or "(nothing read)",
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - no answer: do not stop the agent
                emit(  # the check is skipped and the write goes on: the audit says so (the failed call has its own line)
                    self,
                    {
                        "hook": "before_tool_call",
                        "tool": tool,
                        "rule": "procedure_evidence",
                        "outcome": "error",
                        "reason": f"{type(exc).__name__}: {exc}; the check of \"{c.get('condition')}\" was skipped",
                        "procedure": p.get("name"),
                    },
                )
                continue
            rec = {
                "hook": "before_tool_call",
                "tool": tool,
                "rule": "procedure_evidence",
                "procedure": p.get("name"),
                "condition": c.get("condition"),
                "answer": ans.get("answer"),
                "evidence": str(ans.get("evidence"))[:300],
            }
            if ans.get("answer") == "no":
                self.evidence[p["id"]] = self.evidence.get(p["id"], 0) + 1
                rec["outcome"] = "deny"
                self.records.append(rec)
                return HookResult(
                    message=f"SOP CHECK FAILED ({p.get('name')}): \"{c.get('condition')}\" is not shown by the "
                    f"data you read ({str(ans.get('evidence'))[:300]}). The SOP says: if not, "
                    f"{c.get('if_not') or 'stop'} [SOP: \"{str(c.get('quote', ''))[:160]}\"]. Do not "
                    f"continue this procedure for the case unless you find that evidence.",
                    details=[rec],
                )
            rec["outcome"] = "allow"
            self.records.append(rec)
        return None

    def _general_gate(self, tool: str) -> Optional[HookResult]:
        g = self.general
        if self.general_gated or not self.gate_on or not g or GENERIC_TOOL.search(tool):
            return None
        items = (g.get("preconditions") or []) + (g.get("exceptions") or [])
        if not items:
            return None
        self.general_gated = True
        lines = [
            "SOP CHECK before this action - the SOP's general rules (outside its procedures); confirm none applies:"
        ]
        for x in items[: 2 * MAX_ITEMS]:
            lines.append(
                f"- If {x.get('when') or x.get('condition')} then {x.get('then') or x.get('if_not')} "
                f"[SOP: \"{str(x.get('quote', ''))[:120]}\"]"
            )
        lines.append(f"If none applies, call {tool} again. If one applies, do what the SOP says instead.")
        rec = {"hook": "before_tool_call", "tool": tool, "rule": "procedure_general", "outcome": "deny"}
        self.records.append(rec)
        return HookResult(message="\n".join(lines), details=[rec])

    @audited_failure
    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> HookResult:
        if not self.is_write(tool):
            return HookResult()
        found = self._which(tool, args)
        if not found:
            res = self._general_gate(tool)
            return res if res is not None else HookResult()
        for p, tpl in found:
            if p["id"] in self.gated:
                res = self._evidence_check(p, tool, args)
                if res is not None:
                    return res
            if self.templates_on and tpl is not None:
                rx = template_regex(str(tpl.get("template") or ""))
                key = (p["id"], tpl.get("template"))
                if (
                    rx is not None
                    and key not in self.template_stopped
                    and not any(rx.search(s) for s in _strings(args))
                ):
                    self.template_stopped.add(key)
                    rec = {
                        "hook": "before_tool_call",
                        "tool": tool,
                        "rule": "procedure_template",
                        "outcome": "deny",
                        "procedure": p.get("name"),
                        "template": tpl.get("template"),
                    }
                    self.records.append(rec)
                    return HookResult(
                        message=f"SOP TEMPLATE ({p.get('name')}): {tpl.get('artifact', 'this output')} must "
                        f"follow \"{tpl.get('template')}\" [SOP: \"{str(tpl.get('quote', ''))[:160]}\"]. "
                        f"Fix the value and call {tool} again.",
                        details=[rec],
                    )
            if self.gate_on and p["id"] not in self.gated and (p.get("preconditions") or p.get("exceptions")):
                self.gated.add(p["id"])
                lines = [
                    f"SOP CHECK before you start \"{p.get('name')}\" for this case - confirm from the data you have:"
                ]
                for c in (p.get("preconditions") or [])[:MAX_ITEMS]:
                    lines.append(
                        f"- Must hold: {c.get('condition')} (if not: {c.get('if_not') or 'stop'}) "
                        f"[SOP: \"{str(c.get('quote', ''))[:140]}\"]"
                    )
                for x in (p.get("exceptions") or [])[:MAX_ITEMS]:
                    lines.append(
                        f"- Exception: if {x.get('when')} then {x.get('then')} [SOP: \"{str(x.get('quote', ''))[:140]}\"]"
                    )
                if self.general:
                    for x in (
                        (self.general.get("preconditions") or []) + (self.general.get("exceptions") or [])
                    )[:4]:
                        lines.append(
                            f"- General rule: if {x.get('when') or x.get('condition')} then "
                            f"{x.get('then') or x.get('if_not')} [SOP: \"{str(x.get('quote', ''))[:120]}\"]"
                        )
                lines.append(
                    f"If they hold, call {tool} again. If not, follow the SOP instead (stop, hold, forward or "
                    f"escalate) and do not start this procedure for the case."
                )
                rec = {
                    "hook": "before_tool_call",
                    "tool": tool,
                    "rule": "procedure_gate",
                    "outcome": "deny",
                    "procedure": p.get("name"),
                }
                self.records.append(rec)
                return HookResult(message="\n".join(lines), details=[rec])
        return HookResult()


def guide_from_build(path: str, **kw: Any) -> ProcedureGuide:
    data = json.loads(open(path).read())
    build = data.get("build", data)
    return ProcedureGuide(build.get("procedures") or [], **kw)
