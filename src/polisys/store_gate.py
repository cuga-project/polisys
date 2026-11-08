"""Must-not check at write time over a procedure store: before an acting tool call, ask whether it breaks a
must-not rule or a decision of the procedures the agent opened, given the data the agent read.

A strong agent can have the rule in front of it and still decide wrongly ("they were present at the meeting"
read as "written authorization"). This hook asks a separate model one narrow question per acting call. A verdict
blocks only when its evidence is quoted from data the agent read (checked in code), and each (rule, case) blocks
once: the agent may then explain why the rule does not apply and act.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Set, Tuple

from polisys.hooks import AgentHooks

from .llm import parse_json
from .tool_kind import acts as _acts
from .trace import audited_failure

WINDOW = re.compile(
    r"within (?:the )?(?:last |past |preceding )?(\d+) (calendar days|business days|days|months|weeks|years)",
    re.I,
)
SYSTEM = "You check one action of an agent against SOP rules. Output JSON only."
# checkers.gate.missing_needs_read: a rule broken because something is missing counts only when the record that should hold
# it was read (an absence cannot be quoted, so the quote check alone does not stop such a block).
MISSING_NEEDS_READ = (
    " If the rule is broken because something is missing (an approval, a document, an attachment, a field), answer"
    ' "yes" only if the data shows that the record which should hold it was read and does not hold it; if that record'
    ' was not read, answer "no".'
)
USER = """{label}:
{rules}
{dates}
Data the agent read (most recent last, may be cut):
<<<
{data}
>>>

The agent is about to do this action:
{action}

Does this action break one of the rules above for the case it belongs to? Answer "yes" only if a rule clearly
forbids it given the data (for example a required approval or authorization is missing, or the SOP says not to
reply). Quote the words from the data that show it.{extra} Return
{{"breaks": true|false, "rule": "<the rule>", "case": "<who/what>", "evidence": "<exact quote from the data>",
"why": "<short>"}}"""


def _parse_day(s: str) -> Any:
    import datetime as dt

    s = str(s or "").strip()[:10]
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return None


def _minus(day: Any, n: int, unit: str, store: Any) -> Any:
    """The first day of a window of n units ending today."""
    import datetime as dt

    if unit.startswith("month"):
        m = day.month - n
        y, m = day.year + (m - 1) // 12, (m - 1) % 12 + 1
        import calendar

        return dt.date(y, m, min(day.day, calendar.monthrange(y, m)[1]))
    if unit.startswith("year"):
        return day.replace(year=day.year - n)
    if unit.startswith("week"):
        return day - dt.timedelta(weeks=n)
    if unit.startswith("business") and store is not None and hasattr(store, "is_business_day"):
        d, left = day, n
        while left:
            d -= dt.timedelta(days=1)
            if store.is_business_day(d):
                left -= 1
        return d
    return day - dt.timedelta(days=n)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _words(s: str) -> List[str]:
    """Words of a text with JSON escapes removed, so a quote matches data that was read as escaped JSON."""
    s = re.sub(r"\\u[0-9a-fA-F]{4}|\\[ntr]", " ", str(s or ""))
    return re.findall(r"[a-z0-9]+", s.lower())


def _seq_in(words: List[str], hay: str) -> bool:
    return len(words) >= 4 and f" {' '.join(words)} " in hay


def quoted_in(evidence: str, data: str, n: int = 12, policy: str = "") -> bool:
    """The evidence is a quote from the case data: its first n words, or the first n words of a quoted span in it
    ("..." or “...”), occur in a row in the data (at least 4 words). A span that is also in the policy text does not
    count: a sentence of the SOP is a rule, not evidence about the case (the agent may have read the SOP file)."""
    hay = f" {' '.join(_words(data))} "
    pol = f" {' '.join(_words(policy))} " if policy else ""
    spans = [evidence] + re.findall(r'"([^"]{12,})"|\u201c([^\u201d]{12,})\u201d', evidence)
    for sp in spans:
        sp = sp if isinstance(sp, str) else next((x for x in sp if x), "")
        ws = _words(sp)[:n]
        if _seq_in(ws, hay) and not (pol and _seq_in(ws, pol)):
            return True
    return False


def _has_gate_rules(pkg: Any) -> bool:
    """The package holds what ``StoreGate._rules`` reads from a store."""
    escalation = pkg.global_.get("escalation") or {}
    return bool(escalation.get("must_not")) or any(p.must_not or p.decisions for p in pkg.procedures)


def gate_verdict(rec: Dict[str, Any]) -> str:
    """What the gate did for one record of a check. It blocks when the model says the action breaks a rule, the quote it
    gives is in the case data, and the same rule for the same case was not blocked before. The audit reads this too."""
    if not rec.get("breaks"):
        return "allow"
    if not rec.get("grounded"):
        return "allow_ungrounded"
    if rec.get("repeat"):
        return "allow_repeat"
    return "block"


class StoreGate(AgentHooks):
    def __init__(
        self,
        store: Any,
        model: Callable[[str, str], str],
        data_chars: int = 24000,
        today: str = "",
        sop_text: str = "",
        scope: Any = None,
    ):
        """``store``: rules from the procedures the agent opened; or ``store=None`` and ``sop_text``: the whole SOP.
        ``scope``: a ``router.ScopeReader``, which says what is in scope. With ``store`` and no ``scope``, the scope is the
        procedures the agent opened with their ``next`` links (the measured one), read from the store's own graph."""
        self.store, self.model, self.data_chars = store, model, data_chars
        self._scope_reader = scope  # not a public attribute: the description of a gate (dry_run) lists those, and it is not wiring
        self.sop_text = sop_text
        if scope is not None and not scope.is_default:
            self.scope_names = (
                scope.names
            )  # only a method that is not the measured one: it shows in the wiring description
        self._scope_note = (
            ""  # the procedures in scope of the last check, for the audit, when the method is not the default
        )
        self.today = _parse_day(today)
        self.reads: List[str] = []
        self.blocked: Set[Tuple[str, str]] = set()  # (rule, case): the gate stops a rule once for each case
        self.records: List[Dict[str, Any]] = []
        self._last = ""  # what the last check did (see ``last_decision``)
        self.extra = ""  # added to the question (MISSING_NEEDS_READ); empty keeps the measured prompt

    @classmethod
    def from_package(
        cls,
        pkg: Any,
        model: Callable[[str, str], str],
        store: Any = None,
        sop_text: str = "",
        router: Any = None,
        expander: Any = None,
        **kwargs: Any,
    ) -> "StoreGate":
        """A gate over a ``PolicyPackage``. Give one of:

        - ``store``: the store behind the agent's ``policy_tools``. The gate learns which procedures the agent opened from
          ``store.calls``, so a store of its own would never see them and the gate would not check anything;
        - ``sop_text``: the gate checks that whole text, as the constructor does with ``store=None``.

        ``router`` and ``expander`` (``router.py``) say which procedures and rules are in scope; the defaults are the
        procedures the agent opened and their ``next`` links.

        Raises ``ValueError`` if neither or both are given, or if with ``store`` the package has nothing the gate can
        check: no must-not rule or decision in a procedure, and no escalation must-not."""
        if (store is None) == (not sop_text):
            raise ValueError("give the store behind the agent's policy_tools, or sop_text: one of them")
        if store is not None and not _has_gate_rules(pkg):
            raise ValueError("the package has no must-not rules, decisions or escalation must-nots to check")
        if store is not None:
            from .router import ScopeReader, make_expander, make_router

            kwargs["scope"] = ScopeReader(
                router or make_router("agent_opened"), expander or make_expander("next_links"), pkg, store
            )
        return cls(store, model, sop_text=sop_text, **kwargs)

    @property
    def last_decision(self) -> str:
        """What the last call of ``before_tool_call`` did: ``n/a`` (not an acting tool, or no rule to check), ``allow``,
        ``block_once`` or ``error`` (the model failed and the call went through)."""
        return self._last

    @property
    def scope(self) -> Any:
        """The ``router.ScopeReader`` of the gate (with a store; ``None`` for a gate over the whole SOP text)."""
        if (
            self._scope_reader is None and self.store is not None
        ):  # a gate made on a store alone: its own graph is the package
            from .package import from_store_graph
            from .router import default_reader

            self._scope_reader = default_reader(from_store_graph(self.store.g), self.store)
        return self._scope_reader

    def _rules(self, query: str = "", action: str = "", context: str = "") -> List[str]:
        """The rules the model checks an action against: from ``Scope`` (``router.py``), or the whole SOP text.
        ``query``: the action and the data read recently, for a router that searches; ``action`` and ``context``: the same,
        apart, for a router that reads a prompt."""
        if self.store is None:
            return [self.sop_text] if self.sop_text else []
        out: List[str] = []
        scope = self.scope.read(query, action, context)
        if not self.scope.is_default:
            self._scope_note = f"{self.scope.names}: {', '.join(scope.procedures) or '(no procedure)'}"
        for r in scope.rules:
            if r.kind == "must_not":
                out.append(f"[{r.owner}] MUST NOT: {r.text}")
            else:
                d = r.decision
                out.append(
                    f"[{r.owner}] DECISION {d.get('id')}: {d.get('rule')}"
                    + (f"; then: {d['then']}" if d.get("then") else "")
                    + f"; else: {d.get('else', '')}"
                )
        return list(dict.fromkeys(out))

    def _date_note(self, rules: List[str]) -> str:
        """Today and the exact start of every 'within N days/months' window named in the rules (code, not the model)."""
        if self.today is None:
            return ""
        out = []
        for n, unit in dict.fromkeys(WINDOW.findall(" ".join(rules))):
            start = _minus(self.today, int(n), unit.lower(), self.store)
            if start:
                out.append(f"- within {n} {unit} before today: on or after {start.isoformat()}")
        if not out:
            return f"\nToday is {self.today.isoformat()}.\n"
        return (
            f"\nToday is {self.today.isoformat()}. Date windows in the rules, computed exactly (a governing date before "
            "the start is OUT of the window; use the date the rule names, e.g. collection date, not report date):\n"
            + "\n".join(out)
            + "\n"
        )

    def _relevant_data(self, action: str) -> str:
        """Reads that name the action's entities (names, IDs, addresses), most matches first, then the recent tail;
        an early read of the case's request must not fall out of the window."""
        keys = set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+|\b[A-Z]{1,6}-\d+\b|\b[A-Z][a-z]{2,}\b", action))
        keys = {k for k in keys if sum(k in r for r in self.reads) <= max(3, len(self.reads) // 3)}
        tail = "\n".join(self.reads)[-self.data_chars // 3 :]
        ranked = sorted(
            (r for r in self.reads if r[-200:] not in tail),
            key=lambda r: -sum(k in r for k in keys),
        )
        picked, size = [], 0
        for r in ranked:
            if not keys or not any(k in r for k in keys) or size + len(r) > self.data_chars * 2 // 3:
                continue
            picked.append(r)
            size += len(r)
        return "\n".join(picked + [tail])

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> Any:
        if not _acts(tool) and not tool.startswith("policy__"):
            self.reads.append(f"{tool}: {str(result)[:6000]}")
        return None

    @audited_failure
    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> Any:
        self._last = "n/a"
        if not _acts(tool) or re.search(r"mark|flag|move|label", tool, re.I):
            return None
        action = f"{tool}({json.dumps(args, default=str)[:1500]})"
        context = "\n".join(self.reads[-2:])[-2000:]
        self._scope_note = ""
        try:
            rules = self._rules(f"{action}\n{context}", action, context)
        except Exception as exc:  # noqa: BLE001 - a scope that fails is a check that failed: the call goes through, and says so
            self.records.append(
                {"gate": "error", "tool": tool, "error": f"scope: {type(exc).__name__}: {exc}"[:200]}
            )
            self._last = "error"
            return None
        if not rules:
            return None
        data = self._relevant_data(action)
        try:
            v = parse_json(
                self.model(
                    SYSTEM,
                    USER.format(
                        label="The SOP the agent must follow (full text)"
                        if self.store is None
                        else "SOP rules of the procedures the agent is following (must-not rules and decisions)",
                        rules="\n".join(rules),
                        dates=self._date_note(rules),
                        data=data,
                        action=action,
                        extra=self.extra,
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed check never blocks
            self.records.append({"gate": "error", "tool": tool, "error": str(exc)[:200]})
            self._last = "error"
            return None
        rec = {
            "gate": "checked",
            "tool": tool,
            "breaks": bool(v.get("breaks")),
            "rule": v.get("rule"),
            "case": v.get("case"),
        }
        if self._scope_note:
            rec["scope"] = self._scope_note
        if not v.get("breaks"):
            self.records.append(rec)
            self._last = "allow"
            return None
        grounded = quoted_in(v.get("evidence") or "", data, policy="\n".join(rules))
        key = (_norm(v.get("rule"))[:80], _norm(v.get("case"))[:40])
        rec.update({"evidence": v.get("evidence"), "grounded": grounded, "repeat": key in self.blocked})
        self.records.append(rec)
        self._last = "allow"
        if gate_verdict(rec) != "block":
            return None
        self._last = "block_once"
        self.blocked.add(key)
        return (
            f"Blocked by an SOP check: this action seems to break the rule \"{v.get('rule')}\" for {v.get('case')}. "
            f"Evidence in the data: \"{v.get('evidence')}\". {v.get('why') or ''} Follow the SOP for this case (hold, "
            "ask for the missing approval, or escalate as the SOP says). If the rule does not apply, say why and retry."
        )
