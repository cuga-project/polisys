"""Procedure store: an SOP compiled into a procedure graph, queried by the agent through policy tools.

The store holds one SOP as a graph (JSON): global rules, a triage procedure, a routing table (category → ticket
name, assignee, folder, channel), housekeeping, priority and escalation rules, procedures (when they apply, data
inputs, decisions, ordered steps, deadlines, must-not rules, typed links to other procedures), exact templates with
their slots, definitions and a business-day calendar.

The agent reads the SOP through four tools instead of searching a long document:

- ``policy__list_procedures``: the general rules, triage, routing and one line per procedure;
- ``policy__get_procedure``: the procedures for a case, with the procedures they link to (``next`` links) and the
  full text of every template they name;
- ``policy__get_template``: exact template text and its placeholders;
- ``policy__add_days``: date arithmetic in business or calendar days with the SOP's holidays.

Nothing here names a domain or benchmark: the content comes from the graph file. The graph can be built by hand,
by a strong model, or by the offline intake.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any, Dict, Iterable, List, Optional

from .model import step_text

TEMPLATE_REF = re.compile(r"Template\s+([A-Z]-\d+|\d+)")


class ProcedureStore:
    def __init__(self, graph: Dict[str, Any]):
        self.g = graph
        self.procs = {p["id"]: p for p in graph.get("procedures") or []}
        self.templates = {str(k): v for k, v in (graph.get("templates") or {}).items()}
        cal = graph.get("calendar") or {}
        self.workdays = set(cal.get("workdays") or [0, 1, 2, 3, 4])
        self.holidays = {dt.date.fromisoformat(d) for d in cal.get("holidays") or []}
        self.calls: List[Dict[str, Any]] = []  # what the agent asked for (for verifiers and traces)

    @classmethod
    def load(cls, path: str) -> "ProcedureStore":
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def from_package(cls, pkg: Any) -> "ProcedureStore":
        """The store of a ``PolicyPackage``. A package from a store graph reads the same text as that graph."""
        from .package import to_store_graph

        return cls(to_store_graph(pkg))

    # ── reading ─────────────────────────────────────────────────────────────

    @staticmethod
    def _block(title: str, lines: Iterable[str]) -> List[str]:
        lines = [str(x) for x in lines if x]
        return [f"## {title}"] + [f"- {x}" for x in lines] if lines else []

    def list_procedures(self) -> str:
        g = self.g
        out = [f"# SOP {g.get('id', '')} as a procedure graph"]
        out += self._block(
            "General rules (apply to every case)",
            [f"[{r['src']}] {r['rule']}" for r in g.get("global_rules") or []],
        )
        for key in ("triage", "housekeeping", "escalation"):
            node = g.get(key) or {}
            if node:
                out += self._block(
                    f"{node.get('name', key)} [{node.get('src', '')}]", node.get("steps") or []
                )
                out += [f"- MUST NOT: {m}" for m in node.get("must_not") or []]
        pr = g.get("priority") or {}
        if pr:
            out += self._block(f"{pr.get('name', 'Priority')} [{pr.get('src', '')}]", pr.get("levels") or [])
        if g.get("routing"):
            out += ["## Routing by category (ticket name, assignee, folder, channel)"]
            for cat, r in g["routing"].items():
                out.append(f"- {cat}: " + "; ".join(f"{k}: {v}" for k, v in r.items() if v))
        out += ["## Procedures (call policy__get_procedure with the ids that apply to each case)"]
        for p in self.procs.values():
            out.append(f"- {p['id']} {p['name']} ({p.get('category', '')}): {p.get('applies_when', '')}")
        if g.get("definitions"):
            out += self._block("Definitions", [f"{k}: {v}" for k, v in g["definitions"].items()])
        return "\n".join(out)

    def _expand(self, ids: List[str]) -> List[str]:
        seen: List[str] = []
        for pid in ids:
            for x in [pid] + [
                lk["to"] for lk in (self.procs.get(pid) or {}).get("links") or [] if lk.get("type") == "next"
            ]:
                if x in self.procs and x not in seen:
                    seen.append(x)
        return seen

    @classmethod
    def _parts(cls, p: Dict[str, Any]) -> List[str]:
        """Preconditions, exceptions, templates and escalations of a procedure, when it has any (a package from a skill or
        a build has them; a hand-made graph has none, so its text does not change). A part that has no fields is shown by
        its quote."""

        def line(item: Any, fields: List[str]) -> str:
            if not isinstance(item, dict):
                return str(item)
            known = [str(item[k]) for k in fields if item.get(k)]
            return " | ".join(known) if known else str(item.get("quote") or "")

        out: List[str] = []
        for key, title, fields in (
            ("preconditions", "Preconditions (before you start)", ["condition", "if_not"]),
            ("exceptions", "Exceptions", ["when", "then"]),
            ("templates", "Output formats", ["artifact", "template"]),
            ("escalations", "Escalations", ["when", "notify", "channel"]),
        ):
            out += cls._block(title, [line(x, fields) for x in p.get(key) or []])
        return out

    def procedure_text(self, p: Dict[str, Any], linked_from: Optional[str] = None) -> str:
        head = f"# {p['id']} {p['name']} [{p.get('src', '')}]" + (
            f" (linked from {linked_from})" if linked_from else ""
        )
        out = [head, f"Applies when: {p.get('applies_when', '')}"]
        if p.get("priority") not in (None, "", [], {}):  # a graph that has none gives the text it always gave
            out.append(f"Priority: {p['priority']}")
        route = (self.g.get("routing") or {}).get(p.get("category") or "")
        if route:
            out.append("Routing: " + "; ".join(f"{k}: {v}" for k, v in route.items() if v))
        out += self._block(
            "Data to read", [f"{i.get('fact')}: {i.get('source')}" for i in p.get("inputs") or []]
        )
        out += self._block(
            "Decisions",
            [
                f"{d.get('id')}: {d.get('rule')}"
                + (f" → {d['then']}" if d.get("then") else "")
                + (f"; else → {d['else']}" if d.get("else") else "")
                for d in p.get("decisions") or []
            ],
        )
        out += self._block("Steps", [step_text(s) for s in p.get("steps") or []])
        out += self._parts(p)
        out += self._block("Deadlines", p.get("deadlines") or [])
        out += self._block("MUST NOT", p.get("must_not") or [])
        out += self._block("Notes", p.get("notes") or [])
        text = json.dumps(p)
        attached = [
            f"[{r['src']}] {r['rule']}"
            for r in self.g.get("global_rules") or []
            if r.get("attach_when") and re.search(r["attach_when"], text, re.I)
        ]
        out += self._block("General rules that apply to these steps", attached)
        nxt = [
            f"{lk['to']}" + (f" when {lk['when']}" if lk.get("when") else "") for lk in p.get("links") or []
        ]
        if nxt:
            out.append("Linked procedures: " + ", ".join(nxt))
        return "\n".join(out)

    def get_procedure(self, ids: List[str]) -> str:
        ids = [str(i).strip().lstrip("§") for i in ids]
        unknown = [i for i in ids if i not in self.procs]
        chosen = self._expand([i for i in ids if i in self.procs])
        self.calls.append({"tool": "get_procedure", "ids": ids, "expanded": chosen})
        if not chosen:
            return (
                "No such procedure ids: " + ", ".join(unknown) + ". Call policy__list_procedures for the ids."
            )
        parts = [
            self.procedure_text(self.procs[i], None if i in ids else "a requested procedure") for i in chosen
        ]
        nums = sorted(
            {n for i in chosen for n in TEMPLATE_REF.findall(json.dumps(self.procs[i]))},
            key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else 0, n),
        )
        if nums:
            parts.append(
                "# Templates named above (exact text; fill every [placeholder]; personalize the greeting only)"
            )
            parts += [self.template_text(n) for n in nums]
        if unknown:
            parts.append("Unknown ids ignored: " + ", ".join(unknown))
        return "\n\n".join(parts)

    def template_text(self, n: Any) -> str:
        key = re.sub(r"(?i)^\s*template\s*", "", str(n)).strip()
        t = self.templates.get(key) or {k.lower(): v for k, v in self.templates.items()}.get(key.lower())
        if not t:
            return f"Template {n}: not in the SOP."
        return f"## Template {t['number']} – {t['name']}\nPlaceholders: {', '.join(t.get('slots') or [])}\n---\n{t['text']}\n---"

    def get_template(self, numbers: List[Any]) -> str:
        self.calls.append({"tool": "get_template", "numbers": numbers})
        return "\n\n".join(self.template_text(n) for n in numbers)

    # ── dates ───────────────────────────────────────────────────────────────

    def is_business_day(self, d: dt.date) -> bool:
        return d.weekday() in self.workdays and d not in self.holidays

    def add_days(self, date: str, days: int, unit: str = "business") -> str:
        d = _parse_date(date)
        if d is None:
            return f"Could not read the date {date!r}; use MM/DD/YYYY."
        if unit.startswith("cal"):
            r = d + dt.timedelta(days=days)
        else:
            r, step, left = d, 1 if days >= 0 else -1, abs(days)
            while left:
                r += dt.timedelta(days=step)
                if self.is_business_day(r):
                    left -= 1
        note = "" if self.is_business_day(r) else " (not a business day)"
        return f"{r.strftime('%m/%d/%Y')} ({r.strftime('%A')}){note}"

    def required_steps(self) -> Dict[str, List[str]]:
        """Steps of every procedure the agent opened, for an end verifier."""
        ids = []
        for c in self.calls:
            ids += [i for i in c.get("expanded") or [] if i not in ids]
        return {
            i: [step_text(s) for s in self.procs[i].get("steps") or []]
            + [f"MUST NOT: {m}" for m in self.procs[i].get("must_not") or []]
            for i in ids
        }


def _parse_date(s: str) -> Optional[dt.date]:
    s = (s or "").strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def policy_tools(store: ProcedureStore) -> List[Any]:
    """LangChain tools over a ProcedureStore (app ``policy``)."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _Ids(BaseModel):
        ids: List[str] = Field(
            description="Procedure ids from policy__list_procedures, e.g. ['13.1'] or ['12.1']"
        )

    class _Nums(BaseModel):
        numbers: List[str] = Field(description="Template numbers, e.g. ['15', '17']")

    class _Days(BaseModel):
        date: str = Field(description="Start date, MM/DD/YYYY")
        days: int = Field(description="Number of days to add (negative to subtract)")
        unit: str = Field(
            default="business", description="'business' (skips weekends and SOP holidays) or 'calendar'"
        )

    return [
        StructuredTool.from_function(
            func=lambda: store.list_procedures(),
            name="policy__list_procedures",
            description="The SOP as a procedure graph: general rules, triage, routing, housekeeping, escalation, priority, and the list of procedures with when each applies. Call this first.",
        ),
        StructuredTool.from_function(
            func=lambda ids: store.get_procedure(ids),
            name="policy__get_procedure",
            description="Full procedures for a case (data to read, decisions, ordered steps, deadlines, must-not rules), including the procedures they link to and the exact text of the templates they use.",
            args_schema=_Ids,
        ),
        StructuredTool.from_function(
            func=lambda numbers: store.get_template(numbers),
            name="policy__get_template",
            description="Exact text and placeholders of SOP templates.",
            args_schema=_Nums,
        ),
        StructuredTool.from_function(
            func=lambda date, days, unit="business": store.add_days(date, days, unit),
            name="policy__add_days",
            description="Add business days (weekends and SOP holidays skipped) or calendar days to a date.",
            args_schema=_Days,
        ),
    ]


POLICY_TOOLS_NOTE = (
    "The SOP is also available as a procedure graph through the policy__ tools. Call policy__list_procedures once "
    "at the start; for each case call policy__get_procedure with the ids that apply and follow its steps, decisions "
    "and must-not rules; use the exact template text it returns; use policy__add_days for business-day deadlines."
)
