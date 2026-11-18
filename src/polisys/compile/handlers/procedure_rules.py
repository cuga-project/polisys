"""What must be checked when a procedure is carried out: preconditions, exceptions, templates and escalations.

Tool-keyed rule types cannot hold the parts of an SOP that matter most for following it: "do not start offboarding
without written authorization from X or Y" (a condition on the whole procedure), "if the offer was rescinded, do not
onboard" (an exception), "Jira summary: ONBOARD-[Last Name]-[Start Date MMDDYYYY]" (a template for one step's output),
"if a case is not covered, post in #hr-escalations" (an escalation). One model call per procedure reads the
procedure's section and the sections it refers to, and fills these lists on the ``Procedure``. Every item needs a
verbatim quote (``validate.quote_ok``) or it is dropped. Nothing here is specific to a document or a benchmark.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..llm import parse_json
from ..model import CoverageItem, Procedure, Statement
from ..validate import quote_ok

CONTEXT_CHARS = 14_000
KINDS = ("preconditions", "exceptions", "templates", "escalations")

SYSTEM = "You read one procedure of a company SOP and list what must be checked when someone carries it out. Output JSON only."
USER = """Procedure "{name}", steps:
{steps}

Related policy text (the procedure's section and the sections it refers to):
<<<
{context}
>>>

List only what the text says, each with "quote": the words of the text, copied exactly (one sentence or less):
- "preconditions": what must hold before the procedure starts for a case (approvals, written authorization, eligibility,
  documents received): {{"condition", "if_not": "stop|hold|escalate|ask", "quote"}};
- "exceptions": cases where the procedure must not run, must stop, must be held, or changes:
  {{"when", "then", "quote"}};
- "templates": exact formats that a step's output must follow (ticket summary, email subject or body, message text,
  file or column names), with placeholders such as [Last Name]: {{"step": <number>, "artifact", "template", "quote"}};
- "escalations": who must be notified, and where, when a condition holds: {{"when", "notify", "channel", "quote"}}.
Return {{"preconditions": [], "exceptions": [], "templates": [], "escalations": []}}"""


def _section_key(st: Statement) -> str:
    return f"{st.id.split(':')[0]}:{st.source.get('section', '')}"


PREV_SIBLINGS = 2  # sections right before a procedure in the same chapter often hold its requirements ("before PA submission")


def _before(key: str, by_section: Dict[str, List[Statement]]) -> List[str]:
    """For "doc:8.3": the chapter "doc:8" and the preceding siblings "doc:8.1", "doc:8.2" (those that exist)."""
    doc, _, sec = key.partition(":")
    m = re.fullmatch(r"(\d+(?:\.\d+)*)\.(\d+)", sec)
    if not m:
        return []
    parent, n = m.group(1), int(m.group(2))
    keys = [f"{doc}:{parent}"] + [f"{doc}:{parent}.{i}" for i in range(max(1, n - PREV_SIBLINGS), n)]
    return [k for k in keys if k in by_section]


def enrich_procedures(
    procs: List[Procedure],
    statements: List[Statement],
    links: List[Any],
    model: Optional[Callable[[str, str], str]],
) -> Tuple[List[Procedure], List[CoverageItem]]:
    if model is None or not procs:
        return procs, []
    by_section: Dict[str, List[Statement]] = defaultdict(list)
    for st in statements:
        by_section[_section_key(st)].append(st)
    by_id = {st.id: st for st in statements}
    refs: Dict[str, set] = defaultdict(set)  # section -> sections its statements refer to

    def section_of(ref: str) -> Optional[str]:  # a link end is a statement id or a section id ("doc:section")
        if ref in by_id:
            return _section_key(by_id[ref])
        return ref if ref in by_section else None

    for lk in links:
        kind = str(getattr(getattr(lk, "type", None), "value", getattr(lk, "type", "")))
        src, dst = section_of(str(getattr(lk, "src", ""))), section_of(str(getattr(lk, "dst", "")))
        if kind in ("refers_to", "depends_on", "exception_of") and src and dst:
            refs[src].add(dst)
    cov: List[CoverageItem] = []
    from concurrent.futures import ThreadPoolExecutor

    def one(p: Procedure) -> Tuple[Procedure, List[CoverageItem]]:
        keys = (
            [p.id]
            + [k for k in _before(p.id, by_section) if k not in refs.get(p.id, set())]
            + sorted(refs.get(p.id, set()) - {p.id})
        )
        pool = [st for k in keys for st in by_section.get(k, [])]
        context = "\n".join(f"[§{st.source.get('section', '')}] {st.text}" for st in pool)[:CONTEXT_CHARS]
        steps = "\n".join(f"{i}. {s.quote}" for i, s in enumerate(p.steps, 1))
        try:
            data = parse_json(model(SYSTEM, USER.format(name=p.name, steps=steps, context=context)))
        except Exception:  # noqa: BLE001 - a procedure without these lists is still a procedure
            return p, []
        source = f"{steps}\n{context}"
        out: List[CoverageItem] = []
        for kind in KINDS:
            kept = []
            for it in data.get(kind) or []:
                if not isinstance(it, dict) or not quote_ok(str(it.get("quote") or ""), source):
                    continue
                kept.append({k: v for k, v in it.items() if isinstance(v, (str, int, float))})
                sid = next((st.id for st in pool if quote_ok(str(it["quote"]), st.text)), None)
                if sid:
                    out.append(
                        CoverageItem(
                            sid,
                            "procedure_rule",
                            f"{kind} of {p.name}",
                            dict(by_id[sid].source),
                            str(it["quote"]),
                        )
                    )
            setattr(p, kind, kept)
        return p, out

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(one, procs))
    for _, c in results:
        cov += c
    return [p for p, _ in results], cov


TOC_RE = re.compile(r"(table[ -]of[ -])?contents", re.I)
CASE_TYPES = {
    "prohibition",
    "exception_override",
    "routing",
    "notification_duty",
    "eligibility_filter",
    "approval",
    "state_transition",
}
BATCH = 40
G_SYSTEM = (
    "You read statements of a company SOP that are outside its step-by-step procedures and decide which ones "
    "change how a case is handled. Output JSON only."
)
G_USER = """Procedures of this SOP:
{procs}

Statements (outside the procedures):
{statements}

For each statement that changes how a case is handled (a condition that stops, holds or forbids work, a case that must
go to someone else, an approval needed first, someone to notify), give:
{{"s": <statement number>, "procedure": <procedure number it affects, or 0 if it applies to all work>,
  "kind": "precondition|exception|escalation", "when": "<the condition>", "then": "<stop|hold|forward|escalate|notify|...>",
  "notify": "<who or which channel, if any>", "quote": "<the statement's words, copied exactly, one sentence or less>"}}
Skip definitions, descriptions and statements that change nothing. Return {{"items": [...]}}"""


def general_rules(
    procs: List[Procedure], statements: List[Statement], model: Optional[Callable[[str, str], str]]
) -> Tuple[List[Procedure], List[CoverageItem]]:
    """Rules outside the procedures (a rescinded offer, "if a case is not covered, escalate"): attached to the procedure
    they affect, or to a "General rules" entry that applies to all work."""
    if model is None:
        return procs, []
    in_proc = {p.id for p in procs}
    cands = [
        st
        for st in statements
        if _section_key(st) not in in_proc
        and not TOC_RE.fullmatch(str(st.source.get("section", "")).strip())
        and {str(getattr(t, "value", t)) for t in st.types} & CASE_TYPES
    ]
    if not cands:
        return procs, []
    doc = cands[0].id.split(":")[0]
    general = Procedure(id=f"{doc}:general", name="General rules (outside the procedures)")
    plist = "\n".join(f"{i}. {p.name}" for i, p in enumerate(procs, 1))
    cov: List[CoverageItem] = []
    for k in range(0, len(cands), BATCH):
        batch = cands[k : k + BATCH]
        text = "\n".join(f"{i}. [§{st.source.get('section', '')}] {st.text}" for i, st in enumerate(batch, 1))
        try:
            data = parse_json(model(G_SYSTEM, G_USER.format(procs=plist, statements=text)))
        except Exception:  # noqa: BLE001
            continue
        for it in data.get("items") or []:
            if not isinstance(it, dict) or not isinstance(it.get("s"), int) or not 1 <= it["s"] <= len(batch):
                continue
            st = batch[it["s"] - 1]
            if not quote_ok(str(it.get("quote") or ""), st.text):
                continue
            n = it.get("procedure") if isinstance(it.get("procedure"), int) else 0
            target = procs[n - 1] if 1 <= n <= len(procs) else general
            item = {
                "when": str(it.get("when") or ""),
                "then": str(it.get("then") or ""),
                "quote": str(it["quote"]),
                "from_section": str(st.source.get("section", "")),
            }
            kind = it.get("kind")
            if kind == "precondition":
                target.preconditions.append(
                    {
                        "condition": item["when"],
                        "if_not": item["then"],
                        "quote": item["quote"],
                        "from_section": item["from_section"],
                    }
                )
            elif kind == "escalation":
                target.escalations.append({**item, "notify": str(it.get("notify") or "")})
            else:
                target.exceptions.append(item)
            cov.append(
                CoverageItem(
                    st.id, "procedure_rule", f"{kind} of {target.name}", dict(st.source), item["quote"]
                )
            )
    if general.preconditions or general.exceptions or general.escalations:
        procs = procs + [general]
    return procs, cov
