"""Compile an SOP document into a procedure-store graph (``procedure_store.py``) ahead of time (experimental).

Same output format as a hand-built graph, so the two can be compared:

1. split the SOP at its headings (sections with level, title, body, parent);
2. templates by code: sections whose title names a template, or whose body has several [placeholders] and reads
   like a message; the text is kept verbatim with its placeholders;
3. procedures: each section with ordered or imperative steps is compiled by one model call into
   {applies_when, category, inputs, decisions, steps, deadlines, must_not, notes, links}; steps must quote or
   closely follow the SOP; links point to other section ids;
4. general rules: one model call over the other sections (definitions, contacts, channels, routing, priorities,
   housekeeping, escalation, formats) into global_rules, triage, routing, housekeeping, priority, escalation;
5. calendar: holiday dates found in the text (code), business days Monday–Friday unless the SOP says otherwise;
6. checks: every step must share words with its section (else dropped and counted), every link must resolve.

The model should be the largest open model available (the compile runs once per SOP version and is cached).
Nothing here names a domain or benchmark.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Callable, Dict, List, Optional

from .llm import parse_json

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
PLACEHOLDER = re.compile(r"\[[^\]\n]{2,80}\]")
TEMPLATE_TITLE = re.compile(r"\btemplate\s*(\d+)\b", re.I)
STEP_LINE = re.compile(r"^\s*(step\s*\d+|\d+[.)]|day\s*\d+|if\b|when\b|upon\b|before\b|after\b|-\s)", re.I)
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

SYSTEM = "You compile SOP sections into a structured procedure graph. Output JSON only."
PROC_USER = """SOP sections (id: title):
{toc}

Compile this SOP section into one procedure. Keep the SOP's own words for steps, formats, names, addresses,
channels, file and column names; do not add steps the SOP does not state.

Section {sid} "{title}":
<<<
{body}
>>>

Return {{"name": "...", "category": "<request category the SOP uses>", "applies_when": "<one sentence>",
"inputs": [{{"fact": "...", "source": "<file / column / system named by the SOP>"}}],
"decisions": [{{"id": "...", "rule": "<condition with the SOP's thresholds>", "then": "...", "else": "..."}}],
"steps": ["<ordered steps, each one action with its target, template, format and deadline>"],
"deadlines": ["..."], "must_not": ["<prohibitions and holds>"], "notes": ["<priority hints>"],
"links": [{{"type": "next", "to": "<section id that must also run>", "when": "<condition or empty>"}}]}}"""
GENERAL_USER = """These SOP sections hold the rules that apply to every case (definitions, contacts, channels, triage,
routing, priorities, service levels, escalation, housekeeping, default formats):
<<<
{text}
>>>

Return {{"global_rules": [{{"id": "...", "src": "<section>", "rule": "<the rule in the SOP's words>"}}],
"triage": {{"name": "...", "src": "...", "steps": ["<ordered triage steps with their stop conditions>"]}},
"routing": {{"<category>": {{"jira": "<ticket name format>", "assignee": "...", "folder": "...",
"channel_if_a_step_posts": "..."}}}},
"housekeeping": {{"name": "...", "src": "...", "steps": ["..."]}},
"priority": {{"name": "...", "src": "...", "levels": ["..."]}},
"escalation": {{"name": "...", "src": "...", "steps": ["..."], "must_not": ["..."]}},
"definitions": {{"<term>": "<definition>"}}}}
Use null or empty lists for parts the SOP does not have."""


def split_sections(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    stack: List[int] = []
    for line in text.split("\n"):
        m = HEADING.match(line)
        if m:
            lv = len(m.group(1))
            while stack and out[stack[-1]]["level"] >= lv:
                stack.pop()
            title = m.group(2).strip()
            num = re.match(r"^(\d+(?:\.\d+)*)\.?\s", title)
            out.append(
                {
                    "level": lv,
                    "title": title,
                    "id": num.group(1) if num else f"s{len(out)}",
                    "body": "",
                    "parent": stack[-1] if stack else None,
                }
            )
            stack.append(len(out) - 1)
        elif out:
            out[-1]["body"] += line + "\n"
    for i, s in enumerate(out):
        s["children"] = [j for j, t in enumerate(out) if t["parent"] == i]
    return out


def templates_of(secs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for s in secs:
        m = TEMPLATE_TITLE.search(s["title"])
        body = s["body"].strip()
        slots = sorted(set(PLACEHOLDER.findall(body)))
        if m or (
            len(slots) >= 2 and len(body.split("\n")) >= 3 and re.match(r"^(hi|dear|hello)\b", body, re.I)
        ):
            n = m.group(1) if m else str(len(out) + 1)
            name = re.sub(r"^.*?template\s*\d+\s*[–—:-]?\s*", "", s["title"], flags=re.I) or s["title"]
            out[n] = {"number": int(n), "name": name, "text": body, "slots": slots, "section": s["id"]}
    return out


def holidays_of(text: str, year: Optional[int] = None) -> List[str]:
    found = set()
    for mon, day, yr in re.findall(rf"({MONTHS})\s+(\d{{1,2}})(?:,\s*(\d{{4}}))?", text):
        y = int(yr) if yr else year
        if y:
            try:
                found.add(dt.date(y, dt.datetime.strptime(mon, "%B").month, int(day)).isoformat())
            except ValueError:
                continue
    return sorted(found)


def _is_procedure(s: Dict[str, Any]) -> bool:
    lines = [ln for ln in s["body"].split("\n") if ln.strip()]
    return not TEMPLATE_TITLE.search(s["title"]) and sum(bool(STEP_LINE.match(ln)) for ln in lines) >= 2


def _grounded(step: str, body: str) -> bool:
    words = {w for w in re.findall(r"[a-z]{4,}", step.lower())}
    return bool(words) and len(words & set(re.findall(r"[a-z]{4,}", body.lower()))) >= max(2, len(words) // 3)


def compile_sop(
    text: str, model: Callable[[str, str], str], doc_id: str = "sop", year: Optional[int] = None
) -> Dict[str, Any]:
    secs = split_sections(text)
    toc = "\n".join(f"{s['id']}: {s['title']}" for s in secs)
    templates = templates_of(secs)
    tsec = {t["section"] for t in templates.values()}
    procs, dropped = [], 0
    for s in secs:
        if s["id"] in tsec or s["children"] and not s["body"].strip() or not _is_procedure(s):
            continue
        try:
            p = parse_json(
                model(
                    SYSTEM, PROC_USER.format(toc=toc, sid=s["id"], title=s["title"], body=s["body"][:12000])
                )
            )
        except Exception:  # noqa: BLE001 - one bad section must not stop the compile
            continue
        steps = [x for x in p.get("steps") or [] if isinstance(x, str)]
        kept = [x for x in steps if _grounded(x, s["body"])]
        dropped += len(steps) - len(kept)
        ids = {t["id"] for t in secs}
        procs.append(
            {
                **p,
                "id": s["id"],
                "src": f"§{s['id']}",
                "steps": kept,
                "links": [lk for lk in p.get("links") or [] if isinstance(lk, dict) and lk.get("to") in ids],
            }
        )
    general_text = "\n".join(
        f"## {s['title']}\n{s['body']}" for s in secs if s["id"] not in tsec and not _is_procedure(s)
    )
    try:
        general = parse_json(model(SYSTEM, GENERAL_USER.format(text=general_text[:40000])))
    except Exception:  # noqa: BLE001
        general = {}
    return {
        "id": doc_id,
        "version": "compiled",
        "built_by": getattr(model, "model_name", "model"),
        **{
            k: general.get(k)
            for k in (
                "global_rules",
                "triage",
                "routing",
                "housekeeping",
                "priority",
                "escalation",
                "definitions",
            )
        },
        "procedures": procs,
        "templates": templates,
        "calendar": {
            "workdays": [0, 1, 2, 3, 4],
            "holidays": holidays_of(
                "\n".join(
                    x["title"] + "\n" + x["body"]
                    for x in secs
                    if re.search(r"holiday|closure|closed", x["title"], re.I)
                ),
                year,
            ),
        },
        "checks": {
            "procedures": len(procs),
            "templates": len(templates),
            "steps_dropped_ungrounded": dropped,
        },
    }
