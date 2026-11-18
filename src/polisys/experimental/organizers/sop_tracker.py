"""SOP progress tracker for conversations: before each agent turn, say where the agent is in the SOP.

A support agent follows an SOP across many turns: it asks the customer to run checks, reads the results, applies
fixes, and must escalate when the SOP says so. Agents lose their place: they stop after the first fix, skip a check,
or miss an escalation condition the conversation already shows. The tracker reads the SOP (split at its headings)
and the conversation, and answers two narrow questions with a small model:

1. which SOP sections govern the customer's current issue (a choice from the heading list);
2. for those sections: the checks and actions the SOP prescribes, each done / pending / not needed with evidence
   from the conversation, and whether an SOP escalation condition is met.

Every item must quote the SOP verbatim and every "done" or "met" must quote the conversation, or it is dropped.
The note goes into the conversation before the agent's turn (with the customer's message), not into the system
prompt. Nothing here names a domain: sections, steps and conditions all come from the SOP text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from .llm import parse_json
from .validate import quote_ok

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
MAX_SECTIONS = 6
MAX_PENDING = 5
RESULT_CHARS = 300

Q_SECTIONS = """SOP sections (id: heading path):
{toc}

Conversation so far:
{conv}

Which sections (at most {k}) contain the procedure the agent must follow now for the customer's current issue?
Prefer the most specific sections (the troubleshooting or handling steps), not overviews or tool lists.
Return {{"sections": ["<id>", ...]}}"""

MAX_CHAPTERS = 3  # chosen per turn
MAX_KEPT = 4  # kept over the conversation (the first ones chosen stay)
Q_CHAPTERS = """SOP chapters (id: title, and the headings inside):
{toc}

Conversation so far:
{conv}

Which chapters (at most {k}) contain the procedure the agent must follow for the customer's issue, including the
chapters for anything the issue depends on? Prefer troubleshooting or handling chapters, not overviews or tool lists.
Return {{"chapters": ["<id>", ...]}}"""

Q_TRACK = """SOP text:
{text}

Conversation so far (the agent's tool calls and results are included):
{conv}

List the checks and actions that this SOP text prescribes for the customer's current issue, in SOP order. For each:
- "action": a short description,
- "quote": the exact SOP words that prescribe it (copied, not paraphrased),
- "status": "done" (the conversation shows it was done, with its result), "pending" (prescribed for this issue
  and not done yet; also a conditional check whose condition is still unknown, because the agent has to find out),
  or "not_needed" (the conversation shows the SOP's own condition for it is not met, for example an earlier check
  showed that part is fine),
- "evidence": the exact conversation words that show it was done or is not needed ("" for pending).
Then "escalation": is a condition met under which the SOP says to escalate, transfer, or stop?
{{"met": true or false, "quote": "<exact SOP words of that rule>", "evidence": "<exact conversation words>"}}
Return {{"steps": [...], "escalation": {{...}}}}"""

SYSTEM_SECTIONS = "You match a support conversation to the sections of the company's procedure document (SOP). Output JSON only."
SYSTEM_TRACK = "You track a support agent's progress through the company's procedure (SOP). Use only the SOP text and the conversation. Output JSON only."


def split_sections(text: str) -> List[Dict[str, str]]:
    """Markdown → sections at every heading; each keeps its heading path ("Manual > Diagnosing > SIM") and body."""
    out: List[Dict[str, str]] = []
    path: List[str] = []
    body: List[str] = []

    def flush() -> None:
        if path and "".join(body).strip():
            out.append({"id": str(len(out) + 1), "title": " > ".join(path), "text": "\n".join(body).strip()})

    for line in text.splitlines():
        m = HEADING.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip("* ")]
            body.clear()
        else:
            body.append(line)
    flush()
    return out


def render_conversation(turns: List[Dict[str, Any]]) -> str:
    """turns: {"role": "customer"|"agent"|"tool", "text"} or tool calls {"role": "tool", "name", "args", "result"}."""
    lines = []
    for t in turns:
        if t.get("role") == "tool":
            res = str(t.get("result", ""))[:RESULT_CHARS]
            lines.append(
                f"[agent tool call] {t.get('name')}({json.dumps(t.get('args') or {})[:200]}) -> {res}"
            )
        else:
            lines.append(f"{t.get('role')}: {t.get('text', '')}")
    return "\n".join(lines)


class SopTracker:
    """``note(turns)`` → the text to put before the agent's next turn, or None when there is nothing to say."""

    def __init__(
        self,
        sop_text: str,
        model: Callable[[str, str], str],
        max_sections: int = MAX_SECTIONS,
        unit: str = "chapter",
    ):
        """``unit="chapter"``: choose top-level chapters (all their subsections come along) and keep every chapter
        chosen earlier in the conversation, so a step does not drop out of view between turns. "section": v1."""
        self.sections = split_sections(sop_text)
        self.by_id = {s["id"]: s for s in self.sections}
        self.model = model
        self.max_sections = max_sections
        self.unit = unit
        self.chapters: Dict[str, List[Dict[str, str]]] = {}
        for sec in self.sections:
            self.chapters.setdefault(sec["title"].split(" > ")[0], []).append(sec)
        self.chapter_ids = {str(i): name for i, name in enumerate(self.chapters, 1)}
        self.kept: List[str] = []  # chapters chosen so far in this conversation
        self.records: List[Dict[str, Any]] = []
        self._last: Optional[str] = None

    def _ask(self, system: str, user: str) -> Dict[str, Any]:
        try:
            return parse_json(self.model(system, user)) or {}
        except Exception as e:  # noqa: BLE001 — a failed question means no note, never a failed turn
            self.records.append({"error": f"{type(e).__name__}: {e}"[:200]})
            return {}

    def choose(self, conv: str) -> List[Dict[str, str]]:
        if self.unit == "chapter":
            return self._choose_chapters(conv)
        toc = "\n".join(f"{s['id']}: {s['title']}" for s in self.sections)
        ids = (
            self._ask(SYSTEM_SECTIONS, Q_SECTIONS.format(toc=toc, conv=conv, k=self.max_sections)).get(
                "sections"
            )
            or []
        )
        return [self.by_id[str(i)] for i in ids if str(i) in self.by_id][: self.max_sections]

    def _choose_chapters(self, conv: str) -> List[Dict[str, str]]:
        toc = "\n".join(
            f"{i}: {name} (contains: "
            + "; ".join(x["title"].split(" > ")[-1] for x in self.chapters[name])[:400]
            + ")"
            for i, name in self.chapter_ids.items()
        )
        ids = (
            self._ask(SYSTEM_SECTIONS, Q_CHAPTERS.format(toc=toc, conv=conv, k=MAX_CHAPTERS)).get("chapters")
            or []
        )
        for i in ids:
            name = self.chapter_ids.get(str(i))
            if name and name not in self.kept and len(self.kept) < MAX_KEPT:
                self.kept.append(name)  # a chapter, once chosen, stays for the whole conversation
        return [sec for name in self.kept for sec in self.chapters[name]]

    def track(self, secs: List[Dict[str, str]], conv: str) -> Dict[str, Any]:
        text = "\n\n".join(f"## {s['title']}\n{s['text']}" for s in secs)
        reply = self._ask(SYSTEM_TRACK, Q_TRACK.format(text=text, conv=conv))
        sop = " ".join(s["text"] for s in secs)
        steps = []
        for st in reply.get("steps") or []:
            if not isinstance(st, dict) or not quote_ok(str(st.get("quote", "")), sop):
                continue  # not in the SOP: dropped
            status = st.get("status")
            if status == "done" and not quote_ok(str(st.get("evidence", "")), conv):
                continue  # claimed done without conversation evidence: not shown
            if status == "not_needed" and not quote_ok(str(st.get("evidence", "")), conv):
                status = "pending"  # nothing in the conversation shows its condition is false: the agent must find out
            if status in ("done", "pending", "not_needed"):
                steps.append(
                    {
                        "action": str(st.get("action", ""))[:160],
                        "quote": str(st["quote"])[:300],
                        "status": status,
                    }
                )
        esc = reply.get("escalation") if isinstance(reply.get("escalation"), dict) else {}
        met = (
            bool(esc.get("met"))
            and quote_ok(str(esc.get("quote", "")), sop)
            and quote_ok(str(esc.get("evidence", "")), conv)
        )
        return {
            "steps": steps,
            "escalation": {
                "quote": str(esc.get("quote", ""))[:300],
                "evidence": str(esc.get("evidence", ""))[:300],
            }
            if met
            else None,
        }

    def note(self, turns: List[Dict[str, Any]]) -> Optional[str]:
        conv = render_conversation(turns)
        secs = self.choose(conv)
        if not secs:
            return None
        t = self.track(secs, conv)
        pending = [s for s in t["steps"] if s["status"] == "pending"][:MAX_PENDING]
        done = [s["action"] for s in t["steps"] if s["status"] == "done"]
        self.records.append({"sections": [s["title"] for s in secs], **t})
        if not pending and not t["escalation"]:
            return None
        lines = [
            "[SOP progress check, written by the company's procedure checker, not by the customer]",
            "Procedure: " + "; ".join(dict.fromkeys(s["title"].split(" > ")[0] for s in secs)) + ".",
        ]
        if done:
            lines.append("Done so far: " + "; ".join(done) + ".")
        if t["escalation"]:
            e = t["escalation"]
            lines.append(
                f'The SOP says: "{e["quote"]}". The conversation shows: "{e["evidence"]}". Follow this SOP rule now.'
            )
        elif pending:
            lines.append("Next per the SOP (not done yet, in order):")
            lines += [f'{i}. {s["action"]} (SOP: "{s["quote"]}")' for i, s in enumerate(pending, 1)]
        text = "\n".join(lines)
        if text == self._last:
            return None  # same as the last note: say nothing new
        self._last = text
        return text
