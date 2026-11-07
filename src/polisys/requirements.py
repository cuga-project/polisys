"""Task requirements: what the final state must contain for one task, from the request and the policy text.

Graders of agent benchmarks are lists of required effects (a message sent with given text, a record with given field
values, a file with given rows, nothing sent to a forbidden recipient). Most failures are a missing or wrong effect,
not a broken prohibition. This module turns the request plus the policy text that applies (a given document, or
policies the agent found in data) into a checklist of such effects, each with a verbatim quote, so that code can
track and check it. Item kinds:

- ``effect``: a change the task must make (send, post, create, update, write a file, columns, values);
- ``step``: an action the policy requires on the way (look up, verify, link, pause, notify) that leaves a trace;
- ``guard``: something that must not happen in this task (recipient, action, value);
- ``format``: a template, naming rule or exact text the policy prescribes for an output of this task;
- ``decision``: a rule that decides each record of a set (include / exclude / status / escalate) - one decision per
  record, filled by the agent with evidence.

Only items the task triggers are listed: a whole handbook is not turned into a checklist. A quote that is not in the
request or the policy text (``validate.quote_ok``) drops the item, so the model cannot add requirements.
Nothing here is specific to a benchmark: inputs are the request, the policy text and optional tool names.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .followups import REPLY_CHARS, REPLY_NOTE
from .llm import parse_json, today_line
from .validate import quote_ok

KINDS = ("effect", "step", "guard", "format", "decision")
MAX_POLICY_CHARS = 120_000
MAX_DATA_CHARS = 160_000

SYSTEM = (
    "You read a work request and the company policy that applies, and list what the finished work must contain, "
    "so that code can check it. Output JSON only."
)
USER = """Work request:
<<<
{request}
>>>

Company policy text (may be long; use only the parts this request triggers):
<<<
{policy}
>>>
{data}{tools}
List the requirements for THIS request only. Each item:
- "kind": "effect" (a change the work must make: message sent, record created or updated, file written, with
  recipients, fields, columns or values), "step" (an action the policy requires on the way that leaves a trace),
  "guard" (must not happen in this work), "format" (template, naming rule or exact text for an output of this work),
  or "decision" (a rule that decides each record of a set: include, exclude, status, approval, escalation);
- "text": the requirement in one short sentence (at most 30 words), with the concrete names, values and formats;
- "quote": the words of the request, policy or data that require THIS item, copied exactly (at most 20 words). A
  sentence about another record, value or topic is not a quote for this item; with no such sentence, leave the item out;
- "when": the condition under which it applies, or "";
- "check": {{"tool": "<tool name if known, else empty>", "targets": [recipients, channels, files, records named],
  "values": [exact texts, values or column names the result must contain]}}.
When the policy says what to do for each case or record (a new hire, an invoice, a ticket, an email), apply it to
every record in the data that the request covers: one item per required effect per record, with that record's
names, IDs, dates and amounts filled in and the policy's templates and naming rules applied. A record the policy
says to skip, hold or exclude gets a "guard" item (what must not be done for it) and, if the policy says so, the
effect that records the exception.
Keep every qualifier and condition of the quote in the item (words that limit which records count, thresholds,
dates). When the data does not show which records meet a qualifier, keep the qualifier in "text" and do not name
records for it. Before naming a record in an effect or include decision, read that record's own status, notes and
flags: a record whose own data says not to act on it now, or that does not meet the condition, gets a "guard" item
instead.
Do not list general advice, reading steps without a trace, or rules that this request does not trigger.
Return {{"items": [...]}}"""


def compile_requirements(
    model: Callable[[str, str], str],
    request: str,
    policy: str = "",
    tool_names: Optional[List[str]] = None,
    data: str = "",
    today: str = "",
) -> Dict[str, Any]:
    """The checklist for one task: ``{"items": [...], "dropped": [...], "raw": n}``. ``data``: what the agent has
    read so far (emails, records, files); with it, per-record procedures are filled in per record."""
    tools = f"\nTools that change something: {', '.join(tool_names)[:6000]}\n" if tool_names else ""
    block = (
        f"\nData the agent has read (emails, messages, records, files):\n<<<\n{data[:MAX_DATA_CHARS]}\n>>>\n"
        if data
        else ""
    )
    reply = parse_json(
        model(
            SYSTEM,
            today_line(today)
            + USER.format(
                request=request[:20_000],
                policy=(policy or "(none)")[:MAX_POLICY_CHARS],
                tools=tools,
                data=block,
            ),
        )
    )
    source = f"{request}\n{policy}\n{data[:MAX_DATA_CHARS]}"
    items, dropped = [], []
    for i, it in enumerate(reply.get("items") or [], 1):
        if not isinstance(it, dict) or it.get("kind") not in KINDS or not it.get("text"):
            dropped.append({"item": it, "why": "bad shape"})
            continue
        if not quote_ok(str(it.get("quote") or ""), source):
            dropped.append({"item": it, "why": "quote not in request or policy"})
            continue
        check = it.get("check") if isinstance(it.get("check"), dict) else {}
        items.append(
            {
                "id": f"R{len(items) + 1}",
                "kind": it["kind"],
                "text": str(it["text"])[:300],
                "quote": str(it["quote"])[:400],
                "when": str(it.get("when") or "")[:200],
                "check": {
                    "tool": str(check.get("tool") or ""),
                    "targets": [str(x) for x in check.get("targets") or [] if isinstance(x, (str, int))][:10],
                    "values": [str(x) for x in check.get("values") or [] if isinstance(x, (str, int, float))][
                        :15
                    ],
                },
            }
        )
    return {"items": items, "dropped": dropped, "raw": len(reply.get("items") or [])}


def render(items: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{it['id']} [{it['kind']}] {it['text']}" + (f" (when: {it['when']})" if it.get("when") else "")
        for it in items
    )


__all__ = ["compile_requirements", "render", "KINDS"]


GATE_SYSTEM = "You check finished work against a checklist. Output JSON only."
GATE_USER = """Checklist:
{checklist}

Changes the agent made (successful calls that change something, in order):
{writes}

For each checklist item, say whether the changes above do it: "done", "missing" (clearly not done at all), "wrong"
(clearly done with a different recipient, record, value or format than the item says), "not_applicable" (its
condition did not hold), or "unsure" (the changes do not show it either way, for example IDs you cannot map to the
names in the item).{reply} Say "missing" or "wrong" only when the changes show it. A "guard" item is "done" when nothing
forbidden happened, "wrong" when it did. Return {{"items": [{{"id": "R1", "status": "done|missing|wrong|not_applicable", "why": "<short>"}}]}}"""


def gate(
    model: Callable[[str, str], str],
    items: List[Dict[str, Any]],
    writes: List[Dict[str, Any]],
    max_chars: int = 60_000,
    answer: str = "",
) -> Dict[str, Dict[str, str]]:
    """Status per checklist item from the agent's successful writes: ``{id: {"status", "why"}}``. ``answer``: the agent's
    reply; with it, an item the reply leaves out on purpose with a reason is "not_applicable"."""
    if not items:
        return {}
    lines = []
    for w in writes:
        res = w.get("result")
        lines.append(
            f"- {w['tool']}({_json(w.get('args'))[:1500]})"
            + (f" -> {_json(res)[:600]}" if res is not None else "")
        )
    reply = parse_json(
        model(
            GATE_SYSTEM,
            GATE_USER.format(
                checklist=render(items),
                writes="\n".join(lines)[:max_chars] or "(none)",
                reply=REPLY_NOTE.format(answer=answer[:REPLY_CHARS]) if answer else "",
            ),
        )
    )
    out = {}
    for x in reply.get("items") or []:
        if (
            isinstance(x, dict)
            and x.get("id")
            and x.get("status") in ("done", "missing", "wrong", "not_applicable", "unsure")
        ):
            out[str(x["id"])] = {"status": x["status"], "why": str(x.get("why") or "")[:200]}
    return out


def _json(v: Any) -> str:
    import json

    try:
        return json.dumps(v, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(v)


__all__ += ["gate"]
