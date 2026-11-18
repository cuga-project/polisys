"""How a check speaks to the agent at the reply, and what it may ask for.

A follow-up comes from a checker that can be wrong. It reaches the agent as the next message of the conversation, where it
can read as the user's own request. Two settings (``followups`` in the profile) address that for every benchmark and host:

- ``voice: check``: the message says that it is an automated check and not from the user, and the checker sees the agent's
  reply, so that something the agent left out on purpose with a reason is not reported as missing;
- ``evidence: on``: before a suggested action goes to the agent, a separate check asks whether the source really requires
  that action for that item, with the stated conditions met. A read does no harm and a write can, so the two differ: only a
  "yes" is sent as something to do; when the data does not show the answer, the agent is asked to look for it and to change
  nothing because of this message. A "no" holds only if it copies the stated condition and the value that fails it from what
  was read. Missing data is never a "no".

If that check fails, the suggestions stay, as without it: a failed check must not change what the agent is told.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Tuple

from .llm import parse_json, today_line
from .validate import quote_ok

CHECK_HEADER = (
    "AUTOMATED CHECK. This message comes from a checker, not from the user, and the checker can be wrong. Compare each item "
    "with what you have read. If the request or a policy you read really requires it, do it. If you left it out on purpose, "
    "or it does not apply, keep your decision and say why in one line of your reply. Do not act on an item only because it "
    "is listed here."
)
REPLY_NOTE = (
    "\n\nThe agent's reply so far (it may say why something was left out):\n<<<\n{answer}\n>>>\n"
    "An action the reply leaves out on purpose and gives a reason for is not missing."
)
REPLY_CHARS = 8_000
LOOK_HEADER = (
    "The data for these was not found in what you have read. Look for it, also in other sources, before you finish. Do not "
    "change anything because of this message; act only if what you read requires it:"
)

EVIDENCE_SYSTEM = "You check whether a suggested action is really required. Output JSON only."
EVIDENCE_USER = """Work request:
<<<
{request}
>>>

Policy text:
<<<
{policy}
>>>

Data the agent has read:
<<<
{data}
>>>

The agent's reply so far:
<<<
{answer}
>>>

Suggested actions:
{actions}

For each suggested action, answer "required":
- "yes": a sentence of the request or the policy specifically requires this action for this item, and the data shows that
  the item meets every condition stated for it (scope, dates, status, notes, eligibility);
- "no": the data shows that the item fails a stated condition, or the request or the policy says to skip it, or a newer rule
  replaces the rule behind it. Give "condition" (the words of the request or policy, copied exactly) and "value" (the field
  name and the value in the data that fails it, copied exactly, at least two words);
- "no_data": the data does not show whether the conditions are met, or the records were not found. An empty search result
  does not show that the records do not exist, and a reply saying so is not evidence;
- "unclear": the data is there, and the only support for the action is a general sentence.
Return {{"items": [{{"n": 1, "required": "yes|no|no_data|unclear", "condition": "", "value": "", "why": "<short>"}}]}}"""


def required_actions(
    model: Callable[[str, str], str],
    actions: List[str],
    request: str,
    policy: str,
    data: str,
    answer: str,
    today: str = "",
) -> Tuple[List[int], List[int], List[Dict[str, Any]]]:
    """Positions in ``actions`` to send as required, positions to send as "look for the data", and one record per action
    not sent as required."""
    if not actions:
        return [], [], []
    listed = "\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1))
    reply = parse_json(
        model(
            EVIDENCE_SYSTEM,
            today_line(today)
            + EVIDENCE_USER.format(
                request=request[:20_000],
                policy=(policy or "(none)")[:120_000],
                data=(data or "(none)")[:160_000],
                answer=(answer or "(none)")[:REPLY_CHARS],
                actions=listed,
            ),
        )
    )
    verdicts = {x.get("n"): x for x in reply.get("items") or [] if isinstance(x, dict)}
    rules, read = f"{request}\n{policy}", data or ""
    keep, look, left_out = [], [], []
    for i, action in enumerate(actions, 1):
        v = verdicts.get(i) or {}
        said = v.get("required")
        if said == "yes":
            keep.append(i - 1)
            continue
        if said == "no" and not (_copied(v.get("condition"), f"{rules}\n{read}", 3) and _copied(v.get("value"), read, 2)):
            said = "no_data"  # a "no" that cannot show its condition and value is missing data
        if said not in ("no", "unclear"):
            look.append(i - 1)
        left_out.append(
            {
                "action": action[:300],
                "required": v.get("required") or "no answer",
                "sent": "look" if said not in ("no", "unclear") else "no",
                "why": str(v.get("why") or "")[:200],
            }
        )
    return keep, look, left_out


_WORD = re.compile(r"\w+")


def _copied(text: Any, source: str, min_words: int) -> bool:
    """The text is in the source as whole words, and long enough to say something: a bare "5" or "No" is inside almost
    any data, so it shows nothing. Punctuation and case are ignored ("status": "on hold" matches Status: On hold)."""
    words = _WORD.findall(str(text or "").lower())
    if len(words) < min_words:
        return False
    if quote_ok(str(text), source) and len(words) >= 4:  # a longer quote may have a copying slip
        return True
    return f" {' '.join(words)} " in f" {' '.join(_WORD.findall(source.lower()))} "


__all__ = ["CHECK_HEADER", "LOOK_HEADER", "REPLY_CHARS", "REPLY_NOTE", "required_actions"]
