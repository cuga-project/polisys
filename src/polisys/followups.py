"""How a check speaks to the agent at the reply, and what it may ask for.

A follow-up comes from a checker that can be wrong. It reaches the agent as the next message of the conversation, where it
can read as the user's own request. Two settings (``followups`` in the profile) address that for every benchmark and host:

- ``voice: check``: the message says that it is an automated check and not from the user, and the checker sees the agent's
  reply, so that something the agent left out on purpose with a reason is not reported as missing;
- ``evidence: on``: before a suggested action goes to the agent, a separate check asks whether the source really requires
  that action for that item, with the stated conditions met. Only a "yes" is sent.

If that check fails, the suggestions stay, as without it: a failed check must not change what the agent is told.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from .llm import parse_json

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
- "no": the item fails a stated condition, the request or the policy says to skip it, a newer rule replaces the rule behind
  it, or the reply gives a valid reason for leaving it out;
- "unclear": the only support is a general sentence, or the data does not show that the conditions are met.
Return {{"items": [{{"n": 1, "required": "yes|no|unclear", "why": "<short>"}}]}}"""


def required_actions(
    model: Callable[[str, str], str],
    actions: List[str],
    request: str,
    policy: str,
    data: str,
    answer: str,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Positions in ``actions`` that the source requires, and one record per action left out."""
    if not actions:
        return [], []
    listed = "\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1))
    reply = parse_json(
        model(
            EVIDENCE_SYSTEM,
            EVIDENCE_USER.format(
                request=request[:20_000],
                policy=(policy or "(none)")[:120_000],
                data=(data or "(none)")[:160_000],
                answer=(answer or "(none)")[:REPLY_CHARS],
                actions=listed,
            ),
        )
    )
    verdicts = {x.get("n"): x for x in reply.get("items") or [] if isinstance(x, dict)}
    keep, left_out = [], []
    for i, action in enumerate(actions, 1):
        v = verdicts.get(i) or {}
        if v.get("required") == "yes":
            keep.append(i - 1)
        else:
            left_out.append(
                {"action": action[:300], "required": v.get("required") or "no answer", "why": str(v.get("why") or "")[:200]}
            )
    return keep, left_out


__all__ = ["CHECK_HEADER", "REPLY_CHARS", "REPLY_NOTE", "required_actions"]
