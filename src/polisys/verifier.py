"""The verifier: a small model checks one planned write against policy statements, with the data that matters.

Used for statements that no code checker can decide (``verify`` rules, and types routed to the verifier by
``method_choice.json``). It gets the user's request, the numbered statements, the records that the call names
(``FactStore.for_call``) and the call itself. A model error or an unreadable answer **allows** the call and is
recorded as ``verifier_error``: an error must never block work.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from .llm import parse_json
from .model import Verdict

SYSTEM = "You check one planned action against company policy statements. Output JSON only."
USER = """User request: {request}

Policy statements:
{statements}

Data the assistant has seen (records that mention values in this call first; may be cut):
{facts}

Planned action: {tool}({args})
Arguments this tool accepts: {tool_args}

Does this action itself clearly break a statement? Judge only this action, not steps that may come later.
Do not report a violation for something this tool has no argument or way to do.
If a statement has an exception ("unless ...") that applies here, it is not broken.
Return {{"violations": [{{"statement": <number>, "reason": "...", "fix": "what to do instead"}}]}}"""


def format_statements(statements: List[Dict[str, Any]]) -> str:
    lines = []
    for s in statements:
        unless = "; ".join(u.get("condition", "") for u in s.get("unless") or [])
        lines.append(
            f"{s['n']}. [{s.get('source', '')}] {s['text']}" + (f" (unless: {unless})" if unless else "")
        )
    return "\n".join(lines)


class Verifier:
    def __init__(self, model: Callable[[str, str], str]):
        self.model = model

    def check(
        self,
        request: str,
        statements: List[Dict[str, Any]],
        tool: str,
        args: Dict[str, Any],
        facts: str,
        tool_args: Optional[List[str]] = None,
    ) -> List[Verdict]:
        """One verdict per broken statement (``deny``), or a single ``allow`` (also on errors)."""
        if not statements:
            return []
        t0 = time.monotonic()
        try:
            reply = parse_json(
                self.model(
                    SYSTEM,
                    USER.format(
                        request=request,
                        statements=format_statements(statements),
                        facts=facts,
                        tool=tool,
                        args=json.dumps(args, default=str)[:3000],
                        tool_args=", ".join(tool_args or []) or "(unknown)",
                    ),
                )
            )
        except Exception as exc:
            return [
                Verdict(
                    outcome="allow",
                    checker="verifier",
                    reason=f"verifier_error: {type(exc).__name__}",
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            ]
        by_n = {s["n"]: s for s in statements}
        out = []
        for v in reply.get("violations") or []:
            if not isinstance(v, dict) or v.get("statement") not in by_n:
                continue
            s = by_n[v["statement"]]
            out.append(
                Verdict(
                    outcome="deny",
                    rule_id=s.get("rule_id"),
                    checker="verifier",
                    reason=str(v.get("reason", "")),
                    remedy=str(v.get("fix", "")),
                    latency_ms=(time.monotonic() - t0) * 1000,
                    source={"source": s.get("source")},
                    quote=s["text"],
                )
            )
        return out or [
            Verdict(outcome="allow", checker="verifier", latency_ms=(time.monotonic() - t0) * 1000)
        ]
