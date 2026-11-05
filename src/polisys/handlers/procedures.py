"""Procedures (SOP steps) as step graphs.

A procedure is a run of numbered steps in one section ("1. Confirm the vendor ... 2. Check for duplicates ..."), as a
numbered list, as text ("Step 3: Create a Jira ticket ..."), or as a bulleted list of labelled steps ("- Generate
workbook: ...", at least 3, half of them done with a tool). A table of contents is not a procedure.
Each step becomes a ``Step``; consecutive steps are joined by ``next`` edges. A step's ``action`` is the tool that
performs it: a tool name written in the step (``vendor.screen_sanctions`` or ``vendor_screen_sanctions``), or, with
a model, the tool the model picks from the catalog (only names that exist are kept). A step whose text starts with
"if" or "when", or says "optional" / "may", is optional; its condition is kept.

Nothing here is specific to a document: a procedure is found by its numbered-list shape only.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from ..llm import parse_json
from ..model import Procedure, Statement, Step

OPTIONAL = re.compile(r"^(if|when|where)\b|\b(optional|optionally|may)\b", re.I)
STEP_TEXT = re.compile(
    r"^\s*step\s+(\d{1,2})\s*[:.)\-]\s*", re.I
)  # "Step 3: Create a Jira ticket" written as text
TOC = re.compile(r"(table[ -]of[ -])?contents", re.I)
MIN_STEPS = 2
MIN_BULLET_STEPS = (
    3  # a run of labelled bullets is a procedure only if it is this long and half its steps use a tool
)

SYSTEM = "You map the steps of a company procedure to the tools that perform them. Output JSON only."
USER = """Tools (name: description):
{catalog}

Procedure "{name}", steps:
{steps}

For each step give the one tool that performs it, or "" if no tool does (for example a check done by reading).
Return {{"steps": [{{"n": <number>, "tool": "<tool name or empty>"}}]}}"""


def _tool_in_text(text: str, names: List[str]) -> Optional[str]:
    low = re.sub(
        r"(?<=\w)\.(?=\w)", "_", text.lower()
    )  # "vendor.screen_sanctions" == "vendor_screen_sanctions"
    for name in sorted(names, key=len, reverse=True):
        form = re.sub(r"(?<=\w)\.(?=\w)", "_", name.lower())
        if re.search(rf"(?<![\w]){re.escape(form)}(?![\w])", low):
            return name
    return None


def procedures(
    statements: List[Statement],
    specs: List[Dict[str, Any]],
    model: Optional[Callable[[str, str], str]] = None,
    titles: Optional[Dict[str, str]] = None,
) -> List[Procedure]:
    names = [t["name"] for t in specs]
    groups: "OrderedDict[str, List[Statement]]" = OrderedDict()
    for st in statements:
        if st.source.get("step") is not None or STEP_TEXT.match(st.text):
            groups.setdefault(f"{st.id.split(':')[0]}:{st.source.get('section', '')}", []).append(st)
    out: List[Procedure] = []
    for key, sts in groups.items():
        if len(sts) < MIN_STEPS:
            continue
        sec = key.split(":", 1)[1]
        if TOC.fullmatch(str((titles or {}).get(key) or "").strip()) or TOC.fullmatch(sec.strip()):
            continue  # a table of contents lists sections; it is not a procedure
        name = (titles or {}).get(key) or f"Procedure §{sec}"
        actions = [_tool_in_text(st.text, names) for st in sts]
        if model is not None and not all(actions):
            catalog = "\n".join(f"{t['name']}: {(t.get('description') or '')[:120]}" for t in specs)
            try:
                reply = parse_json(
                    model(
                        SYSTEM,
                        USER.format(
                            catalog=catalog,
                            name=name,
                            steps="\n".join(f"{i}. {st.text}" for i, st in enumerate(sts, 1)),
                        ),
                    )
                )
                for a in reply.get("steps") or []:
                    i = a.get("n") if isinstance(a, dict) else None
                    if (
                        isinstance(i, int)
                        and 1 <= i <= len(sts)
                        and not actions[i - 1]
                        and a.get("tool") in names
                    ):
                        actions[i - 1] = a["tool"]
            except Exception:
                pass
        if all(st.source.get("bullet") for st in sts) and (
            len(sts) < MIN_BULLET_STEPS or 2 * sum(map(bool, actions)) < len(sts)
        ):
            continue  # labelled bullets that are not done with tools are a list of facts, not a procedure
        steps = [
            Step(
                id=st.id,
                action=act or "",
                required=not OPTIONAL.search(st.text),
                condition=st.text if OPTIONAL.search(st.text) else None,
                source=dict(st.source),
                quote=st.text,
            )
            for st, act in zip(sts, actions)
        ]
        edges = [{"from": a.id, "to": b.id} for a, b in zip(steps, steps[1:])]
        out.append(
            Procedure(
                id=key,
                name=name,
                trigger={"tools": sorted({s.action for s in steps if s.action})},
                steps=steps,
                edges=edges,
                source=dict(sts[0].source),
                quote=" / ".join(s.text for s in sts),
            )
        )
    return out
