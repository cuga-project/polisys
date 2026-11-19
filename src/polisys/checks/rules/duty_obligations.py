"""``obligations_from_duties``: the duty rules of a package as obligations, for the obligation ledger."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from polisys.values import _values
from polisys.checks.rules.duties import duty_tools
from polisys.checks.rules.rule_set_hooks import RuleSetHooks


def obligations_from_duties(
    rules: Iterable[Dict[str, Any]],
    specs: Optional[Iterable[Dict[str, Any]]] = None,
    level: str = "observe",
) -> List[Any]:
    """The ``duty`` rules as obligations of the producer ``duty`` (design section 5): "after this write, also do that".

    An item opens at its trigger write (``scope.trigger``: ``tools`` and ``when``, tested on the trigger's arguments) and is
    met by a later successful call of the ``then`` tool or of a tool for the same kind of record (``duty_tools``). A value of
    ``then.when`` that the rule names alone becomes a ``literal`` that the call must carry in that argument; a list of
    values or ``"*"`` is not compared. An exception of the rule (``unless``) stays in the text of the item, as in the old
    follow-up: the agent decides whether it applies. Nothing here reads data or secrets: only the rule. ``specs``: the tool
    specs of the run, for the tools of the same kind; ``level``: the level of the items."""
    from polisys.package.model import Obligation

    by_name = {t["name"]: t for t in specs or [] if isinstance(t, dict) and t.get("name")}
    out: List[Any] = []
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("type") != "duty":
            continue
        trigger, then = rule.get("trigger") or {}, rule.get("then") or {}
        if not trigger.get("tool") or not then.get("tool"):
            continue
        tools = sorted(duty_tools(then, by_name), key=lambda t: (t != then["tool"], t))
        match: Dict[str, Any] = {}
        values: Dict[str, Any] = {}
        for arg, wanted in (then.get("when") or {}).items():
            alone = [str(v) for v in ([wanted] if isinstance(wanted, str) else _values(wanted))]
            if len(alone) != 1 or alone[0] == "*" or not alone[0].strip():
                continue
            name = re.sub(r"\W", "_", str(arg))
            if not re.match(r"[A-Za-z_]", name):
                name = f"v_{name}"
            match[str(arg)] = {"pattern": "{" + name + "}"}
            values[name] = {"from": "literal", "value": alone[0]}
        text = RuleSetHooks._describe(rule)
        unless = "; ".join(
            str(u.get("condition", "")) for u in rule.get("unless") or [] if isinstance(u, dict)
        )
        if unless:
            text = f"{text} (not needed if: {unless[:160]})"
        out.append(
            Obligation(
                id=f"duty_{rule.get('id') or len(out)}",
                origin="duty",
                scope={"trigger": {"tools": [trigger["tool"]], "when": trigger.get("when") or {}}},
                effect={"kind": "write", "tools": tools, "match": match},
                values=values,
                check="code",
                level=level,
                source={"section": str(rule.get("source") or ""), "quote": text},
            )
        )
    return out
