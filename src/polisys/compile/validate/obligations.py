"""The obligations of a policy package: their enums and scope, and that each value binding names something that exists."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from polisys.compile.handlers.tables import _props
from polisys.package.model import PolicyPackage
from polisys.compile.validate.slots import EFFECT_KINDS, OBLIGATION_ENUMS, PackageError, VALUE_ORIGINS, _empty


def _obligation_errors(
    pkg: PolicyPackage, by_name: Optional[Dict[str, Dict[str, Any]]]
) -> List[PackageError]:
    out: List[PackageError] = []
    for o in pkg.obligations:
        for field_name, known in OBLIGATION_ENUMS.items():
            value = getattr(o, field_name)
            if isinstance(value, str) and value not in known:
                out.append(
                    PackageError(
                        o.id,
                        "invalid_field",
                        f"obligation {o.id}: {field_name} {value!r} is not one of {', '.join(known)}",
                    )
                )
        effect = o.effect if isinstance(o.effect, dict) else {}
        if effect.get("kind") not in EFFECT_KINDS:
            out.append(
                PackageError(
                    o.id,
                    "unfilled_slot",
                    f"obligation {o.id}: effect.kind {effect.get('kind')!r} is not one of {', '.join(EFFECT_KINDS)}",
                )
            )
        tools = effect.get("tools") if isinstance(effect.get("tools"), list) else []
        if effect.get("kind") == "write" and not tools:
            out.append(
                PackageError(
                    o.id, "unfilled_slot", f"obligation {o.id}: a write names no tool in effect.tools"
                )
            )
        for tool in tools:
            if by_name is not None and tool not in by_name:
                out.append(PackageError(o.id, "unknown_tool", f"obligation {o.id}: unknown tool {tool}"))
        for name, spec in (o.values or {}).items():
            origin = spec.get("from") if isinstance(spec, dict) else None
            if origin not in VALUE_ORIGINS:
                out.append(
                    PackageError(
                        o.id,
                        "invalid_field",
                        f"obligation {o.id}: value {name} has origin {origin!r}, expected one of {', '.join(VALUE_ORIGINS)}",
                    )
                )
    return out


# The contract of an obligation's values (design section 5, "Value binding"). ``effect.match`` maps a tool argument to
# ``{"pattern": text}``; a pattern holds ``{name}`` or ``{name:format}`` for the values of the obligation (``{{`` and
# ``}}`` are braces). A value is bound when a match pattern names it, directly or through a bound ``template`` value.
# Only a bound value is compared with the call; an unbound one is shown and never compared. Only a bound ``literal``,
# ``template`` or ``computed`` value can be called wrong.
CODE_ARGS = (
    "code",
    "command",
)  # the argument that holds the code of a code tool (``executePython``, ``executeBash``)
PLACEHOLDER_NAME = re.compile(r"\{([A-Za-z_]\w*)(?::[^{}]*)?\}")
VALUE_KEY = {  # the key each origin must carry
    "literal": "value",
    "template": "pattern",
    "computed": "fn",
    "case_fact": "field",
    "request": "about",
}


def _placeholders(text: Any) -> List[str]:
    if not isinstance(text, str):
        return []
    return PLACEHOLDER_NAME.findall(text.replace("{{", "").replace("}}", ""))


def _binding_errors(pkg: PolicyPackage, by_name: Optional[Dict[str, Dict[str, Any]]]) -> List[PackageError]:
    """Value binding of each obligation: what ``effect.match`` and the values say about each other."""
    out: List[PackageError] = []
    asked: set = set()  # (obligation id, value name) that a question calls open
    known = {o.id: o.values if isinstance(o.values, dict) else {} for o in pkg.obligations}
    for q in pkg.policy_questions:
        for ref in (
            q.get("open_values") or []
            if isinstance(q, dict) and isinstance(q.get("open_values"), list)
            else []
        ):
            oid, _, name = ref.rpartition(".") if isinstance(ref, str) else ("", "", "")
            if name in known.get(oid, {}):
                asked.add((oid, name))
            else:
                out.append(
                    PackageError(
                        oid,
                        "unknown_open_value",
                        f"policy question: open_values {ref!r} is not `<obligation id>.<value name>` of an obligation",
                    )
                )
    for o in pkg.obligations:
        values = o.values if isinstance(o.values, dict) else {}
        effect = o.effect if isinstance(o.effect, dict) else {}
        match = effect.get("match") if isinstance(effect.get("match"), dict) else {}
        listed = effect.get("tools")
        tools = [t for t in listed if isinstance(t, str)] if isinstance(listed, list) else []

        def err(code: str, message: str, severity: str = "error") -> None:
            out.append(PackageError(o.id, code, f"obligation {o.id}: {message}", severity))

        for name, spec in values.items():
            origin = spec.get("from") if isinstance(spec, dict) else None
            if not isinstance(origin, str) or origin not in VALUE_KEY:
                continue
            key = VALUE_KEY[origin]
            if _empty(spec.get(key)) and not (origin == "literal" and key in spec):
                err("unfilled_slot", f"value {name} ({origin}) has no {key}")
        for arg, m in match.items():
            if not isinstance(m, dict) or not isinstance(m.get("pattern"), str) or not m["pattern"]:
                err("invalid_field", f"match.{arg} must be an object with a text pattern")
                continue
            if by_name is not None and tools and all(t in by_name for t in tools):
                lacking = [t for t in tools if arg not in _props(by_name[t])]
                if arg in CODE_ARGS and len(lacking) < len(tools):
                    pass  # a code argument is `code` for one code tool and `command` for another: see below
                elif lacking:
                    err("unknown_arg", f"match.{arg} is not an argument of {', '.join(lacking)}")
            for ph in _placeholders(m["pattern"]):
                if ph not in values:
                    err("unknown_value_name", f"match.{arg} names {{{ph}}}, which is not in values")
        if (
            by_name is not None
            and tools
            and all(t in by_name for t in tools)
            and any(a in match for a in CODE_ARGS)
        ):
            for t in tools:  # one match for each code tool: the argument that holds its code
                own = [a for a in CODE_ARGS if a in _props(by_name[t])]
                if own and own[0] not in match:
                    err(
                        "unknown_arg",
                        f"{t} takes its code in `{own[0]}`, and match has no pattern for it: bind the code argument "
                        f"of every code tool ({', '.join(CODE_ARGS)})",
                    )
        bound: set = set()
        todo = [ph for m in match.values() if isinstance(m, dict) for ph in _placeholders(m.get("pattern"))]
        while todo:
            name = todo.pop()
            if name in bound or name not in values:
                continue
            bound.add(name)
            spec = values[name]
            if isinstance(spec, dict) and spec.get("from") == "template":
                for ph in _placeholders(spec.get("pattern")):
                    if ph not in values:
                        err("unknown_value_name", f"value {name} names {{{ph}}}, which is not in values")
                    todo.append(ph)
        for name, spec in values.items():
            if name not in bound:
                err(
                    "unbound_value",
                    f"unmatched value {name}: it is in no effect.match pattern, so it is shown and never compared",
                    "warning",
                )
        for name, spec in values.items():
            if isinstance(spec, dict) and spec.get("from") == "literal" and (o.id, name) in asked:
                err(
                    "literal_open_question",
                    f"value {name} is a literal and a policy question lists it as open: leave the value out until the "
                    "owner answers",
                )
    return out
