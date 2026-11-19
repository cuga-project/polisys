"""The fields of a policy package: every field has the type the model says, the tables and the globals have their form, and
the parts name each other correctly."""

from __future__ import annotations

import dataclasses
import re
import typing
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from polisys.package.model import PACKAGE_SCHEMA, PolicyPackage, json_key
from polisys.compile.validate.slots import ISO_DATE, KNOWN_EXTRA, PackageError, _check_slots, _severity, _show


@lru_cache(maxsize=None)
def _hints(cls: type) -> Dict[str, Any]:
    return typing.get_type_hints(cls)


def _conforms(value: Any, tp: Any) -> bool:
    """``value`` has the shape of the annotation ``tp``. Nested model objects only need to be of the right class:
    their own fields are checked when the walk reaches them."""
    if tp is Any:
        return True
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is typing.Union:
        return any(_conforms(value, a) for a in args)
    if tp is type(None):
        return value is None
    if origin in (list, List):
        return isinstance(value, list) and (not args or all(_conforms(v, args[0]) for v in value))
    if origin in (dict, Dict):
        return isinstance(value, dict) and (
            not args or all(_conforms(k, args[0]) and _conforms(v, args[1]) for k, v in value.items())
        )
    if isinstance(tp, type) and issubclass(tp, Enum):
        return isinstance(value, tp) or (isinstance(value, str) and value in {m.value for m in tp})
    if tp is bool:
        return isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tp is str:
        return isinstance(value, str)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return isinstance(value, tp)
    return True


def _model_errors(obj: Any, path: str, out: List[PackageError]) -> None:
    """Fields whose value does not have the annotated type (``from_json`` lets them through), and keys that went to
    ``extra`` because the model does not have them. Walks every model object under ``obj``."""
    hints = _hints(type(obj))
    own = getattr(obj, "id", "")
    label = own if isinstance(own, str) else ""
    for f in dataclasses.fields(obj):
        value, key = getattr(obj, f.name), json_key(f)
        if not _conforms(value, hints[f.name]):
            want = str(hints[f.name]).replace("typing.", "").replace("polisys.package.model.", "")
            out.append(
                PackageError(label, "invalid_field", f"{path}{key}: expected {want}, got {_show(value)}")
            )
            continue
        items = value if isinstance(value, list) else [value]
        for i, item in enumerate(items):
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                _model_errors(item, f"{path}{key}[{i}]." if isinstance(value, list) else f"{path}{key}.", out)
    known = KNOWN_EXTRA if isinstance(obj, PolicyPackage) else set()
    extra = getattr(obj, "extra", None)
    for key in extra if isinstance(extra, dict) else {}:  # a wrong-typed ``extra`` was reported above
        if isinstance(key, str) and not key.startswith("_") and key not in known:
            out.append(
                PackageError(
                    label, "unknown_field", f"{path}extra.{key}: the model has no field {key!r}", "warning"
                )
            )


# The parts of a package that the model types only as ``Dict[str, Any]``, and what the checks below read from them.
QUESTION_SLOTS: Dict[str, Tuple[Any, bool]] = {
    "statement_ids": ("strs", True),
    "rule_ids": ("strs", False),  # the rules a question is about, when it is about a pair of rules
    "question": ("text", True),
    "open_values": (
        "strs",
        False,
    ),  # "<obligation id>.<value name>" of a literal that the question leaves open
    "default_level": ("level", False),
}
DEFINITION_SLOTS: Dict[str, Tuple[Any, bool]] = {
    "term": ("text", False),
    "meaning": ("text", False),
    "source": ({"statement": ("str", False)}, False),
}
OBLIGATION_SLOTS: Dict[str, Tuple[Any, bool]] = {
    "scope": ("dict", False),
    "effect": ({"kind": ("str", False), "tools": ("strs", False)}, False),
    "values": ("dict", False),
    "source": ({"quote": ("str", False)}, False),
}


def _table_errors(
    label: str, where: str, slots: Dict[str, Tuple[Any, bool]], data: Any
) -> List[PackageError]:
    """``_check_slots`` for a part of the package: a wrong type is an ``invalid_field`` (the checks below read it)."""
    if not isinstance(data, dict):
        return [PackageError(label, "invalid_field", f"{where}: must be an object, got {_show(data)}")]
    return [
        PackageError(
            label, "invalid_field" if code == "invalid_slots" else code, f"{where}.{msg}", _severity(code)
        )
        for code, msg in _check_slots(slots, data, "")
    ]


def _shape_errors(pkg: PolicyPackage) -> List[PackageError]:
    out: List[PackageError] = []
    for i, q in enumerate(pkg.policy_questions):
        out += _table_errors("", f"policy_questions[{i}]", QUESTION_SLOTS, q)
    definitions = pkg.global_.get("definitions")
    if definitions is not None and not isinstance(definitions, list):
        out.append(
            PackageError("", "invalid_field", f"global.definitions: must be a list, got {_show(definitions)}")
        )
    for i, d in enumerate(definitions if isinstance(definitions, list) else []):
        out += _table_errors("", f"global.definitions[{i}]", DEFINITION_SLOTS, d)
    for o in pkg.obligations:
        fields = {name: getattr(o, name) for name in OBLIGATION_SLOTS}
        out += _table_errors(o.id, f"obligation {o.id}", OBLIGATION_SLOTS, fields)
    return out


# What ``package.to_store_graph`` keeps of ``global``, and the shape ``ProcedureStore`` reads from each key. A key outside
# this list is dropped without an error, and a value of another shape crashes the store or loses its content.
GLOBAL_KEYS = (
    "global_rules",
    "triage",
    "routing",
    "housekeeping",
    "priority",
    "escalation",
    "templates",
    "calendar",
    "definitions",
)
GRAPH_NODES = ("triage", "housekeeping", "escalation")  # {name, src, steps, must_not}


def _str_items(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _global_findings(g: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(code, message) for each ``global`` value that ``ProcedureStore`` would crash on or not read."""
    import datetime as dt

    out: List[Tuple[str, str]] = []

    def bad(where: str, want: str, value: Any) -> None:
        out.append(("invalid_field", f"global.{where}: must be {want}, got {_show(value)}"))

    for key in g:
        if key not in GLOBAL_KEYS:
            out.append(
                (
                    "unknown_field",
                    f"global.{key}: the runtime reads only {', '.join(GLOBAL_KEYS)}; this key is dropped",
                )
            )
    rules = g.get("global_rules")
    if rules is not None:
        if not isinstance(rules, list):
            bad("global_rules", "a list of {id, src, rule}", rules)
        for i, r in enumerate(rules if isinstance(rules, list) else []):
            if not isinstance(r, dict) or not all(
                isinstance(r.get(k), str) and r[k] for k in ("id", "src", "rule")
            ):
                bad(f"global_rules[{i}]", "an object with text id, src and rule", r)
            elif r.get("attach_when") is not None:
                try:
                    re.compile(r["attach_when"])
                except (re.error, TypeError):
                    bad(f"global_rules[{i}].attach_when", "a regular expression", r["attach_when"])
    for key in GRAPH_NODES:
        node = g.get(key)
        if node is None:
            continue
        if not isinstance(node, dict):
            bad(key, "an object {name, src, steps, must_not}", node)
            continue
        for field_name in ("name", "src"):
            if field_name in node and not isinstance(node[field_name], str):
                bad(f"{key}.{field_name}", "a string", node[field_name])
        for field_name in ("steps", "must_not"):
            if field_name in node and not _str_items(node[field_name]):
                bad(f"{key}.{field_name}", "a list of strings", node[field_name])
    priority = g.get("priority")
    if priority is not None:
        if not isinstance(priority, dict):
            bad("priority", "an object {name, src, levels}", priority)
        else:
            for field_name in ("name", "src"):
                if field_name in priority and not isinstance(priority[field_name], str):
                    bad(f"priority.{field_name}", "a string", priority[field_name])
            if "levels" in priority and not _str_items(priority["levels"]):
                bad("priority.levels", "a list of strings", priority["levels"])
    routing = g.get("routing")
    if routing is not None:
        if not isinstance(routing, dict) or not all(isinstance(v, dict) for v in routing.values()):
            bad("routing", "an object: category -> {field: value}", routing)
    templates = g.get("templates")
    if templates is not None:
        if not isinstance(templates, dict):
            bad("templates", "an object keyed by the number the text cites", templates)
        for key, t in (templates if isinstance(templates, dict) else {}).items():
            if not isinstance(t, dict) or not all(
                isinstance(t.get(k), str) and t[k] for k in ("name", "text")
            ):
                bad(f"templates.{key}", "an object with text name and text (and number, slots)", t)
            elif str(t.get("number")) != str(key):
                bad(f"templates.{key}.number", f"the key {key!r}", t.get("number"))
            elif "slots" in t and not _str_items(t["slots"]):
                bad(f"templates.{key}.slots", "a list of strings", t["slots"])
    calendar = g.get("calendar")
    if calendar is not None:
        if not isinstance(calendar, dict):
            bad("calendar", "an object {workdays, holidays}", calendar)
        else:
            days = calendar.get("workdays")
            if days is not None and not (
                isinstance(days, list)
                and all(isinstance(d, int) and not isinstance(d, bool) and 0 <= d <= 6 for d in days)
            ):
                bad("calendar.workdays", "a list of weekday numbers 0 (Monday) to 6", days)
            holidays = calendar.get("holidays")
            if holidays is not None:
                ok = _str_items(holidays)
                if ok:
                    try:
                        [dt.date.fromisoformat(h) for h in holidays]
                    except ValueError:
                        ok = False
                if not ok:
                    bad("calendar.holidays", "a list of ISO dates (YYYY-MM-DD)", holidays)
    definitions = g.get("definitions")
    for i, d in enumerate(definitions if isinstance(definitions, list) else []):
        if isinstance(d, dict) and not (isinstance(d.get("term"), str) and d["term"]):
            out.append(("invalid_field", f"global.definitions[{i}].term: must be a non-empty string"))
    return out


def _structure_errors(pkg: PolicyPackage, legacy: bool = False) -> List[PackageError]:
    """The schema name, fields of the wrong type, keys the model does not have, and the shape of what the model types
    only as a dict."""
    out: List[PackageError] = []
    if pkg.schema != PACKAGE_SCHEMA:
        out.append(
            PackageError("", "invalid_field", f"schema is {pkg.schema!r}, expected {PACKAGE_SCHEMA!r}")
        )
    _model_errors(pkg, "", out)
    if not any(
        e.code == "invalid_field" for e in out
    ):  # the dicts are only known to be dicts if the model walk passed
        out += _shape_errors(pkg)
        if not legacy:  # a converted package keeps the global part of its source as it was
            out += [
                PackageError("", code, msg, _severity(code)) for code, msg in _global_findings(pkg.global_)
            ]
    return out


def _content_errors(pkg: PolicyPackage) -> List[PackageError]:
    """Values of the right type that are still wrong: a date that is not a date, a rule id used twice."""
    out: List[PackageError] = []
    for d in pkg.documents:
        if isinstance(d.effective_date, str) and not ISO_DATE.match(d.effective_date):
            out.append(
                PackageError(
                    d.id,
                    "invalid_field",
                    f"document {d.id}: effective_date {d.effective_date!r} is not an ISO date",
                )
            )
    wrapper_ids: set = set()
    inner_ids: set = set()
    for r in pkg.rules:
        inner = r.rule.get("id") if isinstance(r.rule, dict) else None
        twice = r.id in wrapper_ids or (isinstance(inner, str) and inner in inner_ids)
        if not r.id:
            out.append(PackageError(r.id, "duplicate_rule_id", "a rule has no id"))
        elif twice:
            # ``RuleSetHooks.add_rules`` keeps every rule of one load, but the brief lists an id once, and a block_once or a
            # repeated denial is counted by id
            where = f"id {r.id!r}" if r.id in wrapper_ids else f"inner id {inner!r}"
            out.append(
                PackageError(
                    r.id,
                    "duplicate_rule_id",
                    f"rule {where} is used by another rule: both load, but the brief lists only the first"
                    " and block_once and repeated denials count them together",
                )
            )
        wrapper_ids.add(r.id)
        if isinstance(inner, str) and inner:
            inner_ids.add(inner)
    return out
