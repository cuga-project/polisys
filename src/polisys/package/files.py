"""The policy package: one JSON file per policy (``PolicyPackage``, design section 4), and converters from the artifacts
that exist today.

Three source shapes:

- a **runner build file**: ``build`` and ``policy_set``, and sometimes flat ``rules``, ``procedures``, ``policy_hash``
  and other keys at the top;
- a **builder output**: the fields of ``PolicyBuild`` at the top, where each rule is ``{id, type, rule: {...}, ...}``;
- a **store graph** (``procedure_store.py``): global rules, triage, routing, procedures with string steps, templates.

**Round trip is per format.** ``to_build(from_build(x)) == x`` and ``to_store_graph(from_store_graph(x)) == x``,
including key order. The two formats do not hold the same content, so conversion never merges them: one policy can have
one package per source until a merged package replaces them.

Common fields become real fields of the model. What the model does not have goes to ``extra`` at the same level, so
nothing is lost. Reserved names in ``extra`` hold what only the round trip needs:

- ``_layout`` (package): the source format and the keys of the source dicts, in order;
- ``_build``, ``_policy_set`` (package): fields of the nested ``build`` and ``policy_set`` dicts with no place in the
  model, such as the wrapped ``build.rules`` when flat top-level rules won;
- ``_keys`` (any object): the keys of its source dict, only when they differ from the usual ones for that object;
- ``_flat`` (rule): the source rule was the runtime dict itself, so ``to_build`` writes ``rule`` back as it is.

``to_build`` and ``to_store_graph`` also accept a package from another format or from no format. The result then has the
usual shape of the target format. A store graph has no rules, so ``to_store_graph`` leaves them out. ``cls`` and
``level`` are never written to a source format: ``to_build`` restores ``may_deny`` as it was found, and it does not
translate a level that was changed after conversion.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .handlers.prose import CHECKER
from .model import (
    CoverageItem,
    Document,
    Link,
    Obligation,
    PACKAGE_SCHEMA,
    PolicyPackage,
    Procedure,
    RULE_LEVELS,
    Rule,
    Section,
    Statement,
    Step,
    dumps,
    from_json,
    json_key,
    step_text,
    to_json,
)

BUILD, STORE_GRAPH = "build", "store_graph"
LAYOUT, BUILD_BAG, POLICY_SET_BAG, KEYS, FLAT = "_layout", "_build", "_policy_set", "_keys", "_flat"
_RESERVED = {LAYOUT, BUILD_BAG, POLICY_SET_BAG, KEYS, FLAT}
_DERIVED = {"cls", "level"}  # package fields that a source format never had

BUILD_KEYS = (
    "version",
    "rules",
    "procedures",
    "links",
    "definitions",
    "coverage",
    "validation",
    "generator",
    "stats",
    "policy_questions",
)
_BUILD_MODELLED = set(BUILD_KEYS) - {"stats"}
GRAPH_PROCEDURE_KEYS = [
    "id",
    "name",
    "category",
    "applies_when",
    "inputs",
    "decisions",
    "steps",
    "must_not",
    "links",
    "deadlines",
    "notes",
    "src",
]
GRAPH_GLOBAL_KEYS = (
    "global_rules",
    "triage",
    "routing",
    "housekeeping",
    "priority",
    "escalation",
    "templates",
    "calendar",
)
GRAPH_KEYS = (
    "id",
    "version",
    "built_by",
    "global_rules",
    "triage",
    "routing",
    "housekeeping",
    "priority",
    "escalation",
    "procedures",
    "templates",
    "definitions",
    "calendar",
)


# -- rule class and level ----------------------------------------------------------------------------------------------


def rule_class(rule_type: str) -> str:
    """hard, judgment or guidance, from the rule type."""
    return {"verify": "judgment", "guidance": "guidance"}.get(rule_type, "hard")


def rule_level(rule_type: str, may_deny: bool = True, rule: Optional[Dict[str, Any]] = None) -> str:
    """The level of a converted rule: what ``RuleSetHooks`` does with it today when no method choice applies. A method
    choice in a profile can still change that at runtime. ``may_deny``: False if the rule is marked remind-only anywhere.
    ``rule``: the runtime dict, where absent and ``may_deny: true`` differ for the "no assumed data" types."""
    rule = rule or {}
    if rule_type == "guidance":
        return "observe"  # shown in the brief, never checked
    if rule.get("schema_conflict"):
        return "observe"  # the tool's own allowed values stand
    if rule_type in ("grounded_values", "id_provenance") and (
        may_deny is False or rule.get("may_deny") is not True
    ):
        return "observe"  # never measured, so recorded only unless a rule says otherwise
    if may_deny is False or rule_type in ("verify", "duty"):
        return "remind"  # a verifier verdict only reminds; a duty is a follow-up and never denies
    return "deny"


# -- reading a source format -------------------------------------------------------------------------------------------


def _shape(cls: type, target: str) -> Optional[List[str]]:
    """The usual keys of an object in a source format, or None if it should have no dict form there."""
    if cls is Procedure and target == STORE_GRAPH:
        return GRAPH_PROCEDURE_KEYS
    if cls is Step and target == STORE_GRAPH:
        return None  # a string
    return [json_key(f) for f in dataclasses.fields(cls) if not f.metadata.get("omit_empty")]


def _mark(obj: Any, raw: Dict[str, Any], target: str) -> None:
    """Record the source keys of ``obj`` and of the objects inside it, where they are not the usual ones."""
    keys = list(raw)
    if keys != _shape(type(obj), target):
        obj.extra[KEYS] = keys
    for f in dataclasses.fields(obj):
        value, source = getattr(obj, f.name), raw.get(f.name)
        if isinstance(value, list) and isinstance(source, list):
            for item, item_raw in zip(value, source):
                if dataclasses.is_dataclass(item) and isinstance(item_raw, dict):
                    _mark(item, item_raw, target)


def _typed(cls: type, raw: Dict[str, Any], target: str = BUILD) -> Any:
    obj = from_json(cls, raw)
    _mark(obj, raw, target)
    return obj


def _rule(raw: Dict[str, Any], wrappers: Dict[str, Dict[str, Any]]) -> Rule:
    """A rule in either shape (see ``rule_audit.load_rules``). A flat rule takes what it lacks from the wrapped rule of
    the same id in ``build.rules``, because a flat rule has no statement ids, checker or source location."""
    if isinstance(raw.get("rule"), dict):
        rule = _typed(Rule, raw)
    else:
        meta = wrappers.get(raw.get("id")) or {}
        rule_type = raw.get("type", "")
        rule = Rule(
            id=raw.get("id", ""),
            type=rule_type,
            rule=raw,
            statement_ids=list(raw.get("statement_ids") or meta.get("statement_ids") or []),
            checker=(meta.get("checker") if meta.get("type") == rule_type else None)
            or CHECKER.get(rule_type, "code"),
            may_deny=raw.get("may_deny", True),
            source=dict(meta.get("source") or {}),
            quote=raw.get("quote") or meta.get("quote") or "",
            extra={FLAT: True},
        )
    may_deny = rule.may_deny is not False and rule.rule.get("may_deny", True) is not False
    rule.cls = rule.cls or rule_class(rule.type)
    rule.level = rule.level or rule_level(rule.type, may_deny, rule.rule)
    return rule


def from_build(build: Dict[str, Any]) -> PolicyPackage:
    """A runner build file or a builder output → package. Top-level ``rules`` and ``procedures`` win over those in
    ``build``, as for every reader today; the others stay in ``extra`` for the way back."""
    raw = copy.deepcopy(build)
    runner = isinstance(raw.get("build"), dict)
    body = raw["build"] if runner else raw
    policy_set = raw["policy_set"] if isinstance(raw.get("policy_set"), dict) else None

    wrappers = {r.get("id"): r for r in body.get("rules") or [] if isinstance(r.get("rule"), dict)}
    raw_rules = raw["rules"] if "rules" in raw else body.get("rules") or []
    raw_procedures = raw["procedures"] if "procedures" in raw else body.get("procedures") or []
    pkg = PolicyPackage(
        version=body.get("version", ""),
        policy_hash=raw.get("policy_hash", ""),
        generator=body.get("generator") or {},
        documents=[_typed(Document, d) for d in (policy_set or {}).get("documents") or []],
        rules=[_rule(r, wrappers) for r in raw_rules],
        procedures=[_typed(Procedure, p) for p in raw_procedures],
        global_={"definitions": body["definitions"]} if "definitions" in body else {},
        links=[_typed(Link, x) for x in body.get("links") or []],
        coverage=[_typed(CoverageItem, x) for x in body.get("coverage") or []],
        validation=body.get("validation") or [],
        policy_questions=body.get("policy_questions") or [],
    )

    modelled = {"policy_hash", "rules", "procedures"} | ({"build"} if runner else _BUILD_MODELLED)
    if policy_set is not None:
        modelled.add("policy_set")
    pkg.extra = {k: v for k, v in raw.items() if k not in modelled}
    layout: Dict[str, Any] = {"format": BUILD, "top": list(raw)}
    if runner:
        layout["build"] = list(body)
        bag = {k: v for k, v in body.items() if k not in _BUILD_MODELLED}
        if "rules" in raw and "rules" in body:
            bag["rules"] = body["rules"]
        if "procedures" in raw and "procedures" in body and raw["procedures"] != body["procedures"]:
            bag["procedures"] = body["procedures"]
        if bag:
            pkg.extra[BUILD_BAG] = bag
    if policy_set is not None:
        layout["policy_set"] = list(policy_set)
        bag = {
            k: v
            for k, v in policy_set.items()
            if k != "documents" and not (k == "links" and v == body.get("links"))
        }
        if bag:
            pkg.extra[POLICY_SET_BAG] = bag
    pkg.extra[LAYOUT] = layout
    return pkg


def _documents_from_statements(doc_id: str, statements: Sequence[Any]) -> List[Document]:
    """Statements (objects or dicts) → one document, with a section per ``section`` key or ``source.section``."""
    sections: Dict[str, Section] = {}
    for item in statements:
        st = to_json(item) if dataclasses.is_dataclass(item) else dict(item)
        section_id = str(st.pop("section", None) or (st.get("source") or {}).get("section") or "preamble")
        title = st.pop("title", "") or ""
        sections.setdefault(section_id, Section(id=section_id, title=title)).statements.append(
            from_json(Statement, st)
        )
    return [Document(id=doc_id, kind="statements", sections=list(sections.values()))]


def from_store_graph(graph: Dict[str, Any], statements: Optional[List[Any]] = None) -> PolicyPackage:
    """A store graph → package. ``statements``: the policy statements behind the graph, if known; a graph does not
    hold them, so they become ``documents`` and take no part in ``to_store_graph``."""
    raw = copy.deepcopy(graph)
    definitions = raw.get("definitions")
    global_ = {k: raw[k] for k in GRAPH_GLOBAL_KEYS if k in raw}
    modelled = set(GRAPH_GLOBAL_KEYS) | {"version", "built_by", "procedures"}
    if isinstance(definitions, dict):
        global_["definitions"] = [{"term": t, "meaning": m} for t, m in definitions.items()]
        modelled.add("definitions")
    extra = {k: v for k, v in raw.items() if k not in modelled}
    extra[LAYOUT] = {"format": STORE_GRAPH, "top": list(raw)}
    return PolicyPackage(
        version=raw.get("version", ""),
        generator={"built_by": raw["built_by"]} if "built_by" in raw else {},
        documents=_documents_from_statements(str(raw.get("id", "")), statements) if statements else [],
        procedures=[_typed(Procedure, p, STORE_GRAPH) for p in raw.get("procedures") or []],
        global_=global_,
        extra=extra,
    )


# -- writing a source format -------------------------------------------------------------------------------------------


def _layout(pkg: PolicyPackage, fmt: str) -> Optional[Dict[str, Any]]:
    layout = pkg.extra.get(LAYOUT)
    return layout if layout and layout.get("format") == fmt else None


def _unset(f: dataclasses.Field, value: Any) -> bool:
    if f.default is not dataclasses.MISSING:
        return value == f.default
    if f.default_factory is not dataclasses.MISSING:
        return value == f.default_factory()
    return value in ("", None)


def _emit(obj: Any, target: str, own: bool) -> Any:
    """A model object as a plain dict of the ``target`` format. ``own``: the package came from that format, so the keys
    recorded on the object apply. Fields outside those keys are written only if they hold a value."""
    if isinstance(obj, list):
        return [_emit(x, target, own) for x in obj]
    if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
        return to_json(obj)
    extra = getattr(obj, "extra", None) or {}
    if isinstance(obj, Rule) and extra.get(FLAT):
        return to_json(obj.rule)
    if isinstance(obj, Step) and target == STORE_GRAPH and not (own and KEYS in extra):
        return step_text(obj)
    by_key = {json_key(f): f for f in dataclasses.fields(obj) if f.name != "extra"}
    keys = extra[KEYS] if own and KEYS in extra else _shape(type(obj), target) or []
    out: Dict[str, Any] = {}
    for key in keys:
        if key in by_key:
            out[key] = _emit(getattr(obj, by_key[key].name), target, own)
        elif key in extra:
            out[key] = to_json(extra[key])
    for key, f in by_key.items():
        value = getattr(obj, f.name)
        if key not in out and f.name not in _DERIVED and not _unset(f, value):
            out[key] = _emit(value, target, own)
    for key, value in extra.items():
        if key not in out and key not in _RESERVED:
            out[key] = to_json(value)
    return out


def _pick(keys: Sequence[str], values: Dict[str, Any], bag: Dict[str, Any]) -> Dict[str, Any]:
    """The dict with these keys, in this order. A value in ``bag`` (the source's own) wins over a modelled one."""
    return {k: to_json(bag[k] if k in bag else values[k]) for k in keys if k in bag or k in values}


def to_build(pkg: PolicyPackage) -> Dict[str, Any]:
    """Package → the build file it came from, or a builder output if it came from elsewhere."""
    layout = _layout(pkg, BUILD)
    own = layout is not None
    rules = _emit(pkg.rules, BUILD, own)
    procedures = _emit(pkg.procedures, BUILD, own)
    links = _emit(pkg.links, BUILD, own)
    values: Dict[str, Any] = {
        "version": pkg.version,
        "rules": rules,
        "procedures": procedures,
        "links": links,
        "definitions": to_json(pkg.global_.get("definitions") or []),
        "coverage": _emit(pkg.coverage, BUILD, own),
        "validation": to_json(pkg.validation),
        "generator": to_json(pkg.generator),
    }
    if pkg.policy_questions or (own and "policy_questions" in ((layout or {}).get("build") or layout["top"])):
        values["policy_questions"] = to_json(
            pkg.policy_questions
        )  # an empty list is not written unless the source had it
    if not own:
        return _pick(BUILD_KEYS, {**values, "stats": to_json(pkg.extra.get("stats") or {})}, {})
    top_values = {**values, "policy_hash": pkg.policy_hash}
    if "policy_set" in layout:
        top_values["policy_set"] = _pick(
            layout["policy_set"],
            {"documents": _emit(pkg.documents, BUILD, own), "links": links},
            pkg.extra.get(POLICY_SET_BAG) or {},
        )
    if "build" in layout:
        top_values["build"] = _pick(layout["build"], values, pkg.extra.get(BUILD_BAG) or {})
    return _pick(layout["top"], top_values, {k: v for k, v in pkg.extra.items() if k not in _RESERVED})


def to_store_graph(pkg: PolicyPackage) -> Dict[str, Any]:
    """Package → the store graph it came from, or a new graph with the usual keys if it came from elsewhere."""
    layout = _layout(pkg, STORE_GRAPH)
    own = layout is not None
    g = to_json(pkg.global_)
    values: Dict[str, Any] = {k: g[k] for k in GRAPH_GLOBAL_KEYS if k in g}
    values["definitions"] = {d["term"]: d.get("meaning", "") for d in g.get("definitions") or []}
    values["version"] = pkg.version
    values["built_by"] = pkg.generator.get("built_by", "")
    values["procedures"] = _emit(pkg.procedures, STORE_GRAPH, own)
    if own:
        return _pick(layout["top"], values, {k: v for k, v in pkg.extra.items() if k not in _RESERVED})
    values["id"] = pkg.extra.get("id") or (pkg.documents[0].id if pkg.documents else "")
    return {k: values[k] for k in GRAPH_KEYS if values.get(k) not in (None, "", [], {})}


# -- what the components take ------------------------------------------------------------------------------------------


def runtime_rules(pkg: PolicyPackage) -> List[Dict[str, Any]]:
    """The rule dicts ``RuleSetHooks`` runs: the rule's own dict and its ``level``. A wrapper's ``may_deny: false`` is
    copied in and its ``true`` is not, because absent and ``true`` differ for ``grounded_values`` and ``id_provenance``.
    An unknown level raises ``ValueError``: at run time it would let every write through with only a reminder."""
    out = []
    for r in pkg.rules:
        rule = dict(r.rule)
        if r.may_deny is False:
            rule["may_deny"] = False
        if r.level:
            rule["level"] = r.level
        level = rule.get("level")
        if level not in ("", None) and level not in RULE_LEVELS:
            raise ValueError(f"rule {r.id}: level {level!r} is not one of {', '.join(RULE_LEVELS)}")
        out.append(rule)
    return out


def _rules_by_statement(pkg: PolicyPackage) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for r in pkg.rules:
        for statement_id in r.statement_ids:
            out.setdefault(statement_id, []).append(r.rule.get("id") or r.id)
    return out


def exception_links(pkg: PolicyPackage) -> Dict[str, List[str]]:
    """Rule id -> ids of the rules that are an exception of it. A link joins statements (``exception_of``: the source is an
    exception to the target), and a rule holds statements (``statement_ids``), so each rule that holds the source is an
    exception of each rule that holds the target, except a rule that holds both ends: it has the exception in its own text
    (folded in), and it is neither the exception nor the rule of that link. A link with no rule at one end gives nothing
    here, and ``validate_package`` reports it (``exception_without_rule``). ``RuleSetHooks.from_package(..., with_exceptions=True)`` reads this."""
    held = _rules_by_statement(pkg)
    out: Dict[str, List[str]] = {}
    for link in pkg.links:
        if str(getattr(link.type, "value", link.type)) != "exception_of":
            continue
        both = set(held.get(link.src, [])) & set(held.get(link.dst, []))
        for target in held.get(link.dst, []):
            for exception in held.get(link.src, []):
                if target in both or exception in both:
                    continue
                if exception not in out.setdefault(target, []):
                    out[target].append(exception)
    return {k: v for k, v in out.items() if v}


def obligations_from_package(pkg: PolicyPackage, case: Any = None) -> List[Obligation]:
    """The obligations of the producer ``package`` (design section 5): the obligations the package holds, with no change.

    With a ``case`` (anything with a ``procedure`` attribute, such as ``obligations.Case``), only the ones that apply to it: an
    obligation of ``package`` or ``data`` origin whose ``scope.procedure`` is the case's procedure or is not set (it applies
    to every case). Obligations of ``request`` and ``duty`` origin open at the user message and at their trigger, not per
    case, so they are given with every case. With no case: all of them, which is what the ledger takes and binds itself."""
    out = []
    for ob in pkg.obligations:
        scope = ob.scope if isinstance(ob.scope, dict) else {}
        procedure = str(scope.get("procedure") or "")
        if case is None or ob.origin not in ("package", "data") or not procedure:
            out.append(ob)
        elif procedure == str(getattr(case, "procedure", "") or ""):
            out.append(ob)
    return out


def procedure_dicts(pkg: PolicyPackage) -> List[Dict[str, Any]]:
    """The procedures as plain dicts of a build, which ``ProcedureGuide`` and ``SopLoop`` take. From a build package
    they are that build's own. From a store graph a step has ``text`` and no ``action``: see ``step_text``."""
    return _emit(pkg.procedures, BUILD, _layout(pkg, BUILD) is not None)


# -- files -------------------------------------------------------------------------------------------------------------


def save_package(pkg: PolicyPackage, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(pkg) + "\n", encoding="utf-8")


def load_package(path: Union[str, Path]) -> PolicyPackage:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != PACKAGE_SCHEMA:
        raise ValueError(f"{path}: schema is {data.get('schema')!r}, expected {PACKAGE_SCHEMA!r}")
    return from_json(PolicyPackage, data)
