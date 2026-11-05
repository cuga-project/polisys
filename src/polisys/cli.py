"""Command line for the policy layer. Used by people, CI and the policy compiler skill (any agent harness).

    python -m polisys.cli convert policy.pdf appendix.md -o set.json [--meta meta.json] [--model M]
    python -m polisys.cli build set.json --tools tools.json -o build.json [--model M] [--cache DIR]
    python -m polisys.cli validate build.json --tools tools.json [--set set.json]
    python -m polisys.cli validate package.json [--tools tools.json] [--legacy]
    python -m polisys.cli coverage build.json
    python -m polisys.cli counts package.json [package.json ...]
    python -m polisys.cli pack set.json draft.json -o package.json --harness H --model M [--tests tests.json] [--tools tools.json]

``pack`` makes the one ``policy-package/1`` file of a policy from the converted policy set and a hand-written draft (the
step of the policy compiler skill, ``skill/SKILL.md``): it adds the schema, the version (one hash of the statements, so a
change of the splitter moves it), the documents (with the ``modality`` the draft gives each statement) and ``generator``.
It fills a step's ``text`` from its ``id`` when left out. It stops, and writes nothing, on an id it can check that does not
exist (statements, steps, ``scope.procedure``, ``links.to``, ``rule_ids``, definition sources, test statements), on a step
text that does not contain its statement, an unknown draft key or modality, and a test with no tool or no allow or deny.
It writes the statements with no modality to ``<output>.unmarked.txt``. It does not validate the content: ``validate`` on
its output does (rules, coverage, quotes and must-nots against the text, runtime shapes).

``--model`` is a LiteLLM model name (``OPENAI_BASE_URL``, ``OPENAI_API_KEY``); ``openrouter:<name>`` uses OpenRouter
(``OPENROUTER_API_KEY``). ``validate`` exits with code 1 when errors remain (``schema_conflict`` is reported, not
an error). It takes a build, or a policy package: a file with a package key (``schema``, ``documents``, ...) is read as
one, and its ``schema`` must be ``policy-package/1``. For a package ``--tools`` is optional (without it the checks that
need tool schemas are skipped, and the output says so) and ``--legacy`` is for a package converted from an older
artifact: gaps in coverage and rules with no tool are warnings. ``meta.json`` maps a document id (file name without
extension) to ``trust``, ``effective_date``, ``authority_rank``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from .model import PACKAGE_SCHEMA, Modality, PolicyBuild, PolicyPackage, PolicySet, from_json, to_json

WARN_ONLY = {"schema_conflict"}
PACKAGE_KEYS = {
    "schema",
    "documents",
    "obligations",
    "global",
    "policy_questions",
    "tests",
}  # a build has none


def _model(name: Optional[str]):
    if not name:
        return None
    from .llm import openai_json_model, openrouter_json_model

    return (
        openrouter_json_model(name.split(":", 1)[1])
        if name.startswith("openrouter:")
        else openai_json_model(name)
    )


def _tools(path: str) -> List[dict]:
    data = json.loads(Path(path).read_text())
    return data if isinstance(data, list) else data.get("tools", [])


def cmd_convert(a: argparse.Namespace) -> int:
    from .documents import load_policy_set

    meta = json.loads(Path(a.meta).read_text()) if a.meta else {}
    ps = load_policy_set([Path(f) for f in a.files], meta, model=_model(a.model))
    Path(a.output).write_text(json.dumps(to_json(ps), indent=1, ensure_ascii=False))
    print(
        f"{len(ps.documents)} documents, {sum(len(d.statements) for d in ps.documents)} statements -> {a.output}"
    )
    return 0


def cmd_build(a: argparse.Namespace) -> int:
    from .builder import build_policy_set

    ps = from_json(PolicySet, json.loads(Path(a.set).read_text()))
    model = _model(a.model)
    build = build_policy_set(
        ps,
        _tools(a.tools),
        model,
        cache_dir=Path(a.cache) if a.cache else None,
        generator={"kind": a.generator, "harness": a.harness, "model": a.model or "none"},
    )
    Path(a.output).write_text(json.dumps(to_json(build), indent=1, ensure_ascii=False))
    if a.typed_set:  # the policy set with the statement types this build assigned
        Path(a.typed_set).write_text(json.dumps(to_json(ps), indent=1, ensure_ascii=False))
    print(json.dumps(build.stats))
    errors = [e for e in build.validation if e["code"] not in WARN_ONLY]
    return 1 if errors else 0


def _is_warning(e) -> bool:
    return getattr(e, "severity", "") == "warning" or e.code in WARN_ONLY


def _fail(message: str) -> int:
    print(f"ERROR invalid_field : {message}")
    print("1 errors, 0 warnings")
    return 1


def cmd_validate(a: argparse.Namespace) -> int:
    from .validate import validate_build, validate_package

    try:
        data = json.loads(Path(a.build).read_text(encoding="utf-8"))
        tools = _tools(a.tools) if a.tools else None
    except (OSError, ValueError) as exc:  # a file that is missing, or is not JSON
        return _fail(f"cannot read the input: {type(exc).__name__}: {exc}")
    if not isinstance(data, dict):
        return _fail(f"{a.build} must hold a JSON object, got {type(data).__name__}")
    if PACKAGE_KEYS & set(
        data
    ):  # a build has none of these keys, so a package with a typo in ``schema`` is not read as one
        if data.get("schema") != PACKAGE_SCHEMA:
            return _fail(f"schema is {data.get('schema')!r}, expected {PACKAGE_SCHEMA!r}")
        try:
            pkg = from_json(PolicyPackage, data)
        except Exception as exc:  # any shape the decoder cannot read is one finding, not a traceback
            return _fail(f"cannot read the package: {type(exc).__name__}: {exc}")
        if tools is None:
            print("NOTE no --tools: tool, argument and enum checks were skipped")
        errors = validate_package(
            pkg,
            tools,
            legacy=a.legacy,
            priority_classes=[c for c in (a.priority_classes or "").split(",") if c],
        )
    else:
        if tools is None:
            print("--tools is required to validate a build", file=sys.stderr)
            return 2
        build = from_json(PolicyBuild, data)
        ps = from_json(PolicySet, json.loads(Path(a.set).read_text())) if a.set else None
        errors = validate_build(build, tools, ps)
    for e in errors:
        print(f"{'WARN' if _is_warning(e) else 'ERROR'} {e.code} {e.statement_id}: {e.message}")
    hard = [e for e in errors if not _is_warning(e)]
    print(f"{len(hard)} errors, {len(errors) - len(hard)} warnings")
    return 1 if hard else 0


MUST_WORDS = re.compile(
    r"\b(must|must not|shall|never|do not|don't|cannot|can't|may not|required|requires|always|only|not allowed|"
    r"prohibited|forbidden|ensure|should)\b",
    re.I,
)
DRAFT_KEYS = (
    "modality",
    "rules",
    "procedures",
    "obligations",
    "global",
    "links",
    "coverage",
    "policy_questions",
    "tests",
)
TEST_VERDICTS = ("allow", "deny")


def _draft_rule(raw: object) -> object:
    """A draft rule may leave out what the other fields already say: ``type`` (from ``rule``), the id inside ``rule``, the
    ``checker`` and the quote inside ``rule``. Anything that is not an object is left as it is, for ``validate``."""
    from .handlers.prose import CHECKER

    if not isinstance(raw, dict) or not isinstance(raw.get("rule"), dict):
        return raw
    rule = dict(raw["rule"])
    out = dict(raw)
    out.setdefault("type", rule.get("type", ""))
    for key in ("id", "type"):
        if raw.get(key) and not rule.get(key):
            rule[key] = raw[key]
    if raw.get("quote") and not rule.get("quote"):
        rule["quote"] = raw["quote"]
    out["rule"] = rule
    if out.get("type"):
        out.setdefault("checker", CHECKER.get(out["type"], "code"))
    return out


def _known(value: object, known: object) -> bool:
    """The value is a string that is in ``known``: an id of another type (a list, a number) is never known, and never raises."""
    return isinstance(value, str) and value in known  # type: ignore[operator]


def _listed(value: object) -> list:
    return value if isinstance(value, list) else []


def _fill_steps(draft: dict, ps: PolicySet) -> None:
    """A step that names a statement and has no ``text`` gets the statement's words."""
    text = {st.id: st.text for d in ps.documents for st in d.statements}
    for p in _listed(draft.get("procedures")):
        for step in _listed(p.get("steps")) if isinstance(p, dict) else []:
            if isinstance(step, dict) and not step.get("text") and _known(step.get("id"), text):
                step["text"] = text[step["id"]]


def _structure_problems(draft: dict, ps: PolicySet) -> List[str]:
    """Ids inside procedures, obligations, questions and definitions that point at nothing."""
    from .validate import _fold

    text = {st.id: _fold(st.text) for d in ps.documents for st in d.statements}
    problems: List[str] = []
    procedures = [p for p in _listed(draft.get("procedures")) if isinstance(p, dict)]
    ids = {p.get("id") for p in procedures if isinstance(p.get("id"), str)}
    for p in procedures:
        for i, step in enumerate(_listed(p.get("steps"))):
            where = f"procedures[{p.get('id')}].steps[{i}]"
            if not isinstance(step, dict):
                problems.append(f"{where}: a step is an object with id and text, got {type(step).__name__}")
                continue
            sid = step.get("id")
            if sid and not _known(sid, text):
                problems.append(f"{where}: {sid!r} is not a statement of the policy set")
            elif sid and isinstance(step.get("text"), str) and text[sid] not in _fold(step["text"]):
                problems.append(
                    f"{where}: the text does not contain the statement {sid} (steps are its words)"
                )
        for i, link in enumerate(_listed(p.get("links"))):
            if isinstance(link, dict) and not _known(link.get("to"), ids):
                problems.append(
                    f"procedures[{p.get('id')}].links[{i}]: {link.get('to')!r} is not a procedure"
                )
    for i, o in enumerate(_listed(draft.get("obligations"))):
        scope = o.get("scope") if isinstance(o, dict) else None
        if isinstance(scope, dict) and "procedure" in scope and not _known(scope["procedure"], ids):
            problems.append(
                f"obligations[{i}]: scope.procedure {scope.get('procedure')!r} is not a procedure"
            )
    rules = {
        r.get("id")
        for r in _listed(draft.get("rules"))
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    }
    for i, q in enumerate(_listed(draft.get("policy_questions"))):
        for rid in _listed(q.get("rule_ids")) if isinstance(q, dict) else []:
            if not _known(rid, rules):
                problems.append(f"policy_questions[{i}]: rule_ids {rid!r} is not a rule of the draft")
    g = draft.get("global")
    for i, d in enumerate(_listed(g.get("definitions")) if isinstance(g, dict) else []):
        source = d.get("source") if isinstance(d, dict) else None
        sid = source.get("statement") if isinstance(source, dict) else None
        if sid and not _known(sid, text):
            problems.append(f"global.definitions[{i}]: {sid!r} is not a statement of the policy set")
    return problems


def _pack_problems(draft: dict, ps: PolicySet, tool_names: Optional[set]) -> List[str]:
    """What is wrong in a draft that ``pack`` can see: ids that do not exist and tests that cannot run."""
    statements = {st.id for d in ps.documents for st in d.statements}
    problems: List[str] = []
    bad_keys = sorted(set(draft) - set(DRAFT_KEYS))
    if bad_keys:
        problems.append(f"unknown draft keys {bad_keys}; the keys are {', '.join(DRAFT_KEYS)}")
    modality = draft.get("modality") or {}
    if not isinstance(modality, dict):
        problems.append("modality must be an object: statement id -> must, must_not, may or should")
        modality = {}
    for sid, value in modality.items():
        if not _known(sid, statements):
            problems.append(f"modality: {sid!r} is not a statement of the policy set")
        if not isinstance(value, str) or value not in {m.value for m in Modality}:
            problems.append(
                f"modality of {sid}: {value!r} is not one of {', '.join(m.value for m in Modality)}"
            )

    def ids_of(items: object, key: str, label: str) -> None:
        for i, item in enumerate(items if isinstance(items, list) else []):
            ids = item.get(key) if isinstance(item, dict) else None
            for sid in ids if isinstance(ids, list) else [ids] if ids else []:
                if not _known(sid, statements):
                    problems.append(f"{label}[{i}]: {sid!r} is not a statement of the policy set")

    ids_of(draft.get("rules"), "statement_ids", "rules")
    ids_of(draft.get("policy_questions"), "statement_ids", "policy_questions")
    ids_of(draft.get("coverage"), "statement_id", "coverage")
    problems += _structure_problems(draft, ps)
    tests = draft.get("tests")
    if tests is not None and not isinstance(tests, list):
        problems.append("tests must be a list of cases")
    for i, case in enumerate(tests if isinstance(tests, list) else []):
        if not isinstance(case, dict) or not case.get("tool") or case.get("expected") not in TEST_VERDICTS:
            problems.append(f"tests[{i}]: a case needs a tool and expected: allow or deny")
        elif tool_names is not None and not _known(case["tool"], tool_names):
            problems.append(f"tests[{i}]: unknown tool {case['tool']}")
        elif case.get("statement_id") and not _known(case["statement_id"], statements):
            problems.append(f"tests[{i}]: {case['statement_id']!r} is not a statement of the policy set")
    return problems


def cmd_pack(a: argparse.Namespace) -> int:
    """policy set + draft -> one policy package. The draft holds what a person or an agent wrote; the set holds the
    documents. The package is not written if the draft has a problem ``pack`` can see."""
    try:
        ps = from_json(PolicySet, json.loads(Path(a.set).read_text(encoding="utf-8")))
        draft = json.loads(Path(a.draft).read_text(encoding="utf-8"))
        extra_tests = json.loads(Path(a.tests).read_text(encoding="utf-8")) if a.tests else None
        tools = _tools(a.tools) if a.tools else None
    except (OSError, ValueError) as exc:
        return _fail(f"cannot read the input: {type(exc).__name__}: {exc}")
    if not isinstance(draft, dict):
        return _fail(f"{a.draft} must hold a JSON object, got {type(draft).__name__}")
    if extra_tests is not None:
        draft = {**draft, "tests": extra_tests}
    names = {t.get("name") for t in tools if isinstance(t, dict)} if tools is not None else None
    _fill_steps(draft, ps)
    problems = _pack_problems(draft, ps, names)
    if problems:
        for p in problems:
            print(f"ERROR pack : {p}")
        print(f"{len(problems)} errors: nothing written")
        return 1
    modality = draft.get("modality") or {}
    for d in ps.documents:
        for st in d.statements:
            if st.id in modality:
                st.modality = Modality(modality[st.id])
    data = {
        "schema": PACKAGE_SCHEMA,
        "version": ps.version,
        "policy_hash": ps.version,
        "generator": {"kind": "skill", "harness": a.harness, "model": a.model},
        "documents": to_json(ps.documents),
        "rules": [_draft_rule(r) for r in draft.get("rules") or []],
        "procedures": draft.get("procedures") or [],
        "obligations": draft.get("obligations") or [],
        "global": draft.get("global") or {},
        "links": draft.get("links") or [],
        "coverage": draft.get("coverage") or [],
        "policy_questions": draft.get("policy_questions") or [],
        "tests": draft.get("tests") or [],
    }
    try:
        pkg = from_json(PolicyPackage, data)
    except Exception as exc:  # a shape the decoder cannot read (an unknown link type, a number for a list)
        return _fail(f"cannot read the draft: {type(exc).__name__}: {exc}")
    from .package import save_package

    save_package(pkg, a.output)
    statements = [st for d in pkg.documents for st in d.statements]
    unmarked = [st for st in statements if st.modality is None]
    print(
        f"{len(statements)} statements ({len(unmarked)} with no modality), {len(pkg.rules)} rules, "
        f"{len(pkg.procedures)} procedures, {len(pkg.obligations)} obligations, "
        f"{len(pkg.policy_questions)} policy questions, {len(pkg.tests)} tests -> {a.output}"
    )
    if (
        unmarked
    ):  # a heading, a trigger or a definition needs no outcome; a rule the author forgot to mark does
        loud = [st for st in unmarked if MUST_WORDS.search(st.text)]
        quiet = [st for st in unmarked if st not in loud]
        listing = Path(a.output).with_suffix(".unmarked.txt")
        listing.write_text(
            "\n".join(f"{st.id}\t{'LOUD' if st in loud else ''}\t{st.text}" for st in loud + quiet) + "\n",
            encoding="utf-8",
        )
        print(
            f"NOTE {len(unmarked)} statements have no modality, so no outcome is required for them; "
            f"{len(loud)} of them have must-style words. All are in {listing}. Check that none says must:"
        )
        for st in loud[:20]:
            print(f"  {st.id}: {st.text[:100]}")
        if len(loud) > 20:
            print(f"  ... and {len(loud) - 20} more with must-style words, in the file")
    return 0


def cmd_coverage(a: argparse.Namespace) -> int:
    build = from_json(PolicyBuild, json.loads(Path(a.build).read_text()))
    print("coverage:", dict(Counter(c.kind for c in build.coverage)))
    print(
        "rules:",
        dict(Counter(r.type for r in build.rules)),
        "procedures:",
        len(build.procedures),
        "links:",
        dict(Counter(lk.type.value for lk in build.links)),
    )
    for c in build.coverage:
        if c.kind in ("dangling_link", "classify_fallback", "prose_fallback", "conflict", "out_of_scope"):
            print(f"- {c.kind} {c.statement_id}: {c.reason}")
    return 0


def cmd_counts(a: argparse.Namespace) -> int:
    """One JSON object per package file: what it holds (``counts.package_counts``). No model, no tool schema."""
    from .counts import package_counts
    from .package import load_package

    out = {}
    for path in a.packages:
        try:
            out[path] = package_counts(load_package(path))
        except (OSError, ValueError) as exc:
            return _fail(f"cannot read {path}: {type(exc).__name__}: {exc}")
    print(json.dumps(out if len(out) > 1 else next(iter(out.values())), indent=1, ensure_ascii=False))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m polisys.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="documents -> policy set JSON")
    c.add_argument("files", nargs="+")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--meta")
    c.add_argument("--model", help="learn each document's heading and reference style with this model")
    c.set_defaults(fn=cmd_convert)
    b = sub.add_parser("build", help="policy set JSON -> policy build JSON")
    b.add_argument("set")
    b.add_argument("--tools", required=True)
    b.add_argument("-o", "--output", required=True)
    b.add_argument("--model")
    b.add_argument("--cache")
    b.add_argument("--generator", default="pipeline", choices=["pipeline", "skill"])
    b.add_argument("--harness", default="cli")
    b.add_argument("--typed-set", help="also write the policy set with the assigned statement types")
    b.set_defaults(fn=cmd_build)
    v = sub.add_parser(
        "validate", help="check a build or a policy package against the tool schemas (and the policy set)"
    )
    v.add_argument("build", help="a build file or a policy package file")
    v.add_argument("--tools", help="tool schemas; required for a build")
    v.add_argument("--set", help="the policy set of a build")
    v.add_argument("--legacy", action="store_true", help="a package converted from an older artifact")
    v.add_argument(
        "--priority-classes",
        help="the priority classes of the config, highest first, separated by commas (a class not listed is a warning)",
    )
    v.set_defaults(fn=cmd_validate)
    pk = sub.add_parser("pack", help="policy set + draft -> one policy package (the skill's step)")
    pk.add_argument("set", help="the policy set from `convert`")
    pk.add_argument("draft", help="rules, procedures, obligations, ... written by hand")
    pk.add_argument("-o", "--output", required=True)
    pk.add_argument("--harness", required=True, help="who ran the skill, for example claude-code")
    pk.add_argument("--model", required=True, help="the model that wrote the draft")
    pk.add_argument("--tests", help="decision tests (a list of cases); replace the draft's own `tests`")
    pk.add_argument("--tools", help="tool schemas: the tools of the tests must exist")
    pk.set_defaults(fn=cmd_pack)
    cv = sub.add_parser("coverage", help="what became of every statement")
    cv.add_argument("build")
    cv.set_defaults(fn=cmd_coverage)
    cn = sub.add_parser("counts", help="what a policy package holds, in numbers")
    cn.add_argument("packages", nargs="+")
    cn.set_defaults(fn=cmd_counts)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
