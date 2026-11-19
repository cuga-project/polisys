"""Text the package must come from: a quote, a must-not or a value that is not in the policy text was invented by the model."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from polisys.compile.handlers.tables import _props
from polisys.package.model import PolicyPackage
from polisys.compile.validate.rule import _fold
from polisys.compile.validate.slots import PackageError, _show
from polisys.compile.validate.fields import GRAPH_NODES
from polisys.compile.validate.obligations import PLACEHOLDER_NAME, _placeholders

MIN_PART_WORDS = 3  # a must-not with a condition is two parts of verbatim words, each of at least this many


def _flat(text: Any) -> str:
    """Words only, lower case, one space between them, a space at both ends: for finding a run of words in the text."""
    return " " + " ".join(re.findall(r"\w+", _fold(text))) + " "


class _Text:
    """The words of each document, statements joined in order (a sentence the converter split is whole again)."""

    def __init__(self, pkg: PolicyPackage):
        docs = [[st.text for st in d.statements] for d in pkg.documents]
        self.exact = [" ".join(_fold(t) for t in texts) for texts in docs if texts]
        self.words = [_flat(" ".join(texts)) for texts in docs if texts]

    def __bool__(self) -> bool:
        return bool(self.exact)

    def has(self, text: Any) -> bool:
        """The text is in a document, word for word (spaces and emphasis marks aside)."""
        t = _fold(text).strip(" .;:,\"'")
        return bool(t) and any(t in e for e in self.exact)

    def has_words(self, words: str) -> bool:
        return any(words in w for w in self.words)

    def in_parts(self, text: Any, most: int) -> bool:
        """The text is verbatim, or is at most ``most`` runs of words that are each verbatim and at least ``MIN_PART_WORDS``
        long. A run that is in the text keeps every shorter run that starts at the same word in the text, so ``longest[i]``
        tells which runs from word ``i`` are allowed, and a split with the fewest runs is found by trying every allowed
        length (the longest first run can leave a rest that is too short, so it is not the only choice)."""
        words = _flat(text).split()
        if not words:
            return False
        if self.has_words(f" {' '.join(words)} "):
            return True
        n = len(words)
        longest = []
        for i in range(n):
            j = i + 1
            while j <= n and self.has_words(f" {' '.join(words[i:j])} "):
                j += 1
            longest.append(j - 1 - i)
        fewest = [0] * (n + 1)  # fewest runs that cover words[i:]; n + 1 when none does
        for i in range(n - 1, -1, -1):
            best = n + 1
            for size in range(MIN_PART_WORDS, longest[i] + 1):
                if i + size <= n:
                    best = min(best, 1 + fewest[i + size])
            fewest[i] = best
        return fewest[0] <= most

    def must_not_ok(self, text: Any) -> bool:
        """A must-not is verbatim, or a condition and a prohibition that are each verbatim."""
        return self.in_parts(text, 2)


PRONOUN_START = re.compile(
    r"^\W*(they|them|their|it|its|he|she|his|her|we|this|these|those|that|such)\b", re.I
)


def bare_must_not(text: str) -> bool:
    """A must-not that cannot be read alone: under 5 words, or one that starts with a pronoun."""
    return len(re.findall(r"\w+", text)) < 5 or bool(PRONOUN_START.match(text))


def _must_nots(pkg: PolicyPackage) -> Iterator[Tuple[str, str]]:
    """(where, text) of every must-not the runtime shows or checks: those of procedures and of the global nodes."""
    for p in pkg.procedures:
        for m in p.must_not if isinstance(p.must_not, list) else []:
            yield f"procedure {p.id}", m
    for key in GRAPH_NODES:
        node = pkg.global_.get(key)
        listed = node.get("must_not") if isinstance(node, dict) else None
        for m in listed if isinstance(listed, list) else []:
            if isinstance(m, str):
                yield f"global.{key}", m


def _tool_typed(prop: Any) -> bool:
    """The argument takes a value of the tool itself: an enum or a boolean, also as one of the forms of ``anyOf``,
    ``oneOf`` or a list of types (an optional boolean is ``anyOf`` a boolean and null)."""
    if not isinstance(prop, dict):
        return False
    kind = prop.get("type")
    if prop.get("enum") or kind == "boolean" or (isinstance(kind, list) and "boolean" in kind):
        return True
    return any(_tool_typed(alt) for key in ("anyOf", "oneOf", "allOf") for alt in prop.get(key) or [])


def _unfounded_texts(
    pkg: PolicyPackage, text: _Text, by_name: Optional[Dict[str, Dict[str, Any]]]
) -> List[PackageError]:
    """Warnings for what the runtime shows or compares and the text does not contain: global rules, decisions, triage,
    housekeeping and escalation steps, definitions, notes, ``applies_when`` (verbatim, or up to three verbatim runs of words),
    template text (verbatim), a literal value and the fixed parts of a template value (verbatim). A literal on an argument
    with an enum, or a boolean, is a value of the tool and is not looked up."""
    out: List[PackageError] = []

    def warn(where: str, value: Any, strict: bool = False, most: int = 3) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        if not (text.has(value) if strict else text.in_parts(value, most)):
            out.append(
                PackageError(
                    where,
                    "not_in_text",
                    f"{where}: not found in the policy text{'' if strict else ' (as it is, or in up to 3 runs of its words)'}: "
                    f"{_show(value)}",
                    "warning",
                )
            )

    g = pkg.global_
    for i, r in enumerate(g.get("global_rules") if isinstance(g.get("global_rules"), list) else []):
        warn(f"global.global_rules[{i}]", r.get("rule") if isinstance(r, dict) else None)
    for key in GRAPH_NODES:
        node = g.get(key)
        for i, step in enumerate(
            node.get("steps") if isinstance(node, dict) and isinstance(node.get("steps"), list) else []
        ):
            warn(f"global.{key}.steps[{i}]", step)
    templates = g.get("templates")
    for key, t in (templates if isinstance(templates, dict) else {}).items():
        warn(f"global.templates.{key}", t.get("text") if isinstance(t, dict) else None, strict=True)
    definitions = g.get("definitions")
    for i, d in enumerate(definitions if isinstance(definitions, list) else []):
        if isinstance(d, dict):
            warn(f"global.definitions[{i}].term", d.get("term"), strict=True)
            warn(f"global.definitions[{i}].meaning", d.get("meaning"))
    for p in pkg.procedures:
        warn(f"procedure {p.id}: applies_when", p.applies_when)
        for i, note in enumerate(p.notes):
            warn(f"procedure {p.id}: notes[{i}]", note)
        for i, d in enumerate(p.decisions):
            # only the condition: `then` and `else` are mostly flow text ("continue with step 2")
            warn(f"procedure {p.id}: decisions[{i}].rule", d.get("rule") if isinstance(d, dict) else None)
    for o in pkg.obligations:
        values = o.values if isinstance(o.values, dict) else {}
        effect = o.effect if isinstance(o.effect, dict) else {}
        match = effect.get("match") if isinstance(effect.get("match"), dict) else {}
        listed = effect.get("tools")
        tools = (
            [t for t in listed if isinstance(t, str) and by_name and t in by_name]
            if isinstance(listed, list)
            else []
        )
        for name, spec in values.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("from") == "literal" and isinstance(spec.get("value"), str):
                arg_names = [
                    arg
                    for arg, m in match.items()
                    if isinstance(m, dict) and name in _placeholders(m.get("pattern"))
                ]
                typed = False
                for arg in arg_names:
                    for t in tools:
                        prop = _props(by_name[t]).get(arg)
                        typed = typed or _tool_typed(prop)
                if not typed:
                    warn(f"obligation {o.id}: value {name}", spec["value"], strict=True)
            elif spec.get("from") == "template" and isinstance(spec.get("pattern"), str):
                fixed = PLACEHOLDER_NAME.split(spec["pattern"].replace("{{", "").replace("}}", ""))[::2]
                for piece in fixed:
                    if len(re.findall(r"[A-Za-z0-9]", piece)) >= 4:
                        warn(f"obligation {o.id}: value {name}", piece.strip(), strict=True)
    return out


def _invented_errors(pkg: PolicyPackage, by_name: Optional[Dict[str, Dict[str, Any]]]) -> List[PackageError]:
    """Content that the policy text does not contain: quotes, step text and must-nots that are not its words, a step
    action that is no tool, a role the text never names. Skipped when the package carries no text."""
    text = _Text(pkg)
    if not text:
        return []
    out: List[PackageError] = []

    def not_verbatim(label: str, what: str, value: Any) -> None:
        out.append(
            PackageError(
                "", "quote_not_verbatim", f"{label}: {what} is not the words of the policy: {_show(value)}"
            )
        )

    for o in pkg.obligations:
        quote = o.source.get("quote") if isinstance(o.source, dict) else None
        if quote and not text.has(quote):
            not_verbatim(f"obligation {o.id}", "source.quote", quote)
    for p in pkg.procedures:
        for kind in ("preconditions", "exceptions", "templates", "escalations"):
            for i, item in enumerate(getattr(p, kind)):
                quote = item.get("quote") if isinstance(item, dict) else None
                if not quote:
                    out.append(
                        PackageError(p.id, "unfilled_slot", f"procedure {p.id}: {kind}[{i}] has no quote")
                    )
                elif not text.has(quote):
                    not_verbatim(f"procedure {p.id}", f"{kind}[{i}].quote", quote)
        for i, step in enumerate(p.steps):
            words = step.text or step.quote
            if words and not text.has(words):
                not_verbatim(f"procedure {p.id}", f"steps[{i}]", words)
            if step.action and step.action != "reply" and by_name is not None and step.action not in by_name:
                out.append(
                    PackageError(
                        p.id,
                        "unknown_tool",
                        f"procedure {p.id}: steps[{i}].action {step.action!r} is not a tool",
                    )
                )
    for where, must_not in _must_nots(pkg):
        if not text.must_not_ok(must_not):
            not_verbatim(where, "must_not", must_not)
    for where, must_not in _must_nots(pkg):
        if isinstance(must_not, str) and bare_must_not(must_not):
            out.append(
                PackageError(
                    where,
                    "bare_must_not",
                    f"{where}: the must-not {_show(must_not)} cannot be read alone (under 5 words, or it starts with a "
                    "pronoun): write the condition first",
                    "warning",
                )
            )
    out += _unfounded_texts(pkg, text, by_name)
    roles = [
        (r.id, role)
        for r in pkg.rules
        if r.type == "role_permission" and isinstance(r.rule, dict) and isinstance(r.rule.get("roles"), list)
        for role in r.rule["roles"]
    ]
    for rule_id, role in roles:
        if isinstance(role, str) and not text.has_words(_flat(role)):
            out.append(
                PackageError(
                    rule_id, "unknown_role", f"rule {rule_id}: the role {role!r} is not named in the policy"
                )
            )
    if (pkg.generator or {}).get("kind") == "skill":
        for r in pkg.rules:
            if r.type == "verify" and not r.why:
                out.append(
                    PackageError(
                        r.id,
                        "verify_without_why",
                        f"rule {r.id}: a verify rule needs `why`: what keeps a typed rule from fitting",
                        "warning",
                    )
                )
    return out
