"""Calls against one obligation: the test of one argument (``arg_outcome``), of one call, and of the calls of a run
(``check_calls``, ``match``). ``gaps_of`` says why an obligation cannot be fully checked."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from polisys.compile.handlers.tables import _props
from polisys.package.model import Obligation
from polisys.experimental.finish.ledger.constants import (
    ADDRESS_ARGS, CODE_ARGS, CODE_TOOL, MET, OPEN, UNKNOWN, WRITES_IN_CODE, WRITES_IN_SHELL, WRONG, _is_free_text,
)
from polisys.experimental.finish.ledger.text import (
    _as_text, _call_text, _fold_part, _names, _unescape, _without_secrets, fold,
)
from polisys.experimental.finish.ledger.value_patterns import GAP, INFERRED, Token, _compute_plan
from polisys.experimental.finish.ledger.cases import Case
from polisys.experimental.finish.ledger.values import plan_of

PASS, FAIL, NOT_SURE, NO_TEST, PARTIAL = "pass", "fail", "not_sure", "no_test", "partial"


def _schema_of(specs: Optional[Dict[str, Any]], tool: str, arg: str) -> Dict[str, Any]:
    spec = (specs or {}).get(tool)
    if not spec:
        return {}
    prop = (_props(spec) or {}).get(arg) or {}
    if not isinstance(prop, dict):
        return {}
    for key in ("anyOf", "oneOf"):
        for option in prop.get(key) or []:
            if isinstance(option, dict) and option.get("type") != "null":
                return {**prop, **option}
    return prop


def _is_json_text(value: str) -> bool:
    text = value.strip()
    if not text or text[0] not in "{[":
        return False
    try:
        return isinstance(json.loads(text), (dict, list))
    except ValueError:
        return False


def _kind(tool: str, arg: str, value: Any, schema: Dict[str, Any]) -> str:
    kind = schema.get("type")
    if arg in CODE_ARGS and CODE_TOOL.search(tool):
        return "code"
    if schema.get("enum") or kind == "boolean" or isinstance(value, bool):
        return "exact"
    if kind in ("integer", "number") or isinstance(value, (int, float)):
        return "number"
    if arg in ADDRESS_ARGS:
        return "address"
    if isinstance(value, dict) or kind == "object":
        return "json"
    if isinstance(value, (list, tuple)) or kind == "array":
        return "list"
    if isinstance(value, str) and _is_json_text(value):
        return "json"
    return "text"


def _regex(tokens: Sequence[Token], *, full: bool = False, skeleton: bool = False) -> Optional[re.Pattern]:
    """The tokens as a regular expression over folded text. A gap is any text; for the skeleton, so is an inferred value.
    ``None`` when nothing is left to compare. A contains test holds at word edges, so ``100`` is not in ``1000``, and
    ``2026-04-28`` is in ``2026-04-28T09:00:00``."""
    parts: List[Optional[str]] = []  # text, or None for a gap
    for text, trust, _ in tokens:
        if trust == GAP or (skeleton and trust == INFERRED):
            if not parts or parts[-1] is not None:
                parts.append(None)
        elif text:
            if parts and parts[-1] is not None:
                parts[-1] += text
            else:
                parts.append(text)
    if not any(p is not None and fold(p) for p in parts):
        return None
    folded = [None if p is None else _fold_part(p) for p in parts]
    if folded[0] is not None:
        folded[0] = folded[0].lstrip()
    if folded[-1] is not None:
        folded[-1] = folded[-1].rstrip()
    body = "".join(r".*?" if p is None else re.escape(p) for p in folded)
    if full:
        return re.compile(f"^{body}$", re.S)
    if any(p and "@" in p for p in folded):  # an address ends at its end: a@x.com is not in a@x.com.evil.org
        left = r"(?<![a-z0-9._+-])" if folded[0] and folded[0][0].isalnum() else ""
        right = r"(?!\.?[a-z0-9])(?!-)" if folded[-1] and folded[-1][-1].isalnum() else ""
        return re.compile(f"{left}{body}{right}", re.S)
    return re.compile(f"{_edge_before(folded[0])}{body}{_edge_after(folded[-1])}", re.S)


def _edge_before(text: Optional[str]) -> str:
    """A digit only stops at a digit, so a date is found in a timestamp; a letter stops at a letter or a digit."""
    if not text or not text[0].isalnum():
        return ""
    return r"(?<![0-9])" if text[0].isdigit() else r"(?<![a-z0-9])"


def _edge_after(text: Optional[str]) -> str:
    if not text or not text[-1].isalnum():
        return ""
    return r"(?![0-9])" if text[-1].isdigit() else r"(?![a-z0-9])"


def _elements(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [p for p in re.split(r"[,;]", value) if p.strip()]
    return [value]


def _numbers_equal(want: str, got: Any) -> Optional[bool]:
    try:
        return float(want) == float(str(got).strip())
    except ValueError:
        return None


def _id_against_name(tokens: Sequence[Token], value: Any) -> bool:
    """A name in the pattern against an id in the call: they cannot be compared, so the call does not contradict it. A value
    is an id when it is one word with a digit or an ``@``, or one word against a name of several words."""
    want = fold("".join(t[0] for t in tokens))
    got = fold(_as_text(value))
    if not got or " " in got or re.search(r"\d|@", want):
        return False
    return bool(re.search(r"\d|@", got)) or " " in want


def arg_outcome(tool: str, arg: str, tokens: Sequence[Token], value: Any, schema: Dict[str, Any]) -> str:
    """One argument of one call against its pattern: ``pass``, ``fail`` (a fixed part differs), ``not_sure`` (only an inferred
    part differs, or code and JSON do not show it), ``no_test`` (nothing is known to compare), or ``partial`` (the fixed text
    fits and the value of the case is not known, so another case's call fits as well)."""
    outcome = _arg_outcome(tool, arg, tokens, value, schema)
    if outcome == PASS and any(t[1] == GAP for t in tokens) and not any(t[1] == INFERRED for t in tokens):
        return PARTIAL
    return outcome


def _arg_outcome(tool: str, arg: str, tokens: Sequence[Token], value: Any, schema: Dict[str, Any]) -> str:
    kind = _kind(tool, arg, value, schema)
    if _regex(tokens) is None:
        return NO_TEST
    if kind == "code" or kind == "json":
        text = fold(_unescape(_as_text(value)))
        return PASS if _regex(tokens).search(text) else NOT_SURE
    if kind in ("exact", "number"):
        text = fold(_as_text(value))
        if kind == "number" and all(t[1] != GAP for t in tokens):
            same = _numbers_equal("".join(t[0] for t in tokens), value)
            if same is not None:
                return PASS if same else _mismatch(tokens, value, full=True, text=text)
        return (
            PASS if _regex(tokens, full=True).search(text) else _mismatch(tokens, value, full=True, text=text)
        )
    if kind in ("address", "list"):
        elements = _elements(value)
        pattern = _regex(tokens)
        if any(pattern.search(fold(_unescape(_as_text(e)))) for e in elements):
            return PASS
        if any(isinstance(e, (dict, list)) for e in elements):
            return NOT_SURE  # objects can say the same in other words
        if len(elements) == 1:
            return _mismatch(tokens, elements[0], full=False, text=fold(_as_text(elements[0])))
        return (
            NOT_SURE
            if any(_id_against_name(tokens, e) for e in elements)
            else _mismatch(tokens, "", full=False, text=fold(" ".join(_as_text(e) for e in elements)))
        )
    text = fold(_as_text(value))
    return PASS if _regex(tokens).search(text) else _mismatch(tokens, value, full=False, text=text)


def _mismatch(tokens: Sequence[Token], value: Any, *, full: bool, text: str) -> str:
    if _id_against_name(tokens, value):
        return NOT_SURE
    skeleton = _regex(tokens, full=full, skeleton=True)
    if skeleton is not None and not skeleton.search(text):
        return FAIL
    return NOT_SURE


def _shows_writing(args: Dict[str, Any]) -> bool:
    """Does a code call show a sign of writing? The limit of this check is in ``WRITES_IN_CODE``."""
    for arg in CODE_ARGS:
        text = _unescape(_as_text(args.get(arg)))
        if text and (WRITES_IN_CODE.search(text) or (arg == "command" and WRITES_IN_SHELL.search(text))):
            return True
    return False


@dataclass
class CallOutcome:
    status: str  # met, unknown or wrong
    compared: bool = False  # a value of the case was compared, and it passed
    note: str = ""
    evidence: bool = True  # a call that does not meet the item still says something about it


def _call_outcome(
    tool: str,
    args: Dict[str, Any],
    plan: Dict[str, List[Token]],
    specs: Optional[Dict[str, Any]],
    keys: Optional[Sequence[str]] = None,
) -> CallOutcome:
    """One call against an item. A call that does not meet the item is evidence about it only if it is about this case: it
    names a key of the case, or passes an identifying argument. Otherwise one unrelated call of a tool would silence each
    missing item of that tool. With ``keys`` ``None`` no case is known, and the call counts. A code call always counts,
    because code can say the same in other words; so does a call where nothing could be compared."""
    results: List[Tuple[str, str]] = []
    code_tool = CODE_TOOL.search(tool) is not None
    for arg, tokens in plan.items():
        if arg in CODE_ARGS and code_tool and arg not in args:
            continue  # the pattern of the other code tool
        if arg not in args or args[arg] is None:
            results.append((NOT_SURE if _regex(tokens) is not None else NO_TEST, arg))
            continue
        results.append((arg_outcome(tool, arg, tokens, args[arg], _schema_of(specs, tool, arg)), arg))
    fails = [a for r, a in results if r == FAIL]
    doubts = [a for r, a in results if r == NOT_SURE]
    passed = [a for r, a in results if r == PASS]
    vague = [a for r, a in results if r in (PARTIAL, NO_TEST)]
    about_it = (
        code_tool
        or keys is None
        or any(not _is_free_text(a) for a in passed)
        or _names(keys, _call_text(args))
        or (bool(vague) and not fails and not doubts)
    )
    if fails:
        return CallOutcome(
            WRONG, False, f"{tool}: {', '.join(fails)} differs from what the policy prescribes", about_it
        )
    if doubts:
        return CallOutcome(UNKNOWN, False, f"{tool}: code cannot tell for {', '.join(doubts)}", about_it)
    if code_tool and not passed:  # a code tool matches only on a literal fragment
        return CallOutcome(UNKNOWN, False, f"{tool}: no literal fragment to find in the code", about_it)
    if code_tool and not _shows_writing(args):
        return CallOutcome(
            UNKNOWN, False, f"{tool}: the code holds the fragment and shows no sign of writing", about_it
        )
    if vague and not passed:  # nothing was compared, so nothing tells this case from another
        return CallOutcome(
            UNKNOWN, False, f"{tool}: no known value to compare for {', '.join(vague)}", about_it
        )
    return CallOutcome(MET, bool(passed), f"{tool}({_as_text(_without_secrets(args))[:300]})")


@dataclass
class Checked:
    status: str
    reason: str = ""
    # indexes of the calls that compared a value and passed
    met_calls: List[int] = field(default_factory=list)


def check_calls(
    ob: Obligation,
    write_log: Sequence[Dict[str, Any]],
    plan: Dict[str, List[Token]],
    *,
    specs: Optional[Dict[str, Any]] = None,
    wrong_values: str = "known_origin_only",
    taken: Iterable[int] = (),
    answer: Optional[str] = None,
    tools_in_run: Optional[Set[str]] = None,
    keys: Optional[Sequence[str]] = None,
    other_keys: Sequence[str] = (),
) -> Checked:
    """The status of one obligation from the calls that succeeded. ``taken``: calls that meet another item. They still meet
    this one if they match, and a call that does not match this one is then no evidence against it. ``keys``: the keys of
    the item's case; ``None``: no case is known. ``other_keys``: of the other cases: a call that names one of them and none
    of this case's is another case's call, and says nothing here."""
    effect = ob.effect if isinstance(ob.effect, dict) else {}
    tools = [t for t in effect.get("tools") or [] if isinstance(t, str)]
    if ob.check != "code":
        return Checked(UNKNOWN, f"check {ob.check} is not built: only code checks run")
    if not tools:
        return _no_tool(ob, plan, effect, answer)
    if tools_in_run is not None and not set(tools) & tools_in_run:
        return Checked(UNKNOWN, f"{', '.join(tools)} not in this run")
    taken = set(taken)
    notes: List[str] = []
    doubt = False
    met: List[int] = []
    met_note = ""
    for i, call in enumerate(write_log):
        if call.get("tool") not in tools:
            continue
        if other_keys and _names(other_keys, _call_text(call.get("args"))):
            if not (keys and _names(keys, _call_text(call.get("args")))):
                continue
        outcome = _call_outcome(call["tool"], call.get("args") or {}, plan, specs, keys)
        if outcome.status == MET:
            met_note = met_note or outcome.note
            if outcome.compared:
                met.append(i)
        elif i not in taken and outcome.evidence:
            notes.append(outcome.note)
            doubt = doubt or outcome.status == UNKNOWN
    if met_note:
        return Checked(MET, met_note, met)
    if not notes:
        return Checked(OPEN, f"no successful call of {', '.join(tools)} matches")
    if doubt or wrong_values == "never":  # one call that may be right is enough to say nothing
        return Checked(UNKNOWN, "; ".join(notes)[:400])
    return Checked(WRONG, "; ".join(notes)[:400])


def _no_tool(ob: Obligation, plan: Dict[str, Any], effect: Dict[str, Any], answer: Optional[str]) -> Checked:
    """An effect with no tool. A reply can be read when the policy gives its words; otherwise code cannot tell."""
    if effect.get("kind") != "reply" or answer is None:
        return Checked(UNKNOWN, f"a {effect.get('kind')} with no tool: code has no call to compare")
    wanted = [
        v
        for v in (ob.values if isinstance(ob.values, dict) else {}).values()
        if isinstance(v, dict) and v.get("from") == "literal" and v.get("value")
    ]
    if wanted and all(fold(v["value"]) in fold(answer) for v in wanted):
        return Checked(MET, "the reply holds the words of the policy")
    return Checked(UNKNOWN, "a reply can say the same in other words")


def match(
    ob: Obligation,
    write_log: Sequence[Dict[str, Any]],
    *,
    case: Optional[Case] = None,
    store: Any = None,
    today: str = "",
    specs: Optional[Dict[str, Any]] = None,
    wrong_values: str = "known_origin_only",
    answer: Optional[str] = None,
) -> str:
    """The status of one obligation against the calls that succeeded: ``open``, ``met``, ``unknown`` or ``wrong``."""
    return check_calls(
        ob,
        write_log,
        plan_of(ob, case, store, today),
        specs=specs,
        wrong_values=wrong_values,
        answer=answer,
        keys=case.keys if case is not None and case.keys else None,
    ).status


def gaps_of(ob: Obligation) -> List[str]:
    """Why the ledger cannot fully check this obligation, for a report on a package. Empty: it can."""
    out: List[str] = []
    effect = ob.effect if isinstance(ob.effect, dict) else {}
    tools = effect.get("tools") or []
    if ob.check != "code":
        out.append(f"check_{ob.check}")
    if not tools:
        out.append(f"{effect.get('kind')}_no_tool")
    elif any(CODE_TOOL.search(str(t)) for t in tools) and not effect.get("match"):
        out.append("code_tool_no_pattern")
    specs = ob.values if isinstance(ob.values, dict) else {}
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            out.append("value_not_an_object")
        elif spec.get("from") == "computed" and _compute_plan(spec) is None:
            out.append("computed_no_code")
    return sorted(set(out))
