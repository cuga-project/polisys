"""Default values of the policy, enforced by code.

A policy often states the value to use when the request gives none ("priority: if not provided, default is ..."). A tool
often has another default of its own, so a call that leaves the argument out writes the tool's value, not the policy's.

``build_defaults`` asks a model once per policy text and tool set for the stated defaults: tool, argument, value, and the
policy line. Code keeps an entry only if the tool is a write tool of the run, the argument is one of its arguments, the
line is in the policy text word for word, the value is a whole word of that line, and the tool accepts the value (when
its schema lists allowed values). Two entries with different values for one argument are both dropped.

``DefaultValueHooks`` then decides by code alone: a call of such a tool with the argument missing or empty is blocked
once per call, with the default and the policy line. The agent passes the default, or the value the request gives.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

from polisys.hooks import AgentHooks, HookResult
from polisys.experimental.guidance.tool_notes import _arguments, allowed_values, quoted_in, tool_lines
from polisys.experimental.guidance.work_method_notes import call_key
from polisys.llm import parse_json
from polisys.runtime.context import Piece

if TYPE_CHECKING:
    from polisys.runtime.assembly import Assembly


@dataclass
class Settings:
    """Default values the policy states for arguments of write tools, listed once per policy text and tool set and
    enforced by code: a write call that leaves such an argument out is blocked once with the default (``experimental/checks/default_values.py``)."""

    mode: Literal["off", "on"] = "off"
    model: Optional[str] = None  # the model that lists the defaults; "agent": the model the agent runs on
    cache_dir: Optional[str] = None
    # stated: also the values the policy requires for an argument when the call sets another one
    scope: Literal["defaults", "stated"] = "defaults"


PROMPT_VERSION = "1"
SYSTEM = "You list the default values that a written policy states for arguments of tools. Output JSON only."
USER = """POLICY:
{policy}

TOOLS (with their arguments):
{tools}

List each argument of a tool above for which the policy states the value to use when the request gives none (a default).
- Only defaults the policy states in words, with the value. Not a value the policy merely allows, and not a format.
- "value" is the default exactly as the policy writes it. "quote" is the policy line that states it, word for word.
- Use only tools and arguments from the list. When the policy states no default, return an empty list.

Return {{"defaults": [{{"tool": "<name>", "argument": "<argument>", "value": "<default>", "quote": "<policy line>"}}]}}"""

# scope "stated": the defaults, and the values the policy requires for one argument when the call sets another one.
USER_STATED = """POLICY:
{policy}

TOOLS (with their arguments):
{tools}

List the values the policy states for arguments of the tools above. Two kinds:
- "default": the value to use when the request gives none. A default of a new record belongs to the tool that creates
  the record, not to a tool that updates it.
- "required_when": the policy says that when a call does one thing, another argument must have a stated value (for
  example: when a record gets an owner, its state must be a given state). Give in "when_argument" the argument of the
  same tool whose presence in the call means that thing.
Rules:
- Only values the policy states in words. Not a value the policy merely allows, and not a format.
- "value" is the value exactly as the policy writes it. "quote" is the policy line that states it, word for word.
- Use only tools and arguments from the list. When the policy states none, return an empty list.

Return {{"defaults": [{{"tool": "<name>", "argument": "<argument>", "value": "<value>", "kind": "default | required_when", "when_argument": "<argument, for required_when>", "quote": "<policy line>"}}]}}"""

BLOCK_WHEN = (
    'The policy states: "{quote}". This call of {tool} sets {when_argument}, so the policy asks for {argument}={value}; '
    "the call passes {given}. Pass {argument}={value}, unless the request itself states another value for {argument}. "
    "Then make the call again."
)

BLOCK = (
    'The policy states a default for {argument}: {value}. Policy: "{quote}". This call of {tool} gives no {argument}, '
    "so the tool would use its own default. If the request gives no other value for it, pass {argument}={value}. Then "
    "make the call again."
)


def _has_word(text: str, value: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text, re.I) is not None


def check_defaults(
    raw: Any, policy_text: str, tool_specs: List[Dict[str, Any]], is_write: Callable[[str], bool], stated: bool = False
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """``(kept, dropped)``; each dropped entry carries the reason. ``stated``: values required when another argument is
    set are kept too (their ``kind`` is ``required_when``)."""
    specs = {s["name"]: s for s in tool_specs}
    kept: Dict[tuple, Dict[str, str]] = {}
    dropped: List[Dict[str, str]] = []
    clash = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        tool, arg = str(item.get("tool") or ""), str(item.get("argument") or "")
        value, quote = " ".join(str(item.get("value") or "").split()), str(item.get("quote") or "")
        why = ""
        if tool not in specs or not is_write(tool):
            why = "not a write tool of the run"
        elif arg not in _arguments(specs[tool]):
            why = "not an argument of the tool"
        elif not value or not quoted_in(quote, policy_text):
            why = "no value, or quote not in the policy text"
        elif not _has_word(quote, value):
            why = "the value is not in the quoted line"
        elif allowed_values(specs[tool], arg) and value not in allowed_values(specs[tool], arg):
            why = "the tool does not accept the value"
        kind = str(item.get("kind") or "default").strip().lower()
        when = str(item.get("when_argument") or "")
        if not why and stated and kind == "required_when":
            if when not in _arguments(specs[tool]) or when == arg:
                why = "no argument that tells when the value is required"
        elif not why and kind != "default":
            why = "not a default"
        key = (tool, arg, when if kind == "required_when" else "")
        if not why and key in kept and kept[key]["value"] != value:
            clash.add(key)
            why = "two defaults for one argument"
        if why:
            dropped.append({"tool": tool, "argument": arg, "why": why})
        else:
            kept[key] = {"tool": tool, "argument": arg, "value": value, "quote": " ".join(quote.split())}
            if kind == "required_when":
                kept[key].update(kind=kind, when_argument=when)
    for key in clash:
        kept.pop(key, None)
    return list(kept.values()), dropped


def build_defaults(
    policy_text: str,
    tool_specs: List[Dict[str, Any]],
    model: Callable[[str, str], str],
    is_write: Callable[[str], bool],
    cache_dir: Optional[str] = None,
    scope: str = "defaults",
) -> Dict[str, Any]:
    """``{"defaults": [...], "dropped": [...], "cached": bool}``. With a cache folder, parallel runs share one answer."""
    write_specs = [s for s in tool_specs if is_write(s["name"])]
    stated = scope == "stated"

    def make() -> Dict[str, Any]:
        user = (USER_STATED if stated else USER).format(policy=policy_text, tools=tool_lines(write_specs))
        for _ in range(2):  # an answer with no list (cut, or not JSON) is asked once more, then fails
            reply = parse_json(model(SYSTEM, user))
            if isinstance(reply.get("defaults"), list):
                kept, dropped = check_defaults(reply["defaults"], policy_text, tool_specs, is_write, stated)
                return {"defaults": kept, "dropped": dropped, "cached": False}
        raise ValueError("the model's answer holds no list of defaults")

    if not cache_dir:
        return make()
    import fcntl

    folder = Path(cache_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(
        [PROMPT_VERSION + ("s1" if stated else ""), policy_text, sorted(s["name"] for s in write_specs),
         getattr(model, "model_name", "")],
        ensure_ascii=False,
    )
    path = folder / f"defaults-{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:24]}.json"
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if path.exists():
                return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
            data = make()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
            return data
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, (str, list, dict)) and not value)


class DefaultValueHooks(AgentHooks):
    """Blocks once a write call that leaves out an argument for which the policy states a default. The list of defaults
    is read at the first tool call (from the cache, or from the model), not when the run is wired."""

    def __init__(self, load: Callable[[], Dict[str, Any]]) -> None:
        self._load = load
        self.data: Optional[Dict[str, Any]] = None
        self.told: List[str] = []
        self.records: List[Dict[str, Any]] = []

    def session_start(self, thread_id: Optional[str]) -> None:
        self.told = []

    def defaults(self) -> List[Dict[str, str]]:
        if self.data is None:
            try:
                self.data = self._load()
            except Exception as exc:  # the run goes on without the check, and the audit says so
                self.data = {"defaults": [], "dropped": []}
                self.records.append(
                    {"component": "default_values", "outcome": "defaults_failed", "reason": f"{type(exc).__name__}: {exc}"[:200]}
                )
            else:
                self.records.append(
                    {
                        "component": "default_values",
                        "outcome": "defaults_loaded",
                        "defaults": len(self.data.get("defaults") or []),
                        "dropped": len(self.data.get("dropped") or []),
                        "cached": bool(self.data.get("cached")),
                    }
                )
        return self.data.get("defaults") or []

    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> Optional[HookResult]:
        key = call_key(tool, args)  # once per call: two calls of one turn with the same gap are both blocked
        for d in self.defaults():
            if d["tool"] != tool or key in self.told:
                continue
            given = (args or {}).get(d["argument"])
            if d.get("kind") == "required_when":
                if _empty((args or {}).get(d["when_argument"])) or str(given).strip().lower() == d["value"].lower():
                    continue
                self.told.append(key)
                self.records.append(
                    {"component": "default_values", "outcome": "required_value_block", "tool": tool, "argument": d["argument"]}
                )
                return HookResult(message=BLOCK_WHEN.format(given="nothing" if _empty(given) else repr(given), **d))
            if not _empty(given):
                continue
            self.told.append(key)
            self.records.append(
                {"component": "default_values", "outcome": "default_block", "tool": tool, "argument": d["argument"]}
            )
            return HookResult(message=BLOCK.format(**d))
        return None


# -- wiring: what build_runtime adds to a run for this component (runtime/registry.py) ----------------------------------


def build(a: Assembly) -> Piece:
    from polisys.tool_kind import is_write_tool

    cfg = a.cfg.checkers.default_values
    specs = a.run_tool_specs()
    if not specs:
        raise a.fail("default_values", "checkers.default_values needs RunContext.tool_specs (the tools of the run)")
    model = a.own_model("default_values", cfg.model, timeout=180, max_tokens=8000)
    text = a.policy_text("default_values")
    hook = DefaultValueHooks(lambda: build_defaults(text, specs, model, is_write_tool, cfg.cache_dir, cfg.scope))
    return Piece(hooks=[hook])
