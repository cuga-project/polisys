"""Dry run of the policy wiring: a recording model factory and a stable description of what was wired.

Which hooks a run gets, with which options and which models, is hidden in constructors and closures. A wrong wiring gives
no error: it runs and measures something else. So the wiring has a description that two runs can compare: hook classes
and their public attributes, the tools and the note the layer adds, which slots of the run are used, every model made
with its name, ``timeout`` and ``max_tokens``, and whether the gate reads the store behind the agent's policy tools.

- ``RecordingModels`` has the factory methods of ``runtime.Models``. It records what the real factory would get, with the
  defaults of the real functions in ``llm.py`` and ``detectors.py``. Its models raise when called, so a dry run cannot
  make a model call.
- ``describe_wiring`` describes the parts; ``describe_parts`` describes a ``RuntimeParts``.

Long values appear as a length and a hash, so two descriptions differ when the content differs. Nothing here starts an
agent or opens a network connection.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import types
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import detectors, llm

SLOTS = ("run", "turn_note", "wrap_tools", "after_reply")
SHORT_TEXT = 200  # a longer text is described by its length and hash
SHORT_VALUE = 400  # a list or dict with a longer JSON form is described by its length and hash
MAX_DEPTH = 3
OWN_PREFIX = "polisys"
# A rule dict from a package has these two keys and the same rule from a build does not. RuleSetHooks reads `level` only
# where the rule's old outcome is kept, which the package parity check on real data shows, and never `statement_ids`.
RULE_KEYS_ONLY_IN_PACKAGE = frozenset({"level", "statement_ids"})
RULE_ATTRS = frozenset({"rules", "by_tool", "global_rules", "global_verify"})

_REAL = {
    "json": llm.openai_json_model,
    "openrouter_json": llm.openrouter_json_model,
    "detector": detectors.systemone_detector,
    "questions": detectors.systemone_questions,
}


class DryRunModelCall(RuntimeError):
    """A model was called in a dry run."""


class RecordingModels:
    """The model factory of a dry run. ``made`` lists every model the wiring asked for."""

    def __init__(self) -> None:
        self.made: List[Dict[str, Any]] = []
        self._by_usage: Dict[int, Dict[str, Any]] = {}

    def _make(self, kind: str, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        bound = inspect.signature(_REAL[kind]).bind(*args, **kwargs)
        bound.apply_defaults()
        entry = {"kind": kind, **{k: v for k, v in bound.arguments.items() if k != "api_key"}}

        def call(*_: Any, **__: Any) -> Any:
            raise DryRunModelCall(f"dry run: the {kind} model {entry.get('model')} was called")

        usage: Dict[str, Any] = {"calls": 0}
        call.model_name = entry.get("model")  # type: ignore[attr-defined]
        call.usage = usage  # type: ignore[attr-defined]
        self.made.append(entry)
        self._by_usage[id(usage)] = entry
        return call

    def json_model(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        return self._make("json", *args, **kwargs)

    def openrouter_json_model(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        return self._make("openrouter_json", *args, **kwargs)

    def detector(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        return self._make("detector", *args, **kwargs)

    def questions(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        return self._make("questions", *args, **kwargs)

    def find(self, value: Any) -> Optional[Dict[str, Any]]:
        """The entry of a model. The wrappers in ``trace.py`` keep the ``usage`` dict, so a wrapped model is found too."""
        usage = getattr(value, "usage", None)
        return self._by_usage.get(id(usage)) if isinstance(usage, dict) else None


def _callable_name(value: Any) -> str:
    """A named function of polisys by its name. A closure has no stable name wherever it is made, so it is only a closure."""
    module = getattr(value, "__module__", "") or ""
    name = getattr(value, "__qualname__", type(value).__name__)
    if "<locals>" in name:
        return "<closure>"
    return f"{module}.{name}" if module.startswith("polisys.") else "<function from outside polisys>"


def _plain(value: Any, drop: frozenset = frozenset()) -> Any:
    """JSON form of a value, with sets sorted, so that equal content gives equal text."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v, drop) for k, v in value.items() if k not in drop}
    if isinstance(value, (list, tuple)):
        return [_plain(v, drop) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(v, drop) for v in value), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value), drop)
    if callable(value):
        return _callable_name(value)
    return f"<{type(value).__name__}>"


def _sha(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _describe(
    value: Any, models: Optional[RecordingModels], depth: int = 0, drop: frozenset = frozenset()
) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= SHORT_TEXT else {"text_chars": len(value), "sha": _sha(value)}
    entry = models.find(value) if models else None
    if entry is not None:
        return {"model": dict(entry)}
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        plain = _plain(value, drop)
        if len(json.dumps(plain, sort_keys=True, ensure_ascii=False)) <= SHORT_VALUE:
            return plain
        return {"items": len(value), "sha": _sha(plain)}
    if type(value).__module__.startswith(OWN_PREFIX):
        return _describe_object(value, models, depth)
    if isinstance(value, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        return _callable_name(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return f"<{type(value).__name__}>"


def _describe_object(value: Any, models: Optional[RecordingModels], depth: int = 0) -> Dict[str, Any]:
    name = type(value).__name__
    if depth >= MAX_DEPTH:
        return {"class": name}
    attrs = {}
    for key, item in sorted(vars(value).items()):
        if key.startswith("_"):
            continue
        drop = RULE_KEYS_ONLY_IN_PACKAGE if name == "RuleSetHooks" and key in RULE_ATTRS else frozenset()
        attrs[key] = _describe(item, models, depth + 1, drop)
    return {"class": name, "attrs": attrs}


def _gate_reads_tools_store(hooks: Iterable[Any], tools: Iterable[Any]) -> Optional[Dict[str, bool]]:
    """Does a gate on a store see a procedure that the agent opens through the policy tools?

    The gate learns what is open from its store's ``calls``, so a store of its own would make it check almost nothing,
    with no error. This opens one procedure through the tool and looks; it then takes the call back. ``None`` when the
    wiring has no such gate."""
    opener = next((t for t in tools if t.name == "policy__get_procedure"), None)
    out = {}
    for hook in hooks:
        store = getattr(hook, "store", None)
        procs = getattr(store, "procs", None)
        if not isinstance(procs, dict):
            continue  # no store, or the fact store of a rule hook
        seen = False
        if opener is not None and procs:
            pid = next(iter(procs))
            before = len(store.calls)
            asyncio.run(opener.ainvoke({"ids": [pid]}))
            seen = any(pid in (c.get("expanded") or []) for c in store.calls[before:])
            del store.calls[before:]
        out[type(hook).__name__] = seen
    return out or None


def describe_wiring(
    hooks: Iterable[Any] = (),
    tools: Iterable[Any] = (),
    system_note: str = "",
    slots: Optional[Dict[str, Any]] = None,
    models: Optional[RecordingModels] = None,
) -> Dict[str, Any]:
    """The description of a wiring. ``tools``: only those the layer adds. ``slots``: slot name -> the object behind it, or
    a dict of the objects and values behind it."""
    hooks, tools = list(hooks), list(tools)
    slots = slots or {}

    def slot(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _describe(v, models, 1) for k, v in sorted(value.items())}
        return None if value is None else _describe(value, models)

    return {
        "hooks": [_describe(h, models) for h in hooks],
        "tools": [{"name": t.name, "description_sha": _sha(t.description or "")} for t in tools],
        "system_note": system_note,
        "slots": {name: slot(slots.get(name)) for name in SLOTS},
        "models": sorted((models.made if models else []), key=lambda m: json.dumps(m, sort_keys=True)),
        "gate_reads_tools_store": _gate_reads_tools_store(hooks, tools),
    }


def slot_target(fn: Any) -> Any:
    """What stands behind a slot of ``RuntimeParts``: ``fn.target``, the object of a bound method, or ``fn`` itself."""
    if fn is None:
        return None
    target = getattr(fn, "target", None) or getattr(fn, "__self__", None) or fn
    wiring = getattr(target, "wiring", None)
    return wiring() if callable(wiring) else target


def describe_parts(parts: Any, models: Optional[RecordingModels] = None) -> Dict[str, Any]:
    return describe_wiring(
        parts.hooks,
        parts.tools,
        parts.system_note,
        {name: slot_target(getattr(parts, name)) for name in SLOTS},
        models,
    )
