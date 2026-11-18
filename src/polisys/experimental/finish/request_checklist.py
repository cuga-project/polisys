"""What the user request asks for, checked against the agent's writes (a deliverables checklist).

The layer already checks policies; many failed tasks instead stop before an action the request names ("send the
summary to the controller", "post in #executive", "send a report"). This hook reads only the user request (which it
is given in ``on_user_message``) and the agent's own calls; it never reads data or policy sources.

- ``on_user_message``: one model call lists the deliverables: an action, the kind of tool that does it (a word of a
  tool name), the targets (recipient, channel, record) and the values the result must contain. Targets and values
  are kept only when they appear in the request text, so the model cannot add requirements.
- ``before_tool_call`` (code): when the model bound an allowed value to an argument because the request states it
  ("send a report" -> ``type = report``), a call of that tool with another value is reminded once. Reminders go to
  ``before_llm_call`` (a ``before_tool_call`` message blocks).
- ``after_tool_call``: successful writes are kept.
- ``before_reply`` (code): a deliverable is done when a successful call of its tool has the bound values and every
  word of its literal targets and values (addresses, #channels, quoted text) in its arguments or result.
  The per-call value reminder (``enum_check``) is off by default: a value set later in the run is normal
  (``resolved`` then ``closed``), so the reply check judges the bound values. Missing ones send the agent back once, naming them.

Nothing here is specific to a benchmark: it uses the request text, the tool names and the tool schemas.

``obligations_from_request`` is the same extraction as the producer ``request`` of the obligation ledger (design section 5):
it gives obligations instead of a hook, and the ledger checks and sends. ``DeliverablesHooks`` is not built by the runtime
any more (a profile with a producer has only the ledger send follow-ups); the class stays for the replay tools.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from polisys.hooks import AgentHooks, HookResult

from .llm import parse_json
from .ruleset import canon, is_write_tool

SYSTEM = "You list what a user request asks to be done, so that code can check it was done. Output JSON only."
USER = """Tools that change something (name(argument=[allowed values], ...)):
{tools}

User request:
<<<
{request}
>>>

List each action the request explicitly asks for that changes something (sending, posting, creating, updating,
assigning, linking, closing). Skip reading and searching. For each give:
- "action": a short description;
- "tool": the one tool above that does it;
- "args": only for arguments with allowed values, and only when the request itself states the value for this action
  (for example "send a report" -> {{"type": "report"}}); use the allowed value; otherwise {{}};
- "targets": recipients, channels, people or record names the request names for it, copied exactly from the request;
- "values": short words or phrases the request requires in the result (a quoted text, a name), copied exactly.
Only what the request says. Return {{"items": [...]}}"""

DISPLAY_ID = re.compile(r"[A-Za-z]{2,}[-_]?\d{3,}")  # a record number the agent maps to an internal ID first
KIND_STOP = {
    "create",
    "update",
    "link",
    "add",
    "set",
    "new",
    "details",
    "to",
    "for",
    "get",
    "send",
    "record",
    "by",
    "delete",
    "remove",
    "map",
    "assign",
    "post",
    "a",
    "an",
    "the",
}
STOP_VALUES = {"new", "open", "none", "true", "false", "other", "all", "yes", "no"}
MAX_FOLLOWUPS = 1


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", canon(text).lower())


def literal(text: str, request: str) -> bool:
    """A target or value code can find in a call: an address, a #channel, or text quoted in the request. A description
    ("the user assigned to the incident") is not checked; the action itself still is."""
    t = text.strip()
    if "@" in t or t.startswith("#"):
        return True
    return any(
        f"{q}{t}{e}" in request for q, e in (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("`", "`"))
    )


def same_kind(a: str, b: str) -> bool:
    """Tools for the same kind of record (a shared name word other than a verb): ``update_service`` and
    ``update_configuration_item`` are not, ``link_new_incident_sla`` and ``update_incident_sla_details`` are."""
    if a == b:
        return True
    wa, wb = (
        set(re.findall(r"[a-z]+", a.lower())) - KIND_STOP,
        set(re.findall(r"[a-z]+", b.lower())) - KIND_STOP,
    )
    return bool(wa & wb)


def _flat(value: Any) -> str:
    return canon(json.dumps(value, default=str) if not isinstance(value, str) else value).lower()


class DeliverablesHooks(AgentHooks):
    def __init__(
        self,
        model: Optional[Callable[[str, str], str]],
        tool_specs: List[Dict[str, Any]],
        is_write: Callable[[str], bool] = is_write_tool,
        enum_check: bool = False,
        reply_check: bool = True,
        note_channel: str = "tool",
    ):
        self.model = model
        self.specs = {t["name"]: t for t in tool_specs}
        self.is_write = is_write
        self.enum_check = enum_check
        self.reply_check = reply_check
        self.note_channel = note_channel
        self.request = ""
        self.items: List[Dict[str, Any]] = []
        self.writes: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.records: List[Dict[str, Any]] = []
        self._reminded: set = set()
        self._followups = 0

    # -- request ------------------------------------------------------------------------------------------------------
    def on_user_message(self, message: str, thread_id: Optional[str]) -> HookResult:
        if self.request:  # a follow-up turn (the hook's own message) is not a new request
            return HookResult()
        self.request = message or ""
        if self.model is not None and self.reply_check:
            self.items = self.extract(self.request)
            self.records.append({"hook": "on_user_message", "outcome": "deliverables", "items": self.items})
        return HookResult()

    def extract(self, request: str) -> List[Dict[str, Any]]:
        names = sorted(n for n in self.specs if self.is_write(n))
        lines = []
        for n in names:
            props = (self.specs[n].get("inputSchema") or {}).get("properties") or {}
            enums = [f"{a}=[{'|'.join(map(str, p['enum']))}]" for a, p in props.items() if p.get("enum")]
            lines.append(f"{n}({', '.join(enums)})")
        try:
            data = parse_json(
                self.model(SYSTEM, USER.format(tools="\n".join(lines)[:12000], request=request[:8000]))
            )
        except Exception as e:  # no checklist is better than a wrong one
            self.records.append(
                {"hook": "on_user_message", "outcome": "deliverables_error", "reason": str(e)[:200]}
            )
            return []
        req = canon(request).lower()
        items = []
        for it in data.get("items") or []:
            if not isinstance(it, dict) or it.get("tool") not in self.specs:
                continue
            tool = it["tool"]
            props = (self.specs[tool].get("inputSchema") or {}).get("properties") or {}
            args = {}
            for a, v in (it.get("args") or {}).items() if isinstance(it.get("args"), dict) else []:
                enum = [str(x) for x in (props.get(a) or {}).get("enum") or []]
                if str(v) in enum and _words(str(v)) and set(_words(str(v))) <= set(_words(request)):
                    args[a] = str(v)  # an allowed value the request itself says

            def kept(xs):
                xs = [xs] if isinstance(xs, str) else xs if isinstance(xs, list) else []
                return [
                    x
                    for x in xs
                    if isinstance(x, str) and len(x) > 1 and canon(x).lower() in req and literal(x, request)
                ]

            item = {
                "action": str(it.get("action") or "")[:160],
                "tool": tool,
                "args": args,
                "targets": kept(it.get("targets")),
                "values": kept(it.get("values")),
            }
            items.append(
                item
            )  # a tool name alone is matched by record kind, so another tool of the same kind counts
        return items

    # -- writes -------------------------------------------------------------------------------------------------------
    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> HookResult:
        if not self.enum_check or not self.is_write(tool):
            return HookResult()
        details = []
        for it in self.items:
            if it["tool"] != tool:
                continue
            for arg, want in it["args"].items():
                got = args.get(arg)
                if (got is not None and str(got).lower() == want.lower()) or (tool, arg) in self._reminded:
                    continue
                self._reminded.add((tool, arg))
                self.notes.append(
                    f"REQUEST CHECK: the request asks for {arg} '{want}' ({it['action']}), but this "
                    f"{tool} call has {arg} '{got}'. Use '{want}' unless the request means something else."
                )
                rec = {
                    "hook": "before_tool_call",
                    "tool": tool,
                    "rule": "request_value",
                    "outcome": "remind",
                    "arg": arg,
                    "want": want,
                    "got": got,
                }
                self.records.append(rec)
                details.append(rec)
        return HookResult(details=details)

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> HookResult:
        failed = isinstance(result, dict) and ("error" in result or result.get("success") is False)
        if self.is_write(tool) and not failed:
            self.writes.append({"tool": tool, "args": args, "result": result})
        if self.note_channel == "tool" and self.notes:  # value reminders go with the call they are about
            notes, self.notes = self.notes, []
            return HookResult(message="\n".join(notes))
        return HookResult()

    def before_llm_call(self, thread_id: Optional[str]) -> HookResult:
        notes, self.notes = self.notes, []
        return HookResult(message="\n".join(notes) or None)

    # -- reply --------------------------------------------------------------------------------------------------------
    def _same_kind(self, a: str, b: str) -> bool:
        return same_kind(a, b)

    def missing(self) -> List[Dict[str, Any]]:
        everything = " ".join(_flat(w["args"]) + " " + _flat(w.get("result")) for w in self.writes)
        seen = set(_words(everything))
        out = []
        for it in self.items:
            literal_ok = all(
                set(_words(x)) <= seen
                for x in it["targets"] + it["values"]
                if not DISPLAY_ID.fullmatch(x.strip())
            )
            done = literal_ok and any(
                self._same_kind(w["tool"], it["tool"])
                and all(
                    str(w["args"].get(a, "")).lower() == v.lower()
                    for a, v in it["args"].items()
                    if a in w["args"] or w["tool"] == it["tool"]
                )
                for w in self.writes
            )
            if not done:
                out.append(it)
        return out

    def before_reply(self, answer: str) -> HookResult:
        if not self.reply_check or not self.items or self._followups >= MAX_FOLLOWUPS:
            return HookResult()
        gaps = self.missing()
        rec = {
            "hook": "before_reply",
            "rule": "deliverables",
            "outcome": "followup" if gaps else "complete",
            "missing": gaps,
        }
        self.records.append(rec)
        if not gaps:
            return HookResult(details=[rec])
        self._followups += 1
        lines = [
            "REQUEST CHECK: before you finish, the request asks for these and no successful call has done them yet:"
        ]
        for it in gaps:
            need = "; ".join(
                x
                for x in [
                    ", ".join(it["targets"]),
                    ", ".join(f"'{v}'" for v in it["values"]),
                    ", ".join(f"{a}={v}" for a, v in it["args"].items()),
                ]
                if x
            )
            lines.append(f"- {it['action']} ({need})")
        lines.append("Do them now if they are still needed, or say why they are not.")
        return HookResult(message="\n".join(lines), details=[rec])


# -- the producer ``request`` of the obligation ledger ------------------------------------------------------------------

REQUEST_USER = USER.replace(
    "Only what the request says. Return",
    '- "quote": the words of the request that ask for it, copied exactly, as one unbroken span (at least 8 characters);\n'
    '- "asked": true only if the request itself asks the agent to do this work; false if the request only mentions it\n'
    "  (an address or a ticket named in passing, a thing to know for context, work somebody else does).\n"
    "Only what the request says. Return",
)
MIN_QUOTE_CHARS = 8
ADDRESS_ARGS = (
    "to",
)  # the one argument that holds an address of the message, when the request names one address


def _string_args(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    props = (spec.get("inputSchema") or {}).get("properties") or {}
    return {a: p for a, p in props.items() if isinstance(p, dict)}


def obligations_from_request(
    model: Callable[[str, str], str],
    request: str,
    specs: List[Dict[str, Any]],
    level: str = "observe",
    is_write: Callable[[str], bool] = is_write_tool,
) -> List[Any]:
    """What the user request asks to be done, as obligations of the producer ``request`` (design section 5). One model call
    reads the request and the schemas of the tools that write; code keeps an item only if

    - its tool is a tool that writes, from ``specs``;
    - it has a ``quote``: words of the request itself, the whole quote found word by word (an item with no evidence in the request is not
      created, so the model cannot add requirements) and the model says the request asks for the work (``asked``); code cannot
      tell an ask from a mention, so a wrong answer of the model gets through: Task 4.3 measures it;
    - each target and value is copied from the request and can be found in a call (an address, a #channel, quoted text);
      an allowed value of an enum argument is kept when the request says it.

    The obligation names the tools of the same kind of record that can do it (``same_kind``) and, only where an argument can
    be named, a pattern for it: an enum value, a single address (argument ``to``), a single channel (the one argument whose
    name holds ``channel``). Any other target or value stays a value no pattern names, which the follow-up shows and code
    never compares. The items open at the user message. Raises what the model call or the parsing raises: the ledger writes
    that down and goes on with no request item (no checklist is better than a wrong one)."""
    from .model import Obligation
    from .obligations import _hay
    from .obligations import _words as _owords

    by_name = {t["name"]: t for t in specs if isinstance(t, dict) and t.get("name")}
    names = sorted(n for n in by_name if is_write(n))
    if not names or not (request or "").strip():
        return []
    lines = []
    for n in names:
        enums = [
            f"{a}=[{'|'.join(map(str, p['enum']))}]"
            for a, p in _string_args(by_name[n]).items()
            if p.get("enum")
        ]
        lines.append(f"{n}({', '.join(enums)})")
    data = parse_json(
        model(SYSTEM, REQUEST_USER.format(tools="\n".join(lines)[:12000], request=request[:8000]))
    )
    req, hay = canon(request).lower(), _hay(request)

    def kept(xs: Any) -> List[str]:
        xs = [xs] if isinstance(xs, str) else xs if isinstance(xs, list) else []
        return [
            x
            for x in xs
            if isinstance(x, str) and len(x) > 1 and canon(x).lower() in req and literal(x, request)
        ]

    out: List[Any] = []
    for it in data.get("items") or [] if isinstance(data, dict) else []:
        if not isinstance(it, dict) or it.get("tool") not in names:
            continue
        quote = re.sub(r"\s+", " ", str(it.get("quote") or "")).strip()
        if it.get("asked") is not True:
            continue  # the model says the request only mentions it
        whole = " ".join(_owords(quote))
        if len(whole) < MIN_QUOTE_CHARS or f" {whole} " not in hay:
            continue  # no evidence in the request: the whole quote must be there, not one fragment of it
        tool = it["tool"]
        props = _string_args(by_name[tool])
        enums: Dict[str, str] = {}
        for a, v in (it.get("args") or {}).items() if isinstance(it.get("args"), dict) else []:
            allowed = [str(x) for x in (props.get(a) or {}).get("enum") or []]
            if str(v) in allowed and _words(str(v)) and set(_words(str(v))) <= set(_words(request)):
                enums[a] = str(v)
        targets, values = kept(it.get("targets")), kept(it.get("values"))
        tools = [tool] + [
            n
            for n in names
            if n != tool and same_kind(n, tool) and all(a in _string_args(by_name[n]) for a in enums)
        ]  # the named tool first
        match: Dict[str, Any] = {}
        spec_values: Dict[str, Any] = {}
        for a, v in enums.items():
            match[a] = {"pattern": "{" + a + "}"}
            spec_values[a] = {"from": "literal", "value": v}
        addresses = [t for t in targets if "@" in t]
        channels = [t for t in targets if t.startswith("#")]

        def named(arg: str) -> bool:
            return all(arg in _string_args(by_name[n]) for n in tools) and arg not in match

        if len(addresses) == 1 and named("to"):
            match["to"] = {"pattern": "{target_to}"}
            spec_values["target_to"] = {"from": "literal", "value": addresses[0]}
        chan_args = sorted(
            {a for n in tools for a in _string_args(by_name[n]) if re.search("channel", a, re.I)}
        )
        if len(channels) == 1 and len(chan_args) == 1 and named(chan_args[0]):
            match[chan_args[0]] = {"pattern": "{target_channel}"}
            spec_values["target_channel"] = {"from": "literal", "value": channels[0]}
        shown = {v["value"] for v in spec_values.values()}
        for k, text in enumerate([t for t in targets if t not in shown], 1):
            spec_values[f"target_{k}"] = {"from": "literal", "value": text}
        for k, text in enumerate([v for v in values if v not in shown], 1):
            spec_values[f"value_{k}"] = {"from": "literal", "value": text}
        out.append(
            Obligation(
                id=f"req_{len(out) + 1}",
                origin="request",
                scope={},
                effect={"kind": "write", "tools": tools, "match": match},
                values=spec_values,
                check="code",
                level=level,
                source={"section": "request", "quote": quote},
            )
        )
    return out
