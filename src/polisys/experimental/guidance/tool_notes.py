"""Tool notes (``notes.tool_notes``): help for each tool, written once for a policy text and a tool set and shown in the
tool's own description, so the agent reads it whenever it considers that tool.

A note only helps. It never forbids a call and never narrows what a tool may be used for: the policy text already says what
is forbidden, and a restriction repeated on a tool makes an agent stop where the policy does not ask it to.

| Kind | Content | Kept only if |
|---|---|---|
| ``before`` | a condition of the policy for the call, and the tool that makes it true | ``fix_tool`` is a write tool of the run (a tool that only reads can check a condition, not make it true) |
| ``values`` | a value the policy fixes for one argument: a default, a format, a template | ``argument`` is an argument of the tool |
| ``after`` | a further step the policy requires in a given situation | ``then_tool`` is a tool of the run |

One model call writes the notes from the policy text and the tool list. Code keeps a note only if it passes the check of
its kind, its quote is in the policy text word for word, and its text is short. A ``before`` note whose remedy the policy
does not state (``inferred``) is kept only with ``unmet: fix_with_tool``: a decision of the deployment, not of the policy.

A conflict of lists (the policy's allowed values for an argument differ from the tool's own) is reported by the model in a
list of its own and written by code, only with ``authority.value_lists: tool`` and only when a value of the tool's list is
missing from the quoted policy list: the note then says that the tool's list is the one in force.

The notes of a policy text and tool set are cached on disk (``cache_dir``), so every run under the same policy gets the same
notes. Nothing here reads a task, a request or a grader.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

from polisys.llm import parse_json
from polisys.runtime.context import Piece

if TYPE_CHECKING:
    from polisys.runtime.assembly import Assembly


@dataclass
class Settings:
    """Notes on each tool, written once from the policy text and the tool list and shown in the tool descriptions
    (``experimental/guidance/tool_notes.py``): what to check before the call, which values the policy fixes, what to do after."""

    mode: Literal["off", "on"] = "off"
    model: Optional[str] = None  # the model that writes the notes; "agent": the model the agent runs on
    # A condition whose remedy the policy does not state: as_stated keeps no note for it; fix_with_tool keeps the note that
    # names an available tool which makes the condition true. A decision of the deployment, not of the policy.
    unmet: Literal["as_stated", "fix_with_tool"] = "as_stated"
    cache_dir: Optional[str] = None  # folder for the notes of each policy text and tool set; None: asked in every run
    # The kinds of note shown to the agent; all are written and cached either way. "conflict": the tool's list of allowed
    # values is in force where the policy's list lacks a value of it (only with authority.value_lists: tool).
    show: List[Literal["before", "values", "after", "conflict"]] = field(
        default_factory=lambda: ["before", "values", "after", "conflict"]
    )


PROMPT_VERSION = "5"
KINDS = ("before", "values", "after")
MAX_NOTE_CHARS = 360
MAX_TOOL_CHARS = 1400
MIN_QUOTE_CHARS = 12
HEADER = "Policy notes for this tool:"
LABELS = {"before": "Before calling", "values": "Values", "after": "After it returns"}

SYSTEM = "You prepare short notes for an assistant that will call tools under a written policy. Output JSON only."
USER = """POLICY:
{policy}

TOOLS (with their arguments):
{tools}

Write notes that help the assistant do its work under the policy. A note never forbids a call and never narrows what a tool may be used for: the policy text already says what is forbidden. There are three kinds of note:
- "before": the policy sets a condition for calling this tool, and one of the tools above can change the system so that the condition becomes true. Say how to check the condition and which tool makes it true, and give that tool in "fix_tool". A tool that only reads cannot be the fix_tool. When no tool above can make the condition true, write no note.
- "values": the policy fixes a value for one argument of this tool: a default to pass when the request gives none, a format, a template, a length. Give the argument in "argument". For a format or a template, say what each field of it stands for, in the policy's own words for its terms.
- "after": the policy requires a further step once this tool succeeded in a given situation. Say the situation, and give the tool that does the step in "then_tool". The step is added work for that situation; it does not limit other uses of either tool.

Separately, in "conflicts", report each argument for which the policy states a list of allowed values and that list lacks a value of the tool's own allowed values shown above. Give the tool, the argument, and the policy's list as the quote. An argument whose two lists hold the same values is not a conflict.

Rules:
- Each note names one tool from the list and quotes the policy statement it comes from, word for word, in "quote".
- For "before": when the policy itself states what to do if the condition is not met, write that and set "inferred" to false. When it does not, set "inferred" to true.
- A statement about one kind of record or one operation is noted only on the tools of that operation. Do not extend it to other tools.
- Use only tools and arguments from the list.
- One or two sentences per note, addressed to the assistant ("Check ...", "Pass ..."). A note says what to do. It does not say "halt", "stop", "do not" or "only".

Return {{"notes": [{{"tool": "<name>", "kind": "before|values|after", "text": "<note>", "quote": "<policy text>", "fix_tool": "<before only>", "argument": "<values only>", "then_tool": "<after only>", "inferred": false}}], "conflicts": [{{"tool": "<name>", "argument": "<argument>", "quote": "<the policy's list of allowed values>"}}]}}"""

CONFLICT = (
    "The tool accepts these values for {argument}: {values}. The policy text lists other values for it. The tool's list is "
    "the one in force: when the request asks for a value that the tool accepts, pass it."
)


@dataclass
class Note:
    tool: str
    kind: str
    text: str
    quote: str
    inferred: bool = False
    ref: str = ""  # before: the tool that fixes the condition; values: the argument; after: the tool of the next step
    conflict: bool = False  # values: the policy's list differs from the tool's own list


def _squash(text: str) -> str:
    """Lower case, one space between words, plain dashes and quotes, no Markdown marks: how a quote is compared."""
    text = (text or "").lower()
    text = text.translate({ord(c): "-" for c in "‐‑‒–—−"})
    text = text.translate({ord(c): "'" for c in "‘’"} | {ord(c): '"' for c in "“”"})
    text = re.sub(r"[*_`#>|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quoted_in(quote: str, policy_text: str) -> bool:
    """Is the quote in the policy text, word for word (case, spacing and Markdown marks aside)?"""
    q = _squash(quote)
    return len(q) >= MIN_QUOTE_CHARS and q in _squash(policy_text)


class NotesError(ValueError):
    """The model's answer holds no list of notes (cut at the token limit, or not JSON)."""


def tool_lines(tool_specs: List[Dict[str, Any]]) -> str:
    """The tools for a prompt: name and description, then each argument with its description and allowed values."""
    lines = []
    for spec in tool_specs:
        schema = spec.get("inputSchema") or {}
        required = set(schema.get("required") or [])
        desc = " ".join(str(spec.get("description") or "").split())[:300]
        lines.append(f"- {spec['name']}: {desc}")
        for name, prop in (schema.get("properties") or {}).items():
            prop = prop if isinstance(prop, dict) else {}
            bits = [" ".join(str(prop.get("description") or "").split())[:160]]
            options = [prop] + [a for a in prop.get("anyOf") or [] if isinstance(a, dict)]
            enum = next((o["enum"] for o in options if o.get("enum")), None)
            if enum:
                bits.append("allowed: " + ", ".join(map(str, enum)))
            text = "; ".join(b for b in bits if b)
            mark = " (required)" if name in required else ""
            lines.append(f"    {name}{mark}: {text}" if text else f"    {name}{mark}")
    return "\n".join(lines)


def _arguments(spec: Dict[str, Any]) -> Dict[str, Any]:
    return (spec.get("inputSchema") or {}).get("properties") or {}


def allowed_values(spec: Dict[str, Any], argument: str) -> List[str]:
    """The tool's own allowed values of an argument; empty when its schema has none."""
    prop = _arguments(spec).get(argument)
    prop = prop if isinstance(prop, dict) else {}
    options = [prop] + [a for a in prop.get("anyOf") or [] if isinstance(a, dict)]
    return [str(v) for v in next((o["enum"] for o in options if o.get("enum")), [])]


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(_squash(word))}(?![\w-])", _squash(text)) is not None


def check_notes(
    raw: Any,
    policy_text: str,
    tool_specs: List[Dict[str, Any]],
    unmet: str = "as_stated",
    value_lists: str = "policy",
    conflicts: Any = None,
    is_write: Optional[Callable[[str], bool]] = None,
) -> tuple[List[Note], List[Dict[str, str]]]:
    """The notes that pass the checks, and why each other one was dropped. ``conflicts``: the model's list of arguments
    whose policy list differs from the tool's own. ``is_write``: which tools change the system (default: ``ruleset``'s)."""
    if is_write is None:
        from polisys.checks.rules import is_write_tool as is_write
    kept: List[Note] = []
    dropped: List[Dict[str, str]] = []
    specs = {s["name"]: s for s in tool_specs}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        tool, kind = str(item.get("tool") or ""), str(item.get("kind") or "")
        text = " ".join(str(item.get("text") or "").split())
        quote = str(item.get("quote") or "")
        inferred = item.get("inferred") is True
        ref = str(item.get({"before": "fix_tool", "values": "argument", "after": "then_tool"}.get(kind, "")) or "")
        why = ""
        if tool not in specs:
            why = "tool not in the run"
        elif kind not in KINDS:
            why = "unknown kind"
        elif not text or len(text) > MAX_NOTE_CHARS:
            why = "no text or too long"
        elif not quoted_in(quote, policy_text):
            why = "quote not in the policy text"
        elif kind == "before" and (ref not in specs or not is_write(ref)):
            why = "no write tool of the run fixes the condition"
        elif kind == "after" and ref not in specs:
            why = "no tool of the run does the next step"
        elif kind == "values" and ref not in _arguments(specs[tool]):
            why = "not an argument of this tool"
        elif kind == "before" and inferred and unmet != "fix_with_tool":
            why = "remedy not stated in the policy"
        if why:
            dropped.append({"tool": tool, "why": why})
        else:
            kept.append(Note(tool, kind, text, quote, inferred, ref))
    for item in conflicts if isinstance(conflicts, list) else []:
        if not isinstance(item, dict):
            continue
        tool, ref, quote = str(item.get("tool") or ""), str(item.get("argument") or ""), str(item.get("quote") or "")
        values = allowed_values(specs[tool], ref) if tool in specs else []
        why = ""
        if tool not in specs:
            why = "tool not in the run"
        elif not values:
            why = "the tool has no list for this argument"
        elif not quoted_in(quote, policy_text):
            why = "quote not in the policy text"
        elif all(_has_word(quote, v) for v in values):
            why = "the quoted policy list holds every value of the tool"
        elif value_lists != "tool":
            why = "the policy's list stands"
        if why:
            dropped.append({"tool": tool, "why": why})
        else:
            text = CONFLICT.format(argument=ref, values=", ".join(values))
            kept.append(Note(tool, "values", text, quote, False, ref, True))
    return kept, dropped


def cache_key(policy_text: str, tool_names: List[str], model_name: str, unmet: str, value_lists: str = "policy") -> str:
    blob = json.dumps(
        [PROMPT_VERSION, policy_text, sorted(tool_names), model_name, unmet, value_lists], ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def build_notes(
    policy_text: str,
    tool_specs: List[Dict[str, Any]],
    model: Callable[[str, str], str],
    unmet: str = "as_stated",
    cache_dir: Optional[str] = None,
    value_lists: str = "policy",
) -> Dict[str, Any]:
    """``{"notes": [Note as dict], "dropped": [...], "cached": bool}`` for this policy text and tool set. With a cache folder,
    a second caller waits for the first one's file instead of asking the model again, so parallel runs share one answer."""
    names = [s["name"] for s in tool_specs]
    path = None
    if cache_dir:
        folder = Path(cache_dir).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        key = cache_key(policy_text, names, getattr(model, "model_name", ""), unmet, value_lists)
        path = folder / f"{key}.json"

    def make() -> Dict[str, Any]:
        user = USER.format(policy=policy_text, tools=tool_lines(tool_specs))
        for _ in range(2):  # an answer cut at the token limit is not "no notes": ask once more, then fail
            reply = parse_json(model(SYSTEM, user))
            if isinstance(reply.get("notes"), list):
                kept, dropped = check_notes(
                    reply["notes"], policy_text, tool_specs, unmet, value_lists, reply.get("conflicts")
                )
                return {"notes": [asdict(n) for n in kept], "dropped": dropped, "cached": False}
        raise NotesError("the model's answer holds no list of notes (cut at the token limit, or not JSON)")

    if path is None:
        return make()
    import fcntl

    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                data["cached"] = True
                return data
            data = make()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
            return data
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def note_kind(note: Dict[str, Any]) -> str:
    """``conflict`` for a note on a list of values that the tool holds and the policy lacks, else the note's kind."""
    return "conflict" if note.get("conflict") else str(note.get("kind") or "")


def describe(notes: List[Dict[str, Any]], show: Optional[List[str]] = None) -> Dict[str, str]:
    """Per tool, the text added to its description: the notes by kind, cut at ``MAX_TOOL_CHARS``. With ``show``, only the
    notes of those kinds (see ``note_kind``)."""
    by_tool: Dict[str, List[Dict[str, Any]]] = {}
    for n in notes:
        if show is not None and note_kind(n) not in show:
            continue
        by_tool.setdefault(n["tool"], []).append(n)
    out = {}
    for tool, items in by_tool.items():
        lines = [HEADER]
        for kind in KINDS:
            for n in items:
                if n["kind"] == kind:
                    lines.append(f"- {LABELS[kind]}: {n['text']}")
        text = "\n".join(lines)
        out[tool] = text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS].rsplit("\n", 1)[0]
    return out


class ToolNotes:
    """The notes of one run. ``wrap_tools`` gives the agent its tools with the notes in their descriptions. The notes are
    read (from the cache, or from the model) at the first ``wrap_tools`` call, not when the run is wired, so that wiring a
    run asks no model."""

    def __init__(self, load: Callable[[], Dict[str, Any]], show: Optional[List[str]] = None):
        self._load = load
        self.show = show
        self.data: Optional[Dict[str, Any]] = None
        self.text: Dict[str, str] = {}
        self.records: List[Dict[str, Any]] = []

    def wrap_tools(self, tools: List[Any], turns: Any = None, live: Any = None) -> List[Any]:
        if self.data is None:
            try:
                self.data = self._load()
            except Exception as exc:  # the run goes on without notes, and the audit says so: never a silent empty arm
                self.data = {"notes": [], "dropped": []}
                self.records.append(
                    {"component": "tool_notes", "outcome": "notes_failed", "reason": f"{type(exc).__name__}: {exc}"[:200]}
                )
            self.text = describe(self.data.get("notes") or [], self.show)
        shown = [t for t in tools if self.text.get(getattr(t, "name", ""))]
        all_notes = self.data.get("notes") or []
        self.records.append(
            {
                "component": "tool_notes",
                "outcome": "notes_shown",
                "notes": sum(1 for n in all_notes if self.show is None or note_kind(n) in self.show),
                "written": len(all_notes),
                "tools": len(shown),
                "dropped": len(self.data.get("dropped") or []),
                "cached": bool(self.data.get("cached")),
            }
        )
        out = []
        for tool in tools:
            note = self.text.get(getattr(tool, "name", ""))
            if not note:
                out.append(tool)
                continue
            out.append(tool.model_copy(update={"description": f"{tool.description or ''}\n\n{note}".strip()}))
        return out


# -- wiring: what build_runtime adds to a run for this component (runtime/registry.py) ----------------------------------


def build(a: Assembly) -> Optional[Piece]:
    cfg = a.cfg.notes.tool_notes
    if not cfg.show:
        raise a.fail("tool_notes", "notes.tool_notes is on with no kind of note to show")
    specs = a.run_tool_specs()
    if not specs:
        raise a.fail("tool_notes", "notes.tool_notes needs RunContext.tool_specs (the tools of the run)")
    model = a.own_model("tool_notes", cfg.model, timeout=300, max_tokens=24000)
    text = a.policy_text("tool_notes")
    lists = a.cfg.authority.value_lists
    notes = ToolNotes(lambda: build_notes(text, specs, model, cfg.unmet, cfg.cache_dir, lists), list(cfg.show))
    return Piece(wrap_tools=notes.wrap_tools, target=notes)
