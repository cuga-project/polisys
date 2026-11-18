"""What a task gives the policy layer (``RunContext``), what the layer gives the runner (``RuntimeParts``), and what one
component adds to a run (``Piece``, ``Component``). This module imports nothing else from polisys at run time, so a
component's file can import it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Tuple

if TYPE_CHECKING:
    from polisys.runtime.assembly import Assembly
    from polisys.runtime.profile import PolicyConfig
    from polisys.runtime.trace import AuditLog, PolicyTrace


@dataclass
class RunContext:
    """What one task gives the policy layer."""

    task_id: str = ""
    request: str = ""
    system_prompt: str = ""
    tool_specs: List[Dict[str, Any]] = field(default_factory=list)  # {name, description, inputSchema, ...}
    tools_in_run: Optional[List[str]] = None  # None: every tool in tool_specs
    principal: Optional[Dict[str, Any]] = None  # the logged-in user as the platform knows it
    world_time: str = ""  # the task's date; empty: no date
    data_search: List[Tuple[str, Callable[[str], Any]]] = field(
        default_factory=list
    )  # (tool name, search(query))
    # () -> {source: [record, ...]}: every record of every app, read through the host's read tools (sources.listing). A
    # table is one record with a "rows" list. None: the host cannot list.
    data_list: Optional[Callable[[], Dict[str, List[Any]]]] = None
    # Where the run writes audit.jsonl. None: the audit stays in memory.
    out_dir: Optional[Path] = None
    policy_version: str = ""  # which policy the task is under, also when it has no package
    # How the host agent can show a hook's before_llm_call text: "system" (in the system prompt) or "message" (as the
    # newest message). notes.brief: message needs a host that says "message".
    host_note_placement: str = "system"
    # What the host's hook registry offers (``polisys.hooks.HOST_FEATURES``). None: not told, no check. A host that is told
    # to the runtime and lacks ``tool_blocked`` cannot run the obligation ledger: ``build_runtime`` refuses it.
    host_features: Optional[FrozenSet[str]] = None
    # Which package ran, as the result says (<version>+<file hash>). "": the version of the package.
    package_version: str = ""
    policy_text: str = ""  # the SOP text when it is not the system prompt
    # The tool that sends the customer a message in the middle of a turn. "": this benchmark has no such tool. None: not
    # said, and the navigator raises, because its name is in no tool spec and a wrong value gives no error.
    message_tool: Optional[str] = None
    # The model the agent runs on, as ``control.by_model`` names it. "": not said, and the level takes ``control.default_level``.
    agent_model: str = ""
    tool_apps: Optional[Dict[str, str]] = None  # tool name -> app ("gmail", "slack"); notes.read_first needs it


@dataclass
class RuntimeParts:
    """What a runner uses. Each slot is ``None`` when no component fills it, and then the runner does what it does with no
    policy layer.

    - ``run(agent, request, thread_id, timeout)``: replaces ``agent.invoke``.
    - ``turn_note(turns)``: a text to add to the next user turn, or ``None``.
    - ``wrap_tools(tools, turns, live)``: the tools to build the agent with.
    - ``after_reply(answer, turns, live)``: a new prompt instead of sending the reply, or ``None``.

    ``turn_note`` and ``after_reply`` make blocking model calls: call them off the event loop. ``turns`` and ``live`` are
    the runner's lists (see ``sop_navigator.HandoffGuard``). ``usage()``: by component, the model and its counters.
    ``records()``: the verdict records of all components, read when called.

    ``components``: the object behind each component that is on, for what a runner reads after the run (the loop's
    receipts, the store's calls).

    ``trace`` and ``audit``: the model-call events and the decision lines of the run (``runtime/trace.py``); ``None`` when the
    config says ``trace: false`` and when no component is on (a baseline run has neither, and writes no file)."""

    hooks: List[Any] = field(default_factory=list)
    tools: List[Any] = field(default_factory=list)
    system_note: str = ""
    run: Optional[Callable[..., Any]] = None
    turn_note: Optional[Callable[..., Any]] = None
    wrap_tools: Optional[Callable[..., Any]] = None
    after_reply: Optional[Callable[..., Any]] = None
    usage: Callable[[], Dict[str, Any]] = dict
    records: Callable[[], List[Any]] = list
    components: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[PolicyTrace] = None
    audit: Optional[AuditLog] = None


class RuntimeWiringError(ValueError):
    """The config, the package and the task cannot give a run that does what the config says."""


@dataclass
class Piece:
    """What one component adds to a run. ``target``: the object behind it (default: its first hook)."""

    hooks: List[Any] = field(default_factory=list)
    tools: List[Any] = field(default_factory=list)
    system_note: str = ""
    run: Optional[Callable[..., Any]] = None
    turn_note: Optional[Callable[..., Any]] = None
    wrap_tools: Optional[Callable[..., Any]] = None
    after_reply: Optional[Callable[..., Any]] = None
    target: Any = None
    # Other objects whose records are decisions (the hooks and the target are audited already).
    audited: List[Any] = field(default_factory=list)


@dataclass
class Component:
    """``enabled(cfg)``: is it on. ``build(assembly)``: its piece, or ``None`` when this package has nothing for it."""

    enabled: Callable[[PolicyConfig], bool]
    build: Callable[[Assembly], Optional[Piece]]
