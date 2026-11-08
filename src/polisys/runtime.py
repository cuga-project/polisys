"""Runtime assembly: one config, one package and one task give the hooks, tools, notes and slots of a run (design spec,
sections 3 and 13). This is the only place that wires the policy layer.

    parts = build_runtime(load_config(path), load_package(package_path), RunContext(task_id=..., tool_specs=...))

``COMPONENTS`` is the registry: a name, when the config turns it on, and how to build it. The order of the entries is the
order of the hooks. Adding a method is one entry, plus its config key when it has options. A component gets what it needs
as parameters from the config and the context. It reads no policy switch, and neither does this module.

**Wiring that can have no effect raises.** The run must not go on and measure something else with no word about it: a
gate with no model name would use the agent's model; a gate on a store that the agent's tools do not use checks almost
nothing; a guide on a package whose steps name no tool never stops. ``RuntimeWiringError`` names the task, the policy
version and the component.

This module knows no benchmark. A benchmark's facts come in through ``RunContext``; ``build_runtime`` runs inside each
task's child process.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from loguru import logger

from .levels import (
    Resolution,
    Shadow,
    WriteChecks,
    explain_level,
    gate_strength,
    resolve,
)
from .model import PolicyPackage
from .policy_config import PolicyConfig
from .trace import AUDIT_FILE, AuditLog, PolicyTrace


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

    ``trace`` and ``audit``: the model-call events and the decision lines of the run (``trace.py``); ``None`` when the
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


class Models:
    """The real model factory. A dry run gives ``dry_run.RecordingModels`` instead."""

    def json_model(self, model: str, **kwargs: Any) -> Callable[[str, str], str]:
        from .llm import openai_json_model

        return openai_json_model(model, **kwargs)

    def openrouter_json_model(self, model: str, **kwargs: Any) -> Callable[[str, str], str]:
        from .llm import openrouter_json_model

        return openrouter_json_model(model, **kwargs)

    @staticmethod
    def _need_key(model: str) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):  # with no key, every answer would be "unknown"
            raise ValueError(f"{model} needs OPENROUTER_API_KEY, and it is not set")

    def questions(self, model: str) -> Callable[..., Any]:
        from .detectors import systemone_questions

        self._need_key(model)
        return systemone_questions(model)

    def detector(self, model: str) -> Callable[..., Any]:
        from .detectors import systemone_detector

        self._need_key(model)
        return systemone_detector(model)


class Assembly:
    """What the components of one run share: the inputs, the one procedure store, and the models made so far."""

    def __init__(
        self,
        cfg: PolicyConfig,
        pkg: Optional[PolicyPackage],
        ctx: RunContext,
        models: Any,
        resolution: Optional[Resolution] = None,
    ):
        self.cfg, self.pkg, self.ctx, self.models = cfg, pkg, ctx, models
        # The level of the run, and what the level switch decided; a test that builds an Assembly alone gets the static level.
        static = explain_level(cfg, ctx.agent_model).level
        self.resolution = resolution or Resolution(cfg, static, static, [])
        self.level = self.resolution.level
        self.notes: List[Tuple[str, str]] = list(self.resolution.notes)  # written to the audit when it starts
        self.held: Dict[str, Any] = {}  # the checks that WriteChecks runs, by component
        self.made: List[Tuple[str, Any]] = []
        self.package_read = False
        self._store: Any = None
        self._scope: Any = None
        self.loop: Any = None  # the SOP loop of this run, when it has one
        self.trace: Optional[PolicyTrace] = PolicyTrace() if cfg.trace else None
        self.audit: Optional[AuditLog] = None
        if cfg.trace:
            version = ctx.package_version or (getattr(pkg, "version", "") if pkg is not None else "")
            path = Path(ctx.out_dir) / AUDIT_FILE if ctx.out_dir else None
            self.audit = AuditLog(path, task=ctx.task_id, package_version=version)
            self.trace.audit = self.audit  # a call that fails is a line of the audit

    def fail(self, component: str, why: str) -> RuntimeWiringError:
        version = self.ctx.policy_version or getattr(self.pkg, "version", "") or "unknown"
        return RuntimeWiringError(
            f"task {self.ctx.task_id}, policy version {version}, component {component}: {why}"
        )

    def package(self, component: str) -> Any:
        if self.pkg is None:
            raise self.fail(
                component, f"needs a policy package, and none is available (profile: {self.cfg.package})"
            )
        self.package_read = True
        return self.pkg

    def store(self, component: str) -> Any:
        """The one ``ProcedureStore`` of this run. The gate learns from it what the agent opened through the tools."""
        if self._store is None:
            from .procedure_store import ProcedureStore

            self._store = ProcedureStore.from_package(self.package(component))
        return self._store

    def scope_reader(self, component: str) -> Any:
        """The one ``ScopeReader`` of this run (``router.py``): the router and expander of ``checkers.router``, over the
        package and the shared store. The gate reads its scope from it (the ledger has no retrieval scope for now). A router
        that needs a model gets the traced one of ``checkers.router.model`` (``llm_cards``; the measured router had a timeout
        of 240 and 16000 tokens) or of ``checkers.narrow_questions`` (``bm25_kev``)."""
        if self._scope is None:
            from .router import ScopeReader, make_expander, make_router, router_needs

            r = self.cfg.checkers.router
            try:
                needs = router_needs(r.kind)
                options: Dict[str, Any] = {"top_k": r.top_k}
                if needs == "model":
                    options["model"] = self.model("router", r.model, timeout=240, max_tokens=16000)
                elif needs == "ask":
                    options["ask"] = self.questions("router", self.cfg.checkers.narrow_questions)
                router = make_router(r.kind, **options)
                expander = make_expander(r.expand, hops=r.hops)
            except ValueError as exc:
                raise self.fail(component, f"checkers.router: {exc}") from exc
            pkg = self.package(component)
            self._scope = ScopeReader(router, expander, pkg, self.store(component))
            if expander.problem(pkg):  # a method that changes nothing would read as "no gain" in a comparison
                raise self.fail(component, f"checkers.router.expand {r.expand}: {expander.problem(pkg)}")
        return self._scope

    def model(self, component: str, name: Optional[str], openrouter: bool = False, **kwargs: Any) -> Any:
        if not name:  # llm.py would fall back to the agent's model
            raise self.fail(component, "its model is none")
        make = self.models.openrouter_json_model if openrouter else self.models.json_model
        model = make(name, **kwargs)
        if self.trace is not None:
            model = self.trace.wrap_model(model, component)
        self.made.append((component, model))
        return model

    def questions(self, component: str, name: Optional[str]) -> Any:
        if not name:
            raise self.fail(component, "needs checkers.narrow_questions, and it is none")
        ask = self.models.questions(name)
        if self.trace is not None:
            ask = self.trace.wrap_ask(ask, component)
        self.made.append((component, ask))
        return ask

    def detector(self, name: str) -> Any:
        detector = self.models.detector(name)
        if self.trace is not None:
            detector = self.trace.wrap_detector(detector, "detector")
        self.made.append(("detector", detector))
        return detector

    def note_channel(self) -> str:
        return {"tool_result": "tool", "system": "system"}[self.cfg.notes.reminders]

    def brief(self) -> bool:
        return self.cfg.notes.brief != "off"

    def method_choice(self) -> Optional[Dict[str, Any]]:
        path = self.cfg.checkers.method_choice
        return json.loads(Path(path).read_text()) if path else None

    def verifier_for(self, component: str) -> Callable[[str], Any]:
        """Model name -> verifier, for rule types whose measured best model differs."""
        from .verifier import Verifier

        cache: Dict[str, Any] = {}

        def verifier_for(name: str) -> Any:
            if name not in cache:
                via, _, rest = name.partition(":")
                on_router = via == "openrouter" and bool(rest)
                cache[name] = Verifier(
                    self.model(component, rest if on_router else name, openrouter=on_router)
                )
            return cache[name]

        return verifier_for


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


# -- the components ----------------------------------------------------------------------------------------------------


def _ledger_on(cfg: PolicyConfig) -> bool:
    """The obligation ledger is on when the profile names a producer. Then it is the only part that sends follow-ups: the
    requirements tracker is refused (``_check_settings``), the procedure monitor and the duty rules keep their records and send
    nothing, the intake's check of the reply is off (its write checks stay), and the old request checklist (``deliverables``)
    is the producer ``request``."""
    return bool(cfg.obligations.producers)


def _rules_on_package(cfg: PolicyConfig) -> bool:
    return cfg.checkers.rules == "on" and (bool(cfg.package) or not cfg.sources.data)


def _build_rules(a: Assembly) -> Piece:
    from .package import runtime_rules
    from .ruleset import RuleSetHooks
    from .verifier import Verifier

    pkg = a.package("rules")
    rules = runtime_rules(pkg)
    if not rules:
        raise a.fail("rules", "the package has no rules")
    if not a.ctx.tool_specs:
        raise a.fail("rules", "the task gives no tool specs, so no rule can be matched to a call")
    choice = a.method_choice()
    verifier = verifier_for = None
    check = a.cfg.checkers.verifier
    specific = check.scope == "specific_statements"
    # specific_statements: only a judgment rule of the package, on a tool it names (spec section 4: a general statement never
    # goes to a per-call model); a package with none builds no verifier.
    if check.mode == "on" and any(
        r.get("type") == "verify"
        and (not specific or (not r.get("routed_from") and (r.get("tools") or r.get("tool"))))
        for r in rules
    ):
        verifier = Verifier(a.model("verifier", check.model))
        verifier_for = a.verifier_for("verifier") if choice else None
    hooks = RuleSetHooks.from_package(
        pkg,
        a.ctx.tool_specs,
        principal=a.ctx.principal,
        tools_in_run=a.ctx.tools_in_run,
        method_choice=choice,
        verifier=verifier,
        verifier_for=verifier_for,
        verifier_may_deny=check.may_deny,
        brief=a.brief(),
        brief_guidance=a.cfg.notes.brief_guidance,
        # a rule list shows each rule with its exceptions (spec section 8); other shapes keep the brief as measured
        with_exceptions="rules" in a.resolution.shapes,
        note_channel=a.note_channel(),
        cap_effect=a.cfg.decisions.after_repeated_denials,
        duty_follow_up=not _ledger_on(a.cfg),  # with a producer, only the ledger sends follow-ups
        verify_scope=check.scope,
    )
    return Piece(hooks=[hooks])


def _build_loop(a: Assembly) -> Piece:
    from .sop_loop import LoopHooks, SopLoop

    cfg = a.cfg.organizers.loop
    index = None
    if cfg.style == "discovery":
        from .package import to_build
        from .sop_retrieval import SopIndex

        narrow = a.cfg.checkers.narrow_questions
        index = SopIndex(to_build(a.package("loop")), ask=a.questions("loop", narrow) if narrow else None)
    options = {
        "rules_from_data": cfg.rules_from_data,
        "mode": cfg.style,
        "require_evidence": a.cfg.decisions.require_evidence,
        "index": index,
    }
    model = a.model("loop", cfg.model, timeout=300, max_tokens=8000)
    if a.pkg is not None:
        loop = SopLoop.from_package(a.package("loop"), model, **options)
    else:  # with no package, the loop asks the model for the rules in the data
        loop = SopLoop(model, [], **options)
    a.loop = loop
    rec = LoopHooks()

    async def run(agent: Any, request: str, thread_id: str, timeout: float) -> Any:
        return await loop.run(agent, request, thread_id, rec, timeout)

    run.target = loop  # type: ignore[attr-defined]
    return Piece(hooks=[rec], run=run, target=loop)


def _build_monitor(a: Assembly) -> Optional[Piece]:
    from .procedure_monitor import ProcedureMonitor

    pkg = a.package("procedure_monitor")
    if not pkg.procedures:  # most rule lists have no procedures: nothing to follow, as in the old wiring
        return None
    shapes = a.resolution.shapes
    if "rules" in shapes and not {"procedures", "flow"} & set(shapes):
        return None  # a declared rule list: its procedures are the policy's sections, and no step is "still missing"
    return Piece(
        hooks=[
            ProcedureMonitor(pkg.procedures, tools_in_run=a.ctx.tools_in_run, follow_up=not _ledger_on(a.cfg))
        ]
    )


def _build_intake(a: Assembly) -> Piece:
    from .decisions import Decisions
    from .intake import OnlinePolicyIntake

    checkers, decisions = a.cfg.checkers, a.cfg.decisions
    verify = checkers.verifier.mode == "on"
    if verify and checkers.verifier.model != checkers.compile:
        raise a.fail(
            "intake",
            f"it has one model, and checkers.compile ({checkers.compile}) differs from checkers.verifier.model "
            f"({checkers.verifier.model})",
        )
    if a.cfg.sources.discovery and not a.ctx.data_search:
        raise a.fail(
            "intake", "sources.discovery is on, and the task gives no search over messages or documents"
        )
    if a.cfg.sources.listing and a.ctx.data_list is None:
        raise a.fail("intake", "sources.listing is on, and the task gives no listing of its records")
    if checkers.rules == "on" and not a.ctx.tool_specs:
        raise a.fail("intake", "checkers.rules is on, and the task gives no tool specs to compile rules for")
    detector = a.detector(checkers.detector) if checkers.detector else None
    hook = OnlinePolicyIntake(
        a.model("intake", checkers.compile),
        verify_writes=verify,
        check_reply=verify and not _ledger_on(a.cfg),  # with a producer only the ledger sends follow-ups
        detector=detector,
        tool_specs=a.ctx.tool_specs if checkers.rules == "on" else None,
        method_choice=a.method_choice(),
        read_before_write=checkers.read_before_write,
        verifier_may_deny=checkers.verifier.may_deny,
        guard_code=checkers.guard_code,
        brief=a.brief(),
        note_channel=a.note_channel(),
        today=a.ctx.world_time[:10] or None,  # every check judges dates against it, not only guard code
        track_supersedes=a.cfg.authority.supersedes,
        discover=a.ctx.data_search if a.cfg.sources.discovery else None,
        listing=a.ctx.data_list if a.cfg.sources.listing else None,
        listing_data=a.cfg.sources.listing and a.cfg.sources.listing_data,
        decisions=Decisions(a.questions("decisions", checkers.narrow_questions))
        if decisions.mode == "on"
        else None,
        model_effect=decisions.effect,
        cap_effect=decisions.after_repeated_denials,
        record_checks=decisions.record_checks,
        retry_effect=decisions.retry_effect,
        voice=a.cfg.followups.voice,
        evidence=a.cfg.followups.evidence,
    )
    return Piece(hooks=[hook])


def _build_requirements_tracker(a: Assembly) -> Piece:
    from .tracker import RequirementsTracker

    cfg = a.cfg.organizers.requirements_tracker
    model = a.model("requirements_tracker", cfg.model, timeout=400, max_tokens=16000)
    policy = a.ctx.system_prompt if cfg.read_system_prompt else ""
    if cfg.read_system_prompt and not policy:
        raise a.fail("requirements_tracker", "read_system_prompt is on, and the task gives no system prompt")
    return Piece(
        hooks=[
            RequirementsTracker(
                model,
                policy=policy,
                wrong_mode=cfg.wrong_items,
                voice=a.cfg.followups.voice,
                evidence=a.cfg.followups.evidence,
                today=a.ctx.world_time[:10],
            )
        ]
    )


def _build_guide(a: Assembly) -> Piece:
    from .procedure_guide import ProcedureGuide

    cfg = a.cfg.organizers.guide
    model = None
    if cfg.evidence_model:
        model = a.model("guide", cfg.evidence_model, timeout=180)
    guide = ProcedureGuide.from_package(
        a.package("guide"), model=model, gate=cfg.stop_first_write, templates=cfg.stop_templates
    )
    if not any(s.get("action") for p in guide.procs for s in p.get("steps") or [] if isinstance(s, dict)):
        raise a.fail(
            "guide",
            "no step of the package names a tool, so the guide has no tool to stop on (a graph package?)",
        )
    return Piece(hooks=[guide])


def _build_procedure_tools(a: Assembly) -> Piece:
    from .procedure_store import POLICY_TOOLS_NOTE, policy_tools

    store = a.store("procedure_tools")
    if not store.procs:
        raise a.fail("procedure_tools", "the package has no procedures for the agent to read")
    return Piece(tools=policy_tools(store), system_note=POLICY_TOOLS_NOTE, target=store)


def _build_procedure_push(a: Assembly) -> Piece:
    from .procedure_push import ProcedurePush
    from .router import make_router

    cfg = a.cfg.organizers.procedure_push
    store = a.store("procedure_push")
    if not store.procs:
        raise a.fail("procedure_push", "the package has no procedures to hand over")
    model = a.model("procedure_push", cfg.model, timeout=240, max_tokens=16000)
    router = make_router("llm_cards", model=model, top_k=cfg.top_k)
    push = ProcedurePush(
        store, a.package("procedure_push"), router, request=a.ctx.request or "", max_selections=cfg.max_selections
    )
    return Piece(hooks=[push])


def _build_read_first(a: Assembly) -> Piece:
    from .read_first import ReadFirst
    from .ruleset import is_write_tool

    if not a.ctx.tool_apps:
        raise a.fail("read_first", "notes.read_first needs RunContext.tool_apps (which app each tool belongs to)")
    return Piece(hooks=[ReadFirst(a.ctx.tool_apps, is_write_tool, request=a.ctx.request or "")])


def _build_work_method(a: Assembly) -> Piece:
    from .work_method import ReadbackHooks, note

    names = list(a.cfg.notes.work_method)
    if "readback" in names and "readback_sop" in names:
        raise a.fail("work_method", "readback and readback_sop are two forms of one turn; list one")
    if "readback_sop" in names:
        hooks = [ReadbackHooks(store=a.store("work_method"))]
    else:
        hooks = [ReadbackHooks()] if "readback" in names else []
    return Piece(hooks=hooks, system_note=note(names), target=hooks[0] if hooks else None)


def _build_gate(a: Assembly) -> Piece:
    from .router import router_reads_opened
    from .store_gate import StoreGate

    cfg = a.cfg.checkers.gate
    cfg_router = a.cfg.checkers.router
    status, text = gate_strength(a.cfg, a.ctx.agent_model)
    if status == "weaker":
        raise a.fail("gate", f"{text}; write min_strength none to allow it")
    if status == "unknown":  # not an error: the strength is not known, and the profile said on
        a.notes.append(("gate_min_strength", f"the gate stays on because the profile says on: {text}"))
    elif status == "ok":
        a.notes.append(("gate_min_strength", text))
    model = a.model("gate", cfg.model, timeout=180, max_tokens=cfg.max_tokens)
    today = a.ctx.world_time[:10]
    if cfg.scope == "active_procedures":
        opens = a.cfg.organizers.procedure_tools == "on" or a.cfg.organizers.procedure_push.mode == "on"
        if not opens and router_reads_opened(cfg_router.kind):
            raise a.fail(
                "gate",
                "scope active_procedures with the router agent_opened needs organizers.procedure_tools or procedure_push: "
                "nothing else opens a procedure (a router that searches does not need them)",
            )
        reader = a.scope_reader("gate")
        gate = StoreGate.from_package(
            a.package("gate"),
            model,
            store=a.store("gate"),
            router=reader.router,
            expander=reader.expander,
            today=today,
        )
    else:
        gate = StoreGate.from_package(
            None, model, sop_text=a.ctx.policy_text or a.ctx.system_prompt, today=today
        )
    if cfg.missing_needs_read or cfg.trigger_check:
        from .store_gate import MISSING_NEEDS_READ, TRIGGER_CHECK

        gate.extra = (MISSING_NEEDS_READ if cfg.missing_needs_read else "") + (TRIGGER_CHECK if cfg.trigger_check else "")
    if cfg.seed_request:
        if not a.ctx.request:
            raise a.fail("gate", "seed_request is on, and the task gives no request")
        gate.reads.append(f"user request: {a.ctx.request}")
    return Piece(hooks=[gate])


def _build_sop_tracker(a: Assembly) -> Piece:
    from .sop_tracker import SopTracker

    model = a.model("sop_tracker", a.cfg.organizers.sop_tracker.model, timeout=90, max_tokens=8000)
    tracker = SopTracker(a.ctx.policy_text or a.ctx.system_prompt, model)
    if not tracker.sections:
        raise a.fail(
            "sop_tracker", "the task's policy text has no sections, so the tracker has nothing to say"
        )
    return Piece(turn_note=tracker.note, target=tracker)


def _build_navigator(a: Assembly) -> Piece:
    from .sop_navigator import HandoffGuard, SopNavigator

    cfg = a.cfg.organizers.navigator
    names = [t["name"] for t in a.ctx.tool_specs]
    if a.ctx.message_tool is None:
        raise a.fail(
            "navigator",
            "the task does not say which tool sends a message in the middle of a turn; give RunContext.message_tool, "
            'or "" when the benchmark has none',
        )
    if not cfg.handoff_text and not cfg.handoff_tools:
        raise a.fail("navigator", "no hand-off phrase and no hand-off tool, so it can never fire")
    in_run = set(names if a.ctx.tools_in_run is None else a.ctx.tools_in_run)
    missing = sorted(set(cfg.handoff_tools) - in_run)
    if missing:
        raise a.fail("navigator", f"the hand-off tools {missing} are not in this run")
    if cfg.max_blocks == 0:
        raise a.fail("navigator", "max_blocks is 0, so it can never block")
    navigator = SopNavigator.from_package(
        a.package("navigator"),
        a.ctx.policy_text or a.ctx.system_prompt,
        names,
        a.questions("navigator", a.cfg.checkers.narrow_questions),
        semantic_evidence=cfg.semantic_evidence,
    )
    if not navigator.chapters:
        raise a.fail(
            "navigator", "no chapter of the policy holds enough actions, so no hand-off is ever open"
        )
    guard = HandoffGuard(navigator, cfg.handoff_text, cfg.handoff_tools, cfg.max_blocks, a.ctx.message_tool)
    return Piece(
        wrap_tools=guard.wrap_tools, after_reply=guard.after_reply, target=guard, audited=[navigator]
    )


def _build_obligations(a: Assembly) -> Piece:
    from .deliverables import obligations_from_request
    from .obligations import CONVERSATION, ONE_REQUEST, SOP_LOOP, ObligationLedger, cards_of
    from .package import obligations_from_package, runtime_rules
    from .ruleset import obligations_from_duties

    cfg = a.cfg.obligations
    producers = set(cfg.producers)
    if not a.ctx.tool_specs:
        raise a.fail("obligations", "the task gives no tool specs, so no argument test can be chosen")
    specs = [{k: v for k, v in t.items() if k != "app"} for t in a.ctx.tool_specs]
    if a.ctx.tools_in_run is not None:
        in_run = set(a.ctx.tools_in_run)
        specs = [t for t in specs if t["name"] in in_run]
    # The producers `package` and `duty` read the package; `request` reads the request, at the user message.
    pkg = a.package("obligations") if producers & {"package", "duty"} else a.pkg
    made: List[Any] = []
    if "package" in producers:
        made += obligations_from_package(pkg)
        if not made:
            raise a.fail("obligations", "the package has no obligations, so the ledger has nothing to check")
    if "duty" in producers:  # a policy may have no duty rule: that is no error while another producer is on
        made += obligations_from_duties(runtime_rules(pkg), specs)
    if not made and "request" not in producers:
        raise a.fail(
            "obligations",
            "the producers give no obligation (the package has no duty rule), so the ledger has nothing to check",
        )
    # The items of a request and of duties are `observe` until a measurement writes a level (Task 4.3), as the package's are.
    # With no `obligations.item_level` the level switch decides, and the ledger is on at `observe`: it records and sends
    # nothing until a replay allows it (spec section 5). An explicit `item_level` is what the profile asks for.
    effective_item = cfg.item_level or "observe"
    nav = a.cfg.organizers.navigator
    run_type = ONE_REQUEST
    if nav.handoff_tools or nav.handoff_text:
        if a.cfg.organizers.loop.mode == "on":
            raise a.fail(
                "obligations", "a run with hand-off phrases or tools and a loop has no one final reply"
            )
        if a.ctx.message_tool is None:
            raise a.fail(
                "obligations",
                "the task does not say which tool sends a message in the middle of a turn; give "
                'RunContext.message_tool, or "" when the benchmark has none',
            )
        in_run = set(
            [t["name"] for t in a.ctx.tool_specs] if a.ctx.tools_in_run is None else a.ctx.tools_in_run
        )
        missing = sorted(set(nav.handoff_tools) - in_run)
        if missing:
            raise a.fail("obligations", f"the hand-off tools {missing} are not in this run")
        run_type = CONVERSATION
    elif a.ctx.message_tool:
        raise a.fail(
            "obligations",
            "the task is a conversation, and there is no hand-off phrase and no hand-off tool, so the ledger would act on an "
            "ordinary turn; give organizers.navigator.handoff_text or handoff_tools",
        )
    elif a.cfg.organizers.loop.mode == "on":
        run_type = SOP_LOOP
    # The ledger's level limits its follow-ups (0 and 1: none, 2: one, 3: two). Under the level switch that is the run's level.
    # An explicit `item_level` keeps the earlier reading, which the profiles written for it depend on: the level of the items
    # says what the ledger may do.
    if cfg.item_level:
        level = {"observe": 0, "show": 1}.get(cfg.item_level, 3 if cfg.max_follow_ups >= 2 else 2)
    else:
        level = a.level
    policy_text = "\n".join(
        st.text
        for d in (pkg.documents if pkg is not None else [])
        for s in d.sections
        for st in s.statements
        if st.text
    )
    request_producer = None
    if "request" in producers:
        if run_type == CONVERSATION:
            raise a.fail(
                "obligations",
                "the producer request reads one request, and a conversation has none: turn it off",
            )
        ask = a.model("obligations", cfg.model)
        request_producer = lambda text: obligations_from_request(ask, text, specs)  # noqa: E731
    bindable = any(o.origin in ("package", "data") for o in made)
    ledger = ObligationLedger(
        made,
        a.model("obligations", cfg.model, timeout=300, max_tokens=16000) if bindable else None,
        cards=cards_of(made, pkg.procedures if pkg is not None else ()),
        tool_specs=a.ctx.tool_specs,
        tools_in_run=a.ctx.tools_in_run,
        run_type=run_type,
        level=level,
        max_follow_ups=cfg.max_follow_ups,
        wrong_values=cfg.wrong_values,
        item_level=effective_item,
        request=a.ctx.request,
        store=a.store("obligations") if pkg is not None else None,
        today=a.ctx.world_time[:10],
        policy_text=policy_text,
        handoff_tools=nav.handoff_tools,
        handoff_text=nav.handoff_text,
        message_tool=a.ctx.message_tool or "",
        request_producer=request_producer,
    )
    if a.loop is not None:
        a.loop._phase_listener = ledger.loop_phase
    return Piece(
        hooks=[ledger], after_reply=ledger.after_reply if run_type == CONVERSATION else None, target=ledger
    )


COMPONENTS: Dict[str, Component] = {
    "rules": Component(_rules_on_package, _build_rules),
    "loop": Component(lambda cfg: cfg.organizers.loop.mode == "on", _build_loop),
    "procedure_monitor": Component(_rules_on_package, _build_monitor),
    "intake": Component(lambda cfg: cfg.sources.data, _build_intake),
    "requirements_tracker": Component(
        lambda cfg: cfg.organizers.requirements_tracker.mode == "on", _build_requirements_tracker
    ),
    "guide": Component(lambda cfg: cfg.organizers.guide.mode == "on", _build_guide),
    "procedure_tools": Component(lambda cfg: cfg.organizers.procedure_tools == "on", _build_procedure_tools),
    "gate": Component(lambda cfg: cfg.checkers.gate.mode == "on", _build_gate),
    "procedure_push": Component(lambda cfg: cfg.organizers.procedure_push.mode == "on", _build_procedure_push),
    "read_first": Component(lambda cfg: cfg.notes.read_first, _build_read_first),
    "work_method": Component(lambda cfg: bool(cfg.notes.work_method), _build_work_method),
    "sop_tracker": Component(lambda cfg: cfg.organizers.sop_tracker.mode == "on", _build_sop_tracker),
    "navigator": Component(lambda cfg: cfg.organizers.navigator.mode == "on", _build_navigator),
    "obligations": Component(_ledger_on, _build_obligations),
}


# -- settings that cannot work ------------------------------------------------------------------------------------------

# Nothing reads these yet (a later task does). Any other value would be ignored with no word about it.
NOT_READ_YET: Dict[str, Any] = {
    "authority.company_domains": [],
    "authority.may_change_rules": [],
}
PRODUCERS_BUILT = {"request", "package", "duty"}


def _value(cfg: PolicyConfig, path: str) -> Any:
    node: Any = cfg
    for key in path.split("."):
        node = getattr(node, key)
    return node


def _check_settings(a: Assembly) -> None:
    cfg, checkers, organizers = a.cfg, a.cfg.checkers, a.cfg.organizers
    features = a.ctx.host_features
    if features is not None:
        if _ledger_on(cfg) and "tool_blocked" not in features:
            raise a.fail(
                "obligations",
                "needs a host whose hook registry tells every hook about a stopped call (tool_blocked), and RunContext."
                "host_features says this host does not: a follow-up could ask for work that another hook blocked",
            )
        if "failure_report" not in features:
            logger.warning(
                "polisys: the host's hook registry does not report a hook that raises, so a failed check leaves no audit line"
            )
    modes = {
        "checkers.rules": checkers.rules,
        "checkers.gate.mode": checkers.gate.mode,
        "checkers.verifier.mode": checkers.verifier.mode,
        "decisions.mode": cfg.decisions.mode,
        "organizers.procedure_tools": organizers.procedure_tools,
        "organizers.guide.mode": organizers.guide.mode,
        "organizers.loop.mode": organizers.loop.mode,
        "organizers.sop_tracker.mode": organizers.sop_tracker.mode,
        "organizers.navigator.mode": organizers.navigator.mode,
        "organizers.requirements_tracker.mode": organizers.requirements_tracker.mode,
    }
    for path, mode in {**modes, "checkers.write_checks": checkers.write_checks}.items():
        if mode == "auto":  # levels.resolve replaces every auto before this runs
            raise a.fail(
                path,
                "is still auto: build_runtime resolves it, and a component built alone must not skip that",
            )
    if cfg.package and "{policy}" in cfg.package and not a.ctx.policy_version:
        raise a.fail(
            "package", f"the profile names {cfg.package}, and the task gives no policy_version to fill in"
        )
    if cfg.package and a.pkg is None:
        raise a.fail("package", f"the profile names the package {cfg.package}, and none is available")
    if not cfg.package and a.pkg is not None:
        raise a.fail("package", "the profile says no package, and one was given")
    for path, default in NOT_READ_YET.items():
        if _value(cfg, path) != default:
            raise a.fail(path, "is set, and nothing reads it yet")
    from .router import UnknownMethod, check_names, expander_options, router_names, router_options

    try:
        check_names(checkers.router.kind, checkers.router.expand)
    except UnknownMethod as exc:
        raise a.fail(
            "checkers.router.kind"
            if checkers.router.kind not in router_names()
            else "checkers.router.expand",
            str(exc),
        ) from exc
    r = checkers.router
    read = set(router_options(r.kind)) | set(expander_options(r.expand))
    for name in ("model", "top_k", "hops"):  # a setting the chosen methods do not read would change nothing
        if getattr(r, name) != getattr(type(r)(), name) and name not in read:
            raise a.fail(
                f"checkers.router.{name}",
                f"is set, and the router {r.kind} with the expander {r.expand} does not read it",
            )
    unbuilt = set(cfg.obligations.producers) - PRODUCERS_BUILT
    if unbuilt:
        raise a.fail("obligations.producers", f"{sorted(unbuilt)}: no such producer exists yet")
    # The verifier of the online intake keeps its measured path (it takes every statement that did not compile, whatever the
    # scope says); the scope sets what the verifier of the package's rules takes.
    write = checkers.write_checks == "on"
    if write and not (_rules_on_package(cfg) or checkers.gate.mode == "on"):
        raise a.fail(
            "checkers.write_checks",
            "is on, and neither the package's rules nor the gate is on, so it has no check to run",
        )
    if a.resolution.risk_active and not write:
        raise a.fail(
            "control.risk.level3_tools",
            "is set, and separate hooks cannot give one tool more checks: it needs checkers.write_checks, and no check is "
            "on that it could hold",
        )
    if a.resolution.risk_active and not a.resolution.auto_stages:
        raise a.fail(
            "control.risk.level3_tools",
            "is set, and it does nothing: a check set to on or off runs the same for every tool, so one tool can get more "
            "checks only from a check left to auto: set rules, gate or verifier to auto, or remove the setting",
        )
    # The control settings change something only where something reads the level: a component left to ``auto``,
    # ``write_checks``, a ledger with no ``item_level``, level 0 (shadow), and the gate's comparison of strength.
    control, res = cfg.control, a.resolution
    gate_compares = checkers.gate.mode == "on" and checkers.gate.min_strength == "agent"
    unread = []
    if control.level != "by_model" and not res.level_read:
        unread.append("control.level")
    if control.by_model and not (res.level_read or gate_compares):
        unread.append("control.by_model")
    if control.default_level != 1 and not (control.level == "by_model" and res.level_read):
        unread.append("control.default_level")
    # A declared rule list is read by the rule hooks (the brief shows each rule with its exceptions) and by the procedure monitor
    # (it builds none), whatever else the level decides.
    rule_list_read = "rules" in cfg.shapes and _rules_on_package(cfg)
    if cfg.shapes and not (res.had_auto or rule_list_read):
        unread.append("shapes")
    if unread:
        raise a.fail(
            unread[0],
            "is set, and nothing reads it: the profile lists every component, so no level decides anything (a component "
            "left to auto, checkers.write_checks, a ledger with no item_level or level 0 would read it)",
        )
    # A setting that only one path reads, with that path off.
    data = cfg.sources.data
    rules = checkers.rules == "on"
    loop = organizers.loop.mode == "on"
    ledger = _ledger_on(cfg)
    later = (
        a.resolution.pending
    )  # components that auto left off: the level decides, and a higher level would read these
    if ledger and organizers.navigator.mode == "on":
        raise a.fail("obligations", "the navigator checks the same hand-offs: turn one of them off")
    if ledger and organizers.requirements_tracker.mode == "on":
        raise a.fail(
            "obligations",
            "the requirements tracker sends its own follow-up, and with a producer only the ledger sends them: turn one of "
            "them off",
        )
    router = checkers.router
    default_router = type(router)()
    changed = [
        f
        for f in ("kind", "expand", "model", "top_k", "hops")
        if getattr(router, f) != getattr(default_router, f)
    ]
    gate_reads = (
        checkers.gate.mode == "on" and checkers.gate.scope == "active_procedures"
    ) or "gate" in later
    readers = [
        (
            f"checkers.router.{changed[0] if changed else 'kind'}",
            bool(changed),
            gate_reads,
            "checkers.gate with scope active_procedures: only the gate reads the scope",
        ),
        ("priority_classes", bool(cfg.priority_classes), bool(cfg.package), "a package"),
        (
            "notes.brief_guidance",
            cfg.notes.brief_guidance,
            cfg.notes.brief != "off" and (_rules_on_package(cfg) or "rules" in later),
            "the brief (notes.brief is not off) and checkers.rules on a package",
        ),
        ("obligations.max_follow_ups", cfg.obligations.max_follow_ups != 1, ledger, "a producer"),
        (
            "obligations.wrong_values",
            cfg.obligations.wrong_values != "known_origin_only",
            ledger,
            "a producer",
        ),
        ("obligations.item_level", cfg.obligations.item_level is not None, ledger, "a producer"),
        ("sources.discovery", cfg.sources.discovery, data, "sources.data"),
        ("sources.listing", cfg.sources.listing, data, "sources.data"),
        ("authority.supersedes", cfg.authority.supersedes, data, "sources.data"),
        ("checkers.compile", bool(checkers.compile), data, "sources.data"),
        ("checkers.detector", bool(checkers.detector), data, "sources.data"),
        ("checkers.guard_code", checkers.guard_code != "off", data, "sources.data"),
        ("checkers.read_before_write", checkers.read_before_write != "off", data, "sources.data"),
        ("decisions.mode", cfg.decisions.mode == "on", data, "sources.data"),
        (
            "decisions.require_evidence",
            cfg.decisions.require_evidence,
            loop or "loop" in later,
            "organizers.loop",
        ),
        (
            "checkers.gate.seed_request",
            checkers.gate.seed_request,
            checkers.gate.mode == "on" or "gate" in later,
            "checkers.gate",
        ),
        ("checkers.rules", rules, data or bool(cfg.package), "a package or sources.data"),
        (
            "checkers.verifier.mode",
            checkers.verifier.mode == "on",
            rules or data or "rules" in later,
            "checkers.rules or sources.data",
        ),
        (
            "checkers.method_choice",
            bool(checkers.method_choice),
            rules or data or "rules" in later,
            "checkers.rules or sources.data",
        ),
        (
            "checkers.narrow_questions",
            bool(checkers.narrow_questions),
            cfg.decisions.mode == "on"
            or (router.kind == "bm25_kev" and gate_reads)
            or organizers.navigator.mode == "on"
            or (loop and organizers.loop.style == "discovery")
            or bool(later & {"decisions", "navigator", "loop"}),
            "decisions, the navigator, the discovery loop or the router bm25_kev",
        ),
    ]
    for path, is_set, is_read, reader in readers:
        if is_set and not is_read:
            raise a.fail(path, f"is set, and nothing reads it: it needs {reader}")
    if cfg.priority_classes and a.pkg is not None:
        # The ranks themselves decide a conflict when the package is validated or its conflicts are settled
        # (``validate_package`` and ``resolve_conflicts`` take the list); a run reports the documents it cannot rank.
        from .validate import priority_findings

        for found in priority_findings(a.pkg, cfg.priority_classes):
            a.notes.append(("priority_classes", found.message))
    reads_request = (
        organizers.requirements_tracker.mode == "on"
        or "request" in cfg.obligations.producers
        or ledger
        or (checkers.gate.mode == "on" and checkers.gate.seed_request)
        or (cfg.sources.request and "requirements_tracker" in later)  # the level decides whether it reads it
    )
    if cfg.sources.request != reads_request:
        raise a.fail(
            "sources.request",
            f"is {cfg.sources.request}, and the components that read the request are "
            f"{'on' if reads_request else 'off'} (the requirements tracker, the producers request and package, a gate "
            "with seed_request)",
        )
    if cfg.notes.brief == "message":
        if a.ctx.host_note_placement != "message":
            raise a.fail("notes.brief", "is message, and the host's note placement (RunContext.host_note_placement) is not")


# -- assembly ----------------------------------------------------------------------------------------------------------


def build_runtime(
    cfg: PolicyConfig, pkg: Optional[PolicyPackage], ctx: RunContext, *, models: Any = None
) -> RuntimeParts:
    """The parts of one run. ``pkg`` is the opened package, or ``None`` when there is none for this task; a config that
    needs one then raises. ``models``: the model factory, by default the real one; a dry run gives
    ``dry_run.RecordingModels``. Raises ``RuntimeWiringError`` on wiring that can have no effect.

    The level switch (``levels.py``) runs first: it replaces every ``auto`` of the config by ``on`` or ``off`` from the control
    level of ``ctx.agent_model`` and the shapes of the policy sources, and what it decided goes to the audit. A profile that
    lists every component has no ``auto``, so nothing changes for it. At level 0 every hook is wrapped in ``Shadow``: it runs
    and says nothing to the agent."""
    resolution = resolve(
        cfg,
        pkg,
        ctx.agent_model,
        policy_text=ctx.policy_text or ctx.system_prompt,
        message_tool=ctx.message_tool,
        tool_names=[
            t["name"] for t in ctx.tool_specs if ctx.tools_in_run is None or t["name"] in ctx.tools_in_run
        ],
    )
    cfg = resolution.cfg
    a = Assembly(cfg, pkg, ctx, models if models is not None else Models(), resolution)
    _check_settings(a)
    parts = RuntimeParts()
    for name, component in COMPONENTS.items():
        try:
            piece = component.build(a) if component.enabled(cfg) else None
        except RuntimeWiringError:
            raise
        except Exception as exc:
            raise a.fail(name, f"{type(exc).__name__}: {exc}") from exc
        if piece is not None:
            _add(a, parts, name, piece)
    if a.held:
        _add_write_checks(a, parts)
    if pkg is not None and not a.package_read:
        raise a.fail(
            "package", f"the profile names the package {cfg.package}, and no component that is on reads it"
        )

    def usage() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for component, model in a.made:
            several = any(c == component and m.model_name != model.model_name for c, m in a.made)
            row = out.setdefault(
                f"{component}:{model.model_name}" if several else component, {"model": model.model_name}
            )
            for k, v in (getattr(model, "usage", {}) or {}).items():
                row[k] = row.get(k, 0) + v
        return out

    def records() -> List[Any]:
        slots = [t for t in parts.components.values() if t not in parts.hooks and hasattr(t, "records")]
        return [r for keeper in parts.hooks + slots for r in keeper.records]

    parts.usage, parts.records = usage, records
    if a.audit is not None and parts.components:  # a baseline run, with nothing on, has no trace and no file
        a.audit.start()
        parts.trace, parts.audit = a.trace, a.audit
        for rule_id, text in a.notes:  # what the level switch decided, after the file is made empty
            a.audit.record(
                "levels", None, {"hook": "build", "rule": rule_id, "outcome": "observe", "reason": text}
            )
    return parts


# The checks that WriteChecks runs, when ``checkers.write_checks`` is on: their hooks do not go to the registry themselves.
WRITE_CHECK_PARTS = ("rules", "gate")


def _add(a: Assembly, parts: RuntimeParts, name: str, piece: Piece) -> None:
    held = a.cfg.checkers.write_checks == "on" and name in WRITE_CHECK_PARTS
    shadow = a.level == 0
    if shadow:  # level 0 stops nothing and changes nothing the agent sees
        changes = [
            slot
            for slot in ("tools", "system_note", "run", "turn_note", "wrap_tools")
            if getattr(piece, slot)
        ]
        if changes:
            raise a.fail(
                name,
                f"level 0 stops nothing and shows the agent nothing, and this component gives {changes}: set it off, or "
                "auto, or raise the level",
            )
    hooks = piece.hooks
    if held:
        a.held[name] = piece.hooks[0]
        hooks = []
        if name == "rules":
            parts.hooks.append(_HOLD)  # the place of the write checks among the hooks
    elif shadow and piece.hooks:
        hooks = [Shadow(h, name) for h in piece.hooks]
        for wrapper in hooks:
            _attach_shadow(a, wrapper)
    parts.hooks += hooks
    parts.tools += piece.tools
    if piece.system_note:
        parts.system_note = (
            f"{parts.system_note}\n\n{piece.system_note}" if parts.system_note else piece.system_note
        )
    for slot in ("run", "turn_note", "wrap_tools", "after_reply"):
        value = getattr(piece, slot)
        if value is not None:
            if getattr(parts, slot) is not None:
                raise a.fail(name, f"a second component fills the slot {slot}, and a run has one")
            if slot == "after_reply" and shadow:
                quiet = Shadow(None, name)  # level 0: the new prompt is a line, not sent
                _attach_shadow(a, quiet)
                value = quiet.quiet_after_reply(value)
            setattr(parts, slot, value)
    target = piece.target if piece.target is not None else (piece.hooks[0] if piece.hooks else None)
    if target is not None:
        parts.components[name] = target
    if a.audit is not None:
        for owner in [*piece.hooks, target, *piece.audited]:
            a.audit.attach(name, owner)


def _attach_shadow(a: Assembly, wrapper: Shadow) -> None:
    if a.audit is not None:
        a.audit.attach("shadow", wrapper)


_HOLD = object()  # a marker in ``parts.hooks`` until ``_add_write_checks`` puts the hook there


def _add_write_checks(a: Assembly, parts: RuntimeParts) -> None:
    """One hook for the rules and the gate, in the place where the rules' hook would be (spec section 6)."""
    cfg = a.cfg
    check = WriteChecks(
        lambda tool: explain_level(cfg, a.ctx.agent_model, tool).level,
        rules=a.held.get("rules"),
        gate=a.held.get("gate"),
        auto=a.resolution.auto_stages,
    )
    hook: Any = Shadow(check, "write_checks") if a.level == 0 else check
    index = parts.hooks.index(_HOLD) if _HOLD in parts.hooks else len(parts.hooks)
    parts.hooks[:] = [h for h in parts.hooks if h is not _HOLD]
    parts.hooks.insert(index, hook)
    parts.components["write_checks"] = check
    if a.audit is not None:
        a.audit.attach("write_checks", check)
        if hook is not check:
            _attach_shadow(a, hook)
