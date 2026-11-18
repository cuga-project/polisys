"""Runtime assembly: one config, one package and one task give the hooks, tools, notes and slots of a run (design spec,
sections 3 and 13). ``build_runtime`` is the only place that wires the policy layer.

    parts = build_runtime(load_config(path), load_package(package_path), RunContext(task_id=..., tool_specs=...))

``COMPONENTS`` (``runtime/registry.py``) is the registry: a name, when the config turns it on, and how to build it. The
order of the entries is the order of the hooks. Each component's ``build`` function is in the component's own file; it
gets an ``Assembly`` (the config, the package, the task and the models made so far) and returns a ``Piece``. A component
reads no policy switch, and neither does this module. ``runtime/settings_check.py`` refuses settings that cannot work.

**Wiring that can have no effect raises.** The run must not go on and measure something else with no word about it: a
gate with no model name would use the agent's model; a gate on a store that the agent's tools do not use checks almost
nothing; a guide on a package whose steps name no tool never stops. ``RuntimeWiringError`` names the task, the policy
version and the component.

This module knows no benchmark. A benchmark's facts come in through ``RunContext``; ``build_runtime`` runs inside each
task's child process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from polisys.package.model import PolicyPackage
from polisys.runtime.context import Component, Piece, RunContext, RuntimeParts, RuntimeWiringError
from polisys.runtime.levels import Resolution, Shadow, WriteChecks, explain_level, resolve
from polisys.runtime.models import Models
from polisys.runtime.profile import PolicyConfig, ledger_on
from polisys.runtime.registry import COMPONENTS
from polisys.runtime.settings_check import check_settings
from polisys.runtime.trace import AUDIT_FILE, AuditLog, PolicyTrace

__all__ = [
    "COMPONENTS",
    "Assembly",
    "Component",
    "Models",
    "Piece",
    "RunContext",
    "RuntimeParts",
    "RuntimeWiringError",
    "build_runtime",
]


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
            from polisys.experimental.guidance.procedure_store import ProcedureStore

            self._store = ProcedureStore.from_package(self.package(component))
        return self._store

    def scope_reader(self, component: str) -> Any:
        """The one ``ScopeReader`` of this run (``checks/router.py``): the router and expander of ``checkers.router``, over the
        package and the shared store. The gate reads its scope from it (the ledger has no retrieval scope for now). A router
        that needs a model gets the traced one of ``checkers.router.model`` (``llm_cards``; the measured router had a timeout
        of 240 and 16000 tokens) or of ``checkers.narrow_questions`` (``bm25_kev``)."""
        if self._scope is None:
            from polisys.checks.router import ScopeReader, make_expander, make_router, router_needs

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
        from polisys.checks.verifier import Verifier

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

    @property
    def ledger_on(self) -> bool:
        """With a producer, only the obligation ledger sends follow-ups (``profile.ledger_on``)."""
        return ledger_on(self.cfg)

    def own_model(self, component: str, name: Optional[str], **kwargs: Any) -> Any:
        """The model of a component whose setting may say ``agent``: the model the agent runs on."""
        if name == "agent":
            if not self.ctx.agent_model:
                raise self.fail(component, "its model is 'agent', and RunContext.agent_model is empty")
            name = self.ctx.agent_model
        return self.model(component, name, **kwargs)

    def policy_text(self, component: str) -> str:
        text = self.ctx.policy_text or self.ctx.system_prompt
        if not text:
            raise self.fail(component, "it reads the policy text, and RunContext has neither policy_text nor system_prompt")
        return text

    def run_tool_specs(self) -> List[Dict[str, Any]]:
        return [t for t in self.ctx.tool_specs if self.ctx.tools_in_run is None or t["name"] in self.ctx.tools_in_run]


# -- assembly ----------------------------------------------------------------------------------------------------------


def build_runtime(
    cfg: PolicyConfig, pkg: Optional[PolicyPackage], ctx: RunContext, *, models: Any = None
) -> RuntimeParts:
    """The parts of one run. ``pkg`` is the opened package, or ``None`` when there is none for this task; a config that
    needs one then raises. ``models``: the model factory, by default the real one; a dry run gives
    ``dry_run.RecordingModels``. Raises ``RuntimeWiringError`` on wiring that can have no effect.

    The level switch (``runtime/levels.py``) runs first: it replaces every ``auto`` of the config by ``on`` or ``off`` from the control
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
    check_settings(a)
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
