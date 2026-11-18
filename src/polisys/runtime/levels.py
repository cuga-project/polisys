"""The level switch (design spec sections 6, 7 and 8): how much control a run gets, and the hook that runs the write checks.

    resolve_level(cfg, agent_model, tool)  ->  0..3      the control level of a call
    components_for(level, shape)           ->  {component: bool}
    resolve(cfg, pkg, agent_model, ...)    ->  Resolution  (a copy of the config with every ``auto`` replaced)
    WriteChecks(level_of, rules=, gate=)                 the write checks in the spec's order, stopping at the first block
    Shadow(inner, name)                                  level 0: runs a hook, drops what it would have said

**The levels.** 0 observe: every check runs and records what it would do, nothing reaches the agent. 1 advise: the brief and
the notes; hard rules deny. 2 check: procedure tools, the judgment gate, narrow questions where wired. 3 control: the
verifier, the guide, the SOP loop, the trackers. Level 0 is a mode of the whole run, not a floor: it is the one level that
``risk.level3_tools`` does not raise. A baseline run has no parts at all, which is not level 0.

**Which level.** (1) static: ``control.level`` 0 to 3, or ``by_model``: the agent's model in ``control.by_model``; a model
that is not in the table takes ``control.default_level``, and the audit says so. (2) by risk: a tool that matches a pattern of
``control.risk.level3_tools`` gets level 3 for any model that is at level 1 or more.

**Which components.** ``components_for(level, shape)`` is the table of spec sections 7 and 8, by the shape of the policy
source. It decides only the components that the profile marks ``auto``: an explicit ``on`` or ``off`` always wins, so the best
measured profiles, which list every component, are not changed by this module. The table is a hypothesis until the level by
model experiment (spec section 7). A component that ``auto`` turns on but that cannot be built for this run (no model, no
procedures, no hand-off settings) is turned off and the reason goes to the audit: ``auto`` never raises, ``on`` does.

**Model strength.** ``control.by_model`` is also the table of strength: a lower level is a stronger model. ``gate.min_strength:
agent`` needs the gate's model at the same level as the agent's or lower. A model that is not in the table is unknown: it
does not raise, it is reported, and the gate stays on only when the profile says ``on``.

**WriteChecks.** One hook for the typed rules, the gate and the verifier, in the order of spec section 6, cheapest first.
It stops at the first block, so a hard deny means the gate model is not called. A check whose level is not reached is skipped
and writes no line; each check that ran writes one line (component ``write_checks``, rule id the stage), a model check that
failed open writes ``error``. The order realizes the precedence ``deny`` > ``hold`` > ``block_once`` > ``remind`` > ``allow``:
only ``deny`` and ``block_once`` stop a call, the first one decides, and a reminder goes to the tool result as it does today.
No stage holds a call yet (``hold`` is the intake's ``retry_effect``). The narrow questions of Kev are not a stage here: they
are wired on the online path only.
"""

from __future__ import annotations

import copy
import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from polisys.hooks import AgentHooks, HookResult
from loguru import logger

from .policy_config import PolicyConfig
from .router import router_reads_opened
from .trace import fresh, hook_failure

LEVELS = (0, 1, 2, 3)
CHECKS = ("rules", "gate", "verifier", "decisions")
ORGANIZERS = ("procedure_tools", "guide", "loop", "sop_tracker", "navigator", "requirements_tracker")
COMPONENTS = CHECKS + ORGANIZERS
SHAPES = ("procedures", "data", "flow", "rules")

# The first level at which a component is on, by shape (spec sections 7 and 8). Level 0 turns every check of the shape on, in
# shadow, and no organizer.
FIRST_LEVEL: Dict[str, Dict[str, int]] = {
    # a given SOP with procedures: router, gate and ledger; guide and loop at level 3
    "procedures": {"rules": 1, "procedure_tools": 2, "gate": 2, "verifier": 3, "guide": 3, "loop": 3},
    # rules in business data: discovery, online intake, narrow questions, the request tracker at level 3
    "data": {"rules": 1, "decisions": 2, "verifier": 3, "requirements_tracker": 3},
    # a step flow with references: a note with each customer turn, the hand-off check at level 3
    "flow": {"rules": 1, "sop_tracker": 2, "navigator": 3, "verifier": 3},
    # a rule list with exceptions: hard rules and the brief
    "rules": {"rules": 1, "verifier": 3},
}

# Where each component's mode lives in the config.
MODE_PATH: Dict[str, Tuple[str, ...]] = {
    "rules": ("checkers", "rules"),
    "gate": ("checkers", "gate", "mode"),
    "verifier": ("checkers", "verifier", "mode"),
    "decisions": ("decisions", "mode"),
    "procedure_tools": ("organizers", "procedure_tools"),
    "guide": ("organizers", "guide", "mode"),
    "loop": ("organizers", "loop", "mode"),
    "sop_tracker": ("organizers", "sop_tracker", "mode"),
    "navigator": ("organizers", "navigator", "mode"),
    "requirements_tracker": ("organizers", "requirements_tracker", "mode"),
}
# A stage of the write checks, with the level from which it runs.
STAGE_LEVEL = {"rules": 1, "gate": 2, "verifier": 3}
PRECEDENCE = ("deny", "hold", "block_once", "remind", "allow")


# -- the table ----------------------------------------------------------------------------------------------------------


def components_for(level: int, shape: Union[str, Iterable[str]]) -> Dict[str, bool]:
    """Which components are on at a level, for one shape or the union of several (one run can mix shapes, spec section 8).
    Every component is a key. This is a table of defaults: ``resolve`` applies it to ``auto`` components only."""
    if isinstance(level, bool) or level not in LEVELS:
        raise ValueError(f"level {level!r} is not one of {', '.join(map(str, LEVELS))}")
    shapes = [shape] if isinstance(shape, str) else list(shape)
    out = {name: False for name in COMPONENTS}
    for one in shapes:
        if one not in FIRST_LEVEL:
            raise ValueError(f"shape {one!r} is not one of {', '.join(SHAPES)}")
        for name, first in FIRST_LEVEL[one].items():
            if level >= first or (level == 0 and name in CHECKS):
                out[name] = True
    return out


def shapes_of(cfg: PolicyConfig, pkg: Any) -> List[str]:
    """The shapes of this run: ``cfg.shapes`` when it says, else from what the run has. A package with procedures is a given
    SOP; with ``refers_to`` links also a step flow; with rules and no procedures a rule list; ``sources.data`` is rules in
    business data. A step flow with no package (its policy is only text) has to be declared."""
    if cfg.shapes:
        return list(cfg.shapes)
    out: List[str] = []
    if pkg is not None:
        if pkg.procedures:
            out.append("procedures")
        if any(str(getattr(link.type, "value", link.type)) == "refers_to" for link in pkg.links):
            out.append("flow")
        if pkg.rules and not pkg.procedures:
            out.append("rules")
    if cfg.sources.data:
        out.append("data")
    return out


# -- the level of a call ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelInfo:
    level: int
    why: str
    known: bool = True  # False: the model is not in the table and the default level was taken


def explain_level(cfg: PolicyConfig, agent_model: str = "", tool: Optional[str] = None) -> LevelInfo:
    """The level and the reason for it. ``tool=None``: the level of the run."""
    control = cfg.control
    if control.level != "by_model":
        info = LevelInfo(int(control.level), f"control.level is {control.level}")
    elif agent_model and agent_model in control.by_model:
        info = LevelInfo(control.by_model[agent_model], f"control.by_model has {agent_model}")
    else:
        who = (
            f"the agent model {agent_model!r} is not in control.by_model"
            if agent_model
            else "no agent model is given"
        )
        info = LevelInfo(
            control.default_level, f"{who}: control.default_level {control.default_level}", False
        )
    if tool and info.level >= 1 and _risky(cfg, tool):
        return LevelInfo(
            3, f"{tool} matches control.risk.level3_tools; the run is at level {info.level}", info.known
        )
    return info


def resolve_level(cfg: PolicyConfig, agent_model: str = "", tool: Optional[str] = None) -> int:
    """The control level of a call: static by the agent's model, raised to 3 for a tool in ``risk.level3_tools``. Level 0 is
    a mode of the run, so no tool is raised out of it."""
    return explain_level(cfg, agent_model, tool).level


def _matches(pattern: str, tool: str) -> bool:
    names = {tool, tool.rsplit("__", 1)[-1]}  # the tool of an app can carry a prefix: crm__delete_account
    return any(fnmatch.fnmatchcase(name, pattern) for name in names)


def _risky(cfg: PolicyConfig, tool: str) -> bool:
    return any(_matches(pattern, tool) for pattern in cfg.control.risk.level3_tools)


def gate_strength(cfg: PolicyConfig, agent_model: str) -> Tuple[str, str]:
    """Compare the gate's model with the agent's by ``control.by_model`` (a lower level is a stronger model).

    ``("none", ...)``: ``min_strength`` is none, nothing is compared. ``("ok", ...)``: the gate is as strong as the agent or
    stronger. ``("weaker", ...)``. ``("unknown", ...)``: a model is not in the table, or none is given."""
    gate = cfg.checkers.gate
    if gate.min_strength == "none":
        return "none", "gate.min_strength is none: the gate's model is not compared with the agent's"
    table = cfg.control.by_model
    missing = [
        f"{role} {name!r}"
        for role, name in (("gate model", gate.model), ("agent model", agent_model))
        if not name or name not in table
    ]
    if missing:
        return "unknown", f"{' and '.join(missing)} not in control.by_model, so their strength is not known"
    if table[gate.model] <= table[agent_model]:
        return (
            "ok",
            f"gate {gate.model} (level {table[gate.model]}) is as strong as agent {agent_model} (level {table[agent_model]}) or stronger",
        )
    return (
        "weaker",
        f"gate {gate.model} (level {table[gate.model]}) is weaker than agent {agent_model} (level {table[agent_model]})",
    )


# -- auto -> on or off --------------------------------------------------------------------------------------------------


@dataclass
class Resolution:
    cfg: PolicyConfig  # a copy with no ``auto`` left
    level: int  # of the run
    check_level: int  # of the checks that exist: 3 when a tool is in risk.level3_tools
    shapes: List[str]
    notes: List[Tuple[str, str]] = field(default_factory=list)  # (id, text) for the audit
    # Components that ``auto`` left off. A setting whose reader is one of them is not a wiring mistake: the level decides, and a
    # higher level would read it.
    pending: Set[str] = field(default_factory=set)
    # The write checks that ``auto`` turned on: only these follow the level of a call. A check the profile says ``on`` runs
    # for every call, whatever the level (explicit wins).
    auto_stages: Set[str] = field(default_factory=set)
    # Does anything read the control level? Without an ``auto``, ``write_checks`` on, a ledger or level 0, a control setting
    # changes nothing, and ``build_runtime`` raises on it.
    level_read: bool = False
    risk_active: bool = False  # some pattern of ``risk.level3_tools`` matches a tool of the task
    had_auto: bool = False  # the profile left some component to the level


def _get(cfg: PolicyConfig, name: str) -> str:
    node: Any = cfg
    for key in MODE_PATH[name]:
        node = getattr(node, key)
    return node


def _set(cfg: PolicyConfig, name: str, value: str) -> None:
    *head, last = MODE_PATH[name]
    node: Any = cfg
    for key in head:
        node = getattr(node, key)
    setattr(node, last, value)


def _has_actions(pkg: Any) -> bool:
    """A step of the package names a tool, so the guide has one to stop on (the check ``_build_guide`` makes)."""
    if pkg is None:
        return False
    from .package import procedure_dicts

    return any(
        s.get("action") for p in procedure_dicts(pkg) for s in p.get("steps") or [] if isinstance(s, dict)
    )


def _has_judgment(pkg: Any) -> bool:
    """A package judgment rule on a tool it names (what the verifier of scope ``specific_statements`` takes)."""
    if pkg is None:
        return False
    for rule in pkg.rules:
        body = rule.rule if isinstance(rule.rule, dict) else {}
        if (
            body.get("type") == "verify"
            and not body.get("routed_from")
            and (body.get("tools") or body.get("tool"))
        ):
            return True
    return False


def resolve(
    cfg: PolicyConfig,
    pkg: Any,
    agent_model: str = "",
    policy_text: str = "",
    message_tool: Optional[str] = None,
    tool_names: Sequence[str] = (),
) -> Resolution:
    """Replace every ``auto`` of the config by ``on`` or ``off``. Explicit modes stay. ``policy_text``: the SOP text of the
    task, for what reads the text and not a package (the SOP tracker, the gate on the whole SOP). ``tool_names``: the tools of
    the task, to check the patterns of ``risk.level3_tools`` against."""
    from .package import runtime_rules
    from .store_gate import _has_gate_rules

    out = copy.deepcopy(cfg)
    auto_checks = {n for n in ("rules", "gate", "verifier") if _get(out, n) == "auto"}
    had_auto = any(_get(out, n) == "auto" for n in MODE_PATH)
    info = explain_level(cfg, agent_model)
    patterns = cfg.control.risk.level3_tools
    unmatched = [p for p in patterns if tool_names and not any(_matches(p, t) for t in tool_names)]
    risk_active = bool(patterns) and len(unmatched) < len(patterns)  # a typo in every pattern raises no tool
    check_level = 3 if risk_active and info.level >= 1 else info.level
    shapes = shapes_of(cfg, pkg)
    notes: List[Tuple[str, str]] = [
        ("level", f"level {info.level}: {info.why}"),
        ("shape", f"shapes: {', '.join(shapes) or 'none (no package, no sources.data)'}"),
    ]
    for pattern in unmatched:
        notes.append(("risk_pattern", f"control.risk.level3_tools {pattern!r} matches no tool of the task"))
    if patterns and cfg.sources.data and info.level < 3:
        notes.append(
            (
                "risk_verifier",
                "the verifier of a run with sources.data is the online intake's and follows the level of the run: a tool in "
                "control.risk.level3_tools gets no verifier here",
            )
        )
    if check_level != info.level:
        notes.append(
            (
                "check_level",
                f"the checks are built for level {check_level}: a tool is in control.risk.level3_tools",
            )
        )
    base, checks = components_for(info.level, shapes), components_for(check_level, shapes)
    pending: Set[str] = set()
    ledger = bool(cfg.obligations.producers)
    has_rules = pkg is not None and bool(runtime_rules(pkg))

    def can_build(name: str) -> str:
        """Why this component cannot be built for this run, or ''. ``out`` already holds what was decided before it."""
        c = out
        if name == "rules":
            return "" if c.sources.data or has_rules else "the package has no rules and sources.data is off"
        if name == "procedure_tools":
            return "" if pkg is not None and pkg.procedures else "the package has no procedures to read"
        if name == "gate":
            if not c.checkers.gate.model:
                return "checkers.gate.model is none"
            status, text = gate_strength(c, agent_model)
            if status in ("weaker", "unknown"):
                return f"{text}; with auto the gate stays off, and only 'on' keeps it"
            if c.checkers.gate.scope == "active_procedures":
                if _get(c, "procedure_tools") != "on" and router_reads_opened(c.checkers.router.kind):
                    return "scope active_procedures with the router agent_opened needs procedure_tools, and they are off"
                if pkg is None or not _has_gate_rules(pkg):
                    return "the package has no must-not rule or decision to check"
            elif not policy_text:
                return "scope whole_sop needs the SOP text, and the task gives none"
            return ""
        if name == "verifier":
            if not c.checkers.verifier.model:
                return "checkers.verifier.model is none"
            if c.sources.data:
                if c.checkers.verifier.model != c.checkers.compile:
                    return "the online intake has one model: checkers.compile differs from checkers.verifier.model"
                return ""
            if _get(c, "rules") != "on":
                return "the verifier reads the rules, and they are off"
            if c.checkers.verifier.scope == "specific_statements" and not _has_judgment(pkg):
                return "the package has no judgment rule on a tool"
            return ""
        if name == "decisions":
            if not c.sources.data:
                return "narrow questions are wired on the online path, and sources.data is off"
            return "" if c.checkers.narrow_questions else "checkers.narrow_questions is none"
        if name == "guide":
            return "" if _has_actions(pkg) else "no step of the package names a tool"
        if name == "loop":
            return "" if c.organizers.loop.model else "organizers.loop.model is none"
        if name == "sop_tracker":
            # Package structure cannot tell a step flow from a rule list with references: a rule list and a step flow can both
            # have procedures, steps and ``refers_to`` links. So the tracker under ``auto`` needs a declared flow.
            if "flow" not in c.shapes:
                return "a step flow is declared in policy.shapes, not inferred"
            if not c.organizers.sop_tracker.model:
                return "organizers.sop_tracker.model is none"
            return "" if policy_text else "the task gives no SOP text"
        if name == "navigator":
            nav = c.organizers.navigator
            if ledger:
                return "the obligation ledger checks the same hand-offs"
            if not (nav.handoff_text or nav.handoff_tools):
                return "no hand-off phrase and no hand-off tool"
            if message_tool is None:
                return "the task does not say which tool sends a message"
            if pkg is None or not c.checkers.narrow_questions:
                return "it needs a package and checkers.narrow_questions"
            return ""
        if name == "requirements_tracker":
            if ledger:
                return "with a producer only the ledger sends follow-ups"
            if not c.sources.request:
                return "sources.request is off, so the request is not read"
            return (
                ""
                if c.organizers.requirements_tracker.model
                else "organizers.requirements_tracker.model is none"
            )
        raise ValueError(name)

    # order: what another component needs comes first
    for name in (
        "rules",
        "procedure_tools",
        "gate",
        "verifier",
        "decisions",
        *("guide", "loop", "sop_tracker", "navigator", "requirements_tracker"),
    ):
        if _get(out, name) != "auto":
            continue
        # A check that WriteChecks holds (the rules, the gate, the verifier of the package's rules) can be raised for one tool,
        # so it is built for the highest level any tool has. What runs for the whole run, as the intake's verifier and its narrow
        # questions do, follows the level of the run.
        per_tool = name in ("rules", "gate") or (name == "verifier" and not out.sources.data)
        at = check_level if per_tool else info.level
        wanted = (checks if per_tool else base)[name]
        why = (
            can_build(name)
            if wanted
            else f"not on at level {at} for the shapes {', '.join(shapes) or 'none'}"
        )
        _set(out, name, "off" if why else "on")
        if why:
            pending.add(name)
        notes.append((f"auto:{name}", f"off: {why}" if why else f"on at level {at}"))

    if out.checkers.write_checks == "auto":
        rules_built = _get(out, "rules") == "on" and (
            not out.sources.data or has_rules
        )  # else the intake holds the rules
        built = [
            n for n in ("rules", "gate", "verifier") if _get(out, n) == "on" and (n == "gate" or rules_built)
        ]
        decided = [n for n in built if n in auto_checks]
        # Two checks to put in order, one of them decided by the level; or a tool that needs its own level, which only
        # WriteChecks can give (it never raises under auto).
        on = (len(built) >= 2 and bool(decided)) or (risk_active and bool(built))
        out.checkers.write_checks = "on" if on else "off"
        notes.append(
            (
                "auto:write_checks",
                f"{out.checkers.write_checks}: on write checks {', '.join(built) or 'none'}; the level switch turned on "
                f"{', '.join(decided) or 'none'}"
                + ("; a tool is in control.risk.level3_tools" if risk_active else ""),
            )
        )
    if not (
        had_auto
        or info.level == 0
        or check_level != info.level
        or out.checkers.write_checks == "on"
        or patterns
    ):
        notes = []  # a profile that lists every component has nothing to report: the level switch decided nothing
    return Resolution(
        out,
        info.level,
        check_level,
        shapes,
        notes,
        pending,
        auto_stages={n for n in auto_checks if _get(out, n) == "on"},
        level_read=had_auto
        or out.checkers.write_checks == "on"
        or (ledger and cfg.obligations.item_level is None)
        or info.level == 0,
        risk_active=risk_active,
        had_auto=had_auto,
    )


# -- level 0 ------------------------------------------------------------------------------------------------------------


class Shadow(AgentHooks):
    """Level 0 around one hook: the hook runs as it always does, and nothing it says reaches the agent. A message that would
    have stopped a call, been added to a tool result or sent back at the reply is a line of the audit with the verdict
    ``would_block``, ``would_note`` or ``would_followup``; the hook's own records are the same decisions it always writes,
    and at this level they are what it would have done. The brief goes to no model call and writes no line, so a run at level
    0 shows the agent what a run with no policy shows it. ``tool_blocked`` is not passed on: no call was blocked."""

    def __init__(self, inner: Any, name: str):
        self.inner, self.name = inner, name
        self.records: List[Dict[str, Any]] = []

    def log(self, hook: str, tool: str, verdict: str, text: Any) -> None:
        self.records.append(
            {"hook": hook, "tool": tool, "rule_id": self.name, "outcome": verdict, "reason": str(text)[:2000]}
        )

    def _quiet(self, hook: str, tool: str, verdict: str, result: Any) -> None:
        message = getattr(result, "message", result)
        answer = getattr(result, "answer", None)
        if isinstance(message, str) and message:
            self.log(hook, tool, verdict, message)
        if isinstance(answer, str) and answer:
            self.log(hook, tool, "would_answer", answer)

    def session_start(self, thread_id: Optional[str]) -> Any:
        self.inner.session_start(thread_id)

    def on_user_message(self, message: str, thread_id: Optional[str]) -> Any:
        self._quiet("on_user_message", "", "would_answer", self.inner.on_user_message(message, thread_id))

    def before_llm_call(self, thread_id: Optional[str]) -> Any:
        self.inner.before_llm_call(thread_id)  # the brief: no line, it is not a verdict

    def after_llm_call(self, text: str) -> Any:
        self.inner.after_llm_call(text)

    def after_plan(self, code: str) -> Any:
        self._quiet("after_plan", "", "would_block", self.inner.after_plan(code))

    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> Any:
        self._quiet("before_tool_call", tool, "would_block", self.inner.before_tool_call(tool, args))

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> Any:
        self._quiet("after_tool_call", tool, "would_note", self.inner.after_tool_call(tool, args, result))

    def before_reply(self, answer: str) -> Any:
        self._quiet("before_reply", "", "would_followup", self.inner.before_reply(answer))

    def on_task_end(self, answer: str) -> Any:
        self.inner.on_task_end(answer)

    def quiet_after_reply(self, after_reply: Callable[..., Any]) -> Callable[..., Any]:
        """The slot ``after_reply`` of a component, with its answer dropped (the new prompt it would have sent is a line)."""

        def quiet(answer: Any, turns: Any, live: Any) -> Any:
            out = after_reply(answer, turns, live)
            if out:
                self.log("after_reply", "", "would_followup", out)
            return None

        return quiet


# -- the write checks ---------------------------------------------------------------------------------------------------


class WriteChecks(AgentHooks):
    """The typed rules, the gate and the verifier as one hook, in the order of spec section 6.

    ``level_of(tool)`` gives the control level of a call (``resolve_level``). A stage that ``auto`` turned on runs from its
    level: rules from 1, the gate from 2, the verifier from 3, and below it is skipped with no line. A stage the profile says
    ``on`` runs for every call (explicit wins); ``auto`` names the stages that follow the level, and the default is all of them. At level 0 every stage runs, none
    stops the call, and each writes what it would have done (``would_deny``, ``would_block_once``, ``would_remind``). Above
    that the first stage that blocks ends the check, so a hard deny means the gate's model is not called. A model check that
    fails open (the gate's model or the verifier raised) is the verdict ``error`` and the next stage runs.

    ``rules``: a ``RuleSetHooks``; ``gate``: a ``StoreGate``. Their other hooks (facts, reads, the brief, reminders in the tool
    result, duties) run for every event, as in the registry, in that order."""

    def __init__(
        self,
        level_of: Callable[[str], int],
        rules: Any = None,
        gate: Any = None,
        auto: Optional[Iterable[str]] = None,
    ):
        self.level_of = level_of
        self._rules = rules
        self._auto = set(STAGE_LEVEL if auto is None else auto)
        self.children = [c for c in (rules, gate) if c is not None]
        self.records: List[Dict[str, Any]] = []
        # (stage, owner of last_decision, the call, who decides)
        self._stages: List[Tuple[str, Any, Callable[[str, Dict[str, Any]], Any], str]] = []
        if rules is not None:
            self._stages.append(("rules", rules, rules.check_code, "code"))
        if gate is not None:
            self._stages.append(("gate", gate, gate.before_tool_call, "model"))
        if rules is not None and getattr(rules, "verifier", None) is not None:
            self._stages.append(("verifier", rules, rules.check_verifier, "model"))

    # -- the stages ----------------------------------------------------------------------------------------------------
    def before_tool_call(self, tool: str, args: Dict[str, Any]) -> Any:
        level = self.level_of(tool)
        queued = len(
            getattr(self._rules, "notes", None) or []
        )  # reminders queued by an earlier stage of this call
        for stage, owner, run, checker in self._stages:
            if level and stage in self._auto and level < STAGE_LEVEL[stage]:
                continue
            try:
                result = run(tool, args)
            except (
                Exception
            ) as exc:  # a broken check must not break the agent; its own line says so (hook_failed)
                logger.warning("write check {} failed: {}", stage, exc)
                self._line(stage, tool, level, "error", f"{type(exc).__name__}: {exc}", checker)
                continue
            decision = owner.last_decision
            if decision == "n/a":
                continue  # nothing applied to this call: the check did not run
            message = result if isinstance(result, str) else getattr(result, "message", None)
            self._line(stage, tool, level, decision, message, checker)
            if message and level:
                if self._rules is not None and stage != "rules":
                    del self._rules.notes[
                        queued:
                    ]  # the call did not run: its reminder must not say it "was run"
                return message
        return None

    def _line(self, stage: str, tool: str, level: int, decision: str, message: Any, checker: str) -> None:
        verdict = (
            f"would_{decision}" if level == 0 and decision in ("deny", "block_once", "remind") else decision
        )
        text = f"level {level}: " + (str(message) if message else "no rule broken")
        self.records.append(
            {
                "hook": "before_tool_call",
                "tool": tool,
                "rule_id": stage,
                "outcome": verdict,
                "reason": text,
                "checker": checker,
            }
        )

    # -- everything else goes to the checks, as the registry would -----------------------------------------------------
    def _fan(self, name: str, *args: Any) -> HookResult:
        messages: List[str] = []
        details: List[Dict[str, Any]] = []
        answer: Optional[str] = None
        for child in self.children:
            call_args = (answer, *args[1:]) if name == "before_reply" and answer is not None else args
            try:
                res = getattr(child, name)(*call_args)
            except Exception as exc:  # the same rule as the registry
                logger.warning("write check {}.{} failed: {}", type(child).__name__, name, exc)
                hook_failure(child if getattr(child, "_audit_sink", None) else self, name, call_args, exc)
                continue
            if isinstance(res, str):
                res = HookResult(message=res or None)
            if res is None:
                continue
            if res.message:
                messages.append(res.message)
            details.extend(res.details)
            if res.answer is not None:
                answer = res.answer
        return HookResult(message="\n\n".join(messages) or None, details=details, answer=answer)

    def session_start(self, thread_id: Optional[str]) -> Any:
        self.records = fresh(self.records)  # a new session, the same list when it is tapped
        return self._fan("session_start", thread_id)

    def on_user_message(self, message: str, thread_id: Optional[str]) -> Any:
        return self._fan("on_user_message", message, thread_id)

    def before_llm_call(self, thread_id: Optional[str]) -> Any:
        return self._fan("before_llm_call", thread_id)

    def after_llm_call(self, text: str) -> Any:
        return self._fan("after_llm_call", text)

    def after_plan(self, code: str) -> Any:
        return self._fan("after_plan", code)

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> Any:
        return self._fan("after_tool_call", tool, args, result)

    def before_reply(self, answer: str) -> Any:
        return self._fan("before_reply", answer)

    def on_task_end(self, answer: str) -> Any:
        return self._fan("on_task_end", answer)
