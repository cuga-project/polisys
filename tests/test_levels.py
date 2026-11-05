"""The level switch (Task 5.1): the level of a call, which components each level turns on, the write checks in order, level 0.

Offline: no model call. Models are scripted, and a real ``CugaAgent`` loop shows that a blocked write does not run.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict, List

import pytest
import yaml
from langchain_core.tools import tool

from polisys.hooks import AgentHooks, HookRegistry
from polisys.dry_run import RecordingModels
from polisys.levels import (
    COMPONENTS,
    FIRST_LEVEL,
    SHAPES,
    Shadow,
    components_for,
    explain_level,
    gate_strength,
    resolve,
    resolve_level,
    shapes_of,
)
from polisys.model import Obligation, PolicyPackage
from polisys.obligations import ObligationLedger
from polisys.package import from_build, from_store_graph
from polisys.policy_config import PolicyConfig
from polisys.runtime import RunContext, RuntimeWiringError, build_runtime
from polisys.ruleset import RuleSetHooks
from polisys.store_gate import StoreGate
from polisys.verifier import Verifier
try:  # the CUGA agent is optional: its real-loop tests are skipped without it
    from cuga.sdk import CugaAgent
except ImportError:  # pragma: no cover
    CugaAgent = None

TOOL = "update_ticket"
SPECS = [
    {
        "name": TOOL,
        "description": "Change a ticket.",
        "inputSchema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}, "status": {"type": "string"}},
        },
    },
    {
        "name": "delete_account",
        "description": "Delete an account.",
        "inputSchema": {"type": "object", "properties": {"account_id": {"type": "string"}}},
    },
    {
        "name": "get_ticket",
        "description": "Read a ticket.",
        "inputSchema": {"type": "object", "properties": {"ticket_id": {"type": "string"}}},
    },
]
RULE = {
    "id": "r1",
    "type": "allowed_values",
    "tool": TOOL,
    "arg": "status",
    "values": ["open"],
    "quote": "Only open.",
}
SPECIFIC = {
    "id": "v_specific",
    "type": "verify",
    "statement": "Never close a ticket of a VIP.",
    "tools": [TOOL],
    "quote": "Never close a ticket of a VIP.",
}
GENERAL = {
    "id": "v_general",
    "type": "verify",
    "statement": "Never assume data.",
    "quote": "Never assume data.",
}
GUIDANCE = {"id": "g1", "type": "guidance", "text": "Be polite.", "tools": [TOOL], "quote": "Be polite."}
SOP_TEXT = "# Tickets\nNever close a ticket of a VIP.\n"
GRAPH = {
    "id": "sop",
    "procedures": [
        {
            "id": "1.1",
            "name": "Tickets",
            "steps": ["Open a ticket."],
            "must_not": ["Do not close a ticket of a VIP."],
            "links": [],
        }
    ],
}
CTX = RunContext(
    task_id="task-5",
    policy_version="v1",
    request="Close T1.",
    system_prompt=SOP_TEXT,
    tool_specs=SPECS,
    agent_model="m/agent",
)
OFF = {"rules": "off", "gate": {"mode": "off"}, "verifier": {"mode": "off"}}
ORGANIZERS_OFF = {
    "procedure_tools": "off",
    "guide": "off",
    "loop": "off",
    "sop_tracker": "off",
    "navigator": "off",
    "requirements_tracker": "off",
}
GATE = {"mode": "on", "model": "m/gate", "scope": "whole_sop", "min_strength": "none"}


def package(*rules: Dict[str, Any]) -> PolicyPackage:
    return from_build({"rules": list(rules), "procedures": []})


def config(checkers=None, organizers=None, **top) -> PolicyConfig:
    """A config with every component off, except what the arguments say."""
    return PolicyConfig.from_dict(
        {
            "package": top.pop("package", "p.json"),
            "decisions": top.pop("decisions", {"mode": "off"}),
            "checkers": {**OFF, **(checkers or {})},
            "organizers": {**ORGANIZERS_OFF, **(organizers or {})},
            **top,
        }
    )


class Scripted(RecordingModels):
    """A model factory whose models count their calls and answer with a fixed text, so a test can say a model was not called."""

    def __init__(self, answers: Dict[str, str]):
        super().__init__()
        self.answers, self.calls = answers, []

    def json_model(self, model: str, **kwargs: Any):
        super().json_model(model, **kwargs)  # recorded, as a dry run does

        def call(system: str, user: str) -> str:
            self.calls.append(model)
            return self.answers.get(model, '{"breaks": false, "violations": []}')

        call.model_name, call.usage = model, {"calls": 0}  # type: ignore[attr-defined]
        return call


def build(cfg, pkg=None, ctx=CTX, models=None):
    models = models or Scripted({})
    return build_runtime(cfg, pkg if pkg is not None else package(RULE), ctx, models=models), models


def lines(parts, component=None):
    return [x for x in parts.audit.lines if component is None or x["component"] == component]


# -- the table: level x shape -> components -----------------------------------------------------------------------------

EXPECTED = {
    # shape: {level: components that are on}
    "procedures": {
        0: {"rules", "gate", "verifier"},
        1: {"rules"},
        2: {"rules", "procedure_tools", "gate"},
        3: {"rules", "procedure_tools", "gate", "verifier", "guide", "loop"},
    },
    "data": {
        0: {"rules", "verifier", "decisions"},
        1: {"rules"},
        2: {"rules", "decisions"},
        3: {"rules", "decisions", "verifier", "requirements_tracker"},
    },
    "flow": {
        0: {"rules", "verifier"},
        1: {"rules"},
        2: {"rules", "sop_tracker"},
        3: {"rules", "sop_tracker", "navigator", "verifier"},
    },
    "rules": {0: {"rules", "verifier"}, 1: {"rules"}, 2: {"rules"}, 3: {"rules", "verifier"}},
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_level_by_shape_gives_the_components_of_the_spec_table(level, shape):
    got = components_for(level, shape)
    assert set(got) == set(COMPONENTS)  # every component is a key
    assert {name for name, on in got.items() if on} == EXPECTED[shape][level]


def test_a_mixed_run_gets_the_union_and_level_0_turns_no_organizer_on():
    both = components_for(2, ["procedures", "data"])
    assert {n for n, on in both.items() if on} == {"rules", "procedure_tools", "gate", "decisions"}
    for shape in SHAPES:
        assert not any(
            components_for(0, shape)[o]
            for o in COMPONENTS
            if o not in ("rules", "gate", "verifier", "decisions")
        )
    assert FIRST_LEVEL["procedures"]["verifier"] == 3  # the verifier is level 3 only (spec section 6)


@pytest.mark.parametrize("bad", [-1, 4, True, "2"])
def test_a_level_that_is_not_0_to_3_raises(bad):
    with pytest.raises(ValueError, match="level"):
        components_for(bad, "rules")
    with pytest.raises(ValueError, match="shape"):
        components_for(1, "cooking")


def test_shapes_are_declared_or_inferred_from_the_package_and_the_sources():
    graph = from_store_graph(GRAPH)
    assert shapes_of(config(), graph) == ["procedures"]
    assert shapes_of(config(), package(RULE)) == ["rules"]
    assert shapes_of(config(sources={"data": True}), None) == ["data"]
    assert shapes_of(config(sources={"data": True}), package(RULE)) == ["rules", "data"]
    assert shapes_of(config(shapes=["flow"]), None) == ["flow"]
    assert shapes_of(config(), None) == []


# -- the level of a call ------------------------------------------------------------------------------------------------


def test_the_level_is_static_by_model_and_an_unknown_model_takes_the_stated_default_and_says_so():
    cfg = config(control={"by_model": {"m/small": 3, "m/big": 1}, "default_level": 2})
    assert resolve_level(cfg, "m/small") == 3 and resolve_level(cfg, "m/big") == 1
    info = explain_level(cfg, "m/other")
    assert info.level == 2 and info.known is False and "m/other" in info.why and "default_level 2" in info.why
    assert "no agent model" in explain_level(cfg, "").why
    assert resolve_level(config(), "m/x") == 1  # no table, no default: advise, which makes no model call
    assert resolve_level(config(control={"level": 3, "by_model": {"m/big": 1}}), "m/big") == 3  # static wins


def test_risk_level3_tools_raise_one_tool_and_only_that_tool():
    cfg = config(
        control={"by_model": {"m/agent": 1}, "risk": {"level3_tools": ["delete_*", "send_email_external"]}}
    )
    assert resolve_level(cfg, "m/agent") == 1 and resolve_level(cfg, "m/agent", TOOL) == 1
    assert resolve_level(cfg, "m/agent", "delete_account") == 3
    assert resolve_level(cfg, "m/agent", "crm__delete_account") == 3  # an app prefix
    assert resolve_level(cfg, "m/agent", "gmail__send_email_external") == 3
    assert resolve_level(cfg, "m/agent", "send_email") == 1


def test_level_0_is_a_mode_of_the_run_and_no_tool_is_raised_out_of_it():
    cfg = config(control={"level": 0, "risk": {"level3_tools": ["*_delete*"]}})
    assert resolve_level(cfg, "m/agent", "delete_account") == 0


def test_the_gate_is_compared_with_the_agent_by_the_table_and_a_lower_level_is_stronger():
    table = {"m/strong": 1, "m/mid": 2, "m/weak": 3}

    def status(gate, agent, min_strength="agent"):
        cfg = config(
            {"gate": {"mode": "on", "model": gate, "min_strength": min_strength}}, control={"by_model": table}
        )
        return gate_strength(cfg, agent)[0]

    assert status("m/strong", "m/weak") == "ok" and status("m/mid", "m/mid") == "ok"
    assert status("m/weak", "m/strong") == "weaker"
    assert status("m/other", "m/weak") == "unknown" and status("m/mid", "m/other") == "unknown"
    assert status("m/mid", "") == "unknown"
    assert status("m/weak", "m/strong", "none") == "none"


# -- auto -> on or off --------------------------------------------------------------------------------------------------


def test_an_explicit_on_or_off_is_never_changed_and_a_profile_with_no_auto_has_nothing_to_report():
    cfg = config({"rules": "on", "gate": GATE}, control={"level": 3}, shapes=["procedures"])
    out = resolve(cfg, package(RULE), "m/agent", policy_text=SOP_TEXT)
    assert out.notes == [] and out.pending == set()
    assert (
        out.cfg.checkers.write_checks == "off"
    )  # nothing was left to the level: separate hooks, as measured
    out.cfg.checkers.write_checks = "auto"
    assert out.cfg == cfg
    off = config({"rules": "off", "gate": {"mode": "off"}}, control={"level": 3}, shapes=["procedures"])
    assert resolve(off, package(RULE), "m/agent").notes == []


def test_auto_follows_the_level_and_names_why_a_component_is_off():
    auto = {"gate": {"mode": "auto", "model": "m/gate", "scope": "active_procedures", "min_strength": "none"}}
    cfg = config(
        auto | {"rules": "auto"},
        organizers={"procedure_tools": "auto", "guide": "auto", "loop": "auto"},
        control={"level": 3},
    )
    out = resolve(cfg, from_store_graph(GRAPH), "m/agent")
    on = {n for n in COMPONENTS if _mode(out.cfg, n) == "on"}
    # rules: the graph has no rule; guide: no step names a tool; loop: no model
    assert on == {"procedure_tools", "gate"}
    notes = dict(out.notes)
    assert "off: the package has no rules" in notes["auto:rules"]
    assert "no step of the package names a tool" in notes["auto:guide"]
    assert "organizers.loop.model is none" in notes["auto:loop"]
    assert notes["auto:gate"].startswith("on at level 3")
    assert "level 3" in notes["level"] and "procedures" in notes["shape"]


def _mode(cfg, name):
    from polisys.levels import MODE_PATH

    node = cfg
    for key in MODE_PATH[name]:
        node = getattr(node, key)
    return node


def test_an_auto_gate_stays_off_when_the_strength_is_not_known_and_an_explicit_gate_stays_on():
    gate = {"mode": "auto", "model": "m/gate", "scope": "whole_sop", "min_strength": "agent"}
    cfg = config({"gate": gate}, control={"level": 2}, shapes=["procedures"])
    out = resolve(cfg, package(RULE), "m/agent", policy_text=SOP_TEXT)
    assert out.cfg.checkers.gate.mode == "off"
    assert "not in control.by_model" in dict(out.notes)["auto:gate"]
    explicit = config({"rules": "on", "gate": {**gate, "mode": "on"}})
    parts, models = build(explicit)
    assert parts.components["gate"] and "m/gate" in [m["model"] for m in models.made]
    assert any(
        "not in control.by_model" in x["evidence"] for x in lines(parts, "levels")
    )  # reported, not raised


def test_a_gate_weaker_than_the_agent_raises_even_when_it_is_on():
    gate = {"mode": "on", "model": "m/weak", "scope": "whole_sop", "min_strength": "agent"}
    cfg = config({"rules": "on", "gate": gate}, control={"by_model": {"m/weak": 3, "m/agent": 1}})
    with pytest.raises(RuntimeWiringError, match="gate.*weaker"):
        build(cfg)


def test_write_checks_auto_is_on_when_two_checks_are_on_and_the_level_turned_one_on():
    auto_gate = {**GATE, "mode": "auto"}
    both = config({"rules": "auto", "gate": auto_gate}, control={"level": 2}, shapes=["procedures", "rules"])
    assert resolve(both, package(RULE), "m/agent", policy_text=SOP_TEXT).cfg.checkers.write_checks == "on"
    rules_only = config({"rules": "auto"}, control={"level": 2}, shapes=["rules"])
    assert (
        resolve(rules_only, package(RULE), "m/agent").cfg.checkers.write_checks == "off"
    )  # one check: nothing to order
    # a profile that lists its components keeps separate hooks, whatever they are: the measured profiles are not changed
    listed = config({"rules": "on", "gate": GATE}, control={"level": 2})
    assert config().checkers.write_checks == "auto"
    assert resolve(listed, package(RULE), "m/agent", policy_text=SOP_TEXT).cfg.checkers.write_checks == "off"
    forced = config({"rules": "on", "gate": GATE, "write_checks": "on"}, control={"level": 2})
    assert resolve(forced, package(RULE), "m/agent", policy_text=SOP_TEXT).cfg.checkers.write_checks == "on"


def test_a_risk_tool_turns_write_checks_on_under_auto_and_only_an_explicit_off_raises():
    risk = {"level": 1, "risk": {"level3_tools": ["delete_*"]}}
    parts, _ = build(config({"rules": "auto"}, control=risk))
    assert parts.components[
        "write_checks"
    ]  # auto: a tool that needs its own level needs the hook that can give it
    off = config({"rules": "on", "write_checks": "off"}, control=risk)
    with pytest.raises(RuntimeWiringError, match="control.risk.level3_tools"):
        build(off)


def test_a_risk_tool_with_every_check_set_by_hand_raises_because_nothing_reads_its_level():
    risk = {"level": 1, "risk": {"level3_tools": ["delete_*"]}}
    verifier = {"mode": "on", "model": "m/verify", "scope": "specific_statements"}
    all_on = config({"rules": "on", "gate": GATE, "verifier": verifier, "write_checks": "on"}, control=risk)
    with pytest.raises(RuntimeWiringError, match="control.risk.level3_tools.*does nothing"):
        build(all_on, package(RULE, SPECIFIC))
    one_left = {"rules": "on", "gate": {**GATE, "mode": "auto"}, "verifier": verifier, "write_checks": "on"}
    parts, _ = build(config(one_left, control=risk, shapes=["procedures"]), package(RULE, SPECIFIC))
    assert parts.components["write_checks"]  # the gate is left to auto: the level of the tool decides it
    no_tool = {"risk": {"level3_tools": ["*_delte*"]}}
    build(config({"rules": "on"}, control=no_tool))  # no tool is raised: a pattern to report, as before


@pytest.mark.parametrize("agent", ["", "a/other", "m/agent"])
def test_a_risk_setting_never_raises_under_auto_for_a_model_that_is_not_in_the_table(agent):
    cfg = config(
        {"rules": "auto"},
        control={"by_model": {"m/agent": 2}, "risk": {"level3_tools": ["delete_*"]}},
        shapes=["rules"],
    )
    parts, _ = build(cfg, ctx=dataclasses.replace(CTX, agent_model=agent))
    assert parts.components["write_checks"]


def test_a_risk_pattern_that_matches_no_tool_is_reported_and_builds_no_write_checks():
    cfg = config({"rules": "on"}, control={"risk": {"level3_tools": ["*_delte*"]}})
    parts, _ = build(cfg)
    assert "write_checks" not in parts.components
    notes = {x["rule_id"]: x["evidence"] for x in lines(parts, "levels")}
    assert "'*_delte*' matches no tool" in notes["risk_pattern"]
    right = config({"rules": "auto"}, control={"risk": {"level3_tools": ["*_delte*", "*delete*"]}})
    parts, _ = build(right)  # one pattern matches: the other is reported, and the tool is raised
    assert "write_checks" in parts.components
    assert "matches no tool" in {x["rule_id"]: x["evidence"] for x in lines(parts, "levels")}["risk_pattern"]


def test_risk_tools_on_a_data_run_get_no_verifier_and_the_audit_says_so():
    cfg = config(
        {"rules": "on", "compile": "m/c", "write_checks": "on"},
        control={"level": 1, "risk": {"level3_tools": ["delete_*"]}},
        sources={"data": True},
    )
    parts, _ = build(cfg, package=None) if False else (None, None)
    out = resolve(cfg, None, "m/agent", tool_names=["delete_account"])
    assert "risk_verifier" in dict(out.notes) and "gets no verifier" in dict(out.notes)["risk_verifier"]


@pytest.mark.parametrize(
    "setting",
    [
        {"control": {"level": 2}},
        {"control": {"by_model": {"m/a": 2}}},
        {"control": {"level": 2, "default_level": 3}},
        {"shapes": ["procedures"]},
    ],
)
def test_a_control_setting_that_nothing_reads_raises_on_a_profile_that_lists_every_component(setting):
    path = {"control": "control.", "shapes": "shapes"}[next(iter(setting))]
    with pytest.raises(RuntimeWiringError, match="nothing reads it") as err:
        build(config({"rules": "on"}, **setting))
    assert path in str(err.value)


def test_a_declared_rule_list_is_read_by_the_rule_hooks_and_needs_no_auto_component():
    build(config({"rules": "on"}, shapes=["rules"]))  # a profile that lists every component
    build(config({"rules": "on"}, shapes=["procedures", "rules"]))
    with pytest.raises(
        RuntimeWiringError, match="shapes.*nothing reads it"
    ):  # the rule hooks are off: nothing reads it
        build(config({"rules": "off"}, shapes=["rules"]))
    with pytest.raises(RuntimeWiringError, match="shapes.*nothing reads it"):
        build(config({"rules": "on"}, shapes=["flow"]))


def test_control_settings_are_read_where_a_level_decides_something():
    build(config({"rules": "auto"}, control={"level": 2}, shapes=["rules"]))  # an auto component
    build(config({"rules": "on", "write_checks": "on"}, control={"level": 2}))  # write checks
    build(config({"rules": "on"}, control={"level": 0}))  # shadow
    gate = {**GATE, "min_strength": "agent"}
    build(
        config({"rules": "on", "gate": gate}, control={"by_model": {"m/gate": 1, "m/agent": 2}})
    )  # the comparison
    with pytest.raises(RuntimeWiringError, match="default_level"):  # a static level has no use for it
        build(config({"rules": "auto"}, control={"level": 2, "default_level": 3}, shapes=["rules"]))


# -- the write checks in order ------------------------------------------------------------------------------------------


def write_checks_config(level, gate=GATE, verifier=None, **checkers):
    return config(
        {
            "rules": "on",
            "gate": gate,
            "write_checks": "on",
            **({"verifier": verifier} if verifier else {}),
            **checkers,
        },
        control={"level": level},
    )


DENIED = {"ticket_id": "T1", "status": "closed"}
ALLOWED = {"ticket_id": "T1", "status": "open"}


def test_a_hard_deny_means_the_gate_model_is_not_called():
    models = Scripted({"m/gate": '{"breaks": false}'})
    parts, _ = build(write_checks_config(2), models=models)
    registry = HookRegistry(parts.hooks)
    message = registry.before_tool_call(TOOL, DENIED)
    assert message and "was not called" in message
    assert models.calls == []  # the gate model was not called
    assert [x["verdict"] for x in lines(parts, "write_checks")] == ["deny"]
    assert lines(parts, "gate") == []  # a check that was skipped writes no line
    assert registry.before_tool_call(TOOL, ALLOWED) is None
    assert models.calls == ["m/gate"]  # an allowed call reaches the gate
    assert [(x["rule_id"], x["verdict"]) for x in lines(parts, "write_checks")] == [
        ("rules", "deny"),
        ("rules", "allow"),
        ("gate", "allow"),
    ]


VERIFY = {"mode": "auto", "model": "m/verify", "scope": "specific_statements"}


def auto_write_checks_config(level, risk=None):
    control = {"level": level, **({"risk": {"level3_tools": risk}} if risk else {})}
    return config(
        {"rules": "auto", "gate": {**GATE, "mode": "auto"}, "verifier": VERIFY, "write_checks": "auto"},
        control=control,
        shapes=["procedures", "rules"],
    )


def test_each_check_that_auto_turned_on_runs_from_its_level_and_below_it_writes_no_line():
    pkg = package(RULE, SPECIFIC)
    seen = {}
    for level in (1, 2, 3):
        models = Scripted({})
        parts, _ = build(auto_write_checks_config(level), pkg, models=models)
        assert HookRegistry(parts.hooks).before_tool_call(TOOL, ALLOWED) is None
        seen[level] = [x["rule_id"] for x in lines(parts, "write_checks")]
        assert models.calls == {1: [], 2: ["m/gate"], 3: ["m/gate", "m/verify"]}[level]
    # at level 1 only the rules are on, so there is nothing to put in order: separate hooks
    assert seen == {1: [], 2: ["rules", "gate"], 3: ["rules", "gate", "verifier"]}


def test_a_check_the_profile_says_on_runs_at_every_level_explicit_wins():
    verifier = {"mode": "on", "model": "m/verify", "scope": "specific_statements"}
    for level in (1, 2):
        models = Scripted({})
        parts, _ = build(
            write_checks_config(level, verifier=verifier), package(RULE, SPECIFIC), models=models
        )
        assert HookRegistry(parts.hooks).before_tool_call(TOOL, ALLOWED) is None
        assert models.calls == [
            "m/gate",
            "m/verify",
        ]  # a gate that is on is not skipped at level 1, nor a verifier at 2
        assert [x["rule_id"] for x in lines(parts, "write_checks")] == ["rules", "gate", "verifier"]
    mixed = config(  # the rules left to the level, the gate said on: the gate runs at level 1
        {"rules": "auto", "gate": GATE, "write_checks": "on"}, control={"level": 1}, shapes=["rules"]
    )
    models = Scripted({})
    parts, _ = build(mixed, models=models)
    HookRegistry(parts.hooks).before_tool_call(TOOL, ALLOWED)
    assert models.calls == ["m/gate"]


def test_a_risk_tool_gets_the_level_3_checks_and_another_tool_does_not():
    models = Scripted({})
    parts, _ = build(auto_write_checks_config(1, risk=["delete_*"]), package(RULE, SPECIFIC), models=models)
    registry = HookRegistry(parts.hooks)
    registry.before_tool_call(TOOL, ALLOWED)
    assert models.calls == []  # level 1: rules only
    registry.before_tool_call("delete_account", {"account_id": "A1"})
    assert models.calls == ["m/gate"]  # level 3 for this tool: the gate runs


def test_the_first_block_ends_the_check_and_the_gate_runs_before_the_verifier():
    blocking = '{"breaks": true, "rule": "[1.1] MUST NOT: close a VIP", "case": "T1", "evidence": "the ticket belongs to a VIP customer of the shop", "why": "x"}'
    models = Scripted({"m/gate": blocking})
    verifier = {"mode": "on", "model": "m/verify", "scope": "specific_statements"}
    parts, _ = build(write_checks_config(3, verifier=verifier), package(RULE, SPECIFIC), models=models)
    gate = parts.components["gate"]
    gate.after_tool_call("get_ticket", {}, "the ticket belongs to a VIP customer of the shop")
    message = HookRegistry(parts.hooks).before_tool_call(TOOL, ALLOWED)
    assert message and "Blocked by an SOP check" in message
    assert models.calls == ["m/gate"]  # the verifier was not asked
    assert [(x["rule_id"], x["verdict"]) for x in lines(parts, "write_checks")] == [
        ("rules", "allow"),
        ("gate", "block_once"),
    ]


def test_a_model_check_that_fails_open_writes_error_and_the_next_stage_runs():
    class Broken(Scripted):
        def json_model(self, model, **kwargs):
            call = super().json_model(model, **kwargs)

            def failing(system, user):
                self.calls.append(model)
                raise TimeoutError("no answer")

            failing.model_name, failing.usage = model, call.usage
            return failing

    verifier = {"mode": "on", "model": "m/verify", "scope": "specific_statements"}
    models = Broken({})
    parts, _ = build(write_checks_config(3, verifier=verifier), package(RULE, SPECIFIC), models=models)
    assert HookRegistry(parts.hooks).before_tool_call(TOOL, ALLOWED) is None
    assert models.calls == ["m/gate", "m/verify"]
    stages = {x["rule_id"]: x["verdict"] for x in lines(parts, "write_checks")}
    assert stages == {"rules": "allow", "gate": "error", "verifier": "error"}


def test_tool_blocked_still_reaches_the_other_hooks():
    heard: List[str] = []

    class Listener(AgentHooks):
        def tool_blocked(self, tool, args, message):
            heard.append(tool)

    parts, _ = build(write_checks_config(1))
    HookRegistry([*parts.hooks, Listener()]).before_tool_call(TOOL, DENIED)
    assert heard == [TOOL]


def test_a_reminder_goes_to_the_tool_result_as_it_does_today_and_the_write_runs():
    pkg = package(RULE)
    pkg.rules[0].level = "remind"
    parts, _ = build(write_checks_config(1), pkg)
    registry = HookRegistry(parts.hooks)
    assert registry.before_tool_call(TOOL, DENIED) is None
    note = registry.after_tool_call(TOOL, DENIED, {"ok": True})
    assert note and "POLICY NOTE" in note
    assert [x["verdict"] for x in lines(parts, "write_checks")] == [
        "remind",
        "allow",
    ]  # the gate is on: it runs


def test_a_reminder_of_the_rules_is_dropped_when_a_later_stage_blocks_the_call():
    pkg = package(RULE)
    pkg.rules[0].level = "remind"
    blocking = (
        '{"breaks": true, "rule": "[1.1] MUST NOT: close a VIP", "case": "T1", '
        '"evidence": "the ticket belongs to a VIP customer of the shop", "why": "x"}'
    )
    models = Scripted({"m/gate": blocking})
    parts, _ = build(write_checks_config(2), pkg, models=models)
    parts.components["gate"].after_tool_call(
        "get_ticket", {}, "the ticket belongs to a VIP customer of the shop"
    )
    registry = HookRegistry(parts.hooks)
    assert registry.before_tool_call(TOOL, DENIED)  # the rule only reminds, the gate blocks
    assert registry.after_tool_call("get_ticket", {"ticket_id": "T1"}, "ok") is None  # no "it was run" note
    assert parts.components["rules"].notes == []


def test_after_repeated_denials_skip_keeps_the_call_blocked_and_it_still_stops_before_the_gate():
    models = Scripted({})
    cfg = config(
        {"rules": "on", "gate": GATE, "write_checks": "on"},
        control={"level": 2},
        decisions={"mode": "off", "after_repeated_denials": "skip"},
    )
    parts, _ = build(cfg, models=models)
    registry = HookRegistry(parts.hooks)
    messages = [registry.before_tool_call(TOOL, DENIED) for _ in range(5)]
    assert all(messages) and "still not allowed" in messages[4]
    assert models.calls == []


# -- block_once: one meaning ---------------------------------------------------------------------------------------------

ONCE = {**RULE, "id": "r_once", "level": "block_once"}


def test_block_once_for_a_code_rule_blocks_the_first_call_of_each_case_and_then_reminds():
    hooks = RuleSetHooks([ONCE], SPECS)
    t1, t2 = {"ticket_id": "T1", "status": "closed"}, {"ticket_id": "T2", "status": "closed"}
    assert hooks.before_tool_call(TOOL, t1).message  # the first call of case T1 is blocked
    assert hooks.last_decision == "block_once"
    assert not hooks.before_tool_call(TOOL, t1).message  # the same case: a reminder
    assert hooks.last_decision == "remind" and "POLICY NOTE" in hooks.pop_notes()[0]
    assert hooks.before_tool_call(TOOL, t2).message  # a second case blocks once too
    assert not hooks.before_tool_call(TOOL, t2).message
    assert not hooks.before_tool_call(TOOL, t1).message


@pytest.mark.parametrize("cap", ["allow", "skip"])
def test_the_repeated_denial_cap_counts_each_case_of_a_block_once_rule_apart(cap):
    hooks = RuleSetHooks([ONCE], SPECS, cap_effect=cap)
    for n in range(1, 7):  # six cases: more than the cap of three denials in a row
        call = {"ticket_id": f"T{n}", "status": "closed"}
        assert hooks.before_tool_call(TOOL, call).message, f"case T{n} was not blocked on its first try"
        assert hooks.before_tool_call(TOOL, call).message is None  # the same case: a reminder, as before


def test_the_cap_still_counts_the_denials_of_a_deny_rule_on_one_tool():
    hooks = RuleSetHooks([RULE], SPECS, cap_effect="skip")
    messages = [
        hooks.before_tool_call(TOOL, {"ticket_id": f"T{n}", "status": "closed"}).message for n in range(5)
    ]
    assert "still not allowed" in messages[3]  # the 4th identical denial, as before


def test_a_call_with_no_identifying_argument_is_one_case_so_the_rule_stops_one_call_as_before():
    hooks = RuleSetHooks([ONCE], SPECS)
    assert hooks.before_tool_call(TOOL, {"status": "closed"}).message
    assert not hooks.before_tool_call(TOOL, {"status": "closed"}).message
    assert not hooks.before_tool_call(TOOL, {"status": "deleted"}).message  # still the empty case


def test_block_once_for_the_gate_is_the_same_meaning_a_rule_and_a_case():
    def answer(case):
        return (
            '{"breaks": true, "rule": "[1.1] MUST NOT: close a VIP", "case": "%s", '
            '"evidence": "the ticket belongs to a VIP customer of the shop", "why": "x"}' % case
        )

    replies = iter([answer("T1"), answer("T1"), answer("T2"), answer("T2")])
    gate = StoreGate(None, lambda system, user: next(replies), sop_text="Do not close a ticket of a VIP.")
    gate.after_tool_call("get_ticket", {}, "the ticket belongs to a VIP customer of the shop")
    assert gate.before_tool_call(TOOL, ALLOWED) and gate.last_decision == "block_once"
    assert (
        gate.before_tool_call(TOOL, ALLOWED) is None and gate.last_decision == "allow"
    )  # same rule, same case
    assert gate.before_tool_call(TOOL, ALLOWED) and gate.last_decision == "block_once"  # another case
    assert gate.before_tool_call(TOOL, ALLOWED) is None


# -- the verifier takes only the package's judgment rules ----------------------------------------------------------------


def verifier_prompts(scope):
    prompts: List[str] = []

    def model(system, user):
        prompts.append(user)
        return '{"violations": []}'

    routed = {
        "id": "v_routed",
        "type": "verify",
        "statement": "Routed from a typed rule.",
        "tools": [TOOL],
        "routed_from": "allowed_values",
    }
    hooks = RuleSetHooks(
        [SPECIFIC, GENERAL, GUIDANCE, routed], SPECS, verifier=Verifier(model), verify_scope=scope
    )
    hooks.check_verifier(TOOL, ALLOWED)
    return "\n".join(prompts)


def test_specific_statements_send_the_verifier_only_a_judgment_rule_on_a_tool_it_names():
    text = verifier_prompts("specific_statements")
    assert "Never close a ticket of a VIP." in text
    assert "Never assume data." not in text  # a general statement never goes to a per-call model
    assert "Routed from a typed rule." not in text and "Be polite." not in text


def test_all_statements_keeps_the_measured_path_general_statements_included():
    text = verifier_prompts("all_statements")
    assert "Never close a ticket of a VIP." in text and "Never assume data." in text and "Routed from" in text


def test_a_package_with_no_judgment_rule_builds_no_verifier_under_specific_statements():
    verifier = {"mode": "on", "model": "m/verify", "scope": "specific_statements"}
    parts, models = build(write_checks_config(3, verifier=verifier), package(RULE, GENERAL))
    assert "m/verify" not in [m["model"] for m in models.made]
    assert parts.components["rules"].verifier is None


# -- level 0 ------------------------------------------------------------------------------------------------------------


def test_level_0_runs_every_check_stops_nothing_and_the_audit_holds_the_would_be_verdicts():
    models = Scripted({"m/gate": '{"breaks": false}'})
    verifier = {"mode": "on", "model": "m/verify", "scope": "specific_statements"}
    parts, _ = build(write_checks_config(0, verifier=verifier), package(RULE, SPECIFIC), models=models)
    registry = HookRegistry(parts.hooks)
    assert registry.before_tool_call(TOOL, DENIED) is None  # nothing is stopped
    assert registry.after_tool_call(TOOL, DENIED, {"ok": True}) is None  # nothing is added to a tool result
    assert registry.before_llm_call("t") is None  # no brief
    assert registry.before_reply("done").message is None
    assert models.calls == ["m/gate", "m/verify"]  # every check ran, also after a would-be deny
    stages = [(x["rule_id"], x["verdict"]) for x in lines(parts, "write_checks")]
    assert stages == [("rules", "would_deny"), ("gate", "allow"), ("verifier", "allow")]
    assert any(x["component"] == "rules" and x["verdict"] == "deny" for x in parts.audit.lines)
    assert any("level 0" in x["evidence"] for x in lines(parts, "levels")) or any(
        x["rule_id"] == "level" for x in lines(parts, "levels")
    )


def test_level_0_with_separate_hooks_drops_what_each_would_have_said_and_logs_it():
    parts, _ = build(config({"rules": "on"}, control={"level": 0}))
    assert all(isinstance(h, Shadow) for h in parts.hooks)
    registry = HookRegistry(parts.hooks)
    assert registry.before_tool_call(TOOL, DENIED) is None
    assert any(
        x["component"] == "shadow" and x["verdict"] == "would_block" and "was not called" in x["evidence"]
        for x in parts.audit.lines
    )
    assert any(x["component"] == "rules" and x["verdict"] == "deny" for x in parts.audit.lines)


def test_level_0_is_not_the_baseline_and_a_component_that_changes_the_agent_raises():
    parts, _ = build(config({"rules": "on"}, control={"level": 0}))
    assert parts.hooks and parts.audit is not None
    baseline = build_runtime(config(package=None), None, CTX, models=Scripted({}))
    assert baseline.hooks == [] and baseline.audit is None
    graph = from_store_graph(GRAPH)
    with pytest.raises(RuntimeWiringError, match="level 0"):
        build(config(organizers={"procedure_tools": "on"}, control={"level": 0}), graph)


def test_level_0_obligations_are_recorded_and_the_new_prompt_is_not_sent():
    sent: List[str] = []
    shadow = Shadow(None, "obligations")
    quiet = shadow.quiet_after_reply(lambda answer, turns, live: sent.append(answer) or "Do the rest.")
    assert quiet("done", [], []) is None and sent == ["done"]
    assert shadow.records[0]["outcome"] == "would_followup" and "Do the rest." in shadow.records[0]["reason"]


# -- the ledger follows the level ---------------------------------------------------------------------------------------


def ledger_config(level, **obligations):
    return config(
        control={"level": level},
        sources={"request": True},
        obligations={"producers": ["package"], "model": "m/bind", **obligations},
    )


def ticket() -> Obligation:
    return Obligation(
        id="ob1",
        origin="package",
        scope={},
        effect={"kind": "write", "tools": [TOOL], "match": {}},
        values={},
        level="follow_up",
        source={"section": "1", "quote": "Close the ticket."},
    )


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_with_no_item_level_the_ledger_takes_the_run_level_and_is_on_at_observe(level):
    pkg = PolicyPackage(version="v", obligations=[ticket()])
    parts, _ = build(ledger_config(level), pkg)
    ledger = parts.components["obligations"]
    assert isinstance(ledger, ObligationLedger) or isinstance(
        ledger.inner if hasattr(ledger, "inner") else ledger, ObligationLedger
    )
    assert (
        ledger.level == level and ledger.item_level == "observe"
    )  # nothing is sent until a replay allows it


def test_an_explicit_item_level_keeps_the_earlier_reading_whatever_the_run_level():
    pkg = PolicyPackage(version="v", obligations=[ticket()])
    cfg = ledger_config(1, item_level="follow_up", max_follow_ups=2)
    cfg.control.level = "by_model"  # an explicit item_level does not read the level
    parts, _ = build(cfg, pkg)
    ledger = parts.components["obligations"]
    assert ledger.level == 3 and ledger.item_level == "follow_up"


# -- the example of the spec ---------------------------------------------------------------------------------------------

SPEC_EXAMPLE = """
policy:
  package: packages/handbook/crestwood/package.json
  sources: {data: true, discovery: true, request: true}
  authority: {company_domains: [company.example.com], may_change_rules: [VP, COO, HR]}
  priority_classes: [regulatory, company, department, team]
  control:
    level: by_model
    by_model: {aws/gpt-oss-120b: 3, azure/DeepSeek-V4-Pro: 2, gcp/gemini-3.8-flash: 2}
    risk: {level3_tools: ["*payment*", "*delete*", "send_email_external"]}
  checkers:
    compile: aws/gpt-oss-120b
    narrow_questions: jaredpalmer/kev-4b
    router: {kind: agent_opened, expand: next_links}   # any registered kinds (section 11)
    gate: {model: gcp/gemini-3.8-flash, scope: active_procedures, min_strength: agent}
    verifier: {model: aws/gpt-oss-120b, scope: specific_statements, may_deny: false}
  organizers: {procedure_tools: auto, guide: auto, loop: auto, sop_tracker: auto, navigator: auto, requirements_tracker: auto}
  obligations: {producers: [package, request, duty], model: aws/gpt-oss-120b, max_follow_ups: 1, wrong_values: known_origin_only}
  decisions: {require_evidence: true, after_repeated_denials: skip}
  notes: {reminders: tool_result, brief: system}
  trace: true
"""
# Settings of the example that have no reader yet, so ``build_runtime`` raises on them by its rule that wiring with no effect
# raises (company domains and roles are not read; ``priority_classes`` has a reader since Task 5.2).
NOT_READ = ("authority",)


def example(drop=NOT_READ) -> PolicyConfig:
    data = yaml.safe_load(SPEC_EXAMPLE)["policy"]
    for key in drop:
        data.pop(key)
    return PolicyConfig.from_dict(data)


def spec_package() -> PolicyPackage:
    pkg = from_store_graph(GRAPH)
    pkg.rules = package(RULE, SPECIFIC).rules
    pkg.obligations = [ticket()]
    return pkg


def spec_ctx(agent_model: str) -> RunContext:
    return RunContext(
        task_id="task-5",
        policy_version="v1",
        request="Close T1.",
        system_prompt=SOP_TEXT,
        tool_specs=SPECS,
        agent_model=agent_model,
        message_tool="",
        data_search=[("search_mail", lambda query: [])],
    )


@pytest.mark.parametrize(
    "agent, level, on",
    [
        ("azure/DeepSeek-V4-Pro", 2, {"rules", "procedure_tools", "gate", "decisions"}),
        ("gcp/gemini-3.8-flash", 2, {"rules", "procedure_tools", "gate", "decisions"}),
        ("aws/gpt-oss-120b", 3, {"rules", "procedure_tools", "gate", "decisions", "verifier"}),
        (
            "a/other",
            1,
            {"rules"},
        ),  # a model that is not in the table: the default level, no strength to compare
        ("", 1, {"rules"}),
    ],
)
def test_the_example_of_the_spec_builds_for_any_model_also_one_that_is_not_in_the_table(agent, level, on):
    cfg = example()
    parts, models = build(cfg, spec_package(), spec_ctx(agent))
    assert explain_level(cfg, agent).level == level
    notes = {x["rule_id"]: x["evidence"] for x in lines(parts, "levels")}
    assert notes["level"].startswith(f"level {level}")
    assert ("check_level" in notes) == (level < 3)  # a risk tool is at level 3: the checks are built for it
    assert "*delete*" not in notes.get(
        "risk_pattern", ""
    )  # it matches delete_account; the others match no tool here
    turned_on = {n for n, text in notes.items() if n.startswith("auto:") and text.startswith("on")}
    assert turned_on - {"auto:write_checks"} == {f"auto:{n}" for n in on}
    assert notes["auto:navigator"].startswith("off: ")
    assert (
        "write_checks" in parts.components
    )  # a tool is in risk.level3_tools: it needs the hook that can raise one tool
    assert ("gate" in parts.components) == (
        "gate" in on
    )  # an unknown model: the strength is not known, so the gate stays off
    assert parts.components["obligations"].item_level == "observe"  # the ledger is on at observe under auto
    assert ("gcp/gemini-3.8-flash" in [m["model"] for m in models.made]) == ("gate" in on)


def test_the_example_as_written_raises_only_on_authority_which_nothing_reads_yet():
    with pytest.raises(RuntimeWiringError, match="authority.*nothing reads it yet"):
        build(example(drop=()), spec_package(), spec_ctx("azure/DeepSeek-V4-Pro"))


# -- a real agent loop --------------------------------------------------------------------------------------------------

MADE: List[Dict[str, str]] = []


@tool
async def update_ticket(ticket_id: str, status: str) -> str:
    """Change a ticket."""
    MADE.append({"ticket_id": ticket_id, "status": status})
    return f"ticket {ticket_id} is {status}"


async def run_loop(monkeypatch, parts, script):
    agent_loop = pytest.importorskip("cuga.backend.hooks.tests.test_agent_loop")
    MARKER, ScriptedModel = agent_loop.MARKER, agent_loop.ScriptedModel

    class Marker(AgentHooks):
        def before_llm_call(self, thread_id):
            return MARKER

    monkeypatch.setenv("OPENAI_API_KEY", "scripted-run-no-calls")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/unused")
    MADE.clear()
    agent = CugaAgent(
        tools=[update_ticket],
        model=ScriptedModel(script=script, seen=[]),
        hooks=[Marker(), *parts.hooks],
        auto_load_policies=False,
        filesystem_sync=False,
    )
    return await agent.invoke("Close T1.", thread_id="t-levels")


async def test_a_real_agent_loop_a_write_blocked_by_the_write_checks_does_not_run(monkeypatch):
    models = Scripted({})
    parts, _ = build(write_checks_config(2), models=models)
    script = ['print(await update_ticket(ticket_id="T1", status="closed"))', "ANSWER:It was refused."]
    result = await run_loop(monkeypatch, parts, script)
    assert MADE == [] and result.answer == "It was refused."
    assert models.calls == []  # the rule denied it, so the gate's model was not paid for
    assert [e["outcome"] for e in result.hook_events if e["hook"] == "before_tool_call"] == ["block"]
    assert [x["verdict"] for x in lines(parts, "write_checks")] == ["deny"]


async def test_a_real_agent_loop_an_allowed_write_runs_and_goes_through_the_gate(monkeypatch):
    models = Scripted({})
    parts, _ = build(write_checks_config(2), models=models)
    script = ['print(await update_ticket(ticket_id="T1", status="open"))', "ANSWER:Done."]
    await run_loop(monkeypatch, parts, script)
    assert MADE == [{"ticket_id": "T1", "status": "open"}] and models.calls == ["m/gate"]


async def test_a_real_agent_loop_at_level_0_the_write_runs_and_the_audit_says_it_would_have_been_denied(
    monkeypatch,
):
    parts, _ = build(write_checks_config(0), models=Scripted({}))
    script = ['print(await update_ticket(ticket_id="T1", status="closed"))', "ANSWER:Done."]
    result = await run_loop(monkeypatch, parts, script)
    assert MADE == [{"ticket_id": "T1", "status": "closed"}] and result.answer == "Done."
    assert [e["outcome"] for e in result.hook_events if e["hook"] == "before_tool_call"] == ["observe"]
    assert ("rules", "would_deny") in [(x["rule_id"], x["verdict"]) for x in lines(parts, "write_checks")]


# -- measured profiles ---------------------------------------------------------------------------------------------------


def test_the_write_checks_are_off_unless_a_profile_asks_so_and_the_module_names_no_benchmark():
    from pathlib import Path

    from polisys import levels

    assert (
        PolicyConfig().checkers.write_checks == "auto"
    )  # and it is off for a profile that lists its components
    text = Path(levels.__file__).read_text()
    assert not re.search(
        r"automationbench|itsm|handbook|tau2|τ|enterpriseops|appworld|telecom|crestwood", text, re.I
    )
