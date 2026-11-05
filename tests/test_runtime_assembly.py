"""Runtime assembly: one place wires the policy layer, and wiring that can have no effect raises."""

import asyncio
import ast
import dataclasses
import inspect
import re
from pathlib import Path

import pytest

from polisys.hooks import AgentHooks
from polisys.dry_run import DryRunModelCall, RecordingModels, describe_parts
from polisys.package import from_build, from_store_graph
from polisys.policy_config import PolicyConfig
from polisys.procedure_store import POLICY_TOOLS_NOTE
from polisys.runtime import (
    COMPONENTS,
    Assembly,
    Component,
    Models,
    Piece,
    RunContext,
    RuntimeWiringError,
    build_runtime,
)
from polisys.sop_navigator import HandoffGuard, SopNavigator
from polisys.store_gate import StoreGate

from .test_from_package import _navigator_build
from .test_sop_navigator import POLICY as NAVIGATOR_POLICY
from .test_sop_navigator import TURNS, fake_ask

TOOL = "update_ticket"
SPECS = [
    {"name": TOOL, "description": "Change a ticket.", "inputSchema": {"type": "object", "properties": {}}}
]
RULE = {
    "id": "r1",
    "type": "allowed_values",
    "tool": TOOL,
    "arg": "status",
    "values": ["open"],
    "quote": "Only open.",
}
VERIFY = {
    "id": "v1",
    "type": "verify",
    "statement": "Do not do it.",
    "tools": [TOOL],
    "quote": "Do not do it.",
}
BUILD = {"rules": [RULE, VERIFY], "procedures": []}
STEP_BUILD = {
    "rules": [RULE],
    "procedures": [
        {"id": "p1", "name": "Tickets", "steps": [{"action": TOOL, "quote": "Update the ticket."}]}
    ],
}
GRAPH = {
    "id": "sop",
    "escalation": {"name": "Escalate", "steps": ["Tell the lead."], "must_not": ["Do not reply."]},
    "procedures": [
        {
            "id": "1.1",
            "name": "Tickets",
            "steps": ["Open a ticket."],
            "must_not": ["Do not open two tickets."],
            "links": [],
        }
    ],
}
NO_RULES_GRAPH = {
    "id": "sop",
    "procedures": [{"id": "1.1", "name": "Tickets", "steps": ["Open a ticket."], "links": []}],
}
SOP_TEXT = "# Tickets\nOpen a ticket.\n"
CTX = RunContext(
    task_id="task-7", policy_version="v-abc", request="Close it.", system_prompt=SOP_TEXT, tool_specs=SPECS
)
OFF = {
    "rules": "off",
    "gate": {"mode": "off"},
    "verifier": {"mode": "off"},
}
ORGANIZERS_OFF = {
    "procedure_tools": "off",
    "guide": "off",
    "loop": "off",
    "sop_tracker": "off",
    "navigator": "off",
    "requirements_tracker": "off",
}


def config(checkers=None, organizers=None, **top):
    """A config with every component off, except what the arguments turn on."""
    return PolicyConfig.from_dict(
        {
            "package": top.pop("package", None),
            "decisions": top.pop("decisions", {"mode": "off"}),
            "checkers": {**OFF, **(checkers or {})},
            "organizers": {**ORGANIZERS_OFF, **(organizers or {})},
            **top,
        }
    )


def build(cfg, pkg=None, ctx=CTX):
    models = RecordingModels()
    return build_runtime(cfg, pkg, ctx, models=models), models


RULES_ON = {"rules": "on", "verifier": {"mode": "on", "model": "m/v", "scope": "all_statements"}}
GATE_ON = {"gate": {"mode": "on", "model": "m/g", "min_strength": "none"}}


# -- the baseline ------------------------------------------------------------------------------------------------------


def test_every_component_off_adds_nothing_to_the_run():
    parts, models = build(config())
    assert parts.hooks == [] and parts.tools == [] and parts.system_note == ""
    assert (parts.run, parts.turn_note, parts.wrap_tools, parts.after_reply) == (None, None, None, None)
    assert parts.usage() == {} and parts.records() == [] and parts.components == {}
    assert models.made == []


# -- each raise names the task, the policy version and the component ---------------------------------------------------


def assert_raises(cfg, pkg, component, words, ctx=CTX):
    with pytest.raises(RuntimeWiringError) as err:
        build(cfg, pkg, ctx)
    text = str(err.value)
    assert ctx.task_id in text and ctx.policy_version in text and component in text, text
    assert words in text, text


def test_a_package_path_with_a_placeholder_and_no_policy_version_raises():
    cfg = config(RULES_ON, package="packages/x/{policy}/builds/package.json")
    ctx = dataclasses.replace(CTX, policy_version="")
    with pytest.raises(RuntimeWiringError, match="no policy_version"):
        build(cfg, None, ctx)
    # a fixed path needs none: the package itself is the answer
    assert build(config(RULES_ON, package="p.json"), from_build(BUILD), ctx)[0].hooks


def test_a_profile_that_needs_a_package_with_none_available_raises_with_task_and_version():
    cfg = config(RULES_ON, package="packages/x/{policy}/builds/package.json")
    assert_raises(cfg, None, "package", "none is available")


def test_a_component_that_needs_a_package_raises_when_the_profile_names_none():
    assert_raises(config(organizers={"guide": "on"}), None, "guide", "needs a policy package")


def test_a_package_that_the_profile_does_not_name_raises():
    assert_raises(config(), from_build(BUILD), "package", "says no package")


def test_a_package_that_no_component_reads_raises():
    assert_raises(config(package="p.json"), from_build(BUILD), "package", "no component that is on reads it")


def test_guide_on_a_graph_package_raises_because_no_step_names_a_tool():
    cfg = config(organizers={"guide": "on"}, package="p.json")
    assert_raises(cfg, from_store_graph(GRAPH), "guide", "no tool to stop on")


def test_guide_on_a_build_package_with_tools_in_its_steps_builds():
    parts, _ = build(config(organizers={"guide": "on"}, package="p.json"), from_build(STEP_BUILD))
    assert type(parts.hooks[0]).__name__ == "ProcedureGuide"


def test_gate_with_no_checkable_items_and_no_sop_text_names_the_context():
    cfg = config(GATE_ON, {"procedure_tools": "on"}, package="p.json")
    assert_raises(
        cfg, from_store_graph(NO_RULES_GRAPH), "gate", "no must-not rules, decisions or escalation must-nots"
    )


def test_gate_over_a_policy_text_with_no_text_names_the_context():
    cfg = config({"gate": {"mode": "on", "model": "m/g", "scope": "whole_sop", "min_strength": "none"}})
    empty = dataclasses.replace(CTX, system_prompt="")
    assert_raises(cfg, None, "gate", "or sop_text", ctx=empty)


def test_gate_on_with_no_model_raises_instead_of_using_the_agents_model():
    cfg = config(
        {"gate": {"mode": "on", "model": None, "min_strength": "none"}},
        {"procedure_tools": "on"},
        package="p.json",
    )
    assert_raises(cfg, from_store_graph(GRAPH), "gate", "its model is none")


@pytest.mark.parametrize(
    "organizers, words",
    [
        ({"loop": {"mode": "on", "model": None}}, "loop"),
        ({"sop_tracker": {"mode": "on", "model": None}}, "sop_tracker"),
        ({"requirements_tracker": {"mode": "on", "model": None}}, "requirements_tracker"),
    ],
)
def test_any_component_on_with_no_model_raises(organizers, words):
    top = {"sources": {"request": True}} if "requirements_tracker" in organizers else {}
    assert_raises(config(organizers=organizers, **top), None, words, "its model is none")


def test_gate_weaker_than_the_agent_raises_and_an_unknown_model_does_not():
    gate = {"mode": "on", "model": "m/g", "min_strength": "agent"}
    cfg = config(
        {"gate": gate},
        {"procedure_tools": "on"},
        package="p.json",
        control={"by_model": {"m/g": 3, "m/a": 2}},
    )
    ctx = dataclasses.replace(CTX, agent_model="m/a")
    with pytest.raises(RuntimeWiringError, match="gate.*weaker"):
        build(cfg, from_store_graph(GRAPH), ctx)
    unknown = config(
        {"gate": gate}, {"procedure_tools": "on"}, package="p.json"
    )  # no table: the strength is not known
    assert build(unknown, from_store_graph(GRAPH))[0].components["gate"]


def test_gate_on_active_procedures_with_procedure_tools_off_raises_because_nothing_opens_one():
    cfg = config(GATE_ON, package="p.json")
    assert_raises(cfg, from_store_graph(GRAPH), "gate", "nothing else opens a procedure")


def navigator_config(**navigator):
    defaults = {"mode": "on", "handoff_text": ["transferring you"], "handoff_tools": []}
    return config({"narrow_questions": "kev"}, {"navigator": {**defaults, **navigator}}, package="p.json")


def navigator_package():
    return from_build(_navigator_build())


NAV_CTX = RunContext(
    task_id="task-7",
    policy_version="v-abc",
    system_prompt=NAVIGATOR_POLICY,
    tool_specs=[
        {"name": n}
        for n in ("check_messages", "grant_permission", "enable_roaming", "reset_network", "transfer")
    ],
    message_tool="send_message",
)


def test_navigator_with_empty_hand_off_lists_raises():
    assert_raises(
        navigator_config(handoff_text=[], handoff_tools=[]),
        navigator_package(),
        "navigator",
        "never fire",
        NAV_CTX,
    )


def test_navigator_with_a_hand_off_tool_that_is_not_in_the_run_raises():
    cfg = navigator_config(handoff_text=[], handoff_tools=["not_in_this_run"])
    assert_raises(cfg, navigator_package(), "navigator", "not_in_this_run", NAV_CTX)


def test_navigator_with_a_misspelled_hand_off_tool_beside_a_phrase_raises():
    cfg = navigator_config(handoff_text=["transferring you"], handoff_tools=["transfer", "tranfser"])
    assert_raises(cfg, navigator_package(), "navigator", "tranfser", NAV_CTX)


def test_navigator_hand_off_tool_is_checked_against_the_tools_of_this_run():
    cfg = navigator_config(handoff_tools=["transfer"])
    ctx = dataclasses.replace(NAV_CTX, tools_in_run=["check_messages", "enable_roaming", "grant_permission"])
    assert_raises(cfg, navigator_package(), "navigator", "not in this run", ctx)


def test_navigator_that_is_not_told_the_message_tool_raises():
    ctx = dataclasses.replace(NAV_CTX, message_tool=None)
    assert_raises(navigator_config(), navigator_package(), "navigator", "message_tool", ctx)


def test_navigator_on_a_benchmark_with_no_message_tool_builds():
    ctx = dataclasses.replace(NAV_CTX, message_tool="")
    parts, _ = build(navigator_config(), navigator_package(), ctx)
    assert parts.components["navigator"].message_tool == ""


def test_navigator_that_may_block_zero_times_raises():
    assert_raises(
        navigator_config(max_blocks=0), navigator_package(), "navigator", "can never block", NAV_CTX
    )


def test_navigator_with_no_narrow_question_model_raises():
    cfg = config(organizers={"navigator": {"mode": "on", "handoff_text": ["x"]}}, package="p.json")
    assert_raises(cfg, navigator_package(), "navigator", "narrow_questions", NAV_CTX)


def test_navigator_on_a_package_with_no_chapter_to_check_raises():
    ctx = dataclasses.replace(NAV_CTX, tool_specs=[{"name": "zzz_nothing_like_it"}])
    assert_raises(navigator_config(), navigator_package(), "navigator", "no chapter", ctx)


def test_rules_on_a_package_with_no_rules_raises():
    assert_raises(config(RULES_ON, package="p.json"), from_store_graph(GRAPH), "rules", "no rules")


def test_rules_on_with_no_tool_specs_raises():
    ctx = dataclasses.replace(CTX, tool_specs=[])
    assert_raises(config(RULES_ON, package="p.json"), from_build(BUILD), "rules", "no tool specs", ctx)


def test_discovery_with_no_search_in_the_task_raises():
    cfg = config(
        {
            "rules": "on",
            "compile": "m/i",
            "verifier": {"mode": "on", "model": "m/i", "scope": "all_statements"},
        },
        sources={"data": True, "discovery": True},
    )
    assert_raises(cfg, None, "intake", "no search")


def test_intake_with_a_verifier_model_other_than_its_model_raises():
    cfg = config(
        {
            "rules": "on",
            "compile": "m/i",
            "verifier": {"mode": "on", "model": "m/other", "scope": "all_statements"},
        },
        sources={"data": True},
    )
    assert_raises(cfg, None, "intake", "it has one model")


def test_the_brief_as_a_message_needs_a_host_that_says_message():
    assert_raises(config(notes={"brief": "message"}), None, "notes.brief", "host_note_placement")
    assert build(config(notes={"brief": "message"}), ctx=RunContext(host_note_placement="message"))[0].hooks == []


def test_auto_is_resolved_by_the_level_switch():
    parts, _ = build(PolicyConfig())  # no package, no data: nothing for any component to do
    assert parts.hooks == [] and parts.components == {}


@pytest.mark.parametrize(
    "top, path",
    [
        ({"priority_classes": ["regulatory"]}, "priority_classes"),
        ({"authority": {"company_domains": ["a.example"]}}, "authority.company_domains"),
        ({"authority": {"may_change_rules": ["VP"]}}, "authority.may_change_rules"),
        ({"obligations": {"max_follow_ups": 2}}, "obligations.max_follow_ups"),
        ({"obligations": {"wrong_values": "never"}}, "obligations.wrong_values"),
        ({"obligations": {"item_level": "show"}}, "obligations.item_level"),
    ],
)
def test_a_setting_that_nothing_reads_yet_raises(top, path):
    assert_raises(config(**top), None, path, "")


@pytest.mark.parametrize(
    "checkers, top, path",
    [
        ({"router": {"kind": "bm25"}}, {}, "checkers.router.kind"),
        ({"router": {"expand": "pagerank"}}, {}, "checkers.router.expand"),
        ({"compile": "m"}, {}, "checkers.compile"),
        ({"detector": "m"}, {}, "checkers.detector"),
        ({"guard_code": "remind"}, {}, "checkers.guard_code"),
        ({"read_before_write": "remind"}, {}, "checkers.read_before_write"),
        ({"narrow_questions": "m"}, {}, "checkers.narrow_questions"),
        ({"method_choice": "m.json"}, {}, "checkers.method_choice"),
        ({"verifier": {"mode": "on", "model": "m", "scope": "all_statements"}}, {}, "checkers.verifier.mode"),
        (
            {"verifier": {"mode": "on", "model": "m", "scope": "specific_statements"}},
            {},
            "checkers.verifier.mode",
        ),
        ({}, {"sources": {"discovery": True}}, "sources.discovery"),
        ({}, {"sources": {"request": True}}, "sources.request"),
        ({}, {"authority": {"supersedes": True}}, "authority.supersedes"),
        ({}, {"decisions": {"mode": "on"}}, "decisions.mode"),
        ({}, {"decisions": {"mode": "off", "require_evidence": True}}, "decisions.require_evidence"),
        ({"gate": {"mode": "off", "seed_request": True}}, {}, "checkers.gate.seed_request"),
        ({"rules": "on"}, {}, "checkers.rules"),
    ],
)
def test_a_setting_whose_only_reader_is_off_raises(checkers, top, path):
    assert_raises(config(checkers, **top), None, path, "")


def test_a_component_that_cannot_build_raises_with_the_context():
    cfg = config(RULES_ON, package="p.json")
    cfg.checkers.method_choice = "/nonexistent/choice.json"
    assert_raises(cfg, from_build(BUILD), "rules", "FileNotFoundError")


def test_two_components_in_one_slot_raise():
    COMPONENTS["extra_run"] = Component(lambda cfg: True, lambda a: Piece(run=lambda *x: None))
    COMPONENTS["other_run"] = Component(lambda cfg: True, lambda a: Piece(run=lambda *x: None))
    try:
        assert_raises(config(), None, "other_run", "slot run")
    finally:
        del COMPONENTS["extra_run"], COMPONENTS["other_run"]


# -- one shared store --------------------------------------------------------------------------------------------------


def test_gate_and_policy_tools_share_one_store_and_the_description_says_so():
    cfg = config(GATE_ON, {"procedure_tools": "on"}, package="p.json")
    parts, models = build(cfg, from_store_graph(GRAPH))
    gate = parts.hooks[0]
    assert isinstance(gate, StoreGate) and gate.store is parts.components["procedure_tools"]
    assert parts.system_note == POLICY_TOOLS_NOTE and [t.name for t in parts.tools][:2] == [
        "policy__list_procedures",
        "policy__get_procedure",
    ]
    assert describe_parts(parts, models)["gate_reads_tools_store"] == {"StoreGate": True}
    next(t for t in parts.tools if t.name == "policy__get_procedure").invoke({"ids": ["1.1"]})
    assert "[1.1] MUST NOT: Do not open two tickets." in gate._rules()


def test_the_description_shows_a_store_that_is_not_shared(monkeypatch):
    from polisys.procedure_store import ProcedureStore

    monkeypatch.setattr(Assembly, "store", lambda self, component: ProcedureStore.from_package(self.pkg))
    cfg = config(GATE_ON, {"procedure_tools": "on"}, package="p.json")
    models = RecordingModels()
    parts = build_runtime(cfg, from_store_graph(GRAPH), CTX, models=models)
    assert describe_parts(parts, models)["gate_reads_tools_store"] == {"StoreGate": False}


def test_describing_a_gate_leaves_the_store_as_it_was():
    parts, models = build(
        config(GATE_ON, {"procedure_tools": "on"}, package="p.json"), from_store_graph(GRAPH)
    )
    describe_parts(parts, models)
    assert parts.hooks[0].store.calls == []


def test_the_store_verifier_is_gone():
    import importlib

    with pytest.raises(ImportError):
        importlib.import_module("polisys.store_verifier")
    assert "store_verifier" not in COMPONENTS
    import polisys.runtime as runtime

    assert not hasattr(runtime, "STORE_VERIFIER_MODEL") and not hasattr(runtime, "_build_store_verifier")


# -- runner differences as parameters ----------------------------------------------------------------------------------


def test_seed_request_puts_the_request_in_the_data_of_the_gate_and_its_default_does_not():
    gate = {"mode": "on", "model": "m/g", "scope": "whole_sop", "min_strength": "none"}
    plain, _ = build(config({"gate": gate}))
    seeded, _ = build(config({"gate": {**gate, "seed_request": True}}, sources={"request": True}))
    assert plain.hooks[0].reads == []
    assert seeded.hooks[0].reads == ["user request: Close it."]


def test_a_seeded_gate_reads_the_request_so_sources_request_must_say_so():
    gate = {"mode": "on", "model": "m/g", "scope": "whole_sop", "min_strength": "none", "seed_request": True}
    assert_raises(config({"gate": gate}), None, "sources.request", "is False, and the components")
    plain = {**gate, "seed_request": False}
    assert_raises(
        config({"gate": plain}, sources={"request": True}),
        None,
        "sources.request",
        "is True, and the components",
    )


def test_seed_request_with_no_request_raises():
    gate = {"mode": "on", "model": "m/g", "scope": "whole_sop", "min_strength": "none", "seed_request": True}
    cfg = config({"gate": gate}, sources={"request": True})
    assert_raises(cfg, None, "gate", "no request", dataclasses.replace(CTX, request=""))


def test_gate_max_tokens_comes_from_the_config_and_timeout_stays_measured():
    gate = {"mode": "on", "model": "m/g", "scope": "whole_sop", "min_strength": "none"}
    _, default = build(config({"gate": gate}))
    _, raised = build(config({"gate": {**gate, "max_tokens": 16000}}))
    assert [(m["timeout"], m["max_tokens"]) for m in default.made] == [(180, 4000)]
    assert [(m["timeout"], m["max_tokens"]) for m in raised.made] == [(180, 16000)]


def test_the_gate_text_is_the_policy_text_when_the_task_has_one_and_else_the_system_prompt():
    gate = {"mode": "on", "model": "m/g", "scope": "whole_sop", "min_strength": "none"}
    assert build(config({"gate": gate}))[0].hooks[0].sop_text == SOP_TEXT
    ctx = dataclasses.replace(CTX, policy_text="# SOP file")
    assert build(config({"gate": gate}), ctx=ctx)[0].hooks[0].sop_text == "# SOP file"


def test_the_tracker_reads_the_system_prompt_only_when_the_config_says_so():
    tracker = {"mode": "on", "model": "m/t", "wrong_items": "verify"}
    plain, _ = build(config(organizers={"requirements_tracker": tracker}, sources={"request": True}))
    told, _ = build(
        config(
            organizers={"requirements_tracker": {**tracker, "read_system_prompt": True}},
            sources={"request": True},
        )
    )
    assert plain.hooks[0].policy == "" and told.hooks[0].policy == SOP_TEXT
    assert plain.hooks[0].wrong_mode == "verify"


def test_loop_styles_reach_the_loop_and_a_discovery_loop_needs_a_package():
    loop = {"mode": "on", "model": "m/l", "style": "discovery"}
    assert_raises(config(organizers={"loop": loop}), None, "loop", "needs a policy package")
    parts, models = build(
        config({"narrow_questions": "kev"}, {"loop": loop}, package="p.json"), from_build(STEP_BUILD)
    )
    loop_object = parts.components["loop"]
    assert loop_object.mode == "discovery" and loop_object.index is not None
    assert {m["kind"] for m in models.made} == {"json", "questions"}


def test_reminder_and_brief_settings_reach_every_hook_that_has_them():
    cfg = config(
        RULES_ON,
        package="p.json",
        notes={"reminders": "system", "brief": "off"},
    )
    parts, _ = build(cfg, from_build(BUILD))
    (rules,) = parts.hooks
    assert (rules.note_channel, rules.brief) == ("system", False)


def test_the_procedure_monitor_comes_with_the_rules_when_the_package_has_procedures():
    with_procedures, _ = build(config(RULES_ON, package="p.json"), from_build(STEP_BUILD))
    without, _ = build(config(RULES_ON, package="p.json"), from_build(BUILD))
    assert [type(h).__name__ for h in with_procedures.hooks] == ["RuleSetHooks", "ProcedureMonitor"]
    assert [type(h).__name__ for h in without.hooks] == ["RuleSetHooks"]


def test_the_rule_hooks_get_a_verifier_only_when_a_rule_needs_it():
    no_verify = {"rules": [RULE], "procedures": []}
    parts, models = build(config(RULES_ON, package="p.json"), from_build(no_verify))
    assert parts.hooks[0].verifier is None and models.made == []
    parts, models = build(config(RULES_ON, package="p.json"), from_build(BUILD))
    assert parts.hooks[0].verifier is not None and [m["model"] for m in models.made] == ["m/v"]


def test_a_method_choice_gives_verifiers_by_model_through_the_factory(tmp_path):
    path = tmp_path / "choice.json"
    path.write_text('{"verify": {"checker": "verifier"}}')
    cfg = config({**RULES_ON, "method_choice": str(path)}, package="p.json")
    parts, models = build(cfg, from_build(BUILD))
    hooks = parts.hooks[0]
    assert hooks.method_choice == {"verify": {"checker": "verifier"}}
    hooks.verifier_for("m/other")
    hooks.verifier_for("openrouter:m/router")
    assert hooks.verifier_for("m/other") is hooks.verifier_for("m/other")
    assert [(m["kind"], m["model"]) for m in models.made] == [
        ("json", "m/v"),
        ("json", "m/other"),
        ("openrouter_json", "m/router"),
    ]
    assert set(parts.usage()) == {"verifier:m/v", "verifier:m/other", "verifier:m/router"}


# -- adding a method is one entry --------------------------------------------------------------------------------------


class DemoHook(AgentHooks):
    records = [{"demo": 1}]


def test_a_component_registered_in_components_reaches_the_run_with_no_other_change():
    def build_demo(a):
        return Piece(
            hooks=[DemoHook()],
            tools=[],
            system_note="Demo note.",
            turn_note=lambda turns: "note",
            target=None,
        )

    COMPONENTS["demo"] = Component(lambda cfg: cfg.trace, build_demo)
    try:
        parts, _ = build(config())
    finally:
        del COMPONENTS["demo"]
    assert [type(h) for h in parts.hooks] == [DemoHook]
    assert parts.system_note == "Demo note." and parts.turn_note([]) == "note"
    assert isinstance(parts.components["demo"], DemoHook)
    assert parts.records() == [{"demo": 1}]
    assert "demo" not in COMPONENTS


# -- the slots ---------------------------------------------------------------------------------------------------------


def test_the_loop_slot_runs_the_loop_with_the_recorder_that_is_also_a_hook(monkeypatch):
    from polisys.sop_loop import LoopHooks, SopLoop

    seen = {}

    async def fake_run(self, agent, request, thread_id, rec, timeout):
        seen.update(agent=agent, request=request, thread=thread_id, rec=rec, timeout=timeout)
        return "result"

    monkeypatch.setattr(SopLoop, "run", fake_run)
    parts, _ = build(config(organizers={"loop": {"mode": "on", "model": "m/l"}}))
    assert asyncio.run(parts.run("agent", "req", "t", 5.0)) == "result"
    assert seen["rec"] is parts.hooks[0] and isinstance(seen["rec"], LoopHooks)
    assert seen["timeout"] == 5.0


def test_the_tracker_slot_is_a_note_for_the_turn_with_no_hook():
    parts, _ = build(config(organizers={"sop_tracker": {"mode": "on", "model": "m/t"}}))
    assert parts.hooks == [] and parts.turn_note is not None
    assert parts.components["sop_tracker"].sections[0]["title"] == "Tickets"


def test_the_tracker_on_a_text_with_no_sections_raises():
    cfg = config(organizers={"sop_tracker": {"mode": "on", "model": "m/t"}})
    assert_raises(
        cfg, None, "sop_tracker", "no sections", dataclasses.replace(CTX, system_prompt="no heading at all")
    )


def navigator_parts(**navigator):
    return build(navigator_config(**navigator), navigator_package(), NAV_CTX)


def test_the_navigator_slots_have_no_hook_and_share_one_guard():
    parts, models = navigator_parts()
    assert parts.hooks == [] and parts.wrap_tools.__self__ is parts.after_reply.__self__
    assert [m["kind"] for m in models.made] == ["questions"]
    description = describe_parts(parts, models)["slots"]
    assert description["wrap_tools"]["handoff_text"] == ["transferring you"]
    assert description["wrap_tools"]["message_tool"] == "send_message"
    assert description["wrap_tools"]["navigator"]["class"] == "SopNavigator"


class ScriptedNavigator:
    """Finds every hand-off premature, and records what it was shown."""

    records = []

    def __init__(self, note="Do the open steps first."):
        self.note, self.seen = note, []

    def check_handoff(self, turns, call_args=None):
        self.seen.append((list(turns), call_args))
        return self.note


def async_tool(name, calls):
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel

    class Args(BaseModel):
        message: str = ""

    async def run(message=""):
        calls.append(name)
        return f"{name} ran"

    return StructuredTool(name=name, description=name, args_schema=Args, coroutine=run)


async def test_a_hand_off_tool_is_blocked_up_to_max_blocks_then_runs():
    calls, turns, live = [], [{"role": "customer", "text": "help"}], []
    navigator = ScriptedNavigator()
    guard = HandoffGuard(navigator, handoff_tools=["transfer"], max_blocks=2)
    transfer = guard.wrap_tools([async_tool("transfer", calls)], turns, live)[0]
    assert await transfer.ainvoke({"message": "x"}) == navigator.note
    assert await transfer.ainvoke({"message": "x"}) == navigator.note
    assert await transfer.ainvoke({"message": "x"}) == "transfer ran"
    assert calls == ["transfer"] and guard.blocks == 2
    assert navigator.seen[0][0] == turns


async def test_a_fine_hand_off_runs_and_other_tools_are_recorded_for_the_turn():
    calls, turns, live = [], [], []
    guard = HandoffGuard(
        ScriptedNavigator(note=None), handoff_tools=["transfer"], message_tool="send_message"
    )
    transfer, read, send = guard.wrap_tools(
        [async_tool("transfer", calls), async_tool("read", calls), async_tool("send_message", calls)],
        turns,
        live,
    )
    assert await transfer.ainvoke({"message": "x"}) == "transfer ran"
    await read.ainvoke({"message": "q"})
    await send.ainvoke({"message": "How is it?"})
    assert guard.blocks == 0
    assert [t["role"] for t in live] == ["tool", "tool", "agent", "customer"]
    assert live[2]["text"] == "How is it?" and live[3]["text"] == "send_message ran"


class AskModels(RecordingModels):
    """A real navigator with a scripted decision model, which keeps the state of each question."""

    def __init__(self):
        super().__init__()
        self.states = []

    def questions(self, *args, **kwargs):
        recorded = super().questions(*args, **kwargs)

        def ask(state, questions):
            self.states.append(state)
            return fake_ask()(state, questions)

        ask.usage, ask.model_name = recorded.usage, recorded.model_name
        return ask


@pytest.mark.parametrize("message_tool, reply_reaches_the_navigator", [("send_message", True), ("", False)])
async def test_the_message_tool_from_the_context_decides_what_the_navigator_reads(
    message_tool, reply_reaches_the_navigator
):
    """The customer's reply to a mid-turn message counts as a customer message only under its right tool name: under a
    wrong one it is a tool entry, and the question whether the problem is solved reads the wrong messages."""
    models = AskModels()
    ctx = dataclasses.replace(NAV_CTX, message_tool=message_tool)
    parts = build_runtime(navigator_config(), navigator_package(), ctx, models=models)
    turns, live, calls = [{"role": "customer", "text": "My picture messages do not send."}], [], []
    send = parts.wrap_tools([async_tool("send_message", calls)], turns, live)[0]

    await send.ainvoke({"message": "Is it fixed now?"})
    parts.after_reply("I am transferring you now.", turns, live)

    solved_state = next(s for s in models.states if s.startswith("Last customer messages"))
    assert ("send_message ran" in solved_state) is reply_reaches_the_navigator


def test_tools_with_no_coroutine_pass_as_they_are():
    plain = type("T", (), {"coroutine": None, "name": "x"})()
    assert HandoffGuard(ScriptedNavigator()).wrap_tools([plain], [], []) == [plain]


def test_a_reply_that_hands_off_too_early_becomes_a_new_prompt_up_to_max_blocks():
    navigator = ScriptedNavigator()
    guard = HandoffGuard(navigator, handoff_text=["Transferring you"], max_blocks=1)
    turns = [{"role": "customer", "text": "help"}]
    live = [{"role": "agent", "text": "mid"}, {"role": "tool", "name": "t"}]
    assert guard.after_reply("Hello there", turns, live) is None
    prompt = guard.after_reply("I am transferring you now", turns, live)
    assert (
        prompt.startswith("[Policy note — your hand-off message was not sent]") and navigator.note in prompt
    )
    assert [t["role"] for t in turns] == [
        "customer",
        "agent",
    ]  # the mid-turn message joins; tool calls are there already
    assert guard.after_reply("I am transferring you now", turns, live) is None  # the cap


def test_a_real_navigator_behind_the_guard_blocks_a_premature_hand_off():
    nav = SopNavigator(
        NAVIGATOR_POLICY, _navigator_build(), ["enable_roaming"], fake_ask(), min_chapter_actions=1
    )
    guard = HandoffGuard(nav, handoff_text=["transfer"], max_blocks=1)
    prompt = guard.after_reply("I will transfer you", list(TURNS), [])
    assert prompt and "toggle_roaming" in prompt and guard.records == nav.records and guard.blocks == 1


# -- the description ---------------------------------------------------------------------------------------------------


def test_the_description_is_the_same_for_two_builds_and_differs_when_a_rule_changes():
    cfg = config(RULES_ON, package="p.json")
    one, models_one = build(cfg, from_build(BUILD))
    two, models_two = build(cfg, from_build(BUILD))
    first = describe_parts(one, models_one)
    assert first == describe_parts(two, models_two)
    assert first["models"][0]["model"] == "m/v"
    long_rules = [{**RULE, "id": f"r{i}", "quote": "Only open, and nothing else at all."} for i in range(20)]
    changed = [*long_rules, {**RULE, "id": "r99", "values": ["closed"]}]
    a, ma = build(cfg, from_build({"rules": long_rules + [{**RULE, "id": "r99"}], "procedures": []}))
    b, mb = build(cfg, from_build({"rules": changed, "procedures": []}))
    described_a, described_b = describe_parts(a, ma), describe_parts(b, mb)
    assert described_a["hooks"][0]["attrs"]["rules"]["items"] == 21
    assert (
        described_a["hooks"][0]["attrs"]["rules"]["sha"] != described_b["hooks"][0]["attrs"]["rules"]["sha"]
    )


def test_a_rule_from_a_package_is_described_like_the_same_rule_from_a_build():
    from polisys.dry_run import describe_wiring
    from polisys.ruleset import RuleSetHooks

    parts, models = build(
        config({"rules": "on"}, package="p.json"), from_build({"rules": [RULE], "procedures": []})
    )
    new = describe_parts(parts, models)["hooks"][0]["attrs"]
    old = describe_wiring([RuleSetHooks([RULE], SPECS)], models=RecordingModels())["hooks"][0]["attrs"]
    assert new == old
    assert parts.hooks[0].rules[0]["level"]  # the package's level is there; the description leaves it out


def test_a_slot_with_a_plain_function_still_shows_as_used():
    from polisys.dry_run import slot_target

    fn = lambda turns: None  # noqa: E731
    assert slot_target(fn) is fn and slot_target(None) is None


# -- models ------------------------------------------------------------------------------------------------------------


def test_recording_models_raise_when_called_and_record_the_defaults_of_the_real_functions():
    models = RecordingModels()
    model = models.json_model("m/x", timeout=5)
    with pytest.raises(DryRunModelCall):
        model("s", "u")
    assert models.made == [
        {
            "kind": "json",
            "model": "m/x",
            "timeout": 5,
            "base_url": None,
            "api_key_env": "OPENAI_API_KEY",
            "max_retries": 3,
            "max_tokens": None,
        }
    ]


def test_recording_models_find_a_model_behind_a_trace_wrapper():
    from polisys.trace import PolicyTrace

    models = RecordingModels()
    model = models.json_model("m/x")
    assert models.find(PolicyTrace().wrap_model(model, "c"))["model"] == "m/x"


def test_recording_models_keep_no_key():
    models = RecordingModels()
    models.questions("kev", api_key="secret")
    assert "api_key" not in models.made[0]


def test_the_real_factory_needs_the_key_of_the_decision_model(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Models().questions("kev")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Models().detector("kev")
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-used")
    assert Models().questions("kev").model_name == "kev"
    assert Models().detector("kev").model_name == "kev"


def test_build_runtime_with_no_factory_uses_the_real_one(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    gate = {"mode": "on", "model": "m/g", "scope": "whole_sop", "min_strength": "none"}
    parts = build_runtime(config({"gate": gate}), None, CTX)
    assert parts.hooks[0].model.model_name == "m/g"
    assert parts.usage() == {
        "gate": {"model": "m/g", "calls": 0, "input_tokens": 0, "output_tokens": 0},
    }


# -- no switch read ----------------------------------------------------------------------------------------------------


LAYER = Path(inspect.getfile(build_runtime)).parent
MODULES = sorted(p.relative_to(LAYER).as_posix() for p in LAYER.rglob("*.py") if "tests" not in p.parts)
# Every read of the environment in the layer: (module, function) -> the names it may read. Only secrets and the address of
# the gateway come from the environment. The audit reads the values of every variable with a secret-looking name (names: none),
# to keep them out of its lines. ``openai_json_model`` also reads the key whose name its caller gives (``api_key_env``), which is
# no upper-case name in the statement.
ALLOWED_READS = {
    ("trace.py", "AuditLog.__init__"): set(),
    ("detectors.py", "systemone_detector"): {"OPENROUTER_API_KEY"},
    ("detectors.py", "systemone_questions"): {"OPENROUTER_API_KEY"},
    ("runtime.py", "Models._need_key"): {"OPENROUTER_API_KEY"},
    ("llm.py", "openai_json_model"): {"OPENAI_BASE_URL"},
}


def environment_reads(source):
    """(enclosing function, upper-case names in the statement) of each mention of ``os.environ`` or ``os.getenv``, in any
    form: a call, a subscript, ``in``, an alias, or ``from os import environ``."""

    def names_in(node):
        return {
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and re.fullmatch(r"[A-Z][A-Z0-9_]*", n.value)
        }

    out = []

    def walk(node, scope, statement):
        for child in ast.iter_child_nodes(node):
            is_scope = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            inner = [*scope, child.name] if is_scope else scope
            here = child if isinstance(child, ast.stmt) else statement
            is_attribute = (
                isinstance(child, ast.Attribute)
                and child.attr in ("environ", "getenv")
                and isinstance(child.value, ast.Name)
                and child.value.id == "os"
            )
            is_import = (
                isinstance(child, ast.ImportFrom)
                and child.module == "os"
                and {a.name for a in child.names} & {"environ", "getenv"}
            )
            if is_attribute or is_import:
                out.append((".".join(scope), names_in(here)))
            walk(child, inner, here)

    walk(ast.parse(source), [], None)
    return out


@pytest.mark.parametrize(
    "source",
    [
        'import os\nos.environ.get("SOME_SWITCH")',
        'import os\nos.environ["SOME_SWITCH"]',
        'import os\n"SOME_SWITCH" in os.environ',
        'import os\nos.getenv("SOME_SWITCH")',
        "import os\nenv = os.environ",
        'from os import environ\nenviron["SOME_SWITCH"]',
        'import os\ndef f():\n    return dict(os.environ).get("SOME_SWITCH")',
    ],
)
def test_the_scan_finds_every_form_of_a_read(source):
    assert environment_reads(source)


def test_the_scan_gives_the_function_and_the_name():
    source = 'import os\nclass C:\n    def m(self):\n        return "SOME_SWITCH" in os.environ'
    assert environment_reads(source) == [("C.m", {"SOME_SWITCH"})]


@pytest.mark.parametrize("name", MODULES)
def test_the_layer_reads_no_policy_switch(name):
    wrong = []
    for function, names in environment_reads((LAYER / name).read_text()):
        key = (name, function)
        if key not in ALLOWED_READS or names - ALLOWED_READS[key]:
            wrong.append((function, sorted(names)))
    assert wrong == []


def test_the_scan_covers_the_modules_that_read_the_environment_and_the_allowed_list_has_no_stale_entry():
    assert {"llm.py", "detectors.py", "package.py", "sop_retrieval.py", "runtime.py"} <= set(MODULES)
    found = {
        (name, function) for name in MODULES for function, _ in environment_reads((LAYER / name).read_text())
    }
    assert found == set(ALLOWED_READS)


@pytest.mark.parametrize("name", ["runtime.py", "dry_run.py"])
def test_the_assembly_names_no_benchmark_and_imports_nothing_from_the_eval_repo(name):
    text = (Path(inspect.getfile(build_runtime)).parent / name).read_text().lower()
    for word in ("automationbench", "handbook", "itsm", "tau2", "enterpriseops", "telecom", "policy_eval"):
        assert word not in text


def test_a_model_client_with_no_name_raises_and_the_environment_gives_no_name(monkeypatch):
    from polisys.llm import openai_json_model

    monkeypatch.setenv("MODEL_NAME", "m/from-the-environment")
    monkeypatch.setenv("POLICY_MODEL_NAME", "m/from-the-environment")
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    for name in (None, ""):
        with pytest.raises(ValueError, match="needs a model name"):
            openai_json_model(name)
    assert openai_json_model("m/named").model_name == "m/named"
