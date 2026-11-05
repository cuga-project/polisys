"""A check that fails open has a line in the audit, intake verdicts name their rule, and the decisions of intake's own rule
set are lines too. In every test the run does what it did before: the audit only says more."""

import json

import pytest

from polisys.hooks import AgentHooks, HookRegistry, HookResult
from polisys import sop_navigator, store_gate
from polisys.decisions import Decisions
from polisys.deliverables import DeliverablesHooks
from polisys.dry_run import RecordingModels
from polisys.intake import OnlinePolicyIntake
from polisys.levels import Shadow, WriteChecks
from polisys.package import from_build
from polisys.procedure_monitor import ProcedureMonitor
from polisys.ruleset import RuleSetHooks
from polisys.sop_planner import SopPlanner
from polisys.runtime import Assembly, Models, build_runtime
from polisys.sop_navigator import SopNavigator
from polisys.store_gate import StoreGate
from polisys.tracker import RequirementsTracker
from polisys.trace import AuditLog, PolicyTrace, audited_failure, hook_failure
from polisys.verifier import Verifier

from .test_audit import RULE, Scripted, ctx, lines_of
from .test_procedure_guide import guide
from .test_runtime_assembly import SPECS, TOOL, config
from .test_sop_navigator import BUILD, POLICY, TURNS

SEND = ("gmail_send_email", {"body": "x"})
STATEMENT = "Never send it."
POLICY_ITEM = {"title": "Send rules", "trust": "trusted", "statements": [STATEMENT]}


def boom(system, user):
    raise RuntimeError("gateway down")


def attached(component, hook):
    audit = AuditLog(task="t", package_version="v")
    audit.attach(component, hook)
    return audit


def intake(model, **kwargs):
    kwargs.setdefault("read_before_write", False)
    kwargs.setdefault("check_reply", False)
    hooks = OnlinePolicyIntake(model, **kwargs)
    hooks.on_user_message("send it", None)
    hooks.policies = [POLICY_ITEM]
    return hooks


def verdict_model(*violations):
    return lambda system, user: json.dumps({"violations": list(violations)})


# -- 1. a check that fails open has a line ------------------------------------------------------------------------------


def test_an_intake_whose_verify_model_raises_has_a_line_and_the_write_still_runs():
    hooks = intake(boom)
    audit = attached("intake", hooks)
    assert HookRegistry([hooks]).before_tool_call(*SEND) is None  # the registry swallows it: the write runs
    [line] = audit.lines
    assert (line["verdict"], line["rule_id"], line["hook"], line["tool"]) == (
        "error",
        "hook_failed",
        "before_tool_call",
        SEND[0],
    )
    assert "gateway down" in line["evidence"] and "without this check" in line["evidence"]
    with pytest.raises(RuntimeError, match="gateway down"):  # and it is raised again, as before
        hooks.before_tool_call(*SEND)


def test_an_intake_whose_reply_check_raises_has_a_line_and_the_reply_is_not_checked():
    hooks = intake(boom, check_reply=True)
    audit = attached("intake", hooks)
    assert HookRegistry([hooks]).before_reply("done").message is None
    [line] = audit.lines
    assert (line["verdict"], line["rule_id"], line["hook"]) == ("error", "hook_failed", "before_reply")


# -- every hook, not only the ones that decide: the registry and the write checks report what they swallow ------------------


def down(*args, **kwargs):
    raise RuntimeError("hook down")


def failed_lines(audit):
    return [(x["component"], x["hook"], x["tool"], x["rule_id"], x["verdict"]) for x in audit.lines]


class Other(AgentHooks):
    """The hook after the broken one: what it says reaches the agent as before."""

    def on_user_message(self, message, thread_id):
        return HookResult(details=[{"other": message}])

    def after_tool_call(self, tool, args, result):
        return "a note of the other hook"

    def before_reply(self, answer):
        return "a follow-up of the other hook"


def hooks_with_no_decorator():
    package = from_build({"rules": [], "procedures": []})
    return {
        "intake": OnlinePolicyIntake(lambda s, u: "{}", tool_specs=SPECS, read_before_write=False),
        "gate": StoreGate(None, lambda s, u: "{}", sop_text="Never send."),
        "procedure_monitor": ProcedureMonitor(package.procedures),
        "deliverables": DeliverablesHooks(lambda s, u: "{}", SPECS),
        "requirements_tracker": RequirementsTracker(lambda s, u: "{}"),
        "sop_tracker": SopPlanner([], lambda s, u: "{}"),
    }


@pytest.mark.parametrize(
    "component",
    ["intake", "gate", "procedure_monitor", "deliverables", "requirements_tracker", "sop_tracker"],
)
def test_a_hook_with_no_decorator_that_raises_is_one_line_of_its_component(component):
    hooks = hooks_with_no_decorator()[component]
    audit = attached(component, hooks)
    hooks.after_tool_call = down
    registry = HookRegistry([hooks, Other()])
    assert (
        registry.after_tool_call("get_it", {"id": 1}, "x") == "a note of the other hook"
    )  # the run is the same
    assert failed_lines(audit) == [(component, "after_tool_call", "get_it", "hook_failed", "error")]
    assert "hook down" in audit.lines[0]["evidence"]


def test_the_intake_learning_nothing_from_a_read_is_a_line_and_no_policy_found():
    hooks = OnlinePolicyIntake(lambda s, u: "{}", tool_specs=SPECS, read_before_write=False)
    audit = attached("intake", hooks)
    hooks.enforcer.after_tool_call = down  # the real hook raises from inside
    assert HookRegistry([hooks]).after_tool_call("get_it", {}, "x") is None
    assert failed_lines(audit) == [("intake", "after_tool_call", "get_it", "hook_failed", "error")]
    hooks.enforcer.on_user_message = down
    assert HookRegistry([hooks]).on_user_message("hello", None).message is None
    assert failed_lines(audit)[-1] == ("intake", "on_user_message", "", "hook_failed", "error")


def test_the_gate_reading_nothing_is_an_error_line_and_never_an_allow():
    gate = StoreGate(None, lambda s, u: "{}", sop_text="Never send.")
    audit = attached("gate", gate)
    gate.reads = None  # the real hook raises from inside
    assert HookRegistry([gate]).after_tool_call("get_it", {}, "x") is None
    assert failed_lines(audit) == [("gate", "after_tool_call", "get_it", "hook_failed", "error")]


def test_a_hook_that_decides_is_one_line_through_the_registry_and_not_two():
    hooks = intake(boom)
    audit = attached("intake", hooks)
    assert HookRegistry([hooks]).before_tool_call(*SEND) is None
    assert failed_lines(audit) == [("intake", "before_tool_call", SEND[0], "hook_failed", "error")]


def test_a_child_of_the_write_checks_is_a_line_of_its_own_component_and_a_decided_failure_is_one_line():
    rules, gate = RuleSetHooks([RULE], SPECS), StoreGate(None, lambda s, u: "{}", sop_text="Never send.")
    audit = attached("rules", rules)
    audit.attach("gate", gate)
    check = WriteChecks(lambda tool: 1, rules=rules, gate=gate)
    audit.attach("write_checks", check)
    gate.after_tool_call = down
    assert (
        check.after_tool_call("get_it", {}, "x").message is None
    )  # as before: the other child ran and the run goes on
    assert failed_lines(audit) == [("gate", "after_tool_call", "get_it", "hook_failed", "error")]

    @audited_failure
    def before_reply(self, answer):  # a hook that decides reports it and raises
        raise RuntimeError("hook down")

    rules.before_reply = lambda answer: before_reply(rules, answer)
    check.before_reply("done")
    assert [x["hook"] for x in audit.lines].count("before_reply") == 1


def test_a_child_with_no_audit_is_a_line_of_the_write_checks():
    rules = RuleSetHooks([RULE], SPECS)
    check = WriteChecks(lambda tool: 1, rules=rules)
    audit = attached("write_checks", check)
    rules.on_user_message = down
    check.on_user_message("hello", None)
    assert failed_lines(audit) == [("write_checks", "on_user_message", "", "hook_failed", "error")]


def test_a_hook_under_level_0_is_a_line_and_stays_quiet():
    inner = RequirementsTracker(lambda s, u: "{}")
    wrapper = Shadow(inner, "requirements_tracker")
    audit = attached("shadow", wrapper)
    inner.after_tool_call = down
    assert HookRegistry([wrapper]).after_tool_call("get_it", {}, "x") is None
    assert failed_lines(audit) == [("shadow", "after_tool_call", "get_it", "hook_failed", "error")]


def test_a_tool_blocked_observer_that_raises_is_a_line_and_the_block_stands():
    class Stop(AgentHooks):
        def before_tool_call(self, tool, args):
            return "stopped"

    hooks = RequirementsTracker(lambda s, u: "{}")
    audit = attached("requirements_tracker", hooks)
    hooks.tool_blocked = down
    assert HookRegistry([Stop(), hooks]).before_tool_call("send", {}) == "stopped"
    assert failed_lines(audit) == [("requirements_tracker", "tool_blocked", "send", "hook_failed", "error")]


def test_a_hook_with_no_audit_writes_nothing_and_a_broken_audit_changes_nothing():
    hooks = RequirementsTracker(lambda s, u: "{}")
    hooks.after_tool_call = down
    assert HookRegistry([hooks, Other()]).after_tool_call("get_it", {}, "x") == "a note of the other hook"
    hooks._audit_sink = down
    hooks._report_failure = down
    assert HookRegistry([hooks, Other()]).after_tool_call("get_it", {}, "x") == "a note of the other hook"
    hook_failure(hooks, "after_tool_call", ("get_it",), RuntimeError("x"))  # raises nothing


def test_what_the_agent_sees_is_the_same_with_and_without_the_audit():
    def run(with_audit):
        hooks = RequirementsTracker(lambda s, u: "{}")
        if with_audit:
            attached("requirements_tracker", hooks)
        hooks.on_user_message = hooks.after_tool_call = hooks.before_reply = down
        registry = HookRegistry([hooks, Other()])
        return (
            registry.on_user_message("hello", None),
            registry.after_tool_call("get_it", {}, "x"),
            registry.before_reply("done"),
            hooks.records,
        )

    assert run(True) == run(False)


def test_a_rule_set_whose_scope_check_raises_has_a_line_and_the_write_still_runs():
    def scope(rule, action):
        raise RuntimeError("kev down")

    hooks = RuleSetHooks([RULE], SPECS, scope_check=scope)
    audit = attached("rules", hooks)
    assert HookRegistry([hooks]).before_tool_call(TOOL, {"status": "closed"}) is None
    [failed] = [x for x in audit.lines if x["verdict"] == "error"]
    assert (failed["rule_id"], failed["tool"]) == ("hook_failed", TOOL) and "kev down" in failed["evidence"]


def test_a_failed_scope_check_inside_intake_is_a_line_of_the_intake():
    def ask(state, questions):
        raise RuntimeError("kev down")

    hooks = OnlinePolicyIntake(
        lambda s, u: "{}",
        tool_specs=SPECS,
        verify_writes=True,
        read_before_write=False,
        check_reply=False,
        decisions=Decisions(ask),
    )
    hooks.enforcer.add_rules([RULE])
    audit = attached("intake", hooks)
    assert HookRegistry([hooks]).before_tool_call(TOOL, {"status": "closed"}) is None
    assert [(x["rule_id"], x["verdict"]) for x in audit.lines] == [("hook_failed", "error")]


def no_chapter_navigator():
    return SopNavigator(
        POLICY,
        BUILD,
        min_chapter_actions=1,
        agent_tools=[],
        ask=lambda state, questions: {k: None for k in questions},
    )


async def test_a_blocked_hand_off_tool_and_the_one_let_through_name_the_tool():
    from polisys.sop_navigator import HandoffGuard

    from .test_runtime_assembly import async_tool
    from .test_sop_navigator import fake_ask

    nav = SopNavigator(POLICY, BUILD, min_chapter_actions=1, agent_tools=[], ask=fake_ask())
    guard = HandoffGuard(nav, handoff_tools=["transfer"], max_blocks=1)
    audit = attached("navigator", nav)
    audit.attach("navigator", guard)
    transfer = guard.wrap_tools([async_tool("transfer", [])], [dict(t) for t in TURNS], [])[0]
    assert "Before handing this case off" in await transfer.ainvoke({"message": "x"})
    assert await transfer.ainvoke({"message": "x"}) == "transfer ran"  # the limit: it goes through
    assert [(x["verdict"], x["hook"], x["tool"]) for x in audit.lines] == [
        ("block", "before_tool_call", "transfer"),
        ("allow_after_cap", "before_tool_call", "transfer"),
    ]


def test_a_hand_off_with_no_chapter_goes_through_and_the_audit_says_it_was_not_checked():
    nav = no_chapter_navigator()
    audit = attached("navigator", nav)
    assert nav.check_handoff(TURNS) is None  # the hand-off goes through
    assert nav.records == []  # no verdict record, as before
    [line] = audit.lines
    assert (line["verdict"], line["rule_id"], line["checker"]) == ("allow", "no_chapter", "")
    assert "not checked" in line["evidence"]
    nav.check_handoff(TURNS)
    assert len(audit.lines) == 2  # asked again on each hand-off, so a line each time


def test_a_guide_whose_evidence_model_raises_has_a_line_and_the_write_goes_on():
    g = guide(boom)
    audit = attached("guide", g)
    g.after_tool_call("read_inbox", {}, {"from": "vp@x", "text": "Navarro termination"})
    args = {"summary": "OFFBOARD-Navarro-04102026"}
    assert g.before_tool_call("jira__create_issue", args).message  # the gate stop
    assert g.before_tool_call("jira__create_issue", args).message is None  # the check fails: no stop
    [failed] = [x for x in audit.lines if x["verdict"] == "error"]
    assert (failed["rule_id"], failed["tool"], failed["hook"]) == (
        "procedure_evidence",
        "jira__create_issue",
        "before_tool_call",
    )
    assert "gateway down" in failed["evidence"] and "skipped" in failed["evidence"]


def test_a_rule_set_whose_verifier_model_raises_says_error_and_not_allow_and_the_write_runs():
    verify = {"id": "v1", "type": "verify", "statement": STATEMENT, "tools": [TOOL], "source": "§9"}
    hooks = RuleSetHooks([verify], SPECS, verifier=Verifier(boom))
    audit = attached("rules", hooks)
    assert hooks.before_tool_call(TOOL, {"status": "open"}).message is None  # as before: the write runs
    [line] = audit.lines
    assert line["verdict"] == "error" and "verifier_error: RuntimeError" in line["evidence"]
    assert hooks.records[-1]["outcome"] == "allow"  # the record for the run is what it was


LOOP_PROCEDURE = {
    "id": "p1",
    "name": "Closing",
    "steps": [{"action": TOOL, "quote": "Update the ticket."}],
    "preconditions": [
        {
            "condition": "written authorization from the lead",
            "if_not": "stop",
            "quote": "Do not close a ticket without written authorization from the lead.",
        }
    ],
}


class FakeAgent:
    async def invoke(self, message, thread_id=None):
        from types import SimpleNamespace

        return SimpleNamespace(answer="done", receipt=None, error=None)


def run_loop(answers):
    """The loop through ``build_runtime``, with a model that answers by the question it is asked."""

    def model(system, user):
        return answers(user)

    cfg = config(organizers={"loop": {"mode": "on", "model": "m/loop"}}, package="p.json")
    pkg = from_build({"version": "v9", "rules": [RULE], "procedures": [LOOP_PROCEDURE]})
    parts = build_runtime(cfg, pkg, ctx(), models=Scripted(answers={"m/loop": model}))
    import asyncio

    asyncio.run(parts.run(FakeAgent(), "Close the ticket of Lee.", "th", 30))
    return parts, parts.components["loop"]


CASES = {"cases": [{"case": "Lee", "summary": "close the ticket", "keys": ["Lee"]}]}


def test_a_loop_question_that_fails_makes_the_case_line_say_error_and_the_case_stays_follow():
    def answers(user):
        if "Condition from the SOP" in user:
            raise RuntimeError("gateway down")
        return json.dumps({**CASES, "procedure": 1, "answer": "no"})

    parts, loop = run_loop(answers)
    [case] = [e for e in loop.log if e["phase"] == "case"]
    assert case["decision"] == "follow"  # what the run does: the same as before
    [line] = [x for x in lines_of(parts, "loop") if x["rule_id"] == "Closing"]
    assert line["verdict"] == "error" and "gateway down" in line["evidence"] and "follow" in line["evidence"]
    assert any(x["rule_id"] == "model_call" and x["verdict"] == "error" for x in lines_of(parts, "loop"))


def test_a_loop_plan_question_that_fails_is_a_line_and_no_case_is_checked():
    def answers(user):
        raise RuntimeError("gateway down")

    parts, loop = run_loop(answers)
    assert [e["phase"] for e in loop.log] == ["plan"]
    plan = [x for x in lines_of(parts, "loop") if x["rule_id"] == "questions"]
    assert [x["verdict"] for x in plan] == ["error"] and "no case was checked" in plan[0]["evidence"]


def test_a_loop_decision_with_no_procedure_names_the_general_rules_and_a_plain_follow_names_none():
    audit = AuditLog()
    case = {
        "phase": "case",
        "case": "Lee",
        "procedure": None,
        "reasons": ["SOP rule applies: if x then hold"],
    }
    audit.record("loop", None, {**case, "decision": "exception"})
    audit.record("loop", None, {**case, "decision": "follow", "reasons": []})
    assert [(x["rule_id"], x["verdict"]) for x in audit.lines] == [
        ("general_rules", "exception"),
        ("", "follow"),
    ]


def test_a_loop_whose_questions_all_work_has_no_error_in_its_lines():
    def answers(user):
        return json.dumps({**CASES, "procedure": 1, "answer": "no", "evidence": "none"})

    parts, loop = run_loop(answers)
    assert "errors" not in loop.log[0] and all(x["verdict"] != "error" for x in lines_of(parts, "loop"))


# -- the three wrappers report a failed call ---------------------------------------------------------------------------


def traced():
    audit = AuditLog(task="t")
    return PolicyTrace(audit=audit), audit


def test_a_model_that_raises_is_an_error_line_and_is_counted_as_an_error():
    trace, audit = traced()
    with pytest.raises(RuntimeError):
        trace.wrap_model(boom, "guide")("S", "U")
    [line] = audit.lines
    assert (line["component"], line["rule_id"], line["checker"], line["verdict"]) == (
        "guide",
        "model_call",
        "model",
        "error",
    )
    assert "gateway down" in line["evidence"]
    assert list(trace.summary().values())[0]["errors"] == 1


def test_a_decision_model_that_raises_is_an_error_line():
    trace, audit = traced()
    with pytest.raises(RuntimeError):
        trace.wrap_ask(lambda state, questions: boom(state, questions), "decisions")("s", {"excluded": "?"})
    [line] = audit.lines
    assert (line["component"], line["rule_id"], line["checker"], line["verdict"]) == (
        "decisions",
        "kev_call",
        "kev",
        "error",
    )
    assert line["evidence"].startswith("excluded: ")


def test_a_detector_that_raises_is_an_error_line():
    trace, audit = traced()
    with pytest.raises(RuntimeError):
        trace.wrap_detector(lambda request, texts: boom(request, texts))("r", ["a"])
    [line] = audit.lines
    assert (line["component"], line["rule_id"], line["checker"], line["verdict"]) == (
        "detector",
        "detector_call",
        "kev",
        "error",
    )


def test_a_decision_model_client_that_gives_up_is_an_error_line_and_its_answer_passes_unchanged():
    """The Kev clients count a failed call in ``usage["errors"]`` and answer with no value: nothing raises."""
    trace, audit = traced()

    def ask(state, questions):
        ask.usage["errors"] += 1
        return {name: None for name in questions}

    ask.usage, ask.model_name = {"calls": 0, "errors": 0, "cost": 0.0}, "kev"
    assert trace.wrap_ask(ask, "navigator")("s", {"solved": "?"}) == {"solved": None}
    [line] = audit.lines
    assert (line["component"], line["verdict"]) == ("navigator", "error") and "no answer" in line["evidence"]
    trace.wrap_ask(lambda state, questions: {"a": 0.1}, "navigator")("s", {"a": "?"})
    assert len(audit.lines) == 1  # a call that worked adds nothing


def test_a_gate_model_that_raises_in_a_run_is_two_lines_and_the_write_runs():
    gate = {"gate": {"mode": "on", "model": "m/g", "min_strength": "none", "scope": "whole_sop"}}
    parts = build_runtime(config(gate), None, ctx(), models=Scripted(answers={"m/g": boom}))
    assert parts.hooks[0].before_tool_call(TOOL, {"status": "open"}) is None
    assert sorted((x["rule_id"], x["verdict"]) for x in lines_of(parts, "gate")) == [
        ("", "error"),  # the gate's own: it did not check
        ("model_call", "error"),  # the call
    ]
    assert [e.get("error") is not None for e in parts.trace.events] == [True]


# -- 2. intake verdicts name their rule ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, verdict",
    [
        ({}, "remind"),
        ({"model_effect": "block_once"}, "block_once"),
        ({"verifier_may_deny": True}, "deny"),
    ],
)
def test_an_intake_verdict_names_the_statement_it_is_about_and_who_decided(kwargs, verdict):
    hooks = intake(verdict_model({"statement": 1, "reason": "it is sent", "fix": "do not"}), **kwargs)
    audit = attached("intake", hooks)
    hooks.before_tool_call(*SEND)
    [line] = audit.lines
    assert line["verdict"] == verdict
    assert (line["rule_id"], line["source"], line["checker"], line["class"]) == (
        STATEMENT,
        "Send rules",
        "model",
        "judgment",
    )
    assert line["evidence"] == "it is sent" and line["hook"] == "before_tool_call" and line["tool"] == SEND[0]


def test_a_write_that_broke_two_statements_is_two_lines():
    hooks = intake(
        verdict_model({"statement": 1, "reason": "a"}, {"statement": 2, "reason": "b"}),
        verifier_may_deny=True,
    )
    hooks.policies = [{**POLICY_ITEM, "statements": [STATEMENT, "Never send it twice."]}]
    audit = attached("intake", hooks)
    hooks.before_tool_call(*SEND)
    assert [(x["rule_id"], x["evidence"]) for x in audit.lines] == [
        (STATEMENT, "a"),
        ("Never send it twice.", "b"),
    ]


def test_a_gap_at_the_reply_names_its_statement():
    reply = json.dumps({"missing": [{"statement": 1, "action": "notify the lead"}]})
    hooks = intake(lambda s, u: reply, check_reply=True)
    audit = attached("intake", hooks)
    assert hooks.before_reply("done").message
    [line] = audit.lines
    assert (line["verdict"], line["rule_id"], line["evidence"], line["checker"], line["hook"]) == (
        "followup",
        STATEMENT,
        "notify the lead",
        "model",
        "before_reply",
    )


def test_the_decision_model_of_intake_is_the_checker_kev():
    def ask(state, questions):
        return {name: 0.9 for name in questions}

    hooks = intake(verdict_model(), decisions=Decisions(ask), model_effect="block_once")
    audit = attached("intake", hooks)
    assert hooks.before_tool_call(*SEND).message
    [line] = audit.lines
    assert (line["verdict"], line["rule_id"], line["checker"], line["class"]) == (
        "block_once",
        "decisions",
        "kev",
        "judgment",
    )


def test_a_read_before_write_stop_and_the_trackers_have_a_checker():
    audit = AuditLog()
    audit.record(
        "intake",
        None,
        {
            "hook": "before_tool_call",
            "tool": "t",
            "outcome": "deny",
            "rule": "read_before_write",
            "unread": ["the SOP"],
        },
    )
    audit.record(
        "requirements_tracker", None, {"hook": "before_reply", "rule": "requirements", "outcome": "followup"}
    )
    audit.record("procedure_monitor", None, {"outcome": "step_done", "procedure": "p1", "step": "s1"})
    audit.record("deliverables", None, {"hook": "on_user_message", "outcome": "deliverables", "items": []})
    audit.record(
        "deliverables", None, {"hook": "before_reply", "rule": "deliverables", "outcome": "complete"}
    )
    assert [(x["rule_id"], x["checker"]) for x in audit.lines] == [
        ("read_before_write", "code"),
        ("requirements", "model"),
        ("p1", "monitor"),
        ("", "model"),
        ("deliverables", "code"),
    ]


# -- 3. the decisions of intake's own rule set --------------------------------------------------------------------------


def intake_with_rule(cap_effect="allow", decisions=None):
    hooks = OnlinePolicyIntake(
        lambda s, u: "{}",
        tool_specs=SPECS,
        verify_writes=True,
        read_before_write=False,
        check_reply=False,
        cap_effect=cap_effect,
        decisions=decisions,
    )
    hooks.enforcer.add_rules([RULE])
    return hooks


def test_the_cap_of_the_online_rule_set_lets_the_write_run_and_is_a_line():
    hooks = intake_with_rule("allow")
    audit = attached("intake", hooks)
    results = [hooks.before_tool_call(TOOL, {"status": "closed"}) for _ in range(4)]
    assert [bool(r.message) for r in results] == [True, True, True, False]  # the 4th runs
    assert [x["verdict"] for x in audit.lines] == ["deny", "deny", "deny", "deny", "allow_after_cap"]
    assert {x["rule_id"] for x in audit.lines} == {"r1"}
    assert len(hooks.records) == 3  # the records are what they were: the 4th call adds none


def test_the_cap_with_skip_has_one_name_and_one_line():
    hooks = intake_with_rule("skip")
    audit = attached("intake", hooks)
    for _ in range(4):
        hooks.before_tool_call(TOOL, {"status": "closed"})
    assert [x["verdict"] for x in audit.lines] == ["deny"] * 4 + ["deny_after_cap"]  # as the rules component
    assert audit.lines[-1]["rule_id"] == "r1" and "same denial" in audit.lines[-1]["evidence"]


def test_a_code_denial_that_the_scope_check_drops_is_a_line_and_the_write_runs():
    def ask(state, questions):
        return {name: 0.0 for name in questions}  # the rule does not apply to this call

    hooks = intake_with_rule(decisions=Decisions(ask))
    audit = attached("intake", hooks)
    assert not hooks.before_tool_call(TOOL, {"status": "closed"}).message  # the write runs
    assert hooks.records == []  # no record changes
    assert [(x["verdict"], x["rule_id"], x["checker"]) for x in audit.lines] == [
        ("out_of_scope", "r1", "code")
    ]


def test_the_rule_set_of_a_run_and_that_of_an_intake_name_the_cap_alike():
    rules = RuleSetHooks([RULE], SPECS, cap_effect="skip")
    audit = attached("rules", rules)
    for _ in range(4):
        rules.before_tool_call(TOOL, {"status": "closed"})
    assert audit.lines[-1]["verdict"] == "deny_after_cap"
    assert (
        rules.before_tool_call(TOOL, {"status": "closed"}).details[-1]["rule"] == "deny_cap"
    )  # the detail keeps its name


# -- what the audit shares with the components --------------------------------------------------------------------------


def test_the_audit_reads_the_threshold_of_the_navigator_and_not_a_copy_of_it(monkeypatch):
    done = TURNS + [  # every open step of the chapter is done, so only the answer on "solved" can differ
        {"role": "agent", "text": "Use toggle_roaming() and toggle_saver()."},
        {"role": "tool", "name": "enable_roaming", "result": "ok"},
    ]
    nav = SopNavigator(
        POLICY,
        BUILD,
        min_chapter_actions=1,
        agent_tools=["enable_roaming"],
        ask=lambda state, qs: {k: 0.6 if k == "solved" else 0.9 if k.startswith("c") else 0.0 for k in qs},
    )
    audit = attached("navigator", nav)
    assert nav.check_handoff(done) is None and "solved" in audit.lines[-1]["evidence"]  # 0.6 is yes at 0.5
    monkeypatch.setattr(sop_navigator, "YES", 0.7)
    nav.chapter = None
    assert nav.check_handoff(done) is None  # at 0.7 it is no, and nothing is open: allowed all the same
    assert "solved" not in audit.lines[-1]["evidence"] and "no open step" in audit.lines[-1]["evidence"]


def test_the_audit_reads_the_verdict_of_the_gate_and_not_a_copy_of_it(monkeypatch):
    def model(system, user):
        return json.dumps({"breaks": True, "rule": "r", "case": "c", "evidence": "quoted data is here"})

    gate = StoreGate(None, model, sop_text="the whole sop")
    audit = attached("gate", gate)
    gate.reads.append("tool: quoted data is here")
    assert "Blocked" in gate.before_tool_call("update_ticket", {"status": "open"})
    assert audit.lines[-1]["verdict"] == "block"
    monkeypatch.setattr(store_gate, "gate_verdict", lambda rec: "allow_repeat")
    gate.blocked.clear()
    assert (
        gate.before_tool_call("update_ticket", {"status": "open"}) is None
    )  # the component follows the function
    assert audit.lines[-1]["verdict"] == "allow_repeat"  # and so does the audit


# -- the file ---------------------------------------------------------------------------------------------------------


def test_a_lone_surrogate_in_a_text_does_not_stop_a_line(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.start()
    audit.record("rules", None, {"hook": "before_tool_call", "outcome": "deny", "reason": "bad \ud800 text"})
    audit.record("rules", None, {"hook": "before_tool_call", "outcome": "deny", "reason": "next"})
    written = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(written) == 2 and audit.path is not None and "ud800" in written[0]


# -- every model of a run is wrapped --------------------------------------------------------------------------------------


def test_each_method_of_the_assembly_that_makes_a_model_wraps_it_in_the_trace():
    models = Scripted(
        answers={"m/x": lambda s, u: "{}"},
        asks={"kev": lambda state, questions: {k: 0.1 for k in questions}},
    )
    a = Assembly(config(), None, ctx(), models)
    a.model("gate", "m/x")("S", "U")
    a.questions("navigator", "kev")("s", {"q": "?"})
    a.detector("kev")("r", ["t"])
    assert [(e["kind"], e["component"]) for e in a.trace.events] == [
        ("llm", "gate"),
        ("decision", "navigator"),
        ("detector", "detector"),
    ]
    assert a.trace.audit is a.audit


def test_a_model_that_a_method_choice_gives_a_verifier_by_model_name_is_wrapped(tmp_path):
    choice = tmp_path / "choice.json"
    choice.write_text('{"verify": {"checker": "verifier", "model": "m/other"}}')
    verify = {"id": "v1", "type": "verify", "statement": STATEMENT, "tools": [TOOL]}
    cfg = config(
        {
            "rules": "on",
            "method_choice": str(choice),
            "verifier": {"mode": "on", "model": "m/v", "scope": "all_statements"},
        },
        package="p.json",
    )
    models = Scripted(
        answers={"m/v": lambda s, u: '{"violations": []}', "m/other": lambda s, u: '{"violations": []}'}
    )
    parts = build_runtime(
        cfg, from_build({"version": "v9", "rules": [verify], "procedures": []}), ctx(), models=models
    )
    parts.hooks[0].before_tool_call(TOOL, {"status": "open"})
    assert [(e["component"], e["model"]) for e in parts.trace.events] == [("verifier", "m/other")]
    assert parts.usage()["verifier:m/other"]["calls"] == 1


def test_the_real_factory_is_wrapped_too(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    gate = {"gate": {"mode": "on", "model": "m/g", "min_strength": "none", "scope": "whole_sop"}}
    parts = build_runtime(config(gate), None, ctx(), models=Models())
    model = parts.hooks[0].model
    assert model.__qualname__.startswith("PolicyTrace.wrap_model") and model.model_name == "m/g"
    assert isinstance(RecordingModels().json_model("m/x"), type(lambda: 0))
