"""The audit: every component that decides writes a line in one shape, and the audit changes nothing in a run."""

import asyncio
import dataclasses
import json
from types import SimpleNamespace

import pytest

from polisys.dry_run import RecordingModels, describe_parts
from polisys.package import from_build
from polisys.runtime import RunContext, build_runtime
from polisys.ruleset import RuleSetHooks
from polisys.trace import AUDIT_FIELDS, EVIDENCE_CHARS, REDACTED, AuditLog, PolicyTrace, Records

from .test_runtime_assembly import (
    NAV_CTX,
    SOP_TEXT,
    SPECS,
    TOOL,
    config,
    navigator_config,
    navigator_package,
)
from .test_sop_navigator import TURNS, fake_ask

VERSION = "v9+abcdef012345"


class Scripted(RecordingModels):
    """The model factory with a scripted answer for each model name, counting calls as the real models do."""

    def __init__(self, answers=None, asks=None):
        super().__init__()
        self.answers, self.asks = answers or {}, asks or {}

    def _scripted(self, recorded, fn):
        def call(*args):
            recorded.usage["calls"] += 1
            return fn(*args)

        call.usage, call.model_name = recorded.usage, recorded.model_name
        return call

    def json_model(self, model, **kwargs):
        return self._scripted(super().json_model(model, **kwargs), self.answers[model])

    def questions(self, model, **kwargs):
        return self._scripted(super().questions(model, **kwargs), self.asks[model])

    def detector(self, model, **kwargs):
        return self._scripted(super().detector(model, **kwargs), self.asks[model])


def ctx(tmp_path=None, **given):
    base = dict(
        task_id="task-9",
        policy_version="v9",
        package_version=VERSION,
        request="Close the ticket of Lee.",
        system_prompt=SOP_TEXT,
        tool_specs=SPECS,
        out_dir=tmp_path,
    )
    return RunContext(**{**base, **given})


def lines_of(parts, component):
    return [x for x in parts.audit.lines if x["component"] == component]


def assert_one_shape(parts):
    assert parts.audit.lines
    for x in parts.audit.lines:
        assert tuple(x) == AUDIT_FIELDS and all(isinstance(v, str) for v in x.values())
        assert x["time"].endswith("Z") and x["package_version"] == VERSION and x["task"] == "task-9"


PRECONDITION = {
    "condition": "written authorization from the lead",
    "if_not": "stop",
    "quote": "Do not close a ticket without written authorization from the lead.",
}
PROCEDURE = {
    "id": "p1",
    "name": "Closing",
    "steps": [{"action": TOOL, "quote": "Update the ticket."}],
    "preconditions": [PRECONDITION],
}
RULE = {
    "id": "r1",
    "type": "allowed_values",
    "tool": TOOL,
    "arg": "status",
    "values": ["open"],
    "quote": "Only open.",
    "source": "§2",
}
PACKAGE = {"version": "v9", "rules": [RULE], "procedures": [PROCEDURE]}


def test_the_guide_stops_the_first_write_of_a_procedure_and_the_audit_says_so():
    cfg = config(organizers={"guide": {"mode": "on"}}, package="p.json")
    parts = build_runtime(cfg, from_build(PACKAGE), ctx(), models=Scripted())
    stop = parts.hooks[0].before_tool_call(TOOL, {"ticket_id": "T-1", "status": "open"})
    assert stop.message
    [line] = lines_of(parts, "guide")
    assert (line["verdict"], line["rule_id"], line["checker"], line["tool"]) == (
        "deny",
        "procedure_gate",
        "code",
        TOOL,
    )
    assert "Closing" in line["evidence"] and line["hook"] == "before_tool_call"
    assert_one_shape(parts)


def test_the_loop_audits_an_explore_stop_with_no_record_and_a_decision_for_each_case():
    loop_json = json.dumps(
        {
            "cases": [{"case": "Lee", "summary": "close the ticket", "keys": ["Lee"]}],
            "procedure": 1,
            "answer": "no",
            "evidence": "no authorization in the data",
        }
    )
    cfg = config(organizers={"loop": {"mode": "on", "model": "m/loop"}}, package="p.json")
    parts = build_runtime(
        cfg, from_build(PACKAGE), ctx(), models=Scripted(answers={"m/loop": lambda system, user: loop_json})
    )
    hooks = parts.hooks[0]
    hooks.mode = "explore"
    assert hooks.before_tool_call(TOOL, {}).message
    # the stop had no record before and has none now: the verdict records are as they were
    assert hooks.records == []
    [stop] = lines_of(parts, "loop")
    assert (stop["rule_id"], stop["verdict"], stop["checker"]) == ("explore_phase", "deny", "code")

    class Agent:
        async def invoke(self, message, thread_id=None):
            return SimpleNamespace(answer="done", receipt=None, error=None)

    asyncio.run(parts.run(Agent(), "Close the ticket of Lee.", "th-1", 30))
    decided = [x for x in lines_of(parts, "loop") if x["rule_id"] == "Closing"]
    assert [(x["verdict"], x["checker"]) for x in decided] == [("stop", "model")]
    assert "Lee" in decided[0]["evidence"] and "authorization" in decided[0]["evidence"]
    # the plan of the loop lists the cases and decides nothing about a rule
    assert not [x for x in parts.audit.lines if "plan" in x["evidence"]]
    assert_one_shape(parts)


NAVIGATOR_CTX = dataclasses.replace(NAV_CTX, task_id="task-9", package_version=VERSION)


def navigator_run():
    return build_runtime(navigator_config(), navigator_package(), NAVIGATOR_CTX, models=Asks())


class Asks(RecordingModels):
    """A real navigator with the decision model of the navigator tests."""

    def questions(self, *args, **kwargs):
        recorded = super().questions(*args, **kwargs)
        ask = fake_ask()
        ask.usage, ask.model_name = recorded.usage, recorded.model_name
        return ask


def test_a_blocked_hand_off_and_a_hand_off_let_through_at_the_limit_are_both_audited():
    parts = navigator_run()
    turns, live = [dict(t) for t in TURNS], []
    for _ in range(2):
        blocked = parts.after_reply("I am transferring you now.", turns, live)  # the note is the new prompt
        assert blocked
    assert parts.after_reply("I am transferring you now.", turns, live) is None  # the limit: it goes through
    lines = lines_of(parts, "navigator")
    assert [x["verdict"] for x in lines] == ["block", "block", "allow_after_cap"]
    assert lines[0]["checker"] == "kev" and lines[0]["source"] and "doc:roaming:1" in lines[0]["rule_id"]
    assert [x["hook"] for x in lines[:2]] == [
        "before_reply",
        "before_reply",
    ]  # a block says where it happened
    assert (lines[2]["rule_id"], lines[2]["hook"], lines[2]["checker"]) == (
        "max_blocks",
        "before_reply",
        "code",
    )
    assert_one_shape(parts)


def test_the_decision_model_and_the_detector_of_the_decision_models_are_traced_and_counted_once():
    reply = {
        "policies": [
            {"record": 1, "title": "Priority", "trust": "trusted", "statements": ["Answer questions"]}
        ]
    }
    cfg = config({"rules": "off", "compile": "m/i", "detector": "kev"}, sources={"data": True})
    models = Scripted(
        answers={"m/i": lambda s, u: json.dumps(reply)},
        asks={"kev": lambda request, texts: [0.9] * len(texts)},
    )
    parts = build_runtime(cfg, None, ctx(), models=models)
    intake = parts.hooks[0]
    intake.on_user_message("answer the mentions", None)
    intake.after_tool_call(
        "gmail_find_email",
        {"query": "p"},
        {
            "messages": [
                {"subject": "Rules", "body": "RESPOND: questions."},
                {"subject": "Lunch", "body": "Pizza on Friday for the whole team, " + "x" * 200},
            ]
        },
    )
    intake._drain(5)

    kinds = sorted((e["kind"], e["component"]) for e in parts.trace.events)
    assert kinds == [("detector", "detector"), ("llm", "intake")]
    usage = parts.usage()
    assert (
        usage["intake"]["calls"] == 1 and usage["detector"]["calls"] == 1
    )  # each call once, not once more by the trace
    verdicts = [x["verdict"] for x in lines_of(parts, "intake")]
    assert "detected" in verdicts and "policy_found" in verdicts


def test_asking_the_decision_model_through_the_trace_returns_what_it_returned_and_counts_once():
    ask = fake_ask(solved=0.9)
    ask.usage, ask.model_name = {"calls": 0, "errors": 0, "cost": 0.0}, "kev"
    wrapped = PolicyTrace().wrap_ask(ask, "navigator")
    assert wrapped("state", {"solved": "?"}) == ask("state", {"solved": "?"}) and wrapped.usage is ask.usage


def test_a_copy_of_a_tapped_list_is_a_plain_list_and_writes_no_line():
    import copy

    hooks = RuleSetHooks([RULE], SPECS)
    audit = AuditLog()
    audit.attach("rules", hooks)
    hooks.before_tool_call(TOOL, {"status": "closed"})
    written = len(audit.lines)
    for duplicate in (copy.deepcopy(hooks.records), copy.copy(hooks.records), hooks.records.copy()):
        assert type(duplicate) is list and duplicate == hooks.records
    assert len(audit.lines) == written


def test_a_hand_off_that_is_fine_is_an_allow_with_the_reason():
    parts = navigator_run()
    parts.components["navigator"].navigator.ask = fake_ask(solved=0.9)
    assert parts.after_reply("I am transferring you now.", [dict(t) for t in TURNS], []) is None
    [line] = lines_of(parts, "navigator")
    assert line["verdict"] == "allow" and "solved" in line["evidence"]


def test_the_requirements_tracker_and_the_sop_tracker_write_their_verdicts():
    def tracker_model(system, user):
        if system.startswith("You check finished work"):
            return json.dumps({"items": [{"id": "R1", "status": "missing", "why": "no write"}]})
        return json.dumps(
            {"items": [{"kind": "effect", "text": "close the ticket", "quote": "Close the ticket of Lee."}]}
        )

    def sop_model(system, user):
        if "SOP chapters" in user:
            return json.dumps({"chapters": ["1"]})
        return json.dumps(
            {
                "steps": [
                    {
                        "action": "open the ticket",
                        "quote": "Open a ticket.",
                        "status": "pending",
                        "evidence": "",
                    }
                ]
            }
        )

    cfg = config(
        organizers={
            "requirements_tracker": {"mode": "on", "model": "m/req", "read_system_prompt": True},
            "sop_tracker": {"mode": "on", "model": "m/sop"},
        },
        sources={"request": True},
    )
    parts = build_runtime(
        cfg, None, ctx(), models=Scripted(answers={"m/req": tracker_model, "m/sop": sop_model})
    )
    tracker = parts.hooks[0]
    tracker.on_user_message("Close the ticket of Lee.", None)
    assert tracker.before_reply("done").message
    assert parts.turn_note([{"role": "customer", "text": "Please open a ticket."}])
    compiled, req = lines_of(parts, "requirements_tracker")
    assert compiled["verdict"] == "compiled" and "close the ticket" in compiled["evidence"]
    assert (compiled["checker"], req["checker"]) == ("model", "model")
    assert (req["verdict"], req["rule_id"], req["hook"]) == ("followup", "requirements", "before_reply")
    [sop] = lines_of(parts, "sop_tracker")
    assert (sop["verdict"], sop["checker"]) == ("remind", "model") and "open the ticket" in sop["evidence"]
    assert_one_shape(parts)


def test_the_procedure_monitor_and_the_rules_share_one_audit():
    cfg = config(checkers={"rules": "on"}, package="p.json")
    parts = build_runtime(cfg, from_build(PACKAGE), ctx(), models=Scripted())
    rules, monitor = parts.hooks
    assert rules.before_tool_call(TOOL, {"status": "closed"}).message
    monitor.after_tool_call(TOOL, {"status": "open"}, {"ok": True})
    [denied] = [x for x in lines_of(parts, "rules") if x["verdict"] == "deny"]
    [step] = lines_of(parts, "procedure_monitor")
    assert (denied["rule_id"], denied["level"], denied["class"], denied["source"]) == (
        "r1",
        "deny",
        "hard",
        "§2",
    )
    assert (step["verdict"], step["rule_id"]) == ("step_done", "p1")
    assert_one_shape(parts)


def test_intake_deliverables_and_the_gate_are_read_through_the_same_path():
    from polisys.deliverables import DeliverablesHooks
    from polisys.intake import OnlinePolicyIntake

    audit = AuditLog(task="task-9", package_version=VERSION)
    reply = {
        "policies": [
            {"record": 1, "title": "Priority", "trust": "trusted", "statements": ["Answer questions"]}
        ]
    }
    intake = OnlinePolicyIntake(lambda s, u: json.dumps(reply), verify_writes=False, check_reply=False)
    audit.attach("intake", intake)
    intake.on_user_message("answer the mentions", None)
    intake.after_tool_call(
        "gmail_find_email",
        {"query": "priority"},
        {"messages": [{"subject": "Rules", "body": "RESPOND: questions."}]},
    )
    intake._drain(5)
    deliverables = DeliverablesHooks(lambda s, u: json.dumps({"items": []}), SPECS)
    audit.attach("deliverables", deliverables)
    deliverables.on_user_message("Close the ticket.", None)

    found = [x for x in audit.lines if x["component"] == "intake"]
    assert [x["verdict"] for x in found] == ["policy_found"]
    assert "Answer questions" in found[0]["evidence"]
    assert [x["verdict"] for x in audit.lines if x["component"] == "deliverables"] == ["deliverables"]
    assert all(tuple(x) == AUDIT_FIELDS for x in audit.lines)


@pytest.mark.parametrize(
    "cap_effect, verdicts",
    [
        ("allow", ["deny", "deny", "deny", "allow_after_cap", "allow_after_cap"]),
        ("skip", ["deny", "deny", "deny", "deny_after_cap", "deny_after_cap"]),
    ],
)
def test_an_intake_record_is_final_when_the_audit_reads_it_also_after_the_cap(cap_effect, verdicts):
    """The cap used to change the outcome of a record after it was appended. The line is written at the append."""
    from polisys.intake import OnlinePolicyIntake

    def model(system, user):
        return json.dumps({"violations": [{"statement": 1, "reason": "no", "fix": "stop"}]})

    hooks = OnlinePolicyIntake(
        model, check_reply=False, read_before_write=False, cap_effect=cap_effect, verifier_may_deny=True
    )
    audit = AuditLog()
    audit.attach("intake", hooks)
    hooks.on_user_message("send it", None)
    hooks.policies = [{"title": "T", "trust": "trusted", "statements": ["Never send it."]}]
    for _ in range(5):
        hooks.before_tool_call("gmail_send_email", {"body": "x"})
    assert [r["outcome"] for r in hooks.records] == verdicts
    assert [x["verdict"] for x in audit.lines] == verdicts


def test_the_guard_code_cap_is_final_when_the_audit_reads_it():
    from polisys.intake import OnlinePolicyIntake

    from .test_guard_code import CHECK, ROWS

    def model(system, user):
        if "Python check" in system:
            return json.dumps({"applies": True, "code": CHECK})
        return json.dumps({"violations": []})

    hooks = OnlinePolicyIntake(
        model, check_reply=False, read_before_write=False, guard_code="deny", today="2026-02-20"
    )
    audit = AuditLog()
    audit.attach("intake", hooks)
    hooks.on_user_message("Process this month's amortization.", None)
    hooks.policies = [
        {
            "title": "Rules",
            "trust": "trusted",
            "statements": ["Only amortize items whose coverage has started."],
        }
    ]
    hooks.after_tool_call("get_rows", {"spreadsheet": "ss_prepaids"}, ROWS)
    for _ in range(5):
        hooks.before_tool_call("gmail_send_email", {"body": "Office Lease Deposit $750"})
    expected = ["deny", "deny", "deny", "allow_after_cap", "allow_after_cap"]
    assert [x["verdict"] for x in audit.lines] == expected
    assert [r["outcome"] for r in hooks.records if r.get("rule") == "guard_code"] == expected
    # the rule is named by its statement, with its source: ``statement`` is only a place in the list of one call
    assert {(x["rule_id"], x["source"], x["checker"], x["class"]) for x in audit.lines} == {
        ("Only amortize items whose coverage has started.", "Rules", "code", "hard")
    }


def test_a_rule_line_says_who_checked_it_cedar_for_roles_the_verifier_for_judgment_and_code_for_the_rest():
    from polisys.verifier import Verifier

    rules = [
        {"id": "role", "type": "role_permission", "tools": [TOOL], "roles": ["agent"], "effect": "forbid"},
        {"id": "judge", "type": "verify", "statement": "Be kind.", "tools": [TOOL], "level": "remind"},
        {"id": "ok", "type": "required_args", "tool": TOOL, "args": ["status"], "level": "deny"},
    ]
    verifier = Verifier(lambda system, user: json.dumps({"violations": [{"statement": 1, "reason": "rude"}]}))
    hooks = RuleSetHooks(rules, SPECS, principal={"role": "agent"}, verifier=verifier)
    audit = AuditLog(task="t", package_version="v")
    audit.attach("rules", hooks)
    hooks.before_tool_call(TOOL, {"status": "open"})
    by_rule = {x["rule_id"]: x for x in audit.lines}
    assert (by_rule["role"]["checker"], by_rule["role"]["class"]) == ("cedar", "hard")
    assert (by_rule["ok"]["checker"], by_rule["ok"]["level"], by_rule["ok"]["verdict"]) == (
        "code",
        "deny",
        "allow",
    )
    # the verifier's own verdict: its rule is the one it broke, and it is a judgment checked by a model
    assert (by_rule["judge"]["checker"], by_rule["judge"]["class"], by_rule["judge"]["evidence"]) == (
        "verifier",
        "judgment",
        "rude",
    )


@pytest.mark.parametrize("cap_effect, verdict", [("allow", "allow_after_cap"), ("skip", "deny_after_cap")])
def test_the_deny_cap_is_audited_with_the_rule_it_let_through_or_kept_blocked(cap_effect, verdict):
    hooks = RuleSetHooks([RULE], SPECS, cap_effect=cap_effect)
    audit = AuditLog()
    audit.attach("rules", hooks)
    for _ in range(4):  # the cap is 3 denials of the same call
        hooks.before_tool_call(TOOL, {"status": "closed"})
    assert [x["verdict"] for x in audit.lines] == ["deny"] * 4 + [verdict]
    capped = audit.lines[-1]
    assert (
        capped["rule_id"] == "r1" and "same denial 4 times" in capped["evidence"] and capped["tool"] == TOOL
    )
    assert len(hooks.records) == 4  # the cap is no verdict record: the records are as they were


# -- what the audit does not do ----------------------------------------------------------------------------------------


def test_a_record_list_behaves_as_a_list_and_a_new_session_keeps_the_tap():
    hooks = RuleSetHooks([RULE], SPECS)
    audit = AuditLog()
    audit.attach("rules", hooks)
    hooks.before_tool_call(TOOL, {"status": "closed"})
    assert isinstance(hooks.records, Records) and isinstance(hooks.records, list) and len(hooks.records) == 1
    hooks.session_start("thread-2")
    assert hooks.records == [] and len(audit.lines) == 1
    hooks.before_tool_call(TOOL, {"status": "closed"})
    assert len(hooks.records) == 1 and len(audit.lines) == 2  # still reported after the reset


def test_the_audit_never_holds_a_key_and_cuts_a_long_text(monkeypatch):
    # test values, made in parts so that no key-shaped literal is in the file
    key = "sk-" + "live-" + "0123456789" + "abcdefghij"
    password = "hunter2" + "-hunter2"  # pragma: allowlist secret
    bearer = "Bearer " + "abcdefghij" + "klmnopqrstuv"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setenv("SOME_PASSWORD", password)
    audit = AuditLog()
    note = f"key {key} and {password} and {bearer} " + "x" * 5000
    audit.record("rules", None, {"hook": "before_tool_call", "outcome": "deny", "reason": note})
    [line] = audit.lines
    assert not any(secret in line["evidence"] for secret in (key, password, bearer, "abcdefghij"))
    assert line["evidence"].count(REDACTED) == 3
    assert len(line["evidence"]) < EVIDENCE_CHARS + 60 and "(cut, " in line["evidence"]


def test_the_audit_does_not_hold_a_prompt():
    hooks_prompt = "SYSTEM PROMPT " * 500
    audit = AuditLog()
    audit.record(
        "gate",
        None,
        {
            "gate": "checked",
            "tool": TOOL,
            "breaks": True,
            "rule": "r",
            "case": "c",
            "evidence": hooks_prompt,
            "grounded": True,
        },
    )
    assert len(audit.lines[0]["evidence"]) < EVIDENCE_CHARS + 60


def test_a_record_the_audit_cannot_read_is_skipped_and_the_component_goes_on():
    audit = AuditLog()

    class Odd:
        def __repr__(self):
            raise RuntimeError("no text")

    audit.record(
        "guide", None, {"outcome": "deny", "rule": "x", "evidence": Odd()}
    )  # str() fails: nothing raises
    audit.record("guide", None, "not a dict")
    audit.record("guide", None, {"outcome": "deny", "rule": "x"})
    assert [x["rule_id"] for x in audit.lines] == ["x"]


def test_the_wrapper_of_a_model_keeps_its_attributes_and_passes_arguments_and_answers_through():
    trace = PolicyTrace()
    seen = []

    def model(system, user, extra=None):
        seen.append((system, user, extra))
        return "answer"

    model.usage, model.model_name = {"calls": 0}, "m/x"
    wrapped = trace.wrap_model(model, "gate")
    assert wrapped("S", "U", extra=1) == "answer" and seen == [("S", "U", 1)]
    assert wrapped.usage is model.usage and wrapped.model_name == "m/x"
    with pytest.raises(ZeroDivisionError):
        trace.wrap_ask(lambda state, questions: 1 / 0, "decisions")("s", {"q": "?"})
    assert [e["kind"] for e in trace.events] == ["llm", "decision"] and "ZeroDivisionError" in trace.events[
        1
    ]["error"]


def test_every_model_of_a_run_is_made_by_a_method_of_the_assembly_that_wraps_it():
    """A component that reached ``models`` itself would call a model that the trace does not see."""
    import ast
    import inspect

    from polisys import runtime

    users = []

    def walk(node, scope):
        for child in ast.iter_child_nodes(node):
            inner = [*scope, child.name] if isinstance(child, (ast.FunctionDef, ast.ClassDef)) else scope
            if isinstance(child, ast.Attribute) and child.attr == "models":
                users.append(".".join(scope))
            walk(child, inner)

    walk(ast.parse(inspect.getsource(runtime)), [])
    assert sorted(set(users)) == [
        "Assembly.__init__",
        "Assembly.detector",
        "Assembly.model",
        "Assembly.questions",
    ]


# -- usage ---------------------------------------------------------------------------------------------------------------


def test_usage_is_counted_once_per_call_and_the_trace_adds_nothing_to_it():
    def silent(system, user):
        return json.dumps({"breaks": False})

    gate = {"gate": {"mode": "on", "model": "m/g", "min_strength": "none", "scope": "whole_sop"}}
    parts = build_runtime(config(gate), None, ctx(), models=Scripted(answers={"m/g": silent}))
    for _ in range(3):
        parts.hooks[0].before_tool_call(TOOL, {"status": "open"})
    row = parts.usage()["gate"]
    assert row["calls"] == 3 and row["model"] == "m/g"  # not 6: the trace did not add its own count
    assert len([e for e in parts.trace.events if e["component"] == "gate"]) == 3


# -- the golden description does not move -------------------------------------------------------------------------------


def described(trace):
    models = Scripted(answers={"m/g": lambda s, u: "{}"})
    cfg = config(
        {"gate": {"mode": "on", "model": "m/g", "min_strength": "none", "scope": "whole_sop"}}, trace=trace
    )
    return json.dumps(
        describe_parts(build_runtime(cfg, None, ctx(), models=models), models), sort_keys=True, default=str
    )


def test_the_wiring_description_is_the_same_with_the_trace_on_and_off():
    assert described(True) == described(False)


def test_no_decision_with_the_trace_off_and_none_in_a_baseline_run(tmp_path):
    off = build_runtime(config(trace=False), None, ctx(tmp_path / "off"))
    base = build_runtime(config(), None, ctx(tmp_path / "base"))
    for parts in (off, base):
        assert parts.audit is None and parts.trace is None
    assert not (tmp_path / "off").exists() and not (tmp_path / "base").exists()


def test_the_file_exists_empty_for_a_policy_run_that_decides_nothing(tmp_path):
    parts = build_runtime(
        config(checkers={"rules": "on"}, package="p.json"), from_build(PACKAGE), ctx(tmp_path)
    )
    assert (tmp_path / "audit.jsonl").read_text() == "" and parts.audit.lines == []


def test_a_second_run_in_the_same_folder_replaces_the_first_audit(tmp_path):
    for _ in range(2):
        parts = build_runtime(
            config(checkers={"rules": "on"}, package="p.json"), from_build(PACKAGE), ctx(tmp_path)
        )
        parts.hooks[0].before_tool_call(TOOL, {"status": "closed"})
    assert len((tmp_path / "audit.jsonl").read_text().splitlines()) == 1
