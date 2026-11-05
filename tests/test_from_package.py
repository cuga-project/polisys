"""Components read a policy package: same outcomes as from old inputs; ``level`` decides where nothing measured does."""

import copy

import pytest

from polisys.model import RULE_LEVELS, Step, Verdict, step_text
from polisys.package import (
    from_build,
    from_store_graph,
    load_package,
    procedure_dicts,
    runtime_rules,
    save_package,
    to_store_graph,
)
from polisys.procedure_guide import ProcedureGuide
from polisys.procedure_store import ProcedureStore
from polisys.ruleset import RuleSetHooks
from polisys.sop_loop import SopLoop, _proc_line
from polisys.sop_navigator import SopNavigator
from polisys.store_gate import StoreGate

from .test_sop_navigator import BUILD as NAVIGATOR_BUILD
from .test_sop_navigator import POLICY as NAVIGATOR_POLICY
from .test_sop_navigator import TURNS, fake_ask

TOOL = "update_ticket"
SPECS = [{"name": TOOL}, {"name": "notify_owner"}]
PRINCIPAL = {"id": "u1", "role": "agent"}

# rule type -> (slots, what was read before, arguments of a call that breaks the rule)
CASES = {
    "required_args": ({"tool": TOOL, "args": ["caller_id"]}, None, {}),
    "allowed_values": ({"tool": TOOL, "arg": "status", "values": ["open"]}, None, {"status": "zzz"}),
    "derived_value": (
        {"tool": TOOL, "arg": "priority", "inputs": ["impact"], "table": {"low": "low"}},
        None,
        {"priority": "high", "impact": "low"},
    ),
    "transition": (
        {"tool": TOOL, "arg": "state", "id_arg": "ticket_id", "edges": {"new": ["open"]}},
        {"id": "T-1", "state": "new"},
        {"ticket_id": "T-1", "state": "done"},
    ),
    "reference_status": (
        {"tool": TOOL, "arg": "ticket_id", "field": "state", "allowed": ["open"]},
        {"id": "T-1", "state": "new"},
        {"ticket_id": "T-1"},
    ),
    "membership": (
        {"tool": TOOL, "arg": "user", "group_arg": "group"},
        {"group": "G-1", "user": "U-2"},
        {"user": "U-1", "group": "G-1"},
    ),
    "role_permission": ({"tools": [TOOL], "roles": ["agent"], "effect": "forbid"}, None, {}),
    "record_condition": (
        {"tools": [TOOL], "field": "notes", "op": "contains", "value": "FSR-"},
        {"id": "T-1", "notes": "FSR-1"},
        {"ticket_id": "T-1"},
    ),
    "threshold": (
        {"tool": TOOL, "arg": "amount", "op": "gt", "value": 100, "then": "forbid"},
        None,
        {"amount": 500},
    ),
    "format": ({"tool": TOOL, "arg": "code", "pattern": r"X\d+"}, None, {"code": "zzz"}),
    "grounded_values": ({}, None, {"option": "unseen"}),
    "id_provenance": ({}, None, {"ref_id": "AB-123"}),
    "verify": ({"statement": "Do not do it.", "tools": [TOOL]}, None, {}),
    "duty": ({"trigger": {"tool": TOOL}, "then": {"tool": "notify_owner"}}, None, {}),
    "guidance": ({"text": "Be kind.", "tools": [TOOL]}, None, {}),
}
TODAY = {  # the outcome at the write with no method choice, for a rule with no may_deny key
    "grounded_values": "observe",
    "id_provenance": "observe",
    "verify": "remind",
    "duty": None,
    "guidance": None,
}
STATES = {
    "plain": {},
    "may_deny_false": {"may_deny": False},
    "may_deny_true": {"may_deny": True},
    "schema_conflict": {"schema_conflict": True},
}
CHOICES = {
    "no_choice": None,
    "code_off": {"checker": "code", "code_may_deny": False},
    "code_on": {"checker": "code", "code_may_deny": True},
    "no_setting": {"checker": "code"},
    "to_verifier": {"checker": "verifier", "code_may_deny": False, "may_deny": True},
    "verifier_off": {"checker": "verifier", "code_may_deny": True, "may_deny": False},
}


class DenyAll:
    """A verifier that finds every statement broken."""

    def check(self, request, statements, tool, args, facts, props):
        return [Verdict(outcome="deny", rule_id=s["rule_id"], reason="broken") for s in statements]


def _rule(rule_type, **state):
    return {
        "id": "r1",
        "type": rule_type,
        "source": "§1",
        "quote": "the rule",
        **CASES[rule_type][0],
        **state,
    }


def _wrapped(rule_type, wrapper_may_deny=True, **state):
    return {
        "id": "r1",
        "type": rule_type,
        "rule": _rule(rule_type, **state),
        "statement_ids": ["s1"],
        "checker": "code",
        "may_deny": wrapper_may_deny,
        "source": {},
        "quote": "the rule",
    }


def _package(rule_type, level=None, **state):
    pkg = from_build({"rules": [_wrapped(rule_type, **state)]})
    if level:
        pkg.rules[0].level = level
    return pkg


def _run(hooks, rule_type, calls=1):
    """Everything the hooks say and record when a call breaks the rule."""
    _, read, args = CASES[rule_type]
    out = {"brief": hooks.before_llm_call(None).message}
    if read:
        hooks.after_tool_call("get_ticket", {}, read)
    out["messages"] = []
    for _ in range(calls):
        out["messages"].append(hooks.before_tool_call(TOOL, dict(args)).message)
        out["messages"].append(hooks.after_tool_call(TOOL, dict(args), {"ok": True}).message)
    out["reply"] = hooks.before_reply("done").message
    out["next_brief"] = hooks.before_llm_call(None).message
    out["records"] = [(r["rule_id"], r["type"], r["outcome"], r["reason"]) for r in hooks.records]
    return out


def _outcomes(hooks, rule_type, calls=1):
    return [r[2] for r in _run(hooks, rule_type, calls)["records"]]


def _hooks_kwargs(rule_type, choice, verifier_may_deny=False):
    method_choice = {rule_type: choice} if choice else None
    return dict(
        principal=PRINCIPAL,
        method_choice=method_choice,
        verifier=DenyAll(),
        verifier_may_deny=verifier_may_deny,
    )


# -- parity with the old inputs ----------------------------------------------------------------------------------------


def _load_rules_as_today(source):
    """What the callers of today pass to the constructor (``rule_audit.load_rules``)."""
    out = []
    for r in source["rules"]:
        if isinstance(r.get("rule"), dict):
            x = dict(r["rule"])
            if r.get("may_deny") is False:
                x["may_deny"] = False
            out.append(x)
        else:
            out.append(dict(r))
    return out


@pytest.mark.parametrize("rule_type", sorted(CASES))
@pytest.mark.parametrize("state", sorted(STATES))
@pytest.mark.parametrize("choice", sorted(CHOICES))
@pytest.mark.parametrize("shape", ["flat", "wrapped", "wrapper_false"])
@pytest.mark.parametrize("verifier_may_deny", [False, True])
def test_outcomes_are_the_same_from_a_package_and_from_old_input(
    rule_type, state, choice, shape, verifier_may_deny
):
    if shape == "flat":
        rules = [_rule(rule_type, **STATES[state])]
    elif shape == "wrapped":
        rules = [_wrapped(rule_type, **STATES[state])]
    else:
        rules = [_wrapped(rule_type, wrapper_may_deny=False, **STATES[state])]
    source = {"rules": rules}
    kwargs = _hooks_kwargs(rule_type, CHOICES[choice], verifier_may_deny)
    old = RuleSetHooks(_load_rules_as_today(source), SPECS, **kwargs)
    new = RuleSetHooks.from_package(from_build(copy.deepcopy(source)), SPECS, **kwargs)
    assert _run(new, rule_type, calls=2) == _run(old, rule_type, calls=2)


@pytest.mark.parametrize("rule_type", sorted(CASES))
def test_a_package_read_from_a_file_gives_the_same_outcome(tmp_path, rule_type):
    path = tmp_path / "package.json"
    save_package(_package(rule_type), path)
    kwargs = _hooks_kwargs(rule_type, None)
    old = RuleSetHooks([_rule(rule_type)], SPECS, **kwargs)
    new = RuleSetHooks.from_package(load_package(path), SPECS, **kwargs)
    assert _run(new, rule_type) == _run(old, rule_type)


@pytest.mark.parametrize("rule_type", sorted(CASES))
def test_the_cases_above_break_their_rule(rule_type):
    """The parity above would pass with no rule firing."""
    hooks = RuleSetHooks.from_package(_package(rule_type), SPECS, **_hooks_kwargs(rule_type, None))
    outcomes = _outcomes(hooks, rule_type)
    expected = TODAY.get(rule_type, "deny")
    if expected:
        assert expected in outcomes
    elif rule_type == "duty":
        assert outcomes == ["duty_open", "followup"]
    else:
        assert outcomes == []


# -- level -------------------------------------------------------------------------------------------------------------


def _format_hooks(level, **kwargs):
    return RuleSetHooks.from_package(_package("format", level=level), SPECS, principal=PRINCIPAL, **kwargs)


def test_runtime_rule_carries_the_level_and_no_wrapper_true():
    rule = runtime_rules(_package("grounded_values"))[0]
    assert rule["level"] == "observe" and "may_deny" not in rule
    rule = runtime_rules(from_build({"rules": [_wrapped("format", wrapper_may_deny=False)]}))[0]
    assert rule["level"] == "remind" and rule["may_deny"] is False


@pytest.mark.parametrize("level", ["Deny", "dney", "hard", "block-once", "DENY", 1, ["deny"]])
def test_an_unknown_level_raises_and_names_the_rule(level):
    with pytest.raises(ValueError, match="rule r1: level"):
        RuleSetHooks.from_package(_package("format", level=level), SPECS)


def test_an_unknown_level_from_a_file_raises(tmp_path):
    path = tmp_path / "package.json"
    save_package(_package("format", level="dney"), path)
    with pytest.raises(ValueError, match="rule r1: level 'dney'"):
        runtime_rules(load_package(path))


def test_an_unknown_level_inside_the_rule_dict_raises():
    pkg = _package("format")
    pkg.rules[0].level, pkg.rules[0].rule["level"] = "", "dney"
    with pytest.raises(ValueError, match="rule r1: level 'dney'"):
        runtime_rules(pkg)


@pytest.mark.parametrize("level", RULE_LEVELS)
def test_every_known_level_is_accepted(level):
    assert runtime_rules(_package("format", level=level))[0]["level"] == level


def test_a_package_rule_with_no_level_runs_the_old_code():
    pkg = _package("format")
    pkg.rules[0].level = ""
    assert "level" not in runtime_rules(pkg)[0]


def test_level_deny_stops_the_call():
    hooks = _format_hooks("deny")
    assert hooks.before_tool_call(TOOL, {"code": "zzz"}).message
    assert hooks.records[-1]["outcome"] == "deny"


def test_level_remind_lets_the_call_run_and_leaves_a_note():
    hooks = _format_hooks("remind")
    assert hooks.before_tool_call(TOOL, {"code": "zzz"}).message is None
    assert hooks.records[-1]["outcome"] == "remind"
    assert "POLICY NOTE" in hooks.after_tool_call(TOOL, {"code": "zzz"}, {"ok": True}).message


def test_level_observe_records_only_and_is_not_in_the_brief():
    hooks = _format_hooks("observe")
    assert hooks.before_llm_call(None).message is None
    assert hooks.before_tool_call(TOOL, {"code": "zzz"}).message is None
    assert hooks.records[-1]["outcome"] == "observe"
    assert hooks.pop_notes() == []


def test_level_block_once_stops_one_call_then_reminds():
    hooks = _format_hooks("block_once")
    assert hooks.before_tool_call(TOOL, {"code": "zzz"}).message
    assert hooks.before_tool_call(TOOL, {"code": "yyy"}).message is None
    assert [r["outcome"] for r in hooks.records] == ["deny", "remind"]
    hooks.session_start(None)
    assert hooks.before_tool_call(TOOL, {"code": "zzz"}).message


def test_level_block_once_counts_only_a_real_stop():
    hooks = _format_hooks("block_once", mode="shadow")
    for _ in range(2):
        hooks.before_tool_call(TOOL, {"code": "zzz"})
    assert [r["outcome"] for r in hooks.records] == ["deny", "deny"]


@pytest.mark.parametrize(
    "level, choice, outcome",
    [
        ("remind", {"checker": "code"}, "deny"),
        ("observe", {"checker": "code", "code_may_deny": True}, "deny"),
        ("deny", {"checker": "code", "code_may_deny": False}, "remind"),
        ("block_once", {"checker": "code", "code_may_deny": False}, "remind"),
    ],
)
def test_a_method_choice_wins_over_the_level(level, choice, outcome):
    hooks = _format_hooks(level, method_choice={"format": choice})
    hooks.before_tool_call(TOOL, {"code": "zzz"})
    assert hooks.records[-1]["outcome"] == outcome


def test_a_rule_routed_to_the_verifier_by_a_method_choice_has_no_level():
    choice = {"format": {"checker": "verifier", "code_may_deny": False, "may_deny": True}}
    hooks = _format_hooks("observe", method_choice=choice, verifier=DenyAll(), verifier_may_deny=True)
    assert hooks.before_tool_call(TOOL, {"code": "zzz"}).message
    assert hooks.records[-1]["outcome"] == "deny"


def _verify_outcome(level, **kwargs):
    hooks = RuleSetHooks.from_package(_package("verify", level=level), SPECS, verifier=DenyAll(), **kwargs)
    hooks.before_tool_call(TOOL, {})
    return hooks.records[-1]["outcome"]


@pytest.mark.parametrize("level", ["observe", "remind", "block_once", "deny"])
def test_verifier_may_deny_true_wins_over_the_level_of_a_verify_rule(level):
    assert _verify_outcome(level, verifier_may_deny=True) == "deny"


@pytest.mark.parametrize(
    "level, outcome",
    [("deny", "remind"), ("block_once", "remind"), ("remind", "remind"), ("observe", "observe")],
)
def test_verifier_may_deny_false_means_no_setting_so_a_verify_rule_never_denies_and_observe_stays(
    level, outcome
):
    assert _verify_outcome(level) == outcome
    assert _verify_outcome(level, verifier_may_deny=False) == outcome


@pytest.mark.parametrize(
    "rule_type, state, outcome",
    [
        ("format", {"schema_conflict": True}, "observe"),
        ("format", {"may_deny": False}, "remind"),
        ("grounded_values", {"may_deny": False}, "observe"),
        ("id_provenance", {"may_deny": False}, "observe"),
    ],
)
def test_limits_stay_at_level_deny(rule_type, state, outcome):
    hooks = RuleSetHooks.from_package(_package(rule_type, level="deny", **state), SPECS)
    hooks.before_tool_call(TOOL, CASES[rule_type][2])
    assert hooks.records[-1]["outcome"] == outcome


@pytest.mark.parametrize("rule_type", ["grounded_values", "id_provenance"])
def test_a_level_can_raise_a_no_assumed_data_rule_only_without_a_method_choice(rule_type):
    pkg = _package(rule_type, level="deny")
    hooks = RuleSetHooks.from_package(pkg, SPECS)
    hooks.before_tool_call(TOOL, CASES[rule_type][2])
    assert hooks.records[-1]["outcome"] == "deny"
    hooks = RuleSetHooks.from_package(pkg, SPECS, method_choice={rule_type: {"checker": "code"}})
    hooks.before_tool_call(TOOL, CASES[rule_type][2])
    assert hooks.records[-1]["outcome"] == "remind"


@pytest.mark.parametrize("level", ["observe", "remind", "block_once", "deny"])
def test_a_duty_keeps_its_follow_up_path(level):
    hooks = RuleSetHooks.from_package(_package("duty", level=level), SPECS)
    assert "also call `notify_owner`" in hooks.before_llm_call(None).message
    assert _outcomes(hooks, "duty") == ["duty_open", "followup"]


@pytest.mark.parametrize("level", ["observe", "remind", "deny"])
def test_guidance_stays_in_the_brief_and_is_not_checked(level):
    hooks = RuleSetHooks.from_package(_package("guidance", level=level), SPECS)
    assert "Be kind." in hooks.before_llm_call(None).message
    assert _outcomes(hooks, "guidance") == []


# -- slots that hold a value alone, or values that are not strings -------------------------------------------------------


@pytest.mark.parametrize(
    "rule, text",
    [
        (
            {"type": "reference_status", "tool": TOOL, "arg": "user", "field": "active", "allowed": [True]},
            "must have active in True",
        ),
        (
            {"type": "allowed_values", "tool": TOOL, "arg": "x", "values": [1, False]},
            "must be one of 1, False",
        ),
        ({"type": "required_args", "tool": TOOL, "args": [1]}, "must include 1"),
        ({"type": "derived_value", "tool": TOOL, "arg": "x", "inputs": [True]}, "computed from True"),
        ({"type": "transition", "tool": TOOL, "arg": "x", "edges": {"a": [True]}}, "a → True"),
        ({"type": "transition", "tool": TOOL, "arg": "x", "edges": {"a": True}}, "a → True"),
        ({"type": "role_permission", "tools": [TOOL], "roles": [True]}, f"roles True may not call {TOOL}"),
        ({"type": "required_args", "tool": TOOL, "args": ["x"], "when": {"on": True}}, "when on is True"),
        (
            {
                "type": "duty",
                "trigger": {"tool": TOOL, "when": {"on": True}},
                "then": {"tool": "n", "when": {"s": True}},
            },
            "when on is True: also call `n` with s=True",
        ),
    ],
)
def test_the_brief_describes_a_rule_with_values_that_are_not_strings(rule, text):
    hooks = RuleSetHooks([{"id": "r1", "source": "§1", **rule}], SPECS, principal=PRINCIPAL)
    assert text in hooks.before_llm_call(None).message


def test_a_write_check_reports_a_value_that_is_not_a_string():
    rule = {
        "id": "r1",
        "type": "reference_status",
        "tool": TOOL,
        "arg": "user",
        "field": "active",
        "allowed": [True],
    }
    hooks = RuleSetHooks([rule], SPECS)
    hooks.after_tool_call("get_user", {}, {"id": "U-1", "active": False})
    assert "allowed: True" in hooks.before_tool_call(TOOL, {"user": "U-1"}).message


def test_a_duty_with_a_value_alone_opens_and_closes():
    rule = {
        "id": "r1",
        "type": "duty",
        "trigger": {"tool": TOOL},
        "then": {"tool": "notify_owner", "when": {"read": True}},
    }
    hooks = RuleSetHooks([rule], SPECS)
    hooks.after_tool_call(TOOL, {}, {"ok": True})
    hooks.after_tool_call("notify_owner", {"read": False}, {"ok": True})
    assert len(hooks.open_duties) == 1
    hooks.after_tool_call("notify_owner", {"read": True}, {"ok": True})
    assert hooks.open_duties == [] and hooks.records[-1]["outcome"] == "duty_done"


# -- step text ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "step, text",
    [
        ("Do it.", "Do it."),
        (Step(text="Do it.", quote="quoted"), "Do it."),
        (Step(quote="quoted", action="mail"), "quoted"),
        (Step(action="mail"), "mail"),
        (Step(), ""),
        ({"text": "Do it."}, "Do it."),
        ({"quote": "quoted", "action": "mail"}, "quoted"),
        ({"quote": "", "action": "mail"}, "mail"),
        ({}, ""),
    ],
)
def test_step_text(step, text):
    assert step_text(step) == text


# -- procedure components ----------------------------------------------------------------------------------------------


def _build_procedures():
    return [
        {
            "id": "p1",
            "name": "Onboard",
            "trigger": {},
            "steps": [
                {
                    "id": "s1",
                    "action": "create_ticket",
                    "required": True,
                    "condition": None,
                    "source": {},
                    "quote": "Create the ONBOARD ticket.",
                }
            ],
            "edges": [],
            "source": {},
            "quote": "",
            "preconditions": [{"condition": "HR approved", "if_not": "stop", "quote": "HR must approve."}],
            "exceptions": [],
            "templates": [
                {"step": 1, "template": "ONBOARD-[Last Name]", "artifact": "the title", "quote": "Title."}
            ],
            "escalations": [],
        },
        {
            "id": "general",
            "name": "General rules",
            "trigger": {},
            "steps": [],
            "edges": [],
            "source": {},
            "quote": "",
            "preconditions": [],
            "exceptions": [{"when": "the case is legal", "then": "stop", "quote": "Legal stops all."}],
            "templates": [],
            "escalations": [],
        },
    ]


def _graph():
    return {
        "id": "SOP-1",
        "version": "1",
        "built_by": "hand",
        "global_rules": [{"src": "§1", "rule": "Be careful.", "attach_when": "ticket"}],
        "routing": {"hr": {"ticket": "HR-[Name]", "assignee": "Ann"}},
        "escalation": {
            "name": "Escalation",
            "src": "§9",
            "steps": ["Call the lead."],
            "must_not": ["Do not reply."],
        },
        "procedures": [
            {
                "id": "1.1",
                "name": "Open a ticket",
                "category": "hr",
                "applies_when": "a request arrives",
                "decisions": [{"id": "D1", "rule": "If urgent", "then": "call", "else": "wait"}],
                "steps": ["Open the ticket with Template 2.", "Tell the requester."],
                "must_not": ["Do not open two tickets."],
                "links": [{"type": "next", "to": "1.2", "when": "after opening"}],
                "src": "§1.1",
            },
            {
                "id": "1.2",
                "name": "Close it",
                "category": "",
                "applies_when": "",
                "steps": ["Close."],
                "src": "§1.2",
            },
        ],
        "templates": {"2": {"number": 2, "name": "Open", "slots": ["Name"], "text": "Hello [Name]"}},
        "definitions": {"case": "one request"},
        "calendar": {"workdays": [0, 1, 2, 3, 4], "holidays": ["2026-12-25"]},
    }


def _texts(store):
    return [store.list_procedures(), store.get_procedure(["1.1"]), store.get_procedure(["1.2", "9.9"])]


def test_procedure_store_from_a_graph_package_reads_the_same_text():
    old, new = ProcedureStore(_graph()), ProcedureStore.from_package(from_store_graph(_graph()))
    assert _texts(new) == _texts(old)
    assert new.required_steps() == old.required_steps()
    assert new.add_days("12/24/2026", 1) == old.add_days("12/24/2026", 1)
    assert "Open the ticket with Template 2." in new.get_procedure(["1.1"])


def test_procedure_store_from_a_build_package_reads_the_step_quotes():
    store = ProcedureStore.from_package(from_build({"procedures": _build_procedures()}))
    assert "- Create the ONBOARD ticket." in store.get_procedure(["p1"])


def test_procedure_dicts_of_a_build_package_are_that_build_s_procedures():
    procedures = _build_procedures()
    assert procedure_dicts(from_build({"build": {"procedures": procedures}})) == procedures
    assert procedure_dicts(from_build({"procedures": procedures, "build": {"version": "1"}})) == procedures


def _guide_run(guide):
    args = {"title": "ONBOARD-Lee"}
    return [guide.before_tool_call("create_ticket", args).message for _ in range(2)] + [
        guide.before_tool_call("send_mail", {"to": "x@y.com"}).message
    ]


def test_procedure_guide_from_a_build_package_stops_as_from_the_build():
    procedures = _build_procedures()
    pkg = from_build({"procedures": copy.deepcopy(procedures)})
    old, new = ProcedureGuide(procedures), ProcedureGuide.from_package(pkg)
    assert new.procs == old.procs and new.general == old.general
    stops = _guide_run(new)
    assert stops == _guide_run(old)
    assert "SOP CHECK" in stops[0] and stops[1] is None and "general rules" in stops[2]


def test_procedure_guide_passes_its_settings():
    guide = ProcedureGuide.from_package(
        from_build({"procedures": _build_procedures()}), gate=False, templates=False
    )
    assert guide.gate_on is False and guide.templates_on is False


def test_procedure_guide_reads_the_text_of_graph_steps_and_has_no_tool_to_stop_on():
    guide = ProcedureGuide.from_package(from_store_graph(_graph()))
    assert "Tell the requester." in guide._text(guide.procs[0])
    assert guide.before_tool_call("create_ticket", {"title": "x"}).message is None


def test_sop_loop_from_a_build_package_has_the_build_s_procedures():
    procedures = _build_procedures()
    model = lambda system, user: "{}"  # noqa: E731
    old = SopLoop(model, procedures, mode="advisory", require_evidence=True)
    new = SopLoop.from_package(
        from_build({"procedures": copy.deepcopy(procedures)}), model, mode="advisory", require_evidence=True
    )
    assert new.model is model and new.mode == "advisory" and new.require_evidence is True
    assert new.procs == old.procs and new.general == old.general


def test_sop_loop_shows_the_text_of_graph_steps():
    loop = SopLoop.from_package(from_store_graph(_graph()), lambda system, user: "{}")
    proc = loop.procs[0]
    assert "Open the ticket with Template 2." in _proc_line(1, proc)
    turn = loop._case_turn(1, 1, {"case": "c"}, proc, "follow", [], [], {}, [])
    assert "1. Open the ticket with Template 2." in turn


def _navigator_build():
    build = copy.deepcopy(NAVIGATOR_BUILD)  # a real build has these; the model requires them
    build["policy_set"]["documents"][0]["kind"] = "sop"
    build["build"]["procedures"][0]["id"] = "p"
    return build


def test_sop_navigator_from_a_build_package_gives_the_same_walk_and_note():
    kwargs = dict(agent_tools=["enable_roaming", "transfer_to_human_agents"], min_chapter_actions=1)
    build = _navigator_build()
    old = SopNavigator(NAVIGATOR_POLICY, build, ask=fake_ask(), **kwargs)
    new = SopNavigator.from_package(from_build(build), NAVIGATOR_POLICY, ask=fake_ask(), **kwargs)
    assert new.ids == old.ids and new.links == old.links and new.exits == old.exits
    assert dict(new.items) == dict(old.items) and new.chapters == old.chapters
    note = new.check_handoff(TURNS)
    assert note and note == old.check_handoff(TURNS)


def test_sop_navigator_from_a_package_with_no_sections_says_so():
    with pytest.raises(ValueError, match="sections"):
        SopNavigator.from_package(from_store_graph(_graph()), NAVIGATOR_POLICY, [], fake_ask())


def gate_model(system, user):
    return (
        '{"breaks": true, "rule": "Do not open two tickets.", "case": "Lee", '
        '"evidence": "the ticket of Lee is already open today", "why": "two"}'
    )


def _no_escalation_graph():
    graph = _graph()
    del graph["escalation"]
    return graph


def test_store_gate_calls_the_model_once_the_agent_opens_a_procedure_on_the_shared_store():
    """The gate sees what the agent opened only through the store behind the agent's tools."""
    prompts = []

    def model(system, user):
        prompts.append(user)
        return gate_model(system, user)

    pkg = from_store_graph(_no_escalation_graph())
    store = ProcedureStore.from_package(pkg)
    gate = StoreGate.from_package(pkg, model, store=store, today="2026-11-05")
    assert gate.store is store and gate.today.isoformat() == "2026-11-05"
    gate.after_tool_call("get_ticket", {}, "the ticket of Lee is already open today")
    assert gate.before_tool_call("create_ticket", {"title": "Lee"}) is None
    assert prompts == []  # nothing opened yet: nothing to check

    store.get_procedure(["1.1"])
    assert "Blocked by an SOP check" in gate.before_tool_call("create_ticket", {"title": "Lee"})
    assert len(prompts) == 1 and "[1.1] MUST NOT: Do not open two tickets." in prompts[0]


def test_store_gate_needs_a_store_or_sop_text():
    pkg = from_store_graph(_graph())
    with pytest.raises(ValueError, match="store"):
        StoreGate.from_package(pkg, gate_model)
    with pytest.raises(ValueError, match="one of them"):
        StoreGate.from_package(
            pkg, gate_model, store=ProcedureStore.from_package(pkg), sop_text="The whole SOP."
        )


def test_store_gate_raises_when_the_package_has_nothing_to_check():
    build_package = from_build({"procedures": _build_procedures()})
    store = ProcedureStore.from_package(build_package)
    with pytest.raises(ValueError, match="must-not"):
        StoreGate.from_package(build_package, gate_model, store=store)


@pytest.mark.parametrize(
    "change",
    [
        lambda g: g["procedures"][0].pop("decisions"),
        lambda g: g["procedures"][0].pop("must_not"),
        lambda g: (g["procedures"][0].pop("decisions"), g["procedures"][0].pop("must_not")),
    ],
)
def test_store_gate_builds_from_any_one_kind_of_rule(change):
    graph = _graph()
    change(graph)
    pkg = from_store_graph(graph)
    assert StoreGate.from_package(pkg, gate_model, store=ProcedureStore.from_package(pkg))


def test_store_gate_with_a_package_with_nothing_to_check_can_use_sop_text():
    gate = StoreGate.from_package(
        from_build({"procedures": _build_procedures()}), gate_model, sop_text="The SOP."
    )
    assert gate.store is None and gate._rules() == ["The SOP."]


def test_store_gate_rules_are_the_same_as_from_the_graph():
    pkg = from_store_graph(_graph())
    old = StoreGate(ProcedureStore(_graph()), gate_model)
    new = StoreGate.from_package(pkg, gate_model, store=ProcedureStore.from_package(pkg))
    for gate in (old, new):
        gate.store.get_procedure(["1.1"])
    assert new._rules() == old._rules()


def test_store_graph_round_trip_is_unchanged():
    assert to_store_graph(from_store_graph(_graph())) == _graph()
