"""Policy package: per-format round trip, rule class and level, real fields, and old JSON that must keep loading."""

import copy
import json

import pytest

from polisys.model import (
    Obligation,
    PolicyBuild,
    PolicyPackage,
    PolicySet,
    Procedure,
    Rule,
    Statement,
    Step,
    StatementType,
    from_json,
    to_json,
)
from polisys.package import (
    from_build,
    from_store_graph,
    load_package,
    rule_class,
    rule_level,
    save_package,
    to_build,
    to_store_graph,
)
from polisys.procedure_store import ProcedureStore
from polisys.ruleset import RuleSetHooks

SOURCE = {"document": "policy-abc", "section": "4.2", "page": None, "line": 65}


def _statement(i: int = 1) -> dict:
    return {
        "id": f"policy-abc:4.2:{i}",
        "text": "Urgency must be low, medium or high.",
        "types": ["allowed_values"],
        "modality": "must",
        "source": SOURCE,
        "quote": "Urgency must be low, medium or high.",
    }


def _policy_set(links: list) -> dict:
    section = {
        "id": "4.2",
        "title": "Priority",
        "page": None,
        "parent": "4",
        "statements": [_statement()],
        "source": {"document": "policy-abc", "section": "4.2"},
        "quote": "",
    }
    document = {
        "id": "policy-abc",
        "kind": "html",
        "title": "Service policy",
        "trust": "trusted",
        "effective_date": None,
        "authority_rank": 0,
        "sections": [section],
        "structure": {"headings": [{"pattern": "^#", "level": 1, "kind": "heading"}]},
        "source": {"file": "policy.html"},
        "quote": "Service policy",
    }
    return {"documents": [document], "links": links, "source": {}, "quote": ""}


def _wrapped(rule_id: str, rule_type: str, may_deny: bool = True, checker: str = "code", **slots) -> dict:
    inner = {
        "id": rule_id,
        "type": rule_type,
        "source": "§4.2 L65",
        "quote": "a quote",
        "by": "model",
        **slots,
    }
    return {
        "id": rule_id,
        "type": rule_type,
        "rule": inner,
        "statement_ids": ["policy-abc:4.2:1"],
        "checker": checker,
        "may_deny": may_deny,
        "source": SOURCE,
        "quote": "a quote",
    }


LINK = {
    "type": "refers_to",
    "src": "policy-abc:4.2:1",
    "dst": "policy-abc:3.1:2",
    "source": SOURCE,
    "quote": "see section 3.1",
}
DEFINITION = {
    "term": "Major incident",
    "meaning": "an outage of a core service",
    "used_by": [],
    "source": SOURCE,
    "quote": "Major incident: an outage of a core service",
}
COVERAGE = {
    "statement_id": "policy-abc:4.2:1",
    "kind": "rule",
    "reason": "read by code from a table or list",
    "source": SOURCE,
    "quote": "Urgency must be low, medium or high.",
}
VALIDATION = {"statement_id": "policy-abc:4.2:1", "code": "schema_conflict", "message": "tool rejects 'low'"}
GENERATOR = {"kind": "pipeline", "harness": "builder", "model": "some/model"}


def runner_with_flat_rules() -> dict:
    """Runner build file whose top-level rules differ from ``build.rules``: repair added and dropped rules."""
    kept = _wrapped("r_kept", "allowed_values", tool="ticket__create", arg="urgency", values=["low", "high"])
    soft = _wrapped(
        "r_soft", "allowed_values", may_deny=False, tool="ticket__create", arg="state", schema_conflict=True
    )
    soft["rule"]["may_deny"] = False
    dropped = _wrapped("r_dropped", "verify", checker="verifier", statement="An old check", tools=[])
    flat_new = {
        "id": "grant_1",
        "type": "role_permission",
        "tool": "ticket__close",
        "roles": ["manager"],
        "from_grant": True,
        "may_deny": True,
        "statement_ids": ["policy-abc:9:1"],
        "source": "§9",
        "quote": "Only managers may close.",
    }
    return copy.deepcopy(
        {
            "policy_hash": "policy-one",
            "rules": [copy.deepcopy(kept["rule"]), copy.deepcopy(soft["rule"]), flat_new],
            "procedures": [],
            "build": {
                "version": "version-of-runner",
                "rules": [kept, soft, dropped],
                "procedures": [],
                "links": [],
                "definitions": [DEFINITION],
                "coverage": [COVERAGE],
                "validation": [VALIDATION],
                "generator": GENERATOR,
                "stats": {"documents": 1, "rules": 3, "by_type": {"allowed_values": 2, "verify": 1}},
            },
            "policy_set": _policy_set([]),
            "repair": {"verify_tools_added": 2, "grant_rules": 1},
            "repair2": {"verify_retyped": 1},
        }
    )


def runner_with_nested_build() -> dict:
    """Runner build file with no flat rules: rules and procedures only in ``build``, and its own ``stats`` at the top."""
    step = {
        "id": "policy-abc:11.1:3",
        "action": "ticket__create",
        "required": True,
        "condition": None,
        "source": SOURCE,
        "quote": "Step 1: Create a ticket ONBOARD-[Last Name].",
    }
    procedure = {
        "id": "policy-abc:11.1",
        "name": "Receiving a request",
        "trigger": {"tools": ["ticket__create"]},
        "steps": [step, {**step, "id": "policy-abc:11.1:5", "action": "", "condition": "if new"}],
        "edges": [{"from": "policy-abc:11.1:3", "to": "policy-abc:11.1:5"}],
        "source": {"document": "policy-abc", "section": "11.1"},
        "quote": "",
        "preconditions": [{"condition": "a manager confirmed", "if_not": "stop", "quote": "q"}],
        "exceptions": [{"when": "emergency", "then": "skip step 2", "quote": "q", "from_section": "2"}],
        "templates": [{"step": 1, "artifact": "ticket", "template": "ONBOARD-[Last Name]", "quote": "q"}],
        "escalations": [{"when": "no reply", "notify": "owner", "channel": "#help", "quote": "q"}],
    }
    return copy.deepcopy(
        {
            "stats": {"sop": "policy.html", "sections": 2, "seconds": 3},
            "build": {
                "version": "version-of-nested",
                "rules": [
                    _wrapped("g1", "guidance", checker="brief", text="Be careful.", tools=[]),
                    _wrapped(
                        "v1", "verify", checker="verifier", statement="A check", tools=["ticket__create"]
                    ),
                    _wrapped(
                        "d1",
                        "duty",
                        checker="monitor",
                        trigger={"tool": "ticket__create"},
                        then={"tool": "mail"},
                    ),
                ],
                "procedures": [procedure],
                "links": [LINK],
                "definitions": [DEFINITION],
                "coverage": [COVERAGE],
                "validation": [],
                "generator": GENERATOR,
                "stats": {"documents": 1, "rules": 3, "model": "some/model", "usage": {"calls": 4}},
            },
            "policy_set": _policy_set([LINK]),
        }
    )


def builder_output() -> dict:
    """``to_json(PolicyBuild)``: a skill build, one rule remind-only by its wrapper alone."""
    return copy.deepcopy(
        {
            "version": "version-of-builder",
            "rules": [
                _wrapped("g_1", "allowed_values", tool="ticket__create", arg="urgency", values=["low"]),
                _wrapped("g_2", "record_condition", may_deny=False, tool="ticket__close", field="state"),
            ],
            "procedures": [],
            "links": [LINK],
            "definitions": [],
            "coverage": [COVERAGE],
            "validation": [],
            "generator": {"kind": "skill", "harness": "claude-code", "model": "a person reviewed it"},
            "stats": {
                "documents": 1,
                "rules": 2,
                "links_by_type": {"refers_to": 1},
                "documents_by_trust": {"trusted": 1},
            },
        }
    )


def store_graph() -> dict:
    return {
        "id": "Policy_ABC",
        "version": "by_hand_v1",
        "built_by": "A person, from the SOP text only",
        "global_rules": [
            {"id": "G-scope", "src": "§1", "rule": "Do not deviate from the SOP."},
            {"id": "G-docs", "src": "§2", "attach_when": "Word document", "rule": "Use the template."},
        ],
        "triage": {
            "id": "triage",
            "src": "§3",
            "name": "Daily triage",
            "steps": ["Read the inbox."],
            "team": {"a": "b"},
        },
        "routing": {
            "Onboarding": {"jira": "ONBOARD-[Name]", "assignee": "Coordinator", "folder": "Onboarding"}
        },
        "housekeeping": {"id": "housekeeping", "src": "§14", "name": "Files", "steps": ["File everything."]},
        "priority": {
            "id": "priority",
            "src": "§4",
            "name": "Priority",
            "levels": ["P1: today", "P2: this week"],
        },
        "escalation": {
            "id": "escalation",
            "src": "§15",
            "name": "Escalate",
            "steps": ["Post."],
            "must_not": ["Wait."],
        },
        "procedures": [
            {
                "id": "11.1",
                "name": "Receiving a notification",
                "category": "Onboarding",
                "applies_when": "A manager confirms a new hire.",
                "inputs": [{"fact": "start date", "source": "the notification"}],
                "decisions": [{"id": "npi", "rule": "NPI is present", "then": "step 2", "else": "stop"}],
                "steps": ["1. Create ONBOARD-[Name].", "2. Email Template 1 to the employee.", ""],
                "must_not": ["Do not send before approval."],
                "links": [{"type": "next", "to": "11.2", "when": "the hire is confirmed"}],
                "deadlines": ["Within 3 business days."],
                "notes": ["P3 if the start date is far away."],
                "src": "§11.1",
            },
            {
                "id": "11.2",
                "name": "Setting up",
                "category": "",
                "applies_when": "",
                "inputs": [],
                "decisions": [],
                "steps": [],
                "must_not": [],
                "links": [],
                "deadlines": [],
                "notes": [],
                "src": "§11.2",
            },
        ],
        "templates": {
            "1": {"number": 1, "name": "Welcome", "text": "Hi [Name],", "slots": ["Name"]},
            "A-1": {"number": "A-1", "name": "Payment", "text": "Dear [Vendor],", "slots": ["Vendor"]},
        },
        "definitions": {"Probation": "the first 90 days", "DOS": "date of service"},
        "calendar": {"workdays": [0, 1, 2, 3, 4], "holidays": ["2026-01-01"]},
    }


def smaller_store_graph() -> dict:
    """A graph with several sections left out and its keys in another order."""
    g = store_graph()
    return {
        k: g[k]
        for k in (
            "id",
            "version",
            "built_by",
            "global_rules",
            "procedures",
            "templates",
            "routing",
            "calendar",
        )
    }


BUILDS = [runner_with_flat_rules, runner_with_nested_build, builder_output]
GRAPHS = [store_graph, smaller_store_graph]


def _same(a: dict, b: dict) -> None:
    """Equal, and with keys in the same order: ``ProcedureStore`` runs regexes on ``json.dumps`` of a procedure."""
    assert a == b
    assert json.dumps(a) == json.dumps(b)


# -- round trip --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("make", BUILDS)
def test_build_to_package_to_build_is_equal(make):
    _same(to_build(from_build(make())), make())


@pytest.mark.parametrize("make", GRAPHS)
def test_store_graph_to_package_to_store_graph_is_equal(make):
    _same(to_store_graph(from_store_graph(make())), make())


@pytest.mark.parametrize(
    "make, read, write",
    [(m, from_build, to_build) for m in BUILDS] + [(m, from_store_graph, to_store_graph) for m in GRAPHS],
)
def test_round_trip_through_a_file(tmp_path, make, read, write):
    path = tmp_path / "a" / "package.json"
    save_package(read(make()), path)
    loaded = load_package(path)
    _same(write(loaded), make())
    assert loaded == read(make())


def test_converters_do_not_share_data_with_their_input():
    source = store_graph()
    pkg = from_store_graph(source)
    pkg.global_["calendar"]["holidays"].append("2030-01-01")
    pkg.procedures[0].must_not.append("x")
    assert source == store_graph()
    out = to_store_graph(from_store_graph(source))
    out["procedures"][0]["must_not"].append("x")
    assert source == store_graph()
    build = to_build(from_build(runner_with_nested_build()))
    assert build["policy_set"]["links"] is not build["build"]["links"]


# -- which rules, and what they carry ----------------------------------------------------------------------------------


def test_top_level_rules_win_over_build_rules():
    pkg = from_build(runner_with_flat_rules())
    assert [r.id for r in pkg.rules] == ["r_kept", "r_soft", "grant_1"]
    assert [r["id"] for r in to_build(pkg)["build"]["rules"]] == ["r_kept", "r_soft", "r_dropped"]


def test_flat_rule_takes_missing_fields_from_the_wrapped_rule_of_the_same_id():
    source = runner_with_flat_rules()
    kept, _, new = from_build(source).rules
    assert kept.rule == source["rules"][0]
    assert "statement_ids" not in kept.rule
    assert kept.statement_ids == ["policy-abc:4.2:1"]
    assert kept.source == SOURCE
    assert kept.quote == "a quote"
    assert new.statement_ids == ["policy-abc:9:1"]
    assert new.checker == "cedar"
    assert new.source == {}


def test_builder_output_rules_stay_wrapped():
    pkg = from_build(builder_output())
    assert pkg.rules[0].rule["tool"] == "ticket__create"
    assert pkg.rules[0].source == SOURCE


def test_nested_build_gives_real_procedure_fields():
    p = from_build(runner_with_nested_build()).procedures[0]
    assert p.trigger == {"tools": ["ticket__create"]}
    assert p.steps[0].action == "ticket__create"
    assert p.steps[0].text == ""
    assert p.steps[1].condition == "if new"
    assert p.preconditions[0]["if_not"] == "stop"
    assert p.exceptions and p.templates and p.escalations
    assert p.extra == {}


def test_other_fields_of_a_build_stay_at_their_level():
    pkg = from_build(runner_with_flat_rules())
    assert pkg.policy_hash == "policy-one"
    assert pkg.version == "version-of-runner"
    assert pkg.generator == GENERATOR
    assert pkg.extra["repair"] == {"verify_tools_added": 2, "grant_rules": 1}
    assert pkg.global_["definitions"] == [DEFINITION]
    assert len(pkg.documents) == 1 and pkg.documents[0].statements[0].types == [StatementType.ALLOWED_VALUES]
    assert pkg.coverage[0].kind == "rule"
    assert pkg.validation == [VALIDATION]
    assert pkg.obligations == []
    assert pkg.links == []
    assert from_build(builder_output()).extra["stats"]["rules"] == 2
    assert from_build(runner_with_nested_build()).extra["stats"]["sop"] == "policy.html"


# -- may_deny, class, level --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule_type, cls",
    [
        ("allowed_values", "hard"),
        ("role_permission", "hard"),
        ("grounded_values", "hard"),
        ("duty", "hard"),
        ("verify", "judgment"),
        ("guidance", "guidance"),
    ],
)
def test_class_from_type(rule_type, cls):
    assert rule_class(rule_type) == cls
    assert from_build({"rules": [_wrapped("x", rule_type)]}).rules[0].cls == cls
    assert from_build({"rules": [{"id": "x", "type": rule_type}]}).rules[0].cls == cls


def _level_today(rule: dict) -> str:
    """What ``RuleSetHooks`` does with a runtime rule when no method choice applies."""
    if rule["type"] == "guidance":
        return "observe"  # only in the brief
    if rule["type"] == "duty":
        return "remind"  # a follow-up after the trigger call, never a denial
    hooks = RuleSetHooks([], [])
    if hooks._observe_only(rule):
        return "observe"
    return "deny" if hooks._may_deny(rule) else "remind"


LEVELS = [
    ("allowed_values", {}, "deny"),
    ("allowed_values", {"may_deny": True}, "deny"),
    ("allowed_values", {"may_deny": False}, "remind"),
    ("allowed_values", {"schema_conflict": True}, "observe"),
    ("allowed_values", {"may_deny": False, "schema_conflict": True}, "observe"),
    ("required_args", {"may_deny": False}, "remind"),
    ("role_permission", {}, "deny"),
    ("grounded_values", {}, "observe"),
    ("grounded_values", {"may_deny": False}, "observe"),
    ("grounded_values", {"may_deny": True}, "deny"),
    ("id_provenance", {}, "observe"),
    ("id_provenance", {"may_deny": False}, "observe"),
    ("id_provenance", {"may_deny": True}, "deny"),
    ("verify", {}, "remind"),
    ("verify", {"may_deny": True}, "remind"),
    ("verify", {"may_deny": False}, "remind"),
    ("duty", {}, "remind"),
    ("duty", {"may_deny": False}, "remind"),
    ("guidance", {}, "observe"),
    ("guidance", {"may_deny": False}, "observe"),
]


@pytest.mark.parametrize("rule_type, state, level", LEVELS)
def test_level_of_a_flat_rule_is_what_the_runtime_does(rule_type, state, level):
    raw = {"id": "x", "type": rule_type, **state}
    assert from_build({"rules": [raw]}).rules[0].level == level
    assert _level_today(raw) == level
    assert to_build(from_build({"rules": [raw]}))["rules"][0] == raw


@pytest.mark.parametrize("rule_type, state, level", LEVELS)
def test_level_of_a_wrapped_rule_reads_the_rule_dict(rule_type, state, level):
    """The wrapper says ``may_deny: true`` on nearly every rule; the runtime sees only the inner dict."""
    raw = _wrapped("x", rule_type)
    raw["rule"].update(state)
    assert from_build({"rules": [raw]}).rules[0].level == level
    assert _level_today(raw["rule"]) == level


@pytest.mark.parametrize(
    "rule_type, inner, level",
    [
        ("allowed_values", {}, "remind"),
        ("grounded_values", {}, "observe"),
        ("grounded_values", {"may_deny": True}, "observe"),
        ("guidance", {}, "observe"),
    ],
)
def test_level_when_only_the_wrapper_says_may_deny_false(rule_type, inner, level):
    """``rule_audit.load_rules`` copies a false wrapper into the runtime dict."""
    raw = _wrapped("x", rule_type, may_deny=False)
    raw["rule"].update(inner)
    assert from_build({"rules": [raw]}).rules[0].level == level
    assert _level_today({**raw["rule"], "may_deny": False}) == level


def test_rule_level_called_directly():
    assert rule_level("format") == "deny"
    assert rule_level("format", False) == "remind"
    assert rule_level("format", True, {"schema_conflict": True}) == "observe"
    assert rule_level("id_provenance", True, {"may_deny": True}) == "deny"
    assert rule_level("id_provenance") == "observe"


def test_may_deny_false_only_inside_the_rule_dict_also_reminds():
    raw = _wrapped("x", "format", may_deny=True)
    raw["rule"]["may_deny"] = False
    pkg = from_build({"rules": [raw]})
    assert pkg.rules[0].level == "remind"
    assert to_build(pkg)["rules"][0] == raw


@pytest.mark.parametrize("state", [{}, {"may_deny": True}, {"may_deny": False}])
def test_may_deny_state_of_a_flat_rule_is_restored_exactly(state):
    """ruleset._may_deny treats absent and True differently, for example on grounded_values."""
    raw = {"id": "x", "type": "grounded_values", **state}
    assert to_build(from_build({"rules": [raw]}))["rules"][0] == raw


def test_may_deny_absent_from_a_wrapped_rule_stays_absent():
    raw = _wrapped("x", "format")
    del raw["may_deny"]
    pkg = from_build({"rules": [raw]})
    assert pkg.rules[0].level == "deny"
    assert to_build(pkg)["rules"][0] == raw


# -- store graph fields ------------------------------------------------------------------------------------------------


def test_store_graph_gives_real_fields():
    pkg = from_store_graph(store_graph())
    p = pkg.procedures[0]
    assert p.applies_when == "A manager confirms a new hire."
    assert p.inputs[0]["fact"] == "start date"
    assert p.decisions[0]["else"] == "stop"
    assert p.steps[0] == Step(text="1. Create ONBOARD-[Name].")
    assert p.must_not == ["Do not send before approval."]
    assert p.links[0]["to"] == "11.2"
    assert p.deadlines == ["Within 3 business days."]
    assert p.notes == ["P3 if the start date is far away."]
    assert p.category == "Onboarding"
    assert p.src == "§11.1"
    assert p.extra == {}
    assert pkg.version == "by_hand_v1"
    assert pkg.generator == {"built_by": "A person, from the SOP text only"}
    assert pkg.extra["id"] == "Policy_ABC"
    assert pkg.rules == [] and pkg.obligations == []
    assert pkg.global_["templates"]["A-1"]["name"] == "Payment"
    assert pkg.global_["definitions"][0] == {"term": "Probation", "meaning": "the first 90 days"}
    assert set(pkg.global_) >= {
        "global_rules",
        "triage",
        "routing",
        "housekeeping",
        "escalation",
        "priority",
        "calendar",
    }


def test_procedure_store_reads_the_same_text_after_a_round_trip():
    graph = store_graph()
    before, after = ProcedureStore(graph), ProcedureStore(to_store_graph(from_store_graph(graph)))
    assert after.list_procedures() == before.list_procedures()
    assert after.get_procedure(["11.1"]) == before.get_procedure(["11.1"])


def test_statements_become_documents_and_leave_the_graph_as_it_was():
    statements = [
        {**_statement(1), "section": "4.2", "title": "Priority"},
        Statement(id="s2", text="Close within a day.", source={"section": "5"}),
    ]
    pkg = from_store_graph(store_graph(), statements)
    assert [s.id for s in pkg.documents[0].sections] == ["4.2", "5"]
    assert pkg.documents[0].sections[0].title == "Priority"
    assert pkg.documents[0].statements[1].text == "Close within a day."
    assert from_store_graph(store_graph()).documents == []
    _same(to_store_graph(pkg), store_graph())


# -- what the model does not have --------------------------------------------------------------------------------------


def test_unknown_fields_stay_at_their_level_and_return():
    graph = store_graph()
    graph["future"] = {"a": 1}
    graph["procedures"][0]["owner"] = "HR"
    graph["procedures"][0]["priority"] = "P1"
    pkg = from_store_graph(graph)
    assert pkg.extra["future"] == {"a": 1}
    assert pkg.procedures[0].extra["owner"] == "HR"
    assert pkg.procedures[0].priority == "P1"
    _same(to_store_graph(pkg), graph)

    build = runner_with_nested_build()
    build["build"]["future"] = 1
    build["policy_set"]["future"] = 2
    build["policy_set"]["links"] = []
    document = build["policy_set"]["documents"][0]
    document["future"] = 3
    document["sections"][0]["future"] = 4
    document["sections"][0]["statements"][0]["future"] = 5
    build["build"]["rules"][0]["future"] = 6
    build["build"]["procedures"][0]["future"] = 7
    build["build"]["procedures"][0]["steps"][0]["future"] = 8
    build["build"]["links"][0]["future"] = 9
    build["build"]["coverage"][0]["future"] = 10
    pkg = from_build(build)
    assert pkg.documents[0].extra["future"] == 3
    assert pkg.documents[0].statements[0].extra["future"] == 5
    assert pkg.rules[0].extra["future"] == 6
    assert pkg.procedures[0].extra["future"] == 7
    assert pkg.procedures[0].steps[0].extra["future"] == 8
    _same(to_build(pkg), build)


def test_absent_keys_stay_absent():
    graph = store_graph()
    for key in ("notes", "deadlines", "category"):
        del graph["procedures"][0][key]
    graph["procedures"][1]["notes"] = []
    _same(to_store_graph(from_store_graph(graph)), graph)

    build = runner_with_nested_build()
    del build["policy_set"]["documents"][0]["structure"]
    del build["policy_set"]["documents"][0]["sections"][0]["page"]
    del build["build"]["procedures"][0]["templates"]
    del build["build"]["procedures"][0]["steps"][0]["condition"]
    del build["build"]["coverage"][0]["reason"]
    del build["build"]["stats"]
    _same(to_build(from_build(build)), build)


def test_a_dict_step_in_a_graph_stays_a_dict():
    graph = store_graph()
    graph["procedures"][0]["steps"].append(
        {"id": "s", "action": "mail", "required": True, "condition": None, "source": {}, "quote": ""}
    )
    graph["procedures"][0]["steps"].append({"text": "Only text"})
    pkg = from_store_graph(graph)
    assert pkg.procedures[0].steps[-2].action == "mail"
    _same(to_store_graph(pkg), graph)


# -- package JSON ------------------------------------------------------------------------------------------------------


def test_package_json_keys(tmp_path):
    path = tmp_path / "package.json"
    save_package(from_store_graph(store_graph()), path)
    data = json.loads(path.read_text())
    assert data["schema"] == "policy-package/1"
    assert "global" in data and "global_" not in data
    for key in (
        "obligations",
        "policy_questions",
        "tests",
        "coverage",
        "validation",
        "links",
        "rules",
        "extra",
    ):
        assert key in data

    save_package(from_build(builder_output()), path)
    rule = json.loads(path.read_text())["rules"][0]
    assert rule["class"] == "hard" and rule["level"] == "deny" and "cls" not in rule


def test_load_package_refuses_another_schema(tmp_path):
    path = tmp_path / "package.json"
    path.write_text(json.dumps({"schema": "policy-package/2"}))
    with pytest.raises(ValueError, match="policy-package/1"):
        load_package(path)


def test_obligation_round_trip():
    ob = Obligation(
        id="ob_12.1_3",
        origin="package",
        scope={"procedure": "12.1", "when": {"case_decision": "proceed"}},
        effect={
            "kind": "write",
            "tools": ["ticket__create"],
            "match": {"summary": {"pattern": "OFF-{last_name}"}},
        },
        values={
            "last_name": {"from": "case_fact", "field": "employee_last_name"},
            "date": {"from": "computed"},
        },
        check="code",
        level="follow_up",
        source={"section": "12.1", "quote": "Step 3: Create a ticket."},
    )
    pkg = PolicyPackage(obligations=[ob])
    back = from_json(PolicyPackage, to_json(pkg))
    assert back.obligations == [ob]
    assert {f for f in to_json(ob)} == {
        "id",
        "origin",
        "scope",
        "effect",
        "values",
        "check",
        "level",
        "source",
    }


def test_a_package_can_be_written_to_the_other_format():
    """Not a round trip: the usual shape of the target, with what the source had that the target can carry."""
    graph_pkg = from_store_graph(store_graph())
    build = to_build(graph_pkg)
    assert list(build) == [
        "version",
        "rules",
        "procedures",
        "links",
        "definitions",
        "coverage",
        "validation",
        "generator",
        "stats",
    ]
    assert build["procedures"][0]["steps"][0]["text"] == "1. Create ONBOARD-[Name]."
    assert build["procedures"][0]["must_not"] == ["Do not send before approval."]

    build_pkg = from_build(runner_with_nested_build())
    graph = to_store_graph(build_pkg)
    assert graph["id"] == "policy-abc"
    assert graph["procedures"][0]["steps"][0] == "Step 1: Create a ticket ONBOARD-[Last Name]."
    assert "rules" not in graph
    assert ProcedureStore(graph).list_procedures()

    new = PolicyPackage(
        version="v1", procedures=[Procedure(id="p", name="P", steps=[Step(text="Do it", action="mail")])]
    )
    assert to_store_graph(new)["procedures"][0]["steps"] == ["Do it"]
    assert to_build(new)["procedures"][0]["steps"][0]["action"] == "mail"


# -- old JSON ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("make", [lambda: runner_with_nested_build()["build"], builder_output])
def test_old_build_json_loads_and_writes_as_before(make):
    """Readers of ``to_json(build)`` must see no new keys."""
    assert to_json(from_json(PolicyBuild, make())) == make()


def test_old_policy_set_json_loads_and_writes_as_before():
    assert to_json(from_json(PolicySet, _policy_set([LINK]))) == _policy_set([LINK])


def test_step_stays_usable_as_before_and_from_a_string():
    assert Step("s1", "mail") == Step(id="s1", action="mail")
    assert to_json(Step("s1", "mail")) == {
        "id": "s1",
        "action": "mail",
        "required": True,
        "condition": None,
        "source": {},
        "quote": "",
    }
    p = from_json(Procedure, {"id": "1", "name": "n", "steps": ["Do it", {"id": "s", "action": "mail"}]})
    assert p.steps == [Step(text="Do it"), Step(id="s", action="mail")]


def test_new_rule_fields_are_left_out_of_json_when_empty():
    assert set(to_json(Rule(id="r", type="format", rule={}))) == {
        "id",
        "type",
        "rule",
        "statement_ids",
        "checker",
        "may_deny",
        "source",
        "quote",
    }


def test_policy_questions_of_a_build_come_back_and_an_old_build_is_unchanged():
    question = {"statement_ids": ["s1"], "question": "Is it guidance?", "default_level": "observe"}
    old = {"policy_hash": "h", "rules": [], "build": {"version": "v", "rules": [], "coverage": []}}
    assert (
        from_build(old).policy_questions == [] and to_build(from_build(old)) == old
    )  # the key is not written
    for questions in ([question], []):  # an empty list that the source had is kept
        new = {
            "policy_hash": "h",
            "rules": [],
            "build": {"version": "v", "rules": [], "coverage": [], "policy_questions": questions},
        }
        pkg = from_build(new)
        assert pkg.policy_questions == questions and to_build(pkg) == new
    builder_output = {"version": "v", "rules": [], "coverage": [], "policy_questions": [question]}
    assert to_build(from_build(builder_output)) == builder_output
