"""Build and package validators: one error code per kind of problem, the build-twice diff, conflict resolution."""

import copy
import json

import pytest

from polisys import cli
from polisys.documents import load_document
from polisys.model import (
    CoverageItem,
    Document,
    Link,
    LinkType,
    Obligation,
    PolicyBuild,
    PolicyPackage,
    PolicySet,
    Procedure,
    RULE_CLASSES,
    Rule,
    Section,
    Statement,
    Step,
    Trust,
    to_json,
)
from polisys.package import save_package
from polisys.validate import diff_builds, resolve_conflicts, validate_build, validate_package

SPECS = [
    {
        "name": "update_ticket",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["open", "closed"]},
                "code": {"type": "string"},
            },
        },
    }
]


def _ps():
    return PolicySet(
        documents=[
            load_document(
                "## 1. Tickets\n\n- Status must be open or closed.\n- Codes look like T-123.\n",
                meta={"id": "d"},
            )
        ]
    )


def _rule(rid, raw, sid="d:1:1"):
    return Rule(id=rid, type=raw["type"], rule=raw, statement_ids=[sid])


def _codes(build, ps=None):
    return sorted(e.code for e in validate_build(build, SPECS, ps))


def test_each_error_code():
    ps = _ps()
    q = "Status must be open or closed"
    cases = {
        "unknown_tool": {
            "type": "allowed_values",
            "tool": "nope",
            "arg": "status",
            "values": ["open"],
            "quote": q,
        },
        "unknown_arg": {
            "type": "allowed_values",
            "tool": "update_ticket",
            "arg": "nope",
            "values": ["open"],
            "quote": q,
        },
        "value_not_in_enum": {
            "type": "allowed_values",
            "tool": "update_ticket",
            "arg": "status",
            "values": ["gone"],
            "quote": q,
        },
        "quote_not_verbatim": {
            "type": "allowed_values",
            "tool": "update_ticket",
            "arg": "status",
            "values": ["open"],
            "quote": "made up",
        },
        "bad_pattern": {
            "type": "format",
            "tool": "update_ticket",
            "arg": "code",
            "pattern": "T-[",
            "quote": "Codes look like T-123",
        },
    }
    for code, raw in cases.items():
        sid = "d:1:2" if raw["type"] == "format" else "d:1:1"
        build = PolicyBuild(
            version="v",
            rules=[_rule("r", raw, sid)],
            coverage=[CoverageItem("d:1:1", "rule"), CoverageItem("d:1:2", "rule")],
        )
        assert _codes(build, ps) == [code], code
    build = PolicyBuild(
        version="v",
        links=[Link(type=LinkType.REFERS_TO, src="d:1:1", dst="d:9")],
        coverage=[CoverageItem("d:1:1", "rule"), CoverageItem("d:1:2", "rule")],
    )
    assert _codes(build, ps) == ["link_target_missing"]
    assert _codes(PolicyBuild(version="v", coverage=[CoverageItem("d:1:1", "rule")]), ps) == [
        "statement_uncovered"
    ]


def test_diff_builds_flags_changed_statement():
    a = PolicyBuild(
        version="v", rules=[_rule("1", {"type": "format", "tool": "t", "arg": "code", "pattern": "T-\\d+"})]
    )
    b = PolicyBuild(
        version="v", rules=[_rule("2", {"type": "format", "tool": "t", "arg": "code", "pattern": "T-\\d{3}"})]
    )
    assert diff_builds(a, a) == [] and diff_builds(a, b) == ["d:1:1"]


# -- policy packages ---------------------------------------------------------------------------------------------------

TOOLS = [
    {
        "name": "update_ticket",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["open", "closed"]},
                "code": {"type": "string"},
            },
        },
    },
    {
        "name": "create_ticket",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["open", "closed"]},
                "summary": {"type": "string"},
            },
        },
    },
]
TEXT = "Status must be open or closed."


def _doc(doc_id, *statements, **kw):
    """A document whose statements are (text, modality) pairs, numbered from 1."""
    sts = [
        Statement(id=f"{doc_id}:1:{n}", text=text, modality=modality, source={"section": "1"})
        for n, (text, modality) in enumerate(statements, 1)
    ]
    return Document(id=doc_id, kind="policy", sections=[Section(id="1", statements=sts)], **kw)


def _allowed(rid, sid, values, when=None, text=TEXT, tool="update_ticket", arg="status"):
    rule = {"id": rid, "type": "allowed_values", "tool": tool, "arg": arg, "values": values, "quote": text}
    if when is not None:
        rule["when"] = when
    return Rule(id=rid, type="allowed_values", rule=rule, statement_ids=[sid], cls="hard", level="deny")


def _raw(rid, rule, sids=(), cls="hard", level="deny"):
    rule = {"id": rid, "quote": TEXT, **rule}
    return Rule(id=rid, type=rule["type"], rule=rule, statement_ids=list(sids), cls=cls, level=level)


def _clean_package():
    doc = _doc("d", (TEXT, "must"))
    return PolicyPackage(version="v", documents=[doc], rules=[_allowed("r1", "d:1:1", ["open", "closed"])])


def _findings(pkg, specs=TOOLS, legacy=False):
    return sorted((e.code, e.severity) for e in validate_package(pkg, specs, legacy=legacy))


def _one(rule, specs=TOOLS):
    """The findings of a package that holds only this rule (legacy, so that having no statements is not a finding)."""
    return validate_package(PolicyPackage(version="v", rules=[rule]), specs, legacy=True)


def test_a_complete_package_has_no_findings():
    assert validate_package(_clean_package(), TOOLS) == []


def test_validate_package_only_reports():
    pkg = _clean_package()
    pkg.rules.append(_raw("bad", {"type": "required_args", "tool": "update_ticket", "args": "code"}))
    before = to_json(pkg)
    validate_package(pkg, TOOLS)
    assert to_json(pkg) == before


@pytest.mark.parametrize(
    "rule, where",
    [
        ({"type": "allowed_values", "tool": "update_ticket", "arg": "status", "values": "open"}, "values"),
        ({"type": "allowed_values", "tool": "update_ticket", "arg": "status", "values": 5}, "values"),
        ({"type": "required_args", "tool": "update_ticket", "args": "code"}, "args"),
        (
            {
                "type": "reference_status",
                "tool": "update_ticket",
                "arg": "code",
                "field": "s",
                "allowed": "open",
            },
            "allowed",
        ),
        (
            {
                "type": "allowed_values",
                "tool": "update_ticket",
                "arg": "status",
                "values": ["open"],
                "when": {"code": "x"},
            },
            "when.code",
        ),
        (
            {
                "type": "duty",
                "trigger": {"tool": "create_ticket"},
                "then": {"tool": "update_ticket", "when": {"status": "closed"}},
            },
            "then.when.status",
        ),
        (
            {
                "type": "duty",
                "trigger": {"tool": "create_ticket"},
                "then": {"tool": "update_ticket", "when": {"status": True}},
            },
            "then.when.status",
        ),
        ({"type": "guidance", "text": "Be kind", "tools": "update_ticket"}, "tools"),
        ({"type": "format", "tool": "update_ticket", "arg": "code", "pattern": ["T-.*"]}, "pattern"),
        (
            {
                "type": "threshold",
                "tool": "update_ticket",
                "arg": "code",
                "op": "gt",
                "value": "five",
                "then": "forbid",
            },
            "value",
        ),
        ({"type": "role_permission", "tools": ["update_ticket"], "roles": "admin"}, "roles"),
        (
            {
                "type": "transition",
                "tool": "update_ticket",
                "arg": "status",
                "id_arg": "code",
                "edges": {"open": "closed"},
            },
            "edges",
        ),
    ],
)
def test_wrong_typed_slot_is_an_error(rule, where):
    """A bare value where a list belongs raises in ``before_tool_call`` (the hook registry swallows it and the write
    runs) or is read as its characters. The validator names the slot instead."""
    found = _one(_raw("r", rule))
    assert [(e.code, e.severity) for e in found] == [("invalid_slots", "error")]
    assert where in found[0].message


@pytest.mark.parametrize(
    "rule",
    [
        {"type": "required_args", "tool": "update_ticket", "args": []},
        {"type": "allowed_values", "tool": "update_ticket", "arg": "", "values": ["open"]},
        {"type": "allowed_values", "tool": "update_ticket", "arg": "status"},
        {"type": "format", "tool": "update_ticket", "arg": "code", "pattern": "<regex>"},
        {"type": "format", "tool": "update_ticket", "arg": "code", "pattern": "TODO"},
        {"type": "duty", "trigger": {"tool": "create_ticket"}, "then": {"when": {"status": ["closed"]}}},
        {"type": "verify", "tools": ["update_ticket"]},
        {"type": "record_condition", "tools": ["update_ticket"], "field": "code", "op": "equals"},
    ],
)
def test_unfilled_slot(rule):
    assert [(e.code, e.severity) for e in _one(_raw("r", rule))] == [("unfilled_slot", "error")]


def test_class_and_level_must_be_set():
    rule = {"type": "allowed_values", "tool": "update_ticket", "arg": "status", "values": ["open"]}
    found = _one(_raw("r", rule, cls="", level=""))
    assert [e.code for e in found] == ["unfilled_slot", "unfilled_slot"]
    assert {e.message.split(": ")[1] for e in found} == {"class is not set", "level is not set"}


def test_unknown_level_and_class():
    rule = {"type": "allowed_values", "tool": "update_ticket", "arg": "status", "values": ["open"]}
    assert [e.code for e in _one(_raw("r", rule, level="block"))] == ["unknown_level"]
    assert [e.code for e in _one(_raw("r", rule, cls="soft"))] == ["unknown_class"]
    # with no level of its own, the level inside the rule is what package.runtime_rules reads (and raises on)
    inner = _raw("r", {**rule, "level": "block"}, level="")
    assert sorted(e.code for e in _one(inner)) == ["unfilled_slot", "unknown_level"]
    assert [e.code for e in _one(_raw("r", {**rule, "level": "block"}, level="deny"))] == []


def test_rule_no_trigger():
    verify = {"type": "verify", "statement": "Be careful", "tools": []}
    pkg = PolicyPackage(version="v", rules=[_raw("v", verify, cls="judgment", level="remind")])
    assert _findings(pkg, legacy=False) == [("coverage_incomplete", "error"), ("rule_no_trigger", "error")]
    assert ("rule_no_trigger", "warning") in _findings(pkg, legacy=True)
    # global by design: no trigger tool is fine
    for kind in ("guidance", "id_provenance", "grounded_values"):
        rule = {"type": kind, "text": "x"} if kind == "guidance" else {"type": kind}
        assert [e.code for e in _one(_raw("g", rule, cls="guidance", level="observe"))] == []
    assert [e.code for e in _one(_raw("r", {"type": "required_args", "args": ["code"]}))] == [
        "rule_no_trigger"
    ]
    duty = {"type": "duty", "trigger": {}, "then": {"tool": "update_ticket"}, "quote": "x"}
    assert [e.code for e in _one(_raw("d", duty))] == ["rule_no_trigger"]


# -- coverage ----------------------------------------------------------------------------------------------------------


def _coverage_package(**kw):
    doc = _doc(
        "d", (TEXT, "must"), ("Never delete a ticket.", "must_not"), ("Tickets may be reopened.", "may")
    )
    return PolicyPackage(
        version="v", documents=[doc], rules=[_allowed("r1", "d:1:1", ["open", "closed"])], **kw
    )


def test_a_must_not_statement_with_no_outcome_is_incomplete():
    found = validate_package(_coverage_package(), TOOLS)
    assert [(e.code, e.severity, e.statement_id) for e in found] == [
        ("coverage_incomplete", "error", "d:1:2")
    ]
    assert "must_not" in found[0].message  # the "may" statement needs none


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coverage": [CoverageItem("d:1:2", "out_of_scope", "not a rule for the agent")]},
        {
            "policy_questions": [
                {"statement_ids": ["d:1:2"], "question": "Does it apply?", "default_level": "remind"}
            ]
        },
        {"procedures": [Procedure(id="p", steps=[Step(id="d:1:2", action="update_ticket")])]},
        {"procedures": [Procedure(id="p", exceptions=[{"quote": "Never delete a ticket.", "then": "stop"}])]},
        {
            "global_": {
                "definitions": [{"term": "ticket", "meaning": "a ticket", "source": {"statement": "d:1:2"}}]
            }
        },
        {
            "obligations": [
                Obligation(id="o", effect={"kind": "reply"}, source={"quote": "Never delete a ticket."})
            ]
        },
    ],
)
def test_each_outcome_covers_a_statement(kwargs):
    assert validate_package(_coverage_package(**kwargs), TOOLS) == []


def test_a_rule_outcome_needs_a_rule_that_names_the_statement():
    said = _coverage_package(coverage=[CoverageItem("d:1:2", "hard_rule")])
    assert _findings(said) == [("coverage_incomplete", "error")]
    for cls in RULE_CLASSES:
        rule = _raw(
            "r2", {"type": "guidance", "text": "Never delete a ticket."}, ["d:1:2"], cls=cls, level="observe"
        )
        backed = _coverage_package()
        backed.rules.append(rule)
        assert validate_package(backed, TOOLS) == []


def test_out_of_scope_needs_a_reason():
    pkg = _coverage_package(coverage=[CoverageItem("d:1:2", "out_of_scope")])
    assert _findings(pkg) == [("coverage_incomplete", "error"), ("coverage_incomplete", "error")]


def test_legacy_coverage_kinds_count_only_for_a_legacy_package():
    pkg = _coverage_package(coverage=[CoverageItem("d:1:2", "verify")])
    assert _findings(pkg, legacy=False) == [("coverage_incomplete", "error")]
    assert _findings(pkg, legacy=True) == []
    # a note about a statement is not an outcome, and a gap in a legacy package is a warning
    pkg = _coverage_package(coverage=[CoverageItem("d:1:2", "dangling_link", "no target")])
    assert _findings(pkg, legacy=True) == [("coverage_incomplete", "warning")]


def test_rules_without_statements_cannot_show_coverage():
    pkg = PolicyPackage(version="v", rules=[_allowed("r1", "d:1:1", ["open"])])
    assert _findings(pkg, legacy=False) == [("coverage_incomplete", "error")]
    assert _findings(pkg, legacy=True) == []


def test_statement_outcomes_give_each_statement_the_outcomes_that_coverage_reads():
    from polisys.validate import statement_outcomes

    pkg = _coverage_package(coverage=[CoverageItem("d:1:2", "out_of_scope", "not for the agent")])
    outcomes = statement_outcomes(pkg)
    assert outcomes["d:1:1"] == ["hard_rule"] and outcomes["d:1:2"] == ["out_of_scope"]
    pkg = _coverage_package(coverage=[CoverageItem("d:1:2", "verify")])
    assert statement_outcomes(pkg)["d:1:2"] == []  # a claim with nothing behind it
    assert statement_outcomes(pkg, legacy=True)["d:1:2"] == ["judgment_rule"]


# -- conflicts ---------------------------------------------------------------------------------------------------------


def _pair(values_a, values_b, when_a=None, when_b=None, docs=None, **kw):
    docs = docs or [_doc("d1", (TEXT, "must")), _doc("d2", (TEXT, "must"))]
    rules = [_allowed("a", "d1:1:1", values_a, when_a), _allowed("b", "d2:1:1", values_b, when_b)]
    return PolicyPackage(version="v", documents=docs, rules=rules, **kw)


def _conflicts(pkg, **kw):
    return [e for e in validate_package(pkg, TOOLS, **kw) if e.code == "rule_conflict"]


def test_disjoint_values_on_the_same_argument_conflict():
    found = _conflicts(_pair(["open"], ["closed"]))
    assert [(e.code, e.severity) for e in found] == [("rule_conflict", "error")]
    assert "update_ticket.status" in found[0].message and "nothing decides" in found[0].message
    assert "resolve_conflicts sets both to remind" in found[0].message


def test_values_with_something_in_common_do_not_conflict():
    assert _conflicts(_pair(["open", "closed"], ["closed"])) == []
    assert _conflicts(_pair(["Open"], ["open"])) == []  # same value, compared as the runtime compares it


@pytest.mark.parametrize(
    "when_a, when_b, conflict",
    [
        (None, None, True),
        ({"code": ["x"]}, {"code": ["y"]}, False),  # no call meets both
        ({"code": ["x"]}, {"code": ["x", "y"]}, True),
        ({"code": ["x"]}, {"summary": ["y"]}, True),  # an argument only one names is free in the other
        ({"code": ["*"]}, {"code": ["y"]}, True),  # "*" = a value is given
        (None, {"code": ["y"]}, True),
    ],
)
def test_conflicts_need_when_conditions_that_overlap(when_a, when_b, conflict):
    assert bool(_conflicts(_pair(["open"], ["closed"], when_a, when_b))) is conflict


def test_a_conflict_blocks_only_if_both_rules_can_deny():
    pkg = _pair(["open"], ["closed"])
    pkg.rules[1].level = "remind"
    assert [(e.code, e.severity) for e in _conflicts(pkg)] == [("rule_conflict", "warning")]


def test_reference_status_with_nothing_in_common_conflicts():
    def status(rid, sid, allowed):
        rule = {
            "type": "reference_status",
            "tool": "update_ticket",
            "arg": "code",
            "field": "state",
            "allowed": allowed,
        }
        return _raw(rid, {**rule, "quote": TEXT}, [sid])

    pkg = _pair(["open"], ["open"])
    pkg.rules = [status("a", "d1:1:1", ["active"]), status("b", "d2:1:1", ["retired"])]
    assert len(_conflicts(pkg)) == 1


def test_a_duty_that_asks_for_a_value_another_rule_forbids_conflicts():
    text = "After a ticket is created it must be closed."
    duty = {
        "type": "duty",
        "trigger": {"tool": "create_ticket"},
        "then": {"tool": "update_ticket", "when": {"status": ["closed"]}},
        "quote": text,
    }
    doc = _doc("d", (TEXT, "must"), (text, "must"))
    pkg = PolicyPackage(
        version="v",
        documents=[doc],
        rules=[_allowed("only_open", "d:1:1", ["open"]), _raw("close", duty, ["d:1:2"], level="remind")],
    )
    found = _conflicts(pkg)
    assert [(e.code, e.severity) for e in found] == [
        ("rule_conflict", "error")
    ]  # the allowed_values rule can deny
    assert "duty requires update_ticket.status ['closed']" in found[0].message
    pkg.rules[0].level = "remind"
    assert [e.severity for e in _conflicts(pkg)] == ["warning"]
    pkg.rules[1].rule["then"]["when"] = {"status": ["open"]}  # a value the other rule allows
    assert _conflicts(pkg) == []


def test_a_newer_date_or_higher_rank_decides_a_conflict():
    newer = _pair(
        ["open"],
        ["closed"],
        docs=[
            _doc("d1", (TEXT, "must"), effective_date="2025-01-01"),
            _doc("d2", (TEXT, "must"), effective_date="2026-01-01"),
        ],
    )
    found = _conflicts(newer)
    assert "b wins" in found[0].message and "resolve_conflicts removes a" in found[0].message
    ranked = _pair(
        ["open"],
        ["closed"],
        docs=[_doc("d1", (TEXT, "must"), authority_rank=2), _doc("d2", (TEXT, "must"), authority_rank=1)],
    )
    assert "a wins" in _conflicts(ranked)[0].message
    tie = _pair(
        ["open"],
        ["closed"],
        docs=[
            _doc("d1", (TEXT, "must"), effective_date="2026-01-01", authority_rank=1),
            _doc("d2", (TEXT, "must"), effective_date="2026-01-01", authority_rank=1),
        ],
    )
    assert "nothing decides" in _conflicts(tie)[0].message
    untrusted = _pair(
        ["open"],
        ["closed"],
        docs=[_doc("d1", (TEXT, "must")), _doc("d2", (TEXT, "must"), trust=Trust.UNTRUSTED)],
    )
    assert "a wins" in _conflicts(untrusted)[0].message


def test_an_exception_link_is_not_a_conflict():
    pkg = _pair(["open"], ["closed"], links=[Link(type=LinkType.EXCEPTION_OF, src="d2:1:1", dst="d1:1:1")])
    assert _conflicts(pkg) == []
    pkg.links = [Link(type=LinkType.REFERS_TO, src="d2:1:1", dst="d1:1:1")]
    assert len(_conflicts(pkg)) == 1


# -- resolve_conflicts -------------------------------------------------------------------------------------------------


def test_resolve_conflicts_removes_the_loser_and_keeps_the_input():
    pkg = _pair(
        ["open"],
        ["closed"],
        docs=[
            _doc("d1", (TEXT, "must"), effective_date="2025-01-01"),
            _doc("d2", (TEXT, "must"), effective_date="2026-01-01"),
        ],
    )
    before = to_json(pkg)
    out, notes = resolve_conflicts(pkg)
    assert to_json(pkg) == before and out is not pkg
    assert [r.id for r in out.rules] == ["b"]
    assert len(notes) == 1 and "removed a" in notes[0]
    # the losing statement is recorded, so coverage stays complete, and nothing conflicts any more
    assert [(c.statement_id, c.kind) for c in out.coverage] == [("d1:1:1", "out_of_scope")]
    assert out.coverage[0].reason == "superseded by b"
    assert validate_package(out, TOOLS) == []


def test_resolve_conflicts_by_rank():
    pkg = _pair(
        ["open"],
        ["closed"],
        docs=[_doc("d1", (TEXT, "must"), authority_rank=1), _doc("d2", (TEXT, "must"), authority_rank=5)],
    )
    out, _ = resolve_conflicts(pkg)
    assert [r.id for r in out.rules] == ["b"]


def test_a_tie_is_never_decided():
    pkg = _pair(["open"], ["closed"])
    before = to_json(pkg)
    out, notes = resolve_conflicts(pkg)
    assert to_json(pkg) == before
    assert [(r.id, r.level) for r in out.rules] == [("a", "remind"), ("b", "remind")]
    assert len(out.policy_questions) == 1
    question = out.policy_questions[0]
    assert question["statement_ids"] == ["d1:1:1", "d2:1:1"] and question["default_level"] == "remind"
    assert question["rule_ids"] == ["a", "b"] and "each hold in a different case" in question["question"]
    assert len(notes) == 1
    # a second run adds nothing, and the validator then reports the conflict as acknowledged, not blocking
    again, notes2 = resolve_conflicts(out)
    assert len(again.policy_questions) == 1
    assert notes2 == ["tie between a and b: both set to at most remind, policy question already open"]
    found = _conflicts(out)
    assert [e.severity for e in found] == ["warning"]
    assert "a policy question about this pair is open" in found[0].message


def test_a_tie_lowers_a_level_and_never_raises_one():
    pkg = _pair(["open"], ["closed"])
    pkg.rules[0].level = "observe"
    out, _ = resolve_conflicts(pkg)
    assert [r.level for r in out.rules] == ["observe", "remind"]


def test_resolve_conflicts_leaves_a_converted_package_alone():
    pkg = _pair(["open"], ["closed"])
    pkg.extra["_layout"] = {"format": "build", "top": []}
    out, notes = resolve_conflicts(pkg)
    assert to_json(out) == to_json(pkg) and "not changed" in notes[0]


# -- tools, fields, obligations ----------------------------------------------------------------------------------------


def test_tool_checks_need_tool_specs():
    rule = _allowed("r1", "d:1:1", ["open"], tool="nope")
    pkg = PolicyPackage(version="v", documents=[_doc("d", (TEXT, "must"))], rules=[rule])
    assert _findings(pkg, TOOLS) == [("unknown_tool", "error")]
    assert _findings(pkg, None) == []  # no spec file: skipped
    other = _allowed("r1", "d:1:1", ["open"], arg="nope")
    pkg.rules = [other]
    assert _findings(pkg, TOOLS) == [("unknown_arg", "error")]


def test_tool_specs_with_parameters_instead_of_inputschema():
    specs = [{"name": "update_ticket", "parameters": {"properties": {"status": {"type": "string"}}}}]
    pkg = _clean_package()
    assert _findings(pkg, specs) == []
    pkg.rules = [_allowed("r1", "d:1:1", ["open"], arg="code")]
    assert _findings(pkg, specs) == [("unknown_arg", "error")]


def test_schema_conflict_is_a_warning():
    pkg = _clean_package()
    pkg.rules = [_allowed("r1", "d:1:1", ["open", "gone"])]  # the policy lists a value the tool lacks
    pkg.rules[0].rule["by"] = "code"
    found = validate_package(pkg, TOOLS)
    assert [(e.code, e.severity) for e in found] == [("schema_conflict", "warning")]
    assert "update_ticket.status" in found[0].message and "gone" in found[0].message


def test_wrong_typed_package_fields_are_errors():
    pkg = _clean_package()
    pkg.schema = "policy-package/9"
    pkg.documents[0].authority_rank = "high"
    pkg.documents[0].trust = "maybe"
    pkg.rules[0].may_deny = "no"
    pkg.rules[0].statement_ids = "d:1:1"
    found = validate_package(pkg, None)
    assert {e.code for e in found} == {"invalid_field"}
    for key in ("schema", "authority_rank", "trust", "may_deny", "statement_ids"):
        assert any(key in e.message for e in found), key


@pytest.mark.parametrize(
    "field, value", [("documents", "text"), ("rules", [None]), ("policy_questions", "q"), ("global_", [])]
)
def test_a_package_of_the_wrong_shape_is_reported_not_raised(field, value):
    pkg = _clean_package()
    setattr(pkg, field, value)
    found = validate_package(pkg, TOOLS)
    assert [e.code for e in found] == ["invalid_field"]


def test_a_date_that_is_not_a_date_is_an_error():
    pkg = _clean_package()
    pkg.documents[0].effective_date = "yesterday"
    found = validate_package(pkg, None)
    assert [(e.code, e.severity) for e in found] == [("invalid_field", "error")]
    assert "effective_date" in found[0].message


def test_duplicate_rule_ids_and_unknown_fields():
    pkg = _clean_package()
    pkg.rules.append(copy.deepcopy(pkg.rules[0]))
    pkg.extra["obligatons"] = []  # a typo goes to ``extra``; the keys package.py keeps are fine
    pkg.extra["stats"] = {}
    pkg.extra["_layout"] = {}
    found = validate_package(pkg, TOOLS)
    assert sorted((e.code, e.severity) for e in found) == [
        ("duplicate_rule_id", "error"),
        ("unknown_field", "warning"),
    ]


def test_obligation_fields():
    ok = Obligation(
        id="ok",
        effect={"kind": "write", "tools": ["update_ticket"], "match": {"status": {"pattern": "{s}"}}},
        values={"s": {"from": "literal", "value": "closed"}},
    )
    bad = Obligation(
        id="bad",
        origin="somewhere",
        check="magic",
        level="loud",
        effect={"kind": "write", "tools": ["nope"]},
        values={"s": {"from": "guess"}},
    )
    empty = Obligation(id="empty", effect={})
    pkg = PolicyPackage(version="v", documents=[_doc("d", (TEXT, "may"))], obligations=[ok, bad, empty])
    found = {(e.statement_id, e.code) for e in validate_package(pkg, TOOLS) if e.severity == "error"}
    assert found == {
        ("bad", "invalid_field"),
        ("bad", "unknown_tool"),
        ("empty", "unfilled_slot"),
    }
    assert (
        sum(e.statement_id == "bad" and e.code == "invalid_field" for e in validate_package(pkg, TOOLS)) == 4
    )


# -- command line ------------------------------------------------------------------------------------------------------


def _write(tmp_path, pkg):
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps(TOOLS))
    path = tmp_path / "package.json"
    save_package(pkg, path)
    return str(path), str(tools)


def test_cli_validates_a_package(tmp_path, capsys):
    path, tools = _write(tmp_path, _clean_package())
    assert cli.main(["validate", path, "--tools", tools]) == 0
    assert "0 errors, 0 warnings" in capsys.readouterr().out
    # no tool file: the tool checks are skipped, the rest runs
    assert cli.main(["validate", path]) == 0


def test_cli_exit_code_follows_severity_and_legacy_flag(tmp_path, capsys):
    path, tools = _write(tmp_path, _coverage_package())
    assert cli.main(["validate", path, "--tools", tools]) == 1
    assert "ERROR coverage_incomplete d:1:2" in capsys.readouterr().out
    assert cli.main(["validate", path, "--tools", tools, "--legacy"]) == 0
    out = capsys.readouterr().out
    assert "WARN coverage_incomplete d:1:2" in out and "0 errors, 1 warnings" in out


def test_cli_reports_a_package_that_cannot_be_read(tmp_path, capsys):
    path, tools = _write(tmp_path, _clean_package())
    data = json.loads(open(path).read())
    data["documents"][0]["trust"] = "maybe"
    open(path, "w").write(json.dumps(data))
    assert cli.main(["validate", path, "--tools", tools]) == 1
    assert "cannot read the package" in capsys.readouterr().out


def test_cli_still_validates_a_build_and_needs_tools_for_it(tmp_path, capsys):
    build = tmp_path / "build.json"
    build.write_text(json.dumps(to_json(PolicyBuild(version="v"))))
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps(TOOLS))
    assert cli.main(["validate", str(build), "--tools", str(tools)]) == 0
    assert cli.main(["validate", str(build)]) == 2


# -- fix round 1 -------------------------------------------------------------------------------------------------------

from polisys.model import from_json  # noqa: E402
from polisys.ruleset import RuleSetHooks  # noqa: E402
from polisys.validate import RULE_SLOTS, _is_broken, _slot_findings, _trigger_tools  # noqa: E402
from polisys.verifier import Verifier  # noqa: E402


@pytest.mark.parametrize(
    "rule, where",
    [
        ({"type": "record_condition", "field": "code", "op": "in", "value": 5}, "value"),
        ({"type": "record_condition", "field": "code", "op": "in", "value": True}, "value"),
        (
            {"type": "record_condition", "field": "code", "op": "in", "value": "open"},
            "value",
        ),  # read as characters
        ({"type": "record_condition", "field": "code", "op": "equals", "value": ["a"]}, "value"),
        ({"type": "record_condition", "field": "code", "op": "gt", "value": "many"}, "value"),
        ({"type": "grounded_values", "tool": "update_ticket"}, "tool"),
        ({"type": "grounded_values", "tools": ["update_ticket"]}, "tools"),
        ({"type": "id_provenance", "tools": ["update_ticket"]}, "tools"),
        ({"type": "verify", "statement": "x", "tools": ["update_ticket"], "types": 5}, "types"),
        ({"type": "grounded_values", "types": 5}, "types"),
        ({"type": "allowed_values", "tool": "t", "arg": "a", "values": ["x"], "types": 5}, "types"),
        ({"type": "allowed_values", "tool": "t", "arg": "a", "values": ["x"], "id": ["x"]}, "id"),
        ({"type": "verify", "statement": "x", "tools": ["t"], "unless": [{"condition": 5}]}, "unless"),
        (
            {"type": "duty", "trigger": {"tool": "t"}, "then": {"tool": "u"}, "unless": [{"condition": 5}]},
            "unless",
        ),
        ({"type": "format", "tool": "t", "arg": "a", "pattern": "x", "when": ["a"]}, "when"),
        (
            {
                "type": "threshold",
                "tool": "t",
                "arg": "a",
                "op": "gt",
                "value": 1,
                "then": "forbid",
                "when": 1,
            },
            "when",
        ),
        (
            {
                "type": "transition",
                "tool": "t",
                "arg": "a",
                "id_arg": "i",
                "edges": {"o": ["c"]},
                "when": "x",
            },
            "when",
        ),
        ({"type": "verify", "statement": "x", "tools": ["t"], "when": ["a"]}, "when"),
        ({"type": "guidance", "text": "x", "tools": ["t"], "when": "a"}, "when"),
        ({"type": "required_args", "args": ["code"], "tools": 5}, "tools"),
        ({"type": "format", "tool": "t", "arg": "a", "pattern": "x", "quote": ["q"]}, "quote"),
    ],
)
def test_shapes_that_fail_at_run_time_are_errors(rule, where):
    found = _one(_raw("r", rule), specs=None)
    assert "invalid_slots" in {e.code for e in found}, found
    assert any(where in e.message for e in found if e.code == "invalid_slots")


def test_record_condition_value_follows_op():
    base = {"type": "record_condition", "tools": ["update_ticket"], "field": "code"}
    fine = [
        ("in", ["a", "b"]),
        ("contains", "abc"),
        ("contains", 5),
        ("equals", False),
        ("equals", 0),
        ("gt", 5),
        ("gt", "5"),
        ("le", 2.5),
        ("empty", None),
        ("empty", "ignored: the runtime does not read a value for empty"),
    ]
    for op, value in fine:
        assert _one(_raw("r", {**base, "op": op, "value": value})) == [], (op, value)
    assert _one(_raw("r", {**base, "op": "empty"})) == []  # no value at all
    for op in ("in", "contains", "equals", "gt"):
        found = _one(_raw("r", {**base, "op": op}))
        assert [(e.code, e.severity) for e in found] == [("unfilled_slot", "error")], op


def test_date_compare_warns_only_for_the_dates_the_runtime_cannot_compare():
    base = {"type": "record_condition", "tools": ["update_ticket"], "field": "code"}
    for op in (
        "gt",
        "ge",
        "lt",
        "le",
    ):  # an ISO date, with or without a time, is compared as a date: nothing to warn
        assert _one(_raw("r", {**base, "op": op, "value": "2025-02-25"})) == []
        assert _one(_raw("r", {**base, "op": op, "value": "2025-02-25T12:08:00-05:00"})) == []
    for value in ("02/25/2025", "25.02.2025", "Feb 25 2025", "25 February 2025", "2025-02", "2025-13-45 EST"):
        found = _one(_raw("r", {**base, "op": "lt", "value": value}))
        assert [(e.code, e.severity) for e in found] == [("date_compare", "warning")], value
        assert "ISO" in found[0].message
    assert _one(_raw("r", {**base, "op": "lt", "value": "2025"})) == []  # a plain number is a number
    assert (
        _one(_raw("r", {**base, "op": "equals", "value": "02/25/2025"})) == []
    )  # equality is not a comparison
    threshold = {"type": "threshold", "tool": "update_ticket", "arg": "code", "op": "lt", "then": "forbid"}
    for value in (
        "2025-02-25",
        "02/25/2025",
    ):  # a threshold reads its argument as a number, whatever the form
        found = _one(_raw("r", {**threshold, "value": value}))
        assert [(e.code, e.severity) for e in found] == [("date_compare", "warning")], value
    assert _one(_raw("r", {**threshold, "value": 100})) == []


def test_a_global_rule_with_an_empty_tool_slot_is_fine():
    assert _one(_raw("r", {"type": "id_provenance", "tools": [], "tool": ""})) == []


# One valid, firing rule per type, and values that each slot kind accepts. Every rule that the slot table accepts must run
# through the hooks without raising: a hook that raises is swallowed by the hook registry, and the write goes through.
RUN_TOOLS = [
    {"name": name, "inputSchema": {"type": "object", "properties": {"status": {}, "code": {}, "summary": {}}}}
    for name in ("update_ticket", "create_ticket", "read_ticket")
]
BASE_RULES = {
    "required_args": {"tool": "update_ticket", "args": ["code"]},
    "allowed_values": {"tool": "update_ticket", "arg": "status", "values": ["open"]},
    "derived_value": {"tool": "update_ticket", "arg": "status", "inputs": ["code"], "table": {"abc": "open"}},
    "transition": {"tool": "update_ticket", "arg": "status", "id_arg": "code", "edges": {"open": ["closed"]}},
    "reference_status": {"tool": "update_ticket", "arg": "code", "field": "state", "allowed": ["active"]},
    "membership": {"tool": "update_ticket", "arg": "code", "group_arg": "summary"},
    "role_permission": {"tools": ["update_ticket"], "roles": ["viewer"]},
    "grounded_values": {},
    "id_provenance": {},
    "duty": {
        "trigger": {"tool": "create_ticket"},
        "then": {"tool": "update_ticket", "when": {"status": ["closed"]}},
    },
    "guidance": {"text": "Be kind.", "tools": ["update_ticket"]},
    "record_condition": {"tools": ["update_ticket"], "field": "code", "op": "equals", "value": "abc"},
    "threshold": {"tool": "update_ticket", "arg": "code", "op": "gt", "value": 5, "then": "forbid"},
    "format": {"tool": "update_ticket", "arg": "code", "pattern": r"T-\d+"},
    "verify": {"statement": "Be careful.", "tools": ["update_ticket"]},
}
SAMPLES = {
    "str": ["x", "Open Case"],
    "text": ["a sentence", "<main_policy>"],
    "strs": [["update_ticket"], ["a", "b"]],
    "scalar": ["x", 5, 2.5, True, False, 0],
    "values": [["x"], [1], [True], [2.5], ["a", 1, False], [""]],
    "number": [5, 2.5, "5", "$500"],
    "when": [{"status": ["open"]}, {"status": [True, 1, "*"]}, {"code": ["x"]}],
    "edges": [{"open": ["closed"]}, {"open": []}],
    "dict": [{"abc": "high"}],
    "unless": [[{"condition": "when it applies"}], [{"condition": ""}], [{}]],
    "any": ["x", 5, [1], True],
}
CALLS = [
    {"status": "open", "code": "abc"},
    {"status": "zzz", "code": "T-5", "summary": "s"},
    {"status": "closed", "code": "TKT-12"},
    {"status": True, "code": 7},
    {"code": "TKT-12"},
]


def _variants(rule_type):
    """Rules of one type that differ from the base rule in one slot, with a value the slot's kind accepts."""
    base = {"id": "r", "type": rule_type, "quote": "q", **BASE_RULES[rule_type]}
    yield base
    for name, (kind, _) in RULE_SLOTS[rule_type].items():
        if isinstance(kind, dict):  # a nested table: duty trigger and then
            for sub, (sub_kind, _) in kind.items():
                for sample in SAMPLES[sub_kind]:
                    yield {**base, name: {**base[name], sub: sample}}
        elif name not in ("id", "type") and (rule_type, name) != (
            "record_condition",
            "value",
        ):  # value: by op, below
            for sample in SAMPLES[kind]:
                yield {**base, name: sample}
    if rule_type == "record_condition":
        for op, value in [("in", ["abc"]), ("in", [1]), ("in", [True]), ("contains", 5), ("equals", False)]:
            yield {**base, "op": op, "value": value}
        for op, value in [("gt", 5), ("gt", "5"), ("le", 2.5), ("empty", None), ("empty", [1])]:
            yield {**base, "op": op, "value": value}


def _run_through_hooks(rule):
    asked = []
    verifier = Verifier(lambda system, user: asked.append(user) or '{"violations": []}')
    hooks = RuleSetHooks([rule], RUN_TOOLS, principal={"role": "viewer"}, verifier=verifier)
    hooks.on_user_message("update ticket TKT-12 to open", None)
    hooks.before_llm_call(None)
    record = {"id": "TKT-12", "code": "abc", "status": "open", "state": "active", "summary": "s"}
    hooks.after_tool_call("read_ticket", {"id": "TKT-12"}, record)
    for args in CALLS:
        hooks.before_tool_call("update_ticket", args)
    hooks.after_tool_call("create_ticket", {"status": "open"}, {"id": "TKT-13"})
    hooks.before_llm_call(None)
    hooks.after_tool_call("update_ticket", {"status": "closed", "code": "TKT-12"}, record)
    hooks.before_reply("done")
    return asked


@pytest.mark.parametrize("rule_type", sorted(BASE_RULES))
def test_every_shape_the_slot_table_allows_runs(rule_type):
    ran = 0
    for rule in _variants(rule_type):
        problems = [f for f in _slot_findings(Rule(id="r", type=rule_type, rule=rule)) if _is_broken([f])]
        assert problems == [], (rule, problems)  # the sample is a shape the table allows
        asked = _run_through_hooks(rule)  # raises if the table allows a shape that fails at run time
        tools = _trigger_tools(rule)
        if rule_type == "verify" and (not tools or "update_ticket" in tools):
            assert len(asked) == len(CALLS), rule  # a verify rule with a bad ``unless`` costs the model call
        ran += 1
    assert ran > 3


# -- malformed packages are reported, not raised -----------------------------------------------------------------------


def _bad_trust(pkg):
    pkg.documents[0].trust = ["x"]
    return TOOLS


def _bad_spec(pkg):
    return [{"inputSchema": {}}, *TOOLS]


def _set(path, value):
    def apply(pkg):
        target = pkg
        for key in path[:-1]:
            target = target[key] if isinstance(key, int) else getattr(target, key)
        if isinstance(path[-1], int):
            target[path[-1]] = value
        else:
            setattr(target, path[-1], value)
        return TOOLS

    return apply


def _question(**kw):
    return {"statement_ids": ["d:1:1"], "question": "q?", "default_level": "remind", **kw}


@pytest.mark.parametrize(
    "make",
    [
        lambda pkg: (
            pkg.rules.append(
                _raw("t", {"type": "required_args", "tool": "update_ticket", "args": ["code"], "tools": 5})
            )
            or TOOLS
        ),
        lambda pkg: pkg.policy_questions.append(_question(statement_ids=5)) or TOOLS,
        lambda pkg: pkg.policy_questions.append(_question(statement_ids=[["x"]])) or TOOLS,
        lambda pkg: pkg.policy_questions.append(_question(question=5)) or TOOLS,
        lambda pkg: pkg.policy_questions.append(_question(default_level="loud")) or TOOLS,
        lambda pkg: pkg.policy_questions.append("a question") or TOOLS,
        lambda pkg: (
            pkg.global_.update(definitions=[{"term": "t", "source": {"statement": ["d:1:1"]}}]) or TOOLS
        ),
        lambda pkg: pkg.global_.update(definitions={"term": "meaning"}) or TOOLS,
        lambda pkg: (
            pkg.obligations.append(Obligation(id="o", effect={"kind": "reply"}, source={"quote": 5})) or TOOLS
        ),
        lambda pkg: (
            pkg.obligations.append(Obligation(id="o", effect={"kind": "write", "tools": [["x"]]})) or TOOLS
        ),
        lambda pkg: pkg.obligations.append(Obligation(id="o", effect=[])) or TOOLS,
        _bad_trust,
        _bad_spec,
        lambda pkg: "not a list of specs",
    ],
)
def test_a_malformed_package_is_reported_not_raised(make):
    pkg = _clean_package()
    specs = make(pkg)
    found = validate_package(pkg, specs)
    assert found and all(isinstance(e.message, str) for e in found)


def _rich_package():
    """One of everything the validator reads, so that the fuzz below reaches each field."""
    quote = "Never delete a ticket."
    doc = _doc(
        "d",
        (TEXT, "must"),
        (quote, "must_not"),
        ("Tickets may be reopened.", "may"),
        ("A viewer cannot change tickets.", "may"),
    )
    rules = [_allowed("a", "d:1:1", ["open", "closed"])]
    for rule_type, base in BASE_RULES.items():
        rule = {"type": rule_type, "quote": quote, **copy.deepcopy(base)}
        if rule_type == "duty":
            rule["then"]["when"] = {"status": ["open"]}  # a value that "a" allows
        if rule_type != "grounded_values":
            rules.append(_raw(rule_type, rule, ["d:1:2"]))
    rules.append(_raw("g", {"type": "grounded_values"}))
    return PolicyPackage(
        version="v",
        documents=[doc],
        rules=rules,
        procedures=[Procedure(id="p", steps=[Step(id="d:1:2", action="update_ticket", quote=quote)])],
        obligations=[
            Obligation(
                id="o",
                effect={"kind": "write", "tools": ["update_ticket"], "match": {"status": {"pattern": "{s}"}}},
                values={"s": {"from": "literal", "value": "closed"}},
                source={"quote": quote},
            )
        ],
        links=[Link(type=LinkType.REFERS_TO, src="d:1:1", dst="d:1:2")],
        coverage=[CoverageItem("d:1:3", "out_of_scope", "r")],
        policy_questions=[_question(rule_ids=["a", "b"])],
        global_={"definitions": [{"term": "t", "meaning": "m", "source": {"statement": "d:1:1"}}]},
    )


JUNK = [5, "x", None, [], {}, [[1]], {"a": 1}, True, [["x"]]]
GONE = object()


def _locations(data, path=()):
    yield path
    if isinstance(data, dict):
        for key, value in data.items():
            yield from _locations(value, path + (key,))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            yield from _locations(value, path + (i,))


def _mutated(data, path, value):
    out = copy.deepcopy(data)
    target = out
    for key in path[:-1]:
        target = target[key]
    if value is GONE:
        target.pop(path[-1]) if isinstance(target, dict) else target.pop()
    else:
        target[path[-1]] = value
    return out


def test_no_wrong_value_anywhere_in_a_package_makes_the_validator_raise():
    """Every field and slot of a package that holds every part is replaced, in turn, by wrong values (three of the nine
    kinds, rotating from one place to the next, and the place left out). A package that cannot be decoded is the command
    line's one ``invalid_field``; one that can be decoded must be validated."""
    data = to_json(_rich_package())
    base = validate_package(from_json(PolicyPackage, data), RUN_TOOLS)
    assert [
        e for e in base if e.severity == "error"
    ] == []  # the base itself is clean (a role rule only warns)
    decoded = 0
    for i, path in enumerate(p for p in _locations(data) if p):
        for value in [*(JUNK[(i + k) % len(JUNK)] for k in (0, 3, 6)), GONE]:
            try:
                pkg = from_json(PolicyPackage, _mutated(data, path, value))
            except Exception:  # noqa: BLE001 - cli.cmd_validate reports any decode failure
                continue
            validate_package(pkg, RUN_TOOLS, legacy=bool(i % 2))
            resolve_conflicts(pkg)
            decoded += 1
    assert decoded > 800
    # a slot that a rule of this type does not have: ``tools: 5`` on a rule with no tools, ``types`` on any rule
    for i, rule in enumerate(data["rules"]):
        for name in ("tool", "tools", "types", "when", "unless", "value", "id", "quote", "statement"):
            for value in (5, [[1]]):
                mutated = copy.deepcopy(data)
                mutated["rules"][i]["rule"][name] = value
                validate_package(from_json(PolicyPackage, mutated), RUN_TOOLS)
                resolve_conflicts(from_json(PolicyPackage, mutated))


# -- command line ------------------------------------------------------------------------------------------------------


def test_cli_does_not_read_a_schema_typo_as_a_build(tmp_path, capsys):
    path, tools = _write(tmp_path, _clean_package())
    data = json.loads(open(path).read())
    data["rules"][0]["rule"]["values"] = "open"  # a bare string: an error in a package
    for schema in ("policy-package/9", None):
        bad = dict(data)
        if schema is None:
            bad.pop("schema")
        else:
            bad["schema"] = schema
        open(path, "w").write(json.dumps(bad))
        assert cli.main(["validate", path, "--tools", tools]) == 1
        assert "schema is" in capsys.readouterr().out
    data["schema"] = "policy-package/1"
    open(path, "w").write(json.dumps(data))
    assert cli.main(["validate", path, "--tools", tools]) == 1
    assert "invalid_slots" in capsys.readouterr().out


def test_cli_says_when_the_tool_checks_are_skipped(tmp_path, capsys):
    path, tools = _write(tmp_path, _clean_package())
    assert cli.main(["validate", path]) == 0
    assert "NOTE no --tools" in capsys.readouterr().out
    assert cli.main(["validate", path, "--tools", tools]) == 0
    assert "NOTE" not in capsys.readouterr().out


def test_cli_reports_input_that_cannot_be_decoded(tmp_path, capsys):
    path, tools = _write(tmp_path, _clean_package())
    data = json.loads(open(path).read())
    data["procedures"] = [{"id": "p", "steps": [[1]]}]
    open(path, "w").write(json.dumps(data))
    assert cli.main(["validate", path, "--tools", tools]) == 1
    assert "cannot read the package" in capsys.readouterr().out
    open(path, "w").write("this is not json")
    assert cli.main(["validate", path, "--tools", tools]) == 1
    assert "cannot read the input" in capsys.readouterr().out
    open(path, "w").write("[1, 2]")
    assert cli.main(["validate", path, "--tools", tools]) == 1
    assert "JSON object" in capsys.readouterr().out
    assert cli.main(["validate", str(tmp_path / "missing.json"), "--tools", tools]) == 1


# -- placeholders and empty values inside lists ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        {"type": "allowed_values", "tool": "update_ticket", "arg": "status", "values": ["TODO"]},
        {"type": "allowed_values", "tool": "update_ticket", "arg": "status", "values": ["open", "<x>"]},
        {
            "type": "reference_status",
            "tool": "update_ticket",
            "arg": "code",
            "field": "s",
            "allowed": ["TBD"],
        },
        {"type": "required_args", "tool": "update_ticket", "args": ["code", ""]},
        {"type": "required_args", "tool": "update_ticket", "args": ["<arg>"]},
        {"type": "role_permission", "tools": ["<tool>"], "roles": ["admin"]},
        {
            "type": "allowed_values",
            "tool": "update_ticket",
            "arg": "status",
            "values": ["open"],
            "when": {"code": ["<id>"]},
        },
        {
            "type": "allowed_values",
            "tool": "update_ticket",
            "arg": "status",
            "values": ["open"],
            "when": {"code": [""]},
        },
        {
            "type": "allowed_values",
            "tool": "update_ticket",
            "arg": "status",
            "values": ["open"],
            "when": {"code": []},
        },
        {
            "type": "duty",
            "trigger": {"tool": "create_ticket"},
            "then": {"tool": "update_ticket", "when": {"code": ["<incident_id>"]}},
        },  # what the 12 real ITSM strings become when they are put in a list
        {
            "type": "derived_value",
            "tool": "update_ticket",
            "arg": "status",
            "inputs": ["code"],
            "table": {"a": "TODO"},
        },
    ],
)
def test_placeholders_and_empty_elements_inside_lists_are_unfilled(rule):
    found = _one(_raw("r", rule))
    assert [(e.code, e.severity) for e in found] == [("unfilled_slot", "error")]


def test_an_empty_string_among_values_is_a_warning_because_it_can_be_the_value():
    """ "When creating a Jira ticket, the ticket description starts empty": ``values: [""]`` is that rule."""
    rule = {"type": "allowed_values", "tool": "update_ticket", "arg": "status", "values": [""]}
    assert [(e.code, e.severity) for e in _one(_raw("r", rule), specs=None)] == [("empty_value", "warning")]
    # the rule still takes part in the conflict check and the run is not stopped by it
    pkg = _pair([""], ["open"])
    assert [e.code for e in validate_package(pkg, TOOLS) if e.code == "rule_conflict"] == ["rule_conflict"]


def test_a_state_with_no_outgoing_transition_is_fine():
    rule = {
        "type": "transition",
        "tool": "update_ticket",
        "arg": "status",
        "id_arg": "code",
        "edges": {"closed": []},
    }
    assert _one(_raw("r", rule)) == []


def test_policy_text_in_angle_brackets_is_not_a_placeholder():
    rule = {"type": "guidance", "text": "<main_policy>", "tools": ["update_ticket"]}
    assert _one(_raw("r", rule, cls="guidance", level="observe")) == []


# -- coverage claims ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["procedure_step", "obligation", "definition", "question", "hard_rule"])
def test_a_coverage_claim_with_nothing_behind_it_covers_nothing(kind):
    pkg = _coverage_package(coverage=[CoverageItem("d:1:2", kind)])
    found = validate_package(pkg, TOOLS)
    assert [(e.code, e.statement_id) for e in found] == [("coverage_incomplete", "d:1:2")]
    assert f"coverage says {kind}" in found[0].message
    assert _findings(pkg, legacy=True) == []  # a legacy package's coverage items count as written


def test_a_short_obligation_quote_does_not_cover_a_statement():
    for quote in ("a", "ticket", "Never delete"):
        pkg = _coverage_package(
            obligations=[Obligation(id="o", effect={"kind": "reply"}, source={"quote": quote})]
        )
        assert _findings(pkg) == [("coverage_incomplete", "error")], quote
    # a quote that is not a statement's text does not either, and it is reported as invented
    pkg = _coverage_package(
        obligations=[
            Obligation(id="o", effect={"kind": "reply"}, source={"quote": "Always keep every ticket"})
        ]
    )
    assert sorted(_findings(pkg)) == [("coverage_incomplete", "error"), ("quote_not_verbatim", "error")]


def test_out_of_scope_beside_another_outcome_is_a_warning_for_a_new_package():
    both = [
        CoverageItem("d:1:1", "out_of_scope", "not for the agent"),
        CoverageItem("d:1:2", "out_of_scope", "r"),
    ]
    pkg = _coverage_package(coverage=both)  # d:1:1 has a rule as well
    found = validate_package(pkg, TOOLS)
    assert [(e.code, e.severity, e.statement_id) for e in found] == [
        ("coverage_conflict", "warning", "d:1:1")
    ]
    assert _findings(pkg, legacy=True) == []
    # a question beside a rule is not a conflict: a tie adds one
    pkg = _coverage_package(coverage=both[1:], policy_questions=[_question()])
    assert validate_package(pkg, TOOLS) == []


# -- ties, chains, duplicates ------------------------------------------------------------------------------------------


def _two_statements_two_pairs():
    """Statements s1 and s2 give two rules each, so two pairs of rules tie."""
    docs = [_doc("d1", (TEXT, "must")), _doc("d2", (TEXT, "must"))]
    rules = [
        _allowed("a1", "d1:1:1", ["open"]),
        _allowed("a2", "d1:1:1", ["x"], tool="create_ticket"),
        _allowed("b1", "d2:1:1", ["closed"]),
        _allowed("b2", "d2:1:1", ["y"], tool="create_ticket"),
    ]
    return PolicyPackage(version="v", documents=docs, rules=rules)


def test_each_tied_pair_gets_its_own_question_and_note():
    out, notes = resolve_conflicts(_two_statements_two_pairs())
    assert [r.level for r in out.rules] == ["remind"] * 4
    assert sorted(q["rule_ids"] for q in out.policy_questions) == [["a1", "b1"], ["a2", "b2"]]
    assert len(notes) == 2 and all(n.startswith("tie between") for n in notes)


def test_a_question_about_other_rules_on_the_same_statements_does_not_hide_a_tie():
    pkg = _pair(["open"], ["closed"])
    pkg.policy_questions = [
        _question(statement_ids=["d1:1:1", "d2:1:1"])
    ]  # no rule_ids: about something else
    found = _conflicts(pkg)
    assert "resolve_conflicts sets both to remind" in found[0].message  # not "a question is open"
    out, notes = resolve_conflicts(pkg)
    assert len(out.policy_questions) == 2 and len(notes) == 1 and "question added" in notes[0]
    assert [r.level for r in out.rules] == ["remind", "remind"]


def test_a_rule_is_removed_only_by_a_winner_that_stays():
    """C {x, z}, A {y}, B {z}; dates B < A < C. A conflicts with B and with C, B and C do not conflict."""
    text = TEXT
    docs = [
        _doc("db", (text, "must"), effective_date="2024-01-01"),
        _doc("da", (text, "must"), effective_date="2025-01-01"),
        _doc("dc", (text, "must"), effective_date="2026-01-01"),
    ]
    rules = [
        _allowed("B", "db:1:1", ["z"]),
        _allowed("A", "da:1:1", ["y"]),
        _allowed("C", "dc:1:1", ["x", "z"]),
    ]
    out, notes = resolve_conflicts(PolicyPackage(version="v", documents=docs, rules=rules))
    assert [r.id for r in out.rules] == ["B", "C"]
    assert len(notes) == 1 and "removed A: C wins" in notes[0]
    assert [c.reason for c in out.coverage] == ["superseded by C"]  # names a rule that stays


def test_rule_ids_that_two_rules_share_are_reported():
    pkg = _clean_package()
    twin = copy.deepcopy(pkg.rules[0])
    pkg.rules.append(twin)  # same wrapper id and inner id
    other = copy.deepcopy(pkg.rules[0])
    other.id = "r3"  # a different wrapper id, the same inner id
    pkg.rules.append(other)
    found = [e for e in validate_package(pkg, TOOLS) if e.code == "duplicate_rule_id"]
    assert len(found) == 2
    assert "id 'r1'" in found[0].message and "inner id 'r1'" in found[1].message
    assert "dropped" not in found[0].message and "brief lists only the first" in found[0].message


# -- required vs forbidden ---------------------------------------------------------------------------------------------


def _duty_package(then_when, other, then_tool="update_ticket"):
    text = "After a ticket is created it must be updated."
    duty = {
        "type": "duty",
        "trigger": {"tool": "create_ticket"},
        "then": {"tool": then_tool, "when": then_when},
        "quote": text,
    }
    doc = _doc("d", (text, "must"), (other["quote"], "must"))
    rules = [
        _raw("duty", duty, ["d:1:1"], level="remind"),
        _raw("other", other, ["d:1:2"], cls=other.pop("cls", "hard")),
    ]
    return PolicyPackage(version="v", documents=[doc], rules=rules)


def test_a_duty_that_asks_for_a_value_over_a_threshold_that_forbids_it_conflicts():
    limit = {
        "type": "threshold",
        "tool": "update_ticket",
        "arg": "code",
        "op": "gt",
        "value": 5,
        "then": "forbid",
        "quote": "Code must not be above 5.",
    }
    found = _conflicts(_duty_package({"code": ["9"]}, dict(limit)))
    assert [(e.code, e.severity) for e in found] == [("rule_conflict", "error")]
    assert "forbids (code gt 5)" in found[0].message
    assert _conflicts(_duty_package({"code": ["3"]}, dict(limit))) == []
    assert _conflicts(_duty_package({"code": ["9", "3"]}, dict(limit))) == []  # one value passes
    approval = {**limit, "then": "requires_approval"}
    assert _conflicts(_duty_package({"code": ["9"]}, approval)) == []  # approval is not a ban


def test_a_duty_that_asks_for_a_value_outside_a_format_conflicts():
    form = {
        "type": "format",
        "tool": "update_ticket",
        "arg": "code",
        "pattern": r"T-\d+",
        "quote": "Codes look like T-123.",
    }
    found = _conflicts(_duty_package({"code": ["abc"]}, dict(form)))
    assert [(e.code, e.severity) for e in found] == [("rule_conflict", "error")]
    assert "does not match the format" in found[0].message
    assert _conflicts(_duty_package({"code": ["T-5"]}, dict(form))) == []
    assert _conflicts(_duty_package({"code": ["abc", "T-5"]}, dict(form))) == []


def test_a_duty_that_needs_a_tool_a_role_rule_forbids_is_a_warning_and_is_left_alone():
    role = {
        "type": "role_permission",
        "tools": ["update_ticket"],
        "roles": ["viewer"],
        "quote": "Viewers may not update tickets.",
    }
    pkg = _duty_package({}, role)
    found = _conflicts(pkg)
    assert [(e.code, e.severity) for e in found] == [("rule_conflict", "warning")]
    assert "viewer" in found[0].message and "resolve_conflicts leaves it" in found[0].message
    out, notes = resolve_conflicts(pkg)
    assert to_json(out) == to_json(pkg) and notes == []
    other_tool = _duty_package({}, dict(role), then_tool="create_ticket")
    assert _conflicts(other_tool) == []


def test_a_literal_that_the_tool_pattern_rejects_is_a_schema_conflict_warning():
    specs = [
        {
            "name": "update_ticket",
            "inputSchema": {"properties": {"code": {"type": "string", "pattern": "^[A-Z]+-\\d+$"}}},
        }
    ]
    rule = {"type": "allowed_values", "tool": "update_ticket", "arg": "code", "values": ["OPS-12", "ops-fin"]}
    found = [(e.code, e.severity) for e in _one(_raw("r", rule), specs=specs)]
    assert found == [("schema_conflict", "warning")]
    ok = {**rule, "values": ["OPS-12"]}
    assert _one(_raw("r", ok), specs=specs) == []
