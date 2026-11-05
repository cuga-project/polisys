"""Procedure intake: "Step N:" steps, table of contents skipped, per-procedure items with verbatim quotes, general rules."""

import json

from polisys.handlers.procedure_rules import enrich_procedures, general_rules
from polisys.handlers.procedures import procedures
from polisys.model import Statement, StatementType


def st(i, sec, text, types=(StatementType.PROCEDURE,)):
    return Statement(id=f"doc:{sec}:{i}", text=text, types=list(types), source={"section": sec})


STATEMENTS = [
    st(1, "contents", "Step 1: Onboarding"),
    st(2, "contents", "Step 2: Offboarding"),
    st(1, "12.1", "Step 1: Open employee_roster.xlsx and confirm the record."),
    st(2, "12.1", "Step 2: Create a Jira ticket: OFFBOARD-[Last Name]-[Last Day MMDDYYYY]."),
    st(
        3,
        "12.1",
        "Do not initiate involuntary offboarding without written authorization from Nicole Ashford.",
        (StatementType.PROHIBITION,),
    ),
    st(
        1,
        "15.6",
        "If a candidate rescinds their acceptance, forward the email to d.okonkwo@x.edu.",
        (StatementType.ROUTING,),
    ),
]
SPECS = [
    {"name": "jira__create_issue", "description": "Create a Jira issue"},
    {"name": "handbook__readFile", "description": "Read a file"},
]


def test_step_text_makes_a_procedure_and_contents_is_skipped():
    procs = procedures(STATEMENTS, SPECS, None, {})
    assert [p.id for p in procs] == ["doc:12.1"]
    assert len(procs[0].steps) == 2


def test_enrich_keeps_only_items_with_verbatim_quotes():
    def model(system, user):
        return json.dumps(
            {
                "preconditions": [
                    {
                        "condition": "written authorization",
                        "if_not": "stop",
                        "quote": "Do not initiate involuntary offboarding without written authorization from Nicole Ashford.",
                    },
                    {
                        "condition": "invented",
                        "if_not": "stop",
                        "quote": "This sentence is not in the SOP at all.",
                    },
                ],
                "exceptions": [],
                "templates": [
                    {
                        "step": 2,
                        "artifact": "Jira summary",
                        "template": "OFFBOARD-[Last Name]-[Last Day MMDDYYYY]",
                        "quote": "Create a Jira ticket: OFFBOARD-[Last Name]-[Last Day MMDDYYYY].",
                    }
                ],
                "escalations": [],
            }
        )

    procs, cov = enrich_procedures(procedures(STATEMENTS, SPECS, None, {}), STATEMENTS, [], model)
    assert [c["condition"] for c in procs[0].preconditions] == ["written authorization"]
    assert procs[0].templates and cov


def test_general_rules_go_to_the_procedure_they_affect_or_to_general():
    def model(system, user):
        return json.dumps(
            {
                "items": [
                    {
                        "s": 1,
                        "procedure": 0,
                        "kind": "exception",
                        "when": "a candidate rescinds",
                        "then": "forward",
                        "quote": "forward the email to d.okonkwo@x.edu.",
                    }
                ]
            }
        )

    procs, cov = general_rules(procedures(STATEMENTS, SPECS, None, {}), STATEMENTS, model)
    general = [p for p in procs if p.id.endswith(":general")]
    assert general and general[0].exceptions[0]["then"] == "forward" and cov


def test_labelled_bullets_are_steps_only_when_done_with_tools():
    from polisys.documents import _Parser

    p = _Parser("sop")
    p.feed(
        "# 2 Finance\n## 2.1 Reconciling\n- Generate workbook: download the Excel file from the Jira issue.\n"
        "- Clear items: comment on the Jira sub-task with the target GL.\n"
        "- Exception handling: create a Jira ticket in the FIN queue.\n"
        "## 2.2 Contacts\n- Support Contact: the finance desk.\n- Hours: weekdays.\n- Escalation owner: the Controller.\n"
    )
    sts = [s for sec in p.sections_out() for s in sec.statements]
    assert [s.source.get("step") for s in sts if s.source.get("bullet")] == [1, 2, 3, 1, 2, 3]
    specs = [
        {"name": "jira__get_issue", "description": "Get a Jira issue"},
        {"name": "jira__add_comment", "description": "Comment on a Jira issue"},
        {"name": "jira__create_issue", "description": "Create a Jira issue"},
    ]

    def model(system, user):
        if "Reconciling" in user or "2.1" in user:
            return json.dumps(
                {
                    "steps": [
                        {"n": 1, "tool": "jira__get_issue"},
                        {"n": 2, "tool": "jira__add_comment"},
                        {"n": 3, "tool": "jira__create_issue"},
                    ]
                }
            )
        return json.dumps({"steps": [{"n": i, "tool": ""} for i in (1, 2, 3)]})

    assert [pr.id for pr in procedures(sts, specs, model, {})] == ["sop:2.1"]
