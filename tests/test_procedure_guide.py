"""Procedure guide: procedure identification, first-write gate, template check, evidence check, general-rules stop."""

import json

from polisys.procedure_guide import ProcedureGuide, literal_head, template_regex

OFFBOARD = {
    "id": "d:12.1",
    "name": "Offboarding",
    "steps": [{"action": "jira__create_issue", "quote": "Step 2: Create ticket"}],
    "preconditions": [
        {
            "condition": "written authorization from the HR Director",
            "if_not": "stop",
            "quote": "Do not initiate involuntary offboarding without written authorization.",
        }
    ],
    "templates": [
        {
            "step": 1,
            "artifact": "Jira summary",
            "template": "OFFBOARD-[Last Name]-[Last Day MMDDYYYY]",
            "quote": "q",
        }
    ],
}
ONBOARD = {
    "id": "d:11.1",
    "name": "Onboarding",
    "steps": [{"action": "jira__create_issue", "quote": "Step 3: Create ticket"}],
    "preconditions": [
        {"condition": "new hire email received", "if_not": "stop", "quote": "Upon receiving the email"}
    ],
    "templates": [
        {
            "step": 1,
            "artifact": "Jira summary",
            "template": "ONBOARD-[Last Name]-[Start Date MMDDYYYY]",
            "quote": "q",
        }
    ],
}
GENERAL = {
    "id": "d:general",
    "name": "General rules",
    "steps": [],
    "exceptions": [
        {"when": "a required document is missing", "then": "do not submit", "quote": "Do not submit"}
    ],
}


def guide(model=None):
    return ProcedureGuide(
        [OFFBOARD, ONBOARD, GENERAL], model=model, is_write=lambda t: not t.startswith("read")
    )


def test_template_regex_and_head():
    rx = template_regex("ONBOARD-[Last Name]-[Start Date MMDDYYYY]")
    assert literal_head("ONBOARD-[Last Name]-[Start Date MMDDYYYY]") == "ONBOARD-"
    assert rx.search("ONBOARD-Yoon-04282026") and not rx.search("ONBOARD-Yoon-2026-04-28")


def test_template_head_picks_the_procedure_and_gate_stops_once():
    g = guide()
    first = g.before_tool_call("jira__create_issue", {"summary": "OFFBOARD-Navarro-04102026"})
    assert "Offboarding" in first.message and "written authorization" in first.message
    assert g.before_tool_call("jira__create_issue", {"summary": "OFFBOARD-Navarro-04102026"}).message is None


def test_wrong_template_is_stopped_once_with_the_expected_format():
    g = guide()
    msg = g.before_tool_call("jira__create_issue", {"summary": "ONBOARD-Kapoor-May 5"}).message
    assert "ONBOARD-[Last Name]-[Start Date MMDDYYYY]" in msg
    assert g.records[-1]["rule"] == "procedure_template"


def test_write_linked_to_no_procedure_gets_the_general_rules_once_and_reads_pass():
    g = guide()
    assert g.before_tool_call("read_inbox", {}).message is None
    msg = g.before_tool_call("google_mail__send_email", {"to": "x@y.com"}).message
    assert "required document is missing" in msg
    assert g.before_tool_call("google_mail__send_email", {"to": "x@y.com"}).message is None


def test_evidence_check_keeps_the_stop_when_proof_is_missing_and_caps_it():
    def model(system, user):
        return json.dumps({"answer": "no", "evidence": "the email is from the VP, not from the HR Director"})

    g = guide(model)
    g.after_tool_call("read_inbox", {}, {"from": "vp@x", "text": "Navarro termination"})
    args = {"summary": "OFFBOARD-Navarro-04102026"}
    assert g.before_tool_call("jira__create_issue", args).message  # gate
    assert "CHECK FAILED" in g.before_tool_call("jira__create_issue", args).message  # retry: no evidence
    assert "CHECK FAILED" in g.before_tool_call("jira__create_issue", args).message
    assert g.before_tool_call("jira__create_issue", args).message is None  # at most two evidence stops


def test_evidence_check_skips_preconditions_that_need_no_proof():
    calls = []
    g = guide(lambda s, u: calls.append(u) or json.dumps({"answer": "no", "evidence": "x"}))
    args = {"summary": "ONBOARD-Yoon-04282026"}
    g.before_tool_call("jira__create_issue", args)
    assert g.before_tool_call("jira__create_issue", args).message is None and not calls
