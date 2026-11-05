"""SOP loop: a held case's normal-procedure writes are stopped (at most twice); other writes and SOP notices pass."""

from polisys.procedure_guide import ProcedureGuide
from polisys.sop_loop import LoopHooks

SUBMIT = {
    "id": "d:8.3",
    "name": "PA Submission",
    "steps": [
        {"action": "send_email", "quote": "Email the payer"},
        {"action": "post_message", "quote": "Post to #team"},
    ],
    "templates": [
        {"step": 1, "template": "[ORDER REQUEST] Acme | [Last Name]"},
        {"step": 2, "template": "[PA SUBMITTED] [CASE_ID]"},
    ],
}


def hooks():
    h = LoopHooks(is_write=lambda t: t.startswith(("send", "post")))
    h.which = ProcedureGuide([SUBMIT], gate=False, templates=False)._which
    h.held.append(
        {
            "case": "Webb",
            "keys": ["Webb"],
            "proc_id": "d:8.3",
            "proc_name": "PA Submission",
            "decision": "hold",
            "reason": "required document invalid",
            "stops": 0,
        }
    )
    return h


def test_held_case_normal_write_is_stopped_twice_then_passes():
    h = hooks()
    args = {"to": "pa@payer.com", "subject": "[ORDER REQUEST] Acme | Webb | Drug"}
    assert "HOLD" in h.before_tool_call("send_email", args).message
    assert h.before_tool_call("send_email", args).message
    assert h.before_tool_call("send_email", args).message is None


def test_hold_notice_and_other_cases_pass():
    h = hooks()
    assert (
        h.before_tool_call(
            "post_message", {"channel": "alerts", "text": "[PA HOLD] Webb invalid file"}
        ).message
        is None
    )
    assert h.before_tool_call("send_email", {"subject": "[ORDER REQUEST] Acme | Okafor | Drug"}).message is None
