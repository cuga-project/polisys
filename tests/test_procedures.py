"""Procedures: numbered steps become a step graph; the monitor reminds, warns on order, and follows up once."""

from polisys.documents import load_document
from polisys.handlers.procedures import procedures
from polisys.procedure_monitor import ProcedureMonitor

SOP = """## 4. Invoices

4.2 Process every invoice in this order:
1. Screen the vendor with `vendor.screen_sanctions`.
2. Check duplicates with check_duplicates.
3. If the total is above $10,000, call notify_controller.
4. Request approval with request_approval.
"""
SPECS = [
    {"name": n}
    for n in (
        "vendor_screen_sanctions",
        "check_duplicates",
        "notify_controller",
        "request_approval",
        "pay_invoice",
    )
]


def _proc():
    return procedures(load_document(SOP, meta={"id": "ap"}).statements, SPECS)


def test_numbered_sop_becomes_steps_with_next_edges():
    (p,) = _proc()
    assert [s.action for s in p.steps] == [
        "vendor_screen_sanctions",
        "check_duplicates",
        "notify_controller",
        "request_approval",
    ]
    assert [s.required for s in p.steps] == [True, True, False, True]
    assert len(p.edges) == 3 and p.edges[0] == {"from": p.steps[0].id, "to": p.steps[1].id}


def test_reminder_names_next_required_step_after_trigger():
    mon = ProcedureMonitor(_proc())
    assert mon.before_llm_call(None).message is None  # not active yet
    mon.after_tool_call("vendor_screen_sanctions", {}, {"ok": True})
    assert "check_duplicates" in mon.before_llm_call(None).message


def test_reply_follow_up_lists_missing_required_step_once():
    mon = ProcedureMonitor(_proc())
    for t in ("vendor_screen_sanctions", "check_duplicates"):
        mon.after_tool_call(t, {}, {"ok": True})
    msg = mon.before_reply("done").message
    assert "request_approval" in msg and "notify_controller" not in msg
    assert mon.before_reply("done").message is None


def test_out_of_order_is_warning_not_deny():
    mon = ProcedureMonitor(_proc())
    res = mon.after_tool_call("request_approval", {}, {"ok": True})
    assert res.message is None and res.details[0]["outcome"] == "order_warning"


def test_step_with_tool_not_in_run_is_skipped():
    mon = ProcedureMonitor(_proc(), tools_in_run=["vendor_screen_sanctions", "check_duplicates"])
    for t in ("vendor_screen_sanctions", "check_duplicates"):
        mon.after_tool_call(t, {}, {"ok": True})
    assert mon.before_reply("done").message is None
