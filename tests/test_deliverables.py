"""Deliverables checklist: extraction keeps only what the request says; the reply check matches writes by kind."""

import json

from polisys.deliverables import DeliverablesHooks

SPECS = [
    {"name": "gmail_send_email", "inputSchema": {"properties": {"to": {}, "body": {}}}},
    {
        "name": "send_notification",
        "inputSchema": {"properties": {"type": {"enum": ["alert", "report", "update"]}, "message": {}}},
    },
    {"name": "update_incident_sla_details", "inputSchema": {"properties": {"stage": {}}}},
    {"name": "find_incident_by_number", "inputSchema": {"properties": {"number": {}}}},
]
REQUEST = 'Send a report to the assignee, email ops@acme.example.com the summary "Q1 done", and pause the SLA for INC-000057.'


def model(system, user):
    return json.dumps(
        {
            "items": [
                {
                    "action": "send report notification",
                    "tool": "send_notification",
                    "args": {"type": "report"},
                    "targets": ["the assignee"],
                    "values": [],
                },
                {
                    "action": "email the summary",
                    "tool": "gmail_send_email",
                    "args": {},
                    "targets": ["ops@acme.example.com"],
                    "values": ["Q1 done", "invented text"],
                },
                {
                    "action": "pause the SLA",
                    "tool": "update_incident_sla_details",
                    "args": {"stage": "paused"},
                    "targets": ["INC-000057"],
                    "values": [],
                },
                {"action": "made-up tool", "tool": "no_such_tool", "args": {}, "targets": [], "values": []},
            ]
        }
    )


def hooks():
    h = DeliverablesHooks(model, SPECS)
    h.on_user_message(REQUEST, None)
    return h


def test_extraction_keeps_only_request_text_and_known_tools():
    h = hooks()
    by_tool = {it["tool"]: it for it in h.items}
    assert "no_such_tool" not in by_tool
    assert by_tool["send_notification"]["args"] == {"type": "report"}
    assert by_tool["send_notification"]["targets"] == []  # a description, not a literal target
    assert by_tool["gmail_send_email"]["values"] == [
        "Q1 done"
    ]  # quoted in the request; invented text dropped
    assert (
        by_tool["update_incident_sla_details"]["args"] == {}
    )  # "paused" is not an allowed value in the schema


def test_reply_names_what_is_missing_and_follows_up_once():
    h = hooks()
    h.after_tool_call("send_notification", {"type": "alert", "message": "x"}, {"ok": True})
    h.after_tool_call("gmail_send_email", {"to": "ops@acme.example.com", "body": "Q1 done."}, {"id": 1})
    h.after_tool_call("update_incident_sla_details", {"stage": "paused"}, {"error": "denied"})
    missing = {it["tool"] for it in h.missing()}
    assert missing == {"send_notification", "update_incident_sla_details"}  # wrong type; failed call
    res = h.before_reply("done")
    assert "type=report" in res.message
    assert h.before_reply("done").message is None  # one follow-up only


def test_reads_are_ignored_and_same_kind_tool_counts():
    h = hooks()
    h.after_tool_call("find_incident_by_number", {"number": "INC-000057"}, {"incident_id": "INC_057"})
    h.after_tool_call("send_notification", {"type": "report", "message": "x"}, {"ok": True})
    h.after_tool_call("gmail_send_email", {"to": "ops@acme.example.com", "body": "Q1 done"}, {"id": 1})
    h.items = [it for it in h.items if it["tool"] != "update_incident_sla_details"]
    assert h.missing() == []
