"""Policy references for the "read before write" reminder: found in the request and in data; read only by an output
that names the source and holds rule text; dead-end searches kept; writes never read a source."""

import json

from polisys.references import ReferenceLedger, find_references

ROWS = {
    "rows": [
        {
            "worksheet_id": "ws_alert_rules",
            "cells": {
                "Rule": "Onboarding policy",
                "Action": "Apply onboarding exclusion rules from separate sheet as per "
                "ws_onboarding_policy worksheet",
            },
        }
    ]
}
ONBOARDING = {
    "rows": [
        {
            "worksheet_id": "ws_onboarding_policy",
            "cells": {
                "Policy": "Recently onboarded",
                "Condition": "Earliest event of ANY type within 7 days",
                "Action": "Exclude entirely: no tag, no email, no SF task",
            },
        }
    ]
}


def test_references_in_requests():
    assert find_references(
        "Process the registrants by region per our regional meeting policy. Handle follow-ups."
    ) == ["per our regional meeting policy"]
    assert find_references("Follow our amortization rules - there may have been some recent changes.") == [
        "Follow our amortization rules"
    ]
    assert find_references("Send the summary to the team.") == []


def test_a_listing_or_title_is_not_a_read_but_the_rules_are():
    led = ReferenceLedger()
    led.add_from("Process the registrants per our regional meeting policy.", "the request")
    led.observe("zoom_list_meetings", {}, {"meetings": [{"topic": "Regional meeting - Europe"}]}, False)
    led.observe(
        "google_drive_find_multiple_files",
        {"title": "policy"},
        {"files": [{"title": "Regional Meeting Policy"}]},
        False,
    )
    assert led.unread()
    led.observe(
        "google_sheets_get_many_rows",
        {"spreadsheet": "ss_regional_policy"},
        {"title": "Regional Meeting Policy", "rows": [{"Rule": "Only regions with at least 5 registrants"}]},
        False,
    )
    assert not led.unread()


def test_source_named_and_held_in_the_same_or_a_later_output():
    led = ReferenceLedger()
    led.observe("google_sheets_find_many_rows", {"worksheet": "ws_alert_rules"}, ROWS, False)
    assert any("ws_onboarding_policy" in r.text for r in led.unread())  # named here, not held here
    led.observe("google_sheets_find_many_rows", {"worksheet": "ws_onboarding_policy"}, ONBOARDING, False)
    assert not [r for r in led.refs if "ws_onboarding_policy" in r.text and r.status != "read"]


def test_writes_never_read_and_empty_searches_are_kept():
    led = ReferenceLedger()
    led.add_from("Follow our amortization rules.", "the request")
    led.observe(
        "gmail_send_email", {"body": "Amortization rules: only items within the period"}, {"sent": True}, True
    )
    led.observe(
        "google_sheets_find_worksheet",
        {"title": "Amortization Rules"},
        json.dumps({"error": "Worksheet 'Amortization Rules' not found"}),
        False,
    )
    (ref,) = led.unread()
    assert ref.empty_searches == ["google_sheets_find_worksheet(title='Amortization Rules')"]
    assert "NOT READ" in led.summary() and "found nothing" in led.summary()


def test_intake_reminds_once_without_blocking_until_the_source_is_read():
    from polisys.intake import OnlinePolicyIntake

    hooks = OnlinePolicyIntake(lambda s, u: '{"policies": []}', check_reply=False)
    hooks.on_user_message("Process the registrants per our regional meeting policy.", None)
    assert hooks.before_tool_call("zoom_create_meeting", {"topic": "Europe"}).message is None  # never blocks
    assert hooks.records[-1]["rule"] == "read_before_write"
    note = hooks.before_llm_call(None).message
    assert "per our regional meeting policy" in note and "NOT READ" in note
    hooks.before_tool_call("zoom_create_meeting", {"topic": "APAC"})
    assert "read before write" not in (hooks.before_llm_call(None).message or "")  # named once
    hooks.after_tool_call(
        "google_sheets_get_many_rows",
        {"spreadsheet": "ss_regional_policy"},
        {"title": "Regional Meeting Policy", "rows": [{"Rule": "Only regions with at least 5"}]},
    )
    n = len(hooks.records)
    hooks.before_tool_call("zoom_create_meeting", {"topic": "Americas"})
    assert not [r for r in hooks.records[n:] if r.get("rule") == "read_before_write"]


def test_block_once_stops_only_the_first_write():
    from polisys.intake import OnlinePolicyIntake

    hooks = OnlinePolicyIntake(
        lambda s, u: '{"policies": []}', check_reply=False, read_before_write="block_once"
    )
    hooks.on_user_message("Process the registrants per our regional meeting policy.", None)
    first = hooks.before_tool_call("zoom_create_meeting", {"topic": "Europe"})
    assert first.message and "was not run" in first.message and "regional meeting policy" in first.message
    assert hooks.before_tool_call("zoom_create_meeting", {"topic": "Europe"}).message is None
    assert [r["outcome"] for r in hooks.records if r.get("rule") == "read_before_write"] == ["deny", "remind"]


def test_reminder_names_read_tools_not_used_yet():
    from polisys.intake import OnlinePolicyIntake

    specs = [
        {"name": n, "inputSchema": {}}
        for n in ("gmail_find_email", "google_sheets_get_many_rows", "gmail_send_email")
    ]
    hooks = OnlinePolicyIntake(lambda s, u: '{"policies": []}', check_reply=False, tool_specs=specs)
    hooks.on_user_message("Follow our amortization rules.", None)
    hooks.after_tool_call("google_sheets_get_many_rows", {"spreadsheet": "ss"}, {"rows": []})
    hooks.before_tool_call("gmail_send_email", {"to": "x@y.com"})
    note = hooks.before_llm_call(None).message
    assert "Read tools you have not used yet: gmail_find_email." in note
