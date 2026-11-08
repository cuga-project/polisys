"""notes.read_first: until the first message to people, remind the agent of named sources it has not read."""

from polisys.read_first import NOTE, ReadFirst

APPS = {
    "gmail_find_email": "gmail",
    "gmail_send_email": "gmail",
    "slack_list_channels": "slack",
    "slack_list_channel_messages": "slack",
    "slack_send_channel_message": "slack",
    "google_sheets_get_many_rows": "google_sheets",
}


def is_write(tool):
    return "send" in tool


def hooks(request):
    h = ReadFirst(APPS, is_write)
    h.on_user_message(request, None)
    return h


def test_named_apps_and_channels_that_were_not_read_are_listed():
    h = hooks("Do the daily sweep per the policy in Sheets, email the report and post to #treasury on Slack.")
    msg = h.before_llm_call(None).message
    assert msg.startswith(NOTE.split("{")[0]) and "google_sheets" in msg and "slack" in msg and "#treasury" in msg
    assert "gmail" not in msg  # the request names no mailbox


def test_a_channel_read_by_its_id_counts_and_the_note_ends_after_the_first_send():
    h = hooks("Post the summary to #treasury on Slack.")
    h.after_tool_call("slack_list_channels", {}, {"channels": [{"id": "C9", "name": "treasury"}]})
    assert "#treasury" in h.before_llm_call(None).message  # listing the channels is not reading it
    h.after_tool_call("slack_list_channel_messages", {"channel": "C9"}, {"messages": []})
    assert h.before_llm_call(None) is None
    h2 = hooks("Post the summary to #treasury on Slack.")
    h2.after_tool_call("slack_send_channel_message", {"channel": "C9", "text": "x"}, {"ok": True})
    assert h2.before_llm_call(None) is None and h2.records[0]["unread"]  # too late after a send; the miss is recorded


def test_a_request_that_names_nothing_gets_no_note():
    assert hooks("Process the new invoices.").before_llm_call(None) is None
